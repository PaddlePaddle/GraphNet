import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_input_ids_: torch.Tensor,
        L_self_modules_model_modules_embed_layer_modules_hidden_layer_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_embed_layer_modules_hidden_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_embed_layer_modules_output_layer_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_embed_layer_modules_output_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_embed_layer_modules_residual_layer_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_embed_layer_modules_residual_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_rotary_emb_buffers_cos_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_rotary_emb_buffers_sin_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_ffn_layer_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_ffn_layer_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_ffn_layer_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_rotary_emb_buffers_cos_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_rotary_emb_buffers_sin_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_ffn_layer_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_ffn_layer_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_ffn_layer_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_rotary_emb_buffers_cos_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_rotary_emb_buffers_sin_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_ffn_layer_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_ffn_layer_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_ffn_layer_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_rotary_emb_buffers_cos_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_rotary_emb_buffers_sin_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_ffn_layer_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_ffn_layer_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_ffn_layer_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_rotary_emb_buffers_cos_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_rotary_emb_buffers_sin_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_ffn_layer_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_ffn_layer_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_ffn_layer_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_rotary_emb_buffers_cos_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_rotary_emb_buffers_sin_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_ffn_layer_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_ffn_layer_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_ffn_layer_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_rotary_emb_buffers_cos_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_rotary_emb_buffers_sin_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_ffn_layer_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_ffn_layer_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_ffn_layer_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_rotary_emb_buffers_cos_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_rotary_emb_buffers_sin_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_ffn_layer_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_ffn_layer_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_ffn_layer_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_rotary_emb_buffers_cos_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_rotary_emb_buffers_sin_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_ffn_layer_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_ffn_layer_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_ffn_layer_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_rotary_emb_buffers_cos_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_rotary_emb_buffers_sin_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_ffn_layer_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_ffn_layer_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_ffn_layer_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_rotary_emb_buffers_cos_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_rotary_emb_buffers_sin_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_ffn_layer_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_ffn_layer_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_ffn_layer_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_rotary_emb_buffers_cos_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_rotary_emb_buffers_sin_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_ffn_layer_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_ffn_layer_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_ffn_layer_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_adaLN_modulation_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_adaLN_modulation_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_adaLN_modulation_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_adaLN_modulation_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_adaLN_modulation_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_adaLN_modulation_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_flow_loss_modules_net_modules_final_layer_modules_adaLN_modulation_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_flow_loss_modules_net_modules_final_layer_modules_adaLN_modulation_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_input_ids_ = L_input_ids_
        l_self_modules_model_modules_embed_layer_modules_hidden_layer_parameters_weight_ = L_self_modules_model_modules_embed_layer_modules_hidden_layer_parameters_weight_
        l_self_modules_model_modules_embed_layer_modules_hidden_layer_parameters_bias_ = L_self_modules_model_modules_embed_layer_modules_hidden_layer_parameters_bias_
        l_self_modules_model_modules_embed_layer_modules_output_layer_parameters_weight_ = L_self_modules_model_modules_embed_layer_modules_output_layer_parameters_weight_
        l_self_modules_model_modules_embed_layer_modules_output_layer_parameters_bias_ = L_self_modules_model_modules_embed_layer_modules_output_layer_parameters_bias_
        l_self_modules_model_modules_embed_layer_modules_residual_layer_parameters_weight_ = L_self_modules_model_modules_embed_layer_modules_residual_layer_parameters_weight_
        l_self_modules_model_modules_embed_layer_modules_residual_layer_parameters_bias_ = L_self_modules_model_modules_embed_layer_modules_residual_layer_parameters_bias_
        l_self_modules_model_modules_layers_modules_0_modules_norm1_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_norm1_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_norm1_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_0_modules_norm1_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_rotary_emb_buffers_cos_cached_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_rotary_emb_buffers_sin_cached_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_norm2_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_norm2_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_norm2_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_0_modules_norm2_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_0_modules_ffn_layer_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_ffn_layer_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_ffn_layer_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_ffn_layer_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_ffn_layer_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_ffn_layer_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_norm1_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_norm1_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_norm1_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_1_modules_norm1_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_rotary_emb_buffers_cos_cached_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_rotary_emb_buffers_sin_cached_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_norm2_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_norm2_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_norm2_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_1_modules_norm2_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_1_modules_ffn_layer_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_ffn_layer_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_ffn_layer_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_ffn_layer_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_ffn_layer_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_ffn_layer_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_norm1_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_norm1_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_norm1_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_2_modules_norm1_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_rotary_emb_buffers_cos_cached_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_rotary_emb_buffers_sin_cached_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_norm2_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_norm2_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_norm2_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_2_modules_norm2_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_2_modules_ffn_layer_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_ffn_layer_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_ffn_layer_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_ffn_layer_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_ffn_layer_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_ffn_layer_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_norm1_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_norm1_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_norm1_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_3_modules_norm1_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_rotary_emb_buffers_cos_cached_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_rotary_emb_buffers_sin_cached_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_norm2_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_norm2_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_norm2_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_3_modules_norm2_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_3_modules_ffn_layer_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_ffn_layer_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_ffn_layer_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_ffn_layer_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_ffn_layer_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_ffn_layer_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_norm1_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_norm1_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_norm1_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_4_modules_norm1_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_rotary_emb_buffers_cos_cached_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_rotary_emb_buffers_sin_cached_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_norm2_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_norm2_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_norm2_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_4_modules_norm2_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_4_modules_ffn_layer_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_ffn_layer_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_ffn_layer_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_ffn_layer_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_ffn_layer_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_ffn_layer_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_norm1_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_norm1_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_norm1_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_5_modules_norm1_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_rotary_emb_buffers_cos_cached_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_rotary_emb_buffers_sin_cached_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_norm2_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_norm2_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_norm2_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_5_modules_norm2_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_5_modules_ffn_layer_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_ffn_layer_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_ffn_layer_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_ffn_layer_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_ffn_layer_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_ffn_layer_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_norm1_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_norm1_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_norm1_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_6_modules_norm1_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_rotary_emb_buffers_cos_cached_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_rotary_emb_buffers_sin_cached_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_norm2_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_norm2_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_norm2_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_6_modules_norm2_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_6_modules_ffn_layer_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_ffn_layer_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_ffn_layer_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_ffn_layer_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_ffn_layer_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_ffn_layer_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_norm1_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_norm1_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_norm1_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_7_modules_norm1_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_rotary_emb_buffers_cos_cached_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_rotary_emb_buffers_sin_cached_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_norm2_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_norm2_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_norm2_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_7_modules_norm2_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_7_modules_ffn_layer_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_ffn_layer_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_ffn_layer_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_ffn_layer_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_ffn_layer_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_ffn_layer_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_norm1_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_norm1_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_norm1_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_8_modules_norm1_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_rotary_emb_buffers_cos_cached_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_rotary_emb_buffers_sin_cached_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_norm2_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_norm2_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_norm2_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_8_modules_norm2_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_8_modules_ffn_layer_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_ffn_layer_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_ffn_layer_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_ffn_layer_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_ffn_layer_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_ffn_layer_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_norm1_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_norm1_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_norm1_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_9_modules_norm1_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_rotary_emb_buffers_cos_cached_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_rotary_emb_buffers_sin_cached_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_norm2_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_norm2_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_norm2_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_9_modules_norm2_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_9_modules_ffn_layer_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_ffn_layer_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_ffn_layer_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_ffn_layer_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_ffn_layer_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_ffn_layer_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_norm1_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_norm1_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_norm1_parameters_bias_ = L_self_modules_model_modules_layers_modules_10_modules_norm1_parameters_bias_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_rotary_emb_buffers_cos_cached_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_rotary_emb_buffers_sin_cached_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_norm2_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_norm2_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_norm2_parameters_bias_ = L_self_modules_model_modules_layers_modules_10_modules_norm2_parameters_bias_
        l_self_modules_model_modules_layers_modules_10_modules_ffn_layer_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_ffn_layer_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_ffn_layer_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_ffn_layer_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_ffn_layer_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_ffn_layer_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_norm1_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_norm1_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_norm1_parameters_bias_ = L_self_modules_model_modules_layers_modules_11_modules_norm1_parameters_bias_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_rotary_emb_buffers_cos_cached_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_rotary_emb_buffers_sin_cached_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_norm2_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_norm2_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_norm2_parameters_bias_ = L_self_modules_model_modules_layers_modules_11_modules_norm2_parameters_bias_
        l_self_modules_model_modules_layers_modules_11_modules_ffn_layer_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_ffn_layer_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_ffn_layer_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_ffn_layer_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_ffn_layer_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_ffn_layer_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_norm_parameters_weight_ = (
            L_self_modules_model_modules_norm_parameters_weight_
        )
        l_self_modules_model_modules_norm_parameters_bias_ = (
            L_self_modules_model_modules_norm_parameters_bias_
        )
        l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_ = (
            L_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_
        )
        l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_ = (
            L_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_
        )
        l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_ = L_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_
        l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_ = L_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_
        l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_ = L_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_
        l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_ = L_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_
        l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_ = (
            L_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_
        )
        l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_ = (
            L_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_
        )
        l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_ = L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_adaLN_modulation_modules_1_parameters_weight_
        l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_ = L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_adaLN_modulation_modules_1_parameters_bias_
        l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_ = L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_
        l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_ = L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_
        l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_ = L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_
        l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_ = L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_
        l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_ = L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_
        l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_ = L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_
        l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_ = L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_adaLN_modulation_modules_1_parameters_weight_
        l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_ = L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_adaLN_modulation_modules_1_parameters_bias_
        l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_ = L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_
        l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_ = L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_
        l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_ = L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_
        l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_ = L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_
        l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_ = L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_
        l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_ = L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_
        l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_ = L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_adaLN_modulation_modules_1_parameters_weight_
        l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_ = L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_adaLN_modulation_modules_1_parameters_bias_
        l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_ = L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_
        l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_ = L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_
        l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_ = L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_
        l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_ = L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_
        l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_ = L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_
        l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_ = L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_
        l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_ = L_self_modules_flow_loss_modules_net_modules_final_layer_modules_adaLN_modulation_modules_1_parameters_weight_
        l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_ = L_self_modules_flow_loss_modules_net_modules_final_layer_modules_adaLN_modulation_modules_1_parameters_bias_
        l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_ = L_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_
        l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_ = L_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_
        mask = torch.ones_like(l_input_ids_, dtype=torch.float32)
        x = torch._C._nn.pad(l_input_ids_, (0, 0), "constant", None)
        l_input_ids_ = None
        mask_1 = torch._C._nn.pad(mask, (0, 0), "constant", None)
        mask = None
        x_1 = x.unfold(dimension=-1, size=16, step=16)
        x = None
        mask_2 = mask_1.unfold(dimension=-1, size=16, step=16)
        mask_1 = None
        x_2 = torch.cat([x_1, mask_2], dim=-1)
        x_1 = mask_2 = None
        linear = torch._C._nn.linear(
            x_2,
            l_self_modules_model_modules_embed_layer_modules_hidden_layer_parameters_weight_,
            l_self_modules_model_modules_embed_layer_modules_hidden_layer_parameters_bias_,
        )
        l_self_modules_model_modules_embed_layer_modules_hidden_layer_parameters_weight_ = l_self_modules_model_modules_embed_layer_modules_hidden_layer_parameters_bias_ = (None)
        hid = torch.nn.functional.silu(linear, inplace=False)
        linear = None
        linear_1 = torch._C._nn.linear(
            hid,
            l_self_modules_model_modules_embed_layer_modules_output_layer_parameters_weight_,
            l_self_modules_model_modules_embed_layer_modules_output_layer_parameters_bias_,
        )
        hid = l_self_modules_model_modules_embed_layer_modules_output_layer_parameters_weight_ = l_self_modules_model_modules_embed_layer_modules_output_layer_parameters_bias_ = (None)
        out = torch.nn.functional.dropout(linear_1, 0.1, False, False)
        linear_1 = None
        res = torch._C._nn.linear(
            x_2,
            l_self_modules_model_modules_embed_layer_modules_residual_layer_parameters_weight_,
            l_self_modules_model_modules_embed_layer_modules_residual_layer_parameters_bias_,
        )
        x_2 = l_self_modules_model_modules_embed_layer_modules_residual_layer_parameters_weight_ = l_self_modules_model_modules_embed_layer_modules_residual_layer_parameters_bias_ = (None)
        out_1 = out + res
        out = res = None
        position_ids = torch.arange(
            0, 180, dtype=torch.int64, device=device(type="cuda", index=0)
        )
        position_ids_1 = position_ids.view(-1, 180)
        position_ids = None
        mask_3 = torch.full(
            (180, 180), -3.4028234663852886e38, device=device(type="cuda", index=0)
        )
        mask_cond = torch.arange(180, device=device(type="cuda", index=0))
        add_1 = mask_cond + 1
        view_1 = add_1.view(180, 1)
        add_1 = None
        lt = mask_cond < view_1
        mask_cond = view_1 = None
        masked_fill_ = mask_3.masked_fill_(lt, 0)
        lt = masked_fill_ = None
        mask_4 = mask_3.to(torch.float32)
        mask_3 = None
        getitem = mask_4[(None, None, slice(None, None, None), slice(None, None, None))]
        mask_4 = None
        causal_4d_mask = getitem.expand(1, 1, 180, 180)
        getitem = None
        hidden_states = torch.nn.functional.layer_norm(
            out_1,
            (768,),
            l_self_modules_model_modules_layers_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_model_modules_layers_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_0_modules_norm1_parameters_weight_ = (
            l_self_modules_model_modules_layers_modules_0_modules_norm1_parameters_bias_
        ) = None
        query_states = torch._C._nn.linear(
            hidden_states,
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        key_states = torch._C._nn.linear(
            hidden_states,
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states = torch._C._nn.linear(
            hidden_states,
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states = l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_2 = query_states.view(1, 180, 12, 64)
        query_states = None
        query_states_1 = view_2.transpose(1, 2)
        view_2 = None
        view_3 = key_states.view(1, 180, 12, 64)
        key_states = None
        key_states_1 = view_3.transpose(1, 2)
        view_3 = None
        view_4 = value_states.view(1, 180, 12, 64)
        value_states = None
        value_states_1 = view_4.transpose(1, 2)
        view_4 = None
        getitem_1 = l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_rotary_emb_buffers_cos_cached_[
            slice(None, 180, None)
        ]
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = (
            None
        )
        cos = getitem_1.to(dtype=torch.float32)
        getitem_1 = None
        getitem_2 = l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_rotary_emb_buffers_sin_cached_[
            slice(None, 180, None)
        ]
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = (
            None
        )
        sin = getitem_2.to(dtype=torch.float32)
        getitem_2 = None
        getitem_3 = cos[position_ids_1]
        cos = None
        cos_1 = getitem_3.unsqueeze(1)
        getitem_3 = None
        getitem_4 = sin[position_ids_1]
        sin = None
        sin_1 = getitem_4.unsqueeze(1)
        getitem_4 = None
        mul = query_states_1 * cos_1
        x1 = query_states_1[(Ellipsis, slice(None, 32, None))]
        x2 = query_states_1[(Ellipsis, slice(32, None, None))]
        query_states_1 = None
        neg = -x2
        x2 = None
        cat_1 = torch.cat((neg, x1), dim=-1)
        neg = x1 = None
        mul_1 = cat_1 * sin_1
        cat_1 = None
        q_embed = mul + mul_1
        mul = mul_1 = None
        mul_2 = key_states_1 * cos_1
        cos_1 = None
        x1_1 = key_states_1[(Ellipsis, slice(None, 32, None))]
        x2_1 = key_states_1[(Ellipsis, slice(32, None, None))]
        key_states_1 = None
        neg_1 = -x2_1
        x2_1 = None
        cat_2 = torch.cat((neg_1, x1_1), dim=-1)
        neg_1 = x1_1 = None
        mul_3 = cat_2 * sin_1
        cat_2 = sin_1 = None
        k_embed = mul_2 + mul_3
        mul_2 = mul_3 = None
        attn_output = torch._C._nn.scaled_dot_product_attention(
            q_embed, k_embed, value_states_1, causal_4d_mask, dropout_p=0.0
        )
        q_embed = k_embed = value_states_1 = None
        transpose_3 = attn_output.transpose(1, 2)
        attn_output = None
        attn_output_1 = transpose_3.contiguous()
        transpose_3 = None
        attn_output_2 = attn_output_1.reshape(1, 180, 768)
        attn_output_1 = None
        attn_output_3 = torch._C._nn.linear(
            attn_output_2,
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_2 = l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_1 = out_1 + attn_output_3
        out_1 = attn_output_3 = None
        hidden_states_2 = torch.nn.functional.layer_norm(
            hidden_states_1,
            (768,),
            l_self_modules_model_modules_layers_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_model_modules_layers_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_0_modules_norm2_parameters_weight_ = (
            l_self_modules_model_modules_layers_modules_0_modules_norm2_parameters_bias_
        ) = None
        linear_7 = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_model_modules_layers_modules_0_modules_ffn_layer_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_0_modules_ffn_layer_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_1 = torch.nn.functional.silu(linear_7, inplace=False)
        linear_7 = None
        linear_8 = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_model_modules_layers_modules_0_modules_ffn_layer_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_2 = l_self_modules_model_modules_layers_modules_0_modules_ffn_layer_modules_up_proj_parameters_weight_ = (None)
        mul_4 = silu_1 * linear_8
        silu_1 = linear_8 = None
        hidden_states_3 = torch._C._nn.linear(
            mul_4,
            l_self_modules_model_modules_layers_modules_0_modules_ffn_layer_modules_down_proj_parameters_weight_,
            None,
        )
        mul_4 = l_self_modules_model_modules_layers_modules_0_modules_ffn_layer_modules_down_proj_parameters_weight_ = (None)
        hidden_states_4 = hidden_states_1 + hidden_states_3
        hidden_states_1 = hidden_states_3 = None
        hidden_states_5 = torch.nn.functional.layer_norm(
            hidden_states_4,
            (768,),
            l_self_modules_model_modules_layers_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_model_modules_layers_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_1_modules_norm1_parameters_weight_ = (
            l_self_modules_model_modules_layers_modules_1_modules_norm1_parameters_bias_
        ) = None
        query_states_2 = torch._C._nn.linear(
            hidden_states_5,
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        key_states_2 = torch._C._nn.linear(
            hidden_states_5,
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_2 = torch._C._nn.linear(
            hidden_states_5,
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_5 = l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_5 = query_states_2.view(1, 180, 12, 64)
        query_states_2 = None
        query_states_3 = view_5.transpose(1, 2)
        view_5 = None
        view_6 = key_states_2.view(1, 180, 12, 64)
        key_states_2 = None
        key_states_3 = view_6.transpose(1, 2)
        view_6 = None
        view_7 = value_states_2.view(1, 180, 12, 64)
        value_states_2 = None
        value_states_3 = view_7.transpose(1, 2)
        view_7 = None
        getitem_9 = l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_rotary_emb_buffers_cos_cached_[
            slice(None, 180, None)
        ]
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = (
            None
        )
        cos_2 = getitem_9.to(dtype=torch.float32)
        getitem_9 = None
        getitem_10 = l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_rotary_emb_buffers_sin_cached_[
            slice(None, 180, None)
        ]
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = (
            None
        )
        sin_2 = getitem_10.to(dtype=torch.float32)
        getitem_10 = None
        getitem_11 = cos_2[position_ids_1]
        cos_2 = None
        cos_3 = getitem_11.unsqueeze(1)
        getitem_11 = None
        getitem_12 = sin_2[position_ids_1]
        sin_2 = None
        sin_3 = getitem_12.unsqueeze(1)
        getitem_12 = None
        mul_5 = query_states_3 * cos_3
        x1_2 = query_states_3[(Ellipsis, slice(None, 32, None))]
        x2_2 = query_states_3[(Ellipsis, slice(32, None, None))]
        query_states_3 = None
        neg_2 = -x2_2
        x2_2 = None
        cat_3 = torch.cat((neg_2, x1_2), dim=-1)
        neg_2 = x1_2 = None
        mul_6 = cat_3 * sin_3
        cat_3 = None
        q_embed_1 = mul_5 + mul_6
        mul_5 = mul_6 = None
        mul_7 = key_states_3 * cos_3
        cos_3 = None
        x1_3 = key_states_3[(Ellipsis, slice(None, 32, None))]
        x2_3 = key_states_3[(Ellipsis, slice(32, None, None))]
        key_states_3 = None
        neg_3 = -x2_3
        x2_3 = None
        cat_4 = torch.cat((neg_3, x1_3), dim=-1)
        neg_3 = x1_3 = None
        mul_8 = cat_4 * sin_3
        cat_4 = sin_3 = None
        k_embed_1 = mul_7 + mul_8
        mul_7 = mul_8 = None
        attn_output_4 = torch._C._nn.scaled_dot_product_attention(
            q_embed_1, k_embed_1, value_states_3, causal_4d_mask, dropout_p=0.0
        )
        q_embed_1 = k_embed_1 = value_states_3 = None
        transpose_7 = attn_output_4.transpose(1, 2)
        attn_output_4 = None
        attn_output_5 = transpose_7.contiguous()
        transpose_7 = None
        attn_output_6 = attn_output_5.reshape(1, 180, 768)
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
            (768,),
            l_self_modules_model_modules_layers_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_model_modules_layers_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_1_modules_norm2_parameters_weight_ = (
            l_self_modules_model_modules_layers_modules_1_modules_norm2_parameters_bias_
        ) = None
        linear_14 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_model_modules_layers_modules_1_modules_ffn_layer_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_1_modules_ffn_layer_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_2 = torch.nn.functional.silu(linear_14, inplace=False)
        linear_14 = None
        linear_15 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_model_modules_layers_modules_1_modules_ffn_layer_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_7 = l_self_modules_model_modules_layers_modules_1_modules_ffn_layer_modules_up_proj_parameters_weight_ = (None)
        mul_9 = silu_2 * linear_15
        silu_2 = linear_15 = None
        hidden_states_8 = torch._C._nn.linear(
            mul_9,
            l_self_modules_model_modules_layers_modules_1_modules_ffn_layer_modules_down_proj_parameters_weight_,
            None,
        )
        mul_9 = l_self_modules_model_modules_layers_modules_1_modules_ffn_layer_modules_down_proj_parameters_weight_ = (None)
        hidden_states_9 = hidden_states_6 + hidden_states_8
        hidden_states_6 = hidden_states_8 = None
        hidden_states_10 = torch.nn.functional.layer_norm(
            hidden_states_9,
            (768,),
            l_self_modules_model_modules_layers_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_model_modules_layers_modules_2_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_2_modules_norm1_parameters_weight_ = (
            l_self_modules_model_modules_layers_modules_2_modules_norm1_parameters_bias_
        ) = None
        query_states_4 = torch._C._nn.linear(
            hidden_states_10,
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        key_states_4 = torch._C._nn.linear(
            hidden_states_10,
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_4 = torch._C._nn.linear(
            hidden_states_10,
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_10 = l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_8 = query_states_4.view(1, 180, 12, 64)
        query_states_4 = None
        query_states_5 = view_8.transpose(1, 2)
        view_8 = None
        view_9 = key_states_4.view(1, 180, 12, 64)
        key_states_4 = None
        key_states_5 = view_9.transpose(1, 2)
        view_9 = None
        view_10 = value_states_4.view(1, 180, 12, 64)
        value_states_4 = None
        value_states_5 = view_10.transpose(1, 2)
        view_10 = None
        getitem_17 = l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_rotary_emb_buffers_cos_cached_[
            slice(None, 180, None)
        ]
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = (
            None
        )
        cos_4 = getitem_17.to(dtype=torch.float32)
        getitem_17 = None
        getitem_18 = l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_rotary_emb_buffers_sin_cached_[
            slice(None, 180, None)
        ]
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = (
            None
        )
        sin_4 = getitem_18.to(dtype=torch.float32)
        getitem_18 = None
        getitem_19 = cos_4[position_ids_1]
        cos_4 = None
        cos_5 = getitem_19.unsqueeze(1)
        getitem_19 = None
        getitem_20 = sin_4[position_ids_1]
        sin_4 = None
        sin_5 = getitem_20.unsqueeze(1)
        getitem_20 = None
        mul_10 = query_states_5 * cos_5
        x1_4 = query_states_5[(Ellipsis, slice(None, 32, None))]
        x2_4 = query_states_5[(Ellipsis, slice(32, None, None))]
        query_states_5 = None
        neg_4 = -x2_4
        x2_4 = None
        cat_5 = torch.cat((neg_4, x1_4), dim=-1)
        neg_4 = x1_4 = None
        mul_11 = cat_5 * sin_5
        cat_5 = None
        q_embed_2 = mul_10 + mul_11
        mul_10 = mul_11 = None
        mul_12 = key_states_5 * cos_5
        cos_5 = None
        x1_5 = key_states_5[(Ellipsis, slice(None, 32, None))]
        x2_5 = key_states_5[(Ellipsis, slice(32, None, None))]
        key_states_5 = None
        neg_5 = -x2_5
        x2_5 = None
        cat_6 = torch.cat((neg_5, x1_5), dim=-1)
        neg_5 = x1_5 = None
        mul_13 = cat_6 * sin_5
        cat_6 = sin_5 = None
        k_embed_2 = mul_12 + mul_13
        mul_12 = mul_13 = None
        attn_output_8 = torch._C._nn.scaled_dot_product_attention(
            q_embed_2, k_embed_2, value_states_5, causal_4d_mask, dropout_p=0.0
        )
        q_embed_2 = k_embed_2 = value_states_5 = None
        transpose_11 = attn_output_8.transpose(1, 2)
        attn_output_8 = None
        attn_output_9 = transpose_11.contiguous()
        transpose_11 = None
        attn_output_10 = attn_output_9.reshape(1, 180, 768)
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
            (768,),
            l_self_modules_model_modules_layers_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_model_modules_layers_modules_2_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_2_modules_norm2_parameters_weight_ = (
            l_self_modules_model_modules_layers_modules_2_modules_norm2_parameters_bias_
        ) = None
        linear_21 = torch._C._nn.linear(
            hidden_states_12,
            l_self_modules_model_modules_layers_modules_2_modules_ffn_layer_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_2_modules_ffn_layer_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_3 = torch.nn.functional.silu(linear_21, inplace=False)
        linear_21 = None
        linear_22 = torch._C._nn.linear(
            hidden_states_12,
            l_self_modules_model_modules_layers_modules_2_modules_ffn_layer_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_12 = l_self_modules_model_modules_layers_modules_2_modules_ffn_layer_modules_up_proj_parameters_weight_ = (None)
        mul_14 = silu_3 * linear_22
        silu_3 = linear_22 = None
        hidden_states_13 = torch._C._nn.linear(
            mul_14,
            l_self_modules_model_modules_layers_modules_2_modules_ffn_layer_modules_down_proj_parameters_weight_,
            None,
        )
        mul_14 = l_self_modules_model_modules_layers_modules_2_modules_ffn_layer_modules_down_proj_parameters_weight_ = (None)
        hidden_states_14 = hidden_states_11 + hidden_states_13
        hidden_states_11 = hidden_states_13 = None
        hidden_states_15 = torch.nn.functional.layer_norm(
            hidden_states_14,
            (768,),
            l_self_modules_model_modules_layers_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_model_modules_layers_modules_3_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_3_modules_norm1_parameters_weight_ = (
            l_self_modules_model_modules_layers_modules_3_modules_norm1_parameters_bias_
        ) = None
        query_states_6 = torch._C._nn.linear(
            hidden_states_15,
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        key_states_6 = torch._C._nn.linear(
            hidden_states_15,
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_6 = torch._C._nn.linear(
            hidden_states_15,
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_15 = l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_11 = query_states_6.view(1, 180, 12, 64)
        query_states_6 = None
        query_states_7 = view_11.transpose(1, 2)
        view_11 = None
        view_12 = key_states_6.view(1, 180, 12, 64)
        key_states_6 = None
        key_states_7 = view_12.transpose(1, 2)
        view_12 = None
        view_13 = value_states_6.view(1, 180, 12, 64)
        value_states_6 = None
        value_states_7 = view_13.transpose(1, 2)
        view_13 = None
        getitem_25 = l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_rotary_emb_buffers_cos_cached_[
            slice(None, 180, None)
        ]
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = (
            None
        )
        cos_6 = getitem_25.to(dtype=torch.float32)
        getitem_25 = None
        getitem_26 = l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_rotary_emb_buffers_sin_cached_[
            slice(None, 180, None)
        ]
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = (
            None
        )
        sin_6 = getitem_26.to(dtype=torch.float32)
        getitem_26 = None
        getitem_27 = cos_6[position_ids_1]
        cos_6 = None
        cos_7 = getitem_27.unsqueeze(1)
        getitem_27 = None
        getitem_28 = sin_6[position_ids_1]
        sin_6 = None
        sin_7 = getitem_28.unsqueeze(1)
        getitem_28 = None
        mul_15 = query_states_7 * cos_7
        x1_6 = query_states_7[(Ellipsis, slice(None, 32, None))]
        x2_6 = query_states_7[(Ellipsis, slice(32, None, None))]
        query_states_7 = None
        neg_6 = -x2_6
        x2_6 = None
        cat_7 = torch.cat((neg_6, x1_6), dim=-1)
        neg_6 = x1_6 = None
        mul_16 = cat_7 * sin_7
        cat_7 = None
        q_embed_3 = mul_15 + mul_16
        mul_15 = mul_16 = None
        mul_17 = key_states_7 * cos_7
        cos_7 = None
        x1_7 = key_states_7[(Ellipsis, slice(None, 32, None))]
        x2_7 = key_states_7[(Ellipsis, slice(32, None, None))]
        key_states_7 = None
        neg_7 = -x2_7
        x2_7 = None
        cat_8 = torch.cat((neg_7, x1_7), dim=-1)
        neg_7 = x1_7 = None
        mul_18 = cat_8 * sin_7
        cat_8 = sin_7 = None
        k_embed_3 = mul_17 + mul_18
        mul_17 = mul_18 = None
        attn_output_12 = torch._C._nn.scaled_dot_product_attention(
            q_embed_3, k_embed_3, value_states_7, causal_4d_mask, dropout_p=0.0
        )
        q_embed_3 = k_embed_3 = value_states_7 = None
        transpose_15 = attn_output_12.transpose(1, 2)
        attn_output_12 = None
        attn_output_13 = transpose_15.contiguous()
        transpose_15 = None
        attn_output_14 = attn_output_13.reshape(1, 180, 768)
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
            (768,),
            l_self_modules_model_modules_layers_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_model_modules_layers_modules_3_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_3_modules_norm2_parameters_weight_ = (
            l_self_modules_model_modules_layers_modules_3_modules_norm2_parameters_bias_
        ) = None
        linear_28 = torch._C._nn.linear(
            hidden_states_17,
            l_self_modules_model_modules_layers_modules_3_modules_ffn_layer_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_3_modules_ffn_layer_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_4 = torch.nn.functional.silu(linear_28, inplace=False)
        linear_28 = None
        linear_29 = torch._C._nn.linear(
            hidden_states_17,
            l_self_modules_model_modules_layers_modules_3_modules_ffn_layer_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_17 = l_self_modules_model_modules_layers_modules_3_modules_ffn_layer_modules_up_proj_parameters_weight_ = (None)
        mul_19 = silu_4 * linear_29
        silu_4 = linear_29 = None
        hidden_states_18 = torch._C._nn.linear(
            mul_19,
            l_self_modules_model_modules_layers_modules_3_modules_ffn_layer_modules_down_proj_parameters_weight_,
            None,
        )
        mul_19 = l_self_modules_model_modules_layers_modules_3_modules_ffn_layer_modules_down_proj_parameters_weight_ = (None)
        hidden_states_19 = hidden_states_16 + hidden_states_18
        hidden_states_16 = hidden_states_18 = None
        hidden_states_20 = torch.nn.functional.layer_norm(
            hidden_states_19,
            (768,),
            l_self_modules_model_modules_layers_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_model_modules_layers_modules_4_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_4_modules_norm1_parameters_weight_ = (
            l_self_modules_model_modules_layers_modules_4_modules_norm1_parameters_bias_
        ) = None
        query_states_8 = torch._C._nn.linear(
            hidden_states_20,
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        key_states_8 = torch._C._nn.linear(
            hidden_states_20,
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_8 = torch._C._nn.linear(
            hidden_states_20,
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_20 = l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_14 = query_states_8.view(1, 180, 12, 64)
        query_states_8 = None
        query_states_9 = view_14.transpose(1, 2)
        view_14 = None
        view_15 = key_states_8.view(1, 180, 12, 64)
        key_states_8 = None
        key_states_9 = view_15.transpose(1, 2)
        view_15 = None
        view_16 = value_states_8.view(1, 180, 12, 64)
        value_states_8 = None
        value_states_9 = view_16.transpose(1, 2)
        view_16 = None
        getitem_33 = l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_rotary_emb_buffers_cos_cached_[
            slice(None, 180, None)
        ]
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = (
            None
        )
        cos_8 = getitem_33.to(dtype=torch.float32)
        getitem_33 = None
        getitem_34 = l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_rotary_emb_buffers_sin_cached_[
            slice(None, 180, None)
        ]
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = (
            None
        )
        sin_8 = getitem_34.to(dtype=torch.float32)
        getitem_34 = None
        getitem_35 = cos_8[position_ids_1]
        cos_8 = None
        cos_9 = getitem_35.unsqueeze(1)
        getitem_35 = None
        getitem_36 = sin_8[position_ids_1]
        sin_8 = None
        sin_9 = getitem_36.unsqueeze(1)
        getitem_36 = None
        mul_20 = query_states_9 * cos_9
        x1_8 = query_states_9[(Ellipsis, slice(None, 32, None))]
        x2_8 = query_states_9[(Ellipsis, slice(32, None, None))]
        query_states_9 = None
        neg_8 = -x2_8
        x2_8 = None
        cat_9 = torch.cat((neg_8, x1_8), dim=-1)
        neg_8 = x1_8 = None
        mul_21 = cat_9 * sin_9
        cat_9 = None
        q_embed_4 = mul_20 + mul_21
        mul_20 = mul_21 = None
        mul_22 = key_states_9 * cos_9
        cos_9 = None
        x1_9 = key_states_9[(Ellipsis, slice(None, 32, None))]
        x2_9 = key_states_9[(Ellipsis, slice(32, None, None))]
        key_states_9 = None
        neg_9 = -x2_9
        x2_9 = None
        cat_10 = torch.cat((neg_9, x1_9), dim=-1)
        neg_9 = x1_9 = None
        mul_23 = cat_10 * sin_9
        cat_10 = sin_9 = None
        k_embed_4 = mul_22 + mul_23
        mul_22 = mul_23 = None
        attn_output_16 = torch._C._nn.scaled_dot_product_attention(
            q_embed_4, k_embed_4, value_states_9, causal_4d_mask, dropout_p=0.0
        )
        q_embed_4 = k_embed_4 = value_states_9 = None
        transpose_19 = attn_output_16.transpose(1, 2)
        attn_output_16 = None
        attn_output_17 = transpose_19.contiguous()
        transpose_19 = None
        attn_output_18 = attn_output_17.reshape(1, 180, 768)
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
            (768,),
            l_self_modules_model_modules_layers_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_model_modules_layers_modules_4_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_4_modules_norm2_parameters_weight_ = (
            l_self_modules_model_modules_layers_modules_4_modules_norm2_parameters_bias_
        ) = None
        linear_35 = torch._C._nn.linear(
            hidden_states_22,
            l_self_modules_model_modules_layers_modules_4_modules_ffn_layer_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_4_modules_ffn_layer_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_5 = torch.nn.functional.silu(linear_35, inplace=False)
        linear_35 = None
        linear_36 = torch._C._nn.linear(
            hidden_states_22,
            l_self_modules_model_modules_layers_modules_4_modules_ffn_layer_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_22 = l_self_modules_model_modules_layers_modules_4_modules_ffn_layer_modules_up_proj_parameters_weight_ = (None)
        mul_24 = silu_5 * linear_36
        silu_5 = linear_36 = None
        hidden_states_23 = torch._C._nn.linear(
            mul_24,
            l_self_modules_model_modules_layers_modules_4_modules_ffn_layer_modules_down_proj_parameters_weight_,
            None,
        )
        mul_24 = l_self_modules_model_modules_layers_modules_4_modules_ffn_layer_modules_down_proj_parameters_weight_ = (None)
        hidden_states_24 = hidden_states_21 + hidden_states_23
        hidden_states_21 = hidden_states_23 = None
        hidden_states_25 = torch.nn.functional.layer_norm(
            hidden_states_24,
            (768,),
            l_self_modules_model_modules_layers_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_model_modules_layers_modules_5_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_5_modules_norm1_parameters_weight_ = (
            l_self_modules_model_modules_layers_modules_5_modules_norm1_parameters_bias_
        ) = None
        query_states_10 = torch._C._nn.linear(
            hidden_states_25,
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        key_states_10 = torch._C._nn.linear(
            hidden_states_25,
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_10 = torch._C._nn.linear(
            hidden_states_25,
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_25 = l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_17 = query_states_10.view(1, 180, 12, 64)
        query_states_10 = None
        query_states_11 = view_17.transpose(1, 2)
        view_17 = None
        view_18 = key_states_10.view(1, 180, 12, 64)
        key_states_10 = None
        key_states_11 = view_18.transpose(1, 2)
        view_18 = None
        view_19 = value_states_10.view(1, 180, 12, 64)
        value_states_10 = None
        value_states_11 = view_19.transpose(1, 2)
        view_19 = None
        getitem_41 = l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_rotary_emb_buffers_cos_cached_[
            slice(None, 180, None)
        ]
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = (
            None
        )
        cos_10 = getitem_41.to(dtype=torch.float32)
        getitem_41 = None
        getitem_42 = l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_rotary_emb_buffers_sin_cached_[
            slice(None, 180, None)
        ]
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = (
            None
        )
        sin_10 = getitem_42.to(dtype=torch.float32)
        getitem_42 = None
        getitem_43 = cos_10[position_ids_1]
        cos_10 = None
        cos_11 = getitem_43.unsqueeze(1)
        getitem_43 = None
        getitem_44 = sin_10[position_ids_1]
        sin_10 = None
        sin_11 = getitem_44.unsqueeze(1)
        getitem_44 = None
        mul_25 = query_states_11 * cos_11
        x1_10 = query_states_11[(Ellipsis, slice(None, 32, None))]
        x2_10 = query_states_11[(Ellipsis, slice(32, None, None))]
        query_states_11 = None
        neg_10 = -x2_10
        x2_10 = None
        cat_11 = torch.cat((neg_10, x1_10), dim=-1)
        neg_10 = x1_10 = None
        mul_26 = cat_11 * sin_11
        cat_11 = None
        q_embed_5 = mul_25 + mul_26
        mul_25 = mul_26 = None
        mul_27 = key_states_11 * cos_11
        cos_11 = None
        x1_11 = key_states_11[(Ellipsis, slice(None, 32, None))]
        x2_11 = key_states_11[(Ellipsis, slice(32, None, None))]
        key_states_11 = None
        neg_11 = -x2_11
        x2_11 = None
        cat_12 = torch.cat((neg_11, x1_11), dim=-1)
        neg_11 = x1_11 = None
        mul_28 = cat_12 * sin_11
        cat_12 = sin_11 = None
        k_embed_5 = mul_27 + mul_28
        mul_27 = mul_28 = None
        attn_output_20 = torch._C._nn.scaled_dot_product_attention(
            q_embed_5, k_embed_5, value_states_11, causal_4d_mask, dropout_p=0.0
        )
        q_embed_5 = k_embed_5 = value_states_11 = None
        transpose_23 = attn_output_20.transpose(1, 2)
        attn_output_20 = None
        attn_output_21 = transpose_23.contiguous()
        transpose_23 = None
        attn_output_22 = attn_output_21.reshape(1, 180, 768)
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
            (768,),
            l_self_modules_model_modules_layers_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_model_modules_layers_modules_5_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_5_modules_norm2_parameters_weight_ = (
            l_self_modules_model_modules_layers_modules_5_modules_norm2_parameters_bias_
        ) = None
        linear_42 = torch._C._nn.linear(
            hidden_states_27,
            l_self_modules_model_modules_layers_modules_5_modules_ffn_layer_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_5_modules_ffn_layer_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_6 = torch.nn.functional.silu(linear_42, inplace=False)
        linear_42 = None
        linear_43 = torch._C._nn.linear(
            hidden_states_27,
            l_self_modules_model_modules_layers_modules_5_modules_ffn_layer_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_27 = l_self_modules_model_modules_layers_modules_5_modules_ffn_layer_modules_up_proj_parameters_weight_ = (None)
        mul_29 = silu_6 * linear_43
        silu_6 = linear_43 = None
        hidden_states_28 = torch._C._nn.linear(
            mul_29,
            l_self_modules_model_modules_layers_modules_5_modules_ffn_layer_modules_down_proj_parameters_weight_,
            None,
        )
        mul_29 = l_self_modules_model_modules_layers_modules_5_modules_ffn_layer_modules_down_proj_parameters_weight_ = (None)
        hidden_states_29 = hidden_states_26 + hidden_states_28
        hidden_states_26 = hidden_states_28 = None
        hidden_states_30 = torch.nn.functional.layer_norm(
            hidden_states_29,
            (768,),
            l_self_modules_model_modules_layers_modules_6_modules_norm1_parameters_weight_,
            l_self_modules_model_modules_layers_modules_6_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_6_modules_norm1_parameters_weight_ = (
            l_self_modules_model_modules_layers_modules_6_modules_norm1_parameters_bias_
        ) = None
        query_states_12 = torch._C._nn.linear(
            hidden_states_30,
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        key_states_12 = torch._C._nn.linear(
            hidden_states_30,
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_12 = torch._C._nn.linear(
            hidden_states_30,
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_30 = l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_20 = query_states_12.view(1, 180, 12, 64)
        query_states_12 = None
        query_states_13 = view_20.transpose(1, 2)
        view_20 = None
        view_21 = key_states_12.view(1, 180, 12, 64)
        key_states_12 = None
        key_states_13 = view_21.transpose(1, 2)
        view_21 = None
        view_22 = value_states_12.view(1, 180, 12, 64)
        value_states_12 = None
        value_states_13 = view_22.transpose(1, 2)
        view_22 = None
        getitem_49 = l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_rotary_emb_buffers_cos_cached_[
            slice(None, 180, None)
        ]
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = (
            None
        )
        cos_12 = getitem_49.to(dtype=torch.float32)
        getitem_49 = None
        getitem_50 = l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_rotary_emb_buffers_sin_cached_[
            slice(None, 180, None)
        ]
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = (
            None
        )
        sin_12 = getitem_50.to(dtype=torch.float32)
        getitem_50 = None
        getitem_51 = cos_12[position_ids_1]
        cos_12 = None
        cos_13 = getitem_51.unsqueeze(1)
        getitem_51 = None
        getitem_52 = sin_12[position_ids_1]
        sin_12 = None
        sin_13 = getitem_52.unsqueeze(1)
        getitem_52 = None
        mul_30 = query_states_13 * cos_13
        x1_12 = query_states_13[(Ellipsis, slice(None, 32, None))]
        x2_12 = query_states_13[(Ellipsis, slice(32, None, None))]
        query_states_13 = None
        neg_12 = -x2_12
        x2_12 = None
        cat_13 = torch.cat((neg_12, x1_12), dim=-1)
        neg_12 = x1_12 = None
        mul_31 = cat_13 * sin_13
        cat_13 = None
        q_embed_6 = mul_30 + mul_31
        mul_30 = mul_31 = None
        mul_32 = key_states_13 * cos_13
        cos_13 = None
        x1_13 = key_states_13[(Ellipsis, slice(None, 32, None))]
        x2_13 = key_states_13[(Ellipsis, slice(32, None, None))]
        key_states_13 = None
        neg_13 = -x2_13
        x2_13 = None
        cat_14 = torch.cat((neg_13, x1_13), dim=-1)
        neg_13 = x1_13 = None
        mul_33 = cat_14 * sin_13
        cat_14 = sin_13 = None
        k_embed_6 = mul_32 + mul_33
        mul_32 = mul_33 = None
        attn_output_24 = torch._C._nn.scaled_dot_product_attention(
            q_embed_6, k_embed_6, value_states_13, causal_4d_mask, dropout_p=0.0
        )
        q_embed_6 = k_embed_6 = value_states_13 = None
        transpose_27 = attn_output_24.transpose(1, 2)
        attn_output_24 = None
        attn_output_25 = transpose_27.contiguous()
        transpose_27 = None
        attn_output_26 = attn_output_25.reshape(1, 180, 768)
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
            (768,),
            l_self_modules_model_modules_layers_modules_6_modules_norm2_parameters_weight_,
            l_self_modules_model_modules_layers_modules_6_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_6_modules_norm2_parameters_weight_ = (
            l_self_modules_model_modules_layers_modules_6_modules_norm2_parameters_bias_
        ) = None
        linear_49 = torch._C._nn.linear(
            hidden_states_32,
            l_self_modules_model_modules_layers_modules_6_modules_ffn_layer_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_6_modules_ffn_layer_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_7 = torch.nn.functional.silu(linear_49, inplace=False)
        linear_49 = None
        linear_50 = torch._C._nn.linear(
            hidden_states_32,
            l_self_modules_model_modules_layers_modules_6_modules_ffn_layer_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_32 = l_self_modules_model_modules_layers_modules_6_modules_ffn_layer_modules_up_proj_parameters_weight_ = (None)
        mul_34 = silu_7 * linear_50
        silu_7 = linear_50 = None
        hidden_states_33 = torch._C._nn.linear(
            mul_34,
            l_self_modules_model_modules_layers_modules_6_modules_ffn_layer_modules_down_proj_parameters_weight_,
            None,
        )
        mul_34 = l_self_modules_model_modules_layers_modules_6_modules_ffn_layer_modules_down_proj_parameters_weight_ = (None)
        hidden_states_34 = hidden_states_31 + hidden_states_33
        hidden_states_31 = hidden_states_33 = None
        hidden_states_35 = torch.nn.functional.layer_norm(
            hidden_states_34,
            (768,),
            l_self_modules_model_modules_layers_modules_7_modules_norm1_parameters_weight_,
            l_self_modules_model_modules_layers_modules_7_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_7_modules_norm1_parameters_weight_ = (
            l_self_modules_model_modules_layers_modules_7_modules_norm1_parameters_bias_
        ) = None
        query_states_14 = torch._C._nn.linear(
            hidden_states_35,
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        key_states_14 = torch._C._nn.linear(
            hidden_states_35,
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_14 = torch._C._nn.linear(
            hidden_states_35,
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_35 = l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_23 = query_states_14.view(1, 180, 12, 64)
        query_states_14 = None
        query_states_15 = view_23.transpose(1, 2)
        view_23 = None
        view_24 = key_states_14.view(1, 180, 12, 64)
        key_states_14 = None
        key_states_15 = view_24.transpose(1, 2)
        view_24 = None
        view_25 = value_states_14.view(1, 180, 12, 64)
        value_states_14 = None
        value_states_15 = view_25.transpose(1, 2)
        view_25 = None
        getitem_57 = l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_rotary_emb_buffers_cos_cached_[
            slice(None, 180, None)
        ]
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = (
            None
        )
        cos_14 = getitem_57.to(dtype=torch.float32)
        getitem_57 = None
        getitem_58 = l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_rotary_emb_buffers_sin_cached_[
            slice(None, 180, None)
        ]
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = (
            None
        )
        sin_14 = getitem_58.to(dtype=torch.float32)
        getitem_58 = None
        getitem_59 = cos_14[position_ids_1]
        cos_14 = None
        cos_15 = getitem_59.unsqueeze(1)
        getitem_59 = None
        getitem_60 = sin_14[position_ids_1]
        sin_14 = None
        sin_15 = getitem_60.unsqueeze(1)
        getitem_60 = None
        mul_35 = query_states_15 * cos_15
        x1_14 = query_states_15[(Ellipsis, slice(None, 32, None))]
        x2_14 = query_states_15[(Ellipsis, slice(32, None, None))]
        query_states_15 = None
        neg_14 = -x2_14
        x2_14 = None
        cat_15 = torch.cat((neg_14, x1_14), dim=-1)
        neg_14 = x1_14 = None
        mul_36 = cat_15 * sin_15
        cat_15 = None
        q_embed_7 = mul_35 + mul_36
        mul_35 = mul_36 = None
        mul_37 = key_states_15 * cos_15
        cos_15 = None
        x1_15 = key_states_15[(Ellipsis, slice(None, 32, None))]
        x2_15 = key_states_15[(Ellipsis, slice(32, None, None))]
        key_states_15 = None
        neg_15 = -x2_15
        x2_15 = None
        cat_16 = torch.cat((neg_15, x1_15), dim=-1)
        neg_15 = x1_15 = None
        mul_38 = cat_16 * sin_15
        cat_16 = sin_15 = None
        k_embed_7 = mul_37 + mul_38
        mul_37 = mul_38 = None
        attn_output_28 = torch._C._nn.scaled_dot_product_attention(
            q_embed_7, k_embed_7, value_states_15, causal_4d_mask, dropout_p=0.0
        )
        q_embed_7 = k_embed_7 = value_states_15 = None
        transpose_31 = attn_output_28.transpose(1, 2)
        attn_output_28 = None
        attn_output_29 = transpose_31.contiguous()
        transpose_31 = None
        attn_output_30 = attn_output_29.reshape(1, 180, 768)
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
            (768,),
            l_self_modules_model_modules_layers_modules_7_modules_norm2_parameters_weight_,
            l_self_modules_model_modules_layers_modules_7_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_7_modules_norm2_parameters_weight_ = (
            l_self_modules_model_modules_layers_modules_7_modules_norm2_parameters_bias_
        ) = None
        linear_56 = torch._C._nn.linear(
            hidden_states_37,
            l_self_modules_model_modules_layers_modules_7_modules_ffn_layer_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_7_modules_ffn_layer_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_8 = torch.nn.functional.silu(linear_56, inplace=False)
        linear_56 = None
        linear_57 = torch._C._nn.linear(
            hidden_states_37,
            l_self_modules_model_modules_layers_modules_7_modules_ffn_layer_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_37 = l_self_modules_model_modules_layers_modules_7_modules_ffn_layer_modules_up_proj_parameters_weight_ = (None)
        mul_39 = silu_8 * linear_57
        silu_8 = linear_57 = None
        hidden_states_38 = torch._C._nn.linear(
            mul_39,
            l_self_modules_model_modules_layers_modules_7_modules_ffn_layer_modules_down_proj_parameters_weight_,
            None,
        )
        mul_39 = l_self_modules_model_modules_layers_modules_7_modules_ffn_layer_modules_down_proj_parameters_weight_ = (None)
        hidden_states_39 = hidden_states_36 + hidden_states_38
        hidden_states_36 = hidden_states_38 = None
        hidden_states_40 = torch.nn.functional.layer_norm(
            hidden_states_39,
            (768,),
            l_self_modules_model_modules_layers_modules_8_modules_norm1_parameters_weight_,
            l_self_modules_model_modules_layers_modules_8_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_8_modules_norm1_parameters_weight_ = (
            l_self_modules_model_modules_layers_modules_8_modules_norm1_parameters_bias_
        ) = None
        query_states_16 = torch._C._nn.linear(
            hidden_states_40,
            l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        key_states_16 = torch._C._nn.linear(
            hidden_states_40,
            l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_16 = torch._C._nn.linear(
            hidden_states_40,
            l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_40 = l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_26 = query_states_16.view(1, 180, 12, 64)
        query_states_16 = None
        query_states_17 = view_26.transpose(1, 2)
        view_26 = None
        view_27 = key_states_16.view(1, 180, 12, 64)
        key_states_16 = None
        key_states_17 = view_27.transpose(1, 2)
        view_27 = None
        view_28 = value_states_16.view(1, 180, 12, 64)
        value_states_16 = None
        value_states_17 = view_28.transpose(1, 2)
        view_28 = None
        getitem_65 = l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_rotary_emb_buffers_cos_cached_[
            slice(None, 180, None)
        ]
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = (
            None
        )
        cos_16 = getitem_65.to(dtype=torch.float32)
        getitem_65 = None
        getitem_66 = l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_rotary_emb_buffers_sin_cached_[
            slice(None, 180, None)
        ]
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = (
            None
        )
        sin_16 = getitem_66.to(dtype=torch.float32)
        getitem_66 = None
        getitem_67 = cos_16[position_ids_1]
        cos_16 = None
        cos_17 = getitem_67.unsqueeze(1)
        getitem_67 = None
        getitem_68 = sin_16[position_ids_1]
        sin_16 = None
        sin_17 = getitem_68.unsqueeze(1)
        getitem_68 = None
        mul_40 = query_states_17 * cos_17
        x1_16 = query_states_17[(Ellipsis, slice(None, 32, None))]
        x2_16 = query_states_17[(Ellipsis, slice(32, None, None))]
        query_states_17 = None
        neg_16 = -x2_16
        x2_16 = None
        cat_17 = torch.cat((neg_16, x1_16), dim=-1)
        neg_16 = x1_16 = None
        mul_41 = cat_17 * sin_17
        cat_17 = None
        q_embed_8 = mul_40 + mul_41
        mul_40 = mul_41 = None
        mul_42 = key_states_17 * cos_17
        cos_17 = None
        x1_17 = key_states_17[(Ellipsis, slice(None, 32, None))]
        x2_17 = key_states_17[(Ellipsis, slice(32, None, None))]
        key_states_17 = None
        neg_17 = -x2_17
        x2_17 = None
        cat_18 = torch.cat((neg_17, x1_17), dim=-1)
        neg_17 = x1_17 = None
        mul_43 = cat_18 * sin_17
        cat_18 = sin_17 = None
        k_embed_8 = mul_42 + mul_43
        mul_42 = mul_43 = None
        attn_output_32 = torch._C._nn.scaled_dot_product_attention(
            q_embed_8, k_embed_8, value_states_17, causal_4d_mask, dropout_p=0.0
        )
        q_embed_8 = k_embed_8 = value_states_17 = None
        transpose_35 = attn_output_32.transpose(1, 2)
        attn_output_32 = None
        attn_output_33 = transpose_35.contiguous()
        transpose_35 = None
        attn_output_34 = attn_output_33.reshape(1, 180, 768)
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
            (768,),
            l_self_modules_model_modules_layers_modules_8_modules_norm2_parameters_weight_,
            l_self_modules_model_modules_layers_modules_8_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_8_modules_norm2_parameters_weight_ = (
            l_self_modules_model_modules_layers_modules_8_modules_norm2_parameters_bias_
        ) = None
        linear_63 = torch._C._nn.linear(
            hidden_states_42,
            l_self_modules_model_modules_layers_modules_8_modules_ffn_layer_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_8_modules_ffn_layer_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_9 = torch.nn.functional.silu(linear_63, inplace=False)
        linear_63 = None
        linear_64 = torch._C._nn.linear(
            hidden_states_42,
            l_self_modules_model_modules_layers_modules_8_modules_ffn_layer_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_42 = l_self_modules_model_modules_layers_modules_8_modules_ffn_layer_modules_up_proj_parameters_weight_ = (None)
        mul_44 = silu_9 * linear_64
        silu_9 = linear_64 = None
        hidden_states_43 = torch._C._nn.linear(
            mul_44,
            l_self_modules_model_modules_layers_modules_8_modules_ffn_layer_modules_down_proj_parameters_weight_,
            None,
        )
        mul_44 = l_self_modules_model_modules_layers_modules_8_modules_ffn_layer_modules_down_proj_parameters_weight_ = (None)
        hidden_states_44 = hidden_states_41 + hidden_states_43
        hidden_states_41 = hidden_states_43 = None
        hidden_states_45 = torch.nn.functional.layer_norm(
            hidden_states_44,
            (768,),
            l_self_modules_model_modules_layers_modules_9_modules_norm1_parameters_weight_,
            l_self_modules_model_modules_layers_modules_9_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_9_modules_norm1_parameters_weight_ = (
            l_self_modules_model_modules_layers_modules_9_modules_norm1_parameters_bias_
        ) = None
        query_states_18 = torch._C._nn.linear(
            hidden_states_45,
            l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        key_states_18 = torch._C._nn.linear(
            hidden_states_45,
            l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_18 = torch._C._nn.linear(
            hidden_states_45,
            l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_45 = l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_29 = query_states_18.view(1, 180, 12, 64)
        query_states_18 = None
        query_states_19 = view_29.transpose(1, 2)
        view_29 = None
        view_30 = key_states_18.view(1, 180, 12, 64)
        key_states_18 = None
        key_states_19 = view_30.transpose(1, 2)
        view_30 = None
        view_31 = value_states_18.view(1, 180, 12, 64)
        value_states_18 = None
        value_states_19 = view_31.transpose(1, 2)
        view_31 = None
        getitem_73 = l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_rotary_emb_buffers_cos_cached_[
            slice(None, 180, None)
        ]
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = (
            None
        )
        cos_18 = getitem_73.to(dtype=torch.float32)
        getitem_73 = None
        getitem_74 = l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_rotary_emb_buffers_sin_cached_[
            slice(None, 180, None)
        ]
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = (
            None
        )
        sin_18 = getitem_74.to(dtype=torch.float32)
        getitem_74 = None
        getitem_75 = cos_18[position_ids_1]
        cos_18 = None
        cos_19 = getitem_75.unsqueeze(1)
        getitem_75 = None
        getitem_76 = sin_18[position_ids_1]
        sin_18 = None
        sin_19 = getitem_76.unsqueeze(1)
        getitem_76 = None
        mul_45 = query_states_19 * cos_19
        x1_18 = query_states_19[(Ellipsis, slice(None, 32, None))]
        x2_18 = query_states_19[(Ellipsis, slice(32, None, None))]
        query_states_19 = None
        neg_18 = -x2_18
        x2_18 = None
        cat_19 = torch.cat((neg_18, x1_18), dim=-1)
        neg_18 = x1_18 = None
        mul_46 = cat_19 * sin_19
        cat_19 = None
        q_embed_9 = mul_45 + mul_46
        mul_45 = mul_46 = None
        mul_47 = key_states_19 * cos_19
        cos_19 = None
        x1_19 = key_states_19[(Ellipsis, slice(None, 32, None))]
        x2_19 = key_states_19[(Ellipsis, slice(32, None, None))]
        key_states_19 = None
        neg_19 = -x2_19
        x2_19 = None
        cat_20 = torch.cat((neg_19, x1_19), dim=-1)
        neg_19 = x1_19 = None
        mul_48 = cat_20 * sin_19
        cat_20 = sin_19 = None
        k_embed_9 = mul_47 + mul_48
        mul_47 = mul_48 = None
        attn_output_36 = torch._C._nn.scaled_dot_product_attention(
            q_embed_9, k_embed_9, value_states_19, causal_4d_mask, dropout_p=0.0
        )
        q_embed_9 = k_embed_9 = value_states_19 = None
        transpose_39 = attn_output_36.transpose(1, 2)
        attn_output_36 = None
        attn_output_37 = transpose_39.contiguous()
        transpose_39 = None
        attn_output_38 = attn_output_37.reshape(1, 180, 768)
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
            (768,),
            l_self_modules_model_modules_layers_modules_9_modules_norm2_parameters_weight_,
            l_self_modules_model_modules_layers_modules_9_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_9_modules_norm2_parameters_weight_ = (
            l_self_modules_model_modules_layers_modules_9_modules_norm2_parameters_bias_
        ) = None
        linear_70 = torch._C._nn.linear(
            hidden_states_47,
            l_self_modules_model_modules_layers_modules_9_modules_ffn_layer_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_9_modules_ffn_layer_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_10 = torch.nn.functional.silu(linear_70, inplace=False)
        linear_70 = None
        linear_71 = torch._C._nn.linear(
            hidden_states_47,
            l_self_modules_model_modules_layers_modules_9_modules_ffn_layer_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_47 = l_self_modules_model_modules_layers_modules_9_modules_ffn_layer_modules_up_proj_parameters_weight_ = (None)
        mul_49 = silu_10 * linear_71
        silu_10 = linear_71 = None
        hidden_states_48 = torch._C._nn.linear(
            mul_49,
            l_self_modules_model_modules_layers_modules_9_modules_ffn_layer_modules_down_proj_parameters_weight_,
            None,
        )
        mul_49 = l_self_modules_model_modules_layers_modules_9_modules_ffn_layer_modules_down_proj_parameters_weight_ = (None)
        hidden_states_49 = hidden_states_46 + hidden_states_48
        hidden_states_46 = hidden_states_48 = None
        hidden_states_50 = torch.nn.functional.layer_norm(
            hidden_states_49,
            (768,),
            l_self_modules_model_modules_layers_modules_10_modules_norm1_parameters_weight_,
            l_self_modules_model_modules_layers_modules_10_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_10_modules_norm1_parameters_weight_ = l_self_modules_model_modules_layers_modules_10_modules_norm1_parameters_bias_ = (None)
        query_states_20 = torch._C._nn.linear(
            hidden_states_50,
            l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        key_states_20 = torch._C._nn.linear(
            hidden_states_50,
            l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_20 = torch._C._nn.linear(
            hidden_states_50,
            l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_50 = l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_32 = query_states_20.view(1, 180, 12, 64)
        query_states_20 = None
        query_states_21 = view_32.transpose(1, 2)
        view_32 = None
        view_33 = key_states_20.view(1, 180, 12, 64)
        key_states_20 = None
        key_states_21 = view_33.transpose(1, 2)
        view_33 = None
        view_34 = value_states_20.view(1, 180, 12, 64)
        value_states_20 = None
        value_states_21 = view_34.transpose(1, 2)
        view_34 = None
        getitem_81 = l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_rotary_emb_buffers_cos_cached_[
            slice(None, 180, None)
        ]
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = (
            None
        )
        cos_20 = getitem_81.to(dtype=torch.float32)
        getitem_81 = None
        getitem_82 = l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_rotary_emb_buffers_sin_cached_[
            slice(None, 180, None)
        ]
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = (
            None
        )
        sin_20 = getitem_82.to(dtype=torch.float32)
        getitem_82 = None
        getitem_83 = cos_20[position_ids_1]
        cos_20 = None
        cos_21 = getitem_83.unsqueeze(1)
        getitem_83 = None
        getitem_84 = sin_20[position_ids_1]
        sin_20 = None
        sin_21 = getitem_84.unsqueeze(1)
        getitem_84 = None
        mul_50 = query_states_21 * cos_21
        x1_20 = query_states_21[(Ellipsis, slice(None, 32, None))]
        x2_20 = query_states_21[(Ellipsis, slice(32, None, None))]
        query_states_21 = None
        neg_20 = -x2_20
        x2_20 = None
        cat_21 = torch.cat((neg_20, x1_20), dim=-1)
        neg_20 = x1_20 = None
        mul_51 = cat_21 * sin_21
        cat_21 = None
        q_embed_10 = mul_50 + mul_51
        mul_50 = mul_51 = None
        mul_52 = key_states_21 * cos_21
        cos_21 = None
        x1_21 = key_states_21[(Ellipsis, slice(None, 32, None))]
        x2_21 = key_states_21[(Ellipsis, slice(32, None, None))]
        key_states_21 = None
        neg_21 = -x2_21
        x2_21 = None
        cat_22 = torch.cat((neg_21, x1_21), dim=-1)
        neg_21 = x1_21 = None
        mul_53 = cat_22 * sin_21
        cat_22 = sin_21 = None
        k_embed_10 = mul_52 + mul_53
        mul_52 = mul_53 = None
        attn_output_40 = torch._C._nn.scaled_dot_product_attention(
            q_embed_10, k_embed_10, value_states_21, causal_4d_mask, dropout_p=0.0
        )
        q_embed_10 = k_embed_10 = value_states_21 = None
        transpose_43 = attn_output_40.transpose(1, 2)
        attn_output_40 = None
        attn_output_41 = transpose_43.contiguous()
        transpose_43 = None
        attn_output_42 = attn_output_41.reshape(1, 180, 768)
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
            (768,),
            l_self_modules_model_modules_layers_modules_10_modules_norm2_parameters_weight_,
            l_self_modules_model_modules_layers_modules_10_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_10_modules_norm2_parameters_weight_ = l_self_modules_model_modules_layers_modules_10_modules_norm2_parameters_bias_ = (None)
        linear_77 = torch._C._nn.linear(
            hidden_states_52,
            l_self_modules_model_modules_layers_modules_10_modules_ffn_layer_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_10_modules_ffn_layer_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_11 = torch.nn.functional.silu(linear_77, inplace=False)
        linear_77 = None
        linear_78 = torch._C._nn.linear(
            hidden_states_52,
            l_self_modules_model_modules_layers_modules_10_modules_ffn_layer_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_52 = l_self_modules_model_modules_layers_modules_10_modules_ffn_layer_modules_up_proj_parameters_weight_ = (None)
        mul_54 = silu_11 * linear_78
        silu_11 = linear_78 = None
        hidden_states_53 = torch._C._nn.linear(
            mul_54,
            l_self_modules_model_modules_layers_modules_10_modules_ffn_layer_modules_down_proj_parameters_weight_,
            None,
        )
        mul_54 = l_self_modules_model_modules_layers_modules_10_modules_ffn_layer_modules_down_proj_parameters_weight_ = (None)
        hidden_states_54 = hidden_states_51 + hidden_states_53
        hidden_states_51 = hidden_states_53 = None
        hidden_states_55 = torch.nn.functional.layer_norm(
            hidden_states_54,
            (768,),
            l_self_modules_model_modules_layers_modules_11_modules_norm1_parameters_weight_,
            l_self_modules_model_modules_layers_modules_11_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_11_modules_norm1_parameters_weight_ = l_self_modules_model_modules_layers_modules_11_modules_norm1_parameters_bias_ = (None)
        query_states_22 = torch._C._nn.linear(
            hidden_states_55,
            l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        key_states_22 = torch._C._nn.linear(
            hidden_states_55,
            l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_22 = torch._C._nn.linear(
            hidden_states_55,
            l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_55 = l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_35 = query_states_22.view(1, 180, 12, 64)
        query_states_22 = None
        query_states_23 = view_35.transpose(1, 2)
        view_35 = None
        view_36 = key_states_22.view(1, 180, 12, 64)
        key_states_22 = None
        key_states_23 = view_36.transpose(1, 2)
        view_36 = None
        view_37 = value_states_22.view(1, 180, 12, 64)
        value_states_22 = None
        value_states_23 = view_37.transpose(1, 2)
        view_37 = None
        getitem_89 = l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_rotary_emb_buffers_cos_cached_[
            slice(None, 180, None)
        ]
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = (
            None
        )
        cos_22 = getitem_89.to(dtype=torch.float32)
        getitem_89 = None
        getitem_90 = l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_rotary_emb_buffers_sin_cached_[
            slice(None, 180, None)
        ]
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = (
            None
        )
        sin_22 = getitem_90.to(dtype=torch.float32)
        getitem_90 = None
        getitem_91 = cos_22[position_ids_1]
        cos_22 = None
        cos_23 = getitem_91.unsqueeze(1)
        getitem_91 = None
        getitem_92 = sin_22[position_ids_1]
        sin_22 = position_ids_1 = None
        sin_23 = getitem_92.unsqueeze(1)
        getitem_92 = None
        mul_55 = query_states_23 * cos_23
        x1_22 = query_states_23[(Ellipsis, slice(None, 32, None))]
        x2_22 = query_states_23[(Ellipsis, slice(32, None, None))]
        query_states_23 = None
        neg_22 = -x2_22
        x2_22 = None
        cat_23 = torch.cat((neg_22, x1_22), dim=-1)
        neg_22 = x1_22 = None
        mul_56 = cat_23 * sin_23
        cat_23 = None
        q_embed_11 = mul_55 + mul_56
        mul_55 = mul_56 = None
        mul_57 = key_states_23 * cos_23
        cos_23 = None
        x1_23 = key_states_23[(Ellipsis, slice(None, 32, None))]
        x2_23 = key_states_23[(Ellipsis, slice(32, None, None))]
        key_states_23 = None
        neg_23 = -x2_23
        x2_23 = None
        cat_24 = torch.cat((neg_23, x1_23), dim=-1)
        neg_23 = x1_23 = None
        mul_58 = cat_24 * sin_23
        cat_24 = sin_23 = None
        k_embed_11 = mul_57 + mul_58
        mul_57 = mul_58 = None
        attn_output_44 = torch._C._nn.scaled_dot_product_attention(
            q_embed_11, k_embed_11, value_states_23, causal_4d_mask, dropout_p=0.0
        )
        q_embed_11 = k_embed_11 = value_states_23 = causal_4d_mask = None
        transpose_47 = attn_output_44.transpose(1, 2)
        attn_output_44 = None
        attn_output_45 = transpose_47.contiguous()
        transpose_47 = None
        attn_output_46 = attn_output_45.reshape(1, 180, 768)
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
            (768,),
            l_self_modules_model_modules_layers_modules_11_modules_norm2_parameters_weight_,
            l_self_modules_model_modules_layers_modules_11_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_11_modules_norm2_parameters_weight_ = l_self_modules_model_modules_layers_modules_11_modules_norm2_parameters_bias_ = (None)
        linear_84 = torch._C._nn.linear(
            hidden_states_57,
            l_self_modules_model_modules_layers_modules_11_modules_ffn_layer_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_11_modules_ffn_layer_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_12 = torch.nn.functional.silu(linear_84, inplace=False)
        linear_84 = None
        linear_85 = torch._C._nn.linear(
            hidden_states_57,
            l_self_modules_model_modules_layers_modules_11_modules_ffn_layer_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_57 = l_self_modules_model_modules_layers_modules_11_modules_ffn_layer_modules_up_proj_parameters_weight_ = (None)
        mul_59 = silu_12 * linear_85
        silu_12 = linear_85 = None
        hidden_states_58 = torch._C._nn.linear(
            mul_59,
            l_self_modules_model_modules_layers_modules_11_modules_ffn_layer_modules_down_proj_parameters_weight_,
            None,
        )
        mul_59 = l_self_modules_model_modules_layers_modules_11_modules_ffn_layer_modules_down_proj_parameters_weight_ = (None)
        hidden_states_59 = hidden_states_56 + hidden_states_58
        hidden_states_56 = hidden_states_58 = None
        hidden_states_60 = torch.nn.functional.layer_norm(
            hidden_states_59,
            (768,),
            l_self_modules_model_modules_norm_parameters_weight_,
            l_self_modules_model_modules_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_59 = (
            l_self_modules_model_modules_norm_parameters_weight_
        ) = l_self_modules_model_modules_norm_parameters_bias_ = None
        hidden_states_61 = hidden_states_60[
            (slice(None, None, None), -1, slice(None, None, None))
        ]
        hidden_states_60 = None
        z = hidden_states_61.repeat(1, 1)
        hidden_states_61 = None
        randn = torch.randn(1, 720)
        noise = randn.to(device(type="cuda", index=0))
        randn = None
        ones = torch.ones(1)
        mul_60 = ones * 0
        ones = None
        truediv = mul_60 / 50
        mul_60 = None
        t = truediv.to(device(type="cuda", index=0))
        truediv = None
        mul_61 = t * 1000
        t = None
        x_3 = torch._C._nn.linear(
            noise,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_2 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_62 = -9.210340371976184 * arange_2
        arange_2 = None
        truediv_1 = mul_62 / 128
        mul_62 = None
        exp = torch.exp(truediv_1)
        truediv_1 = None
        freqs = exp.to(device=device(type="cuda", index=0))
        exp = None
        getitem_98 = mul_61[(slice(None, None, None), None)]
        mul_61 = None
        float_1 = getitem_98.float()
        getitem_98 = None
        getitem_99 = freqs[None]
        freqs = None
        args = float_1 * getitem_99
        float_1 = getitem_99 = None
        cos_24 = torch.cos(args)
        sin_24 = torch.sin(args)
        args = None
        embedding = torch.cat([cos_24, sin_24], dim=-1)
        cos_24 = sin_24 = None
        input_1 = torch._C._nn.linear(
            embedding,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding = None
        input_2 = torch.nn.functional.silu(input_1, inplace=False)
        input_1 = None
        input_3 = torch._C._nn.linear(
            input_2,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_2 = None
        c = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y = input_3 + c
        input_3 = c = None
        input_4 = torch.nn.functional.silu(y, inplace=False)
        input_5 = torch._C._nn.linear(
            input_4,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_4 = None
        chunk = input_5.chunk(3, dim=-1)
        input_5 = None
        shift_mlp = chunk[0]
        scale_mlp = chunk[1]
        gate_mlp = chunk[2]
        chunk = None
        layer_norm_25 = torch.nn.functional.layer_norm(
            x_3,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_51 = 1 + scale_mlp
        scale_mlp = None
        mul_64 = layer_norm_25 * add_51
        layer_norm_25 = add_51 = None
        h = mul_64 + shift_mlp
        mul_64 = shift_mlp = None
        input_6 = torch._C._nn.linear(
            h,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h = None
        input_7 = torch.nn.functional.silu(input_6, inplace=False)
        input_6 = None
        input_8 = torch._C._nn.linear(
            input_7,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_7 = None
        mul_65 = gate_mlp * input_8
        gate_mlp = input_8 = None
        x_4 = x_3 + mul_65
        x_3 = mul_65 = None
        input_9 = torch.nn.functional.silu(y, inplace=False)
        input_10 = torch._C._nn.linear(
            input_9,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_9 = None
        chunk_1 = input_10.chunk(3, dim=-1)
        input_10 = None
        shift_mlp_1 = chunk_1[0]
        scale_mlp_1 = chunk_1[1]
        gate_mlp_1 = chunk_1[2]
        chunk_1 = None
        layer_norm_26 = torch.nn.functional.layer_norm(
            x_4,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_54 = 1 + scale_mlp_1
        scale_mlp_1 = None
        mul_66 = layer_norm_26 * add_54
        layer_norm_26 = add_54 = None
        h_1 = mul_66 + shift_mlp_1
        mul_66 = shift_mlp_1 = None
        input_11 = torch._C._nn.linear(
            h_1,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_1 = None
        input_12 = torch.nn.functional.silu(input_11, inplace=False)
        input_11 = None
        input_13 = torch._C._nn.linear(
            input_12,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_12 = None
        mul_67 = gate_mlp_1 * input_13
        gate_mlp_1 = input_13 = None
        x_5 = x_4 + mul_67
        x_4 = mul_67 = None
        input_14 = torch.nn.functional.silu(y, inplace=False)
        input_15 = torch._C._nn.linear(
            input_14,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_14 = None
        chunk_2 = input_15.chunk(3, dim=-1)
        input_15 = None
        shift_mlp_2 = chunk_2[0]
        scale_mlp_2 = chunk_2[1]
        gate_mlp_2 = chunk_2[2]
        chunk_2 = None
        layer_norm_27 = torch.nn.functional.layer_norm(
            x_5,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_57 = 1 + scale_mlp_2
        scale_mlp_2 = None
        mul_68 = layer_norm_27 * add_57
        layer_norm_27 = add_57 = None
        h_2 = mul_68 + shift_mlp_2
        mul_68 = shift_mlp_2 = None
        input_16 = torch._C._nn.linear(
            h_2,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_2 = None
        input_17 = torch.nn.functional.silu(input_16, inplace=False)
        input_16 = None
        input_18 = torch._C._nn.linear(
            input_17,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_17 = None
        mul_69 = gate_mlp_2 * input_18
        gate_mlp_2 = input_18 = None
        x_6 = x_5 + mul_69
        x_5 = mul_69 = None
        input_19 = torch.nn.functional.silu(y, inplace=False)
        y = None
        input_20 = torch._C._nn.linear(
            input_19,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_19 = None
        chunk_3 = input_20.chunk(2, dim=-1)
        input_20 = None
        shift = chunk_3[0]
        scale = chunk_3[1]
        chunk_3 = None
        layer_norm_28 = torch.nn.functional.layer_norm(x_6, (768,), None, None, 1e-06)
        x_6 = None
        add_60 = 1 + scale
        scale = None
        mul_70 = layer_norm_28 * add_60
        layer_norm_28 = add_60 = None
        x_7 = mul_70 + shift
        mul_70 = shift = None
        x_8 = torch._C._nn.linear(
            x_7,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_7 = None
        sub = x_8 - noise
        x_8 = None
        mul_71 = sub * 0.02
        sub = None
        x_9 = noise + mul_71
        mul_71 = None
        ones_1 = torch.ones(1)
        mul_72 = ones_1 * 1
        ones_1 = None
        truediv_2 = mul_72 / 50
        mul_72 = None
        t_1 = truediv_2.to(device(type="cuda", index=0))
        truediv_2 = None
        mul_73 = t_1 * 1000
        t_1 = None
        x_10 = torch._C._nn.linear(
            x_9,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_3 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_74 = -9.210340371976184 * arange_3
        arange_3 = None
        truediv_3 = mul_74 / 128
        mul_74 = None
        exp_1 = torch.exp(truediv_3)
        truediv_3 = None
        freqs_1 = exp_1.to(device=device(type="cuda", index=0))
        exp_1 = None
        getitem_111 = mul_73[(slice(None, None, None), None)]
        mul_73 = None
        float_2 = getitem_111.float()
        getitem_111 = None
        getitem_112 = freqs_1[None]
        freqs_1 = None
        args_1 = float_2 * getitem_112
        float_2 = getitem_112 = None
        cos_25 = torch.cos(args_1)
        sin_25 = torch.sin(args_1)
        args_1 = None
        embedding_1 = torch.cat([cos_25, sin_25], dim=-1)
        cos_25 = sin_25 = None
        input_21 = torch._C._nn.linear(
            embedding_1,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_1 = None
        input_22 = torch.nn.functional.silu(input_21, inplace=False)
        input_21 = None
        input_23 = torch._C._nn.linear(
            input_22,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_22 = None
        c_1 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y_1 = input_23 + c_1
        input_23 = c_1 = None
        input_24 = torch.nn.functional.silu(y_1, inplace=False)
        input_25 = torch._C._nn.linear(
            input_24,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_24 = None
        chunk_4 = input_25.chunk(3, dim=-1)
        input_25 = None
        shift_mlp_3 = chunk_4[0]
        scale_mlp_3 = chunk_4[1]
        gate_mlp_3 = chunk_4[2]
        chunk_4 = None
        layer_norm_29 = torch.nn.functional.layer_norm(
            x_10,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_64 = 1 + scale_mlp_3
        scale_mlp_3 = None
        mul_76 = layer_norm_29 * add_64
        layer_norm_29 = add_64 = None
        h_3 = mul_76 + shift_mlp_3
        mul_76 = shift_mlp_3 = None
        input_26 = torch._C._nn.linear(
            h_3,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_3 = None
        input_27 = torch.nn.functional.silu(input_26, inplace=False)
        input_26 = None
        input_28 = torch._C._nn.linear(
            input_27,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_27 = None
        mul_77 = gate_mlp_3 * input_28
        gate_mlp_3 = input_28 = None
        x_11 = x_10 + mul_77
        x_10 = mul_77 = None
        input_29 = torch.nn.functional.silu(y_1, inplace=False)
        input_30 = torch._C._nn.linear(
            input_29,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_29 = None
        chunk_5 = input_30.chunk(3, dim=-1)
        input_30 = None
        shift_mlp_4 = chunk_5[0]
        scale_mlp_4 = chunk_5[1]
        gate_mlp_4 = chunk_5[2]
        chunk_5 = None
        layer_norm_30 = torch.nn.functional.layer_norm(
            x_11,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_67 = 1 + scale_mlp_4
        scale_mlp_4 = None
        mul_78 = layer_norm_30 * add_67
        layer_norm_30 = add_67 = None
        h_4 = mul_78 + shift_mlp_4
        mul_78 = shift_mlp_4 = None
        input_31 = torch._C._nn.linear(
            h_4,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_4 = None
        input_32 = torch.nn.functional.silu(input_31, inplace=False)
        input_31 = None
        input_33 = torch._C._nn.linear(
            input_32,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_32 = None
        mul_79 = gate_mlp_4 * input_33
        gate_mlp_4 = input_33 = None
        x_12 = x_11 + mul_79
        x_11 = mul_79 = None
        input_34 = torch.nn.functional.silu(y_1, inplace=False)
        input_35 = torch._C._nn.linear(
            input_34,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_34 = None
        chunk_6 = input_35.chunk(3, dim=-1)
        input_35 = None
        shift_mlp_5 = chunk_6[0]
        scale_mlp_5 = chunk_6[1]
        gate_mlp_5 = chunk_6[2]
        chunk_6 = None
        layer_norm_31 = torch.nn.functional.layer_norm(
            x_12,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_70 = 1 + scale_mlp_5
        scale_mlp_5 = None
        mul_80 = layer_norm_31 * add_70
        layer_norm_31 = add_70 = None
        h_5 = mul_80 + shift_mlp_5
        mul_80 = shift_mlp_5 = None
        input_36 = torch._C._nn.linear(
            h_5,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_5 = None
        input_37 = torch.nn.functional.silu(input_36, inplace=False)
        input_36 = None
        input_38 = torch._C._nn.linear(
            input_37,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_37 = None
        mul_81 = gate_mlp_5 * input_38
        gate_mlp_5 = input_38 = None
        x_13 = x_12 + mul_81
        x_12 = mul_81 = None
        input_39 = torch.nn.functional.silu(y_1, inplace=False)
        y_1 = None
        input_40 = torch._C._nn.linear(
            input_39,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_39 = None
        chunk_7 = input_40.chunk(2, dim=-1)
        input_40 = None
        shift_1 = chunk_7[0]
        scale_1 = chunk_7[1]
        chunk_7 = None
        layer_norm_32 = torch.nn.functional.layer_norm(x_13, (768,), None, None, 1e-06)
        x_13 = None
        add_73 = 1 + scale_1
        scale_1 = None
        mul_82 = layer_norm_32 * add_73
        layer_norm_32 = add_73 = None
        x_14 = mul_82 + shift_1
        mul_82 = shift_1 = None
        x_15 = torch._C._nn.linear(
            x_14,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_14 = None
        sub_1 = x_15 - noise
        x_15 = None
        mul_83 = sub_1 * 0.02
        sub_1 = None
        x_16 = x_9 + mul_83
        x_9 = mul_83 = None
        ones_2 = torch.ones(1)
        mul_84 = ones_2 * 2
        ones_2 = None
        truediv_4 = mul_84 / 50
        mul_84 = None
        t_2 = truediv_4.to(device(type="cuda", index=0))
        truediv_4 = None
        mul_85 = t_2 * 1000
        t_2 = None
        x_17 = torch._C._nn.linear(
            x_16,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_4 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_86 = -9.210340371976184 * arange_4
        arange_4 = None
        truediv_5 = mul_86 / 128
        mul_86 = None
        exp_2 = torch.exp(truediv_5)
        truediv_5 = None
        freqs_2 = exp_2.to(device=device(type="cuda", index=0))
        exp_2 = None
        getitem_124 = mul_85[(slice(None, None, None), None)]
        mul_85 = None
        float_3 = getitem_124.float()
        getitem_124 = None
        getitem_125 = freqs_2[None]
        freqs_2 = None
        args_2 = float_3 * getitem_125
        float_3 = getitem_125 = None
        cos_26 = torch.cos(args_2)
        sin_26 = torch.sin(args_2)
        args_2 = None
        embedding_2 = torch.cat([cos_26, sin_26], dim=-1)
        cos_26 = sin_26 = None
        input_41 = torch._C._nn.linear(
            embedding_2,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_2 = None
        input_42 = torch.nn.functional.silu(input_41, inplace=False)
        input_41 = None
        input_43 = torch._C._nn.linear(
            input_42,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_42 = None
        c_2 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y_2 = input_43 + c_2
        input_43 = c_2 = None
        input_44 = torch.nn.functional.silu(y_2, inplace=False)
        input_45 = torch._C._nn.linear(
            input_44,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_44 = None
        chunk_8 = input_45.chunk(3, dim=-1)
        input_45 = None
        shift_mlp_6 = chunk_8[0]
        scale_mlp_6 = chunk_8[1]
        gate_mlp_6 = chunk_8[2]
        chunk_8 = None
        layer_norm_33 = torch.nn.functional.layer_norm(
            x_17,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_77 = 1 + scale_mlp_6
        scale_mlp_6 = None
        mul_88 = layer_norm_33 * add_77
        layer_norm_33 = add_77 = None
        h_6 = mul_88 + shift_mlp_6
        mul_88 = shift_mlp_6 = None
        input_46 = torch._C._nn.linear(
            h_6,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_6 = None
        input_47 = torch.nn.functional.silu(input_46, inplace=False)
        input_46 = None
        input_48 = torch._C._nn.linear(
            input_47,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_47 = None
        mul_89 = gate_mlp_6 * input_48
        gate_mlp_6 = input_48 = None
        x_18 = x_17 + mul_89
        x_17 = mul_89 = None
        input_49 = torch.nn.functional.silu(y_2, inplace=False)
        input_50 = torch._C._nn.linear(
            input_49,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_49 = None
        chunk_9 = input_50.chunk(3, dim=-1)
        input_50 = None
        shift_mlp_7 = chunk_9[0]
        scale_mlp_7 = chunk_9[1]
        gate_mlp_7 = chunk_9[2]
        chunk_9 = None
        layer_norm_34 = torch.nn.functional.layer_norm(
            x_18,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_80 = 1 + scale_mlp_7
        scale_mlp_7 = None
        mul_90 = layer_norm_34 * add_80
        layer_norm_34 = add_80 = None
        h_7 = mul_90 + shift_mlp_7
        mul_90 = shift_mlp_7 = None
        input_51 = torch._C._nn.linear(
            h_7,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_7 = None
        input_52 = torch.nn.functional.silu(input_51, inplace=False)
        input_51 = None
        input_53 = torch._C._nn.linear(
            input_52,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_52 = None
        mul_91 = gate_mlp_7 * input_53
        gate_mlp_7 = input_53 = None
        x_19 = x_18 + mul_91
        x_18 = mul_91 = None
        input_54 = torch.nn.functional.silu(y_2, inplace=False)
        input_55 = torch._C._nn.linear(
            input_54,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_54 = None
        chunk_10 = input_55.chunk(3, dim=-1)
        input_55 = None
        shift_mlp_8 = chunk_10[0]
        scale_mlp_8 = chunk_10[1]
        gate_mlp_8 = chunk_10[2]
        chunk_10 = None
        layer_norm_35 = torch.nn.functional.layer_norm(
            x_19,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_83 = 1 + scale_mlp_8
        scale_mlp_8 = None
        mul_92 = layer_norm_35 * add_83
        layer_norm_35 = add_83 = None
        h_8 = mul_92 + shift_mlp_8
        mul_92 = shift_mlp_8 = None
        input_56 = torch._C._nn.linear(
            h_8,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_8 = None
        input_57 = torch.nn.functional.silu(input_56, inplace=False)
        input_56 = None
        input_58 = torch._C._nn.linear(
            input_57,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_57 = None
        mul_93 = gate_mlp_8 * input_58
        gate_mlp_8 = input_58 = None
        x_20 = x_19 + mul_93
        x_19 = mul_93 = None
        input_59 = torch.nn.functional.silu(y_2, inplace=False)
        y_2 = None
        input_60 = torch._C._nn.linear(
            input_59,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_59 = None
        chunk_11 = input_60.chunk(2, dim=-1)
        input_60 = None
        shift_2 = chunk_11[0]
        scale_2 = chunk_11[1]
        chunk_11 = None
        layer_norm_36 = torch.nn.functional.layer_norm(x_20, (768,), None, None, 1e-06)
        x_20 = None
        add_86 = 1 + scale_2
        scale_2 = None
        mul_94 = layer_norm_36 * add_86
        layer_norm_36 = add_86 = None
        x_21 = mul_94 + shift_2
        mul_94 = shift_2 = None
        x_22 = torch._C._nn.linear(
            x_21,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_21 = None
        sub_2 = x_22 - noise
        x_22 = None
        mul_95 = sub_2 * 0.02
        sub_2 = None
        x_23 = x_16 + mul_95
        x_16 = mul_95 = None
        ones_3 = torch.ones(1)
        mul_96 = ones_3 * 3
        ones_3 = None
        truediv_6 = mul_96 / 50
        mul_96 = None
        t_3 = truediv_6.to(device(type="cuda", index=0))
        truediv_6 = None
        mul_97 = t_3 * 1000
        t_3 = None
        x_24 = torch._C._nn.linear(
            x_23,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_5 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_98 = -9.210340371976184 * arange_5
        arange_5 = None
        truediv_7 = mul_98 / 128
        mul_98 = None
        exp_3 = torch.exp(truediv_7)
        truediv_7 = None
        freqs_3 = exp_3.to(device=device(type="cuda", index=0))
        exp_3 = None
        getitem_137 = mul_97[(slice(None, None, None), None)]
        mul_97 = None
        float_4 = getitem_137.float()
        getitem_137 = None
        getitem_138 = freqs_3[None]
        freqs_3 = None
        args_3 = float_4 * getitem_138
        float_4 = getitem_138 = None
        cos_27 = torch.cos(args_3)
        sin_27 = torch.sin(args_3)
        args_3 = None
        embedding_3 = torch.cat([cos_27, sin_27], dim=-1)
        cos_27 = sin_27 = None
        input_61 = torch._C._nn.linear(
            embedding_3,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_3 = None
        input_62 = torch.nn.functional.silu(input_61, inplace=False)
        input_61 = None
        input_63 = torch._C._nn.linear(
            input_62,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_62 = None
        c_3 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y_3 = input_63 + c_3
        input_63 = c_3 = None
        input_64 = torch.nn.functional.silu(y_3, inplace=False)
        input_65 = torch._C._nn.linear(
            input_64,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_64 = None
        chunk_12 = input_65.chunk(3, dim=-1)
        input_65 = None
        shift_mlp_9 = chunk_12[0]
        scale_mlp_9 = chunk_12[1]
        gate_mlp_9 = chunk_12[2]
        chunk_12 = None
        layer_norm_37 = torch.nn.functional.layer_norm(
            x_24,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_90 = 1 + scale_mlp_9
        scale_mlp_9 = None
        mul_100 = layer_norm_37 * add_90
        layer_norm_37 = add_90 = None
        h_9 = mul_100 + shift_mlp_9
        mul_100 = shift_mlp_9 = None
        input_66 = torch._C._nn.linear(
            h_9,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_9 = None
        input_67 = torch.nn.functional.silu(input_66, inplace=False)
        input_66 = None
        input_68 = torch._C._nn.linear(
            input_67,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_67 = None
        mul_101 = gate_mlp_9 * input_68
        gate_mlp_9 = input_68 = None
        x_25 = x_24 + mul_101
        x_24 = mul_101 = None
        input_69 = torch.nn.functional.silu(y_3, inplace=False)
        input_70 = torch._C._nn.linear(
            input_69,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_69 = None
        chunk_13 = input_70.chunk(3, dim=-1)
        input_70 = None
        shift_mlp_10 = chunk_13[0]
        scale_mlp_10 = chunk_13[1]
        gate_mlp_10 = chunk_13[2]
        chunk_13 = None
        layer_norm_38 = torch.nn.functional.layer_norm(
            x_25,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_93 = 1 + scale_mlp_10
        scale_mlp_10 = None
        mul_102 = layer_norm_38 * add_93
        layer_norm_38 = add_93 = None
        h_10 = mul_102 + shift_mlp_10
        mul_102 = shift_mlp_10 = None
        input_71 = torch._C._nn.linear(
            h_10,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_10 = None
        input_72 = torch.nn.functional.silu(input_71, inplace=False)
        input_71 = None
        input_73 = torch._C._nn.linear(
            input_72,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_72 = None
        mul_103 = gate_mlp_10 * input_73
        gate_mlp_10 = input_73 = None
        x_26 = x_25 + mul_103
        x_25 = mul_103 = None
        input_74 = torch.nn.functional.silu(y_3, inplace=False)
        input_75 = torch._C._nn.linear(
            input_74,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_74 = None
        chunk_14 = input_75.chunk(3, dim=-1)
        input_75 = None
        shift_mlp_11 = chunk_14[0]
        scale_mlp_11 = chunk_14[1]
        gate_mlp_11 = chunk_14[2]
        chunk_14 = None
        layer_norm_39 = torch.nn.functional.layer_norm(
            x_26,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_96 = 1 + scale_mlp_11
        scale_mlp_11 = None
        mul_104 = layer_norm_39 * add_96
        layer_norm_39 = add_96 = None
        h_11 = mul_104 + shift_mlp_11
        mul_104 = shift_mlp_11 = None
        input_76 = torch._C._nn.linear(
            h_11,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_11 = None
        input_77 = torch.nn.functional.silu(input_76, inplace=False)
        input_76 = None
        input_78 = torch._C._nn.linear(
            input_77,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_77 = None
        mul_105 = gate_mlp_11 * input_78
        gate_mlp_11 = input_78 = None
        x_27 = x_26 + mul_105
        x_26 = mul_105 = None
        input_79 = torch.nn.functional.silu(y_3, inplace=False)
        y_3 = None
        input_80 = torch._C._nn.linear(
            input_79,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_79 = None
        chunk_15 = input_80.chunk(2, dim=-1)
        input_80 = None
        shift_3 = chunk_15[0]
        scale_3 = chunk_15[1]
        chunk_15 = None
        layer_norm_40 = torch.nn.functional.layer_norm(x_27, (768,), None, None, 1e-06)
        x_27 = None
        add_99 = 1 + scale_3
        scale_3 = None
        mul_106 = layer_norm_40 * add_99
        layer_norm_40 = add_99 = None
        x_28 = mul_106 + shift_3
        mul_106 = shift_3 = None
        x_29 = torch._C._nn.linear(
            x_28,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_28 = None
        sub_3 = x_29 - noise
        x_29 = None
        mul_107 = sub_3 * 0.02
        sub_3 = None
        x_30 = x_23 + mul_107
        x_23 = mul_107 = None
        ones_4 = torch.ones(1)
        mul_108 = ones_4 * 4
        ones_4 = None
        truediv_8 = mul_108 / 50
        mul_108 = None
        t_4 = truediv_8.to(device(type="cuda", index=0))
        truediv_8 = None
        mul_109 = t_4 * 1000
        t_4 = None
        x_31 = torch._C._nn.linear(
            x_30,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_6 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_110 = -9.210340371976184 * arange_6
        arange_6 = None
        truediv_9 = mul_110 / 128
        mul_110 = None
        exp_4 = torch.exp(truediv_9)
        truediv_9 = None
        freqs_4 = exp_4.to(device=device(type="cuda", index=0))
        exp_4 = None
        getitem_150 = mul_109[(slice(None, None, None), None)]
        mul_109 = None
        float_5 = getitem_150.float()
        getitem_150 = None
        getitem_151 = freqs_4[None]
        freqs_4 = None
        args_4 = float_5 * getitem_151
        float_5 = getitem_151 = None
        cos_28 = torch.cos(args_4)
        sin_28 = torch.sin(args_4)
        args_4 = None
        embedding_4 = torch.cat([cos_28, sin_28], dim=-1)
        cos_28 = sin_28 = None
        input_81 = torch._C._nn.linear(
            embedding_4,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_4 = None
        input_82 = torch.nn.functional.silu(input_81, inplace=False)
        input_81 = None
        input_83 = torch._C._nn.linear(
            input_82,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_82 = None
        c_4 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y_4 = input_83 + c_4
        input_83 = c_4 = None
        input_84 = torch.nn.functional.silu(y_4, inplace=False)
        input_85 = torch._C._nn.linear(
            input_84,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_84 = None
        chunk_16 = input_85.chunk(3, dim=-1)
        input_85 = None
        shift_mlp_12 = chunk_16[0]
        scale_mlp_12 = chunk_16[1]
        gate_mlp_12 = chunk_16[2]
        chunk_16 = None
        layer_norm_41 = torch.nn.functional.layer_norm(
            x_31,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_103 = 1 + scale_mlp_12
        scale_mlp_12 = None
        mul_112 = layer_norm_41 * add_103
        layer_norm_41 = add_103 = None
        h_12 = mul_112 + shift_mlp_12
        mul_112 = shift_mlp_12 = None
        input_86 = torch._C._nn.linear(
            h_12,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_12 = None
        input_87 = torch.nn.functional.silu(input_86, inplace=False)
        input_86 = None
        input_88 = torch._C._nn.linear(
            input_87,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_87 = None
        mul_113 = gate_mlp_12 * input_88
        gate_mlp_12 = input_88 = None
        x_32 = x_31 + mul_113
        x_31 = mul_113 = None
        input_89 = torch.nn.functional.silu(y_4, inplace=False)
        input_90 = torch._C._nn.linear(
            input_89,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_89 = None
        chunk_17 = input_90.chunk(3, dim=-1)
        input_90 = None
        shift_mlp_13 = chunk_17[0]
        scale_mlp_13 = chunk_17[1]
        gate_mlp_13 = chunk_17[2]
        chunk_17 = None
        layer_norm_42 = torch.nn.functional.layer_norm(
            x_32,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_106 = 1 + scale_mlp_13
        scale_mlp_13 = None
        mul_114 = layer_norm_42 * add_106
        layer_norm_42 = add_106 = None
        h_13 = mul_114 + shift_mlp_13
        mul_114 = shift_mlp_13 = None
        input_91 = torch._C._nn.linear(
            h_13,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_13 = None
        input_92 = torch.nn.functional.silu(input_91, inplace=False)
        input_91 = None
        input_93 = torch._C._nn.linear(
            input_92,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_92 = None
        mul_115 = gate_mlp_13 * input_93
        gate_mlp_13 = input_93 = None
        x_33 = x_32 + mul_115
        x_32 = mul_115 = None
        input_94 = torch.nn.functional.silu(y_4, inplace=False)
        input_95 = torch._C._nn.linear(
            input_94,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_94 = None
        chunk_18 = input_95.chunk(3, dim=-1)
        input_95 = None
        shift_mlp_14 = chunk_18[0]
        scale_mlp_14 = chunk_18[1]
        gate_mlp_14 = chunk_18[2]
        chunk_18 = None
        layer_norm_43 = torch.nn.functional.layer_norm(
            x_33,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_109 = 1 + scale_mlp_14
        scale_mlp_14 = None
        mul_116 = layer_norm_43 * add_109
        layer_norm_43 = add_109 = None
        h_14 = mul_116 + shift_mlp_14
        mul_116 = shift_mlp_14 = None
        input_96 = torch._C._nn.linear(
            h_14,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_14 = None
        input_97 = torch.nn.functional.silu(input_96, inplace=False)
        input_96 = None
        input_98 = torch._C._nn.linear(
            input_97,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_97 = None
        mul_117 = gate_mlp_14 * input_98
        gate_mlp_14 = input_98 = None
        x_34 = x_33 + mul_117
        x_33 = mul_117 = None
        input_99 = torch.nn.functional.silu(y_4, inplace=False)
        y_4 = None
        input_100 = torch._C._nn.linear(
            input_99,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_99 = None
        chunk_19 = input_100.chunk(2, dim=-1)
        input_100 = None
        shift_4 = chunk_19[0]
        scale_4 = chunk_19[1]
        chunk_19 = None
        layer_norm_44 = torch.nn.functional.layer_norm(x_34, (768,), None, None, 1e-06)
        x_34 = None
        add_112 = 1 + scale_4
        scale_4 = None
        mul_118 = layer_norm_44 * add_112
        layer_norm_44 = add_112 = None
        x_35 = mul_118 + shift_4
        mul_118 = shift_4 = None
        x_36 = torch._C._nn.linear(
            x_35,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_35 = None
        sub_4 = x_36 - noise
        x_36 = None
        mul_119 = sub_4 * 0.02
        sub_4 = None
        x_37 = x_30 + mul_119
        x_30 = mul_119 = None
        ones_5 = torch.ones(1)
        mul_120 = ones_5 * 5
        ones_5 = None
        truediv_10 = mul_120 / 50
        mul_120 = None
        t_5 = truediv_10.to(device(type="cuda", index=0))
        truediv_10 = None
        mul_121 = t_5 * 1000
        t_5 = None
        x_38 = torch._C._nn.linear(
            x_37,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_7 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_122 = -9.210340371976184 * arange_7
        arange_7 = None
        truediv_11 = mul_122 / 128
        mul_122 = None
        exp_5 = torch.exp(truediv_11)
        truediv_11 = None
        freqs_5 = exp_5.to(device=device(type="cuda", index=0))
        exp_5 = None
        getitem_163 = mul_121[(slice(None, None, None), None)]
        mul_121 = None
        float_6 = getitem_163.float()
        getitem_163 = None
        getitem_164 = freqs_5[None]
        freqs_5 = None
        args_5 = float_6 * getitem_164
        float_6 = getitem_164 = None
        cos_29 = torch.cos(args_5)
        sin_29 = torch.sin(args_5)
        args_5 = None
        embedding_5 = torch.cat([cos_29, sin_29], dim=-1)
        cos_29 = sin_29 = None
        input_101 = torch._C._nn.linear(
            embedding_5,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_5 = None
        input_102 = torch.nn.functional.silu(input_101, inplace=False)
        input_101 = None
        input_103 = torch._C._nn.linear(
            input_102,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_102 = None
        c_5 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y_5 = input_103 + c_5
        input_103 = c_5 = None
        input_104 = torch.nn.functional.silu(y_5, inplace=False)
        input_105 = torch._C._nn.linear(
            input_104,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_104 = None
        chunk_20 = input_105.chunk(3, dim=-1)
        input_105 = None
        shift_mlp_15 = chunk_20[0]
        scale_mlp_15 = chunk_20[1]
        gate_mlp_15 = chunk_20[2]
        chunk_20 = None
        layer_norm_45 = torch.nn.functional.layer_norm(
            x_38,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_116 = 1 + scale_mlp_15
        scale_mlp_15 = None
        mul_124 = layer_norm_45 * add_116
        layer_norm_45 = add_116 = None
        h_15 = mul_124 + shift_mlp_15
        mul_124 = shift_mlp_15 = None
        input_106 = torch._C._nn.linear(
            h_15,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_15 = None
        input_107 = torch.nn.functional.silu(input_106, inplace=False)
        input_106 = None
        input_108 = torch._C._nn.linear(
            input_107,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_107 = None
        mul_125 = gate_mlp_15 * input_108
        gate_mlp_15 = input_108 = None
        x_39 = x_38 + mul_125
        x_38 = mul_125 = None
        input_109 = torch.nn.functional.silu(y_5, inplace=False)
        input_110 = torch._C._nn.linear(
            input_109,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_109 = None
        chunk_21 = input_110.chunk(3, dim=-1)
        input_110 = None
        shift_mlp_16 = chunk_21[0]
        scale_mlp_16 = chunk_21[1]
        gate_mlp_16 = chunk_21[2]
        chunk_21 = None
        layer_norm_46 = torch.nn.functional.layer_norm(
            x_39,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_119 = 1 + scale_mlp_16
        scale_mlp_16 = None
        mul_126 = layer_norm_46 * add_119
        layer_norm_46 = add_119 = None
        h_16 = mul_126 + shift_mlp_16
        mul_126 = shift_mlp_16 = None
        input_111 = torch._C._nn.linear(
            h_16,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_16 = None
        input_112 = torch.nn.functional.silu(input_111, inplace=False)
        input_111 = None
        input_113 = torch._C._nn.linear(
            input_112,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_112 = None
        mul_127 = gate_mlp_16 * input_113
        gate_mlp_16 = input_113 = None
        x_40 = x_39 + mul_127
        x_39 = mul_127 = None
        input_114 = torch.nn.functional.silu(y_5, inplace=False)
        input_115 = torch._C._nn.linear(
            input_114,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_114 = None
        chunk_22 = input_115.chunk(3, dim=-1)
        input_115 = None
        shift_mlp_17 = chunk_22[0]
        scale_mlp_17 = chunk_22[1]
        gate_mlp_17 = chunk_22[2]
        chunk_22 = None
        layer_norm_47 = torch.nn.functional.layer_norm(
            x_40,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_122 = 1 + scale_mlp_17
        scale_mlp_17 = None
        mul_128 = layer_norm_47 * add_122
        layer_norm_47 = add_122 = None
        h_17 = mul_128 + shift_mlp_17
        mul_128 = shift_mlp_17 = None
        input_116 = torch._C._nn.linear(
            h_17,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_17 = None
        input_117 = torch.nn.functional.silu(input_116, inplace=False)
        input_116 = None
        input_118 = torch._C._nn.linear(
            input_117,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_117 = None
        mul_129 = gate_mlp_17 * input_118
        gate_mlp_17 = input_118 = None
        x_41 = x_40 + mul_129
        x_40 = mul_129 = None
        input_119 = torch.nn.functional.silu(y_5, inplace=False)
        y_5 = None
        input_120 = torch._C._nn.linear(
            input_119,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_119 = None
        chunk_23 = input_120.chunk(2, dim=-1)
        input_120 = None
        shift_5 = chunk_23[0]
        scale_5 = chunk_23[1]
        chunk_23 = None
        layer_norm_48 = torch.nn.functional.layer_norm(x_41, (768,), None, None, 1e-06)
        x_41 = None
        add_125 = 1 + scale_5
        scale_5 = None
        mul_130 = layer_norm_48 * add_125
        layer_norm_48 = add_125 = None
        x_42 = mul_130 + shift_5
        mul_130 = shift_5 = None
        x_43 = torch._C._nn.linear(
            x_42,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_42 = None
        sub_5 = x_43 - noise
        x_43 = None
        mul_131 = sub_5 * 0.02
        sub_5 = None
        x_44 = x_37 + mul_131
        x_37 = mul_131 = None
        ones_6 = torch.ones(1)
        mul_132 = ones_6 * 6
        ones_6 = None
        truediv_12 = mul_132 / 50
        mul_132 = None
        t_6 = truediv_12.to(device(type="cuda", index=0))
        truediv_12 = None
        mul_133 = t_6 * 1000
        t_6 = None
        x_45 = torch._C._nn.linear(
            x_44,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_8 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_134 = -9.210340371976184 * arange_8
        arange_8 = None
        truediv_13 = mul_134 / 128
        mul_134 = None
        exp_6 = torch.exp(truediv_13)
        truediv_13 = None
        freqs_6 = exp_6.to(device=device(type="cuda", index=0))
        exp_6 = None
        getitem_176 = mul_133[(slice(None, None, None), None)]
        mul_133 = None
        float_7 = getitem_176.float()
        getitem_176 = None
        getitem_177 = freqs_6[None]
        freqs_6 = None
        args_6 = float_7 * getitem_177
        float_7 = getitem_177 = None
        cos_30 = torch.cos(args_6)
        sin_30 = torch.sin(args_6)
        args_6 = None
        embedding_6 = torch.cat([cos_30, sin_30], dim=-1)
        cos_30 = sin_30 = None
        input_121 = torch._C._nn.linear(
            embedding_6,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_6 = None
        input_122 = torch.nn.functional.silu(input_121, inplace=False)
        input_121 = None
        input_123 = torch._C._nn.linear(
            input_122,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_122 = None
        c_6 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y_6 = input_123 + c_6
        input_123 = c_6 = None
        input_124 = torch.nn.functional.silu(y_6, inplace=False)
        input_125 = torch._C._nn.linear(
            input_124,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_124 = None
        chunk_24 = input_125.chunk(3, dim=-1)
        input_125 = None
        shift_mlp_18 = chunk_24[0]
        scale_mlp_18 = chunk_24[1]
        gate_mlp_18 = chunk_24[2]
        chunk_24 = None
        layer_norm_49 = torch.nn.functional.layer_norm(
            x_45,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_129 = 1 + scale_mlp_18
        scale_mlp_18 = None
        mul_136 = layer_norm_49 * add_129
        layer_norm_49 = add_129 = None
        h_18 = mul_136 + shift_mlp_18
        mul_136 = shift_mlp_18 = None
        input_126 = torch._C._nn.linear(
            h_18,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_18 = None
        input_127 = torch.nn.functional.silu(input_126, inplace=False)
        input_126 = None
        input_128 = torch._C._nn.linear(
            input_127,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_127 = None
        mul_137 = gate_mlp_18 * input_128
        gate_mlp_18 = input_128 = None
        x_46 = x_45 + mul_137
        x_45 = mul_137 = None
        input_129 = torch.nn.functional.silu(y_6, inplace=False)
        input_130 = torch._C._nn.linear(
            input_129,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_129 = None
        chunk_25 = input_130.chunk(3, dim=-1)
        input_130 = None
        shift_mlp_19 = chunk_25[0]
        scale_mlp_19 = chunk_25[1]
        gate_mlp_19 = chunk_25[2]
        chunk_25 = None
        layer_norm_50 = torch.nn.functional.layer_norm(
            x_46,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_132 = 1 + scale_mlp_19
        scale_mlp_19 = None
        mul_138 = layer_norm_50 * add_132
        layer_norm_50 = add_132 = None
        h_19 = mul_138 + shift_mlp_19
        mul_138 = shift_mlp_19 = None
        input_131 = torch._C._nn.linear(
            h_19,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_19 = None
        input_132 = torch.nn.functional.silu(input_131, inplace=False)
        input_131 = None
        input_133 = torch._C._nn.linear(
            input_132,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_132 = None
        mul_139 = gate_mlp_19 * input_133
        gate_mlp_19 = input_133 = None
        x_47 = x_46 + mul_139
        x_46 = mul_139 = None
        input_134 = torch.nn.functional.silu(y_6, inplace=False)
        input_135 = torch._C._nn.linear(
            input_134,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_134 = None
        chunk_26 = input_135.chunk(3, dim=-1)
        input_135 = None
        shift_mlp_20 = chunk_26[0]
        scale_mlp_20 = chunk_26[1]
        gate_mlp_20 = chunk_26[2]
        chunk_26 = None
        layer_norm_51 = torch.nn.functional.layer_norm(
            x_47,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_135 = 1 + scale_mlp_20
        scale_mlp_20 = None
        mul_140 = layer_norm_51 * add_135
        layer_norm_51 = add_135 = None
        h_20 = mul_140 + shift_mlp_20
        mul_140 = shift_mlp_20 = None
        input_136 = torch._C._nn.linear(
            h_20,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_20 = None
        input_137 = torch.nn.functional.silu(input_136, inplace=False)
        input_136 = None
        input_138 = torch._C._nn.linear(
            input_137,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_137 = None
        mul_141 = gate_mlp_20 * input_138
        gate_mlp_20 = input_138 = None
        x_48 = x_47 + mul_141
        x_47 = mul_141 = None
        input_139 = torch.nn.functional.silu(y_6, inplace=False)
        y_6 = None
        input_140 = torch._C._nn.linear(
            input_139,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_139 = None
        chunk_27 = input_140.chunk(2, dim=-1)
        input_140 = None
        shift_6 = chunk_27[0]
        scale_6 = chunk_27[1]
        chunk_27 = None
        layer_norm_52 = torch.nn.functional.layer_norm(x_48, (768,), None, None, 1e-06)
        x_48 = None
        add_138 = 1 + scale_6
        scale_6 = None
        mul_142 = layer_norm_52 * add_138
        layer_norm_52 = add_138 = None
        x_49 = mul_142 + shift_6
        mul_142 = shift_6 = None
        x_50 = torch._C._nn.linear(
            x_49,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_49 = None
        sub_6 = x_50 - noise
        x_50 = None
        mul_143 = sub_6 * 0.02
        sub_6 = None
        x_51 = x_44 + mul_143
        x_44 = mul_143 = None
        ones_7 = torch.ones(1)
        mul_144 = ones_7 * 7
        ones_7 = None
        truediv_14 = mul_144 / 50
        mul_144 = None
        t_7 = truediv_14.to(device(type="cuda", index=0))
        truediv_14 = None
        mul_145 = t_7 * 1000
        t_7 = None
        x_52 = torch._C._nn.linear(
            x_51,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_9 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_146 = -9.210340371976184 * arange_9
        arange_9 = None
        truediv_15 = mul_146 / 128
        mul_146 = None
        exp_7 = torch.exp(truediv_15)
        truediv_15 = None
        freqs_7 = exp_7.to(device=device(type="cuda", index=0))
        exp_7 = None
        getitem_189 = mul_145[(slice(None, None, None), None)]
        mul_145 = None
        float_8 = getitem_189.float()
        getitem_189 = None
        getitem_190 = freqs_7[None]
        freqs_7 = None
        args_7 = float_8 * getitem_190
        float_8 = getitem_190 = None
        cos_31 = torch.cos(args_7)
        sin_31 = torch.sin(args_7)
        args_7 = None
        embedding_7 = torch.cat([cos_31, sin_31], dim=-1)
        cos_31 = sin_31 = None
        input_141 = torch._C._nn.linear(
            embedding_7,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_7 = None
        input_142 = torch.nn.functional.silu(input_141, inplace=False)
        input_141 = None
        input_143 = torch._C._nn.linear(
            input_142,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_142 = None
        c_7 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y_7 = input_143 + c_7
        input_143 = c_7 = None
        input_144 = torch.nn.functional.silu(y_7, inplace=False)
        input_145 = torch._C._nn.linear(
            input_144,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_144 = None
        chunk_28 = input_145.chunk(3, dim=-1)
        input_145 = None
        shift_mlp_21 = chunk_28[0]
        scale_mlp_21 = chunk_28[1]
        gate_mlp_21 = chunk_28[2]
        chunk_28 = None
        layer_norm_53 = torch.nn.functional.layer_norm(
            x_52,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_142 = 1 + scale_mlp_21
        scale_mlp_21 = None
        mul_148 = layer_norm_53 * add_142
        layer_norm_53 = add_142 = None
        h_21 = mul_148 + shift_mlp_21
        mul_148 = shift_mlp_21 = None
        input_146 = torch._C._nn.linear(
            h_21,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_21 = None
        input_147 = torch.nn.functional.silu(input_146, inplace=False)
        input_146 = None
        input_148 = torch._C._nn.linear(
            input_147,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_147 = None
        mul_149 = gate_mlp_21 * input_148
        gate_mlp_21 = input_148 = None
        x_53 = x_52 + mul_149
        x_52 = mul_149 = None
        input_149 = torch.nn.functional.silu(y_7, inplace=False)
        input_150 = torch._C._nn.linear(
            input_149,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_149 = None
        chunk_29 = input_150.chunk(3, dim=-1)
        input_150 = None
        shift_mlp_22 = chunk_29[0]
        scale_mlp_22 = chunk_29[1]
        gate_mlp_22 = chunk_29[2]
        chunk_29 = None
        layer_norm_54 = torch.nn.functional.layer_norm(
            x_53,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_145 = 1 + scale_mlp_22
        scale_mlp_22 = None
        mul_150 = layer_norm_54 * add_145
        layer_norm_54 = add_145 = None
        h_22 = mul_150 + shift_mlp_22
        mul_150 = shift_mlp_22 = None
        input_151 = torch._C._nn.linear(
            h_22,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_22 = None
        input_152 = torch.nn.functional.silu(input_151, inplace=False)
        input_151 = None
        input_153 = torch._C._nn.linear(
            input_152,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_152 = None
        mul_151 = gate_mlp_22 * input_153
        gate_mlp_22 = input_153 = None
        x_54 = x_53 + mul_151
        x_53 = mul_151 = None
        input_154 = torch.nn.functional.silu(y_7, inplace=False)
        input_155 = torch._C._nn.linear(
            input_154,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_154 = None
        chunk_30 = input_155.chunk(3, dim=-1)
        input_155 = None
        shift_mlp_23 = chunk_30[0]
        scale_mlp_23 = chunk_30[1]
        gate_mlp_23 = chunk_30[2]
        chunk_30 = None
        layer_norm_55 = torch.nn.functional.layer_norm(
            x_54,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_148 = 1 + scale_mlp_23
        scale_mlp_23 = None
        mul_152 = layer_norm_55 * add_148
        layer_norm_55 = add_148 = None
        h_23 = mul_152 + shift_mlp_23
        mul_152 = shift_mlp_23 = None
        input_156 = torch._C._nn.linear(
            h_23,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_23 = None
        input_157 = torch.nn.functional.silu(input_156, inplace=False)
        input_156 = None
        input_158 = torch._C._nn.linear(
            input_157,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_157 = None
        mul_153 = gate_mlp_23 * input_158
        gate_mlp_23 = input_158 = None
        x_55 = x_54 + mul_153
        x_54 = mul_153 = None
        input_159 = torch.nn.functional.silu(y_7, inplace=False)
        y_7 = None
        input_160 = torch._C._nn.linear(
            input_159,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_159 = None
        chunk_31 = input_160.chunk(2, dim=-1)
        input_160 = None
        shift_7 = chunk_31[0]
        scale_7 = chunk_31[1]
        chunk_31 = None
        layer_norm_56 = torch.nn.functional.layer_norm(x_55, (768,), None, None, 1e-06)
        x_55 = None
        add_151 = 1 + scale_7
        scale_7 = None
        mul_154 = layer_norm_56 * add_151
        layer_norm_56 = add_151 = None
        x_56 = mul_154 + shift_7
        mul_154 = shift_7 = None
        x_57 = torch._C._nn.linear(
            x_56,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_56 = None
        sub_7 = x_57 - noise
        x_57 = None
        mul_155 = sub_7 * 0.02
        sub_7 = None
        x_58 = x_51 + mul_155
        x_51 = mul_155 = None
        ones_8 = torch.ones(1)
        mul_156 = ones_8 * 8
        ones_8 = None
        truediv_16 = mul_156 / 50
        mul_156 = None
        t_8 = truediv_16.to(device(type="cuda", index=0))
        truediv_16 = None
        mul_157 = t_8 * 1000
        t_8 = None
        x_59 = torch._C._nn.linear(
            x_58,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_10 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_158 = -9.210340371976184 * arange_10
        arange_10 = None
        truediv_17 = mul_158 / 128
        mul_158 = None
        exp_8 = torch.exp(truediv_17)
        truediv_17 = None
        freqs_8 = exp_8.to(device=device(type="cuda", index=0))
        exp_8 = None
        getitem_202 = mul_157[(slice(None, None, None), None)]
        mul_157 = None
        float_9 = getitem_202.float()
        getitem_202 = None
        getitem_203 = freqs_8[None]
        freqs_8 = None
        args_8 = float_9 * getitem_203
        float_9 = getitem_203 = None
        cos_32 = torch.cos(args_8)
        sin_32 = torch.sin(args_8)
        args_8 = None
        embedding_8 = torch.cat([cos_32, sin_32], dim=-1)
        cos_32 = sin_32 = None
        input_161 = torch._C._nn.linear(
            embedding_8,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_8 = None
        input_162 = torch.nn.functional.silu(input_161, inplace=False)
        input_161 = None
        input_163 = torch._C._nn.linear(
            input_162,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_162 = None
        c_8 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y_8 = input_163 + c_8
        input_163 = c_8 = None
        input_164 = torch.nn.functional.silu(y_8, inplace=False)
        input_165 = torch._C._nn.linear(
            input_164,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_164 = None
        chunk_32 = input_165.chunk(3, dim=-1)
        input_165 = None
        shift_mlp_24 = chunk_32[0]
        scale_mlp_24 = chunk_32[1]
        gate_mlp_24 = chunk_32[2]
        chunk_32 = None
        layer_norm_57 = torch.nn.functional.layer_norm(
            x_59,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_155 = 1 + scale_mlp_24
        scale_mlp_24 = None
        mul_160 = layer_norm_57 * add_155
        layer_norm_57 = add_155 = None
        h_24 = mul_160 + shift_mlp_24
        mul_160 = shift_mlp_24 = None
        input_166 = torch._C._nn.linear(
            h_24,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_24 = None
        input_167 = torch.nn.functional.silu(input_166, inplace=False)
        input_166 = None
        input_168 = torch._C._nn.linear(
            input_167,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_167 = None
        mul_161 = gate_mlp_24 * input_168
        gate_mlp_24 = input_168 = None
        x_60 = x_59 + mul_161
        x_59 = mul_161 = None
        input_169 = torch.nn.functional.silu(y_8, inplace=False)
        input_170 = torch._C._nn.linear(
            input_169,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_169 = None
        chunk_33 = input_170.chunk(3, dim=-1)
        input_170 = None
        shift_mlp_25 = chunk_33[0]
        scale_mlp_25 = chunk_33[1]
        gate_mlp_25 = chunk_33[2]
        chunk_33 = None
        layer_norm_58 = torch.nn.functional.layer_norm(
            x_60,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_158 = 1 + scale_mlp_25
        scale_mlp_25 = None
        mul_162 = layer_norm_58 * add_158
        layer_norm_58 = add_158 = None
        h_25 = mul_162 + shift_mlp_25
        mul_162 = shift_mlp_25 = None
        input_171 = torch._C._nn.linear(
            h_25,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_25 = None
        input_172 = torch.nn.functional.silu(input_171, inplace=False)
        input_171 = None
        input_173 = torch._C._nn.linear(
            input_172,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_172 = None
        mul_163 = gate_mlp_25 * input_173
        gate_mlp_25 = input_173 = None
        x_61 = x_60 + mul_163
        x_60 = mul_163 = None
        input_174 = torch.nn.functional.silu(y_8, inplace=False)
        input_175 = torch._C._nn.linear(
            input_174,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_174 = None
        chunk_34 = input_175.chunk(3, dim=-1)
        input_175 = None
        shift_mlp_26 = chunk_34[0]
        scale_mlp_26 = chunk_34[1]
        gate_mlp_26 = chunk_34[2]
        chunk_34 = None
        layer_norm_59 = torch.nn.functional.layer_norm(
            x_61,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_161 = 1 + scale_mlp_26
        scale_mlp_26 = None
        mul_164 = layer_norm_59 * add_161
        layer_norm_59 = add_161 = None
        h_26 = mul_164 + shift_mlp_26
        mul_164 = shift_mlp_26 = None
        input_176 = torch._C._nn.linear(
            h_26,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_26 = None
        input_177 = torch.nn.functional.silu(input_176, inplace=False)
        input_176 = None
        input_178 = torch._C._nn.linear(
            input_177,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_177 = None
        mul_165 = gate_mlp_26 * input_178
        gate_mlp_26 = input_178 = None
        x_62 = x_61 + mul_165
        x_61 = mul_165 = None
        input_179 = torch.nn.functional.silu(y_8, inplace=False)
        y_8 = None
        input_180 = torch._C._nn.linear(
            input_179,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_179 = None
        chunk_35 = input_180.chunk(2, dim=-1)
        input_180 = None
        shift_8 = chunk_35[0]
        scale_8 = chunk_35[1]
        chunk_35 = None
        layer_norm_60 = torch.nn.functional.layer_norm(x_62, (768,), None, None, 1e-06)
        x_62 = None
        add_164 = 1 + scale_8
        scale_8 = None
        mul_166 = layer_norm_60 * add_164
        layer_norm_60 = add_164 = None
        x_63 = mul_166 + shift_8
        mul_166 = shift_8 = None
        x_64 = torch._C._nn.linear(
            x_63,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_63 = None
        sub_8 = x_64 - noise
        x_64 = None
        mul_167 = sub_8 * 0.02
        sub_8 = None
        x_65 = x_58 + mul_167
        x_58 = mul_167 = None
        ones_9 = torch.ones(1)
        mul_168 = ones_9 * 9
        ones_9 = None
        truediv_18 = mul_168 / 50
        mul_168 = None
        t_9 = truediv_18.to(device(type="cuda", index=0))
        truediv_18 = None
        mul_169 = t_9 * 1000
        t_9 = None
        x_66 = torch._C._nn.linear(
            x_65,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_11 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_170 = -9.210340371976184 * arange_11
        arange_11 = None
        truediv_19 = mul_170 / 128
        mul_170 = None
        exp_9 = torch.exp(truediv_19)
        truediv_19 = None
        freqs_9 = exp_9.to(device=device(type="cuda", index=0))
        exp_9 = None
        getitem_215 = mul_169[(slice(None, None, None), None)]
        mul_169 = None
        float_10 = getitem_215.float()
        getitem_215 = None
        getitem_216 = freqs_9[None]
        freqs_9 = None
        args_9 = float_10 * getitem_216
        float_10 = getitem_216 = None
        cos_33 = torch.cos(args_9)
        sin_33 = torch.sin(args_9)
        args_9 = None
        embedding_9 = torch.cat([cos_33, sin_33], dim=-1)
        cos_33 = sin_33 = None
        input_181 = torch._C._nn.linear(
            embedding_9,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_9 = None
        input_182 = torch.nn.functional.silu(input_181, inplace=False)
        input_181 = None
        input_183 = torch._C._nn.linear(
            input_182,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_182 = None
        c_9 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y_9 = input_183 + c_9
        input_183 = c_9 = None
        input_184 = torch.nn.functional.silu(y_9, inplace=False)
        input_185 = torch._C._nn.linear(
            input_184,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_184 = None
        chunk_36 = input_185.chunk(3, dim=-1)
        input_185 = None
        shift_mlp_27 = chunk_36[0]
        scale_mlp_27 = chunk_36[1]
        gate_mlp_27 = chunk_36[2]
        chunk_36 = None
        layer_norm_61 = torch.nn.functional.layer_norm(
            x_66,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_168 = 1 + scale_mlp_27
        scale_mlp_27 = None
        mul_172 = layer_norm_61 * add_168
        layer_norm_61 = add_168 = None
        h_27 = mul_172 + shift_mlp_27
        mul_172 = shift_mlp_27 = None
        input_186 = torch._C._nn.linear(
            h_27,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_27 = None
        input_187 = torch.nn.functional.silu(input_186, inplace=False)
        input_186 = None
        input_188 = torch._C._nn.linear(
            input_187,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_187 = None
        mul_173 = gate_mlp_27 * input_188
        gate_mlp_27 = input_188 = None
        x_67 = x_66 + mul_173
        x_66 = mul_173 = None
        input_189 = torch.nn.functional.silu(y_9, inplace=False)
        input_190 = torch._C._nn.linear(
            input_189,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_189 = None
        chunk_37 = input_190.chunk(3, dim=-1)
        input_190 = None
        shift_mlp_28 = chunk_37[0]
        scale_mlp_28 = chunk_37[1]
        gate_mlp_28 = chunk_37[2]
        chunk_37 = None
        layer_norm_62 = torch.nn.functional.layer_norm(
            x_67,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_171 = 1 + scale_mlp_28
        scale_mlp_28 = None
        mul_174 = layer_norm_62 * add_171
        layer_norm_62 = add_171 = None
        h_28 = mul_174 + shift_mlp_28
        mul_174 = shift_mlp_28 = None
        input_191 = torch._C._nn.linear(
            h_28,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_28 = None
        input_192 = torch.nn.functional.silu(input_191, inplace=False)
        input_191 = None
        input_193 = torch._C._nn.linear(
            input_192,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_192 = None
        mul_175 = gate_mlp_28 * input_193
        gate_mlp_28 = input_193 = None
        x_68 = x_67 + mul_175
        x_67 = mul_175 = None
        input_194 = torch.nn.functional.silu(y_9, inplace=False)
        input_195 = torch._C._nn.linear(
            input_194,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_194 = None
        chunk_38 = input_195.chunk(3, dim=-1)
        input_195 = None
        shift_mlp_29 = chunk_38[0]
        scale_mlp_29 = chunk_38[1]
        gate_mlp_29 = chunk_38[2]
        chunk_38 = None
        layer_norm_63 = torch.nn.functional.layer_norm(
            x_68,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_174 = 1 + scale_mlp_29
        scale_mlp_29 = None
        mul_176 = layer_norm_63 * add_174
        layer_norm_63 = add_174 = None
        h_29 = mul_176 + shift_mlp_29
        mul_176 = shift_mlp_29 = None
        input_196 = torch._C._nn.linear(
            h_29,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_29 = None
        input_197 = torch.nn.functional.silu(input_196, inplace=False)
        input_196 = None
        input_198 = torch._C._nn.linear(
            input_197,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_197 = None
        mul_177 = gate_mlp_29 * input_198
        gate_mlp_29 = input_198 = None
        x_69 = x_68 + mul_177
        x_68 = mul_177 = None
        input_199 = torch.nn.functional.silu(y_9, inplace=False)
        y_9 = None
        input_200 = torch._C._nn.linear(
            input_199,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_199 = None
        chunk_39 = input_200.chunk(2, dim=-1)
        input_200 = None
        shift_9 = chunk_39[0]
        scale_9 = chunk_39[1]
        chunk_39 = None
        layer_norm_64 = torch.nn.functional.layer_norm(x_69, (768,), None, None, 1e-06)
        x_69 = None
        add_177 = 1 + scale_9
        scale_9 = None
        mul_178 = layer_norm_64 * add_177
        layer_norm_64 = add_177 = None
        x_70 = mul_178 + shift_9
        mul_178 = shift_9 = None
        x_71 = torch._C._nn.linear(
            x_70,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_70 = None
        sub_9 = x_71 - noise
        x_71 = None
        mul_179 = sub_9 * 0.02
        sub_9 = None
        x_72 = x_65 + mul_179
        x_65 = mul_179 = None
        ones_10 = torch.ones(1)
        mul_180 = ones_10 * 10
        ones_10 = None
        truediv_20 = mul_180 / 50
        mul_180 = None
        t_10 = truediv_20.to(device(type="cuda", index=0))
        truediv_20 = None
        mul_181 = t_10 * 1000
        t_10 = None
        x_73 = torch._C._nn.linear(
            x_72,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_12 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_182 = -9.210340371976184 * arange_12
        arange_12 = None
        truediv_21 = mul_182 / 128
        mul_182 = None
        exp_10 = torch.exp(truediv_21)
        truediv_21 = None
        freqs_10 = exp_10.to(device=device(type="cuda", index=0))
        exp_10 = None
        getitem_228 = mul_181[(slice(None, None, None), None)]
        mul_181 = None
        float_11 = getitem_228.float()
        getitem_228 = None
        getitem_229 = freqs_10[None]
        freqs_10 = None
        args_10 = float_11 * getitem_229
        float_11 = getitem_229 = None
        cos_34 = torch.cos(args_10)
        sin_34 = torch.sin(args_10)
        args_10 = None
        embedding_10 = torch.cat([cos_34, sin_34], dim=-1)
        cos_34 = sin_34 = None
        input_201 = torch._C._nn.linear(
            embedding_10,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_10 = None
        input_202 = torch.nn.functional.silu(input_201, inplace=False)
        input_201 = None
        input_203 = torch._C._nn.linear(
            input_202,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_202 = None
        c_10 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y_10 = input_203 + c_10
        input_203 = c_10 = None
        input_204 = torch.nn.functional.silu(y_10, inplace=False)
        input_205 = torch._C._nn.linear(
            input_204,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_204 = None
        chunk_40 = input_205.chunk(3, dim=-1)
        input_205 = None
        shift_mlp_30 = chunk_40[0]
        scale_mlp_30 = chunk_40[1]
        gate_mlp_30 = chunk_40[2]
        chunk_40 = None
        layer_norm_65 = torch.nn.functional.layer_norm(
            x_73,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_181 = 1 + scale_mlp_30
        scale_mlp_30 = None
        mul_184 = layer_norm_65 * add_181
        layer_norm_65 = add_181 = None
        h_30 = mul_184 + shift_mlp_30
        mul_184 = shift_mlp_30 = None
        input_206 = torch._C._nn.linear(
            h_30,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_30 = None
        input_207 = torch.nn.functional.silu(input_206, inplace=False)
        input_206 = None
        input_208 = torch._C._nn.linear(
            input_207,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_207 = None
        mul_185 = gate_mlp_30 * input_208
        gate_mlp_30 = input_208 = None
        x_74 = x_73 + mul_185
        x_73 = mul_185 = None
        input_209 = torch.nn.functional.silu(y_10, inplace=False)
        input_210 = torch._C._nn.linear(
            input_209,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_209 = None
        chunk_41 = input_210.chunk(3, dim=-1)
        input_210 = None
        shift_mlp_31 = chunk_41[0]
        scale_mlp_31 = chunk_41[1]
        gate_mlp_31 = chunk_41[2]
        chunk_41 = None
        layer_norm_66 = torch.nn.functional.layer_norm(
            x_74,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_184 = 1 + scale_mlp_31
        scale_mlp_31 = None
        mul_186 = layer_norm_66 * add_184
        layer_norm_66 = add_184 = None
        h_31 = mul_186 + shift_mlp_31
        mul_186 = shift_mlp_31 = None
        input_211 = torch._C._nn.linear(
            h_31,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_31 = None
        input_212 = torch.nn.functional.silu(input_211, inplace=False)
        input_211 = None
        input_213 = torch._C._nn.linear(
            input_212,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_212 = None
        mul_187 = gate_mlp_31 * input_213
        gate_mlp_31 = input_213 = None
        x_75 = x_74 + mul_187
        x_74 = mul_187 = None
        input_214 = torch.nn.functional.silu(y_10, inplace=False)
        input_215 = torch._C._nn.linear(
            input_214,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_214 = None
        chunk_42 = input_215.chunk(3, dim=-1)
        input_215 = None
        shift_mlp_32 = chunk_42[0]
        scale_mlp_32 = chunk_42[1]
        gate_mlp_32 = chunk_42[2]
        chunk_42 = None
        layer_norm_67 = torch.nn.functional.layer_norm(
            x_75,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_187 = 1 + scale_mlp_32
        scale_mlp_32 = None
        mul_188 = layer_norm_67 * add_187
        layer_norm_67 = add_187 = None
        h_32 = mul_188 + shift_mlp_32
        mul_188 = shift_mlp_32 = None
        input_216 = torch._C._nn.linear(
            h_32,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_32 = None
        input_217 = torch.nn.functional.silu(input_216, inplace=False)
        input_216 = None
        input_218 = torch._C._nn.linear(
            input_217,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_217 = None
        mul_189 = gate_mlp_32 * input_218
        gate_mlp_32 = input_218 = None
        x_76 = x_75 + mul_189
        x_75 = mul_189 = None
        input_219 = torch.nn.functional.silu(y_10, inplace=False)
        y_10 = None
        input_220 = torch._C._nn.linear(
            input_219,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_219 = None
        chunk_43 = input_220.chunk(2, dim=-1)
        input_220 = None
        shift_10 = chunk_43[0]
        scale_10 = chunk_43[1]
        chunk_43 = None
        layer_norm_68 = torch.nn.functional.layer_norm(x_76, (768,), None, None, 1e-06)
        x_76 = None
        add_190 = 1 + scale_10
        scale_10 = None
        mul_190 = layer_norm_68 * add_190
        layer_norm_68 = add_190 = None
        x_77 = mul_190 + shift_10
        mul_190 = shift_10 = None
        x_78 = torch._C._nn.linear(
            x_77,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_77 = None
        sub_10 = x_78 - noise
        x_78 = None
        mul_191 = sub_10 * 0.02
        sub_10 = None
        x_79 = x_72 + mul_191
        x_72 = mul_191 = None
        ones_11 = torch.ones(1)
        mul_192 = ones_11 * 11
        ones_11 = None
        truediv_22 = mul_192 / 50
        mul_192 = None
        t_11 = truediv_22.to(device(type="cuda", index=0))
        truediv_22 = None
        mul_193 = t_11 * 1000
        t_11 = None
        x_80 = torch._C._nn.linear(
            x_79,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_13 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_194 = -9.210340371976184 * arange_13
        arange_13 = None
        truediv_23 = mul_194 / 128
        mul_194 = None
        exp_11 = torch.exp(truediv_23)
        truediv_23 = None
        freqs_11 = exp_11.to(device=device(type="cuda", index=0))
        exp_11 = None
        getitem_241 = mul_193[(slice(None, None, None), None)]
        mul_193 = None
        float_12 = getitem_241.float()
        getitem_241 = None
        getitem_242 = freqs_11[None]
        freqs_11 = None
        args_11 = float_12 * getitem_242
        float_12 = getitem_242 = None
        cos_35 = torch.cos(args_11)
        sin_35 = torch.sin(args_11)
        args_11 = None
        embedding_11 = torch.cat([cos_35, sin_35], dim=-1)
        cos_35 = sin_35 = None
        input_221 = torch._C._nn.linear(
            embedding_11,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_11 = None
        input_222 = torch.nn.functional.silu(input_221, inplace=False)
        input_221 = None
        input_223 = torch._C._nn.linear(
            input_222,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_222 = None
        c_11 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y_11 = input_223 + c_11
        input_223 = c_11 = None
        input_224 = torch.nn.functional.silu(y_11, inplace=False)
        input_225 = torch._C._nn.linear(
            input_224,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_224 = None
        chunk_44 = input_225.chunk(3, dim=-1)
        input_225 = None
        shift_mlp_33 = chunk_44[0]
        scale_mlp_33 = chunk_44[1]
        gate_mlp_33 = chunk_44[2]
        chunk_44 = None
        layer_norm_69 = torch.nn.functional.layer_norm(
            x_80,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_194 = 1 + scale_mlp_33
        scale_mlp_33 = None
        mul_196 = layer_norm_69 * add_194
        layer_norm_69 = add_194 = None
        h_33 = mul_196 + shift_mlp_33
        mul_196 = shift_mlp_33 = None
        input_226 = torch._C._nn.linear(
            h_33,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_33 = None
        input_227 = torch.nn.functional.silu(input_226, inplace=False)
        input_226 = None
        input_228 = torch._C._nn.linear(
            input_227,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_227 = None
        mul_197 = gate_mlp_33 * input_228
        gate_mlp_33 = input_228 = None
        x_81 = x_80 + mul_197
        x_80 = mul_197 = None
        input_229 = torch.nn.functional.silu(y_11, inplace=False)
        input_230 = torch._C._nn.linear(
            input_229,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_229 = None
        chunk_45 = input_230.chunk(3, dim=-1)
        input_230 = None
        shift_mlp_34 = chunk_45[0]
        scale_mlp_34 = chunk_45[1]
        gate_mlp_34 = chunk_45[2]
        chunk_45 = None
        layer_norm_70 = torch.nn.functional.layer_norm(
            x_81,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_197 = 1 + scale_mlp_34
        scale_mlp_34 = None
        mul_198 = layer_norm_70 * add_197
        layer_norm_70 = add_197 = None
        h_34 = mul_198 + shift_mlp_34
        mul_198 = shift_mlp_34 = None
        input_231 = torch._C._nn.linear(
            h_34,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_34 = None
        input_232 = torch.nn.functional.silu(input_231, inplace=False)
        input_231 = None
        input_233 = torch._C._nn.linear(
            input_232,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_232 = None
        mul_199 = gate_mlp_34 * input_233
        gate_mlp_34 = input_233 = None
        x_82 = x_81 + mul_199
        x_81 = mul_199 = None
        input_234 = torch.nn.functional.silu(y_11, inplace=False)
        input_235 = torch._C._nn.linear(
            input_234,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_234 = None
        chunk_46 = input_235.chunk(3, dim=-1)
        input_235 = None
        shift_mlp_35 = chunk_46[0]
        scale_mlp_35 = chunk_46[1]
        gate_mlp_35 = chunk_46[2]
        chunk_46 = None
        layer_norm_71 = torch.nn.functional.layer_norm(
            x_82,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_200 = 1 + scale_mlp_35
        scale_mlp_35 = None
        mul_200 = layer_norm_71 * add_200
        layer_norm_71 = add_200 = None
        h_35 = mul_200 + shift_mlp_35
        mul_200 = shift_mlp_35 = None
        input_236 = torch._C._nn.linear(
            h_35,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_35 = None
        input_237 = torch.nn.functional.silu(input_236, inplace=False)
        input_236 = None
        input_238 = torch._C._nn.linear(
            input_237,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_237 = None
        mul_201 = gate_mlp_35 * input_238
        gate_mlp_35 = input_238 = None
        x_83 = x_82 + mul_201
        x_82 = mul_201 = None
        input_239 = torch.nn.functional.silu(y_11, inplace=False)
        y_11 = None
        input_240 = torch._C._nn.linear(
            input_239,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_239 = None
        chunk_47 = input_240.chunk(2, dim=-1)
        input_240 = None
        shift_11 = chunk_47[0]
        scale_11 = chunk_47[1]
        chunk_47 = None
        layer_norm_72 = torch.nn.functional.layer_norm(x_83, (768,), None, None, 1e-06)
        x_83 = None
        add_203 = 1 + scale_11
        scale_11 = None
        mul_202 = layer_norm_72 * add_203
        layer_norm_72 = add_203 = None
        x_84 = mul_202 + shift_11
        mul_202 = shift_11 = None
        x_85 = torch._C._nn.linear(
            x_84,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_84 = None
        sub_11 = x_85 - noise
        x_85 = None
        mul_203 = sub_11 * 0.02
        sub_11 = None
        x_86 = x_79 + mul_203
        x_79 = mul_203 = None
        ones_12 = torch.ones(1)
        mul_204 = ones_12 * 12
        ones_12 = None
        truediv_24 = mul_204 / 50
        mul_204 = None
        t_12 = truediv_24.to(device(type="cuda", index=0))
        truediv_24 = None
        mul_205 = t_12 * 1000
        t_12 = None
        x_87 = torch._C._nn.linear(
            x_86,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_14 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_206 = -9.210340371976184 * arange_14
        arange_14 = None
        truediv_25 = mul_206 / 128
        mul_206 = None
        exp_12 = torch.exp(truediv_25)
        truediv_25 = None
        freqs_12 = exp_12.to(device=device(type="cuda", index=0))
        exp_12 = None
        getitem_254 = mul_205[(slice(None, None, None), None)]
        mul_205 = None
        float_13 = getitem_254.float()
        getitem_254 = None
        getitem_255 = freqs_12[None]
        freqs_12 = None
        args_12 = float_13 * getitem_255
        float_13 = getitem_255 = None
        cos_36 = torch.cos(args_12)
        sin_36 = torch.sin(args_12)
        args_12 = None
        embedding_12 = torch.cat([cos_36, sin_36], dim=-1)
        cos_36 = sin_36 = None
        input_241 = torch._C._nn.linear(
            embedding_12,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_12 = None
        input_242 = torch.nn.functional.silu(input_241, inplace=False)
        input_241 = None
        input_243 = torch._C._nn.linear(
            input_242,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_242 = None
        c_12 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y_12 = input_243 + c_12
        input_243 = c_12 = None
        input_244 = torch.nn.functional.silu(y_12, inplace=False)
        input_245 = torch._C._nn.linear(
            input_244,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_244 = None
        chunk_48 = input_245.chunk(3, dim=-1)
        input_245 = None
        shift_mlp_36 = chunk_48[0]
        scale_mlp_36 = chunk_48[1]
        gate_mlp_36 = chunk_48[2]
        chunk_48 = None
        layer_norm_73 = torch.nn.functional.layer_norm(
            x_87,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_207 = 1 + scale_mlp_36
        scale_mlp_36 = None
        mul_208 = layer_norm_73 * add_207
        layer_norm_73 = add_207 = None
        h_36 = mul_208 + shift_mlp_36
        mul_208 = shift_mlp_36 = None
        input_246 = torch._C._nn.linear(
            h_36,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_36 = None
        input_247 = torch.nn.functional.silu(input_246, inplace=False)
        input_246 = None
        input_248 = torch._C._nn.linear(
            input_247,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_247 = None
        mul_209 = gate_mlp_36 * input_248
        gate_mlp_36 = input_248 = None
        x_88 = x_87 + mul_209
        x_87 = mul_209 = None
        input_249 = torch.nn.functional.silu(y_12, inplace=False)
        input_250 = torch._C._nn.linear(
            input_249,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_249 = None
        chunk_49 = input_250.chunk(3, dim=-1)
        input_250 = None
        shift_mlp_37 = chunk_49[0]
        scale_mlp_37 = chunk_49[1]
        gate_mlp_37 = chunk_49[2]
        chunk_49 = None
        layer_norm_74 = torch.nn.functional.layer_norm(
            x_88,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_210 = 1 + scale_mlp_37
        scale_mlp_37 = None
        mul_210 = layer_norm_74 * add_210
        layer_norm_74 = add_210 = None
        h_37 = mul_210 + shift_mlp_37
        mul_210 = shift_mlp_37 = None
        input_251 = torch._C._nn.linear(
            h_37,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_37 = None
        input_252 = torch.nn.functional.silu(input_251, inplace=False)
        input_251 = None
        input_253 = torch._C._nn.linear(
            input_252,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_252 = None
        mul_211 = gate_mlp_37 * input_253
        gate_mlp_37 = input_253 = None
        x_89 = x_88 + mul_211
        x_88 = mul_211 = None
        input_254 = torch.nn.functional.silu(y_12, inplace=False)
        input_255 = torch._C._nn.linear(
            input_254,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_254 = None
        chunk_50 = input_255.chunk(3, dim=-1)
        input_255 = None
        shift_mlp_38 = chunk_50[0]
        scale_mlp_38 = chunk_50[1]
        gate_mlp_38 = chunk_50[2]
        chunk_50 = None
        layer_norm_75 = torch.nn.functional.layer_norm(
            x_89,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_213 = 1 + scale_mlp_38
        scale_mlp_38 = None
        mul_212 = layer_norm_75 * add_213
        layer_norm_75 = add_213 = None
        h_38 = mul_212 + shift_mlp_38
        mul_212 = shift_mlp_38 = None
        input_256 = torch._C._nn.linear(
            h_38,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_38 = None
        input_257 = torch.nn.functional.silu(input_256, inplace=False)
        input_256 = None
        input_258 = torch._C._nn.linear(
            input_257,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_257 = None
        mul_213 = gate_mlp_38 * input_258
        gate_mlp_38 = input_258 = None
        x_90 = x_89 + mul_213
        x_89 = mul_213 = None
        input_259 = torch.nn.functional.silu(y_12, inplace=False)
        y_12 = None
        input_260 = torch._C._nn.linear(
            input_259,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_259 = None
        chunk_51 = input_260.chunk(2, dim=-1)
        input_260 = None
        shift_12 = chunk_51[0]
        scale_12 = chunk_51[1]
        chunk_51 = None
        layer_norm_76 = torch.nn.functional.layer_norm(x_90, (768,), None, None, 1e-06)
        x_90 = None
        add_216 = 1 + scale_12
        scale_12 = None
        mul_214 = layer_norm_76 * add_216
        layer_norm_76 = add_216 = None
        x_91 = mul_214 + shift_12
        mul_214 = shift_12 = None
        x_92 = torch._C._nn.linear(
            x_91,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_91 = None
        sub_12 = x_92 - noise
        x_92 = None
        mul_215 = sub_12 * 0.02
        sub_12 = None
        x_93 = x_86 + mul_215
        x_86 = mul_215 = None
        ones_13 = torch.ones(1)
        mul_216 = ones_13 * 13
        ones_13 = None
        truediv_26 = mul_216 / 50
        mul_216 = None
        t_13 = truediv_26.to(device(type="cuda", index=0))
        truediv_26 = None
        mul_217 = t_13 * 1000
        t_13 = None
        x_94 = torch._C._nn.linear(
            x_93,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_15 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_218 = -9.210340371976184 * arange_15
        arange_15 = None
        truediv_27 = mul_218 / 128
        mul_218 = None
        exp_13 = torch.exp(truediv_27)
        truediv_27 = None
        freqs_13 = exp_13.to(device=device(type="cuda", index=0))
        exp_13 = None
        getitem_267 = mul_217[(slice(None, None, None), None)]
        mul_217 = None
        float_14 = getitem_267.float()
        getitem_267 = None
        getitem_268 = freqs_13[None]
        freqs_13 = None
        args_13 = float_14 * getitem_268
        float_14 = getitem_268 = None
        cos_37 = torch.cos(args_13)
        sin_37 = torch.sin(args_13)
        args_13 = None
        embedding_13 = torch.cat([cos_37, sin_37], dim=-1)
        cos_37 = sin_37 = None
        input_261 = torch._C._nn.linear(
            embedding_13,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_13 = None
        input_262 = torch.nn.functional.silu(input_261, inplace=False)
        input_261 = None
        input_263 = torch._C._nn.linear(
            input_262,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_262 = None
        c_13 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y_13 = input_263 + c_13
        input_263 = c_13 = None
        input_264 = torch.nn.functional.silu(y_13, inplace=False)
        input_265 = torch._C._nn.linear(
            input_264,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_264 = None
        chunk_52 = input_265.chunk(3, dim=-1)
        input_265 = None
        shift_mlp_39 = chunk_52[0]
        scale_mlp_39 = chunk_52[1]
        gate_mlp_39 = chunk_52[2]
        chunk_52 = None
        layer_norm_77 = torch.nn.functional.layer_norm(
            x_94,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_220 = 1 + scale_mlp_39
        scale_mlp_39 = None
        mul_220 = layer_norm_77 * add_220
        layer_norm_77 = add_220 = None
        h_39 = mul_220 + shift_mlp_39
        mul_220 = shift_mlp_39 = None
        input_266 = torch._C._nn.linear(
            h_39,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_39 = None
        input_267 = torch.nn.functional.silu(input_266, inplace=False)
        input_266 = None
        input_268 = torch._C._nn.linear(
            input_267,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_267 = None
        mul_221 = gate_mlp_39 * input_268
        gate_mlp_39 = input_268 = None
        x_95 = x_94 + mul_221
        x_94 = mul_221 = None
        input_269 = torch.nn.functional.silu(y_13, inplace=False)
        input_270 = torch._C._nn.linear(
            input_269,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_269 = None
        chunk_53 = input_270.chunk(3, dim=-1)
        input_270 = None
        shift_mlp_40 = chunk_53[0]
        scale_mlp_40 = chunk_53[1]
        gate_mlp_40 = chunk_53[2]
        chunk_53 = None
        layer_norm_78 = torch.nn.functional.layer_norm(
            x_95,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_223 = 1 + scale_mlp_40
        scale_mlp_40 = None
        mul_222 = layer_norm_78 * add_223
        layer_norm_78 = add_223 = None
        h_40 = mul_222 + shift_mlp_40
        mul_222 = shift_mlp_40 = None
        input_271 = torch._C._nn.linear(
            h_40,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_40 = None
        input_272 = torch.nn.functional.silu(input_271, inplace=False)
        input_271 = None
        input_273 = torch._C._nn.linear(
            input_272,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_272 = None
        mul_223 = gate_mlp_40 * input_273
        gate_mlp_40 = input_273 = None
        x_96 = x_95 + mul_223
        x_95 = mul_223 = None
        input_274 = torch.nn.functional.silu(y_13, inplace=False)
        input_275 = torch._C._nn.linear(
            input_274,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_274 = None
        chunk_54 = input_275.chunk(3, dim=-1)
        input_275 = None
        shift_mlp_41 = chunk_54[0]
        scale_mlp_41 = chunk_54[1]
        gate_mlp_41 = chunk_54[2]
        chunk_54 = None
        layer_norm_79 = torch.nn.functional.layer_norm(
            x_96,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_226 = 1 + scale_mlp_41
        scale_mlp_41 = None
        mul_224 = layer_norm_79 * add_226
        layer_norm_79 = add_226 = None
        h_41 = mul_224 + shift_mlp_41
        mul_224 = shift_mlp_41 = None
        input_276 = torch._C._nn.linear(
            h_41,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_41 = None
        input_277 = torch.nn.functional.silu(input_276, inplace=False)
        input_276 = None
        input_278 = torch._C._nn.linear(
            input_277,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_277 = None
        mul_225 = gate_mlp_41 * input_278
        gate_mlp_41 = input_278 = None
        x_97 = x_96 + mul_225
        x_96 = mul_225 = None
        input_279 = torch.nn.functional.silu(y_13, inplace=False)
        y_13 = None
        input_280 = torch._C._nn.linear(
            input_279,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_279 = None
        chunk_55 = input_280.chunk(2, dim=-1)
        input_280 = None
        shift_13 = chunk_55[0]
        scale_13 = chunk_55[1]
        chunk_55 = None
        layer_norm_80 = torch.nn.functional.layer_norm(x_97, (768,), None, None, 1e-06)
        x_97 = None
        add_229 = 1 + scale_13
        scale_13 = None
        mul_226 = layer_norm_80 * add_229
        layer_norm_80 = add_229 = None
        x_98 = mul_226 + shift_13
        mul_226 = shift_13 = None
        x_99 = torch._C._nn.linear(
            x_98,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_98 = None
        sub_13 = x_99 - noise
        x_99 = None
        mul_227 = sub_13 * 0.02
        sub_13 = None
        x_100 = x_93 + mul_227
        x_93 = mul_227 = None
        ones_14 = torch.ones(1)
        mul_228 = ones_14 * 14
        ones_14 = None
        truediv_28 = mul_228 / 50
        mul_228 = None
        t_14 = truediv_28.to(device(type="cuda", index=0))
        truediv_28 = None
        mul_229 = t_14 * 1000
        t_14 = None
        x_101 = torch._C._nn.linear(
            x_100,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_16 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_230 = -9.210340371976184 * arange_16
        arange_16 = None
        truediv_29 = mul_230 / 128
        mul_230 = None
        exp_14 = torch.exp(truediv_29)
        truediv_29 = None
        freqs_14 = exp_14.to(device=device(type="cuda", index=0))
        exp_14 = None
        getitem_280 = mul_229[(slice(None, None, None), None)]
        mul_229 = None
        float_15 = getitem_280.float()
        getitem_280 = None
        getitem_281 = freqs_14[None]
        freqs_14 = None
        args_14 = float_15 * getitem_281
        float_15 = getitem_281 = None
        cos_38 = torch.cos(args_14)
        sin_38 = torch.sin(args_14)
        args_14 = None
        embedding_14 = torch.cat([cos_38, sin_38], dim=-1)
        cos_38 = sin_38 = None
        input_281 = torch._C._nn.linear(
            embedding_14,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_14 = None
        input_282 = torch.nn.functional.silu(input_281, inplace=False)
        input_281 = None
        input_283 = torch._C._nn.linear(
            input_282,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_282 = None
        c_14 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y_14 = input_283 + c_14
        input_283 = c_14 = None
        input_284 = torch.nn.functional.silu(y_14, inplace=False)
        input_285 = torch._C._nn.linear(
            input_284,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_284 = None
        chunk_56 = input_285.chunk(3, dim=-1)
        input_285 = None
        shift_mlp_42 = chunk_56[0]
        scale_mlp_42 = chunk_56[1]
        gate_mlp_42 = chunk_56[2]
        chunk_56 = None
        layer_norm_81 = torch.nn.functional.layer_norm(
            x_101,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_233 = 1 + scale_mlp_42
        scale_mlp_42 = None
        mul_232 = layer_norm_81 * add_233
        layer_norm_81 = add_233 = None
        h_42 = mul_232 + shift_mlp_42
        mul_232 = shift_mlp_42 = None
        input_286 = torch._C._nn.linear(
            h_42,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_42 = None
        input_287 = torch.nn.functional.silu(input_286, inplace=False)
        input_286 = None
        input_288 = torch._C._nn.linear(
            input_287,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_287 = None
        mul_233 = gate_mlp_42 * input_288
        gate_mlp_42 = input_288 = None
        x_102 = x_101 + mul_233
        x_101 = mul_233 = None
        input_289 = torch.nn.functional.silu(y_14, inplace=False)
        input_290 = torch._C._nn.linear(
            input_289,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_289 = None
        chunk_57 = input_290.chunk(3, dim=-1)
        input_290 = None
        shift_mlp_43 = chunk_57[0]
        scale_mlp_43 = chunk_57[1]
        gate_mlp_43 = chunk_57[2]
        chunk_57 = None
        layer_norm_82 = torch.nn.functional.layer_norm(
            x_102,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_236 = 1 + scale_mlp_43
        scale_mlp_43 = None
        mul_234 = layer_norm_82 * add_236
        layer_norm_82 = add_236 = None
        h_43 = mul_234 + shift_mlp_43
        mul_234 = shift_mlp_43 = None
        input_291 = torch._C._nn.linear(
            h_43,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_43 = None
        input_292 = torch.nn.functional.silu(input_291, inplace=False)
        input_291 = None
        input_293 = torch._C._nn.linear(
            input_292,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_292 = None
        mul_235 = gate_mlp_43 * input_293
        gate_mlp_43 = input_293 = None
        x_103 = x_102 + mul_235
        x_102 = mul_235 = None
        input_294 = torch.nn.functional.silu(y_14, inplace=False)
        input_295 = torch._C._nn.linear(
            input_294,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_294 = None
        chunk_58 = input_295.chunk(3, dim=-1)
        input_295 = None
        shift_mlp_44 = chunk_58[0]
        scale_mlp_44 = chunk_58[1]
        gate_mlp_44 = chunk_58[2]
        chunk_58 = None
        layer_norm_83 = torch.nn.functional.layer_norm(
            x_103,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_239 = 1 + scale_mlp_44
        scale_mlp_44 = None
        mul_236 = layer_norm_83 * add_239
        layer_norm_83 = add_239 = None
        h_44 = mul_236 + shift_mlp_44
        mul_236 = shift_mlp_44 = None
        input_296 = torch._C._nn.linear(
            h_44,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_44 = None
        input_297 = torch.nn.functional.silu(input_296, inplace=False)
        input_296 = None
        input_298 = torch._C._nn.linear(
            input_297,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_297 = None
        mul_237 = gate_mlp_44 * input_298
        gate_mlp_44 = input_298 = None
        x_104 = x_103 + mul_237
        x_103 = mul_237 = None
        input_299 = torch.nn.functional.silu(y_14, inplace=False)
        y_14 = None
        input_300 = torch._C._nn.linear(
            input_299,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_299 = None
        chunk_59 = input_300.chunk(2, dim=-1)
        input_300 = None
        shift_14 = chunk_59[0]
        scale_14 = chunk_59[1]
        chunk_59 = None
        layer_norm_84 = torch.nn.functional.layer_norm(x_104, (768,), None, None, 1e-06)
        x_104 = None
        add_242 = 1 + scale_14
        scale_14 = None
        mul_238 = layer_norm_84 * add_242
        layer_norm_84 = add_242 = None
        x_105 = mul_238 + shift_14
        mul_238 = shift_14 = None
        x_106 = torch._C._nn.linear(
            x_105,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_105 = None
        sub_14 = x_106 - noise
        x_106 = None
        mul_239 = sub_14 * 0.02
        sub_14 = None
        x_107 = x_100 + mul_239
        x_100 = mul_239 = None
        ones_15 = torch.ones(1)
        mul_240 = ones_15 * 15
        ones_15 = None
        truediv_30 = mul_240 / 50
        mul_240 = None
        t_15 = truediv_30.to(device(type="cuda", index=0))
        truediv_30 = None
        mul_241 = t_15 * 1000
        t_15 = None
        x_108 = torch._C._nn.linear(
            x_107,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_17 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_242 = -9.210340371976184 * arange_17
        arange_17 = None
        truediv_31 = mul_242 / 128
        mul_242 = None
        exp_15 = torch.exp(truediv_31)
        truediv_31 = None
        freqs_15 = exp_15.to(device=device(type="cuda", index=0))
        exp_15 = None
        getitem_293 = mul_241[(slice(None, None, None), None)]
        mul_241 = None
        float_16 = getitem_293.float()
        getitem_293 = None
        getitem_294 = freqs_15[None]
        freqs_15 = None
        args_15 = float_16 * getitem_294
        float_16 = getitem_294 = None
        cos_39 = torch.cos(args_15)
        sin_39 = torch.sin(args_15)
        args_15 = None
        embedding_15 = torch.cat([cos_39, sin_39], dim=-1)
        cos_39 = sin_39 = None
        input_301 = torch._C._nn.linear(
            embedding_15,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_15 = None
        input_302 = torch.nn.functional.silu(input_301, inplace=False)
        input_301 = None
        input_303 = torch._C._nn.linear(
            input_302,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_302 = None
        c_15 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y_15 = input_303 + c_15
        input_303 = c_15 = None
        input_304 = torch.nn.functional.silu(y_15, inplace=False)
        input_305 = torch._C._nn.linear(
            input_304,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_304 = None
        chunk_60 = input_305.chunk(3, dim=-1)
        input_305 = None
        shift_mlp_45 = chunk_60[0]
        scale_mlp_45 = chunk_60[1]
        gate_mlp_45 = chunk_60[2]
        chunk_60 = None
        layer_norm_85 = torch.nn.functional.layer_norm(
            x_108,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_246 = 1 + scale_mlp_45
        scale_mlp_45 = None
        mul_244 = layer_norm_85 * add_246
        layer_norm_85 = add_246 = None
        h_45 = mul_244 + shift_mlp_45
        mul_244 = shift_mlp_45 = None
        input_306 = torch._C._nn.linear(
            h_45,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_45 = None
        input_307 = torch.nn.functional.silu(input_306, inplace=False)
        input_306 = None
        input_308 = torch._C._nn.linear(
            input_307,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_307 = None
        mul_245 = gate_mlp_45 * input_308
        gate_mlp_45 = input_308 = None
        x_109 = x_108 + mul_245
        x_108 = mul_245 = None
        input_309 = torch.nn.functional.silu(y_15, inplace=False)
        input_310 = torch._C._nn.linear(
            input_309,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_309 = None
        chunk_61 = input_310.chunk(3, dim=-1)
        input_310 = None
        shift_mlp_46 = chunk_61[0]
        scale_mlp_46 = chunk_61[1]
        gate_mlp_46 = chunk_61[2]
        chunk_61 = None
        layer_norm_86 = torch.nn.functional.layer_norm(
            x_109,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_249 = 1 + scale_mlp_46
        scale_mlp_46 = None
        mul_246 = layer_norm_86 * add_249
        layer_norm_86 = add_249 = None
        h_46 = mul_246 + shift_mlp_46
        mul_246 = shift_mlp_46 = None
        input_311 = torch._C._nn.linear(
            h_46,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_46 = None
        input_312 = torch.nn.functional.silu(input_311, inplace=False)
        input_311 = None
        input_313 = torch._C._nn.linear(
            input_312,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_312 = None
        mul_247 = gate_mlp_46 * input_313
        gate_mlp_46 = input_313 = None
        x_110 = x_109 + mul_247
        x_109 = mul_247 = None
        input_314 = torch.nn.functional.silu(y_15, inplace=False)
        input_315 = torch._C._nn.linear(
            input_314,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_314 = None
        chunk_62 = input_315.chunk(3, dim=-1)
        input_315 = None
        shift_mlp_47 = chunk_62[0]
        scale_mlp_47 = chunk_62[1]
        gate_mlp_47 = chunk_62[2]
        chunk_62 = None
        layer_norm_87 = torch.nn.functional.layer_norm(
            x_110,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_252 = 1 + scale_mlp_47
        scale_mlp_47 = None
        mul_248 = layer_norm_87 * add_252
        layer_norm_87 = add_252 = None
        h_47 = mul_248 + shift_mlp_47
        mul_248 = shift_mlp_47 = None
        input_316 = torch._C._nn.linear(
            h_47,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_47 = None
        input_317 = torch.nn.functional.silu(input_316, inplace=False)
        input_316 = None
        input_318 = torch._C._nn.linear(
            input_317,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_317 = None
        mul_249 = gate_mlp_47 * input_318
        gate_mlp_47 = input_318 = None
        x_111 = x_110 + mul_249
        x_110 = mul_249 = None
        input_319 = torch.nn.functional.silu(y_15, inplace=False)
        y_15 = None
        input_320 = torch._C._nn.linear(
            input_319,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_319 = None
        chunk_63 = input_320.chunk(2, dim=-1)
        input_320 = None
        shift_15 = chunk_63[0]
        scale_15 = chunk_63[1]
        chunk_63 = None
        layer_norm_88 = torch.nn.functional.layer_norm(x_111, (768,), None, None, 1e-06)
        x_111 = None
        add_255 = 1 + scale_15
        scale_15 = None
        mul_250 = layer_norm_88 * add_255
        layer_norm_88 = add_255 = None
        x_112 = mul_250 + shift_15
        mul_250 = shift_15 = None
        x_113 = torch._C._nn.linear(
            x_112,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_112 = None
        sub_15 = x_113 - noise
        x_113 = None
        mul_251 = sub_15 * 0.02
        sub_15 = None
        x_114 = x_107 + mul_251
        x_107 = mul_251 = None
        ones_16 = torch.ones(1)
        mul_252 = ones_16 * 16
        ones_16 = None
        truediv_32 = mul_252 / 50
        mul_252 = None
        t_16 = truediv_32.to(device(type="cuda", index=0))
        truediv_32 = None
        mul_253 = t_16 * 1000
        t_16 = None
        x_115 = torch._C._nn.linear(
            x_114,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_18 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_254 = -9.210340371976184 * arange_18
        arange_18 = None
        truediv_33 = mul_254 / 128
        mul_254 = None
        exp_16 = torch.exp(truediv_33)
        truediv_33 = None
        freqs_16 = exp_16.to(device=device(type="cuda", index=0))
        exp_16 = None
        getitem_306 = mul_253[(slice(None, None, None), None)]
        mul_253 = None
        float_17 = getitem_306.float()
        getitem_306 = None
        getitem_307 = freqs_16[None]
        freqs_16 = None
        args_16 = float_17 * getitem_307
        float_17 = getitem_307 = None
        cos_40 = torch.cos(args_16)
        sin_40 = torch.sin(args_16)
        args_16 = None
        embedding_16 = torch.cat([cos_40, sin_40], dim=-1)
        cos_40 = sin_40 = None
        input_321 = torch._C._nn.linear(
            embedding_16,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_16 = None
        input_322 = torch.nn.functional.silu(input_321, inplace=False)
        input_321 = None
        input_323 = torch._C._nn.linear(
            input_322,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_322 = None
        c_16 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y_16 = input_323 + c_16
        input_323 = c_16 = None
        input_324 = torch.nn.functional.silu(y_16, inplace=False)
        input_325 = torch._C._nn.linear(
            input_324,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_324 = None
        chunk_64 = input_325.chunk(3, dim=-1)
        input_325 = None
        shift_mlp_48 = chunk_64[0]
        scale_mlp_48 = chunk_64[1]
        gate_mlp_48 = chunk_64[2]
        chunk_64 = None
        layer_norm_89 = torch.nn.functional.layer_norm(
            x_115,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_259 = 1 + scale_mlp_48
        scale_mlp_48 = None
        mul_256 = layer_norm_89 * add_259
        layer_norm_89 = add_259 = None
        h_48 = mul_256 + shift_mlp_48
        mul_256 = shift_mlp_48 = None
        input_326 = torch._C._nn.linear(
            h_48,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_48 = None
        input_327 = torch.nn.functional.silu(input_326, inplace=False)
        input_326 = None
        input_328 = torch._C._nn.linear(
            input_327,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_327 = None
        mul_257 = gate_mlp_48 * input_328
        gate_mlp_48 = input_328 = None
        x_116 = x_115 + mul_257
        x_115 = mul_257 = None
        input_329 = torch.nn.functional.silu(y_16, inplace=False)
        input_330 = torch._C._nn.linear(
            input_329,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_329 = None
        chunk_65 = input_330.chunk(3, dim=-1)
        input_330 = None
        shift_mlp_49 = chunk_65[0]
        scale_mlp_49 = chunk_65[1]
        gate_mlp_49 = chunk_65[2]
        chunk_65 = None
        layer_norm_90 = torch.nn.functional.layer_norm(
            x_116,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_262 = 1 + scale_mlp_49
        scale_mlp_49 = None
        mul_258 = layer_norm_90 * add_262
        layer_norm_90 = add_262 = None
        h_49 = mul_258 + shift_mlp_49
        mul_258 = shift_mlp_49 = None
        input_331 = torch._C._nn.linear(
            h_49,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_49 = None
        input_332 = torch.nn.functional.silu(input_331, inplace=False)
        input_331 = None
        input_333 = torch._C._nn.linear(
            input_332,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_332 = None
        mul_259 = gate_mlp_49 * input_333
        gate_mlp_49 = input_333 = None
        x_117 = x_116 + mul_259
        x_116 = mul_259 = None
        input_334 = torch.nn.functional.silu(y_16, inplace=False)
        input_335 = torch._C._nn.linear(
            input_334,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_334 = None
        chunk_66 = input_335.chunk(3, dim=-1)
        input_335 = None
        shift_mlp_50 = chunk_66[0]
        scale_mlp_50 = chunk_66[1]
        gate_mlp_50 = chunk_66[2]
        chunk_66 = None
        layer_norm_91 = torch.nn.functional.layer_norm(
            x_117,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_265 = 1 + scale_mlp_50
        scale_mlp_50 = None
        mul_260 = layer_norm_91 * add_265
        layer_norm_91 = add_265 = None
        h_50 = mul_260 + shift_mlp_50
        mul_260 = shift_mlp_50 = None
        input_336 = torch._C._nn.linear(
            h_50,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_50 = None
        input_337 = torch.nn.functional.silu(input_336, inplace=False)
        input_336 = None
        input_338 = torch._C._nn.linear(
            input_337,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_337 = None
        mul_261 = gate_mlp_50 * input_338
        gate_mlp_50 = input_338 = None
        x_118 = x_117 + mul_261
        x_117 = mul_261 = None
        input_339 = torch.nn.functional.silu(y_16, inplace=False)
        y_16 = None
        input_340 = torch._C._nn.linear(
            input_339,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_339 = None
        chunk_67 = input_340.chunk(2, dim=-1)
        input_340 = None
        shift_16 = chunk_67[0]
        scale_16 = chunk_67[1]
        chunk_67 = None
        layer_norm_92 = torch.nn.functional.layer_norm(x_118, (768,), None, None, 1e-06)
        x_118 = None
        add_268 = 1 + scale_16
        scale_16 = None
        mul_262 = layer_norm_92 * add_268
        layer_norm_92 = add_268 = None
        x_119 = mul_262 + shift_16
        mul_262 = shift_16 = None
        x_120 = torch._C._nn.linear(
            x_119,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_119 = None
        sub_16 = x_120 - noise
        x_120 = None
        mul_263 = sub_16 * 0.02
        sub_16 = None
        x_121 = x_114 + mul_263
        x_114 = mul_263 = None
        ones_17 = torch.ones(1)
        mul_264 = ones_17 * 17
        ones_17 = None
        truediv_34 = mul_264 / 50
        mul_264 = None
        t_17 = truediv_34.to(device(type="cuda", index=0))
        truediv_34 = None
        mul_265 = t_17 * 1000
        t_17 = None
        x_122 = torch._C._nn.linear(
            x_121,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_19 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_266 = -9.210340371976184 * arange_19
        arange_19 = None
        truediv_35 = mul_266 / 128
        mul_266 = None
        exp_17 = torch.exp(truediv_35)
        truediv_35 = None
        freqs_17 = exp_17.to(device=device(type="cuda", index=0))
        exp_17 = None
        getitem_319 = mul_265[(slice(None, None, None), None)]
        mul_265 = None
        float_18 = getitem_319.float()
        getitem_319 = None
        getitem_320 = freqs_17[None]
        freqs_17 = None
        args_17 = float_18 * getitem_320
        float_18 = getitem_320 = None
        cos_41 = torch.cos(args_17)
        sin_41 = torch.sin(args_17)
        args_17 = None
        embedding_17 = torch.cat([cos_41, sin_41], dim=-1)
        cos_41 = sin_41 = None
        input_341 = torch._C._nn.linear(
            embedding_17,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_17 = None
        input_342 = torch.nn.functional.silu(input_341, inplace=False)
        input_341 = None
        input_343 = torch._C._nn.linear(
            input_342,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_342 = None
        c_17 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y_17 = input_343 + c_17
        input_343 = c_17 = None
        input_344 = torch.nn.functional.silu(y_17, inplace=False)
        input_345 = torch._C._nn.linear(
            input_344,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_344 = None
        chunk_68 = input_345.chunk(3, dim=-1)
        input_345 = None
        shift_mlp_51 = chunk_68[0]
        scale_mlp_51 = chunk_68[1]
        gate_mlp_51 = chunk_68[2]
        chunk_68 = None
        layer_norm_93 = torch.nn.functional.layer_norm(
            x_122,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_272 = 1 + scale_mlp_51
        scale_mlp_51 = None
        mul_268 = layer_norm_93 * add_272
        layer_norm_93 = add_272 = None
        h_51 = mul_268 + shift_mlp_51
        mul_268 = shift_mlp_51 = None
        input_346 = torch._C._nn.linear(
            h_51,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_51 = None
        input_347 = torch.nn.functional.silu(input_346, inplace=False)
        input_346 = None
        input_348 = torch._C._nn.linear(
            input_347,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_347 = None
        mul_269 = gate_mlp_51 * input_348
        gate_mlp_51 = input_348 = None
        x_123 = x_122 + mul_269
        x_122 = mul_269 = None
        input_349 = torch.nn.functional.silu(y_17, inplace=False)
        input_350 = torch._C._nn.linear(
            input_349,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_349 = None
        chunk_69 = input_350.chunk(3, dim=-1)
        input_350 = None
        shift_mlp_52 = chunk_69[0]
        scale_mlp_52 = chunk_69[1]
        gate_mlp_52 = chunk_69[2]
        chunk_69 = None
        layer_norm_94 = torch.nn.functional.layer_norm(
            x_123,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_275 = 1 + scale_mlp_52
        scale_mlp_52 = None
        mul_270 = layer_norm_94 * add_275
        layer_norm_94 = add_275 = None
        h_52 = mul_270 + shift_mlp_52
        mul_270 = shift_mlp_52 = None
        input_351 = torch._C._nn.linear(
            h_52,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_52 = None
        input_352 = torch.nn.functional.silu(input_351, inplace=False)
        input_351 = None
        input_353 = torch._C._nn.linear(
            input_352,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_352 = None
        mul_271 = gate_mlp_52 * input_353
        gate_mlp_52 = input_353 = None
        x_124 = x_123 + mul_271
        x_123 = mul_271 = None
        input_354 = torch.nn.functional.silu(y_17, inplace=False)
        input_355 = torch._C._nn.linear(
            input_354,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_354 = None
        chunk_70 = input_355.chunk(3, dim=-1)
        input_355 = None
        shift_mlp_53 = chunk_70[0]
        scale_mlp_53 = chunk_70[1]
        gate_mlp_53 = chunk_70[2]
        chunk_70 = None
        layer_norm_95 = torch.nn.functional.layer_norm(
            x_124,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_278 = 1 + scale_mlp_53
        scale_mlp_53 = None
        mul_272 = layer_norm_95 * add_278
        layer_norm_95 = add_278 = None
        h_53 = mul_272 + shift_mlp_53
        mul_272 = shift_mlp_53 = None
        input_356 = torch._C._nn.linear(
            h_53,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_53 = None
        input_357 = torch.nn.functional.silu(input_356, inplace=False)
        input_356 = None
        input_358 = torch._C._nn.linear(
            input_357,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_357 = None
        mul_273 = gate_mlp_53 * input_358
        gate_mlp_53 = input_358 = None
        x_125 = x_124 + mul_273
        x_124 = mul_273 = None
        input_359 = torch.nn.functional.silu(y_17, inplace=False)
        y_17 = None
        input_360 = torch._C._nn.linear(
            input_359,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_359 = None
        chunk_71 = input_360.chunk(2, dim=-1)
        input_360 = None
        shift_17 = chunk_71[0]
        scale_17 = chunk_71[1]
        chunk_71 = None
        layer_norm_96 = torch.nn.functional.layer_norm(x_125, (768,), None, None, 1e-06)
        x_125 = None
        add_281 = 1 + scale_17
        scale_17 = None
        mul_274 = layer_norm_96 * add_281
        layer_norm_96 = add_281 = None
        x_126 = mul_274 + shift_17
        mul_274 = shift_17 = None
        x_127 = torch._C._nn.linear(
            x_126,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_126 = None
        sub_17 = x_127 - noise
        x_127 = None
        mul_275 = sub_17 * 0.02
        sub_17 = None
        x_128 = x_121 + mul_275
        x_121 = mul_275 = None
        ones_18 = torch.ones(1)
        mul_276 = ones_18 * 18
        ones_18 = None
        truediv_36 = mul_276 / 50
        mul_276 = None
        t_18 = truediv_36.to(device(type="cuda", index=0))
        truediv_36 = None
        mul_277 = t_18 * 1000
        t_18 = None
        x_129 = torch._C._nn.linear(
            x_128,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_20 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_278 = -9.210340371976184 * arange_20
        arange_20 = None
        truediv_37 = mul_278 / 128
        mul_278 = None
        exp_18 = torch.exp(truediv_37)
        truediv_37 = None
        freqs_18 = exp_18.to(device=device(type="cuda", index=0))
        exp_18 = None
        getitem_332 = mul_277[(slice(None, None, None), None)]
        mul_277 = None
        float_19 = getitem_332.float()
        getitem_332 = None
        getitem_333 = freqs_18[None]
        freqs_18 = None
        args_18 = float_19 * getitem_333
        float_19 = getitem_333 = None
        cos_42 = torch.cos(args_18)
        sin_42 = torch.sin(args_18)
        args_18 = None
        embedding_18 = torch.cat([cos_42, sin_42], dim=-1)
        cos_42 = sin_42 = None
        input_361 = torch._C._nn.linear(
            embedding_18,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_18 = None
        input_362 = torch.nn.functional.silu(input_361, inplace=False)
        input_361 = None
        input_363 = torch._C._nn.linear(
            input_362,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_362 = None
        c_18 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y_18 = input_363 + c_18
        input_363 = c_18 = None
        input_364 = torch.nn.functional.silu(y_18, inplace=False)
        input_365 = torch._C._nn.linear(
            input_364,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_364 = None
        chunk_72 = input_365.chunk(3, dim=-1)
        input_365 = None
        shift_mlp_54 = chunk_72[0]
        scale_mlp_54 = chunk_72[1]
        gate_mlp_54 = chunk_72[2]
        chunk_72 = None
        layer_norm_97 = torch.nn.functional.layer_norm(
            x_129,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_285 = 1 + scale_mlp_54
        scale_mlp_54 = None
        mul_280 = layer_norm_97 * add_285
        layer_norm_97 = add_285 = None
        h_54 = mul_280 + shift_mlp_54
        mul_280 = shift_mlp_54 = None
        input_366 = torch._C._nn.linear(
            h_54,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_54 = None
        input_367 = torch.nn.functional.silu(input_366, inplace=False)
        input_366 = None
        input_368 = torch._C._nn.linear(
            input_367,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_367 = None
        mul_281 = gate_mlp_54 * input_368
        gate_mlp_54 = input_368 = None
        x_130 = x_129 + mul_281
        x_129 = mul_281 = None
        input_369 = torch.nn.functional.silu(y_18, inplace=False)
        input_370 = torch._C._nn.linear(
            input_369,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_369 = None
        chunk_73 = input_370.chunk(3, dim=-1)
        input_370 = None
        shift_mlp_55 = chunk_73[0]
        scale_mlp_55 = chunk_73[1]
        gate_mlp_55 = chunk_73[2]
        chunk_73 = None
        layer_norm_98 = torch.nn.functional.layer_norm(
            x_130,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_288 = 1 + scale_mlp_55
        scale_mlp_55 = None
        mul_282 = layer_norm_98 * add_288
        layer_norm_98 = add_288 = None
        h_55 = mul_282 + shift_mlp_55
        mul_282 = shift_mlp_55 = None
        input_371 = torch._C._nn.linear(
            h_55,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_55 = None
        input_372 = torch.nn.functional.silu(input_371, inplace=False)
        input_371 = None
        input_373 = torch._C._nn.linear(
            input_372,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_372 = None
        mul_283 = gate_mlp_55 * input_373
        gate_mlp_55 = input_373 = None
        x_131 = x_130 + mul_283
        x_130 = mul_283 = None
        input_374 = torch.nn.functional.silu(y_18, inplace=False)
        input_375 = torch._C._nn.linear(
            input_374,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_374 = None
        chunk_74 = input_375.chunk(3, dim=-1)
        input_375 = None
        shift_mlp_56 = chunk_74[0]
        scale_mlp_56 = chunk_74[1]
        gate_mlp_56 = chunk_74[2]
        chunk_74 = None
        layer_norm_99 = torch.nn.functional.layer_norm(
            x_131,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_291 = 1 + scale_mlp_56
        scale_mlp_56 = None
        mul_284 = layer_norm_99 * add_291
        layer_norm_99 = add_291 = None
        h_56 = mul_284 + shift_mlp_56
        mul_284 = shift_mlp_56 = None
        input_376 = torch._C._nn.linear(
            h_56,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_56 = None
        input_377 = torch.nn.functional.silu(input_376, inplace=False)
        input_376 = None
        input_378 = torch._C._nn.linear(
            input_377,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_377 = None
        mul_285 = gate_mlp_56 * input_378
        gate_mlp_56 = input_378 = None
        x_132 = x_131 + mul_285
        x_131 = mul_285 = None
        input_379 = torch.nn.functional.silu(y_18, inplace=False)
        y_18 = None
        input_380 = torch._C._nn.linear(
            input_379,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_379 = None
        chunk_75 = input_380.chunk(2, dim=-1)
        input_380 = None
        shift_18 = chunk_75[0]
        scale_18 = chunk_75[1]
        chunk_75 = None
        layer_norm_100 = torch.nn.functional.layer_norm(
            x_132, (768,), None, None, 1e-06
        )
        x_132 = None
        add_294 = 1 + scale_18
        scale_18 = None
        mul_286 = layer_norm_100 * add_294
        layer_norm_100 = add_294 = None
        x_133 = mul_286 + shift_18
        mul_286 = shift_18 = None
        x_134 = torch._C._nn.linear(
            x_133,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_133 = None
        sub_18 = x_134 - noise
        x_134 = None
        mul_287 = sub_18 * 0.02
        sub_18 = None
        x_135 = x_128 + mul_287
        x_128 = mul_287 = None
        ones_19 = torch.ones(1)
        mul_288 = ones_19 * 19
        ones_19 = None
        truediv_38 = mul_288 / 50
        mul_288 = None
        t_19 = truediv_38.to(device(type="cuda", index=0))
        truediv_38 = None
        mul_289 = t_19 * 1000
        t_19 = None
        x_136 = torch._C._nn.linear(
            x_135,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_21 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_290 = -9.210340371976184 * arange_21
        arange_21 = None
        truediv_39 = mul_290 / 128
        mul_290 = None
        exp_19 = torch.exp(truediv_39)
        truediv_39 = None
        freqs_19 = exp_19.to(device=device(type="cuda", index=0))
        exp_19 = None
        getitem_345 = mul_289[(slice(None, None, None), None)]
        mul_289 = None
        float_20 = getitem_345.float()
        getitem_345 = None
        getitem_346 = freqs_19[None]
        freqs_19 = None
        args_19 = float_20 * getitem_346
        float_20 = getitem_346 = None
        cos_43 = torch.cos(args_19)
        sin_43 = torch.sin(args_19)
        args_19 = None
        embedding_19 = torch.cat([cos_43, sin_43], dim=-1)
        cos_43 = sin_43 = None
        input_381 = torch._C._nn.linear(
            embedding_19,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_19 = None
        input_382 = torch.nn.functional.silu(input_381, inplace=False)
        input_381 = None
        input_383 = torch._C._nn.linear(
            input_382,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_382 = None
        c_19 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y_19 = input_383 + c_19
        input_383 = c_19 = None
        input_384 = torch.nn.functional.silu(y_19, inplace=False)
        input_385 = torch._C._nn.linear(
            input_384,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_384 = None
        chunk_76 = input_385.chunk(3, dim=-1)
        input_385 = None
        shift_mlp_57 = chunk_76[0]
        scale_mlp_57 = chunk_76[1]
        gate_mlp_57 = chunk_76[2]
        chunk_76 = None
        layer_norm_101 = torch.nn.functional.layer_norm(
            x_136,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_298 = 1 + scale_mlp_57
        scale_mlp_57 = None
        mul_292 = layer_norm_101 * add_298
        layer_norm_101 = add_298 = None
        h_57 = mul_292 + shift_mlp_57
        mul_292 = shift_mlp_57 = None
        input_386 = torch._C._nn.linear(
            h_57,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_57 = None
        input_387 = torch.nn.functional.silu(input_386, inplace=False)
        input_386 = None
        input_388 = torch._C._nn.linear(
            input_387,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_387 = None
        mul_293 = gate_mlp_57 * input_388
        gate_mlp_57 = input_388 = None
        x_137 = x_136 + mul_293
        x_136 = mul_293 = None
        input_389 = torch.nn.functional.silu(y_19, inplace=False)
        input_390 = torch._C._nn.linear(
            input_389,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_389 = None
        chunk_77 = input_390.chunk(3, dim=-1)
        input_390 = None
        shift_mlp_58 = chunk_77[0]
        scale_mlp_58 = chunk_77[1]
        gate_mlp_58 = chunk_77[2]
        chunk_77 = None
        layer_norm_102 = torch.nn.functional.layer_norm(
            x_137,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_301 = 1 + scale_mlp_58
        scale_mlp_58 = None
        mul_294 = layer_norm_102 * add_301
        layer_norm_102 = add_301 = None
        h_58 = mul_294 + shift_mlp_58
        mul_294 = shift_mlp_58 = None
        input_391 = torch._C._nn.linear(
            h_58,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_58 = None
        input_392 = torch.nn.functional.silu(input_391, inplace=False)
        input_391 = None
        input_393 = torch._C._nn.linear(
            input_392,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_392 = None
        mul_295 = gate_mlp_58 * input_393
        gate_mlp_58 = input_393 = None
        x_138 = x_137 + mul_295
        x_137 = mul_295 = None
        input_394 = torch.nn.functional.silu(y_19, inplace=False)
        input_395 = torch._C._nn.linear(
            input_394,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_394 = None
        chunk_78 = input_395.chunk(3, dim=-1)
        input_395 = None
        shift_mlp_59 = chunk_78[0]
        scale_mlp_59 = chunk_78[1]
        gate_mlp_59 = chunk_78[2]
        chunk_78 = None
        layer_norm_103 = torch.nn.functional.layer_norm(
            x_138,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_304 = 1 + scale_mlp_59
        scale_mlp_59 = None
        mul_296 = layer_norm_103 * add_304
        layer_norm_103 = add_304 = None
        h_59 = mul_296 + shift_mlp_59
        mul_296 = shift_mlp_59 = None
        input_396 = torch._C._nn.linear(
            h_59,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_59 = None
        input_397 = torch.nn.functional.silu(input_396, inplace=False)
        input_396 = None
        input_398 = torch._C._nn.linear(
            input_397,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_397 = None
        mul_297 = gate_mlp_59 * input_398
        gate_mlp_59 = input_398 = None
        x_139 = x_138 + mul_297
        x_138 = mul_297 = None
        input_399 = torch.nn.functional.silu(y_19, inplace=False)
        y_19 = None
        input_400 = torch._C._nn.linear(
            input_399,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_399 = None
        chunk_79 = input_400.chunk(2, dim=-1)
        input_400 = None
        shift_19 = chunk_79[0]
        scale_19 = chunk_79[1]
        chunk_79 = None
        layer_norm_104 = torch.nn.functional.layer_norm(
            x_139, (768,), None, None, 1e-06
        )
        x_139 = None
        add_307 = 1 + scale_19
        scale_19 = None
        mul_298 = layer_norm_104 * add_307
        layer_norm_104 = add_307 = None
        x_140 = mul_298 + shift_19
        mul_298 = shift_19 = None
        x_141 = torch._C._nn.linear(
            x_140,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_140 = None
        sub_19 = x_141 - noise
        x_141 = None
        mul_299 = sub_19 * 0.02
        sub_19 = None
        x_142 = x_135 + mul_299
        x_135 = mul_299 = None
        ones_20 = torch.ones(1)
        mul_300 = ones_20 * 20
        ones_20 = None
        truediv_40 = mul_300 / 50
        mul_300 = None
        t_20 = truediv_40.to(device(type="cuda", index=0))
        truediv_40 = None
        mul_301 = t_20 * 1000
        t_20 = None
        x_143 = torch._C._nn.linear(
            x_142,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_22 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_302 = -9.210340371976184 * arange_22
        arange_22 = None
        truediv_41 = mul_302 / 128
        mul_302 = None
        exp_20 = torch.exp(truediv_41)
        truediv_41 = None
        freqs_20 = exp_20.to(device=device(type="cuda", index=0))
        exp_20 = None
        getitem_358 = mul_301[(slice(None, None, None), None)]
        mul_301 = None
        float_21 = getitem_358.float()
        getitem_358 = None
        getitem_359 = freqs_20[None]
        freqs_20 = None
        args_20 = float_21 * getitem_359
        float_21 = getitem_359 = None
        cos_44 = torch.cos(args_20)
        sin_44 = torch.sin(args_20)
        args_20 = None
        embedding_20 = torch.cat([cos_44, sin_44], dim=-1)
        cos_44 = sin_44 = None
        input_401 = torch._C._nn.linear(
            embedding_20,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_20 = None
        input_402 = torch.nn.functional.silu(input_401, inplace=False)
        input_401 = None
        input_403 = torch._C._nn.linear(
            input_402,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_402 = None
        c_20 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y_20 = input_403 + c_20
        input_403 = c_20 = None
        input_404 = torch.nn.functional.silu(y_20, inplace=False)
        input_405 = torch._C._nn.linear(
            input_404,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_404 = None
        chunk_80 = input_405.chunk(3, dim=-1)
        input_405 = None
        shift_mlp_60 = chunk_80[0]
        scale_mlp_60 = chunk_80[1]
        gate_mlp_60 = chunk_80[2]
        chunk_80 = None
        layer_norm_105 = torch.nn.functional.layer_norm(
            x_143,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_311 = 1 + scale_mlp_60
        scale_mlp_60 = None
        mul_304 = layer_norm_105 * add_311
        layer_norm_105 = add_311 = None
        h_60 = mul_304 + shift_mlp_60
        mul_304 = shift_mlp_60 = None
        input_406 = torch._C._nn.linear(
            h_60,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_60 = None
        input_407 = torch.nn.functional.silu(input_406, inplace=False)
        input_406 = None
        input_408 = torch._C._nn.linear(
            input_407,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_407 = None
        mul_305 = gate_mlp_60 * input_408
        gate_mlp_60 = input_408 = None
        x_144 = x_143 + mul_305
        x_143 = mul_305 = None
        input_409 = torch.nn.functional.silu(y_20, inplace=False)
        input_410 = torch._C._nn.linear(
            input_409,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_409 = None
        chunk_81 = input_410.chunk(3, dim=-1)
        input_410 = None
        shift_mlp_61 = chunk_81[0]
        scale_mlp_61 = chunk_81[1]
        gate_mlp_61 = chunk_81[2]
        chunk_81 = None
        layer_norm_106 = torch.nn.functional.layer_norm(
            x_144,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_314 = 1 + scale_mlp_61
        scale_mlp_61 = None
        mul_306 = layer_norm_106 * add_314
        layer_norm_106 = add_314 = None
        h_61 = mul_306 + shift_mlp_61
        mul_306 = shift_mlp_61 = None
        input_411 = torch._C._nn.linear(
            h_61,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_61 = None
        input_412 = torch.nn.functional.silu(input_411, inplace=False)
        input_411 = None
        input_413 = torch._C._nn.linear(
            input_412,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_412 = None
        mul_307 = gate_mlp_61 * input_413
        gate_mlp_61 = input_413 = None
        x_145 = x_144 + mul_307
        x_144 = mul_307 = None
        input_414 = torch.nn.functional.silu(y_20, inplace=False)
        input_415 = torch._C._nn.linear(
            input_414,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_414 = None
        chunk_82 = input_415.chunk(3, dim=-1)
        input_415 = None
        shift_mlp_62 = chunk_82[0]
        scale_mlp_62 = chunk_82[1]
        gate_mlp_62 = chunk_82[2]
        chunk_82 = None
        layer_norm_107 = torch.nn.functional.layer_norm(
            x_145,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_317 = 1 + scale_mlp_62
        scale_mlp_62 = None
        mul_308 = layer_norm_107 * add_317
        layer_norm_107 = add_317 = None
        h_62 = mul_308 + shift_mlp_62
        mul_308 = shift_mlp_62 = None
        input_416 = torch._C._nn.linear(
            h_62,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_62 = None
        input_417 = torch.nn.functional.silu(input_416, inplace=False)
        input_416 = None
        input_418 = torch._C._nn.linear(
            input_417,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_417 = None
        mul_309 = gate_mlp_62 * input_418
        gate_mlp_62 = input_418 = None
        x_146 = x_145 + mul_309
        x_145 = mul_309 = None
        input_419 = torch.nn.functional.silu(y_20, inplace=False)
        y_20 = None
        input_420 = torch._C._nn.linear(
            input_419,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_419 = None
        chunk_83 = input_420.chunk(2, dim=-1)
        input_420 = None
        shift_20 = chunk_83[0]
        scale_20 = chunk_83[1]
        chunk_83 = None
        layer_norm_108 = torch.nn.functional.layer_norm(
            x_146, (768,), None, None, 1e-06
        )
        x_146 = None
        add_320 = 1 + scale_20
        scale_20 = None
        mul_310 = layer_norm_108 * add_320
        layer_norm_108 = add_320 = None
        x_147 = mul_310 + shift_20
        mul_310 = shift_20 = None
        x_148 = torch._C._nn.linear(
            x_147,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_147 = None
        sub_20 = x_148 - noise
        x_148 = None
        mul_311 = sub_20 * 0.02
        sub_20 = None
        x_149 = x_142 + mul_311
        x_142 = mul_311 = None
        ones_21 = torch.ones(1)
        mul_312 = ones_21 * 21
        ones_21 = None
        truediv_42 = mul_312 / 50
        mul_312 = None
        t_21 = truediv_42.to(device(type="cuda", index=0))
        truediv_42 = None
        mul_313 = t_21 * 1000
        t_21 = None
        x_150 = torch._C._nn.linear(
            x_149,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_23 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_314 = -9.210340371976184 * arange_23
        arange_23 = None
        truediv_43 = mul_314 / 128
        mul_314 = None
        exp_21 = torch.exp(truediv_43)
        truediv_43 = None
        freqs_21 = exp_21.to(device=device(type="cuda", index=0))
        exp_21 = None
        getitem_371 = mul_313[(slice(None, None, None), None)]
        mul_313 = None
        float_22 = getitem_371.float()
        getitem_371 = None
        getitem_372 = freqs_21[None]
        freqs_21 = None
        args_21 = float_22 * getitem_372
        float_22 = getitem_372 = None
        cos_45 = torch.cos(args_21)
        sin_45 = torch.sin(args_21)
        args_21 = None
        embedding_21 = torch.cat([cos_45, sin_45], dim=-1)
        cos_45 = sin_45 = None
        input_421 = torch._C._nn.linear(
            embedding_21,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_21 = None
        input_422 = torch.nn.functional.silu(input_421, inplace=False)
        input_421 = None
        input_423 = torch._C._nn.linear(
            input_422,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_422 = None
        c_21 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y_21 = input_423 + c_21
        input_423 = c_21 = None
        input_424 = torch.nn.functional.silu(y_21, inplace=False)
        input_425 = torch._C._nn.linear(
            input_424,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_424 = None
        chunk_84 = input_425.chunk(3, dim=-1)
        input_425 = None
        shift_mlp_63 = chunk_84[0]
        scale_mlp_63 = chunk_84[1]
        gate_mlp_63 = chunk_84[2]
        chunk_84 = None
        layer_norm_109 = torch.nn.functional.layer_norm(
            x_150,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_324 = 1 + scale_mlp_63
        scale_mlp_63 = None
        mul_316 = layer_norm_109 * add_324
        layer_norm_109 = add_324 = None
        h_63 = mul_316 + shift_mlp_63
        mul_316 = shift_mlp_63 = None
        input_426 = torch._C._nn.linear(
            h_63,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_63 = None
        input_427 = torch.nn.functional.silu(input_426, inplace=False)
        input_426 = None
        input_428 = torch._C._nn.linear(
            input_427,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_427 = None
        mul_317 = gate_mlp_63 * input_428
        gate_mlp_63 = input_428 = None
        x_151 = x_150 + mul_317
        x_150 = mul_317 = None
        input_429 = torch.nn.functional.silu(y_21, inplace=False)
        input_430 = torch._C._nn.linear(
            input_429,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_429 = None
        chunk_85 = input_430.chunk(3, dim=-1)
        input_430 = None
        shift_mlp_64 = chunk_85[0]
        scale_mlp_64 = chunk_85[1]
        gate_mlp_64 = chunk_85[2]
        chunk_85 = None
        layer_norm_110 = torch.nn.functional.layer_norm(
            x_151,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_327 = 1 + scale_mlp_64
        scale_mlp_64 = None
        mul_318 = layer_norm_110 * add_327
        layer_norm_110 = add_327 = None
        h_64 = mul_318 + shift_mlp_64
        mul_318 = shift_mlp_64 = None
        input_431 = torch._C._nn.linear(
            h_64,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_64 = None
        input_432 = torch.nn.functional.silu(input_431, inplace=False)
        input_431 = None
        input_433 = torch._C._nn.linear(
            input_432,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_432 = None
        mul_319 = gate_mlp_64 * input_433
        gate_mlp_64 = input_433 = None
        x_152 = x_151 + mul_319
        x_151 = mul_319 = None
        input_434 = torch.nn.functional.silu(y_21, inplace=False)
        input_435 = torch._C._nn.linear(
            input_434,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_434 = None
        chunk_86 = input_435.chunk(3, dim=-1)
        input_435 = None
        shift_mlp_65 = chunk_86[0]
        scale_mlp_65 = chunk_86[1]
        gate_mlp_65 = chunk_86[2]
        chunk_86 = None
        layer_norm_111 = torch.nn.functional.layer_norm(
            x_152,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_330 = 1 + scale_mlp_65
        scale_mlp_65 = None
        mul_320 = layer_norm_111 * add_330
        layer_norm_111 = add_330 = None
        h_65 = mul_320 + shift_mlp_65
        mul_320 = shift_mlp_65 = None
        input_436 = torch._C._nn.linear(
            h_65,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_65 = None
        input_437 = torch.nn.functional.silu(input_436, inplace=False)
        input_436 = None
        input_438 = torch._C._nn.linear(
            input_437,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_437 = None
        mul_321 = gate_mlp_65 * input_438
        gate_mlp_65 = input_438 = None
        x_153 = x_152 + mul_321
        x_152 = mul_321 = None
        input_439 = torch.nn.functional.silu(y_21, inplace=False)
        y_21 = None
        input_440 = torch._C._nn.linear(
            input_439,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_439 = None
        chunk_87 = input_440.chunk(2, dim=-1)
        input_440 = None
        shift_21 = chunk_87[0]
        scale_21 = chunk_87[1]
        chunk_87 = None
        layer_norm_112 = torch.nn.functional.layer_norm(
            x_153, (768,), None, None, 1e-06
        )
        x_153 = None
        add_333 = 1 + scale_21
        scale_21 = None
        mul_322 = layer_norm_112 * add_333
        layer_norm_112 = add_333 = None
        x_154 = mul_322 + shift_21
        mul_322 = shift_21 = None
        x_155 = torch._C._nn.linear(
            x_154,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_154 = None
        sub_21 = x_155 - noise
        x_155 = None
        mul_323 = sub_21 * 0.02
        sub_21 = None
        x_156 = x_149 + mul_323
        x_149 = mul_323 = None
        ones_22 = torch.ones(1)
        mul_324 = ones_22 * 22
        ones_22 = None
        truediv_44 = mul_324 / 50
        mul_324 = None
        t_22 = truediv_44.to(device(type="cuda", index=0))
        truediv_44 = None
        mul_325 = t_22 * 1000
        t_22 = None
        x_157 = torch._C._nn.linear(
            x_156,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_24 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_326 = -9.210340371976184 * arange_24
        arange_24 = None
        truediv_45 = mul_326 / 128
        mul_326 = None
        exp_22 = torch.exp(truediv_45)
        truediv_45 = None
        freqs_22 = exp_22.to(device=device(type="cuda", index=0))
        exp_22 = None
        getitem_384 = mul_325[(slice(None, None, None), None)]
        mul_325 = None
        float_23 = getitem_384.float()
        getitem_384 = None
        getitem_385 = freqs_22[None]
        freqs_22 = None
        args_22 = float_23 * getitem_385
        float_23 = getitem_385 = None
        cos_46 = torch.cos(args_22)
        sin_46 = torch.sin(args_22)
        args_22 = None
        embedding_22 = torch.cat([cos_46, sin_46], dim=-1)
        cos_46 = sin_46 = None
        input_441 = torch._C._nn.linear(
            embedding_22,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_22 = None
        input_442 = torch.nn.functional.silu(input_441, inplace=False)
        input_441 = None
        input_443 = torch._C._nn.linear(
            input_442,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_442 = None
        c_22 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y_22 = input_443 + c_22
        input_443 = c_22 = None
        input_444 = torch.nn.functional.silu(y_22, inplace=False)
        input_445 = torch._C._nn.linear(
            input_444,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_444 = None
        chunk_88 = input_445.chunk(3, dim=-1)
        input_445 = None
        shift_mlp_66 = chunk_88[0]
        scale_mlp_66 = chunk_88[1]
        gate_mlp_66 = chunk_88[2]
        chunk_88 = None
        layer_norm_113 = torch.nn.functional.layer_norm(
            x_157,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_337 = 1 + scale_mlp_66
        scale_mlp_66 = None
        mul_328 = layer_norm_113 * add_337
        layer_norm_113 = add_337 = None
        h_66 = mul_328 + shift_mlp_66
        mul_328 = shift_mlp_66 = None
        input_446 = torch._C._nn.linear(
            h_66,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_66 = None
        input_447 = torch.nn.functional.silu(input_446, inplace=False)
        input_446 = None
        input_448 = torch._C._nn.linear(
            input_447,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_447 = None
        mul_329 = gate_mlp_66 * input_448
        gate_mlp_66 = input_448 = None
        x_158 = x_157 + mul_329
        x_157 = mul_329 = None
        input_449 = torch.nn.functional.silu(y_22, inplace=False)
        input_450 = torch._C._nn.linear(
            input_449,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_449 = None
        chunk_89 = input_450.chunk(3, dim=-1)
        input_450 = None
        shift_mlp_67 = chunk_89[0]
        scale_mlp_67 = chunk_89[1]
        gate_mlp_67 = chunk_89[2]
        chunk_89 = None
        layer_norm_114 = torch.nn.functional.layer_norm(
            x_158,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_340 = 1 + scale_mlp_67
        scale_mlp_67 = None
        mul_330 = layer_norm_114 * add_340
        layer_norm_114 = add_340 = None
        h_67 = mul_330 + shift_mlp_67
        mul_330 = shift_mlp_67 = None
        input_451 = torch._C._nn.linear(
            h_67,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_67 = None
        input_452 = torch.nn.functional.silu(input_451, inplace=False)
        input_451 = None
        input_453 = torch._C._nn.linear(
            input_452,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_452 = None
        mul_331 = gate_mlp_67 * input_453
        gate_mlp_67 = input_453 = None
        x_159 = x_158 + mul_331
        x_158 = mul_331 = None
        input_454 = torch.nn.functional.silu(y_22, inplace=False)
        input_455 = torch._C._nn.linear(
            input_454,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_454 = None
        chunk_90 = input_455.chunk(3, dim=-1)
        input_455 = None
        shift_mlp_68 = chunk_90[0]
        scale_mlp_68 = chunk_90[1]
        gate_mlp_68 = chunk_90[2]
        chunk_90 = None
        layer_norm_115 = torch.nn.functional.layer_norm(
            x_159,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_343 = 1 + scale_mlp_68
        scale_mlp_68 = None
        mul_332 = layer_norm_115 * add_343
        layer_norm_115 = add_343 = None
        h_68 = mul_332 + shift_mlp_68
        mul_332 = shift_mlp_68 = None
        input_456 = torch._C._nn.linear(
            h_68,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_68 = None
        input_457 = torch.nn.functional.silu(input_456, inplace=False)
        input_456 = None
        input_458 = torch._C._nn.linear(
            input_457,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_457 = None
        mul_333 = gate_mlp_68 * input_458
        gate_mlp_68 = input_458 = None
        x_160 = x_159 + mul_333
        x_159 = mul_333 = None
        input_459 = torch.nn.functional.silu(y_22, inplace=False)
        y_22 = None
        input_460 = torch._C._nn.linear(
            input_459,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_459 = None
        chunk_91 = input_460.chunk(2, dim=-1)
        input_460 = None
        shift_22 = chunk_91[0]
        scale_22 = chunk_91[1]
        chunk_91 = None
        layer_norm_116 = torch.nn.functional.layer_norm(
            x_160, (768,), None, None, 1e-06
        )
        x_160 = None
        add_346 = 1 + scale_22
        scale_22 = None
        mul_334 = layer_norm_116 * add_346
        layer_norm_116 = add_346 = None
        x_161 = mul_334 + shift_22
        mul_334 = shift_22 = None
        x_162 = torch._C._nn.linear(
            x_161,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_161 = None
        sub_22 = x_162 - noise
        x_162 = None
        mul_335 = sub_22 * 0.02
        sub_22 = None
        x_163 = x_156 + mul_335
        x_156 = mul_335 = None
        ones_23 = torch.ones(1)
        mul_336 = ones_23 * 23
        ones_23 = None
        truediv_46 = mul_336 / 50
        mul_336 = None
        t_23 = truediv_46.to(device(type="cuda", index=0))
        truediv_46 = None
        mul_337 = t_23 * 1000
        t_23 = None
        x_164 = torch._C._nn.linear(
            x_163,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_25 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_338 = -9.210340371976184 * arange_25
        arange_25 = None
        truediv_47 = mul_338 / 128
        mul_338 = None
        exp_23 = torch.exp(truediv_47)
        truediv_47 = None
        freqs_23 = exp_23.to(device=device(type="cuda", index=0))
        exp_23 = None
        getitem_397 = mul_337[(slice(None, None, None), None)]
        mul_337 = None
        float_24 = getitem_397.float()
        getitem_397 = None
        getitem_398 = freqs_23[None]
        freqs_23 = None
        args_23 = float_24 * getitem_398
        float_24 = getitem_398 = None
        cos_47 = torch.cos(args_23)
        sin_47 = torch.sin(args_23)
        args_23 = None
        embedding_23 = torch.cat([cos_47, sin_47], dim=-1)
        cos_47 = sin_47 = None
        input_461 = torch._C._nn.linear(
            embedding_23,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_23 = None
        input_462 = torch.nn.functional.silu(input_461, inplace=False)
        input_461 = None
        input_463 = torch._C._nn.linear(
            input_462,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_462 = None
        c_23 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y_23 = input_463 + c_23
        input_463 = c_23 = None
        input_464 = torch.nn.functional.silu(y_23, inplace=False)
        input_465 = torch._C._nn.linear(
            input_464,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_464 = None
        chunk_92 = input_465.chunk(3, dim=-1)
        input_465 = None
        shift_mlp_69 = chunk_92[0]
        scale_mlp_69 = chunk_92[1]
        gate_mlp_69 = chunk_92[2]
        chunk_92 = None
        layer_norm_117 = torch.nn.functional.layer_norm(
            x_164,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_350 = 1 + scale_mlp_69
        scale_mlp_69 = None
        mul_340 = layer_norm_117 * add_350
        layer_norm_117 = add_350 = None
        h_69 = mul_340 + shift_mlp_69
        mul_340 = shift_mlp_69 = None
        input_466 = torch._C._nn.linear(
            h_69,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_69 = None
        input_467 = torch.nn.functional.silu(input_466, inplace=False)
        input_466 = None
        input_468 = torch._C._nn.linear(
            input_467,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_467 = None
        mul_341 = gate_mlp_69 * input_468
        gate_mlp_69 = input_468 = None
        x_165 = x_164 + mul_341
        x_164 = mul_341 = None
        input_469 = torch.nn.functional.silu(y_23, inplace=False)
        input_470 = torch._C._nn.linear(
            input_469,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_469 = None
        chunk_93 = input_470.chunk(3, dim=-1)
        input_470 = None
        shift_mlp_70 = chunk_93[0]
        scale_mlp_70 = chunk_93[1]
        gate_mlp_70 = chunk_93[2]
        chunk_93 = None
        layer_norm_118 = torch.nn.functional.layer_norm(
            x_165,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_353 = 1 + scale_mlp_70
        scale_mlp_70 = None
        mul_342 = layer_norm_118 * add_353
        layer_norm_118 = add_353 = None
        h_70 = mul_342 + shift_mlp_70
        mul_342 = shift_mlp_70 = None
        input_471 = torch._C._nn.linear(
            h_70,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_70 = None
        input_472 = torch.nn.functional.silu(input_471, inplace=False)
        input_471 = None
        input_473 = torch._C._nn.linear(
            input_472,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_472 = None
        mul_343 = gate_mlp_70 * input_473
        gate_mlp_70 = input_473 = None
        x_166 = x_165 + mul_343
        x_165 = mul_343 = None
        input_474 = torch.nn.functional.silu(y_23, inplace=False)
        input_475 = torch._C._nn.linear(
            input_474,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_474 = None
        chunk_94 = input_475.chunk(3, dim=-1)
        input_475 = None
        shift_mlp_71 = chunk_94[0]
        scale_mlp_71 = chunk_94[1]
        gate_mlp_71 = chunk_94[2]
        chunk_94 = None
        layer_norm_119 = torch.nn.functional.layer_norm(
            x_166,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_356 = 1 + scale_mlp_71
        scale_mlp_71 = None
        mul_344 = layer_norm_119 * add_356
        layer_norm_119 = add_356 = None
        h_71 = mul_344 + shift_mlp_71
        mul_344 = shift_mlp_71 = None
        input_476 = torch._C._nn.linear(
            h_71,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_71 = None
        input_477 = torch.nn.functional.silu(input_476, inplace=False)
        input_476 = None
        input_478 = torch._C._nn.linear(
            input_477,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_477 = None
        mul_345 = gate_mlp_71 * input_478
        gate_mlp_71 = input_478 = None
        x_167 = x_166 + mul_345
        x_166 = mul_345 = None
        input_479 = torch.nn.functional.silu(y_23, inplace=False)
        y_23 = None
        input_480 = torch._C._nn.linear(
            input_479,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_479 = None
        chunk_95 = input_480.chunk(2, dim=-1)
        input_480 = None
        shift_23 = chunk_95[0]
        scale_23 = chunk_95[1]
        chunk_95 = None
        layer_norm_120 = torch.nn.functional.layer_norm(
            x_167, (768,), None, None, 1e-06
        )
        x_167 = None
        add_359 = 1 + scale_23
        scale_23 = None
        mul_346 = layer_norm_120 * add_359
        layer_norm_120 = add_359 = None
        x_168 = mul_346 + shift_23
        mul_346 = shift_23 = None
        x_169 = torch._C._nn.linear(
            x_168,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_168 = None
        sub_23 = x_169 - noise
        x_169 = None
        mul_347 = sub_23 * 0.02
        sub_23 = None
        x_170 = x_163 + mul_347
        x_163 = mul_347 = None
        ones_24 = torch.ones(1)
        mul_348 = ones_24 * 24
        ones_24 = None
        truediv_48 = mul_348 / 50
        mul_348 = None
        t_24 = truediv_48.to(device(type="cuda", index=0))
        truediv_48 = None
        mul_349 = t_24 * 1000
        t_24 = None
        x_171 = torch._C._nn.linear(
            x_170,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_26 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_350 = -9.210340371976184 * arange_26
        arange_26 = None
        truediv_49 = mul_350 / 128
        mul_350 = None
        exp_24 = torch.exp(truediv_49)
        truediv_49 = None
        freqs_24 = exp_24.to(device=device(type="cuda", index=0))
        exp_24 = None
        getitem_410 = mul_349[(slice(None, None, None), None)]
        mul_349 = None
        float_25 = getitem_410.float()
        getitem_410 = None
        getitem_411 = freqs_24[None]
        freqs_24 = None
        args_24 = float_25 * getitem_411
        float_25 = getitem_411 = None
        cos_48 = torch.cos(args_24)
        sin_48 = torch.sin(args_24)
        args_24 = None
        embedding_24 = torch.cat([cos_48, sin_48], dim=-1)
        cos_48 = sin_48 = None
        input_481 = torch._C._nn.linear(
            embedding_24,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_24 = None
        input_482 = torch.nn.functional.silu(input_481, inplace=False)
        input_481 = None
        input_483 = torch._C._nn.linear(
            input_482,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_482 = None
        c_24 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y_24 = input_483 + c_24
        input_483 = c_24 = None
        input_484 = torch.nn.functional.silu(y_24, inplace=False)
        input_485 = torch._C._nn.linear(
            input_484,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_484 = None
        chunk_96 = input_485.chunk(3, dim=-1)
        input_485 = None
        shift_mlp_72 = chunk_96[0]
        scale_mlp_72 = chunk_96[1]
        gate_mlp_72 = chunk_96[2]
        chunk_96 = None
        layer_norm_121 = torch.nn.functional.layer_norm(
            x_171,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_363 = 1 + scale_mlp_72
        scale_mlp_72 = None
        mul_352 = layer_norm_121 * add_363
        layer_norm_121 = add_363 = None
        h_72 = mul_352 + shift_mlp_72
        mul_352 = shift_mlp_72 = None
        input_486 = torch._C._nn.linear(
            h_72,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_72 = None
        input_487 = torch.nn.functional.silu(input_486, inplace=False)
        input_486 = None
        input_488 = torch._C._nn.linear(
            input_487,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_487 = None
        mul_353 = gate_mlp_72 * input_488
        gate_mlp_72 = input_488 = None
        x_172 = x_171 + mul_353
        x_171 = mul_353 = None
        input_489 = torch.nn.functional.silu(y_24, inplace=False)
        input_490 = torch._C._nn.linear(
            input_489,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_489 = None
        chunk_97 = input_490.chunk(3, dim=-1)
        input_490 = None
        shift_mlp_73 = chunk_97[0]
        scale_mlp_73 = chunk_97[1]
        gate_mlp_73 = chunk_97[2]
        chunk_97 = None
        layer_norm_122 = torch.nn.functional.layer_norm(
            x_172,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_366 = 1 + scale_mlp_73
        scale_mlp_73 = None
        mul_354 = layer_norm_122 * add_366
        layer_norm_122 = add_366 = None
        h_73 = mul_354 + shift_mlp_73
        mul_354 = shift_mlp_73 = None
        input_491 = torch._C._nn.linear(
            h_73,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_73 = None
        input_492 = torch.nn.functional.silu(input_491, inplace=False)
        input_491 = None
        input_493 = torch._C._nn.linear(
            input_492,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_492 = None
        mul_355 = gate_mlp_73 * input_493
        gate_mlp_73 = input_493 = None
        x_173 = x_172 + mul_355
        x_172 = mul_355 = None
        input_494 = torch.nn.functional.silu(y_24, inplace=False)
        input_495 = torch._C._nn.linear(
            input_494,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_494 = None
        chunk_98 = input_495.chunk(3, dim=-1)
        input_495 = None
        shift_mlp_74 = chunk_98[0]
        scale_mlp_74 = chunk_98[1]
        gate_mlp_74 = chunk_98[2]
        chunk_98 = None
        layer_norm_123 = torch.nn.functional.layer_norm(
            x_173,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_369 = 1 + scale_mlp_74
        scale_mlp_74 = None
        mul_356 = layer_norm_123 * add_369
        layer_norm_123 = add_369 = None
        h_74 = mul_356 + shift_mlp_74
        mul_356 = shift_mlp_74 = None
        input_496 = torch._C._nn.linear(
            h_74,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_74 = None
        input_497 = torch.nn.functional.silu(input_496, inplace=False)
        input_496 = None
        input_498 = torch._C._nn.linear(
            input_497,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_497 = None
        mul_357 = gate_mlp_74 * input_498
        gate_mlp_74 = input_498 = None
        x_174 = x_173 + mul_357
        x_173 = mul_357 = None
        input_499 = torch.nn.functional.silu(y_24, inplace=False)
        y_24 = None
        input_500 = torch._C._nn.linear(
            input_499,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_499 = None
        chunk_99 = input_500.chunk(2, dim=-1)
        input_500 = None
        shift_24 = chunk_99[0]
        scale_24 = chunk_99[1]
        chunk_99 = None
        layer_norm_124 = torch.nn.functional.layer_norm(
            x_174, (768,), None, None, 1e-06
        )
        x_174 = None
        add_372 = 1 + scale_24
        scale_24 = None
        mul_358 = layer_norm_124 * add_372
        layer_norm_124 = add_372 = None
        x_175 = mul_358 + shift_24
        mul_358 = shift_24 = None
        x_176 = torch._C._nn.linear(
            x_175,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_175 = None
        sub_24 = x_176 - noise
        x_176 = None
        mul_359 = sub_24 * 0.02
        sub_24 = None
        x_177 = x_170 + mul_359
        x_170 = mul_359 = None
        ones_25 = torch.ones(1)
        mul_360 = ones_25 * 25
        ones_25 = None
        truediv_50 = mul_360 / 50
        mul_360 = None
        t_25 = truediv_50.to(device(type="cuda", index=0))
        truediv_50 = None
        mul_361 = t_25 * 1000
        t_25 = None
        x_178 = torch._C._nn.linear(
            x_177,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_27 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_362 = -9.210340371976184 * arange_27
        arange_27 = None
        truediv_51 = mul_362 / 128
        mul_362 = None
        exp_25 = torch.exp(truediv_51)
        truediv_51 = None
        freqs_25 = exp_25.to(device=device(type="cuda", index=0))
        exp_25 = None
        getitem_423 = mul_361[(slice(None, None, None), None)]
        mul_361 = None
        float_26 = getitem_423.float()
        getitem_423 = None
        getitem_424 = freqs_25[None]
        freqs_25 = None
        args_25 = float_26 * getitem_424
        float_26 = getitem_424 = None
        cos_49 = torch.cos(args_25)
        sin_49 = torch.sin(args_25)
        args_25 = None
        embedding_25 = torch.cat([cos_49, sin_49], dim=-1)
        cos_49 = sin_49 = None
        input_501 = torch._C._nn.linear(
            embedding_25,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_25 = None
        input_502 = torch.nn.functional.silu(input_501, inplace=False)
        input_501 = None
        input_503 = torch._C._nn.linear(
            input_502,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_502 = None
        c_25 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y_25 = input_503 + c_25
        input_503 = c_25 = None
        input_504 = torch.nn.functional.silu(y_25, inplace=False)
        input_505 = torch._C._nn.linear(
            input_504,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_504 = None
        chunk_100 = input_505.chunk(3, dim=-1)
        input_505 = None
        shift_mlp_75 = chunk_100[0]
        scale_mlp_75 = chunk_100[1]
        gate_mlp_75 = chunk_100[2]
        chunk_100 = None
        layer_norm_125 = torch.nn.functional.layer_norm(
            x_178,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_376 = 1 + scale_mlp_75
        scale_mlp_75 = None
        mul_364 = layer_norm_125 * add_376
        layer_norm_125 = add_376 = None
        h_75 = mul_364 + shift_mlp_75
        mul_364 = shift_mlp_75 = None
        input_506 = torch._C._nn.linear(
            h_75,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_75 = None
        input_507 = torch.nn.functional.silu(input_506, inplace=False)
        input_506 = None
        input_508 = torch._C._nn.linear(
            input_507,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_507 = None
        mul_365 = gate_mlp_75 * input_508
        gate_mlp_75 = input_508 = None
        x_179 = x_178 + mul_365
        x_178 = mul_365 = None
        input_509 = torch.nn.functional.silu(y_25, inplace=False)
        input_510 = torch._C._nn.linear(
            input_509,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_509 = None
        chunk_101 = input_510.chunk(3, dim=-1)
        input_510 = None
        shift_mlp_76 = chunk_101[0]
        scale_mlp_76 = chunk_101[1]
        gate_mlp_76 = chunk_101[2]
        chunk_101 = None
        layer_norm_126 = torch.nn.functional.layer_norm(
            x_179,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_379 = 1 + scale_mlp_76
        scale_mlp_76 = None
        mul_366 = layer_norm_126 * add_379
        layer_norm_126 = add_379 = None
        h_76 = mul_366 + shift_mlp_76
        mul_366 = shift_mlp_76 = None
        input_511 = torch._C._nn.linear(
            h_76,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_76 = None
        input_512 = torch.nn.functional.silu(input_511, inplace=False)
        input_511 = None
        input_513 = torch._C._nn.linear(
            input_512,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_512 = None
        mul_367 = gate_mlp_76 * input_513
        gate_mlp_76 = input_513 = None
        x_180 = x_179 + mul_367
        x_179 = mul_367 = None
        input_514 = torch.nn.functional.silu(y_25, inplace=False)
        input_515 = torch._C._nn.linear(
            input_514,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_514 = None
        chunk_102 = input_515.chunk(3, dim=-1)
        input_515 = None
        shift_mlp_77 = chunk_102[0]
        scale_mlp_77 = chunk_102[1]
        gate_mlp_77 = chunk_102[2]
        chunk_102 = None
        layer_norm_127 = torch.nn.functional.layer_norm(
            x_180,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_382 = 1 + scale_mlp_77
        scale_mlp_77 = None
        mul_368 = layer_norm_127 * add_382
        layer_norm_127 = add_382 = None
        h_77 = mul_368 + shift_mlp_77
        mul_368 = shift_mlp_77 = None
        input_516 = torch._C._nn.linear(
            h_77,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_77 = None
        input_517 = torch.nn.functional.silu(input_516, inplace=False)
        input_516 = None
        input_518 = torch._C._nn.linear(
            input_517,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_517 = None
        mul_369 = gate_mlp_77 * input_518
        gate_mlp_77 = input_518 = None
        x_181 = x_180 + mul_369
        x_180 = mul_369 = None
        input_519 = torch.nn.functional.silu(y_25, inplace=False)
        y_25 = None
        input_520 = torch._C._nn.linear(
            input_519,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_519 = None
        chunk_103 = input_520.chunk(2, dim=-1)
        input_520 = None
        shift_25 = chunk_103[0]
        scale_25 = chunk_103[1]
        chunk_103 = None
        layer_norm_128 = torch.nn.functional.layer_norm(
            x_181, (768,), None, None, 1e-06
        )
        x_181 = None
        add_385 = 1 + scale_25
        scale_25 = None
        mul_370 = layer_norm_128 * add_385
        layer_norm_128 = add_385 = None
        x_182 = mul_370 + shift_25
        mul_370 = shift_25 = None
        x_183 = torch._C._nn.linear(
            x_182,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_182 = None
        sub_25 = x_183 - noise
        x_183 = None
        mul_371 = sub_25 * 0.02
        sub_25 = None
        x_184 = x_177 + mul_371
        x_177 = mul_371 = None
        ones_26 = torch.ones(1)
        mul_372 = ones_26 * 26
        ones_26 = None
        truediv_52 = mul_372 / 50
        mul_372 = None
        t_26 = truediv_52.to(device(type="cuda", index=0))
        truediv_52 = None
        mul_373 = t_26 * 1000
        t_26 = None
        x_185 = torch._C._nn.linear(
            x_184,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_28 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_374 = -9.210340371976184 * arange_28
        arange_28 = None
        truediv_53 = mul_374 / 128
        mul_374 = None
        exp_26 = torch.exp(truediv_53)
        truediv_53 = None
        freqs_26 = exp_26.to(device=device(type="cuda", index=0))
        exp_26 = None
        getitem_436 = mul_373[(slice(None, None, None), None)]
        mul_373 = None
        float_27 = getitem_436.float()
        getitem_436 = None
        getitem_437 = freqs_26[None]
        freqs_26 = None
        args_26 = float_27 * getitem_437
        float_27 = getitem_437 = None
        cos_50 = torch.cos(args_26)
        sin_50 = torch.sin(args_26)
        args_26 = None
        embedding_26 = torch.cat([cos_50, sin_50], dim=-1)
        cos_50 = sin_50 = None
        input_521 = torch._C._nn.linear(
            embedding_26,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_26 = None
        input_522 = torch.nn.functional.silu(input_521, inplace=False)
        input_521 = None
        input_523 = torch._C._nn.linear(
            input_522,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_522 = None
        c_26 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y_26 = input_523 + c_26
        input_523 = c_26 = None
        input_524 = torch.nn.functional.silu(y_26, inplace=False)
        input_525 = torch._C._nn.linear(
            input_524,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_524 = None
        chunk_104 = input_525.chunk(3, dim=-1)
        input_525 = None
        shift_mlp_78 = chunk_104[0]
        scale_mlp_78 = chunk_104[1]
        gate_mlp_78 = chunk_104[2]
        chunk_104 = None
        layer_norm_129 = torch.nn.functional.layer_norm(
            x_185,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_389 = 1 + scale_mlp_78
        scale_mlp_78 = None
        mul_376 = layer_norm_129 * add_389
        layer_norm_129 = add_389 = None
        h_78 = mul_376 + shift_mlp_78
        mul_376 = shift_mlp_78 = None
        input_526 = torch._C._nn.linear(
            h_78,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_78 = None
        input_527 = torch.nn.functional.silu(input_526, inplace=False)
        input_526 = None
        input_528 = torch._C._nn.linear(
            input_527,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_527 = None
        mul_377 = gate_mlp_78 * input_528
        gate_mlp_78 = input_528 = None
        x_186 = x_185 + mul_377
        x_185 = mul_377 = None
        input_529 = torch.nn.functional.silu(y_26, inplace=False)
        input_530 = torch._C._nn.linear(
            input_529,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_529 = None
        chunk_105 = input_530.chunk(3, dim=-1)
        input_530 = None
        shift_mlp_79 = chunk_105[0]
        scale_mlp_79 = chunk_105[1]
        gate_mlp_79 = chunk_105[2]
        chunk_105 = None
        layer_norm_130 = torch.nn.functional.layer_norm(
            x_186,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_392 = 1 + scale_mlp_79
        scale_mlp_79 = None
        mul_378 = layer_norm_130 * add_392
        layer_norm_130 = add_392 = None
        h_79 = mul_378 + shift_mlp_79
        mul_378 = shift_mlp_79 = None
        input_531 = torch._C._nn.linear(
            h_79,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_79 = None
        input_532 = torch.nn.functional.silu(input_531, inplace=False)
        input_531 = None
        input_533 = torch._C._nn.linear(
            input_532,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_532 = None
        mul_379 = gate_mlp_79 * input_533
        gate_mlp_79 = input_533 = None
        x_187 = x_186 + mul_379
        x_186 = mul_379 = None
        input_534 = torch.nn.functional.silu(y_26, inplace=False)
        input_535 = torch._C._nn.linear(
            input_534,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_534 = None
        chunk_106 = input_535.chunk(3, dim=-1)
        input_535 = None
        shift_mlp_80 = chunk_106[0]
        scale_mlp_80 = chunk_106[1]
        gate_mlp_80 = chunk_106[2]
        chunk_106 = None
        layer_norm_131 = torch.nn.functional.layer_norm(
            x_187,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_395 = 1 + scale_mlp_80
        scale_mlp_80 = None
        mul_380 = layer_norm_131 * add_395
        layer_norm_131 = add_395 = None
        h_80 = mul_380 + shift_mlp_80
        mul_380 = shift_mlp_80 = None
        input_536 = torch._C._nn.linear(
            h_80,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_80 = None
        input_537 = torch.nn.functional.silu(input_536, inplace=False)
        input_536 = None
        input_538 = torch._C._nn.linear(
            input_537,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_537 = None
        mul_381 = gate_mlp_80 * input_538
        gate_mlp_80 = input_538 = None
        x_188 = x_187 + mul_381
        x_187 = mul_381 = None
        input_539 = torch.nn.functional.silu(y_26, inplace=False)
        y_26 = None
        input_540 = torch._C._nn.linear(
            input_539,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_539 = None
        chunk_107 = input_540.chunk(2, dim=-1)
        input_540 = None
        shift_26 = chunk_107[0]
        scale_26 = chunk_107[1]
        chunk_107 = None
        layer_norm_132 = torch.nn.functional.layer_norm(
            x_188, (768,), None, None, 1e-06
        )
        x_188 = None
        add_398 = 1 + scale_26
        scale_26 = None
        mul_382 = layer_norm_132 * add_398
        layer_norm_132 = add_398 = None
        x_189 = mul_382 + shift_26
        mul_382 = shift_26 = None
        x_190 = torch._C._nn.linear(
            x_189,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_189 = None
        sub_26 = x_190 - noise
        x_190 = None
        mul_383 = sub_26 * 0.02
        sub_26 = None
        x_191 = x_184 + mul_383
        x_184 = mul_383 = None
        ones_27 = torch.ones(1)
        mul_384 = ones_27 * 27
        ones_27 = None
        truediv_54 = mul_384 / 50
        mul_384 = None
        t_27 = truediv_54.to(device(type="cuda", index=0))
        truediv_54 = None
        mul_385 = t_27 * 1000
        t_27 = None
        x_192 = torch._C._nn.linear(
            x_191,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_29 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_386 = -9.210340371976184 * arange_29
        arange_29 = None
        truediv_55 = mul_386 / 128
        mul_386 = None
        exp_27 = torch.exp(truediv_55)
        truediv_55 = None
        freqs_27 = exp_27.to(device=device(type="cuda", index=0))
        exp_27 = None
        getitem_449 = mul_385[(slice(None, None, None), None)]
        mul_385 = None
        float_28 = getitem_449.float()
        getitem_449 = None
        getitem_450 = freqs_27[None]
        freqs_27 = None
        args_27 = float_28 * getitem_450
        float_28 = getitem_450 = None
        cos_51 = torch.cos(args_27)
        sin_51 = torch.sin(args_27)
        args_27 = None
        embedding_27 = torch.cat([cos_51, sin_51], dim=-1)
        cos_51 = sin_51 = None
        input_541 = torch._C._nn.linear(
            embedding_27,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_27 = None
        input_542 = torch.nn.functional.silu(input_541, inplace=False)
        input_541 = None
        input_543 = torch._C._nn.linear(
            input_542,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_542 = None
        c_27 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y_27 = input_543 + c_27
        input_543 = c_27 = None
        input_544 = torch.nn.functional.silu(y_27, inplace=False)
        input_545 = torch._C._nn.linear(
            input_544,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_544 = None
        chunk_108 = input_545.chunk(3, dim=-1)
        input_545 = None
        shift_mlp_81 = chunk_108[0]
        scale_mlp_81 = chunk_108[1]
        gate_mlp_81 = chunk_108[2]
        chunk_108 = None
        layer_norm_133 = torch.nn.functional.layer_norm(
            x_192,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_402 = 1 + scale_mlp_81
        scale_mlp_81 = None
        mul_388 = layer_norm_133 * add_402
        layer_norm_133 = add_402 = None
        h_81 = mul_388 + shift_mlp_81
        mul_388 = shift_mlp_81 = None
        input_546 = torch._C._nn.linear(
            h_81,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_81 = None
        input_547 = torch.nn.functional.silu(input_546, inplace=False)
        input_546 = None
        input_548 = torch._C._nn.linear(
            input_547,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_547 = None
        mul_389 = gate_mlp_81 * input_548
        gate_mlp_81 = input_548 = None
        x_193 = x_192 + mul_389
        x_192 = mul_389 = None
        input_549 = torch.nn.functional.silu(y_27, inplace=False)
        input_550 = torch._C._nn.linear(
            input_549,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_549 = None
        chunk_109 = input_550.chunk(3, dim=-1)
        input_550 = None
        shift_mlp_82 = chunk_109[0]
        scale_mlp_82 = chunk_109[1]
        gate_mlp_82 = chunk_109[2]
        chunk_109 = None
        layer_norm_134 = torch.nn.functional.layer_norm(
            x_193,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_405 = 1 + scale_mlp_82
        scale_mlp_82 = None
        mul_390 = layer_norm_134 * add_405
        layer_norm_134 = add_405 = None
        h_82 = mul_390 + shift_mlp_82
        mul_390 = shift_mlp_82 = None
        input_551 = torch._C._nn.linear(
            h_82,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_82 = None
        input_552 = torch.nn.functional.silu(input_551, inplace=False)
        input_551 = None
        input_553 = torch._C._nn.linear(
            input_552,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_552 = None
        mul_391 = gate_mlp_82 * input_553
        gate_mlp_82 = input_553 = None
        x_194 = x_193 + mul_391
        x_193 = mul_391 = None
        input_554 = torch.nn.functional.silu(y_27, inplace=False)
        input_555 = torch._C._nn.linear(
            input_554,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_554 = None
        chunk_110 = input_555.chunk(3, dim=-1)
        input_555 = None
        shift_mlp_83 = chunk_110[0]
        scale_mlp_83 = chunk_110[1]
        gate_mlp_83 = chunk_110[2]
        chunk_110 = None
        layer_norm_135 = torch.nn.functional.layer_norm(
            x_194,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_408 = 1 + scale_mlp_83
        scale_mlp_83 = None
        mul_392 = layer_norm_135 * add_408
        layer_norm_135 = add_408 = None
        h_83 = mul_392 + shift_mlp_83
        mul_392 = shift_mlp_83 = None
        input_556 = torch._C._nn.linear(
            h_83,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_83 = None
        input_557 = torch.nn.functional.silu(input_556, inplace=False)
        input_556 = None
        input_558 = torch._C._nn.linear(
            input_557,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_557 = None
        mul_393 = gate_mlp_83 * input_558
        gate_mlp_83 = input_558 = None
        x_195 = x_194 + mul_393
        x_194 = mul_393 = None
        input_559 = torch.nn.functional.silu(y_27, inplace=False)
        y_27 = None
        input_560 = torch._C._nn.linear(
            input_559,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_559 = None
        chunk_111 = input_560.chunk(2, dim=-1)
        input_560 = None
        shift_27 = chunk_111[0]
        scale_27 = chunk_111[1]
        chunk_111 = None
        layer_norm_136 = torch.nn.functional.layer_norm(
            x_195, (768,), None, None, 1e-06
        )
        x_195 = None
        add_411 = 1 + scale_27
        scale_27 = None
        mul_394 = layer_norm_136 * add_411
        layer_norm_136 = add_411 = None
        x_196 = mul_394 + shift_27
        mul_394 = shift_27 = None
        x_197 = torch._C._nn.linear(
            x_196,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_196 = None
        sub_27 = x_197 - noise
        x_197 = None
        mul_395 = sub_27 * 0.02
        sub_27 = None
        x_198 = x_191 + mul_395
        x_191 = mul_395 = None
        ones_28 = torch.ones(1)
        mul_396 = ones_28 * 28
        ones_28 = None
        truediv_56 = mul_396 / 50
        mul_396 = None
        t_28 = truediv_56.to(device(type="cuda", index=0))
        truediv_56 = None
        mul_397 = t_28 * 1000
        t_28 = None
        x_199 = torch._C._nn.linear(
            x_198,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_30 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_398 = -9.210340371976184 * arange_30
        arange_30 = None
        truediv_57 = mul_398 / 128
        mul_398 = None
        exp_28 = torch.exp(truediv_57)
        truediv_57 = None
        freqs_28 = exp_28.to(device=device(type="cuda", index=0))
        exp_28 = None
        getitem_462 = mul_397[(slice(None, None, None), None)]
        mul_397 = None
        float_29 = getitem_462.float()
        getitem_462 = None
        getitem_463 = freqs_28[None]
        freqs_28 = None
        args_28 = float_29 * getitem_463
        float_29 = getitem_463 = None
        cos_52 = torch.cos(args_28)
        sin_52 = torch.sin(args_28)
        args_28 = None
        embedding_28 = torch.cat([cos_52, sin_52], dim=-1)
        cos_52 = sin_52 = None
        input_561 = torch._C._nn.linear(
            embedding_28,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_28 = None
        input_562 = torch.nn.functional.silu(input_561, inplace=False)
        input_561 = None
        input_563 = torch._C._nn.linear(
            input_562,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_562 = None
        c_28 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y_28 = input_563 + c_28
        input_563 = c_28 = None
        input_564 = torch.nn.functional.silu(y_28, inplace=False)
        input_565 = torch._C._nn.linear(
            input_564,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_564 = None
        chunk_112 = input_565.chunk(3, dim=-1)
        input_565 = None
        shift_mlp_84 = chunk_112[0]
        scale_mlp_84 = chunk_112[1]
        gate_mlp_84 = chunk_112[2]
        chunk_112 = None
        layer_norm_137 = torch.nn.functional.layer_norm(
            x_199,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_415 = 1 + scale_mlp_84
        scale_mlp_84 = None
        mul_400 = layer_norm_137 * add_415
        layer_norm_137 = add_415 = None
        h_84 = mul_400 + shift_mlp_84
        mul_400 = shift_mlp_84 = None
        input_566 = torch._C._nn.linear(
            h_84,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_84 = None
        input_567 = torch.nn.functional.silu(input_566, inplace=False)
        input_566 = None
        input_568 = torch._C._nn.linear(
            input_567,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_567 = None
        mul_401 = gate_mlp_84 * input_568
        gate_mlp_84 = input_568 = None
        x_200 = x_199 + mul_401
        x_199 = mul_401 = None
        input_569 = torch.nn.functional.silu(y_28, inplace=False)
        input_570 = torch._C._nn.linear(
            input_569,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_569 = None
        chunk_113 = input_570.chunk(3, dim=-1)
        input_570 = None
        shift_mlp_85 = chunk_113[0]
        scale_mlp_85 = chunk_113[1]
        gate_mlp_85 = chunk_113[2]
        chunk_113 = None
        layer_norm_138 = torch.nn.functional.layer_norm(
            x_200,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_418 = 1 + scale_mlp_85
        scale_mlp_85 = None
        mul_402 = layer_norm_138 * add_418
        layer_norm_138 = add_418 = None
        h_85 = mul_402 + shift_mlp_85
        mul_402 = shift_mlp_85 = None
        input_571 = torch._C._nn.linear(
            h_85,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_85 = None
        input_572 = torch.nn.functional.silu(input_571, inplace=False)
        input_571 = None
        input_573 = torch._C._nn.linear(
            input_572,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_572 = None
        mul_403 = gate_mlp_85 * input_573
        gate_mlp_85 = input_573 = None
        x_201 = x_200 + mul_403
        x_200 = mul_403 = None
        input_574 = torch.nn.functional.silu(y_28, inplace=False)
        input_575 = torch._C._nn.linear(
            input_574,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_574 = None
        chunk_114 = input_575.chunk(3, dim=-1)
        input_575 = None
        shift_mlp_86 = chunk_114[0]
        scale_mlp_86 = chunk_114[1]
        gate_mlp_86 = chunk_114[2]
        chunk_114 = None
        layer_norm_139 = torch.nn.functional.layer_norm(
            x_201,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_421 = 1 + scale_mlp_86
        scale_mlp_86 = None
        mul_404 = layer_norm_139 * add_421
        layer_norm_139 = add_421 = None
        h_86 = mul_404 + shift_mlp_86
        mul_404 = shift_mlp_86 = None
        input_576 = torch._C._nn.linear(
            h_86,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_86 = None
        input_577 = torch.nn.functional.silu(input_576, inplace=False)
        input_576 = None
        input_578 = torch._C._nn.linear(
            input_577,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_577 = None
        mul_405 = gate_mlp_86 * input_578
        gate_mlp_86 = input_578 = None
        x_202 = x_201 + mul_405
        x_201 = mul_405 = None
        input_579 = torch.nn.functional.silu(y_28, inplace=False)
        y_28 = None
        input_580 = torch._C._nn.linear(
            input_579,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_579 = None
        chunk_115 = input_580.chunk(2, dim=-1)
        input_580 = None
        shift_28 = chunk_115[0]
        scale_28 = chunk_115[1]
        chunk_115 = None
        layer_norm_140 = torch.nn.functional.layer_norm(
            x_202, (768,), None, None, 1e-06
        )
        x_202 = None
        add_424 = 1 + scale_28
        scale_28 = None
        mul_406 = layer_norm_140 * add_424
        layer_norm_140 = add_424 = None
        x_203 = mul_406 + shift_28
        mul_406 = shift_28 = None
        x_204 = torch._C._nn.linear(
            x_203,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_203 = None
        sub_28 = x_204 - noise
        x_204 = None
        mul_407 = sub_28 * 0.02
        sub_28 = None
        x_205 = x_198 + mul_407
        x_198 = mul_407 = None
        ones_29 = torch.ones(1)
        mul_408 = ones_29 * 29
        ones_29 = None
        truediv_58 = mul_408 / 50
        mul_408 = None
        t_29 = truediv_58.to(device(type="cuda", index=0))
        truediv_58 = None
        mul_409 = t_29 * 1000
        t_29 = None
        x_206 = torch._C._nn.linear(
            x_205,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_31 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_410 = -9.210340371976184 * arange_31
        arange_31 = None
        truediv_59 = mul_410 / 128
        mul_410 = None
        exp_29 = torch.exp(truediv_59)
        truediv_59 = None
        freqs_29 = exp_29.to(device=device(type="cuda", index=0))
        exp_29 = None
        getitem_475 = mul_409[(slice(None, None, None), None)]
        mul_409 = None
        float_30 = getitem_475.float()
        getitem_475 = None
        getitem_476 = freqs_29[None]
        freqs_29 = None
        args_29 = float_30 * getitem_476
        float_30 = getitem_476 = None
        cos_53 = torch.cos(args_29)
        sin_53 = torch.sin(args_29)
        args_29 = None
        embedding_29 = torch.cat([cos_53, sin_53], dim=-1)
        cos_53 = sin_53 = None
        input_581 = torch._C._nn.linear(
            embedding_29,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_29 = None
        input_582 = torch.nn.functional.silu(input_581, inplace=False)
        input_581 = None
        input_583 = torch._C._nn.linear(
            input_582,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_582 = None
        c_29 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y_29 = input_583 + c_29
        input_583 = c_29 = None
        input_584 = torch.nn.functional.silu(y_29, inplace=False)
        input_585 = torch._C._nn.linear(
            input_584,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_584 = None
        chunk_116 = input_585.chunk(3, dim=-1)
        input_585 = None
        shift_mlp_87 = chunk_116[0]
        scale_mlp_87 = chunk_116[1]
        gate_mlp_87 = chunk_116[2]
        chunk_116 = None
        layer_norm_141 = torch.nn.functional.layer_norm(
            x_206,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_428 = 1 + scale_mlp_87
        scale_mlp_87 = None
        mul_412 = layer_norm_141 * add_428
        layer_norm_141 = add_428 = None
        h_87 = mul_412 + shift_mlp_87
        mul_412 = shift_mlp_87 = None
        input_586 = torch._C._nn.linear(
            h_87,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_87 = None
        input_587 = torch.nn.functional.silu(input_586, inplace=False)
        input_586 = None
        input_588 = torch._C._nn.linear(
            input_587,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_587 = None
        mul_413 = gate_mlp_87 * input_588
        gate_mlp_87 = input_588 = None
        x_207 = x_206 + mul_413
        x_206 = mul_413 = None
        input_589 = torch.nn.functional.silu(y_29, inplace=False)
        input_590 = torch._C._nn.linear(
            input_589,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_589 = None
        chunk_117 = input_590.chunk(3, dim=-1)
        input_590 = None
        shift_mlp_88 = chunk_117[0]
        scale_mlp_88 = chunk_117[1]
        gate_mlp_88 = chunk_117[2]
        chunk_117 = None
        layer_norm_142 = torch.nn.functional.layer_norm(
            x_207,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_431 = 1 + scale_mlp_88
        scale_mlp_88 = None
        mul_414 = layer_norm_142 * add_431
        layer_norm_142 = add_431 = None
        h_88 = mul_414 + shift_mlp_88
        mul_414 = shift_mlp_88 = None
        input_591 = torch._C._nn.linear(
            h_88,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_88 = None
        input_592 = torch.nn.functional.silu(input_591, inplace=False)
        input_591 = None
        input_593 = torch._C._nn.linear(
            input_592,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_592 = None
        mul_415 = gate_mlp_88 * input_593
        gate_mlp_88 = input_593 = None
        x_208 = x_207 + mul_415
        x_207 = mul_415 = None
        input_594 = torch.nn.functional.silu(y_29, inplace=False)
        input_595 = torch._C._nn.linear(
            input_594,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_594 = None
        chunk_118 = input_595.chunk(3, dim=-1)
        input_595 = None
        shift_mlp_89 = chunk_118[0]
        scale_mlp_89 = chunk_118[1]
        gate_mlp_89 = chunk_118[2]
        chunk_118 = None
        layer_norm_143 = torch.nn.functional.layer_norm(
            x_208,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_434 = 1 + scale_mlp_89
        scale_mlp_89 = None
        mul_416 = layer_norm_143 * add_434
        layer_norm_143 = add_434 = None
        h_89 = mul_416 + shift_mlp_89
        mul_416 = shift_mlp_89 = None
        input_596 = torch._C._nn.linear(
            h_89,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_89 = None
        input_597 = torch.nn.functional.silu(input_596, inplace=False)
        input_596 = None
        input_598 = torch._C._nn.linear(
            input_597,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_597 = None
        mul_417 = gate_mlp_89 * input_598
        gate_mlp_89 = input_598 = None
        x_209 = x_208 + mul_417
        x_208 = mul_417 = None
        input_599 = torch.nn.functional.silu(y_29, inplace=False)
        y_29 = None
        input_600 = torch._C._nn.linear(
            input_599,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_599 = None
        chunk_119 = input_600.chunk(2, dim=-1)
        input_600 = None
        shift_29 = chunk_119[0]
        scale_29 = chunk_119[1]
        chunk_119 = None
        layer_norm_144 = torch.nn.functional.layer_norm(
            x_209, (768,), None, None, 1e-06
        )
        x_209 = None
        add_437 = 1 + scale_29
        scale_29 = None
        mul_418 = layer_norm_144 * add_437
        layer_norm_144 = add_437 = None
        x_210 = mul_418 + shift_29
        mul_418 = shift_29 = None
        x_211 = torch._C._nn.linear(
            x_210,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_210 = None
        sub_29 = x_211 - noise
        x_211 = None
        mul_419 = sub_29 * 0.02
        sub_29 = None
        x_212 = x_205 + mul_419
        x_205 = mul_419 = None
        ones_30 = torch.ones(1)
        mul_420 = ones_30 * 30
        ones_30 = None
        truediv_60 = mul_420 / 50
        mul_420 = None
        t_30 = truediv_60.to(device(type="cuda", index=0))
        truediv_60 = None
        mul_421 = t_30 * 1000
        t_30 = None
        x_213 = torch._C._nn.linear(
            x_212,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_32 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_422 = -9.210340371976184 * arange_32
        arange_32 = None
        truediv_61 = mul_422 / 128
        mul_422 = None
        exp_30 = torch.exp(truediv_61)
        truediv_61 = None
        freqs_30 = exp_30.to(device=device(type="cuda", index=0))
        exp_30 = None
        getitem_488 = mul_421[(slice(None, None, None), None)]
        mul_421 = None
        float_31 = getitem_488.float()
        getitem_488 = None
        getitem_489 = freqs_30[None]
        freqs_30 = None
        args_30 = float_31 * getitem_489
        float_31 = getitem_489 = None
        cos_54 = torch.cos(args_30)
        sin_54 = torch.sin(args_30)
        args_30 = None
        embedding_30 = torch.cat([cos_54, sin_54], dim=-1)
        cos_54 = sin_54 = None
        input_601 = torch._C._nn.linear(
            embedding_30,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_30 = None
        input_602 = torch.nn.functional.silu(input_601, inplace=False)
        input_601 = None
        input_603 = torch._C._nn.linear(
            input_602,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_602 = None
        c_30 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y_30 = input_603 + c_30
        input_603 = c_30 = None
        input_604 = torch.nn.functional.silu(y_30, inplace=False)
        input_605 = torch._C._nn.linear(
            input_604,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_604 = None
        chunk_120 = input_605.chunk(3, dim=-1)
        input_605 = None
        shift_mlp_90 = chunk_120[0]
        scale_mlp_90 = chunk_120[1]
        gate_mlp_90 = chunk_120[2]
        chunk_120 = None
        layer_norm_145 = torch.nn.functional.layer_norm(
            x_213,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_441 = 1 + scale_mlp_90
        scale_mlp_90 = None
        mul_424 = layer_norm_145 * add_441
        layer_norm_145 = add_441 = None
        h_90 = mul_424 + shift_mlp_90
        mul_424 = shift_mlp_90 = None
        input_606 = torch._C._nn.linear(
            h_90,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_90 = None
        input_607 = torch.nn.functional.silu(input_606, inplace=False)
        input_606 = None
        input_608 = torch._C._nn.linear(
            input_607,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_607 = None
        mul_425 = gate_mlp_90 * input_608
        gate_mlp_90 = input_608 = None
        x_214 = x_213 + mul_425
        x_213 = mul_425 = None
        input_609 = torch.nn.functional.silu(y_30, inplace=False)
        input_610 = torch._C._nn.linear(
            input_609,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_609 = None
        chunk_121 = input_610.chunk(3, dim=-1)
        input_610 = None
        shift_mlp_91 = chunk_121[0]
        scale_mlp_91 = chunk_121[1]
        gate_mlp_91 = chunk_121[2]
        chunk_121 = None
        layer_norm_146 = torch.nn.functional.layer_norm(
            x_214,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_444 = 1 + scale_mlp_91
        scale_mlp_91 = None
        mul_426 = layer_norm_146 * add_444
        layer_norm_146 = add_444 = None
        h_91 = mul_426 + shift_mlp_91
        mul_426 = shift_mlp_91 = None
        input_611 = torch._C._nn.linear(
            h_91,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_91 = None
        input_612 = torch.nn.functional.silu(input_611, inplace=False)
        input_611 = None
        input_613 = torch._C._nn.linear(
            input_612,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_612 = None
        mul_427 = gate_mlp_91 * input_613
        gate_mlp_91 = input_613 = None
        x_215 = x_214 + mul_427
        x_214 = mul_427 = None
        input_614 = torch.nn.functional.silu(y_30, inplace=False)
        input_615 = torch._C._nn.linear(
            input_614,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_614 = None
        chunk_122 = input_615.chunk(3, dim=-1)
        input_615 = None
        shift_mlp_92 = chunk_122[0]
        scale_mlp_92 = chunk_122[1]
        gate_mlp_92 = chunk_122[2]
        chunk_122 = None
        layer_norm_147 = torch.nn.functional.layer_norm(
            x_215,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_447 = 1 + scale_mlp_92
        scale_mlp_92 = None
        mul_428 = layer_norm_147 * add_447
        layer_norm_147 = add_447 = None
        h_92 = mul_428 + shift_mlp_92
        mul_428 = shift_mlp_92 = None
        input_616 = torch._C._nn.linear(
            h_92,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_92 = None
        input_617 = torch.nn.functional.silu(input_616, inplace=False)
        input_616 = None
        input_618 = torch._C._nn.linear(
            input_617,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_617 = None
        mul_429 = gate_mlp_92 * input_618
        gate_mlp_92 = input_618 = None
        x_216 = x_215 + mul_429
        x_215 = mul_429 = None
        input_619 = torch.nn.functional.silu(y_30, inplace=False)
        y_30 = None
        input_620 = torch._C._nn.linear(
            input_619,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_619 = None
        chunk_123 = input_620.chunk(2, dim=-1)
        input_620 = None
        shift_30 = chunk_123[0]
        scale_30 = chunk_123[1]
        chunk_123 = None
        layer_norm_148 = torch.nn.functional.layer_norm(
            x_216, (768,), None, None, 1e-06
        )
        x_216 = None
        add_450 = 1 + scale_30
        scale_30 = None
        mul_430 = layer_norm_148 * add_450
        layer_norm_148 = add_450 = None
        x_217 = mul_430 + shift_30
        mul_430 = shift_30 = None
        x_218 = torch._C._nn.linear(
            x_217,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_217 = None
        sub_30 = x_218 - noise
        x_218 = None
        mul_431 = sub_30 * 0.02
        sub_30 = None
        x_219 = x_212 + mul_431
        x_212 = mul_431 = None
        ones_31 = torch.ones(1)
        mul_432 = ones_31 * 31
        ones_31 = None
        truediv_62 = mul_432 / 50
        mul_432 = None
        t_31 = truediv_62.to(device(type="cuda", index=0))
        truediv_62 = None
        mul_433 = t_31 * 1000
        t_31 = None
        x_220 = torch._C._nn.linear(
            x_219,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_33 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_434 = -9.210340371976184 * arange_33
        arange_33 = None
        truediv_63 = mul_434 / 128
        mul_434 = None
        exp_31 = torch.exp(truediv_63)
        truediv_63 = None
        freqs_31 = exp_31.to(device=device(type="cuda", index=0))
        exp_31 = None
        getitem_501 = mul_433[(slice(None, None, None), None)]
        mul_433 = None
        float_32 = getitem_501.float()
        getitem_501 = None
        getitem_502 = freqs_31[None]
        freqs_31 = None
        args_31 = float_32 * getitem_502
        float_32 = getitem_502 = None
        cos_55 = torch.cos(args_31)
        sin_55 = torch.sin(args_31)
        args_31 = None
        embedding_31 = torch.cat([cos_55, sin_55], dim=-1)
        cos_55 = sin_55 = None
        input_621 = torch._C._nn.linear(
            embedding_31,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_31 = None
        input_622 = torch.nn.functional.silu(input_621, inplace=False)
        input_621 = None
        input_623 = torch._C._nn.linear(
            input_622,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_622 = None
        c_31 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y_31 = input_623 + c_31
        input_623 = c_31 = None
        input_624 = torch.nn.functional.silu(y_31, inplace=False)
        input_625 = torch._C._nn.linear(
            input_624,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_624 = None
        chunk_124 = input_625.chunk(3, dim=-1)
        input_625 = None
        shift_mlp_93 = chunk_124[0]
        scale_mlp_93 = chunk_124[1]
        gate_mlp_93 = chunk_124[2]
        chunk_124 = None
        layer_norm_149 = torch.nn.functional.layer_norm(
            x_220,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_454 = 1 + scale_mlp_93
        scale_mlp_93 = None
        mul_436 = layer_norm_149 * add_454
        layer_norm_149 = add_454 = None
        h_93 = mul_436 + shift_mlp_93
        mul_436 = shift_mlp_93 = None
        input_626 = torch._C._nn.linear(
            h_93,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_93 = None
        input_627 = torch.nn.functional.silu(input_626, inplace=False)
        input_626 = None
        input_628 = torch._C._nn.linear(
            input_627,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_627 = None
        mul_437 = gate_mlp_93 * input_628
        gate_mlp_93 = input_628 = None
        x_221 = x_220 + mul_437
        x_220 = mul_437 = None
        input_629 = torch.nn.functional.silu(y_31, inplace=False)
        input_630 = torch._C._nn.linear(
            input_629,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_629 = None
        chunk_125 = input_630.chunk(3, dim=-1)
        input_630 = None
        shift_mlp_94 = chunk_125[0]
        scale_mlp_94 = chunk_125[1]
        gate_mlp_94 = chunk_125[2]
        chunk_125 = None
        layer_norm_150 = torch.nn.functional.layer_norm(
            x_221,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_457 = 1 + scale_mlp_94
        scale_mlp_94 = None
        mul_438 = layer_norm_150 * add_457
        layer_norm_150 = add_457 = None
        h_94 = mul_438 + shift_mlp_94
        mul_438 = shift_mlp_94 = None
        input_631 = torch._C._nn.linear(
            h_94,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_94 = None
        input_632 = torch.nn.functional.silu(input_631, inplace=False)
        input_631 = None
        input_633 = torch._C._nn.linear(
            input_632,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_632 = None
        mul_439 = gate_mlp_94 * input_633
        gate_mlp_94 = input_633 = None
        x_222 = x_221 + mul_439
        x_221 = mul_439 = None
        input_634 = torch.nn.functional.silu(y_31, inplace=False)
        input_635 = torch._C._nn.linear(
            input_634,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_634 = None
        chunk_126 = input_635.chunk(3, dim=-1)
        input_635 = None
        shift_mlp_95 = chunk_126[0]
        scale_mlp_95 = chunk_126[1]
        gate_mlp_95 = chunk_126[2]
        chunk_126 = None
        layer_norm_151 = torch.nn.functional.layer_norm(
            x_222,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_460 = 1 + scale_mlp_95
        scale_mlp_95 = None
        mul_440 = layer_norm_151 * add_460
        layer_norm_151 = add_460 = None
        h_95 = mul_440 + shift_mlp_95
        mul_440 = shift_mlp_95 = None
        input_636 = torch._C._nn.linear(
            h_95,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_95 = None
        input_637 = torch.nn.functional.silu(input_636, inplace=False)
        input_636 = None
        input_638 = torch._C._nn.linear(
            input_637,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_637 = None
        mul_441 = gate_mlp_95 * input_638
        gate_mlp_95 = input_638 = None
        x_223 = x_222 + mul_441
        x_222 = mul_441 = None
        input_639 = torch.nn.functional.silu(y_31, inplace=False)
        y_31 = None
        input_640 = torch._C._nn.linear(
            input_639,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_639 = None
        chunk_127 = input_640.chunk(2, dim=-1)
        input_640 = None
        shift_31 = chunk_127[0]
        scale_31 = chunk_127[1]
        chunk_127 = None
        layer_norm_152 = torch.nn.functional.layer_norm(
            x_223, (768,), None, None, 1e-06
        )
        x_223 = None
        add_463 = 1 + scale_31
        scale_31 = None
        mul_442 = layer_norm_152 * add_463
        layer_norm_152 = add_463 = None
        x_224 = mul_442 + shift_31
        mul_442 = shift_31 = None
        x_225 = torch._C._nn.linear(
            x_224,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_224 = None
        sub_31 = x_225 - noise
        x_225 = None
        mul_443 = sub_31 * 0.02
        sub_31 = None
        x_226 = x_219 + mul_443
        x_219 = mul_443 = None
        ones_32 = torch.ones(1)
        mul_444 = ones_32 * 32
        ones_32 = None
        truediv_64 = mul_444 / 50
        mul_444 = None
        t_32 = truediv_64.to(device(type="cuda", index=0))
        truediv_64 = None
        mul_445 = t_32 * 1000
        t_32 = None
        x_227 = torch._C._nn.linear(
            x_226,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_34 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_446 = -9.210340371976184 * arange_34
        arange_34 = None
        truediv_65 = mul_446 / 128
        mul_446 = None
        exp_32 = torch.exp(truediv_65)
        truediv_65 = None
        freqs_32 = exp_32.to(device=device(type="cuda", index=0))
        exp_32 = None
        getitem_514 = mul_445[(slice(None, None, None), None)]
        mul_445 = None
        float_33 = getitem_514.float()
        getitem_514 = None
        getitem_515 = freqs_32[None]
        freqs_32 = None
        args_32 = float_33 * getitem_515
        float_33 = getitem_515 = None
        cos_56 = torch.cos(args_32)
        sin_56 = torch.sin(args_32)
        args_32 = None
        embedding_32 = torch.cat([cos_56, sin_56], dim=-1)
        cos_56 = sin_56 = None
        input_641 = torch._C._nn.linear(
            embedding_32,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_32 = None
        input_642 = torch.nn.functional.silu(input_641, inplace=False)
        input_641 = None
        input_643 = torch._C._nn.linear(
            input_642,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_642 = None
        c_32 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y_32 = input_643 + c_32
        input_643 = c_32 = None
        input_644 = torch.nn.functional.silu(y_32, inplace=False)
        input_645 = torch._C._nn.linear(
            input_644,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_644 = None
        chunk_128 = input_645.chunk(3, dim=-1)
        input_645 = None
        shift_mlp_96 = chunk_128[0]
        scale_mlp_96 = chunk_128[1]
        gate_mlp_96 = chunk_128[2]
        chunk_128 = None
        layer_norm_153 = torch.nn.functional.layer_norm(
            x_227,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_467 = 1 + scale_mlp_96
        scale_mlp_96 = None
        mul_448 = layer_norm_153 * add_467
        layer_norm_153 = add_467 = None
        h_96 = mul_448 + shift_mlp_96
        mul_448 = shift_mlp_96 = None
        input_646 = torch._C._nn.linear(
            h_96,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_96 = None
        input_647 = torch.nn.functional.silu(input_646, inplace=False)
        input_646 = None
        input_648 = torch._C._nn.linear(
            input_647,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_647 = None
        mul_449 = gate_mlp_96 * input_648
        gate_mlp_96 = input_648 = None
        x_228 = x_227 + mul_449
        x_227 = mul_449 = None
        input_649 = torch.nn.functional.silu(y_32, inplace=False)
        input_650 = torch._C._nn.linear(
            input_649,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_649 = None
        chunk_129 = input_650.chunk(3, dim=-1)
        input_650 = None
        shift_mlp_97 = chunk_129[0]
        scale_mlp_97 = chunk_129[1]
        gate_mlp_97 = chunk_129[2]
        chunk_129 = None
        layer_norm_154 = torch.nn.functional.layer_norm(
            x_228,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_470 = 1 + scale_mlp_97
        scale_mlp_97 = None
        mul_450 = layer_norm_154 * add_470
        layer_norm_154 = add_470 = None
        h_97 = mul_450 + shift_mlp_97
        mul_450 = shift_mlp_97 = None
        input_651 = torch._C._nn.linear(
            h_97,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_97 = None
        input_652 = torch.nn.functional.silu(input_651, inplace=False)
        input_651 = None
        input_653 = torch._C._nn.linear(
            input_652,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_652 = None
        mul_451 = gate_mlp_97 * input_653
        gate_mlp_97 = input_653 = None
        x_229 = x_228 + mul_451
        x_228 = mul_451 = None
        input_654 = torch.nn.functional.silu(y_32, inplace=False)
        input_655 = torch._C._nn.linear(
            input_654,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_654 = None
        chunk_130 = input_655.chunk(3, dim=-1)
        input_655 = None
        shift_mlp_98 = chunk_130[0]
        scale_mlp_98 = chunk_130[1]
        gate_mlp_98 = chunk_130[2]
        chunk_130 = None
        layer_norm_155 = torch.nn.functional.layer_norm(
            x_229,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_473 = 1 + scale_mlp_98
        scale_mlp_98 = None
        mul_452 = layer_norm_155 * add_473
        layer_norm_155 = add_473 = None
        h_98 = mul_452 + shift_mlp_98
        mul_452 = shift_mlp_98 = None
        input_656 = torch._C._nn.linear(
            h_98,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_98 = None
        input_657 = torch.nn.functional.silu(input_656, inplace=False)
        input_656 = None
        input_658 = torch._C._nn.linear(
            input_657,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_657 = None
        mul_453 = gate_mlp_98 * input_658
        gate_mlp_98 = input_658 = None
        x_230 = x_229 + mul_453
        x_229 = mul_453 = None
        input_659 = torch.nn.functional.silu(y_32, inplace=False)
        y_32 = None
        input_660 = torch._C._nn.linear(
            input_659,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_659 = None
        chunk_131 = input_660.chunk(2, dim=-1)
        input_660 = None
        shift_32 = chunk_131[0]
        scale_32 = chunk_131[1]
        chunk_131 = None
        layer_norm_156 = torch.nn.functional.layer_norm(
            x_230, (768,), None, None, 1e-06
        )
        x_230 = None
        add_476 = 1 + scale_32
        scale_32 = None
        mul_454 = layer_norm_156 * add_476
        layer_norm_156 = add_476 = None
        x_231 = mul_454 + shift_32
        mul_454 = shift_32 = None
        x_232 = torch._C._nn.linear(
            x_231,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_231 = None
        sub_32 = x_232 - noise
        x_232 = None
        mul_455 = sub_32 * 0.02
        sub_32 = None
        x_233 = x_226 + mul_455
        x_226 = mul_455 = None
        ones_33 = torch.ones(1)
        mul_456 = ones_33 * 33
        ones_33 = None
        truediv_66 = mul_456 / 50
        mul_456 = None
        t_33 = truediv_66.to(device(type="cuda", index=0))
        truediv_66 = None
        mul_457 = t_33 * 1000
        t_33 = None
        x_234 = torch._C._nn.linear(
            x_233,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_35 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_458 = -9.210340371976184 * arange_35
        arange_35 = None
        truediv_67 = mul_458 / 128
        mul_458 = None
        exp_33 = torch.exp(truediv_67)
        truediv_67 = None
        freqs_33 = exp_33.to(device=device(type="cuda", index=0))
        exp_33 = None
        getitem_527 = mul_457[(slice(None, None, None), None)]
        mul_457 = None
        float_34 = getitem_527.float()
        getitem_527 = None
        getitem_528 = freqs_33[None]
        freqs_33 = None
        args_33 = float_34 * getitem_528
        float_34 = getitem_528 = None
        cos_57 = torch.cos(args_33)
        sin_57 = torch.sin(args_33)
        args_33 = None
        embedding_33 = torch.cat([cos_57, sin_57], dim=-1)
        cos_57 = sin_57 = None
        input_661 = torch._C._nn.linear(
            embedding_33,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_33 = None
        input_662 = torch.nn.functional.silu(input_661, inplace=False)
        input_661 = None
        input_663 = torch._C._nn.linear(
            input_662,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_662 = None
        c_33 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y_33 = input_663 + c_33
        input_663 = c_33 = None
        input_664 = torch.nn.functional.silu(y_33, inplace=False)
        input_665 = torch._C._nn.linear(
            input_664,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_664 = None
        chunk_132 = input_665.chunk(3, dim=-1)
        input_665 = None
        shift_mlp_99 = chunk_132[0]
        scale_mlp_99 = chunk_132[1]
        gate_mlp_99 = chunk_132[2]
        chunk_132 = None
        layer_norm_157 = torch.nn.functional.layer_norm(
            x_234,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_480 = 1 + scale_mlp_99
        scale_mlp_99 = None
        mul_460 = layer_norm_157 * add_480
        layer_norm_157 = add_480 = None
        h_99 = mul_460 + shift_mlp_99
        mul_460 = shift_mlp_99 = None
        input_666 = torch._C._nn.linear(
            h_99,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_99 = None
        input_667 = torch.nn.functional.silu(input_666, inplace=False)
        input_666 = None
        input_668 = torch._C._nn.linear(
            input_667,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_667 = None
        mul_461 = gate_mlp_99 * input_668
        gate_mlp_99 = input_668 = None
        x_235 = x_234 + mul_461
        x_234 = mul_461 = None
        input_669 = torch.nn.functional.silu(y_33, inplace=False)
        input_670 = torch._C._nn.linear(
            input_669,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_669 = None
        chunk_133 = input_670.chunk(3, dim=-1)
        input_670 = None
        shift_mlp_100 = chunk_133[0]
        scale_mlp_100 = chunk_133[1]
        gate_mlp_100 = chunk_133[2]
        chunk_133 = None
        layer_norm_158 = torch.nn.functional.layer_norm(
            x_235,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_483 = 1 + scale_mlp_100
        scale_mlp_100 = None
        mul_462 = layer_norm_158 * add_483
        layer_norm_158 = add_483 = None
        h_100 = mul_462 + shift_mlp_100
        mul_462 = shift_mlp_100 = None
        input_671 = torch._C._nn.linear(
            h_100,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_100 = None
        input_672 = torch.nn.functional.silu(input_671, inplace=False)
        input_671 = None
        input_673 = torch._C._nn.linear(
            input_672,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_672 = None
        mul_463 = gate_mlp_100 * input_673
        gate_mlp_100 = input_673 = None
        x_236 = x_235 + mul_463
        x_235 = mul_463 = None
        input_674 = torch.nn.functional.silu(y_33, inplace=False)
        input_675 = torch._C._nn.linear(
            input_674,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_674 = None
        chunk_134 = input_675.chunk(3, dim=-1)
        input_675 = None
        shift_mlp_101 = chunk_134[0]
        scale_mlp_101 = chunk_134[1]
        gate_mlp_101 = chunk_134[2]
        chunk_134 = None
        layer_norm_159 = torch.nn.functional.layer_norm(
            x_236,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_486 = 1 + scale_mlp_101
        scale_mlp_101 = None
        mul_464 = layer_norm_159 * add_486
        layer_norm_159 = add_486 = None
        h_101 = mul_464 + shift_mlp_101
        mul_464 = shift_mlp_101 = None
        input_676 = torch._C._nn.linear(
            h_101,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_101 = None
        input_677 = torch.nn.functional.silu(input_676, inplace=False)
        input_676 = None
        input_678 = torch._C._nn.linear(
            input_677,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_677 = None
        mul_465 = gate_mlp_101 * input_678
        gate_mlp_101 = input_678 = None
        x_237 = x_236 + mul_465
        x_236 = mul_465 = None
        input_679 = torch.nn.functional.silu(y_33, inplace=False)
        y_33 = None
        input_680 = torch._C._nn.linear(
            input_679,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_679 = None
        chunk_135 = input_680.chunk(2, dim=-1)
        input_680 = None
        shift_33 = chunk_135[0]
        scale_33 = chunk_135[1]
        chunk_135 = None
        layer_norm_160 = torch.nn.functional.layer_norm(
            x_237, (768,), None, None, 1e-06
        )
        x_237 = None
        add_489 = 1 + scale_33
        scale_33 = None
        mul_466 = layer_norm_160 * add_489
        layer_norm_160 = add_489 = None
        x_238 = mul_466 + shift_33
        mul_466 = shift_33 = None
        x_239 = torch._C._nn.linear(
            x_238,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_238 = None
        sub_33 = x_239 - noise
        x_239 = None
        mul_467 = sub_33 * 0.02
        sub_33 = None
        x_240 = x_233 + mul_467
        x_233 = mul_467 = None
        ones_34 = torch.ones(1)
        mul_468 = ones_34 * 34
        ones_34 = None
        truediv_68 = mul_468 / 50
        mul_468 = None
        t_34 = truediv_68.to(device(type="cuda", index=0))
        truediv_68 = None
        mul_469 = t_34 * 1000
        t_34 = None
        x_241 = torch._C._nn.linear(
            x_240,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_36 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_470 = -9.210340371976184 * arange_36
        arange_36 = None
        truediv_69 = mul_470 / 128
        mul_470 = None
        exp_34 = torch.exp(truediv_69)
        truediv_69 = None
        freqs_34 = exp_34.to(device=device(type="cuda", index=0))
        exp_34 = None
        getitem_540 = mul_469[(slice(None, None, None), None)]
        mul_469 = None
        float_35 = getitem_540.float()
        getitem_540 = None
        getitem_541 = freqs_34[None]
        freqs_34 = None
        args_34 = float_35 * getitem_541
        float_35 = getitem_541 = None
        cos_58 = torch.cos(args_34)
        sin_58 = torch.sin(args_34)
        args_34 = None
        embedding_34 = torch.cat([cos_58, sin_58], dim=-1)
        cos_58 = sin_58 = None
        input_681 = torch._C._nn.linear(
            embedding_34,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_34 = None
        input_682 = torch.nn.functional.silu(input_681, inplace=False)
        input_681 = None
        input_683 = torch._C._nn.linear(
            input_682,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_682 = None
        c_34 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y_34 = input_683 + c_34
        input_683 = c_34 = None
        input_684 = torch.nn.functional.silu(y_34, inplace=False)
        input_685 = torch._C._nn.linear(
            input_684,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_684 = None
        chunk_136 = input_685.chunk(3, dim=-1)
        input_685 = None
        shift_mlp_102 = chunk_136[0]
        scale_mlp_102 = chunk_136[1]
        gate_mlp_102 = chunk_136[2]
        chunk_136 = None
        layer_norm_161 = torch.nn.functional.layer_norm(
            x_241,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_493 = 1 + scale_mlp_102
        scale_mlp_102 = None
        mul_472 = layer_norm_161 * add_493
        layer_norm_161 = add_493 = None
        h_102 = mul_472 + shift_mlp_102
        mul_472 = shift_mlp_102 = None
        input_686 = torch._C._nn.linear(
            h_102,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_102 = None
        input_687 = torch.nn.functional.silu(input_686, inplace=False)
        input_686 = None
        input_688 = torch._C._nn.linear(
            input_687,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_687 = None
        mul_473 = gate_mlp_102 * input_688
        gate_mlp_102 = input_688 = None
        x_242 = x_241 + mul_473
        x_241 = mul_473 = None
        input_689 = torch.nn.functional.silu(y_34, inplace=False)
        input_690 = torch._C._nn.linear(
            input_689,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_689 = None
        chunk_137 = input_690.chunk(3, dim=-1)
        input_690 = None
        shift_mlp_103 = chunk_137[0]
        scale_mlp_103 = chunk_137[1]
        gate_mlp_103 = chunk_137[2]
        chunk_137 = None
        layer_norm_162 = torch.nn.functional.layer_norm(
            x_242,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_496 = 1 + scale_mlp_103
        scale_mlp_103 = None
        mul_474 = layer_norm_162 * add_496
        layer_norm_162 = add_496 = None
        h_103 = mul_474 + shift_mlp_103
        mul_474 = shift_mlp_103 = None
        input_691 = torch._C._nn.linear(
            h_103,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_103 = None
        input_692 = torch.nn.functional.silu(input_691, inplace=False)
        input_691 = None
        input_693 = torch._C._nn.linear(
            input_692,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_692 = None
        mul_475 = gate_mlp_103 * input_693
        gate_mlp_103 = input_693 = None
        x_243 = x_242 + mul_475
        x_242 = mul_475 = None
        input_694 = torch.nn.functional.silu(y_34, inplace=False)
        input_695 = torch._C._nn.linear(
            input_694,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_694 = None
        chunk_138 = input_695.chunk(3, dim=-1)
        input_695 = None
        shift_mlp_104 = chunk_138[0]
        scale_mlp_104 = chunk_138[1]
        gate_mlp_104 = chunk_138[2]
        chunk_138 = None
        layer_norm_163 = torch.nn.functional.layer_norm(
            x_243,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_499 = 1 + scale_mlp_104
        scale_mlp_104 = None
        mul_476 = layer_norm_163 * add_499
        layer_norm_163 = add_499 = None
        h_104 = mul_476 + shift_mlp_104
        mul_476 = shift_mlp_104 = None
        input_696 = torch._C._nn.linear(
            h_104,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_104 = None
        input_697 = torch.nn.functional.silu(input_696, inplace=False)
        input_696 = None
        input_698 = torch._C._nn.linear(
            input_697,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_697 = None
        mul_477 = gate_mlp_104 * input_698
        gate_mlp_104 = input_698 = None
        x_244 = x_243 + mul_477
        x_243 = mul_477 = None
        input_699 = torch.nn.functional.silu(y_34, inplace=False)
        y_34 = None
        input_700 = torch._C._nn.linear(
            input_699,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_699 = None
        chunk_139 = input_700.chunk(2, dim=-1)
        input_700 = None
        shift_34 = chunk_139[0]
        scale_34 = chunk_139[1]
        chunk_139 = None
        layer_norm_164 = torch.nn.functional.layer_norm(
            x_244, (768,), None, None, 1e-06
        )
        x_244 = None
        add_502 = 1 + scale_34
        scale_34 = None
        mul_478 = layer_norm_164 * add_502
        layer_norm_164 = add_502 = None
        x_245 = mul_478 + shift_34
        mul_478 = shift_34 = None
        x_246 = torch._C._nn.linear(
            x_245,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_245 = None
        sub_34 = x_246 - noise
        x_246 = None
        mul_479 = sub_34 * 0.02
        sub_34 = None
        x_247 = x_240 + mul_479
        x_240 = mul_479 = None
        ones_35 = torch.ones(1)
        mul_480 = ones_35 * 35
        ones_35 = None
        truediv_70 = mul_480 / 50
        mul_480 = None
        t_35 = truediv_70.to(device(type="cuda", index=0))
        truediv_70 = None
        mul_481 = t_35 * 1000
        t_35 = None
        x_248 = torch._C._nn.linear(
            x_247,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_37 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_482 = -9.210340371976184 * arange_37
        arange_37 = None
        truediv_71 = mul_482 / 128
        mul_482 = None
        exp_35 = torch.exp(truediv_71)
        truediv_71 = None
        freqs_35 = exp_35.to(device=device(type="cuda", index=0))
        exp_35 = None
        getitem_553 = mul_481[(slice(None, None, None), None)]
        mul_481 = None
        float_36 = getitem_553.float()
        getitem_553 = None
        getitem_554 = freqs_35[None]
        freqs_35 = None
        args_35 = float_36 * getitem_554
        float_36 = getitem_554 = None
        cos_59 = torch.cos(args_35)
        sin_59 = torch.sin(args_35)
        args_35 = None
        embedding_35 = torch.cat([cos_59, sin_59], dim=-1)
        cos_59 = sin_59 = None
        input_701 = torch._C._nn.linear(
            embedding_35,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_35 = None
        input_702 = torch.nn.functional.silu(input_701, inplace=False)
        input_701 = None
        input_703 = torch._C._nn.linear(
            input_702,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_702 = None
        c_35 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y_35 = input_703 + c_35
        input_703 = c_35 = None
        input_704 = torch.nn.functional.silu(y_35, inplace=False)
        input_705 = torch._C._nn.linear(
            input_704,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_704 = None
        chunk_140 = input_705.chunk(3, dim=-1)
        input_705 = None
        shift_mlp_105 = chunk_140[0]
        scale_mlp_105 = chunk_140[1]
        gate_mlp_105 = chunk_140[2]
        chunk_140 = None
        layer_norm_165 = torch.nn.functional.layer_norm(
            x_248,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_506 = 1 + scale_mlp_105
        scale_mlp_105 = None
        mul_484 = layer_norm_165 * add_506
        layer_norm_165 = add_506 = None
        h_105 = mul_484 + shift_mlp_105
        mul_484 = shift_mlp_105 = None
        input_706 = torch._C._nn.linear(
            h_105,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_105 = None
        input_707 = torch.nn.functional.silu(input_706, inplace=False)
        input_706 = None
        input_708 = torch._C._nn.linear(
            input_707,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_707 = None
        mul_485 = gate_mlp_105 * input_708
        gate_mlp_105 = input_708 = None
        x_249 = x_248 + mul_485
        x_248 = mul_485 = None
        input_709 = torch.nn.functional.silu(y_35, inplace=False)
        input_710 = torch._C._nn.linear(
            input_709,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_709 = None
        chunk_141 = input_710.chunk(3, dim=-1)
        input_710 = None
        shift_mlp_106 = chunk_141[0]
        scale_mlp_106 = chunk_141[1]
        gate_mlp_106 = chunk_141[2]
        chunk_141 = None
        layer_norm_166 = torch.nn.functional.layer_norm(
            x_249,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_509 = 1 + scale_mlp_106
        scale_mlp_106 = None
        mul_486 = layer_norm_166 * add_509
        layer_norm_166 = add_509 = None
        h_106 = mul_486 + shift_mlp_106
        mul_486 = shift_mlp_106 = None
        input_711 = torch._C._nn.linear(
            h_106,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_106 = None
        input_712 = torch.nn.functional.silu(input_711, inplace=False)
        input_711 = None
        input_713 = torch._C._nn.linear(
            input_712,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_712 = None
        mul_487 = gate_mlp_106 * input_713
        gate_mlp_106 = input_713 = None
        x_250 = x_249 + mul_487
        x_249 = mul_487 = None
        input_714 = torch.nn.functional.silu(y_35, inplace=False)
        input_715 = torch._C._nn.linear(
            input_714,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_714 = None
        chunk_142 = input_715.chunk(3, dim=-1)
        input_715 = None
        shift_mlp_107 = chunk_142[0]
        scale_mlp_107 = chunk_142[1]
        gate_mlp_107 = chunk_142[2]
        chunk_142 = None
        layer_norm_167 = torch.nn.functional.layer_norm(
            x_250,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_512 = 1 + scale_mlp_107
        scale_mlp_107 = None
        mul_488 = layer_norm_167 * add_512
        layer_norm_167 = add_512 = None
        h_107 = mul_488 + shift_mlp_107
        mul_488 = shift_mlp_107 = None
        input_716 = torch._C._nn.linear(
            h_107,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_107 = None
        input_717 = torch.nn.functional.silu(input_716, inplace=False)
        input_716 = None
        input_718 = torch._C._nn.linear(
            input_717,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_717 = None
        mul_489 = gate_mlp_107 * input_718
        gate_mlp_107 = input_718 = None
        x_251 = x_250 + mul_489
        x_250 = mul_489 = None
        input_719 = torch.nn.functional.silu(y_35, inplace=False)
        y_35 = None
        input_720 = torch._C._nn.linear(
            input_719,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_719 = None
        chunk_143 = input_720.chunk(2, dim=-1)
        input_720 = None
        shift_35 = chunk_143[0]
        scale_35 = chunk_143[1]
        chunk_143 = None
        layer_norm_168 = torch.nn.functional.layer_norm(
            x_251, (768,), None, None, 1e-06
        )
        x_251 = None
        add_515 = 1 + scale_35
        scale_35 = None
        mul_490 = layer_norm_168 * add_515
        layer_norm_168 = add_515 = None
        x_252 = mul_490 + shift_35
        mul_490 = shift_35 = None
        x_253 = torch._C._nn.linear(
            x_252,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_252 = None
        sub_35 = x_253 - noise
        x_253 = None
        mul_491 = sub_35 * 0.02
        sub_35 = None
        x_254 = x_247 + mul_491
        x_247 = mul_491 = None
        ones_36 = torch.ones(1)
        mul_492 = ones_36 * 36
        ones_36 = None
        truediv_72 = mul_492 / 50
        mul_492 = None
        t_36 = truediv_72.to(device(type="cuda", index=0))
        truediv_72 = None
        mul_493 = t_36 * 1000
        t_36 = None
        x_255 = torch._C._nn.linear(
            x_254,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_38 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_494 = -9.210340371976184 * arange_38
        arange_38 = None
        truediv_73 = mul_494 / 128
        mul_494 = None
        exp_36 = torch.exp(truediv_73)
        truediv_73 = None
        freqs_36 = exp_36.to(device=device(type="cuda", index=0))
        exp_36 = None
        getitem_566 = mul_493[(slice(None, None, None), None)]
        mul_493 = None
        float_37 = getitem_566.float()
        getitem_566 = None
        getitem_567 = freqs_36[None]
        freqs_36 = None
        args_36 = float_37 * getitem_567
        float_37 = getitem_567 = None
        cos_60 = torch.cos(args_36)
        sin_60 = torch.sin(args_36)
        args_36 = None
        embedding_36 = torch.cat([cos_60, sin_60], dim=-1)
        cos_60 = sin_60 = None
        input_721 = torch._C._nn.linear(
            embedding_36,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_36 = None
        input_722 = torch.nn.functional.silu(input_721, inplace=False)
        input_721 = None
        input_723 = torch._C._nn.linear(
            input_722,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_722 = None
        c_36 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y_36 = input_723 + c_36
        input_723 = c_36 = None
        input_724 = torch.nn.functional.silu(y_36, inplace=False)
        input_725 = torch._C._nn.linear(
            input_724,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_724 = None
        chunk_144 = input_725.chunk(3, dim=-1)
        input_725 = None
        shift_mlp_108 = chunk_144[0]
        scale_mlp_108 = chunk_144[1]
        gate_mlp_108 = chunk_144[2]
        chunk_144 = None
        layer_norm_169 = torch.nn.functional.layer_norm(
            x_255,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_519 = 1 + scale_mlp_108
        scale_mlp_108 = None
        mul_496 = layer_norm_169 * add_519
        layer_norm_169 = add_519 = None
        h_108 = mul_496 + shift_mlp_108
        mul_496 = shift_mlp_108 = None
        input_726 = torch._C._nn.linear(
            h_108,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_108 = None
        input_727 = torch.nn.functional.silu(input_726, inplace=False)
        input_726 = None
        input_728 = torch._C._nn.linear(
            input_727,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_727 = None
        mul_497 = gate_mlp_108 * input_728
        gate_mlp_108 = input_728 = None
        x_256 = x_255 + mul_497
        x_255 = mul_497 = None
        input_729 = torch.nn.functional.silu(y_36, inplace=False)
        input_730 = torch._C._nn.linear(
            input_729,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_729 = None
        chunk_145 = input_730.chunk(3, dim=-1)
        input_730 = None
        shift_mlp_109 = chunk_145[0]
        scale_mlp_109 = chunk_145[1]
        gate_mlp_109 = chunk_145[2]
        chunk_145 = None
        layer_norm_170 = torch.nn.functional.layer_norm(
            x_256,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_522 = 1 + scale_mlp_109
        scale_mlp_109 = None
        mul_498 = layer_norm_170 * add_522
        layer_norm_170 = add_522 = None
        h_109 = mul_498 + shift_mlp_109
        mul_498 = shift_mlp_109 = None
        input_731 = torch._C._nn.linear(
            h_109,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_109 = None
        input_732 = torch.nn.functional.silu(input_731, inplace=False)
        input_731 = None
        input_733 = torch._C._nn.linear(
            input_732,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_732 = None
        mul_499 = gate_mlp_109 * input_733
        gate_mlp_109 = input_733 = None
        x_257 = x_256 + mul_499
        x_256 = mul_499 = None
        input_734 = torch.nn.functional.silu(y_36, inplace=False)
        input_735 = torch._C._nn.linear(
            input_734,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_734 = None
        chunk_146 = input_735.chunk(3, dim=-1)
        input_735 = None
        shift_mlp_110 = chunk_146[0]
        scale_mlp_110 = chunk_146[1]
        gate_mlp_110 = chunk_146[2]
        chunk_146 = None
        layer_norm_171 = torch.nn.functional.layer_norm(
            x_257,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_525 = 1 + scale_mlp_110
        scale_mlp_110 = None
        mul_500 = layer_norm_171 * add_525
        layer_norm_171 = add_525 = None
        h_110 = mul_500 + shift_mlp_110
        mul_500 = shift_mlp_110 = None
        input_736 = torch._C._nn.linear(
            h_110,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_110 = None
        input_737 = torch.nn.functional.silu(input_736, inplace=False)
        input_736 = None
        input_738 = torch._C._nn.linear(
            input_737,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_737 = None
        mul_501 = gate_mlp_110 * input_738
        gate_mlp_110 = input_738 = None
        x_258 = x_257 + mul_501
        x_257 = mul_501 = None
        input_739 = torch.nn.functional.silu(y_36, inplace=False)
        y_36 = None
        input_740 = torch._C._nn.linear(
            input_739,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_739 = None
        chunk_147 = input_740.chunk(2, dim=-1)
        input_740 = None
        shift_36 = chunk_147[0]
        scale_36 = chunk_147[1]
        chunk_147 = None
        layer_norm_172 = torch.nn.functional.layer_norm(
            x_258, (768,), None, None, 1e-06
        )
        x_258 = None
        add_528 = 1 + scale_36
        scale_36 = None
        mul_502 = layer_norm_172 * add_528
        layer_norm_172 = add_528 = None
        x_259 = mul_502 + shift_36
        mul_502 = shift_36 = None
        x_260 = torch._C._nn.linear(
            x_259,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_259 = None
        sub_36 = x_260 - noise
        x_260 = None
        mul_503 = sub_36 * 0.02
        sub_36 = None
        x_261 = x_254 + mul_503
        x_254 = mul_503 = None
        ones_37 = torch.ones(1)
        mul_504 = ones_37 * 37
        ones_37 = None
        truediv_74 = mul_504 / 50
        mul_504 = None
        t_37 = truediv_74.to(device(type="cuda", index=0))
        truediv_74 = None
        mul_505 = t_37 * 1000
        t_37 = None
        x_262 = torch._C._nn.linear(
            x_261,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_39 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_506 = -9.210340371976184 * arange_39
        arange_39 = None
        truediv_75 = mul_506 / 128
        mul_506 = None
        exp_37 = torch.exp(truediv_75)
        truediv_75 = None
        freqs_37 = exp_37.to(device=device(type="cuda", index=0))
        exp_37 = None
        getitem_579 = mul_505[(slice(None, None, None), None)]
        mul_505 = None
        float_38 = getitem_579.float()
        getitem_579 = None
        getitem_580 = freqs_37[None]
        freqs_37 = None
        args_37 = float_38 * getitem_580
        float_38 = getitem_580 = None
        cos_61 = torch.cos(args_37)
        sin_61 = torch.sin(args_37)
        args_37 = None
        embedding_37 = torch.cat([cos_61, sin_61], dim=-1)
        cos_61 = sin_61 = None
        input_741 = torch._C._nn.linear(
            embedding_37,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_37 = None
        input_742 = torch.nn.functional.silu(input_741, inplace=False)
        input_741 = None
        input_743 = torch._C._nn.linear(
            input_742,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_742 = None
        c_37 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y_37 = input_743 + c_37
        input_743 = c_37 = None
        input_744 = torch.nn.functional.silu(y_37, inplace=False)
        input_745 = torch._C._nn.linear(
            input_744,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_744 = None
        chunk_148 = input_745.chunk(3, dim=-1)
        input_745 = None
        shift_mlp_111 = chunk_148[0]
        scale_mlp_111 = chunk_148[1]
        gate_mlp_111 = chunk_148[2]
        chunk_148 = None
        layer_norm_173 = torch.nn.functional.layer_norm(
            x_262,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_532 = 1 + scale_mlp_111
        scale_mlp_111 = None
        mul_508 = layer_norm_173 * add_532
        layer_norm_173 = add_532 = None
        h_111 = mul_508 + shift_mlp_111
        mul_508 = shift_mlp_111 = None
        input_746 = torch._C._nn.linear(
            h_111,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_111 = None
        input_747 = torch.nn.functional.silu(input_746, inplace=False)
        input_746 = None
        input_748 = torch._C._nn.linear(
            input_747,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_747 = None
        mul_509 = gate_mlp_111 * input_748
        gate_mlp_111 = input_748 = None
        x_263 = x_262 + mul_509
        x_262 = mul_509 = None
        input_749 = torch.nn.functional.silu(y_37, inplace=False)
        input_750 = torch._C._nn.linear(
            input_749,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_749 = None
        chunk_149 = input_750.chunk(3, dim=-1)
        input_750 = None
        shift_mlp_112 = chunk_149[0]
        scale_mlp_112 = chunk_149[1]
        gate_mlp_112 = chunk_149[2]
        chunk_149 = None
        layer_norm_174 = torch.nn.functional.layer_norm(
            x_263,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_535 = 1 + scale_mlp_112
        scale_mlp_112 = None
        mul_510 = layer_norm_174 * add_535
        layer_norm_174 = add_535 = None
        h_112 = mul_510 + shift_mlp_112
        mul_510 = shift_mlp_112 = None
        input_751 = torch._C._nn.linear(
            h_112,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_112 = None
        input_752 = torch.nn.functional.silu(input_751, inplace=False)
        input_751 = None
        input_753 = torch._C._nn.linear(
            input_752,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_752 = None
        mul_511 = gate_mlp_112 * input_753
        gate_mlp_112 = input_753 = None
        x_264 = x_263 + mul_511
        x_263 = mul_511 = None
        input_754 = torch.nn.functional.silu(y_37, inplace=False)
        input_755 = torch._C._nn.linear(
            input_754,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_754 = None
        chunk_150 = input_755.chunk(3, dim=-1)
        input_755 = None
        shift_mlp_113 = chunk_150[0]
        scale_mlp_113 = chunk_150[1]
        gate_mlp_113 = chunk_150[2]
        chunk_150 = None
        layer_norm_175 = torch.nn.functional.layer_norm(
            x_264,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_538 = 1 + scale_mlp_113
        scale_mlp_113 = None
        mul_512 = layer_norm_175 * add_538
        layer_norm_175 = add_538 = None
        h_113 = mul_512 + shift_mlp_113
        mul_512 = shift_mlp_113 = None
        input_756 = torch._C._nn.linear(
            h_113,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_113 = None
        input_757 = torch.nn.functional.silu(input_756, inplace=False)
        input_756 = None
        input_758 = torch._C._nn.linear(
            input_757,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_757 = None
        mul_513 = gate_mlp_113 * input_758
        gate_mlp_113 = input_758 = None
        x_265 = x_264 + mul_513
        x_264 = mul_513 = None
        input_759 = torch.nn.functional.silu(y_37, inplace=False)
        y_37 = None
        input_760 = torch._C._nn.linear(
            input_759,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_759 = None
        chunk_151 = input_760.chunk(2, dim=-1)
        input_760 = None
        shift_37 = chunk_151[0]
        scale_37 = chunk_151[1]
        chunk_151 = None
        layer_norm_176 = torch.nn.functional.layer_norm(
            x_265, (768,), None, None, 1e-06
        )
        x_265 = None
        add_541 = 1 + scale_37
        scale_37 = None
        mul_514 = layer_norm_176 * add_541
        layer_norm_176 = add_541 = None
        x_266 = mul_514 + shift_37
        mul_514 = shift_37 = None
        x_267 = torch._C._nn.linear(
            x_266,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_266 = None
        sub_37 = x_267 - noise
        x_267 = None
        mul_515 = sub_37 * 0.02
        sub_37 = None
        x_268 = x_261 + mul_515
        x_261 = mul_515 = None
        ones_38 = torch.ones(1)
        mul_516 = ones_38 * 38
        ones_38 = None
        truediv_76 = mul_516 / 50
        mul_516 = None
        t_38 = truediv_76.to(device(type="cuda", index=0))
        truediv_76 = None
        mul_517 = t_38 * 1000
        t_38 = None
        x_269 = torch._C._nn.linear(
            x_268,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_40 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_518 = -9.210340371976184 * arange_40
        arange_40 = None
        truediv_77 = mul_518 / 128
        mul_518 = None
        exp_38 = torch.exp(truediv_77)
        truediv_77 = None
        freqs_38 = exp_38.to(device=device(type="cuda", index=0))
        exp_38 = None
        getitem_592 = mul_517[(slice(None, None, None), None)]
        mul_517 = None
        float_39 = getitem_592.float()
        getitem_592 = None
        getitem_593 = freqs_38[None]
        freqs_38 = None
        args_38 = float_39 * getitem_593
        float_39 = getitem_593 = None
        cos_62 = torch.cos(args_38)
        sin_62 = torch.sin(args_38)
        args_38 = None
        embedding_38 = torch.cat([cos_62, sin_62], dim=-1)
        cos_62 = sin_62 = None
        input_761 = torch._C._nn.linear(
            embedding_38,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_38 = None
        input_762 = torch.nn.functional.silu(input_761, inplace=False)
        input_761 = None
        input_763 = torch._C._nn.linear(
            input_762,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_762 = None
        c_38 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y_38 = input_763 + c_38
        input_763 = c_38 = None
        input_764 = torch.nn.functional.silu(y_38, inplace=False)
        input_765 = torch._C._nn.linear(
            input_764,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_764 = None
        chunk_152 = input_765.chunk(3, dim=-1)
        input_765 = None
        shift_mlp_114 = chunk_152[0]
        scale_mlp_114 = chunk_152[1]
        gate_mlp_114 = chunk_152[2]
        chunk_152 = None
        layer_norm_177 = torch.nn.functional.layer_norm(
            x_269,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_545 = 1 + scale_mlp_114
        scale_mlp_114 = None
        mul_520 = layer_norm_177 * add_545
        layer_norm_177 = add_545 = None
        h_114 = mul_520 + shift_mlp_114
        mul_520 = shift_mlp_114 = None
        input_766 = torch._C._nn.linear(
            h_114,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_114 = None
        input_767 = torch.nn.functional.silu(input_766, inplace=False)
        input_766 = None
        input_768 = torch._C._nn.linear(
            input_767,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_767 = None
        mul_521 = gate_mlp_114 * input_768
        gate_mlp_114 = input_768 = None
        x_270 = x_269 + mul_521
        x_269 = mul_521 = None
        input_769 = torch.nn.functional.silu(y_38, inplace=False)
        input_770 = torch._C._nn.linear(
            input_769,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_769 = None
        chunk_153 = input_770.chunk(3, dim=-1)
        input_770 = None
        shift_mlp_115 = chunk_153[0]
        scale_mlp_115 = chunk_153[1]
        gate_mlp_115 = chunk_153[2]
        chunk_153 = None
        layer_norm_178 = torch.nn.functional.layer_norm(
            x_270,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_548 = 1 + scale_mlp_115
        scale_mlp_115 = None
        mul_522 = layer_norm_178 * add_548
        layer_norm_178 = add_548 = None
        h_115 = mul_522 + shift_mlp_115
        mul_522 = shift_mlp_115 = None
        input_771 = torch._C._nn.linear(
            h_115,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_115 = None
        input_772 = torch.nn.functional.silu(input_771, inplace=False)
        input_771 = None
        input_773 = torch._C._nn.linear(
            input_772,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_772 = None
        mul_523 = gate_mlp_115 * input_773
        gate_mlp_115 = input_773 = None
        x_271 = x_270 + mul_523
        x_270 = mul_523 = None
        input_774 = torch.nn.functional.silu(y_38, inplace=False)
        input_775 = torch._C._nn.linear(
            input_774,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_774 = None
        chunk_154 = input_775.chunk(3, dim=-1)
        input_775 = None
        shift_mlp_116 = chunk_154[0]
        scale_mlp_116 = chunk_154[1]
        gate_mlp_116 = chunk_154[2]
        chunk_154 = None
        layer_norm_179 = torch.nn.functional.layer_norm(
            x_271,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_551 = 1 + scale_mlp_116
        scale_mlp_116 = None
        mul_524 = layer_norm_179 * add_551
        layer_norm_179 = add_551 = None
        h_116 = mul_524 + shift_mlp_116
        mul_524 = shift_mlp_116 = None
        input_776 = torch._C._nn.linear(
            h_116,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_116 = None
        input_777 = torch.nn.functional.silu(input_776, inplace=False)
        input_776 = None
        input_778 = torch._C._nn.linear(
            input_777,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_777 = None
        mul_525 = gate_mlp_116 * input_778
        gate_mlp_116 = input_778 = None
        x_272 = x_271 + mul_525
        x_271 = mul_525 = None
        input_779 = torch.nn.functional.silu(y_38, inplace=False)
        y_38 = None
        input_780 = torch._C._nn.linear(
            input_779,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_779 = None
        chunk_155 = input_780.chunk(2, dim=-1)
        input_780 = None
        shift_38 = chunk_155[0]
        scale_38 = chunk_155[1]
        chunk_155 = None
        layer_norm_180 = torch.nn.functional.layer_norm(
            x_272, (768,), None, None, 1e-06
        )
        x_272 = None
        add_554 = 1 + scale_38
        scale_38 = None
        mul_526 = layer_norm_180 * add_554
        layer_norm_180 = add_554 = None
        x_273 = mul_526 + shift_38
        mul_526 = shift_38 = None
        x_274 = torch._C._nn.linear(
            x_273,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_273 = None
        sub_38 = x_274 - noise
        x_274 = None
        mul_527 = sub_38 * 0.02
        sub_38 = None
        x_275 = x_268 + mul_527
        x_268 = mul_527 = None
        ones_39 = torch.ones(1)
        mul_528 = ones_39 * 39
        ones_39 = None
        truediv_78 = mul_528 / 50
        mul_528 = None
        t_39 = truediv_78.to(device(type="cuda", index=0))
        truediv_78 = None
        mul_529 = t_39 * 1000
        t_39 = None
        x_276 = torch._C._nn.linear(
            x_275,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_41 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_530 = -9.210340371976184 * arange_41
        arange_41 = None
        truediv_79 = mul_530 / 128
        mul_530 = None
        exp_39 = torch.exp(truediv_79)
        truediv_79 = None
        freqs_39 = exp_39.to(device=device(type="cuda", index=0))
        exp_39 = None
        getitem_605 = mul_529[(slice(None, None, None), None)]
        mul_529 = None
        float_40 = getitem_605.float()
        getitem_605 = None
        getitem_606 = freqs_39[None]
        freqs_39 = None
        args_39 = float_40 * getitem_606
        float_40 = getitem_606 = None
        cos_63 = torch.cos(args_39)
        sin_63 = torch.sin(args_39)
        args_39 = None
        embedding_39 = torch.cat([cos_63, sin_63], dim=-1)
        cos_63 = sin_63 = None
        input_781 = torch._C._nn.linear(
            embedding_39,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_39 = None
        input_782 = torch.nn.functional.silu(input_781, inplace=False)
        input_781 = None
        input_783 = torch._C._nn.linear(
            input_782,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_782 = None
        c_39 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y_39 = input_783 + c_39
        input_783 = c_39 = None
        input_784 = torch.nn.functional.silu(y_39, inplace=False)
        input_785 = torch._C._nn.linear(
            input_784,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_784 = None
        chunk_156 = input_785.chunk(3, dim=-1)
        input_785 = None
        shift_mlp_117 = chunk_156[0]
        scale_mlp_117 = chunk_156[1]
        gate_mlp_117 = chunk_156[2]
        chunk_156 = None
        layer_norm_181 = torch.nn.functional.layer_norm(
            x_276,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_558 = 1 + scale_mlp_117
        scale_mlp_117 = None
        mul_532 = layer_norm_181 * add_558
        layer_norm_181 = add_558 = None
        h_117 = mul_532 + shift_mlp_117
        mul_532 = shift_mlp_117 = None
        input_786 = torch._C._nn.linear(
            h_117,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_117 = None
        input_787 = torch.nn.functional.silu(input_786, inplace=False)
        input_786 = None
        input_788 = torch._C._nn.linear(
            input_787,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_787 = None
        mul_533 = gate_mlp_117 * input_788
        gate_mlp_117 = input_788 = None
        x_277 = x_276 + mul_533
        x_276 = mul_533 = None
        input_789 = torch.nn.functional.silu(y_39, inplace=False)
        input_790 = torch._C._nn.linear(
            input_789,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_789 = None
        chunk_157 = input_790.chunk(3, dim=-1)
        input_790 = None
        shift_mlp_118 = chunk_157[0]
        scale_mlp_118 = chunk_157[1]
        gate_mlp_118 = chunk_157[2]
        chunk_157 = None
        layer_norm_182 = torch.nn.functional.layer_norm(
            x_277,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_561 = 1 + scale_mlp_118
        scale_mlp_118 = None
        mul_534 = layer_norm_182 * add_561
        layer_norm_182 = add_561 = None
        h_118 = mul_534 + shift_mlp_118
        mul_534 = shift_mlp_118 = None
        input_791 = torch._C._nn.linear(
            h_118,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_118 = None
        input_792 = torch.nn.functional.silu(input_791, inplace=False)
        input_791 = None
        input_793 = torch._C._nn.linear(
            input_792,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_792 = None
        mul_535 = gate_mlp_118 * input_793
        gate_mlp_118 = input_793 = None
        x_278 = x_277 + mul_535
        x_277 = mul_535 = None
        input_794 = torch.nn.functional.silu(y_39, inplace=False)
        input_795 = torch._C._nn.linear(
            input_794,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_794 = None
        chunk_158 = input_795.chunk(3, dim=-1)
        input_795 = None
        shift_mlp_119 = chunk_158[0]
        scale_mlp_119 = chunk_158[1]
        gate_mlp_119 = chunk_158[2]
        chunk_158 = None
        layer_norm_183 = torch.nn.functional.layer_norm(
            x_278,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_564 = 1 + scale_mlp_119
        scale_mlp_119 = None
        mul_536 = layer_norm_183 * add_564
        layer_norm_183 = add_564 = None
        h_119 = mul_536 + shift_mlp_119
        mul_536 = shift_mlp_119 = None
        input_796 = torch._C._nn.linear(
            h_119,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_119 = None
        input_797 = torch.nn.functional.silu(input_796, inplace=False)
        input_796 = None
        input_798 = torch._C._nn.linear(
            input_797,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_797 = None
        mul_537 = gate_mlp_119 * input_798
        gate_mlp_119 = input_798 = None
        x_279 = x_278 + mul_537
        x_278 = mul_537 = None
        input_799 = torch.nn.functional.silu(y_39, inplace=False)
        y_39 = None
        input_800 = torch._C._nn.linear(
            input_799,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_799 = None
        chunk_159 = input_800.chunk(2, dim=-1)
        input_800 = None
        shift_39 = chunk_159[0]
        scale_39 = chunk_159[1]
        chunk_159 = None
        layer_norm_184 = torch.nn.functional.layer_norm(
            x_279, (768,), None, None, 1e-06
        )
        x_279 = None
        add_567 = 1 + scale_39
        scale_39 = None
        mul_538 = layer_norm_184 * add_567
        layer_norm_184 = add_567 = None
        x_280 = mul_538 + shift_39
        mul_538 = shift_39 = None
        x_281 = torch._C._nn.linear(
            x_280,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_280 = None
        sub_39 = x_281 - noise
        x_281 = None
        mul_539 = sub_39 * 0.02
        sub_39 = None
        x_282 = x_275 + mul_539
        x_275 = mul_539 = None
        ones_40 = torch.ones(1)
        mul_540 = ones_40 * 40
        ones_40 = None
        truediv_80 = mul_540 / 50
        mul_540 = None
        t_40 = truediv_80.to(device(type="cuda", index=0))
        truediv_80 = None
        mul_541 = t_40 * 1000
        t_40 = None
        x_283 = torch._C._nn.linear(
            x_282,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_42 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_542 = -9.210340371976184 * arange_42
        arange_42 = None
        truediv_81 = mul_542 / 128
        mul_542 = None
        exp_40 = torch.exp(truediv_81)
        truediv_81 = None
        freqs_40 = exp_40.to(device=device(type="cuda", index=0))
        exp_40 = None
        getitem_618 = mul_541[(slice(None, None, None), None)]
        mul_541 = None
        float_41 = getitem_618.float()
        getitem_618 = None
        getitem_619 = freqs_40[None]
        freqs_40 = None
        args_40 = float_41 * getitem_619
        float_41 = getitem_619 = None
        cos_64 = torch.cos(args_40)
        sin_64 = torch.sin(args_40)
        args_40 = None
        embedding_40 = torch.cat([cos_64, sin_64], dim=-1)
        cos_64 = sin_64 = None
        input_801 = torch._C._nn.linear(
            embedding_40,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_40 = None
        input_802 = torch.nn.functional.silu(input_801, inplace=False)
        input_801 = None
        input_803 = torch._C._nn.linear(
            input_802,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_802 = None
        c_40 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y_40 = input_803 + c_40
        input_803 = c_40 = None
        input_804 = torch.nn.functional.silu(y_40, inplace=False)
        input_805 = torch._C._nn.linear(
            input_804,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_804 = None
        chunk_160 = input_805.chunk(3, dim=-1)
        input_805 = None
        shift_mlp_120 = chunk_160[0]
        scale_mlp_120 = chunk_160[1]
        gate_mlp_120 = chunk_160[2]
        chunk_160 = None
        layer_norm_185 = torch.nn.functional.layer_norm(
            x_283,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_571 = 1 + scale_mlp_120
        scale_mlp_120 = None
        mul_544 = layer_norm_185 * add_571
        layer_norm_185 = add_571 = None
        h_120 = mul_544 + shift_mlp_120
        mul_544 = shift_mlp_120 = None
        input_806 = torch._C._nn.linear(
            h_120,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_120 = None
        input_807 = torch.nn.functional.silu(input_806, inplace=False)
        input_806 = None
        input_808 = torch._C._nn.linear(
            input_807,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_807 = None
        mul_545 = gate_mlp_120 * input_808
        gate_mlp_120 = input_808 = None
        x_284 = x_283 + mul_545
        x_283 = mul_545 = None
        input_809 = torch.nn.functional.silu(y_40, inplace=False)
        input_810 = torch._C._nn.linear(
            input_809,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_809 = None
        chunk_161 = input_810.chunk(3, dim=-1)
        input_810 = None
        shift_mlp_121 = chunk_161[0]
        scale_mlp_121 = chunk_161[1]
        gate_mlp_121 = chunk_161[2]
        chunk_161 = None
        layer_norm_186 = torch.nn.functional.layer_norm(
            x_284,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_574 = 1 + scale_mlp_121
        scale_mlp_121 = None
        mul_546 = layer_norm_186 * add_574
        layer_norm_186 = add_574 = None
        h_121 = mul_546 + shift_mlp_121
        mul_546 = shift_mlp_121 = None
        input_811 = torch._C._nn.linear(
            h_121,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_121 = None
        input_812 = torch.nn.functional.silu(input_811, inplace=False)
        input_811 = None
        input_813 = torch._C._nn.linear(
            input_812,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_812 = None
        mul_547 = gate_mlp_121 * input_813
        gate_mlp_121 = input_813 = None
        x_285 = x_284 + mul_547
        x_284 = mul_547 = None
        input_814 = torch.nn.functional.silu(y_40, inplace=False)
        input_815 = torch._C._nn.linear(
            input_814,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_814 = None
        chunk_162 = input_815.chunk(3, dim=-1)
        input_815 = None
        shift_mlp_122 = chunk_162[0]
        scale_mlp_122 = chunk_162[1]
        gate_mlp_122 = chunk_162[2]
        chunk_162 = None
        layer_norm_187 = torch.nn.functional.layer_norm(
            x_285,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_577 = 1 + scale_mlp_122
        scale_mlp_122 = None
        mul_548 = layer_norm_187 * add_577
        layer_norm_187 = add_577 = None
        h_122 = mul_548 + shift_mlp_122
        mul_548 = shift_mlp_122 = None
        input_816 = torch._C._nn.linear(
            h_122,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_122 = None
        input_817 = torch.nn.functional.silu(input_816, inplace=False)
        input_816 = None
        input_818 = torch._C._nn.linear(
            input_817,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_817 = None
        mul_549 = gate_mlp_122 * input_818
        gate_mlp_122 = input_818 = None
        x_286 = x_285 + mul_549
        x_285 = mul_549 = None
        input_819 = torch.nn.functional.silu(y_40, inplace=False)
        y_40 = None
        input_820 = torch._C._nn.linear(
            input_819,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_819 = None
        chunk_163 = input_820.chunk(2, dim=-1)
        input_820 = None
        shift_40 = chunk_163[0]
        scale_40 = chunk_163[1]
        chunk_163 = None
        layer_norm_188 = torch.nn.functional.layer_norm(
            x_286, (768,), None, None, 1e-06
        )
        x_286 = None
        add_580 = 1 + scale_40
        scale_40 = None
        mul_550 = layer_norm_188 * add_580
        layer_norm_188 = add_580 = None
        x_287 = mul_550 + shift_40
        mul_550 = shift_40 = None
        x_288 = torch._C._nn.linear(
            x_287,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_287 = None
        sub_40 = x_288 - noise
        x_288 = None
        mul_551 = sub_40 * 0.02
        sub_40 = None
        x_289 = x_282 + mul_551
        x_282 = mul_551 = None
        ones_41 = torch.ones(1)
        mul_552 = ones_41 * 41
        ones_41 = None
        truediv_82 = mul_552 / 50
        mul_552 = None
        t_41 = truediv_82.to(device(type="cuda", index=0))
        truediv_82 = None
        mul_553 = t_41 * 1000
        t_41 = None
        x_290 = torch._C._nn.linear(
            x_289,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_43 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_554 = -9.210340371976184 * arange_43
        arange_43 = None
        truediv_83 = mul_554 / 128
        mul_554 = None
        exp_41 = torch.exp(truediv_83)
        truediv_83 = None
        freqs_41 = exp_41.to(device=device(type="cuda", index=0))
        exp_41 = None
        getitem_631 = mul_553[(slice(None, None, None), None)]
        mul_553 = None
        float_42 = getitem_631.float()
        getitem_631 = None
        getitem_632 = freqs_41[None]
        freqs_41 = None
        args_41 = float_42 * getitem_632
        float_42 = getitem_632 = None
        cos_65 = torch.cos(args_41)
        sin_65 = torch.sin(args_41)
        args_41 = None
        embedding_41 = torch.cat([cos_65, sin_65], dim=-1)
        cos_65 = sin_65 = None
        input_821 = torch._C._nn.linear(
            embedding_41,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_41 = None
        input_822 = torch.nn.functional.silu(input_821, inplace=False)
        input_821 = None
        input_823 = torch._C._nn.linear(
            input_822,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_822 = None
        c_41 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y_41 = input_823 + c_41
        input_823 = c_41 = None
        input_824 = torch.nn.functional.silu(y_41, inplace=False)
        input_825 = torch._C._nn.linear(
            input_824,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_824 = None
        chunk_164 = input_825.chunk(3, dim=-1)
        input_825 = None
        shift_mlp_123 = chunk_164[0]
        scale_mlp_123 = chunk_164[1]
        gate_mlp_123 = chunk_164[2]
        chunk_164 = None
        layer_norm_189 = torch.nn.functional.layer_norm(
            x_290,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_584 = 1 + scale_mlp_123
        scale_mlp_123 = None
        mul_556 = layer_norm_189 * add_584
        layer_norm_189 = add_584 = None
        h_123 = mul_556 + shift_mlp_123
        mul_556 = shift_mlp_123 = None
        input_826 = torch._C._nn.linear(
            h_123,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_123 = None
        input_827 = torch.nn.functional.silu(input_826, inplace=False)
        input_826 = None
        input_828 = torch._C._nn.linear(
            input_827,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_827 = None
        mul_557 = gate_mlp_123 * input_828
        gate_mlp_123 = input_828 = None
        x_291 = x_290 + mul_557
        x_290 = mul_557 = None
        input_829 = torch.nn.functional.silu(y_41, inplace=False)
        input_830 = torch._C._nn.linear(
            input_829,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_829 = None
        chunk_165 = input_830.chunk(3, dim=-1)
        input_830 = None
        shift_mlp_124 = chunk_165[0]
        scale_mlp_124 = chunk_165[1]
        gate_mlp_124 = chunk_165[2]
        chunk_165 = None
        layer_norm_190 = torch.nn.functional.layer_norm(
            x_291,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_587 = 1 + scale_mlp_124
        scale_mlp_124 = None
        mul_558 = layer_norm_190 * add_587
        layer_norm_190 = add_587 = None
        h_124 = mul_558 + shift_mlp_124
        mul_558 = shift_mlp_124 = None
        input_831 = torch._C._nn.linear(
            h_124,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_124 = None
        input_832 = torch.nn.functional.silu(input_831, inplace=False)
        input_831 = None
        input_833 = torch._C._nn.linear(
            input_832,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_832 = None
        mul_559 = gate_mlp_124 * input_833
        gate_mlp_124 = input_833 = None
        x_292 = x_291 + mul_559
        x_291 = mul_559 = None
        input_834 = torch.nn.functional.silu(y_41, inplace=False)
        input_835 = torch._C._nn.linear(
            input_834,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_834 = None
        chunk_166 = input_835.chunk(3, dim=-1)
        input_835 = None
        shift_mlp_125 = chunk_166[0]
        scale_mlp_125 = chunk_166[1]
        gate_mlp_125 = chunk_166[2]
        chunk_166 = None
        layer_norm_191 = torch.nn.functional.layer_norm(
            x_292,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_590 = 1 + scale_mlp_125
        scale_mlp_125 = None
        mul_560 = layer_norm_191 * add_590
        layer_norm_191 = add_590 = None
        h_125 = mul_560 + shift_mlp_125
        mul_560 = shift_mlp_125 = None
        input_836 = torch._C._nn.linear(
            h_125,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_125 = None
        input_837 = torch.nn.functional.silu(input_836, inplace=False)
        input_836 = None
        input_838 = torch._C._nn.linear(
            input_837,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_837 = None
        mul_561 = gate_mlp_125 * input_838
        gate_mlp_125 = input_838 = None
        x_293 = x_292 + mul_561
        x_292 = mul_561 = None
        input_839 = torch.nn.functional.silu(y_41, inplace=False)
        y_41 = None
        input_840 = torch._C._nn.linear(
            input_839,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_839 = None
        chunk_167 = input_840.chunk(2, dim=-1)
        input_840 = None
        shift_41 = chunk_167[0]
        scale_41 = chunk_167[1]
        chunk_167 = None
        layer_norm_192 = torch.nn.functional.layer_norm(
            x_293, (768,), None, None, 1e-06
        )
        x_293 = None
        add_593 = 1 + scale_41
        scale_41 = None
        mul_562 = layer_norm_192 * add_593
        layer_norm_192 = add_593 = None
        x_294 = mul_562 + shift_41
        mul_562 = shift_41 = None
        x_295 = torch._C._nn.linear(
            x_294,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_294 = None
        sub_41 = x_295 - noise
        x_295 = None
        mul_563 = sub_41 * 0.02
        sub_41 = None
        x_296 = x_289 + mul_563
        x_289 = mul_563 = None
        ones_42 = torch.ones(1)
        mul_564 = ones_42 * 42
        ones_42 = None
        truediv_84 = mul_564 / 50
        mul_564 = None
        t_42 = truediv_84.to(device(type="cuda", index=0))
        truediv_84 = None
        mul_565 = t_42 * 1000
        t_42 = None
        x_297 = torch._C._nn.linear(
            x_296,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_44 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_566 = -9.210340371976184 * arange_44
        arange_44 = None
        truediv_85 = mul_566 / 128
        mul_566 = None
        exp_42 = torch.exp(truediv_85)
        truediv_85 = None
        freqs_42 = exp_42.to(device=device(type="cuda", index=0))
        exp_42 = None
        getitem_644 = mul_565[(slice(None, None, None), None)]
        mul_565 = None
        float_43 = getitem_644.float()
        getitem_644 = None
        getitem_645 = freqs_42[None]
        freqs_42 = None
        args_42 = float_43 * getitem_645
        float_43 = getitem_645 = None
        cos_66 = torch.cos(args_42)
        sin_66 = torch.sin(args_42)
        args_42 = None
        embedding_42 = torch.cat([cos_66, sin_66], dim=-1)
        cos_66 = sin_66 = None
        input_841 = torch._C._nn.linear(
            embedding_42,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_42 = None
        input_842 = torch.nn.functional.silu(input_841, inplace=False)
        input_841 = None
        input_843 = torch._C._nn.linear(
            input_842,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_842 = None
        c_42 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y_42 = input_843 + c_42
        input_843 = c_42 = None
        input_844 = torch.nn.functional.silu(y_42, inplace=False)
        input_845 = torch._C._nn.linear(
            input_844,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_844 = None
        chunk_168 = input_845.chunk(3, dim=-1)
        input_845 = None
        shift_mlp_126 = chunk_168[0]
        scale_mlp_126 = chunk_168[1]
        gate_mlp_126 = chunk_168[2]
        chunk_168 = None
        layer_norm_193 = torch.nn.functional.layer_norm(
            x_297,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_597 = 1 + scale_mlp_126
        scale_mlp_126 = None
        mul_568 = layer_norm_193 * add_597
        layer_norm_193 = add_597 = None
        h_126 = mul_568 + shift_mlp_126
        mul_568 = shift_mlp_126 = None
        input_846 = torch._C._nn.linear(
            h_126,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_126 = None
        input_847 = torch.nn.functional.silu(input_846, inplace=False)
        input_846 = None
        input_848 = torch._C._nn.linear(
            input_847,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_847 = None
        mul_569 = gate_mlp_126 * input_848
        gate_mlp_126 = input_848 = None
        x_298 = x_297 + mul_569
        x_297 = mul_569 = None
        input_849 = torch.nn.functional.silu(y_42, inplace=False)
        input_850 = torch._C._nn.linear(
            input_849,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_849 = None
        chunk_169 = input_850.chunk(3, dim=-1)
        input_850 = None
        shift_mlp_127 = chunk_169[0]
        scale_mlp_127 = chunk_169[1]
        gate_mlp_127 = chunk_169[2]
        chunk_169 = None
        layer_norm_194 = torch.nn.functional.layer_norm(
            x_298,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_600 = 1 + scale_mlp_127
        scale_mlp_127 = None
        mul_570 = layer_norm_194 * add_600
        layer_norm_194 = add_600 = None
        h_127 = mul_570 + shift_mlp_127
        mul_570 = shift_mlp_127 = None
        input_851 = torch._C._nn.linear(
            h_127,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_127 = None
        input_852 = torch.nn.functional.silu(input_851, inplace=False)
        input_851 = None
        input_853 = torch._C._nn.linear(
            input_852,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_852 = None
        mul_571 = gate_mlp_127 * input_853
        gate_mlp_127 = input_853 = None
        x_299 = x_298 + mul_571
        x_298 = mul_571 = None
        input_854 = torch.nn.functional.silu(y_42, inplace=False)
        input_855 = torch._C._nn.linear(
            input_854,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_854 = None
        chunk_170 = input_855.chunk(3, dim=-1)
        input_855 = None
        shift_mlp_128 = chunk_170[0]
        scale_mlp_128 = chunk_170[1]
        gate_mlp_128 = chunk_170[2]
        chunk_170 = None
        layer_norm_195 = torch.nn.functional.layer_norm(
            x_299,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_603 = 1 + scale_mlp_128
        scale_mlp_128 = None
        mul_572 = layer_norm_195 * add_603
        layer_norm_195 = add_603 = None
        h_128 = mul_572 + shift_mlp_128
        mul_572 = shift_mlp_128 = None
        input_856 = torch._C._nn.linear(
            h_128,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_128 = None
        input_857 = torch.nn.functional.silu(input_856, inplace=False)
        input_856 = None
        input_858 = torch._C._nn.linear(
            input_857,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_857 = None
        mul_573 = gate_mlp_128 * input_858
        gate_mlp_128 = input_858 = None
        x_300 = x_299 + mul_573
        x_299 = mul_573 = None
        input_859 = torch.nn.functional.silu(y_42, inplace=False)
        y_42 = None
        input_860 = torch._C._nn.linear(
            input_859,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_859 = None
        chunk_171 = input_860.chunk(2, dim=-1)
        input_860 = None
        shift_42 = chunk_171[0]
        scale_42 = chunk_171[1]
        chunk_171 = None
        layer_norm_196 = torch.nn.functional.layer_norm(
            x_300, (768,), None, None, 1e-06
        )
        x_300 = None
        add_606 = 1 + scale_42
        scale_42 = None
        mul_574 = layer_norm_196 * add_606
        layer_norm_196 = add_606 = None
        x_301 = mul_574 + shift_42
        mul_574 = shift_42 = None
        x_302 = torch._C._nn.linear(
            x_301,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_301 = None
        sub_42 = x_302 - noise
        x_302 = None
        mul_575 = sub_42 * 0.02
        sub_42 = None
        x_303 = x_296 + mul_575
        x_296 = mul_575 = None
        ones_43 = torch.ones(1)
        mul_576 = ones_43 * 43
        ones_43 = None
        truediv_86 = mul_576 / 50
        mul_576 = None
        t_43 = truediv_86.to(device(type="cuda", index=0))
        truediv_86 = None
        mul_577 = t_43 * 1000
        t_43 = None
        x_304 = torch._C._nn.linear(
            x_303,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_45 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_578 = -9.210340371976184 * arange_45
        arange_45 = None
        truediv_87 = mul_578 / 128
        mul_578 = None
        exp_43 = torch.exp(truediv_87)
        truediv_87 = None
        freqs_43 = exp_43.to(device=device(type="cuda", index=0))
        exp_43 = None
        getitem_657 = mul_577[(slice(None, None, None), None)]
        mul_577 = None
        float_44 = getitem_657.float()
        getitem_657 = None
        getitem_658 = freqs_43[None]
        freqs_43 = None
        args_43 = float_44 * getitem_658
        float_44 = getitem_658 = None
        cos_67 = torch.cos(args_43)
        sin_67 = torch.sin(args_43)
        args_43 = None
        embedding_43 = torch.cat([cos_67, sin_67], dim=-1)
        cos_67 = sin_67 = None
        input_861 = torch._C._nn.linear(
            embedding_43,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_43 = None
        input_862 = torch.nn.functional.silu(input_861, inplace=False)
        input_861 = None
        input_863 = torch._C._nn.linear(
            input_862,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_862 = None
        c_43 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y_43 = input_863 + c_43
        input_863 = c_43 = None
        input_864 = torch.nn.functional.silu(y_43, inplace=False)
        input_865 = torch._C._nn.linear(
            input_864,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_864 = None
        chunk_172 = input_865.chunk(3, dim=-1)
        input_865 = None
        shift_mlp_129 = chunk_172[0]
        scale_mlp_129 = chunk_172[1]
        gate_mlp_129 = chunk_172[2]
        chunk_172 = None
        layer_norm_197 = torch.nn.functional.layer_norm(
            x_304,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_610 = 1 + scale_mlp_129
        scale_mlp_129 = None
        mul_580 = layer_norm_197 * add_610
        layer_norm_197 = add_610 = None
        h_129 = mul_580 + shift_mlp_129
        mul_580 = shift_mlp_129 = None
        input_866 = torch._C._nn.linear(
            h_129,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_129 = None
        input_867 = torch.nn.functional.silu(input_866, inplace=False)
        input_866 = None
        input_868 = torch._C._nn.linear(
            input_867,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_867 = None
        mul_581 = gate_mlp_129 * input_868
        gate_mlp_129 = input_868 = None
        x_305 = x_304 + mul_581
        x_304 = mul_581 = None
        input_869 = torch.nn.functional.silu(y_43, inplace=False)
        input_870 = torch._C._nn.linear(
            input_869,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_869 = None
        chunk_173 = input_870.chunk(3, dim=-1)
        input_870 = None
        shift_mlp_130 = chunk_173[0]
        scale_mlp_130 = chunk_173[1]
        gate_mlp_130 = chunk_173[2]
        chunk_173 = None
        layer_norm_198 = torch.nn.functional.layer_norm(
            x_305,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_613 = 1 + scale_mlp_130
        scale_mlp_130 = None
        mul_582 = layer_norm_198 * add_613
        layer_norm_198 = add_613 = None
        h_130 = mul_582 + shift_mlp_130
        mul_582 = shift_mlp_130 = None
        input_871 = torch._C._nn.linear(
            h_130,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_130 = None
        input_872 = torch.nn.functional.silu(input_871, inplace=False)
        input_871 = None
        input_873 = torch._C._nn.linear(
            input_872,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_872 = None
        mul_583 = gate_mlp_130 * input_873
        gate_mlp_130 = input_873 = None
        x_306 = x_305 + mul_583
        x_305 = mul_583 = None
        input_874 = torch.nn.functional.silu(y_43, inplace=False)
        input_875 = torch._C._nn.linear(
            input_874,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_874 = None
        chunk_174 = input_875.chunk(3, dim=-1)
        input_875 = None
        shift_mlp_131 = chunk_174[0]
        scale_mlp_131 = chunk_174[1]
        gate_mlp_131 = chunk_174[2]
        chunk_174 = None
        layer_norm_199 = torch.nn.functional.layer_norm(
            x_306,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_616 = 1 + scale_mlp_131
        scale_mlp_131 = None
        mul_584 = layer_norm_199 * add_616
        layer_norm_199 = add_616 = None
        h_131 = mul_584 + shift_mlp_131
        mul_584 = shift_mlp_131 = None
        input_876 = torch._C._nn.linear(
            h_131,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_131 = None
        input_877 = torch.nn.functional.silu(input_876, inplace=False)
        input_876 = None
        input_878 = torch._C._nn.linear(
            input_877,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_877 = None
        mul_585 = gate_mlp_131 * input_878
        gate_mlp_131 = input_878 = None
        x_307 = x_306 + mul_585
        x_306 = mul_585 = None
        input_879 = torch.nn.functional.silu(y_43, inplace=False)
        y_43 = None
        input_880 = torch._C._nn.linear(
            input_879,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_879 = None
        chunk_175 = input_880.chunk(2, dim=-1)
        input_880 = None
        shift_43 = chunk_175[0]
        scale_43 = chunk_175[1]
        chunk_175 = None
        layer_norm_200 = torch.nn.functional.layer_norm(
            x_307, (768,), None, None, 1e-06
        )
        x_307 = None
        add_619 = 1 + scale_43
        scale_43 = None
        mul_586 = layer_norm_200 * add_619
        layer_norm_200 = add_619 = None
        x_308 = mul_586 + shift_43
        mul_586 = shift_43 = None
        x_309 = torch._C._nn.linear(
            x_308,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_308 = None
        sub_43 = x_309 - noise
        x_309 = None
        mul_587 = sub_43 * 0.02
        sub_43 = None
        x_310 = x_303 + mul_587
        x_303 = mul_587 = None
        ones_44 = torch.ones(1)
        mul_588 = ones_44 * 44
        ones_44 = None
        truediv_88 = mul_588 / 50
        mul_588 = None
        t_44 = truediv_88.to(device(type="cuda", index=0))
        truediv_88 = None
        mul_589 = t_44 * 1000
        t_44 = None
        x_311 = torch._C._nn.linear(
            x_310,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_46 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_590 = -9.210340371976184 * arange_46
        arange_46 = None
        truediv_89 = mul_590 / 128
        mul_590 = None
        exp_44 = torch.exp(truediv_89)
        truediv_89 = None
        freqs_44 = exp_44.to(device=device(type="cuda", index=0))
        exp_44 = None
        getitem_670 = mul_589[(slice(None, None, None), None)]
        mul_589 = None
        float_45 = getitem_670.float()
        getitem_670 = None
        getitem_671 = freqs_44[None]
        freqs_44 = None
        args_44 = float_45 * getitem_671
        float_45 = getitem_671 = None
        cos_68 = torch.cos(args_44)
        sin_68 = torch.sin(args_44)
        args_44 = None
        embedding_44 = torch.cat([cos_68, sin_68], dim=-1)
        cos_68 = sin_68 = None
        input_881 = torch._C._nn.linear(
            embedding_44,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_44 = None
        input_882 = torch.nn.functional.silu(input_881, inplace=False)
        input_881 = None
        input_883 = torch._C._nn.linear(
            input_882,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_882 = None
        c_44 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y_44 = input_883 + c_44
        input_883 = c_44 = None
        input_884 = torch.nn.functional.silu(y_44, inplace=False)
        input_885 = torch._C._nn.linear(
            input_884,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_884 = None
        chunk_176 = input_885.chunk(3, dim=-1)
        input_885 = None
        shift_mlp_132 = chunk_176[0]
        scale_mlp_132 = chunk_176[1]
        gate_mlp_132 = chunk_176[2]
        chunk_176 = None
        layer_norm_201 = torch.nn.functional.layer_norm(
            x_311,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_623 = 1 + scale_mlp_132
        scale_mlp_132 = None
        mul_592 = layer_norm_201 * add_623
        layer_norm_201 = add_623 = None
        h_132 = mul_592 + shift_mlp_132
        mul_592 = shift_mlp_132 = None
        input_886 = torch._C._nn.linear(
            h_132,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_132 = None
        input_887 = torch.nn.functional.silu(input_886, inplace=False)
        input_886 = None
        input_888 = torch._C._nn.linear(
            input_887,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_887 = None
        mul_593 = gate_mlp_132 * input_888
        gate_mlp_132 = input_888 = None
        x_312 = x_311 + mul_593
        x_311 = mul_593 = None
        input_889 = torch.nn.functional.silu(y_44, inplace=False)
        input_890 = torch._C._nn.linear(
            input_889,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_889 = None
        chunk_177 = input_890.chunk(3, dim=-1)
        input_890 = None
        shift_mlp_133 = chunk_177[0]
        scale_mlp_133 = chunk_177[1]
        gate_mlp_133 = chunk_177[2]
        chunk_177 = None
        layer_norm_202 = torch.nn.functional.layer_norm(
            x_312,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_626 = 1 + scale_mlp_133
        scale_mlp_133 = None
        mul_594 = layer_norm_202 * add_626
        layer_norm_202 = add_626 = None
        h_133 = mul_594 + shift_mlp_133
        mul_594 = shift_mlp_133 = None
        input_891 = torch._C._nn.linear(
            h_133,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_133 = None
        input_892 = torch.nn.functional.silu(input_891, inplace=False)
        input_891 = None
        input_893 = torch._C._nn.linear(
            input_892,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_892 = None
        mul_595 = gate_mlp_133 * input_893
        gate_mlp_133 = input_893 = None
        x_313 = x_312 + mul_595
        x_312 = mul_595 = None
        input_894 = torch.nn.functional.silu(y_44, inplace=False)
        input_895 = torch._C._nn.linear(
            input_894,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_894 = None
        chunk_178 = input_895.chunk(3, dim=-1)
        input_895 = None
        shift_mlp_134 = chunk_178[0]
        scale_mlp_134 = chunk_178[1]
        gate_mlp_134 = chunk_178[2]
        chunk_178 = None
        layer_norm_203 = torch.nn.functional.layer_norm(
            x_313,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_629 = 1 + scale_mlp_134
        scale_mlp_134 = None
        mul_596 = layer_norm_203 * add_629
        layer_norm_203 = add_629 = None
        h_134 = mul_596 + shift_mlp_134
        mul_596 = shift_mlp_134 = None
        input_896 = torch._C._nn.linear(
            h_134,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_134 = None
        input_897 = torch.nn.functional.silu(input_896, inplace=False)
        input_896 = None
        input_898 = torch._C._nn.linear(
            input_897,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_897 = None
        mul_597 = gate_mlp_134 * input_898
        gate_mlp_134 = input_898 = None
        x_314 = x_313 + mul_597
        x_313 = mul_597 = None
        input_899 = torch.nn.functional.silu(y_44, inplace=False)
        y_44 = None
        input_900 = torch._C._nn.linear(
            input_899,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_899 = None
        chunk_179 = input_900.chunk(2, dim=-1)
        input_900 = None
        shift_44 = chunk_179[0]
        scale_44 = chunk_179[1]
        chunk_179 = None
        layer_norm_204 = torch.nn.functional.layer_norm(
            x_314, (768,), None, None, 1e-06
        )
        x_314 = None
        add_632 = 1 + scale_44
        scale_44 = None
        mul_598 = layer_norm_204 * add_632
        layer_norm_204 = add_632 = None
        x_315 = mul_598 + shift_44
        mul_598 = shift_44 = None
        x_316 = torch._C._nn.linear(
            x_315,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_315 = None
        sub_44 = x_316 - noise
        x_316 = None
        mul_599 = sub_44 * 0.02
        sub_44 = None
        x_317 = x_310 + mul_599
        x_310 = mul_599 = None
        ones_45 = torch.ones(1)
        mul_600 = ones_45 * 45
        ones_45 = None
        truediv_90 = mul_600 / 50
        mul_600 = None
        t_45 = truediv_90.to(device(type="cuda", index=0))
        truediv_90 = None
        mul_601 = t_45 * 1000
        t_45 = None
        x_318 = torch._C._nn.linear(
            x_317,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_47 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_602 = -9.210340371976184 * arange_47
        arange_47 = None
        truediv_91 = mul_602 / 128
        mul_602 = None
        exp_45 = torch.exp(truediv_91)
        truediv_91 = None
        freqs_45 = exp_45.to(device=device(type="cuda", index=0))
        exp_45 = None
        getitem_683 = mul_601[(slice(None, None, None), None)]
        mul_601 = None
        float_46 = getitem_683.float()
        getitem_683 = None
        getitem_684 = freqs_45[None]
        freqs_45 = None
        args_45 = float_46 * getitem_684
        float_46 = getitem_684 = None
        cos_69 = torch.cos(args_45)
        sin_69 = torch.sin(args_45)
        args_45 = None
        embedding_45 = torch.cat([cos_69, sin_69], dim=-1)
        cos_69 = sin_69 = None
        input_901 = torch._C._nn.linear(
            embedding_45,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_45 = None
        input_902 = torch.nn.functional.silu(input_901, inplace=False)
        input_901 = None
        input_903 = torch._C._nn.linear(
            input_902,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_902 = None
        c_45 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y_45 = input_903 + c_45
        input_903 = c_45 = None
        input_904 = torch.nn.functional.silu(y_45, inplace=False)
        input_905 = torch._C._nn.linear(
            input_904,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_904 = None
        chunk_180 = input_905.chunk(3, dim=-1)
        input_905 = None
        shift_mlp_135 = chunk_180[0]
        scale_mlp_135 = chunk_180[1]
        gate_mlp_135 = chunk_180[2]
        chunk_180 = None
        layer_norm_205 = torch.nn.functional.layer_norm(
            x_318,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_636 = 1 + scale_mlp_135
        scale_mlp_135 = None
        mul_604 = layer_norm_205 * add_636
        layer_norm_205 = add_636 = None
        h_135 = mul_604 + shift_mlp_135
        mul_604 = shift_mlp_135 = None
        input_906 = torch._C._nn.linear(
            h_135,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_135 = None
        input_907 = torch.nn.functional.silu(input_906, inplace=False)
        input_906 = None
        input_908 = torch._C._nn.linear(
            input_907,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_907 = None
        mul_605 = gate_mlp_135 * input_908
        gate_mlp_135 = input_908 = None
        x_319 = x_318 + mul_605
        x_318 = mul_605 = None
        input_909 = torch.nn.functional.silu(y_45, inplace=False)
        input_910 = torch._C._nn.linear(
            input_909,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_909 = None
        chunk_181 = input_910.chunk(3, dim=-1)
        input_910 = None
        shift_mlp_136 = chunk_181[0]
        scale_mlp_136 = chunk_181[1]
        gate_mlp_136 = chunk_181[2]
        chunk_181 = None
        layer_norm_206 = torch.nn.functional.layer_norm(
            x_319,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_639 = 1 + scale_mlp_136
        scale_mlp_136 = None
        mul_606 = layer_norm_206 * add_639
        layer_norm_206 = add_639 = None
        h_136 = mul_606 + shift_mlp_136
        mul_606 = shift_mlp_136 = None
        input_911 = torch._C._nn.linear(
            h_136,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_136 = None
        input_912 = torch.nn.functional.silu(input_911, inplace=False)
        input_911 = None
        input_913 = torch._C._nn.linear(
            input_912,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_912 = None
        mul_607 = gate_mlp_136 * input_913
        gate_mlp_136 = input_913 = None
        x_320 = x_319 + mul_607
        x_319 = mul_607 = None
        input_914 = torch.nn.functional.silu(y_45, inplace=False)
        input_915 = torch._C._nn.linear(
            input_914,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_914 = None
        chunk_182 = input_915.chunk(3, dim=-1)
        input_915 = None
        shift_mlp_137 = chunk_182[0]
        scale_mlp_137 = chunk_182[1]
        gate_mlp_137 = chunk_182[2]
        chunk_182 = None
        layer_norm_207 = torch.nn.functional.layer_norm(
            x_320,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_642 = 1 + scale_mlp_137
        scale_mlp_137 = None
        mul_608 = layer_norm_207 * add_642
        layer_norm_207 = add_642 = None
        h_137 = mul_608 + shift_mlp_137
        mul_608 = shift_mlp_137 = None
        input_916 = torch._C._nn.linear(
            h_137,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_137 = None
        input_917 = torch.nn.functional.silu(input_916, inplace=False)
        input_916 = None
        input_918 = torch._C._nn.linear(
            input_917,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_917 = None
        mul_609 = gate_mlp_137 * input_918
        gate_mlp_137 = input_918 = None
        x_321 = x_320 + mul_609
        x_320 = mul_609 = None
        input_919 = torch.nn.functional.silu(y_45, inplace=False)
        y_45 = None
        input_920 = torch._C._nn.linear(
            input_919,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_919 = None
        chunk_183 = input_920.chunk(2, dim=-1)
        input_920 = None
        shift_45 = chunk_183[0]
        scale_45 = chunk_183[1]
        chunk_183 = None
        layer_norm_208 = torch.nn.functional.layer_norm(
            x_321, (768,), None, None, 1e-06
        )
        x_321 = None
        add_645 = 1 + scale_45
        scale_45 = None
        mul_610 = layer_norm_208 * add_645
        layer_norm_208 = add_645 = None
        x_322 = mul_610 + shift_45
        mul_610 = shift_45 = None
        x_323 = torch._C._nn.linear(
            x_322,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_322 = None
        sub_45 = x_323 - noise
        x_323 = None
        mul_611 = sub_45 * 0.02
        sub_45 = None
        x_324 = x_317 + mul_611
        x_317 = mul_611 = None
        ones_46 = torch.ones(1)
        mul_612 = ones_46 * 46
        ones_46 = None
        truediv_92 = mul_612 / 50
        mul_612 = None
        t_46 = truediv_92.to(device(type="cuda", index=0))
        truediv_92 = None
        mul_613 = t_46 * 1000
        t_46 = None
        x_325 = torch._C._nn.linear(
            x_324,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_48 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_614 = -9.210340371976184 * arange_48
        arange_48 = None
        truediv_93 = mul_614 / 128
        mul_614 = None
        exp_46 = torch.exp(truediv_93)
        truediv_93 = None
        freqs_46 = exp_46.to(device=device(type="cuda", index=0))
        exp_46 = None
        getitem_696 = mul_613[(slice(None, None, None), None)]
        mul_613 = None
        float_47 = getitem_696.float()
        getitem_696 = None
        getitem_697 = freqs_46[None]
        freqs_46 = None
        args_46 = float_47 * getitem_697
        float_47 = getitem_697 = None
        cos_70 = torch.cos(args_46)
        sin_70 = torch.sin(args_46)
        args_46 = None
        embedding_46 = torch.cat([cos_70, sin_70], dim=-1)
        cos_70 = sin_70 = None
        input_921 = torch._C._nn.linear(
            embedding_46,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_46 = None
        input_922 = torch.nn.functional.silu(input_921, inplace=False)
        input_921 = None
        input_923 = torch._C._nn.linear(
            input_922,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_922 = None
        c_46 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y_46 = input_923 + c_46
        input_923 = c_46 = None
        input_924 = torch.nn.functional.silu(y_46, inplace=False)
        input_925 = torch._C._nn.linear(
            input_924,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_924 = None
        chunk_184 = input_925.chunk(3, dim=-1)
        input_925 = None
        shift_mlp_138 = chunk_184[0]
        scale_mlp_138 = chunk_184[1]
        gate_mlp_138 = chunk_184[2]
        chunk_184 = None
        layer_norm_209 = torch.nn.functional.layer_norm(
            x_325,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_649 = 1 + scale_mlp_138
        scale_mlp_138 = None
        mul_616 = layer_norm_209 * add_649
        layer_norm_209 = add_649 = None
        h_138 = mul_616 + shift_mlp_138
        mul_616 = shift_mlp_138 = None
        input_926 = torch._C._nn.linear(
            h_138,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_138 = None
        input_927 = torch.nn.functional.silu(input_926, inplace=False)
        input_926 = None
        input_928 = torch._C._nn.linear(
            input_927,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_927 = None
        mul_617 = gate_mlp_138 * input_928
        gate_mlp_138 = input_928 = None
        x_326 = x_325 + mul_617
        x_325 = mul_617 = None
        input_929 = torch.nn.functional.silu(y_46, inplace=False)
        input_930 = torch._C._nn.linear(
            input_929,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_929 = None
        chunk_185 = input_930.chunk(3, dim=-1)
        input_930 = None
        shift_mlp_139 = chunk_185[0]
        scale_mlp_139 = chunk_185[1]
        gate_mlp_139 = chunk_185[2]
        chunk_185 = None
        layer_norm_210 = torch.nn.functional.layer_norm(
            x_326,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_652 = 1 + scale_mlp_139
        scale_mlp_139 = None
        mul_618 = layer_norm_210 * add_652
        layer_norm_210 = add_652 = None
        h_139 = mul_618 + shift_mlp_139
        mul_618 = shift_mlp_139 = None
        input_931 = torch._C._nn.linear(
            h_139,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_139 = None
        input_932 = torch.nn.functional.silu(input_931, inplace=False)
        input_931 = None
        input_933 = torch._C._nn.linear(
            input_932,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_932 = None
        mul_619 = gate_mlp_139 * input_933
        gate_mlp_139 = input_933 = None
        x_327 = x_326 + mul_619
        x_326 = mul_619 = None
        input_934 = torch.nn.functional.silu(y_46, inplace=False)
        input_935 = torch._C._nn.linear(
            input_934,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_934 = None
        chunk_186 = input_935.chunk(3, dim=-1)
        input_935 = None
        shift_mlp_140 = chunk_186[0]
        scale_mlp_140 = chunk_186[1]
        gate_mlp_140 = chunk_186[2]
        chunk_186 = None
        layer_norm_211 = torch.nn.functional.layer_norm(
            x_327,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_655 = 1 + scale_mlp_140
        scale_mlp_140 = None
        mul_620 = layer_norm_211 * add_655
        layer_norm_211 = add_655 = None
        h_140 = mul_620 + shift_mlp_140
        mul_620 = shift_mlp_140 = None
        input_936 = torch._C._nn.linear(
            h_140,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_140 = None
        input_937 = torch.nn.functional.silu(input_936, inplace=False)
        input_936 = None
        input_938 = torch._C._nn.linear(
            input_937,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_937 = None
        mul_621 = gate_mlp_140 * input_938
        gate_mlp_140 = input_938 = None
        x_328 = x_327 + mul_621
        x_327 = mul_621 = None
        input_939 = torch.nn.functional.silu(y_46, inplace=False)
        y_46 = None
        input_940 = torch._C._nn.linear(
            input_939,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_939 = None
        chunk_187 = input_940.chunk(2, dim=-1)
        input_940 = None
        shift_46 = chunk_187[0]
        scale_46 = chunk_187[1]
        chunk_187 = None
        layer_norm_212 = torch.nn.functional.layer_norm(
            x_328, (768,), None, None, 1e-06
        )
        x_328 = None
        add_658 = 1 + scale_46
        scale_46 = None
        mul_622 = layer_norm_212 * add_658
        layer_norm_212 = add_658 = None
        x_329 = mul_622 + shift_46
        mul_622 = shift_46 = None
        x_330 = torch._C._nn.linear(
            x_329,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_329 = None
        sub_46 = x_330 - noise
        x_330 = None
        mul_623 = sub_46 * 0.02
        sub_46 = None
        x_331 = x_324 + mul_623
        x_324 = mul_623 = None
        ones_47 = torch.ones(1)
        mul_624 = ones_47 * 47
        ones_47 = None
        truediv_94 = mul_624 / 50
        mul_624 = None
        t_47 = truediv_94.to(device(type="cuda", index=0))
        truediv_94 = None
        mul_625 = t_47 * 1000
        t_47 = None
        x_332 = torch._C._nn.linear(
            x_331,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_49 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_626 = -9.210340371976184 * arange_49
        arange_49 = None
        truediv_95 = mul_626 / 128
        mul_626 = None
        exp_47 = torch.exp(truediv_95)
        truediv_95 = None
        freqs_47 = exp_47.to(device=device(type="cuda", index=0))
        exp_47 = None
        getitem_709 = mul_625[(slice(None, None, None), None)]
        mul_625 = None
        float_48 = getitem_709.float()
        getitem_709 = None
        getitem_710 = freqs_47[None]
        freqs_47 = None
        args_47 = float_48 * getitem_710
        float_48 = getitem_710 = None
        cos_71 = torch.cos(args_47)
        sin_71 = torch.sin(args_47)
        args_47 = None
        embedding_47 = torch.cat([cos_71, sin_71], dim=-1)
        cos_71 = sin_71 = None
        input_941 = torch._C._nn.linear(
            embedding_47,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_47 = None
        input_942 = torch.nn.functional.silu(input_941, inplace=False)
        input_941 = None
        input_943 = torch._C._nn.linear(
            input_942,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_942 = None
        c_47 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y_47 = input_943 + c_47
        input_943 = c_47 = None
        input_944 = torch.nn.functional.silu(y_47, inplace=False)
        input_945 = torch._C._nn.linear(
            input_944,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_944 = None
        chunk_188 = input_945.chunk(3, dim=-1)
        input_945 = None
        shift_mlp_141 = chunk_188[0]
        scale_mlp_141 = chunk_188[1]
        gate_mlp_141 = chunk_188[2]
        chunk_188 = None
        layer_norm_213 = torch.nn.functional.layer_norm(
            x_332,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_662 = 1 + scale_mlp_141
        scale_mlp_141 = None
        mul_628 = layer_norm_213 * add_662
        layer_norm_213 = add_662 = None
        h_141 = mul_628 + shift_mlp_141
        mul_628 = shift_mlp_141 = None
        input_946 = torch._C._nn.linear(
            h_141,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_141 = None
        input_947 = torch.nn.functional.silu(input_946, inplace=False)
        input_946 = None
        input_948 = torch._C._nn.linear(
            input_947,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_947 = None
        mul_629 = gate_mlp_141 * input_948
        gate_mlp_141 = input_948 = None
        x_333 = x_332 + mul_629
        x_332 = mul_629 = None
        input_949 = torch.nn.functional.silu(y_47, inplace=False)
        input_950 = torch._C._nn.linear(
            input_949,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_949 = None
        chunk_189 = input_950.chunk(3, dim=-1)
        input_950 = None
        shift_mlp_142 = chunk_189[0]
        scale_mlp_142 = chunk_189[1]
        gate_mlp_142 = chunk_189[2]
        chunk_189 = None
        layer_norm_214 = torch.nn.functional.layer_norm(
            x_333,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_665 = 1 + scale_mlp_142
        scale_mlp_142 = None
        mul_630 = layer_norm_214 * add_665
        layer_norm_214 = add_665 = None
        h_142 = mul_630 + shift_mlp_142
        mul_630 = shift_mlp_142 = None
        input_951 = torch._C._nn.linear(
            h_142,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_142 = None
        input_952 = torch.nn.functional.silu(input_951, inplace=False)
        input_951 = None
        input_953 = torch._C._nn.linear(
            input_952,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_952 = None
        mul_631 = gate_mlp_142 * input_953
        gate_mlp_142 = input_953 = None
        x_334 = x_333 + mul_631
        x_333 = mul_631 = None
        input_954 = torch.nn.functional.silu(y_47, inplace=False)
        input_955 = torch._C._nn.linear(
            input_954,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_954 = None
        chunk_190 = input_955.chunk(3, dim=-1)
        input_955 = None
        shift_mlp_143 = chunk_190[0]
        scale_mlp_143 = chunk_190[1]
        gate_mlp_143 = chunk_190[2]
        chunk_190 = None
        layer_norm_215 = torch.nn.functional.layer_norm(
            x_334,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_668 = 1 + scale_mlp_143
        scale_mlp_143 = None
        mul_632 = layer_norm_215 * add_668
        layer_norm_215 = add_668 = None
        h_143 = mul_632 + shift_mlp_143
        mul_632 = shift_mlp_143 = None
        input_956 = torch._C._nn.linear(
            h_143,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_143 = None
        input_957 = torch.nn.functional.silu(input_956, inplace=False)
        input_956 = None
        input_958 = torch._C._nn.linear(
            input_957,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_957 = None
        mul_633 = gate_mlp_143 * input_958
        gate_mlp_143 = input_958 = None
        x_335 = x_334 + mul_633
        x_334 = mul_633 = None
        input_959 = torch.nn.functional.silu(y_47, inplace=False)
        y_47 = None
        input_960 = torch._C._nn.linear(
            input_959,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_959 = None
        chunk_191 = input_960.chunk(2, dim=-1)
        input_960 = None
        shift_47 = chunk_191[0]
        scale_47 = chunk_191[1]
        chunk_191 = None
        layer_norm_216 = torch.nn.functional.layer_norm(
            x_335, (768,), None, None, 1e-06
        )
        x_335 = None
        add_671 = 1 + scale_47
        scale_47 = None
        mul_634 = layer_norm_216 * add_671
        layer_norm_216 = add_671 = None
        x_336 = mul_634 + shift_47
        mul_634 = shift_47 = None
        x_337 = torch._C._nn.linear(
            x_336,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_336 = None
        sub_47 = x_337 - noise
        x_337 = None
        mul_635 = sub_47 * 0.02
        sub_47 = None
        x_338 = x_331 + mul_635
        x_331 = mul_635 = None
        ones_48 = torch.ones(1)
        mul_636 = ones_48 * 48
        ones_48 = None
        truediv_96 = mul_636 / 50
        mul_636 = None
        t_48 = truediv_96.to(device(type="cuda", index=0))
        truediv_96 = None
        mul_637 = t_48 * 1000
        t_48 = None
        x_339 = torch._C._nn.linear(
            x_338,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        arange_50 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_638 = -9.210340371976184 * arange_50
        arange_50 = None
        truediv_97 = mul_638 / 128
        mul_638 = None
        exp_48 = torch.exp(truediv_97)
        truediv_97 = None
        freqs_48 = exp_48.to(device=device(type="cuda", index=0))
        exp_48 = None
        getitem_722 = mul_637[(slice(None, None, None), None)]
        mul_637 = None
        float_49 = getitem_722.float()
        getitem_722 = None
        getitem_723 = freqs_48[None]
        freqs_48 = None
        args_48 = float_49 * getitem_723
        float_49 = getitem_723 = None
        cos_72 = torch.cos(args_48)
        sin_72 = torch.sin(args_48)
        args_48 = None
        embedding_48 = torch.cat([cos_72, sin_72], dim=-1)
        cos_72 = sin_72 = None
        input_961 = torch._C._nn.linear(
            embedding_48,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_48 = None
        input_962 = torch.nn.functional.silu(input_961, inplace=False)
        input_961 = None
        input_963 = torch._C._nn.linear(
            input_962,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_962 = None
        c_48 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        y_48 = input_963 + c_48
        input_963 = c_48 = None
        input_964 = torch.nn.functional.silu(y_48, inplace=False)
        input_965 = torch._C._nn.linear(
            input_964,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_964 = None
        chunk_192 = input_965.chunk(3, dim=-1)
        input_965 = None
        shift_mlp_144 = chunk_192[0]
        scale_mlp_144 = chunk_192[1]
        gate_mlp_144 = chunk_192[2]
        chunk_192 = None
        layer_norm_217 = torch.nn.functional.layer_norm(
            x_339,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_675 = 1 + scale_mlp_144
        scale_mlp_144 = None
        mul_640 = layer_norm_217 * add_675
        layer_norm_217 = add_675 = None
        h_144 = mul_640 + shift_mlp_144
        mul_640 = shift_mlp_144 = None
        input_966 = torch._C._nn.linear(
            h_144,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_144 = None
        input_967 = torch.nn.functional.silu(input_966, inplace=False)
        input_966 = None
        input_968 = torch._C._nn.linear(
            input_967,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_967 = None
        mul_641 = gate_mlp_144 * input_968
        gate_mlp_144 = input_968 = None
        x_340 = x_339 + mul_641
        x_339 = mul_641 = None
        input_969 = torch.nn.functional.silu(y_48, inplace=False)
        input_970 = torch._C._nn.linear(
            input_969,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_969 = None
        chunk_193 = input_970.chunk(3, dim=-1)
        input_970 = None
        shift_mlp_145 = chunk_193[0]
        scale_mlp_145 = chunk_193[1]
        gate_mlp_145 = chunk_193[2]
        chunk_193 = None
        layer_norm_218 = torch.nn.functional.layer_norm(
            x_340,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_678 = 1 + scale_mlp_145
        scale_mlp_145 = None
        mul_642 = layer_norm_218 * add_678
        layer_norm_218 = add_678 = None
        h_145 = mul_642 + shift_mlp_145
        mul_642 = shift_mlp_145 = None
        input_971 = torch._C._nn.linear(
            h_145,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_145 = None
        input_972 = torch.nn.functional.silu(input_971, inplace=False)
        input_971 = None
        input_973 = torch._C._nn.linear(
            input_972,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_972 = None
        mul_643 = gate_mlp_145 * input_973
        gate_mlp_145 = input_973 = None
        x_341 = x_340 + mul_643
        x_340 = mul_643 = None
        input_974 = torch.nn.functional.silu(y_48, inplace=False)
        input_975 = torch._C._nn.linear(
            input_974,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_974 = None
        chunk_194 = input_975.chunk(3, dim=-1)
        input_975 = None
        shift_mlp_146 = chunk_194[0]
        scale_mlp_146 = chunk_194[1]
        gate_mlp_146 = chunk_194[2]
        chunk_194 = None
        layer_norm_219 = torch.nn.functional.layer_norm(
            x_341,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        add_681 = 1 + scale_mlp_146
        scale_mlp_146 = None
        mul_644 = layer_norm_219 * add_681
        layer_norm_219 = add_681 = None
        h_146 = mul_644 + shift_mlp_146
        mul_644 = shift_mlp_146 = None
        input_976 = torch._C._nn.linear(
            h_146,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_146 = None
        input_977 = torch.nn.functional.silu(input_976, inplace=False)
        input_976 = None
        input_978 = torch._C._nn.linear(
            input_977,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_977 = None
        mul_645 = gate_mlp_146 * input_978
        gate_mlp_146 = input_978 = None
        x_342 = x_341 + mul_645
        x_341 = mul_645 = None
        input_979 = torch.nn.functional.silu(y_48, inplace=False)
        y_48 = None
        input_980 = torch._C._nn.linear(
            input_979,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_979 = None
        chunk_195 = input_980.chunk(2, dim=-1)
        input_980 = None
        shift_48 = chunk_195[0]
        scale_48 = chunk_195[1]
        chunk_195 = None
        layer_norm_220 = torch.nn.functional.layer_norm(
            x_342, (768,), None, None, 1e-06
        )
        x_342 = None
        add_684 = 1 + scale_48
        scale_48 = None
        mul_646 = layer_norm_220 * add_684
        layer_norm_220 = add_684 = None
        x_343 = mul_646 + shift_48
        mul_646 = shift_48 = None
        x_344 = torch._C._nn.linear(
            x_343,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_343 = None
        sub_48 = x_344 - noise
        x_344 = None
        mul_647 = sub_48 * 0.02
        sub_48 = None
        x_345 = x_338 + mul_647
        x_338 = mul_647 = None
        ones_49 = torch.ones(1)
        mul_648 = ones_49 * 49
        ones_49 = None
        truediv_98 = mul_648 / 50
        mul_648 = None
        t_49 = truediv_98.to(device(type="cuda", index=0))
        truediv_98 = None
        mul_649 = t_49 * 1000
        t_49 = None
        x_346 = torch._C._nn.linear(
            x_345,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_,
        )
        l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_ = (
            l_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_
        ) = None
        arange_51 = torch.arange(start=0, end=128, dtype=torch.float32)
        mul_650 = -9.210340371976184 * arange_51
        arange_51 = None
        truediv_99 = mul_650 / 128
        mul_650 = None
        exp_49 = torch.exp(truediv_99)
        truediv_99 = None
        freqs_49 = exp_49.to(device=device(type="cuda", index=0))
        exp_49 = None
        getitem_735 = mul_649[(slice(None, None, None), None)]
        mul_649 = None
        float_50 = getitem_735.float()
        getitem_735 = None
        getitem_736 = freqs_49[None]
        freqs_49 = None
        args_49 = float_50 * getitem_736
        float_50 = getitem_736 = None
        cos_73 = torch.cos(args_49)
        sin_73 = torch.sin(args_49)
        args_49 = None
        embedding_49 = torch.cat([cos_73, sin_73], dim=-1)
        cos_73 = sin_73 = None
        input_981 = torch._C._nn.linear(
            embedding_49,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_,
        )
        embedding_49 = l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_ = l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_ = (None)
        input_982 = torch.nn.functional.silu(input_981, inplace=False)
        input_981 = None
        input_983 = torch._C._nn.linear(
            input_982,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_,
        )
        input_982 = l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_ = l_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_ = (None)
        c_49 = torch._C._nn.linear(
            z,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_,
        )
        z = (
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_
        ) = (
            l_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_
        ) = None
        y_49 = input_983 + c_49
        input_983 = c_49 = None
        input_984 = torch.nn.functional.silu(y_49, inplace=False)
        input_985 = torch._C._nn.linear(
            input_984,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_984 = l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_weight_ = l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_ada_ln_modulation_modules_1_parameters_bias_ = (None)
        chunk_196 = input_985.chunk(3, dim=-1)
        input_985 = None
        shift_mlp_147 = chunk_196[0]
        scale_mlp_147 = chunk_196[1]
        gate_mlp_147 = chunk_196[2]
        chunk_196 = None
        layer_norm_221 = torch.nn.functional.layer_norm(
            x_346,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_,
            1e-06,
        )
        l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_ = l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_ = (None)
        add_688 = 1 + scale_mlp_147
        scale_mlp_147 = None
        mul_652 = layer_norm_221 * add_688
        layer_norm_221 = add_688 = None
        h_147 = mul_652 + shift_mlp_147
        mul_652 = shift_mlp_147 = None
        input_986 = torch._C._nn.linear(
            h_147,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        h_147 = l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_ = l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_ = (None)
        input_987 = torch.nn.functional.silu(input_986, inplace=False)
        input_986 = None
        input_988 = torch._C._nn.linear(
            input_987,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_,
        )
        input_987 = l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_ = l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_ = (None)
        mul_653 = gate_mlp_147 * input_988
        gate_mlp_147 = input_988 = None
        x_347 = x_346 + mul_653
        x_346 = mul_653 = None
        input_989 = torch.nn.functional.silu(y_49, inplace=False)
        input_990 = torch._C._nn.linear(
            input_989,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_989 = l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_weight_ = l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_ada_ln_modulation_modules_1_parameters_bias_ = (None)
        chunk_197 = input_990.chunk(3, dim=-1)
        input_990 = None
        shift_mlp_148 = chunk_197[0]
        scale_mlp_148 = chunk_197[1]
        gate_mlp_148 = chunk_197[2]
        chunk_197 = None
        layer_norm_222 = torch.nn.functional.layer_norm(
            x_347,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_,
            1e-06,
        )
        l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_ = l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_ = (None)
        add_691 = 1 + scale_mlp_148
        scale_mlp_148 = None
        mul_654 = layer_norm_222 * add_691
        layer_norm_222 = add_691 = None
        h_148 = mul_654 + shift_mlp_148
        mul_654 = shift_mlp_148 = None
        input_991 = torch._C._nn.linear(
            h_148,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        h_148 = l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_ = l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_ = (None)
        input_992 = torch.nn.functional.silu(input_991, inplace=False)
        input_991 = None
        input_993 = torch._C._nn.linear(
            input_992,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_,
        )
        input_992 = l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_ = l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_ = (None)
        mul_655 = gate_mlp_148 * input_993
        gate_mlp_148 = input_993 = None
        x_348 = x_347 + mul_655
        x_347 = mul_655 = None
        input_994 = torch.nn.functional.silu(y_49, inplace=False)
        input_995 = torch._C._nn.linear(
            input_994,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_994 = l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_weight_ = l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_ada_ln_modulation_modules_1_parameters_bias_ = (None)
        chunk_198 = input_995.chunk(3, dim=-1)
        input_995 = None
        shift_mlp_149 = chunk_198[0]
        scale_mlp_149 = chunk_198[1]
        gate_mlp_149 = chunk_198[2]
        chunk_198 = None
        layer_norm_223 = torch.nn.functional.layer_norm(
            x_348,
            (768,),
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_,
            1e-06,
        )
        l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_ = l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_ = (None)
        add_694 = 1 + scale_mlp_149
        scale_mlp_149 = None
        mul_656 = layer_norm_223 * add_694
        layer_norm_223 = add_694 = None
        h_149 = mul_656 + shift_mlp_149
        mul_656 = shift_mlp_149 = None
        input_996 = torch._C._nn.linear(
            h_149,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_,
        )
        h_149 = l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_ = l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_ = (None)
        input_997 = torch.nn.functional.silu(input_996, inplace=False)
        input_996 = None
        input_998 = torch._C._nn.linear(
            input_997,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_,
        )
        input_997 = l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_ = l_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_ = (None)
        mul_657 = gate_mlp_149 * input_998
        gate_mlp_149 = input_998 = None
        x_349 = x_348 + mul_657
        x_348 = mul_657 = None
        input_999 = torch.nn.functional.silu(y_49, inplace=False)
        y_49 = None
        input_1000 = torch._C._nn.linear(
            input_999,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_,
        )
        input_999 = l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_weight_ = l_self_modules_flow_loss_modules_net_modules_final_layer_modules_ada_ln_modulation_modules_1_parameters_bias_ = (None)
        chunk_199 = input_1000.chunk(2, dim=-1)
        input_1000 = None
        shift_49 = chunk_199[0]
        scale_49 = chunk_199[1]
        chunk_199 = None
        layer_norm_224 = torch.nn.functional.layer_norm(
            x_349, (768,), None, None, 1e-06
        )
        x_349 = None
        add_697 = 1 + scale_49
        scale_49 = None
        mul_658 = layer_norm_224 * add_697
        layer_norm_224 = add_697 = None
        x_350 = mul_658 + shift_49
        mul_658 = shift_49 = None
        x_351 = torch._C._nn.linear(
            x_350,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_,
            l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_,
        )
        x_350 = l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_ = l_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_ = (None)
        sub_49 = x_351 - noise
        x_351 = noise = None
        mul_659 = sub_49 * 0.02
        sub_49 = None
        x_352 = x_345 + mul_659
        x_345 = mul_659 = None
        reshape_12 = x_352.reshape(1, -1, 720)
        x_352 = None
        x_353 = reshape_12.transpose(0, 1)
        reshape_12 = None
        return (x_353,)
