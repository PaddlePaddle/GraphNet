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
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
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
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_
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
        cache_position = torch.arange(0, 32, device=device(type="cuda", index=0))
        position_ids = cache_position.unsqueeze(0)
        attention_mask = l_kwargs_attention_mask_.to(
            device=device(type="cuda", index=0), dtype=torch.bool
        )
        l_kwargs_attention_mask_ = None
        kv_arange = torch.arange(32, device=device(type="cuda", index=0))
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
            torch._functorch.predispatch._vmap_increment_nesting(32, "error")
        )
        _vmap_increment_nesting_2 = None
        child_2 = torch._functorch.predispatch._add_batch_dim(cache_position, 0, 3)
        cache_position = None
        lazy_load_decompositions_3 = (
            torch._functorch.predispatch.lazy_load_decompositions()
        )
        lazy_load_decompositions_3 = None
        _vmap_increment_nesting_3 = (
            torch._functorch.predispatch._vmap_increment_nesting(32, "error")
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
            result_2, 4, 32, 0
        )
        result_2 = None
        _vmap_decrement_nesting = torch._functorch.predispatch._vmap_decrement_nesting()
        _vmap_decrement_nesting = None
        batched_outputs_1 = torch._functorch.predispatch._remove_batch_dim(
            batched_outputs, 3, 32, 0
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
        cos_2 = cos_1.to(dtype=torch.float32)
        cos_1 = None
        sin_2 = sin_1.to(dtype=torch.float32)
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
        to_7 = hidden_states_1.to(torch.float32)
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
        view = linear.view((1, 32, -1, 64))
        linear = None
        query_states = view.transpose(1, 2)
        view = None
        linear_1 = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_1 = linear_1.view((1, 32, -1, 64))
        linear_1 = None
        key_states = view_1.transpose(1, 2)
        view_1 = None
        linear_2 = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_2 = l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_2 = linear_2.view((1, 32, -1, 64))
        linear_2 = None
        value_states = view_2.transpose(1, 2)
        view_2 = None
        cos_3 = cos_2.unsqueeze(1)
        sin_3 = sin_2.unsqueeze(1)
        mul_4 = query_states * cos_3
        x1 = query_states[(Ellipsis, slice(None, 32, None))]
        x2 = query_states[(Ellipsis, slice(32, None, None))]
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
        x1_1 = key_states[(Ellipsis, slice(None, 32, None))]
        x2_1 = key_states[(Ellipsis, slice(32, None, None))]
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
        hidden_states_3 = getitem_6.expand(1, 4, 3, 32, 64)
        getitem_6 = None
        key = hidden_states_3.reshape(1, 12, 32, 64)
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
        hidden_states_4 = getitem_7.expand(1, 4, 3, 32, 64)
        getitem_7 = None
        value = hidden_states_4.reshape(1, 12, 32, 64)
        hidden_states_4 = None
        attention_mask_1 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 32, None),
            )
        ]
        attn_output = torch._C._nn.scaled_dot_product_attention(
            q_embed,
            key,
            value,
            attn_mask=attention_mask_1,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        q_embed = key = value = attention_mask_1 = None
        transpose_4 = attn_output.transpose(1, 2)
        attn_output = None
        attn_output_1 = transpose_4.contiguous()
        transpose_4 = None
        reshape_2 = attn_output_1.reshape(1, 32, -1)
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
        to_9 = hidden_states_7.to(torch.float32)
        hidden_states_7 = None
        hidden_states_8 = (
            l_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_
            * to_9
        )
        l_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_ = (
            to_9
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
        to_11 = hidden_states_11.to(torch.float32)
        hidden_states_11 = None
        hidden_states_12 = (
            l_self_modules_model_modules_layers_modules_1_modules_input_layernorm_parameters_weight_
            * to_11
        )
        l_self_modules_model_modules_layers_modules_1_modules_input_layernorm_parameters_weight_ = (
            to_11
        ) = None
        linear_7 = torch._C._nn.linear(
            hidden_states_12,
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_3 = linear_7.view((1, 32, -1, 64))
        linear_7 = None
        query_states_1 = view_3.transpose(1, 2)
        view_3 = None
        linear_8 = torch._C._nn.linear(
            hidden_states_12,
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_4 = linear_8.view((1, 32, -1, 64))
        linear_8 = None
        key_states_1 = view_4.transpose(1, 2)
        view_4 = None
        linear_9 = torch._C._nn.linear(
            hidden_states_12,
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_12 = l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_5 = linear_9.view((1, 32, -1, 64))
        linear_9 = None
        value_states_1 = view_5.transpose(1, 2)
        view_5 = None
        cos_4 = cos_2.unsqueeze(1)
        sin_4 = sin_2.unsqueeze(1)
        mul_13 = query_states_1 * cos_4
        x1_2 = query_states_1[(Ellipsis, slice(None, 32, None))]
        x2_2 = query_states_1[(Ellipsis, slice(32, None, None))]
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
        x1_3 = key_states_1[(Ellipsis, slice(None, 32, None))]
        x2_3 = key_states_1[(Ellipsis, slice(32, None, None))]
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
        hidden_states_13 = getitem_13.expand(1, 4, 3, 32, 64)
        getitem_13 = None
        key_1 = hidden_states_13.reshape(1, 12, 32, 64)
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
        hidden_states_14 = getitem_14.expand(1, 4, 3, 32, 64)
        getitem_14 = None
        value_1 = hidden_states_14.reshape(1, 12, 32, 64)
        hidden_states_14 = None
        attention_mask_2 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 32, None),
            )
        ]
        attn_output_4 = torch._C._nn.scaled_dot_product_attention(
            q_embed_1,
            key_1,
            value_1,
            attn_mask=attention_mask_2,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        q_embed_1 = key_1 = value_1 = attention_mask_2 = None
        transpose_8 = attn_output_4.transpose(1, 2)
        attn_output_4 = None
        attn_output_5 = transpose_8.contiguous()
        transpose_8 = None
        reshape_5 = attn_output_5.reshape(1, 32, -1)
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
        to_13 = hidden_states_17.to(torch.float32)
        hidden_states_17 = None
        hidden_states_18 = (
            l_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_
            * to_13
        )
        l_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_ = (
            to_13
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
        to_15 = hidden_states_21.to(torch.float32)
        hidden_states_21 = None
        hidden_states_22 = (
            l_self_modules_model_modules_layers_modules_2_modules_input_layernorm_parameters_weight_
            * to_15
        )
        l_self_modules_model_modules_layers_modules_2_modules_input_layernorm_parameters_weight_ = (
            to_15
        ) = None
        linear_14 = torch._C._nn.linear(
            hidden_states_22,
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_6 = linear_14.view((1, 32, -1, 64))
        linear_14 = None
        query_states_2 = view_6.transpose(1, 2)
        view_6 = None
        linear_15 = torch._C._nn.linear(
            hidden_states_22,
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_7 = linear_15.view((1, 32, -1, 64))
        linear_15 = None
        key_states_2 = view_7.transpose(1, 2)
        view_7 = None
        linear_16 = torch._C._nn.linear(
            hidden_states_22,
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_22 = l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_8 = linear_16.view((1, 32, -1, 64))
        linear_16 = None
        value_states_2 = view_8.transpose(1, 2)
        view_8 = None
        cos_5 = cos_2.unsqueeze(1)
        sin_5 = sin_2.unsqueeze(1)
        mul_22 = query_states_2 * cos_5
        x1_4 = query_states_2[(Ellipsis, slice(None, 32, None))]
        x2_4 = query_states_2[(Ellipsis, slice(32, None, None))]
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
        x1_5 = key_states_2[(Ellipsis, slice(None, 32, None))]
        x2_5 = key_states_2[(Ellipsis, slice(32, None, None))]
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
        hidden_states_23 = getitem_20.expand(1, 4, 3, 32, 64)
        getitem_20 = None
        key_2 = hidden_states_23.reshape(1, 12, 32, 64)
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
        hidden_states_24 = getitem_21.expand(1, 4, 3, 32, 64)
        getitem_21 = None
        value_2 = hidden_states_24.reshape(1, 12, 32, 64)
        hidden_states_24 = None
        attention_mask_3 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 32, None),
            )
        ]
        attn_output_8 = torch._C._nn.scaled_dot_product_attention(
            q_embed_2,
            key_2,
            value_2,
            attn_mask=attention_mask_3,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        q_embed_2 = key_2 = value_2 = attention_mask_3 = None
        transpose_12 = attn_output_8.transpose(1, 2)
        attn_output_8 = None
        attn_output_9 = transpose_12.contiguous()
        transpose_12 = None
        reshape_8 = attn_output_9.reshape(1, 32, -1)
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
        to_17 = hidden_states_27.to(torch.float32)
        hidden_states_27 = None
        hidden_states_28 = (
            l_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_
            * to_17
        )
        l_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_ = (
            to_17
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
        to_19 = hidden_states_31.to(torch.float32)
        hidden_states_31 = None
        hidden_states_32 = (
            l_self_modules_model_modules_layers_modules_3_modules_input_layernorm_parameters_weight_
            * to_19
        )
        l_self_modules_model_modules_layers_modules_3_modules_input_layernorm_parameters_weight_ = (
            to_19
        ) = None
        linear_21 = torch._C._nn.linear(
            hidden_states_32,
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_9 = linear_21.view((1, 32, -1, 64))
        linear_21 = None
        query_states_3 = view_9.transpose(1, 2)
        view_9 = None
        linear_22 = torch._C._nn.linear(
            hidden_states_32,
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_10 = linear_22.view((1, 32, -1, 64))
        linear_22 = None
        key_states_3 = view_10.transpose(1, 2)
        view_10 = None
        linear_23 = torch._C._nn.linear(
            hidden_states_32,
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_32 = l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_11 = linear_23.view((1, 32, -1, 64))
        linear_23 = None
        value_states_3 = view_11.transpose(1, 2)
        view_11 = None
        cos_6 = cos_2.unsqueeze(1)
        sin_6 = sin_2.unsqueeze(1)
        mul_31 = query_states_3 * cos_6
        x1_6 = query_states_3[(Ellipsis, slice(None, 32, None))]
        x2_6 = query_states_3[(Ellipsis, slice(32, None, None))]
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
        x1_7 = key_states_3[(Ellipsis, slice(None, 32, None))]
        x2_7 = key_states_3[(Ellipsis, slice(32, None, None))]
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
        hidden_states_33 = getitem_27.expand(1, 4, 3, 32, 64)
        getitem_27 = None
        key_3 = hidden_states_33.reshape(1, 12, 32, 64)
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
        hidden_states_34 = getitem_28.expand(1, 4, 3, 32, 64)
        getitem_28 = None
        value_3 = hidden_states_34.reshape(1, 12, 32, 64)
        hidden_states_34 = None
        attention_mask_4 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 32, None),
            )
        ]
        attn_output_12 = torch._C._nn.scaled_dot_product_attention(
            q_embed_3,
            key_3,
            value_3,
            attn_mask=attention_mask_4,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        q_embed_3 = key_3 = value_3 = attention_mask_4 = None
        transpose_16 = attn_output_12.transpose(1, 2)
        attn_output_12 = None
        attn_output_13 = transpose_16.contiguous()
        transpose_16 = None
        reshape_11 = attn_output_13.reshape(1, 32, -1)
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
        to_21 = hidden_states_37.to(torch.float32)
        hidden_states_37 = None
        hidden_states_38 = (
            l_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_
            * to_21
        )
        l_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_ = (
            to_21
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
        to_23 = hidden_states_41.to(torch.float32)
        hidden_states_41 = None
        hidden_states_42 = (
            l_self_modules_model_modules_layers_modules_4_modules_input_layernorm_parameters_weight_
            * to_23
        )
        l_self_modules_model_modules_layers_modules_4_modules_input_layernorm_parameters_weight_ = (
            to_23
        ) = None
        linear_28 = torch._C._nn.linear(
            hidden_states_42,
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_12 = linear_28.view((1, 32, -1, 64))
        linear_28 = None
        query_states_4 = view_12.transpose(1, 2)
        view_12 = None
        linear_29 = torch._C._nn.linear(
            hidden_states_42,
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_13 = linear_29.view((1, 32, -1, 64))
        linear_29 = None
        key_states_4 = view_13.transpose(1, 2)
        view_13 = None
        linear_30 = torch._C._nn.linear(
            hidden_states_42,
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_42 = l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_14 = linear_30.view((1, 32, -1, 64))
        linear_30 = None
        value_states_4 = view_14.transpose(1, 2)
        view_14 = None
        cos_7 = cos_2.unsqueeze(1)
        sin_7 = sin_2.unsqueeze(1)
        mul_40 = query_states_4 * cos_7
        x1_8 = query_states_4[(Ellipsis, slice(None, 32, None))]
        x2_8 = query_states_4[(Ellipsis, slice(32, None, None))]
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
        x1_9 = key_states_4[(Ellipsis, slice(None, 32, None))]
        x2_9 = key_states_4[(Ellipsis, slice(32, None, None))]
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
        hidden_states_43 = getitem_34.expand(1, 4, 3, 32, 64)
        getitem_34 = None
        key_4 = hidden_states_43.reshape(1, 12, 32, 64)
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
        hidden_states_44 = getitem_35.expand(1, 4, 3, 32, 64)
        getitem_35 = None
        value_4 = hidden_states_44.reshape(1, 12, 32, 64)
        hidden_states_44 = None
        attention_mask_5 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 32, None),
            )
        ]
        attn_output_16 = torch._C._nn.scaled_dot_product_attention(
            q_embed_4,
            key_4,
            value_4,
            attn_mask=attention_mask_5,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        q_embed_4 = key_4 = value_4 = attention_mask_5 = None
        transpose_20 = attn_output_16.transpose(1, 2)
        attn_output_16 = None
        attn_output_17 = transpose_20.contiguous()
        transpose_20 = None
        reshape_14 = attn_output_17.reshape(1, 32, -1)
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
        to_25 = hidden_states_47.to(torch.float32)
        hidden_states_47 = None
        hidden_states_48 = (
            l_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_
            * to_25
        )
        l_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_ = (
            to_25
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
        to_27 = hidden_states_51.to(torch.float32)
        hidden_states_51 = None
        hidden_states_52 = (
            l_self_modules_model_modules_layers_modules_5_modules_input_layernorm_parameters_weight_
            * to_27
        )
        l_self_modules_model_modules_layers_modules_5_modules_input_layernorm_parameters_weight_ = (
            to_27
        ) = None
        linear_35 = torch._C._nn.linear(
            hidden_states_52,
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_15 = linear_35.view((1, 32, -1, 64))
        linear_35 = None
        query_states_5 = view_15.transpose(1, 2)
        view_15 = None
        linear_36 = torch._C._nn.linear(
            hidden_states_52,
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_16 = linear_36.view((1, 32, -1, 64))
        linear_36 = None
        key_states_5 = view_16.transpose(1, 2)
        view_16 = None
        linear_37 = torch._C._nn.linear(
            hidden_states_52,
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_52 = l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_17 = linear_37.view((1, 32, -1, 64))
        linear_37 = None
        value_states_5 = view_17.transpose(1, 2)
        view_17 = None
        cos_8 = cos_2.unsqueeze(1)
        sin_8 = sin_2.unsqueeze(1)
        mul_49 = query_states_5 * cos_8
        x1_10 = query_states_5[(Ellipsis, slice(None, 32, None))]
        x2_10 = query_states_5[(Ellipsis, slice(32, None, None))]
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
        x1_11 = key_states_5[(Ellipsis, slice(None, 32, None))]
        x2_11 = key_states_5[(Ellipsis, slice(32, None, None))]
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
        hidden_states_53 = getitem_41.expand(1, 4, 3, 32, 64)
        getitem_41 = None
        key_5 = hidden_states_53.reshape(1, 12, 32, 64)
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
        hidden_states_54 = getitem_42.expand(1, 4, 3, 32, 64)
        getitem_42 = None
        value_5 = hidden_states_54.reshape(1, 12, 32, 64)
        hidden_states_54 = None
        attention_mask_6 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 32, None),
            )
        ]
        attn_output_20 = torch._C._nn.scaled_dot_product_attention(
            q_embed_5,
            key_5,
            value_5,
            attn_mask=attention_mask_6,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        q_embed_5 = key_5 = value_5 = attention_mask_6 = None
        transpose_24 = attn_output_20.transpose(1, 2)
        attn_output_20 = None
        attn_output_21 = transpose_24.contiguous()
        transpose_24 = None
        reshape_17 = attn_output_21.reshape(1, 32, -1)
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
        to_29 = hidden_states_57.to(torch.float32)
        hidden_states_57 = None
        hidden_states_58 = (
            l_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_
            * to_29
        )
        l_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_ = (
            to_29
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
        to_31 = hidden_states_61.to(torch.float32)
        hidden_states_61 = None
        hidden_states_62 = (
            l_self_modules_model_modules_layers_modules_6_modules_input_layernorm_parameters_weight_
            * to_31
        )
        l_self_modules_model_modules_layers_modules_6_modules_input_layernorm_parameters_weight_ = (
            to_31
        ) = None
        linear_42 = torch._C._nn.linear(
            hidden_states_62,
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_18 = linear_42.view((1, 32, -1, 64))
        linear_42 = None
        query_states_6 = view_18.transpose(1, 2)
        view_18 = None
        linear_43 = torch._C._nn.linear(
            hidden_states_62,
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_19 = linear_43.view((1, 32, -1, 64))
        linear_43 = None
        key_states_6 = view_19.transpose(1, 2)
        view_19 = None
        linear_44 = torch._C._nn.linear(
            hidden_states_62,
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_62 = l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_20 = linear_44.view((1, 32, -1, 64))
        linear_44 = None
        value_states_6 = view_20.transpose(1, 2)
        view_20 = None
        cos_9 = cos_2.unsqueeze(1)
        sin_9 = sin_2.unsqueeze(1)
        mul_58 = query_states_6 * cos_9
        x1_12 = query_states_6[(Ellipsis, slice(None, 32, None))]
        x2_12 = query_states_6[(Ellipsis, slice(32, None, None))]
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
        x1_13 = key_states_6[(Ellipsis, slice(None, 32, None))]
        x2_13 = key_states_6[(Ellipsis, slice(32, None, None))]
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
        hidden_states_63 = getitem_48.expand(1, 4, 3, 32, 64)
        getitem_48 = None
        key_6 = hidden_states_63.reshape(1, 12, 32, 64)
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
        hidden_states_64 = getitem_49.expand(1, 4, 3, 32, 64)
        getitem_49 = None
        value_6 = hidden_states_64.reshape(1, 12, 32, 64)
        hidden_states_64 = None
        attention_mask_7 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 32, None),
            )
        ]
        attn_output_24 = torch._C._nn.scaled_dot_product_attention(
            q_embed_6,
            key_6,
            value_6,
            attn_mask=attention_mask_7,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        q_embed_6 = key_6 = value_6 = attention_mask_7 = None
        transpose_28 = attn_output_24.transpose(1, 2)
        attn_output_24 = None
        attn_output_25 = transpose_28.contiguous()
        transpose_28 = None
        reshape_20 = attn_output_25.reshape(1, 32, -1)
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
        to_33 = hidden_states_67.to(torch.float32)
        hidden_states_67 = None
        hidden_states_68 = (
            l_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_
            * to_33
        )
        l_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_ = (
            to_33
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
        to_35 = hidden_states_71.to(torch.float32)
        hidden_states_71 = None
        hidden_states_72 = (
            l_self_modules_model_modules_layers_modules_7_modules_input_layernorm_parameters_weight_
            * to_35
        )
        l_self_modules_model_modules_layers_modules_7_modules_input_layernorm_parameters_weight_ = (
            to_35
        ) = None
        linear_49 = torch._C._nn.linear(
            hidden_states_72,
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_21 = linear_49.view((1, 32, -1, 64))
        linear_49 = None
        query_states_7 = view_21.transpose(1, 2)
        view_21 = None
        linear_50 = torch._C._nn.linear(
            hidden_states_72,
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_22 = linear_50.view((1, 32, -1, 64))
        linear_50 = None
        key_states_7 = view_22.transpose(1, 2)
        view_22 = None
        linear_51 = torch._C._nn.linear(
            hidden_states_72,
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_72 = l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_23 = linear_51.view((1, 32, -1, 64))
        linear_51 = None
        value_states_7 = view_23.transpose(1, 2)
        view_23 = None
        cos_10 = cos_2.unsqueeze(1)
        sin_10 = sin_2.unsqueeze(1)
        mul_67 = query_states_7 * cos_10
        x1_14 = query_states_7[(Ellipsis, slice(None, 32, None))]
        x2_14 = query_states_7[(Ellipsis, slice(32, None, None))]
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
        x1_15 = key_states_7[(Ellipsis, slice(None, 32, None))]
        x2_15 = key_states_7[(Ellipsis, slice(32, None, None))]
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
        hidden_states_73 = getitem_55.expand(1, 4, 3, 32, 64)
        getitem_55 = None
        key_7 = hidden_states_73.reshape(1, 12, 32, 64)
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
        hidden_states_74 = getitem_56.expand(1, 4, 3, 32, 64)
        getitem_56 = None
        value_7 = hidden_states_74.reshape(1, 12, 32, 64)
        hidden_states_74 = None
        attention_mask_8 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 32, None),
            )
        ]
        attn_output_28 = torch._C._nn.scaled_dot_product_attention(
            q_embed_7,
            key_7,
            value_7,
            attn_mask=attention_mask_8,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        q_embed_7 = key_7 = value_7 = attention_mask_8 = None
        transpose_32 = attn_output_28.transpose(1, 2)
        attn_output_28 = None
        attn_output_29 = transpose_32.contiguous()
        transpose_32 = None
        reshape_23 = attn_output_29.reshape(1, 32, -1)
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
        to_37 = hidden_states_77.to(torch.float32)
        hidden_states_77 = None
        hidden_states_78 = (
            l_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_
            * to_37
        )
        l_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_ = (
            to_37
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
        to_39 = hidden_states_81.to(torch.float32)
        hidden_states_81 = None
        hidden_states_82 = (
            l_self_modules_model_modules_layers_modules_8_modules_input_layernorm_parameters_weight_
            * to_39
        )
        l_self_modules_model_modules_layers_modules_8_modules_input_layernorm_parameters_weight_ = (
            to_39
        ) = None
        linear_56 = torch._C._nn.linear(
            hidden_states_82,
            l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_24 = linear_56.view((1, 32, -1, 64))
        linear_56 = None
        query_states_8 = view_24.transpose(1, 2)
        view_24 = None
        linear_57 = torch._C._nn.linear(
            hidden_states_82,
            l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_25 = linear_57.view((1, 32, -1, 64))
        linear_57 = None
        key_states_8 = view_25.transpose(1, 2)
        view_25 = None
        linear_58 = torch._C._nn.linear(
            hidden_states_82,
            l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_82 = l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_26 = linear_58.view((1, 32, -1, 64))
        linear_58 = None
        value_states_8 = view_26.transpose(1, 2)
        view_26 = None
        cos_11 = cos_2.unsqueeze(1)
        sin_11 = sin_2.unsqueeze(1)
        mul_76 = query_states_8 * cos_11
        x1_16 = query_states_8[(Ellipsis, slice(None, 32, None))]
        x2_16 = query_states_8[(Ellipsis, slice(32, None, None))]
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
        x1_17 = key_states_8[(Ellipsis, slice(None, 32, None))]
        x2_17 = key_states_8[(Ellipsis, slice(32, None, None))]
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
        hidden_states_83 = getitem_62.expand(1, 4, 3, 32, 64)
        getitem_62 = None
        key_8 = hidden_states_83.reshape(1, 12, 32, 64)
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
        hidden_states_84 = getitem_63.expand(1, 4, 3, 32, 64)
        getitem_63 = None
        value_8 = hidden_states_84.reshape(1, 12, 32, 64)
        hidden_states_84 = None
        attention_mask_9 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 32, None),
            )
        ]
        attn_output_32 = torch._C._nn.scaled_dot_product_attention(
            q_embed_8,
            key_8,
            value_8,
            attn_mask=attention_mask_9,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        q_embed_8 = key_8 = value_8 = attention_mask_9 = None
        transpose_36 = attn_output_32.transpose(1, 2)
        attn_output_32 = None
        attn_output_33 = transpose_36.contiguous()
        transpose_36 = None
        reshape_26 = attn_output_33.reshape(1, 32, -1)
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
        to_41 = hidden_states_87.to(torch.float32)
        hidden_states_87 = None
        hidden_states_88 = (
            l_self_modules_model_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_
            * to_41
        )
        l_self_modules_model_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_ = (
            to_41
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
        to_43 = hidden_states_91.to(torch.float32)
        hidden_states_91 = None
        hidden_states_92 = (
            l_self_modules_model_modules_layers_modules_9_modules_input_layernorm_parameters_weight_
            * to_43
        )
        l_self_modules_model_modules_layers_modules_9_modules_input_layernorm_parameters_weight_ = (
            to_43
        ) = None
        linear_63 = torch._C._nn.linear(
            hidden_states_92,
            l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_27 = linear_63.view((1, 32, -1, 64))
        linear_63 = None
        query_states_9 = view_27.transpose(1, 2)
        view_27 = None
        linear_64 = torch._C._nn.linear(
            hidden_states_92,
            l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_28 = linear_64.view((1, 32, -1, 64))
        linear_64 = None
        key_states_9 = view_28.transpose(1, 2)
        view_28 = None
        linear_65 = torch._C._nn.linear(
            hidden_states_92,
            l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_92 = l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_29 = linear_65.view((1, 32, -1, 64))
        linear_65 = None
        value_states_9 = view_29.transpose(1, 2)
        view_29 = None
        cos_12 = cos_2.unsqueeze(1)
        sin_12 = sin_2.unsqueeze(1)
        mul_85 = query_states_9 * cos_12
        x1_18 = query_states_9[(Ellipsis, slice(None, 32, None))]
        x2_18 = query_states_9[(Ellipsis, slice(32, None, None))]
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
        x1_19 = key_states_9[(Ellipsis, slice(None, 32, None))]
        x2_19 = key_states_9[(Ellipsis, slice(32, None, None))]
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
        hidden_states_93 = getitem_69.expand(1, 4, 3, 32, 64)
        getitem_69 = None
        key_9 = hidden_states_93.reshape(1, 12, 32, 64)
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
        hidden_states_94 = getitem_70.expand(1, 4, 3, 32, 64)
        getitem_70 = None
        value_9 = hidden_states_94.reshape(1, 12, 32, 64)
        hidden_states_94 = None
        attention_mask_10 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 32, None),
            )
        ]
        attn_output_36 = torch._C._nn.scaled_dot_product_attention(
            q_embed_9,
            key_9,
            value_9,
            attn_mask=attention_mask_10,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        q_embed_9 = key_9 = value_9 = attention_mask_10 = None
        transpose_40 = attn_output_36.transpose(1, 2)
        attn_output_36 = None
        attn_output_37 = transpose_40.contiguous()
        transpose_40 = None
        reshape_29 = attn_output_37.reshape(1, 32, -1)
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
        to_45 = hidden_states_97.to(torch.float32)
        hidden_states_97 = None
        hidden_states_98 = (
            l_self_modules_model_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_
            * to_45
        )
        l_self_modules_model_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_ = (
            to_45
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
        to_47 = hidden_states_101.to(torch.float32)
        hidden_states_101 = None
        hidden_states_102 = (
            l_self_modules_model_modules_layers_modules_10_modules_input_layernorm_parameters_weight_
            * to_47
        )
        l_self_modules_model_modules_layers_modules_10_modules_input_layernorm_parameters_weight_ = (
            to_47
        ) = None
        linear_70 = torch._C._nn.linear(
            hidden_states_102,
            l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_30 = linear_70.view((1, 32, -1, 64))
        linear_70 = None
        query_states_10 = view_30.transpose(1, 2)
        view_30 = None
        linear_71 = torch._C._nn.linear(
            hidden_states_102,
            l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_31 = linear_71.view((1, 32, -1, 64))
        linear_71 = None
        key_states_10 = view_31.transpose(1, 2)
        view_31 = None
        linear_72 = torch._C._nn.linear(
            hidden_states_102,
            l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_102 = l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_32 = linear_72.view((1, 32, -1, 64))
        linear_72 = None
        value_states_10 = view_32.transpose(1, 2)
        view_32 = None
        cos_13 = cos_2.unsqueeze(1)
        sin_13 = sin_2.unsqueeze(1)
        mul_94 = query_states_10 * cos_13
        x1_20 = query_states_10[(Ellipsis, slice(None, 32, None))]
        x2_20 = query_states_10[(Ellipsis, slice(32, None, None))]
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
        x1_21 = key_states_10[(Ellipsis, slice(None, 32, None))]
        x2_21 = key_states_10[(Ellipsis, slice(32, None, None))]
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
        hidden_states_103 = getitem_76.expand(1, 4, 3, 32, 64)
        getitem_76 = None
        key_10 = hidden_states_103.reshape(1, 12, 32, 64)
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
        hidden_states_104 = getitem_77.expand(1, 4, 3, 32, 64)
        getitem_77 = None
        value_10 = hidden_states_104.reshape(1, 12, 32, 64)
        hidden_states_104 = None
        attention_mask_11 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 32, None),
            )
        ]
        attn_output_40 = torch._C._nn.scaled_dot_product_attention(
            q_embed_10,
            key_10,
            value_10,
            attn_mask=attention_mask_11,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        q_embed_10 = key_10 = value_10 = attention_mask_11 = None
        transpose_44 = attn_output_40.transpose(1, 2)
        attn_output_40 = None
        attn_output_41 = transpose_44.contiguous()
        transpose_44 = None
        reshape_32 = attn_output_41.reshape(1, 32, -1)
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
        to_49 = hidden_states_107.to(torch.float32)
        hidden_states_107 = None
        hidden_states_108 = (
            l_self_modules_model_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_
            * to_49
        )
        l_self_modules_model_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_ = (
            to_49
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
        to_51 = hidden_states_111.to(torch.float32)
        hidden_states_111 = None
        hidden_states_112 = (
            l_self_modules_model_modules_layers_modules_11_modules_input_layernorm_parameters_weight_
            * to_51
        )
        l_self_modules_model_modules_layers_modules_11_modules_input_layernorm_parameters_weight_ = (
            to_51
        ) = None
        linear_77 = torch._C._nn.linear(
            hidden_states_112,
            l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_33 = linear_77.view((1, 32, -1, 64))
        linear_77 = None
        query_states_11 = view_33.transpose(1, 2)
        view_33 = None
        linear_78 = torch._C._nn.linear(
            hidden_states_112,
            l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_34 = linear_78.view((1, 32, -1, 64))
        linear_78 = None
        key_states_11 = view_34.transpose(1, 2)
        view_34 = None
        linear_79 = torch._C._nn.linear(
            hidden_states_112,
            l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_112 = l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_35 = linear_79.view((1, 32, -1, 64))
        linear_79 = None
        value_states_11 = view_35.transpose(1, 2)
        view_35 = None
        cos_14 = cos_2.unsqueeze(1)
        cos_2 = None
        sin_14 = sin_2.unsqueeze(1)
        sin_2 = None
        mul_103 = query_states_11 * cos_14
        x1_22 = query_states_11[(Ellipsis, slice(None, 32, None))]
        x2_22 = query_states_11[(Ellipsis, slice(32, None, None))]
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
        x1_23 = key_states_11[(Ellipsis, slice(None, 32, None))]
        x2_23 = key_states_11[(Ellipsis, slice(32, None, None))]
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
        hidden_states_113 = getitem_83.expand(1, 4, 3, 32, 64)
        getitem_83 = None
        key_11 = hidden_states_113.reshape(1, 12, 32, 64)
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
        hidden_states_114 = getitem_84.expand(1, 4, 3, 32, 64)
        getitem_84 = None
        value_11 = hidden_states_114.reshape(1, 12, 32, 64)
        hidden_states_114 = None
        attention_mask_12 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 32, None),
            )
        ]
        causal_mask = None
        attn_output_44 = torch._C._nn.scaled_dot_product_attention(
            q_embed_11,
            key_11,
            value_11,
            attn_mask=attention_mask_12,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        q_embed_11 = key_11 = value_11 = attention_mask_12 = None
        transpose_48 = attn_output_44.transpose(1, 2)
        attn_output_44 = None
        attn_output_45 = transpose_48.contiguous()
        transpose_48 = None
        reshape_35 = attn_output_45.reshape(1, 32, -1)
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
        to_53 = hidden_states_117.to(torch.float32)
        hidden_states_117 = None
        hidden_states_118 = (
            l_self_modules_model_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_
            * to_53
        )
        l_self_modules_model_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_ = (
            to_53
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
        hidden_states_119 = None
        pow_25 = hidden_states_120.pow(2)
        variance_24 = pow_25.mean(-1, keepdim=True)
        pow_25 = None
        add_72 = variance_24 + 1e-05
        variance_24 = None
        rsqrt_24 = torch.rsqrt(add_72)
        add_72 = None
        hidden_states_121 = hidden_states_120 * rsqrt_24
        hidden_states_120 = rsqrt_24 = None
        to_55 = hidden_states_121.to(torch.float32)
        hidden_states_121 = None
        hidden_states_122 = l_self_modules_model_modules_norm_parameters_weight_ * to_55
        l_self_modules_model_modules_norm_parameters_weight_ = to_55 = None
        getitem_86 = hidden_states_122[
            (slice(None, None, None), slice(0, None, None), slice(None, None, None))
        ]
        hidden_states_122 = None
        logits = torch._C._nn.linear(
            getitem_86, l_self_modules_lm_head_parameters_weight_, None
        )
        getitem_86 = l_self_modules_lm_head_parameters_weight_ = None
        return (logits,)
