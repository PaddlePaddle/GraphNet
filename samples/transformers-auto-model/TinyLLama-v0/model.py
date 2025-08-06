import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_kwargs_input_ids_: torch.Tensor,
        L_self_modules_model_modules_embed_tokens_parameters_weight_: torch.nn.parameter.Parameter,
        L_kwargs_attention_mask_: torch.Tensor,
        L_self_modules_model_modules_rotary_emb_buffers_inv_freq_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_0_modules_input_layernorm_variance_epsilon: torch.Tensor,
        L_self_modules_model_modules_layers_modules_0_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_variance_epsilon: torch.Tensor,
        L_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_input_layernorm_variance_epsilon: torch.Tensor,
        L_self_modules_model_modules_layers_modules_1_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_variance_epsilon: torch.Tensor,
        L_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_input_layernorm_variance_epsilon: torch.Tensor,
        L_self_modules_model_modules_layers_modules_2_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_variance_epsilon: torch.Tensor,
        L_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_input_layernorm_variance_epsilon: torch.Tensor,
        L_self_modules_model_modules_layers_modules_3_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_variance_epsilon: torch.Tensor,
        L_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_input_layernorm_variance_epsilon: torch.Tensor,
        L_self_modules_model_modules_layers_modules_4_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_variance_epsilon: torch.Tensor,
        L_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_input_layernorm_variance_epsilon: torch.Tensor,
        L_self_modules_model_modules_layers_modules_5_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_variance_epsilon: torch.Tensor,
        L_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_input_layernorm_variance_epsilon: torch.Tensor,
        L_self_modules_model_modules_layers_modules_6_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_variance_epsilon: torch.Tensor,
        L_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_input_layernorm_variance_epsilon: torch.Tensor,
        L_self_modules_model_modules_layers_modules_7_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_variance_epsilon: torch.Tensor,
        L_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_norm_variance_epsilon: torch.Tensor,
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
        l_self_modules_model_modules_layers_modules_0_modules_input_layernorm_variance_epsilon = L_self_modules_model_modules_layers_modules_0_modules_input_layernorm_variance_epsilon
        l_self_modules_model_modules_layers_modules_0_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_scaling = (
            L_self_modules_model_modules_layers_modules_0_modules_self_attn_scaling
        )
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_variance_epsilon = L_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_variance_epsilon
        l_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_input_layernorm_variance_epsilon = L_self_modules_model_modules_layers_modules_1_modules_input_layernorm_variance_epsilon
        l_self_modules_model_modules_layers_modules_1_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_scaling = (
            L_self_modules_model_modules_layers_modules_1_modules_self_attn_scaling
        )
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_variance_epsilon = L_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_variance_epsilon
        l_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_input_layernorm_variance_epsilon = L_self_modules_model_modules_layers_modules_2_modules_input_layernorm_variance_epsilon
        l_self_modules_model_modules_layers_modules_2_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_scaling = (
            L_self_modules_model_modules_layers_modules_2_modules_self_attn_scaling
        )
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_variance_epsilon = L_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_variance_epsilon
        l_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_input_layernorm_variance_epsilon = L_self_modules_model_modules_layers_modules_3_modules_input_layernorm_variance_epsilon
        l_self_modules_model_modules_layers_modules_3_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_scaling = (
            L_self_modules_model_modules_layers_modules_3_modules_self_attn_scaling
        )
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_variance_epsilon = L_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_variance_epsilon
        l_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_input_layernorm_variance_epsilon = L_self_modules_model_modules_layers_modules_4_modules_input_layernorm_variance_epsilon
        l_self_modules_model_modules_layers_modules_4_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_scaling = (
            L_self_modules_model_modules_layers_modules_4_modules_self_attn_scaling
        )
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_variance_epsilon = L_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_variance_epsilon
        l_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_input_layernorm_variance_epsilon = L_self_modules_model_modules_layers_modules_5_modules_input_layernorm_variance_epsilon
        l_self_modules_model_modules_layers_modules_5_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_scaling = (
            L_self_modules_model_modules_layers_modules_5_modules_self_attn_scaling
        )
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_variance_epsilon = L_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_variance_epsilon
        l_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_input_layernorm_variance_epsilon = L_self_modules_model_modules_layers_modules_6_modules_input_layernorm_variance_epsilon
        l_self_modules_model_modules_layers_modules_6_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_scaling = (
            L_self_modules_model_modules_layers_modules_6_modules_self_attn_scaling
        )
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_variance_epsilon = L_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_variance_epsilon
        l_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_input_layernorm_variance_epsilon = L_self_modules_model_modules_layers_modules_7_modules_input_layernorm_variance_epsilon
        l_self_modules_model_modules_layers_modules_7_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_scaling = (
            L_self_modules_model_modules_layers_modules_7_modules_self_attn_scaling
        )
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_variance_epsilon = L_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_variance_epsilon
        l_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_norm_variance_epsilon = (
            L_self_modules_model_modules_norm_variance_epsilon
        )
        l_self_modules_model_modules_norm_parameters_weight_ = (
            L_self_modules_model_modules_norm_parameters_weight_
        )
        l_self_modules_lm_head_parameters_weight_ = (
            L_self_modules_lm_head_parameters_weight_
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
        l_kwargs_input_ids_ = (
            l_self_modules_model_modules_embed_tokens_parameters_weight_
        ) = None
        cache_position = torch.arange(0, 21, device=device(type="cpu"))
        position_ids = cache_position.unsqueeze(0)
        attention_mask = l_kwargs_attention_mask_.to(
            device=device(type="cpu"), dtype=torch.bool
        )
        l_kwargs_attention_mask_ = None
        kv_arange = torch.arange(21, device=device(type="cpu"))
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
            21, "error"
        )
        _vmap_increment_nesting_2 = None
        child_2 = torch._C._functorch._add_batch_dim(cache_position, 0, 3)
        cache_position = None
        lazy_load_decompositions_3 = torch._functorch.vmap.lazy_load_decompositions()
        lazy_load_decompositions_3 = None
        _vmap_increment_nesting_3 = torch._C._functorch._vmap_increment_nesting(
            21, "error"
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
        batched_outputs = torch._C._functorch._remove_batch_dim(result_2, 4, 21, 0)
        result_2 = None
        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting()
        _vmap_decrement_nesting = None
        batched_outputs_1 = torch._C._functorch._remove_batch_dim(
            batched_outputs, 3, 21, 0
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
        item = (
            l_self_modules_model_modules_layers_modules_0_modules_input_layernorm_variance_epsilon.item()
        )
        l_self_modules_model_modules_layers_modules_0_modules_input_layernorm_variance_epsilon = (
            None
        )
        add = variance + item
        variance = item = None
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
            None,
        )
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view = linear.view((1, 21, -1, 4))
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
        view_1 = linear_1.view((1, 21, -1, 4))
        linear_1 = None
        key_states = view_1.transpose(1, 2)
        view_1 = None
        linear_2 = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_2 = l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_2 = linear_2.view((1, 21, -1, 4))
        linear_2 = None
        value_states = view_2.transpose(1, 2)
        view_2 = None
        cos_3 = cos_2.unsqueeze(1)
        sin_3 = sin_2.unsqueeze(1)
        mul_4 = query_states * cos_3
        x1 = query_states[(Ellipsis, slice(None, 2, None))]
        x2 = query_states[(Ellipsis, slice(2, None, None))]
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
        x1_1 = key_states[(Ellipsis, slice(None, 2, None))]
        x2_1 = key_states[(Ellipsis, slice(2, None, None))]
        key_states = None
        neg_1 = -x2_1
        x2_1 = None
        cat_2 = torch.cat((neg_1, x1_1), dim=-1)
        neg_1 = x1_1 = None
        mul_7 = cat_2 * sin_3
        cat_2 = sin_3 = None
        k_embed = mul_6 + mul_7
        mul_6 = mul_7 = None
        attention_mask_1 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 21, None),
            )
        ]
        query = q_embed.contiguous()
        q_embed = None
        key = k_embed.contiguous()
        k_embed = None
        value = value_states.contiguous()
        value_states = None
        item_1 = (
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_scaling.item()
        )
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_scaling = None
        attn_output = torch._C._nn.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask_1,
            dropout_p=0.0,
            scale=item_1,
            is_causal=False,
        )
        query = key = value = attention_mask_1 = item_1 = None
        transpose_4 = attn_output.transpose(1, 2)
        attn_output = None
        attn_output_1 = transpose_4.contiguous()
        transpose_4 = None
        reshape = attn_output_1.reshape(1, 21, -1)
        attn_output_1 = None
        attn_output_2 = reshape.contiguous()
        reshape = None
        attn_output_3 = torch._C._nn.linear(
            attn_output_2,
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_2 = l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_3 = inputs_embeds + attn_output_3
        inputs_embeds = attn_output_3 = None
        hidden_states_4 = hidden_states_3.to(torch.float32)
        pow_2 = hidden_states_4.pow(2)
        variance_1 = pow_2.mean(-1, keepdim=True)
        pow_2 = None
        item_2 = (
            l_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_variance_epsilon.item()
        )
        l_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_variance_epsilon = (
            None
        )
        add_4 = variance_1 + item_2
        variance_1 = item_2 = None
        rsqrt_1 = torch.rsqrt(add_4)
        add_4 = None
        hidden_states_5 = hidden_states_4 * rsqrt_1
        hidden_states_4 = rsqrt_1 = None
        to_7 = hidden_states_5.to(torch.float16)
        hidden_states_5 = None
        hidden_states_6 = (
            l_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_
            * to_7
        )
        l_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_ = (
            to_7
        ) = None
        linear_4 = torch._C._nn.linear(
            hidden_states_6,
            l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu = torch.nn.functional.silu(linear_4, inplace=False)
        linear_4 = None
        linear_5 = torch._C._nn.linear(
            hidden_states_6,
            l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_6 = l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_10 = silu * linear_5
        silu = linear_5 = None
        down_proj = torch._C._nn.linear(
            mul_10,
            l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_10 = l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_7 = hidden_states_3 + down_proj
        hidden_states_3 = down_proj = None
        hidden_states_8 = hidden_states_7.to(torch.float32)
        pow_3 = hidden_states_8.pow(2)
        variance_2 = pow_3.mean(-1, keepdim=True)
        pow_3 = None
        item_3 = (
            l_self_modules_model_modules_layers_modules_1_modules_input_layernorm_variance_epsilon.item()
        )
        l_self_modules_model_modules_layers_modules_1_modules_input_layernorm_variance_epsilon = (
            None
        )
        add_6 = variance_2 + item_3
        variance_2 = item_3 = None
        rsqrt_2 = torch.rsqrt(add_6)
        add_6 = None
        hidden_states_9 = hidden_states_8 * rsqrt_2
        hidden_states_8 = rsqrt_2 = None
        to_9 = hidden_states_9.to(torch.float16)
        hidden_states_9 = None
        hidden_states_10 = (
            l_self_modules_model_modules_layers_modules_1_modules_input_layernorm_parameters_weight_
            * to_9
        )
        l_self_modules_model_modules_layers_modules_1_modules_input_layernorm_parameters_weight_ = (
            to_9
        ) = None
        linear_7 = torch._C._nn.linear(
            hidden_states_10,
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_3 = linear_7.view((1, 21, -1, 4))
        linear_7 = None
        query_states_1 = view_3.transpose(1, 2)
        view_3 = None
        linear_8 = torch._C._nn.linear(
            hidden_states_10,
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_4 = linear_8.view((1, 21, -1, 4))
        linear_8 = None
        key_states_1 = view_4.transpose(1, 2)
        view_4 = None
        linear_9 = torch._C._nn.linear(
            hidden_states_10,
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_10 = l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_5 = linear_9.view((1, 21, -1, 4))
        linear_9 = None
        value_states_1 = view_5.transpose(1, 2)
        view_5 = None
        cos_4 = cos_2.unsqueeze(1)
        sin_4 = sin_2.unsqueeze(1)
        mul_13 = query_states_1 * cos_4
        x1_2 = query_states_1[(Ellipsis, slice(None, 2, None))]
        x2_2 = query_states_1[(Ellipsis, slice(2, None, None))]
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
        x1_3 = key_states_1[(Ellipsis, slice(None, 2, None))]
        x2_3 = key_states_1[(Ellipsis, slice(2, None, None))]
        key_states_1 = None
        neg_3 = -x2_3
        x2_3 = None
        cat_4 = torch.cat((neg_3, x1_3), dim=-1)
        neg_3 = x1_3 = None
        mul_16 = cat_4 * sin_4
        cat_4 = sin_4 = None
        k_embed_1 = mul_15 + mul_16
        mul_15 = mul_16 = None
        attention_mask_2 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 21, None),
            )
        ]
        query_1 = q_embed_1.contiguous()
        q_embed_1 = None
        key_1 = k_embed_1.contiguous()
        k_embed_1 = None
        value_1 = value_states_1.contiguous()
        value_states_1 = None
        item_4 = (
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_scaling.item()
        )
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_scaling = None
        attn_output_4 = torch._C._nn.scaled_dot_product_attention(
            query_1,
            key_1,
            value_1,
            attn_mask=attention_mask_2,
            dropout_p=0.0,
            scale=item_4,
            is_causal=False,
        )
        query_1 = key_1 = value_1 = attention_mask_2 = item_4 = None
        transpose_8 = attn_output_4.transpose(1, 2)
        attn_output_4 = None
        attn_output_5 = transpose_8.contiguous()
        transpose_8 = None
        reshape_1 = attn_output_5.reshape(1, 21, -1)
        attn_output_5 = None
        attn_output_6 = reshape_1.contiguous()
        reshape_1 = None
        attn_output_7 = torch._C._nn.linear(
            attn_output_6,
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_6 = l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_11 = hidden_states_7 + attn_output_7
        hidden_states_7 = attn_output_7 = None
        hidden_states_12 = hidden_states_11.to(torch.float32)
        pow_4 = hidden_states_12.pow(2)
        variance_3 = pow_4.mean(-1, keepdim=True)
        pow_4 = None
        item_5 = (
            l_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_variance_epsilon.item()
        )
        l_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_variance_epsilon = (
            None
        )
        add_10 = variance_3 + item_5
        variance_3 = item_5 = None
        rsqrt_3 = torch.rsqrt(add_10)
        add_10 = None
        hidden_states_13 = hidden_states_12 * rsqrt_3
        hidden_states_12 = rsqrt_3 = None
        to_11 = hidden_states_13.to(torch.float16)
        hidden_states_13 = None
        hidden_states_14 = (
            l_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_
            * to_11
        )
        l_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_ = (
            to_11
        ) = None
        linear_11 = torch._C._nn.linear(
            hidden_states_14,
            l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_1 = torch.nn.functional.silu(linear_11, inplace=False)
        linear_11 = None
        linear_12 = torch._C._nn.linear(
            hidden_states_14,
            l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_14 = l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_19 = silu_1 * linear_12
        silu_1 = linear_12 = None
        down_proj_1 = torch._C._nn.linear(
            mul_19,
            l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_19 = l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_15 = hidden_states_11 + down_proj_1
        hidden_states_11 = down_proj_1 = None
        hidden_states_16 = hidden_states_15.to(torch.float32)
        pow_5 = hidden_states_16.pow(2)
        variance_4 = pow_5.mean(-1, keepdim=True)
        pow_5 = None
        item_6 = (
            l_self_modules_model_modules_layers_modules_2_modules_input_layernorm_variance_epsilon.item()
        )
        l_self_modules_model_modules_layers_modules_2_modules_input_layernorm_variance_epsilon = (
            None
        )
        add_12 = variance_4 + item_6
        variance_4 = item_6 = None
        rsqrt_4 = torch.rsqrt(add_12)
        add_12 = None
        hidden_states_17 = hidden_states_16 * rsqrt_4
        hidden_states_16 = rsqrt_4 = None
        to_13 = hidden_states_17.to(torch.float16)
        hidden_states_17 = None
        hidden_states_18 = (
            l_self_modules_model_modules_layers_modules_2_modules_input_layernorm_parameters_weight_
            * to_13
        )
        l_self_modules_model_modules_layers_modules_2_modules_input_layernorm_parameters_weight_ = (
            to_13
        ) = None
        linear_14 = torch._C._nn.linear(
            hidden_states_18,
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_6 = linear_14.view((1, 21, -1, 4))
        linear_14 = None
        query_states_2 = view_6.transpose(1, 2)
        view_6 = None
        linear_15 = torch._C._nn.linear(
            hidden_states_18,
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_7 = linear_15.view((1, 21, -1, 4))
        linear_15 = None
        key_states_2 = view_7.transpose(1, 2)
        view_7 = None
        linear_16 = torch._C._nn.linear(
            hidden_states_18,
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_18 = l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_8 = linear_16.view((1, 21, -1, 4))
        linear_16 = None
        value_states_2 = view_8.transpose(1, 2)
        view_8 = None
        cos_5 = cos_2.unsqueeze(1)
        sin_5 = sin_2.unsqueeze(1)
        mul_22 = query_states_2 * cos_5
        x1_4 = query_states_2[(Ellipsis, slice(None, 2, None))]
        x2_4 = query_states_2[(Ellipsis, slice(2, None, None))]
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
        x1_5 = key_states_2[(Ellipsis, slice(None, 2, None))]
        x2_5 = key_states_2[(Ellipsis, slice(2, None, None))]
        key_states_2 = None
        neg_5 = -x2_5
        x2_5 = None
        cat_6 = torch.cat((neg_5, x1_5), dim=-1)
        neg_5 = x1_5 = None
        mul_25 = cat_6 * sin_5
        cat_6 = sin_5 = None
        k_embed_2 = mul_24 + mul_25
        mul_24 = mul_25 = None
        attention_mask_3 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 21, None),
            )
        ]
        query_2 = q_embed_2.contiguous()
        q_embed_2 = None
        key_2 = k_embed_2.contiguous()
        k_embed_2 = None
        value_2 = value_states_2.contiguous()
        value_states_2 = None
        item_7 = (
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_scaling.item()
        )
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_scaling = None
        attn_output_8 = torch._C._nn.scaled_dot_product_attention(
            query_2,
            key_2,
            value_2,
            attn_mask=attention_mask_3,
            dropout_p=0.0,
            scale=item_7,
            is_causal=False,
        )
        query_2 = key_2 = value_2 = attention_mask_3 = item_7 = None
        transpose_12 = attn_output_8.transpose(1, 2)
        attn_output_8 = None
        attn_output_9 = transpose_12.contiguous()
        transpose_12 = None
        reshape_2 = attn_output_9.reshape(1, 21, -1)
        attn_output_9 = None
        attn_output_10 = reshape_2.contiguous()
        reshape_2 = None
        attn_output_11 = torch._C._nn.linear(
            attn_output_10,
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_10 = l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_19 = hidden_states_15 + attn_output_11
        hidden_states_15 = attn_output_11 = None
        hidden_states_20 = hidden_states_19.to(torch.float32)
        pow_6 = hidden_states_20.pow(2)
        variance_5 = pow_6.mean(-1, keepdim=True)
        pow_6 = None
        item_8 = (
            l_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_variance_epsilon.item()
        )
        l_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_variance_epsilon = (
            None
        )
        add_16 = variance_5 + item_8
        variance_5 = item_8 = None
        rsqrt_5 = torch.rsqrt(add_16)
        add_16 = None
        hidden_states_21 = hidden_states_20 * rsqrt_5
        hidden_states_20 = rsqrt_5 = None
        to_15 = hidden_states_21.to(torch.float16)
        hidden_states_21 = None
        hidden_states_22 = (
            l_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_
            * to_15
        )
        l_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_ = (
            to_15
        ) = None
        linear_18 = torch._C._nn.linear(
            hidden_states_22,
            l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_2 = torch.nn.functional.silu(linear_18, inplace=False)
        linear_18 = None
        linear_19 = torch._C._nn.linear(
            hidden_states_22,
            l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_22 = l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_28 = silu_2 * linear_19
        silu_2 = linear_19 = None
        down_proj_2 = torch._C._nn.linear(
            mul_28,
            l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_28 = l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_23 = hidden_states_19 + down_proj_2
        hidden_states_19 = down_proj_2 = None
        hidden_states_24 = hidden_states_23.to(torch.float32)
        pow_7 = hidden_states_24.pow(2)
        variance_6 = pow_7.mean(-1, keepdim=True)
        pow_7 = None
        item_9 = (
            l_self_modules_model_modules_layers_modules_3_modules_input_layernorm_variance_epsilon.item()
        )
        l_self_modules_model_modules_layers_modules_3_modules_input_layernorm_variance_epsilon = (
            None
        )
        add_18 = variance_6 + item_9
        variance_6 = item_9 = None
        rsqrt_6 = torch.rsqrt(add_18)
        add_18 = None
        hidden_states_25 = hidden_states_24 * rsqrt_6
        hidden_states_24 = rsqrt_6 = None
        to_17 = hidden_states_25.to(torch.float16)
        hidden_states_25 = None
        hidden_states_26 = (
            l_self_modules_model_modules_layers_modules_3_modules_input_layernorm_parameters_weight_
            * to_17
        )
        l_self_modules_model_modules_layers_modules_3_modules_input_layernorm_parameters_weight_ = (
            to_17
        ) = None
        linear_21 = torch._C._nn.linear(
            hidden_states_26,
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_9 = linear_21.view((1, 21, -1, 4))
        linear_21 = None
        query_states_3 = view_9.transpose(1, 2)
        view_9 = None
        linear_22 = torch._C._nn.linear(
            hidden_states_26,
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_10 = linear_22.view((1, 21, -1, 4))
        linear_22 = None
        key_states_3 = view_10.transpose(1, 2)
        view_10 = None
        linear_23 = torch._C._nn.linear(
            hidden_states_26,
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_26 = l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_11 = linear_23.view((1, 21, -1, 4))
        linear_23 = None
        value_states_3 = view_11.transpose(1, 2)
        view_11 = None
        cos_6 = cos_2.unsqueeze(1)
        sin_6 = sin_2.unsqueeze(1)
        mul_31 = query_states_3 * cos_6
        x1_6 = query_states_3[(Ellipsis, slice(None, 2, None))]
        x2_6 = query_states_3[(Ellipsis, slice(2, None, None))]
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
        x1_7 = key_states_3[(Ellipsis, slice(None, 2, None))]
        x2_7 = key_states_3[(Ellipsis, slice(2, None, None))]
        key_states_3 = None
        neg_7 = -x2_7
        x2_7 = None
        cat_8 = torch.cat((neg_7, x1_7), dim=-1)
        neg_7 = x1_7 = None
        mul_34 = cat_8 * sin_6
        cat_8 = sin_6 = None
        k_embed_3 = mul_33 + mul_34
        mul_33 = mul_34 = None
        attention_mask_4 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 21, None),
            )
        ]
        query_3 = q_embed_3.contiguous()
        q_embed_3 = None
        key_3 = k_embed_3.contiguous()
        k_embed_3 = None
        value_3 = value_states_3.contiguous()
        value_states_3 = None
        item_10 = (
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_scaling.item()
        )
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_scaling = None
        attn_output_12 = torch._C._nn.scaled_dot_product_attention(
            query_3,
            key_3,
            value_3,
            attn_mask=attention_mask_4,
            dropout_p=0.0,
            scale=item_10,
            is_causal=False,
        )
        query_3 = key_3 = value_3 = attention_mask_4 = item_10 = None
        transpose_16 = attn_output_12.transpose(1, 2)
        attn_output_12 = None
        attn_output_13 = transpose_16.contiguous()
        transpose_16 = None
        reshape_3 = attn_output_13.reshape(1, 21, -1)
        attn_output_13 = None
        attn_output_14 = reshape_3.contiguous()
        reshape_3 = None
        attn_output_15 = torch._C._nn.linear(
            attn_output_14,
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_14 = l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_27 = hidden_states_23 + attn_output_15
        hidden_states_23 = attn_output_15 = None
        hidden_states_28 = hidden_states_27.to(torch.float32)
        pow_8 = hidden_states_28.pow(2)
        variance_7 = pow_8.mean(-1, keepdim=True)
        pow_8 = None
        item_11 = (
            l_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_variance_epsilon.item()
        )
        l_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_variance_epsilon = (
            None
        )
        add_22 = variance_7 + item_11
        variance_7 = item_11 = None
        rsqrt_7 = torch.rsqrt(add_22)
        add_22 = None
        hidden_states_29 = hidden_states_28 * rsqrt_7
        hidden_states_28 = rsqrt_7 = None
        to_19 = hidden_states_29.to(torch.float16)
        hidden_states_29 = None
        hidden_states_30 = (
            l_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_
            * to_19
        )
        l_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_ = (
            to_19
        ) = None
        linear_25 = torch._C._nn.linear(
            hidden_states_30,
            l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_3 = torch.nn.functional.silu(linear_25, inplace=False)
        linear_25 = None
        linear_26 = torch._C._nn.linear(
            hidden_states_30,
            l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_30 = l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_37 = silu_3 * linear_26
        silu_3 = linear_26 = None
        down_proj_3 = torch._C._nn.linear(
            mul_37,
            l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_37 = l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_31 = hidden_states_27 + down_proj_3
        hidden_states_27 = down_proj_3 = None
        hidden_states_32 = hidden_states_31.to(torch.float32)
        pow_9 = hidden_states_32.pow(2)
        variance_8 = pow_9.mean(-1, keepdim=True)
        pow_9 = None
        item_12 = (
            l_self_modules_model_modules_layers_modules_4_modules_input_layernorm_variance_epsilon.item()
        )
        l_self_modules_model_modules_layers_modules_4_modules_input_layernorm_variance_epsilon = (
            None
        )
        add_24 = variance_8 + item_12
        variance_8 = item_12 = None
        rsqrt_8 = torch.rsqrt(add_24)
        add_24 = None
        hidden_states_33 = hidden_states_32 * rsqrt_8
        hidden_states_32 = rsqrt_8 = None
        to_21 = hidden_states_33.to(torch.float16)
        hidden_states_33 = None
        hidden_states_34 = (
            l_self_modules_model_modules_layers_modules_4_modules_input_layernorm_parameters_weight_
            * to_21
        )
        l_self_modules_model_modules_layers_modules_4_modules_input_layernorm_parameters_weight_ = (
            to_21
        ) = None
        linear_28 = torch._C._nn.linear(
            hidden_states_34,
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_12 = linear_28.view((1, 21, -1, 4))
        linear_28 = None
        query_states_4 = view_12.transpose(1, 2)
        view_12 = None
        linear_29 = torch._C._nn.linear(
            hidden_states_34,
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_13 = linear_29.view((1, 21, -1, 4))
        linear_29 = None
        key_states_4 = view_13.transpose(1, 2)
        view_13 = None
        linear_30 = torch._C._nn.linear(
            hidden_states_34,
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_34 = l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_14 = linear_30.view((1, 21, -1, 4))
        linear_30 = None
        value_states_4 = view_14.transpose(1, 2)
        view_14 = None
        cos_7 = cos_2.unsqueeze(1)
        sin_7 = sin_2.unsqueeze(1)
        mul_40 = query_states_4 * cos_7
        x1_8 = query_states_4[(Ellipsis, slice(None, 2, None))]
        x2_8 = query_states_4[(Ellipsis, slice(2, None, None))]
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
        x1_9 = key_states_4[(Ellipsis, slice(None, 2, None))]
        x2_9 = key_states_4[(Ellipsis, slice(2, None, None))]
        key_states_4 = None
        neg_9 = -x2_9
        x2_9 = None
        cat_10 = torch.cat((neg_9, x1_9), dim=-1)
        neg_9 = x1_9 = None
        mul_43 = cat_10 * sin_7
        cat_10 = sin_7 = None
        k_embed_4 = mul_42 + mul_43
        mul_42 = mul_43 = None
        attention_mask_5 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 21, None),
            )
        ]
        query_4 = q_embed_4.contiguous()
        q_embed_4 = None
        key_4 = k_embed_4.contiguous()
        k_embed_4 = None
        value_4 = value_states_4.contiguous()
        value_states_4 = None
        item_13 = (
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_scaling.item()
        )
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_scaling = None
        attn_output_16 = torch._C._nn.scaled_dot_product_attention(
            query_4,
            key_4,
            value_4,
            attn_mask=attention_mask_5,
            dropout_p=0.0,
            scale=item_13,
            is_causal=False,
        )
        query_4 = key_4 = value_4 = attention_mask_5 = item_13 = None
        transpose_20 = attn_output_16.transpose(1, 2)
        attn_output_16 = None
        attn_output_17 = transpose_20.contiguous()
        transpose_20 = None
        reshape_4 = attn_output_17.reshape(1, 21, -1)
        attn_output_17 = None
        attn_output_18 = reshape_4.contiguous()
        reshape_4 = None
        attn_output_19 = torch._C._nn.linear(
            attn_output_18,
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_18 = l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_35 = hidden_states_31 + attn_output_19
        hidden_states_31 = attn_output_19 = None
        hidden_states_36 = hidden_states_35.to(torch.float32)
        pow_10 = hidden_states_36.pow(2)
        variance_9 = pow_10.mean(-1, keepdim=True)
        pow_10 = None
        item_14 = (
            l_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_variance_epsilon.item()
        )
        l_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_variance_epsilon = (
            None
        )
        add_28 = variance_9 + item_14
        variance_9 = item_14 = None
        rsqrt_9 = torch.rsqrt(add_28)
        add_28 = None
        hidden_states_37 = hidden_states_36 * rsqrt_9
        hidden_states_36 = rsqrt_9 = None
        to_23 = hidden_states_37.to(torch.float16)
        hidden_states_37 = None
        hidden_states_38 = (
            l_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_
            * to_23
        )
        l_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_ = (
            to_23
        ) = None
        linear_32 = torch._C._nn.linear(
            hidden_states_38,
            l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_4 = torch.nn.functional.silu(linear_32, inplace=False)
        linear_32 = None
        linear_33 = torch._C._nn.linear(
            hidden_states_38,
            l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_38 = l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_46 = silu_4 * linear_33
        silu_4 = linear_33 = None
        down_proj_4 = torch._C._nn.linear(
            mul_46,
            l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_46 = l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_39 = hidden_states_35 + down_proj_4
        hidden_states_35 = down_proj_4 = None
        hidden_states_40 = hidden_states_39.to(torch.float32)
        pow_11 = hidden_states_40.pow(2)
        variance_10 = pow_11.mean(-1, keepdim=True)
        pow_11 = None
        item_15 = (
            l_self_modules_model_modules_layers_modules_5_modules_input_layernorm_variance_epsilon.item()
        )
        l_self_modules_model_modules_layers_modules_5_modules_input_layernorm_variance_epsilon = (
            None
        )
        add_30 = variance_10 + item_15
        variance_10 = item_15 = None
        rsqrt_10 = torch.rsqrt(add_30)
        add_30 = None
        hidden_states_41 = hidden_states_40 * rsqrt_10
        hidden_states_40 = rsqrt_10 = None
        to_25 = hidden_states_41.to(torch.float16)
        hidden_states_41 = None
        hidden_states_42 = (
            l_self_modules_model_modules_layers_modules_5_modules_input_layernorm_parameters_weight_
            * to_25
        )
        l_self_modules_model_modules_layers_modules_5_modules_input_layernorm_parameters_weight_ = (
            to_25
        ) = None
        linear_35 = torch._C._nn.linear(
            hidden_states_42,
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_15 = linear_35.view((1, 21, -1, 4))
        linear_35 = None
        query_states_5 = view_15.transpose(1, 2)
        view_15 = None
        linear_36 = torch._C._nn.linear(
            hidden_states_42,
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_16 = linear_36.view((1, 21, -1, 4))
        linear_36 = None
        key_states_5 = view_16.transpose(1, 2)
        view_16 = None
        linear_37 = torch._C._nn.linear(
            hidden_states_42,
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_42 = l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_17 = linear_37.view((1, 21, -1, 4))
        linear_37 = None
        value_states_5 = view_17.transpose(1, 2)
        view_17 = None
        cos_8 = cos_2.unsqueeze(1)
        sin_8 = sin_2.unsqueeze(1)
        mul_49 = query_states_5 * cos_8
        x1_10 = query_states_5[(Ellipsis, slice(None, 2, None))]
        x2_10 = query_states_5[(Ellipsis, slice(2, None, None))]
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
        x1_11 = key_states_5[(Ellipsis, slice(None, 2, None))]
        x2_11 = key_states_5[(Ellipsis, slice(2, None, None))]
        key_states_5 = None
        neg_11 = -x2_11
        x2_11 = None
        cat_12 = torch.cat((neg_11, x1_11), dim=-1)
        neg_11 = x1_11 = None
        mul_52 = cat_12 * sin_8
        cat_12 = sin_8 = None
        k_embed_5 = mul_51 + mul_52
        mul_51 = mul_52 = None
        attention_mask_6 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 21, None),
            )
        ]
        query_5 = q_embed_5.contiguous()
        q_embed_5 = None
        key_5 = k_embed_5.contiguous()
        k_embed_5 = None
        value_5 = value_states_5.contiguous()
        value_states_5 = None
        item_16 = (
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_scaling.item()
        )
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_scaling = None
        attn_output_20 = torch._C._nn.scaled_dot_product_attention(
            query_5,
            key_5,
            value_5,
            attn_mask=attention_mask_6,
            dropout_p=0.0,
            scale=item_16,
            is_causal=False,
        )
        query_5 = key_5 = value_5 = attention_mask_6 = item_16 = None
        transpose_24 = attn_output_20.transpose(1, 2)
        attn_output_20 = None
        attn_output_21 = transpose_24.contiguous()
        transpose_24 = None
        reshape_5 = attn_output_21.reshape(1, 21, -1)
        attn_output_21 = None
        attn_output_22 = reshape_5.contiguous()
        reshape_5 = None
        attn_output_23 = torch._C._nn.linear(
            attn_output_22,
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_22 = l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_43 = hidden_states_39 + attn_output_23
        hidden_states_39 = attn_output_23 = None
        hidden_states_44 = hidden_states_43.to(torch.float32)
        pow_12 = hidden_states_44.pow(2)
        variance_11 = pow_12.mean(-1, keepdim=True)
        pow_12 = None
        item_17 = (
            l_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_variance_epsilon.item()
        )
        l_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_variance_epsilon = (
            None
        )
        add_34 = variance_11 + item_17
        variance_11 = item_17 = None
        rsqrt_11 = torch.rsqrt(add_34)
        add_34 = None
        hidden_states_45 = hidden_states_44 * rsqrt_11
        hidden_states_44 = rsqrt_11 = None
        to_27 = hidden_states_45.to(torch.float16)
        hidden_states_45 = None
        hidden_states_46 = (
            l_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_
            * to_27
        )
        l_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_ = (
            to_27
        ) = None
        linear_39 = torch._C._nn.linear(
            hidden_states_46,
            l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_5 = torch.nn.functional.silu(linear_39, inplace=False)
        linear_39 = None
        linear_40 = torch._C._nn.linear(
            hidden_states_46,
            l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_46 = l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_55 = silu_5 * linear_40
        silu_5 = linear_40 = None
        down_proj_5 = torch._C._nn.linear(
            mul_55,
            l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_55 = l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_47 = hidden_states_43 + down_proj_5
        hidden_states_43 = down_proj_5 = None
        hidden_states_48 = hidden_states_47.to(torch.float32)
        pow_13 = hidden_states_48.pow(2)
        variance_12 = pow_13.mean(-1, keepdim=True)
        pow_13 = None
        item_18 = (
            l_self_modules_model_modules_layers_modules_6_modules_input_layernorm_variance_epsilon.item()
        )
        l_self_modules_model_modules_layers_modules_6_modules_input_layernorm_variance_epsilon = (
            None
        )
        add_36 = variance_12 + item_18
        variance_12 = item_18 = None
        rsqrt_12 = torch.rsqrt(add_36)
        add_36 = None
        hidden_states_49 = hidden_states_48 * rsqrt_12
        hidden_states_48 = rsqrt_12 = None
        to_29 = hidden_states_49.to(torch.float16)
        hidden_states_49 = None
        hidden_states_50 = (
            l_self_modules_model_modules_layers_modules_6_modules_input_layernorm_parameters_weight_
            * to_29
        )
        l_self_modules_model_modules_layers_modules_6_modules_input_layernorm_parameters_weight_ = (
            to_29
        ) = None
        linear_42 = torch._C._nn.linear(
            hidden_states_50,
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_18 = linear_42.view((1, 21, -1, 4))
        linear_42 = None
        query_states_6 = view_18.transpose(1, 2)
        view_18 = None
        linear_43 = torch._C._nn.linear(
            hidden_states_50,
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_19 = linear_43.view((1, 21, -1, 4))
        linear_43 = None
        key_states_6 = view_19.transpose(1, 2)
        view_19 = None
        linear_44 = torch._C._nn.linear(
            hidden_states_50,
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_50 = l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_20 = linear_44.view((1, 21, -1, 4))
        linear_44 = None
        value_states_6 = view_20.transpose(1, 2)
        view_20 = None
        cos_9 = cos_2.unsqueeze(1)
        sin_9 = sin_2.unsqueeze(1)
        mul_58 = query_states_6 * cos_9
        x1_12 = query_states_6[(Ellipsis, slice(None, 2, None))]
        x2_12 = query_states_6[(Ellipsis, slice(2, None, None))]
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
        x1_13 = key_states_6[(Ellipsis, slice(None, 2, None))]
        x2_13 = key_states_6[(Ellipsis, slice(2, None, None))]
        key_states_6 = None
        neg_13 = -x2_13
        x2_13 = None
        cat_14 = torch.cat((neg_13, x1_13), dim=-1)
        neg_13 = x1_13 = None
        mul_61 = cat_14 * sin_9
        cat_14 = sin_9 = None
        k_embed_6 = mul_60 + mul_61
        mul_60 = mul_61 = None
        attention_mask_7 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 21, None),
            )
        ]
        query_6 = q_embed_6.contiguous()
        q_embed_6 = None
        key_6 = k_embed_6.contiguous()
        k_embed_6 = None
        value_6 = value_states_6.contiguous()
        value_states_6 = None
        item_19 = (
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_scaling.item()
        )
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_scaling = None
        attn_output_24 = torch._C._nn.scaled_dot_product_attention(
            query_6,
            key_6,
            value_6,
            attn_mask=attention_mask_7,
            dropout_p=0.0,
            scale=item_19,
            is_causal=False,
        )
        query_6 = key_6 = value_6 = attention_mask_7 = item_19 = None
        transpose_28 = attn_output_24.transpose(1, 2)
        attn_output_24 = None
        attn_output_25 = transpose_28.contiguous()
        transpose_28 = None
        reshape_6 = attn_output_25.reshape(1, 21, -1)
        attn_output_25 = None
        attn_output_26 = reshape_6.contiguous()
        reshape_6 = None
        attn_output_27 = torch._C._nn.linear(
            attn_output_26,
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_26 = l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_51 = hidden_states_47 + attn_output_27
        hidden_states_47 = attn_output_27 = None
        hidden_states_52 = hidden_states_51.to(torch.float32)
        pow_14 = hidden_states_52.pow(2)
        variance_13 = pow_14.mean(-1, keepdim=True)
        pow_14 = None
        item_20 = (
            l_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_variance_epsilon.item()
        )
        l_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_variance_epsilon = (
            None
        )
        add_40 = variance_13 + item_20
        variance_13 = item_20 = None
        rsqrt_13 = torch.rsqrt(add_40)
        add_40 = None
        hidden_states_53 = hidden_states_52 * rsqrt_13
        hidden_states_52 = rsqrt_13 = None
        to_31 = hidden_states_53.to(torch.float16)
        hidden_states_53 = None
        hidden_states_54 = (
            l_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_
            * to_31
        )
        l_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_ = (
            to_31
        ) = None
        linear_46 = torch._C._nn.linear(
            hidden_states_54,
            l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_6 = torch.nn.functional.silu(linear_46, inplace=False)
        linear_46 = None
        linear_47 = torch._C._nn.linear(
            hidden_states_54,
            l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_54 = l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_64 = silu_6 * linear_47
        silu_6 = linear_47 = None
        down_proj_6 = torch._C._nn.linear(
            mul_64,
            l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_64 = l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_55 = hidden_states_51 + down_proj_6
        hidden_states_51 = down_proj_6 = None
        hidden_states_56 = hidden_states_55.to(torch.float32)
        pow_15 = hidden_states_56.pow(2)
        variance_14 = pow_15.mean(-1, keepdim=True)
        pow_15 = None
        item_21 = (
            l_self_modules_model_modules_layers_modules_7_modules_input_layernorm_variance_epsilon.item()
        )
        l_self_modules_model_modules_layers_modules_7_modules_input_layernorm_variance_epsilon = (
            None
        )
        add_42 = variance_14 + item_21
        variance_14 = item_21 = None
        rsqrt_14 = torch.rsqrt(add_42)
        add_42 = None
        hidden_states_57 = hidden_states_56 * rsqrt_14
        hidden_states_56 = rsqrt_14 = None
        to_33 = hidden_states_57.to(torch.float16)
        hidden_states_57 = None
        hidden_states_58 = (
            l_self_modules_model_modules_layers_modules_7_modules_input_layernorm_parameters_weight_
            * to_33
        )
        l_self_modules_model_modules_layers_modules_7_modules_input_layernorm_parameters_weight_ = (
            to_33
        ) = None
        linear_49 = torch._C._nn.linear(
            hidden_states_58,
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_21 = linear_49.view((1, 21, -1, 4))
        linear_49 = None
        query_states_7 = view_21.transpose(1, 2)
        view_21 = None
        linear_50 = torch._C._nn.linear(
            hidden_states_58,
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_22 = linear_50.view((1, 21, -1, 4))
        linear_50 = None
        key_states_7 = view_22.transpose(1, 2)
        view_22 = None
        linear_51 = torch._C._nn.linear(
            hidden_states_58,
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_58 = l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_23 = linear_51.view((1, 21, -1, 4))
        linear_51 = None
        value_states_7 = view_23.transpose(1, 2)
        view_23 = None
        cos_10 = cos_2.unsqueeze(1)
        cos_2 = None
        sin_10 = sin_2.unsqueeze(1)
        sin_2 = None
        mul_67 = query_states_7 * cos_10
        x1_14 = query_states_7[(Ellipsis, slice(None, 2, None))]
        x2_14 = query_states_7[(Ellipsis, slice(2, None, None))]
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
        x1_15 = key_states_7[(Ellipsis, slice(None, 2, None))]
        x2_15 = key_states_7[(Ellipsis, slice(2, None, None))]
        key_states_7 = None
        neg_15 = -x2_15
        x2_15 = None
        cat_16 = torch.cat((neg_15, x1_15), dim=-1)
        neg_15 = x1_15 = None
        mul_70 = cat_16 * sin_10
        cat_16 = sin_10 = None
        k_embed_7 = mul_69 + mul_70
        mul_69 = mul_70 = None
        attention_mask_8 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 21, None),
            )
        ]
        causal_mask = None
        query_7 = q_embed_7.contiguous()
        q_embed_7 = None
        key_7 = k_embed_7.contiguous()
        k_embed_7 = None
        value_7 = value_states_7.contiguous()
        value_states_7 = None
        item_22 = (
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_scaling.item()
        )
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_scaling = None
        attn_output_28 = torch._C._nn.scaled_dot_product_attention(
            query_7,
            key_7,
            value_7,
            attn_mask=attention_mask_8,
            dropout_p=0.0,
            scale=item_22,
            is_causal=False,
        )
        query_7 = key_7 = value_7 = attention_mask_8 = item_22 = None
        transpose_32 = attn_output_28.transpose(1, 2)
        attn_output_28 = None
        attn_output_29 = transpose_32.contiguous()
        transpose_32 = None
        reshape_7 = attn_output_29.reshape(1, 21, -1)
        attn_output_29 = None
        attn_output_30 = reshape_7.contiguous()
        reshape_7 = None
        attn_output_31 = torch._C._nn.linear(
            attn_output_30,
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_30 = l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_59 = hidden_states_55 + attn_output_31
        hidden_states_55 = attn_output_31 = None
        hidden_states_60 = hidden_states_59.to(torch.float32)
        pow_16 = hidden_states_60.pow(2)
        variance_15 = pow_16.mean(-1, keepdim=True)
        pow_16 = None
        item_23 = (
            l_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_variance_epsilon.item()
        )
        l_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_variance_epsilon = (
            None
        )
        add_46 = variance_15 + item_23
        variance_15 = item_23 = None
        rsqrt_15 = torch.rsqrt(add_46)
        add_46 = None
        hidden_states_61 = hidden_states_60 * rsqrt_15
        hidden_states_60 = rsqrt_15 = None
        to_35 = hidden_states_61.to(torch.float16)
        hidden_states_61 = None
        hidden_states_62 = (
            l_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_
            * to_35
        )
        l_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_ = (
            to_35
        ) = None
        linear_53 = torch._C._nn.linear(
            hidden_states_62,
            l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_7 = torch.nn.functional.silu(linear_53, inplace=False)
        linear_53 = None
        linear_54 = torch._C._nn.linear(
            hidden_states_62,
            l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_62 = l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_73 = silu_7 * linear_54
        silu_7 = linear_54 = None
        down_proj_7 = torch._C._nn.linear(
            mul_73,
            l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_73 = l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_63 = hidden_states_59 + down_proj_7
        hidden_states_59 = down_proj_7 = None
        hidden_states_64 = hidden_states_63.to(torch.float32)
        hidden_states_63 = None
        pow_17 = hidden_states_64.pow(2)
        variance_16 = pow_17.mean(-1, keepdim=True)
        pow_17 = None
        item_24 = l_self_modules_model_modules_norm_variance_epsilon.item()
        l_self_modules_model_modules_norm_variance_epsilon = None
        add_48 = variance_16 + item_24
        variance_16 = item_24 = None
        rsqrt_16 = torch.rsqrt(add_48)
        add_48 = None
        hidden_states_65 = hidden_states_64 * rsqrt_16
        hidden_states_64 = rsqrt_16 = None
        to_37 = hidden_states_65.to(torch.float16)
        hidden_states_65 = None
        hidden_states_66 = l_self_modules_model_modules_norm_parameters_weight_ * to_37
        l_self_modules_model_modules_norm_parameters_weight_ = to_37 = None
        getitem_42 = hidden_states_66[
            (slice(None, None, None), slice(0, None, None), slice(None, None, None))
        ]
        hidden_states_66 = None
        logits = torch._C._nn.linear(
            getitem_42, l_self_modules_lm_head_parameters_weight_, None
        )
        getitem_42 = l_self_modules_lm_head_parameters_weight_ = None
        return (logits,)
