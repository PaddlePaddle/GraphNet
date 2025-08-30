import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_inputs_embeds_: torch.Tensor,
        L_attention_mask_: torch.Tensor,
        L_self_modules_rotary_emb_buffers_inv_freq_: torch.Tensor,
        L_self_modules_layers_modules_0_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_self_attn_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_self_attn_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_self_attn_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_self_attn_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_self_attn_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_self_attn_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_self_attn_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_self_attn_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_self_attn_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_self_attn_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_self_attn_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_self_attn_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_self_attn_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_self_attn_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_self_attn_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_self_attn_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_self_attn_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_self_attn_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_self_attn_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_self_attn_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_self_attn_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_self_attn_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_self_attn_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_self_attn_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_self_attn_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_self_attn_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_self_attn_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_self_attn_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_self_attn_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_self_attn_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_self_attn_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_self_attn_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_self_attn_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_self_attn_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_self_attn_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_self_attn_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_self_attn_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_self_attn_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_final_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_final_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_inputs_embeds_ = L_inputs_embeds_
        l_attention_mask_ = L_attention_mask_
        l_self_modules_rotary_emb_buffers_inv_freq_ = (
            L_self_modules_rotary_emb_buffers_inv_freq_
        )
        l_self_modules_layers_modules_0_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_0_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_0_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_layers_modules_0_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_0_modules_self_attn_modules_dense_parameters_weight_ = L_self_modules_layers_modules_0_modules_self_attn_modules_dense_parameters_weight_
        l_self_modules_layers_modules_0_modules_self_attn_modules_dense_parameters_bias_ = L_self_modules_layers_modules_0_modules_self_attn_modules_dense_parameters_bias_
        l_self_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_1_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_1_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_1_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_layers_modules_1_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_1_modules_self_attn_modules_dense_parameters_weight_ = L_self_modules_layers_modules_1_modules_self_attn_modules_dense_parameters_weight_
        l_self_modules_layers_modules_1_modules_self_attn_modules_dense_parameters_bias_ = L_self_modules_layers_modules_1_modules_self_attn_modules_dense_parameters_bias_
        l_self_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_2_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_2_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_2_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_layers_modules_2_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_self_attn_modules_dense_parameters_weight_ = L_self_modules_layers_modules_2_modules_self_attn_modules_dense_parameters_weight_
        l_self_modules_layers_modules_2_modules_self_attn_modules_dense_parameters_bias_ = L_self_modules_layers_modules_2_modules_self_attn_modules_dense_parameters_bias_
        l_self_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_3_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_3_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_3_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_layers_modules_3_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_3_modules_self_attn_modules_dense_parameters_weight_ = L_self_modules_layers_modules_3_modules_self_attn_modules_dense_parameters_weight_
        l_self_modules_layers_modules_3_modules_self_attn_modules_dense_parameters_bias_ = L_self_modules_layers_modules_3_modules_self_attn_modules_dense_parameters_bias_
        l_self_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_4_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_4_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_4_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_layers_modules_4_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_4_modules_self_attn_modules_dense_parameters_weight_ = L_self_modules_layers_modules_4_modules_self_attn_modules_dense_parameters_weight_
        l_self_modules_layers_modules_4_modules_self_attn_modules_dense_parameters_bias_ = L_self_modules_layers_modules_4_modules_self_attn_modules_dense_parameters_bias_
        l_self_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_5_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_5_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_5_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_layers_modules_5_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_5_modules_self_attn_modules_dense_parameters_weight_ = L_self_modules_layers_modules_5_modules_self_attn_modules_dense_parameters_weight_
        l_self_modules_layers_modules_5_modules_self_attn_modules_dense_parameters_bias_ = L_self_modules_layers_modules_5_modules_self_attn_modules_dense_parameters_bias_
        l_self_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_6_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_6_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_6_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_layers_modules_6_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_6_modules_self_attn_modules_dense_parameters_weight_ = L_self_modules_layers_modules_6_modules_self_attn_modules_dense_parameters_weight_
        l_self_modules_layers_modules_6_modules_self_attn_modules_dense_parameters_bias_ = L_self_modules_layers_modules_6_modules_self_attn_modules_dense_parameters_bias_
        l_self_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_7_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_7_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_7_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_layers_modules_7_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_7_modules_self_attn_modules_dense_parameters_weight_ = L_self_modules_layers_modules_7_modules_self_attn_modules_dense_parameters_weight_
        l_self_modules_layers_modules_7_modules_self_attn_modules_dense_parameters_bias_ = L_self_modules_layers_modules_7_modules_self_attn_modules_dense_parameters_bias_
        l_self_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_8_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_8_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_8_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_layers_modules_8_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_8_modules_self_attn_modules_dense_parameters_weight_ = L_self_modules_layers_modules_8_modules_self_attn_modules_dense_parameters_weight_
        l_self_modules_layers_modules_8_modules_self_attn_modules_dense_parameters_bias_ = L_self_modules_layers_modules_8_modules_self_attn_modules_dense_parameters_bias_
        l_self_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_9_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_9_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_9_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_layers_modules_9_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_9_modules_self_attn_modules_dense_parameters_weight_ = L_self_modules_layers_modules_9_modules_self_attn_modules_dense_parameters_weight_
        l_self_modules_layers_modules_9_modules_self_attn_modules_dense_parameters_bias_ = L_self_modules_layers_modules_9_modules_self_attn_modules_dense_parameters_bias_
        l_self_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_10_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_10_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_10_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_layers_modules_10_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_10_modules_self_attn_modules_dense_parameters_weight_ = L_self_modules_layers_modules_10_modules_self_attn_modules_dense_parameters_weight_
        l_self_modules_layers_modules_10_modules_self_attn_modules_dense_parameters_bias_ = L_self_modules_layers_modules_10_modules_self_attn_modules_dense_parameters_bias_
        l_self_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_11_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_11_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_11_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_layers_modules_11_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_11_modules_self_attn_modules_dense_parameters_weight_ = L_self_modules_layers_modules_11_modules_self_attn_modules_dense_parameters_weight_
        l_self_modules_layers_modules_11_modules_self_attn_modules_dense_parameters_bias_ = L_self_modules_layers_modules_11_modules_self_attn_modules_dense_parameters_bias_
        l_self_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_12_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_12_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_12_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_layers_modules_12_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_12_modules_self_attn_modules_dense_parameters_weight_ = L_self_modules_layers_modules_12_modules_self_attn_modules_dense_parameters_weight_
        l_self_modules_layers_modules_12_modules_self_attn_modules_dense_parameters_bias_ = L_self_modules_layers_modules_12_modules_self_attn_modules_dense_parameters_bias_
        l_self_modules_layers_modules_12_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_12_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_12_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_12_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_12_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_12_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_12_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_12_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_13_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_13_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_13_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_layers_modules_13_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_13_modules_self_attn_modules_dense_parameters_weight_ = L_self_modules_layers_modules_13_modules_self_attn_modules_dense_parameters_weight_
        l_self_modules_layers_modules_13_modules_self_attn_modules_dense_parameters_bias_ = L_self_modules_layers_modules_13_modules_self_attn_modules_dense_parameters_bias_
        l_self_modules_layers_modules_13_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_13_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_13_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_13_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_13_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_13_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_13_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_13_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_14_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_14_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_14_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_layers_modules_14_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_14_modules_self_attn_modules_dense_parameters_weight_ = L_self_modules_layers_modules_14_modules_self_attn_modules_dense_parameters_weight_
        l_self_modules_layers_modules_14_modules_self_attn_modules_dense_parameters_bias_ = L_self_modules_layers_modules_14_modules_self_attn_modules_dense_parameters_bias_
        l_self_modules_layers_modules_14_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_14_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_14_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_14_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_14_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_14_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_14_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_14_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_15_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_15_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_15_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_layers_modules_15_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_15_modules_self_attn_modules_dense_parameters_weight_ = L_self_modules_layers_modules_15_modules_self_attn_modules_dense_parameters_weight_
        l_self_modules_layers_modules_15_modules_self_attn_modules_dense_parameters_bias_ = L_self_modules_layers_modules_15_modules_self_attn_modules_dense_parameters_bias_
        l_self_modules_layers_modules_15_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_15_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_15_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_15_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_15_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_15_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_15_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_15_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_16_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_16_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_16_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_layers_modules_16_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_16_modules_self_attn_modules_dense_parameters_weight_ = L_self_modules_layers_modules_16_modules_self_attn_modules_dense_parameters_weight_
        l_self_modules_layers_modules_16_modules_self_attn_modules_dense_parameters_bias_ = L_self_modules_layers_modules_16_modules_self_attn_modules_dense_parameters_bias_
        l_self_modules_layers_modules_16_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_16_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_16_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_16_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_16_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_16_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_16_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_16_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_17_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_17_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_17_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_layers_modules_17_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_17_modules_self_attn_modules_dense_parameters_weight_ = L_self_modules_layers_modules_17_modules_self_attn_modules_dense_parameters_weight_
        l_self_modules_layers_modules_17_modules_self_attn_modules_dense_parameters_bias_ = L_self_modules_layers_modules_17_modules_self_attn_modules_dense_parameters_bias_
        l_self_modules_layers_modules_17_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_17_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_17_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_17_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_17_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_17_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_17_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_17_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_18_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_18_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_18_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_layers_modules_18_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_18_modules_self_attn_modules_dense_parameters_weight_ = L_self_modules_layers_modules_18_modules_self_attn_modules_dense_parameters_weight_
        l_self_modules_layers_modules_18_modules_self_attn_modules_dense_parameters_bias_ = L_self_modules_layers_modules_18_modules_self_attn_modules_dense_parameters_bias_
        l_self_modules_layers_modules_18_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_18_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_18_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_18_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_18_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_18_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_18_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_18_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_19_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_19_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_19_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_layers_modules_19_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_19_modules_self_attn_modules_dense_parameters_weight_ = L_self_modules_layers_modules_19_modules_self_attn_modules_dense_parameters_weight_
        l_self_modules_layers_modules_19_modules_self_attn_modules_dense_parameters_bias_ = L_self_modules_layers_modules_19_modules_self_attn_modules_dense_parameters_bias_
        l_self_modules_layers_modules_19_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_19_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_19_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_19_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_19_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_19_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_19_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_19_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_20_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_20_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_20_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_layers_modules_20_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_20_modules_self_attn_modules_dense_parameters_weight_ = L_self_modules_layers_modules_20_modules_self_attn_modules_dense_parameters_weight_
        l_self_modules_layers_modules_20_modules_self_attn_modules_dense_parameters_bias_ = L_self_modules_layers_modules_20_modules_self_attn_modules_dense_parameters_bias_
        l_self_modules_layers_modules_20_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_20_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_20_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_20_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_20_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_20_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_20_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_20_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_21_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_21_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_21_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_layers_modules_21_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_21_modules_self_attn_modules_dense_parameters_weight_ = L_self_modules_layers_modules_21_modules_self_attn_modules_dense_parameters_weight_
        l_self_modules_layers_modules_21_modules_self_attn_modules_dense_parameters_bias_ = L_self_modules_layers_modules_21_modules_self_attn_modules_dense_parameters_bias_
        l_self_modules_layers_modules_21_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_21_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_21_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_21_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_21_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_21_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_21_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_21_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_22_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_22_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_22_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_layers_modules_22_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_22_modules_self_attn_modules_dense_parameters_weight_ = L_self_modules_layers_modules_22_modules_self_attn_modules_dense_parameters_weight_
        l_self_modules_layers_modules_22_modules_self_attn_modules_dense_parameters_bias_ = L_self_modules_layers_modules_22_modules_self_attn_modules_dense_parameters_bias_
        l_self_modules_layers_modules_22_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_22_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_22_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_22_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_22_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_22_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_22_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_22_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_23_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_23_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_23_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_layers_modules_23_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_23_modules_self_attn_modules_dense_parameters_weight_ = L_self_modules_layers_modules_23_modules_self_attn_modules_dense_parameters_weight_
        l_self_modules_layers_modules_23_modules_self_attn_modules_dense_parameters_bias_ = L_self_modules_layers_modules_23_modules_self_attn_modules_dense_parameters_bias_
        l_self_modules_layers_modules_23_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_23_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_23_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_23_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_23_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_23_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_23_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_23_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_final_layernorm_parameters_weight_ = (
            L_self_modules_final_layernorm_parameters_weight_
        )
        l_self_modules_final_layernorm_parameters_bias_ = (
            L_self_modules_final_layernorm_parameters_bias_
        )
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
        inputs_embeds = torch.nn.functional.dropout(l_inputs_embeds_, 0.0, False, False)
        l_inputs_embeds_ = None
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
        cos_2 = cos_1.to(dtype=torch.float32)
        cos_1 = None
        sin_2 = sin_1.to(dtype=torch.float32)
        sin_1 = None
        _set_grad_enabled_1 = torch._C._set_grad_enabled(True)
        _set_grad_enabled_1 = None
        _log_api_usage_once = torch._C._log_api_usage_once("python.nn_module")
        _log_api_usage_once = None
        hidden_states = torch.nn.functional.layer_norm(
            inputs_embeds,
            (2048,),
            l_self_modules_layers_modules_0_modules_input_layernorm_parameters_weight_,
            l_self_modules_layers_modules_0_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_0_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_layers_modules_0_modules_input_layernorm_parameters_bias_
        ) = None
        linear = torch._C._nn.linear(
            hidden_states,
            l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_1 = linear.view((1, 2, -1, 64))
        linear = None
        query_states = view_1.transpose(1, 2)
        view_1 = None
        linear_1 = torch._C._nn.linear(
            hidden_states,
            l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_2 = linear_1.view((1, 2, -1, 64))
        linear_1 = None
        key_states = view_2.transpose(1, 2)
        view_2 = None
        linear_2 = torch._C._nn.linear(
            hidden_states,
            l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_3 = linear_2.view((1, 2, -1, 64))
        linear_2 = None
        value_states = view_3.transpose(1, 2)
        view_3 = None
        query_rot = query_states[(Ellipsis, slice(None, 32, None))]
        query_pass = query_states[(Ellipsis, slice(32, None, None))]
        query_states = None
        key_rot = key_states[(Ellipsis, slice(None, 32, None))]
        key_pass = key_states[(Ellipsis, slice(32, None, None))]
        key_states = None
        cos_3 = cos_2.unsqueeze(1)
        sin_3 = sin_2.unsqueeze(1)
        mul_3 = query_rot * cos_3
        x1 = query_rot[(Ellipsis, slice(None, 16, None))]
        x2 = query_rot[(Ellipsis, slice(16, None, None))]
        query_rot = None
        neg = -x2
        x2 = None
        cat_1 = torch.cat((neg, x1), dim=-1)
        neg = x1 = None
        mul_4 = cat_1 * sin_3
        cat_1 = None
        q_embed = mul_3 + mul_4
        mul_3 = mul_4 = None
        mul_5 = key_rot * cos_3
        cos_3 = None
        x1_1 = key_rot[(Ellipsis, slice(None, 16, None))]
        x2_1 = key_rot[(Ellipsis, slice(16, None, None))]
        key_rot = None
        neg_1 = -x2_1
        x2_1 = None
        cat_2 = torch.cat((neg_1, x1_1), dim=-1)
        neg_1 = x1_1 = None
        mul_6 = cat_2 * sin_3
        cat_2 = sin_3 = None
        k_embed = mul_5 + mul_6
        mul_5 = mul_6 = None
        query_states_1 = torch.cat((q_embed, query_pass), dim=-1)
        q_embed = query_pass = None
        key_states_1 = torch.cat((k_embed, key_pass), dim=-1)
        k_embed = key_pass = None
        attention_mask_1 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query = query_states_1.contiguous()
        query_states_1 = None
        key = key_states_1.contiguous()
        value = value_states.contiguous()
        attn_output = torch._C._nn.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask_1,
            dropout_p=0.0,
            scale=0.125,
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
            l_self_modules_layers_modules_0_modules_self_attn_modules_dense_parameters_weight_,
            l_self_modules_layers_modules_0_modules_self_attn_modules_dense_parameters_bias_,
        )
        attn_output_2 = l_self_modules_layers_modules_0_modules_self_attn_modules_dense_parameters_weight_ = l_self_modules_layers_modules_0_modules_self_attn_modules_dense_parameters_bias_ = (None)
        attn_outputs = torch.nn.functional.dropout(attn_output_3, 0.0, False, False)
        attn_output_3 = None
        hidden_states_1 = torch._C._nn.linear(
            hidden_states,
            l_self_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states = (
            l_self_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_7 = 0.5 * hidden_states_1
        pow_1 = torch.pow(hidden_states_1, 3.0)
        mul_8 = 0.044715 * pow_1
        pow_1 = None
        add_2 = hidden_states_1 + mul_8
        hidden_states_1 = mul_8 = None
        mul_9 = 0.7978845608028654 * add_2
        add_2 = None
        tanh = torch.tanh(mul_9)
        mul_9 = None
        add_3 = 1.0 + tanh
        tanh = None
        hidden_states_2 = mul_7 * add_3
        mul_7 = add_3 = None
        hidden_states_3 = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_2 = (
            l_self_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        feed_forward_hidden_states = torch.nn.functional.dropout(
            hidden_states_3, 0.0, False, False
        )
        hidden_states_3 = None
        add_4 = attn_outputs + feed_forward_hidden_states
        attn_outputs = feed_forward_hidden_states = None
        hidden_states_4 = add_4 + inputs_embeds
        add_4 = inputs_embeds = None
        hidden_states_5 = torch.nn.functional.layer_norm(
            hidden_states_4,
            (2048,),
            l_self_modules_layers_modules_1_modules_input_layernorm_parameters_weight_,
            l_self_modules_layers_modules_1_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_1_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_layers_modules_1_modules_input_layernorm_parameters_bias_
        ) = None
        linear_6 = torch._C._nn.linear(
            hidden_states_5,
            l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_4 = linear_6.view((1, 2, -1, 64))
        linear_6 = None
        query_states_2 = view_4.transpose(1, 2)
        view_4 = None
        linear_7 = torch._C._nn.linear(
            hidden_states_5,
            l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_5 = linear_7.view((1, 2, -1, 64))
        linear_7 = None
        key_states_2 = view_5.transpose(1, 2)
        view_5 = None
        linear_8 = torch._C._nn.linear(
            hidden_states_5,
            l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_6 = linear_8.view((1, 2, -1, 64))
        linear_8 = None
        value_states_1 = view_6.transpose(1, 2)
        view_6 = None
        query_rot_1 = query_states_2[(Ellipsis, slice(None, 32, None))]
        query_pass_1 = query_states_2[(Ellipsis, slice(32, None, None))]
        query_states_2 = None
        key_rot_1 = key_states_2[(Ellipsis, slice(None, 32, None))]
        key_pass_1 = key_states_2[(Ellipsis, slice(32, None, None))]
        key_states_2 = None
        cos_4 = cos_2.unsqueeze(1)
        sin_4 = sin_2.unsqueeze(1)
        mul_11 = query_rot_1 * cos_4
        x1_2 = query_rot_1[(Ellipsis, slice(None, 16, None))]
        x2_2 = query_rot_1[(Ellipsis, slice(16, None, None))]
        query_rot_1 = None
        neg_2 = -x2_2
        x2_2 = None
        cat_5 = torch.cat((neg_2, x1_2), dim=-1)
        neg_2 = x1_2 = None
        mul_12 = cat_5 * sin_4
        cat_5 = None
        q_embed_1 = mul_11 + mul_12
        mul_11 = mul_12 = None
        mul_13 = key_rot_1 * cos_4
        cos_4 = None
        x1_3 = key_rot_1[(Ellipsis, slice(None, 16, None))]
        x2_3 = key_rot_1[(Ellipsis, slice(16, None, None))]
        key_rot_1 = None
        neg_3 = -x2_3
        x2_3 = None
        cat_6 = torch.cat((neg_3, x1_3), dim=-1)
        neg_3 = x1_3 = None
        mul_14 = cat_6 * sin_4
        cat_6 = sin_4 = None
        k_embed_1 = mul_13 + mul_14
        mul_13 = mul_14 = None
        query_states_3 = torch.cat((q_embed_1, query_pass_1), dim=-1)
        q_embed_1 = query_pass_1 = None
        key_states_3 = torch.cat((k_embed_1, key_pass_1), dim=-1)
        k_embed_1 = key_pass_1 = None
        attention_mask_2 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_1 = query_states_3.contiguous()
        query_states_3 = None
        key_1 = key_states_3.contiguous()
        value_1 = value_states_1.contiguous()
        attn_output_4 = torch._C._nn.scaled_dot_product_attention(
            query_1,
            key_1,
            value_1,
            attn_mask=attention_mask_2,
            dropout_p=0.0,
            scale=0.125,
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
            l_self_modules_layers_modules_1_modules_self_attn_modules_dense_parameters_weight_,
            l_self_modules_layers_modules_1_modules_self_attn_modules_dense_parameters_bias_,
        )
        attn_output_6 = l_self_modules_layers_modules_1_modules_self_attn_modules_dense_parameters_weight_ = l_self_modules_layers_modules_1_modules_self_attn_modules_dense_parameters_bias_ = (None)
        attn_outputs_1 = torch.nn.functional.dropout(attn_output_7, 0.0, False, False)
        attn_output_7 = None
        hidden_states_6 = torch._C._nn.linear(
            hidden_states_5,
            l_self_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_5 = (
            l_self_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_15 = 0.5 * hidden_states_6
        pow_2 = torch.pow(hidden_states_6, 3.0)
        mul_16 = 0.044715 * pow_2
        pow_2 = None
        add_8 = hidden_states_6 + mul_16
        hidden_states_6 = mul_16 = None
        mul_17 = 0.7978845608028654 * add_8
        add_8 = None
        tanh_1 = torch.tanh(mul_17)
        mul_17 = None
        add_9 = 1.0 + tanh_1
        tanh_1 = None
        hidden_states_7 = mul_15 * add_9
        mul_15 = add_9 = None
        hidden_states_8 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_7 = (
            l_self_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        feed_forward_hidden_states_1 = torch.nn.functional.dropout(
            hidden_states_8, 0.0, False, False
        )
        hidden_states_8 = None
        add_10 = attn_outputs_1 + feed_forward_hidden_states_1
        attn_outputs_1 = feed_forward_hidden_states_1 = None
        hidden_states_9 = add_10 + hidden_states_4
        add_10 = hidden_states_4 = None
        hidden_states_10 = torch.nn.functional.layer_norm(
            hidden_states_9,
            (2048,),
            l_self_modules_layers_modules_2_modules_input_layernorm_parameters_weight_,
            l_self_modules_layers_modules_2_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_2_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_layers_modules_2_modules_input_layernorm_parameters_bias_
        ) = None
        linear_12 = torch._C._nn.linear(
            hidden_states_10,
            l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_7 = linear_12.view((1, 2, -1, 64))
        linear_12 = None
        query_states_4 = view_7.transpose(1, 2)
        view_7 = None
        linear_13 = torch._C._nn.linear(
            hidden_states_10,
            l_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_8 = linear_13.view((1, 2, -1, 64))
        linear_13 = None
        key_states_4 = view_8.transpose(1, 2)
        view_8 = None
        linear_14 = torch._C._nn.linear(
            hidden_states_10,
            l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_9 = linear_14.view((1, 2, -1, 64))
        linear_14 = None
        value_states_2 = view_9.transpose(1, 2)
        view_9 = None
        query_rot_2 = query_states_4[(Ellipsis, slice(None, 32, None))]
        query_pass_2 = query_states_4[(Ellipsis, slice(32, None, None))]
        query_states_4 = None
        key_rot_2 = key_states_4[(Ellipsis, slice(None, 32, None))]
        key_pass_2 = key_states_4[(Ellipsis, slice(32, None, None))]
        key_states_4 = None
        cos_5 = cos_2.unsqueeze(1)
        sin_5 = sin_2.unsqueeze(1)
        mul_19 = query_rot_2 * cos_5
        x1_4 = query_rot_2[(Ellipsis, slice(None, 16, None))]
        x2_4 = query_rot_2[(Ellipsis, slice(16, None, None))]
        query_rot_2 = None
        neg_4 = -x2_4
        x2_4 = None
        cat_9 = torch.cat((neg_4, x1_4), dim=-1)
        neg_4 = x1_4 = None
        mul_20 = cat_9 * sin_5
        cat_9 = None
        q_embed_2 = mul_19 + mul_20
        mul_19 = mul_20 = None
        mul_21 = key_rot_2 * cos_5
        cos_5 = None
        x1_5 = key_rot_2[(Ellipsis, slice(None, 16, None))]
        x2_5 = key_rot_2[(Ellipsis, slice(16, None, None))]
        key_rot_2 = None
        neg_5 = -x2_5
        x2_5 = None
        cat_10 = torch.cat((neg_5, x1_5), dim=-1)
        neg_5 = x1_5 = None
        mul_22 = cat_10 * sin_5
        cat_10 = sin_5 = None
        k_embed_2 = mul_21 + mul_22
        mul_21 = mul_22 = None
        query_states_5 = torch.cat((q_embed_2, query_pass_2), dim=-1)
        q_embed_2 = query_pass_2 = None
        key_states_5 = torch.cat((k_embed_2, key_pass_2), dim=-1)
        k_embed_2 = key_pass_2 = None
        attention_mask_3 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_2 = query_states_5.contiguous()
        query_states_5 = None
        key_2 = key_states_5.contiguous()
        value_2 = value_states_2.contiguous()
        attn_output_8 = torch._C._nn.scaled_dot_product_attention(
            query_2,
            key_2,
            value_2,
            attn_mask=attention_mask_3,
            dropout_p=0.0,
            scale=0.125,
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
            l_self_modules_layers_modules_2_modules_self_attn_modules_dense_parameters_weight_,
            l_self_modules_layers_modules_2_modules_self_attn_modules_dense_parameters_bias_,
        )
        attn_output_10 = l_self_modules_layers_modules_2_modules_self_attn_modules_dense_parameters_weight_ = l_self_modules_layers_modules_2_modules_self_attn_modules_dense_parameters_bias_ = (None)
        attn_outputs_2 = torch.nn.functional.dropout(attn_output_11, 0.0, False, False)
        attn_output_11 = None
        hidden_states_11 = torch._C._nn.linear(
            hidden_states_10,
            l_self_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_10 = (
            l_self_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_23 = 0.5 * hidden_states_11
        pow_3 = torch.pow(hidden_states_11, 3.0)
        mul_24 = 0.044715 * pow_3
        pow_3 = None
        add_14 = hidden_states_11 + mul_24
        hidden_states_11 = mul_24 = None
        mul_25 = 0.7978845608028654 * add_14
        add_14 = None
        tanh_2 = torch.tanh(mul_25)
        mul_25 = None
        add_15 = 1.0 + tanh_2
        tanh_2 = None
        hidden_states_12 = mul_23 * add_15
        mul_23 = add_15 = None
        hidden_states_13 = torch._C._nn.linear(
            hidden_states_12,
            l_self_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_12 = (
            l_self_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        feed_forward_hidden_states_2 = torch.nn.functional.dropout(
            hidden_states_13, 0.0, False, False
        )
        hidden_states_13 = None
        add_16 = attn_outputs_2 + feed_forward_hidden_states_2
        attn_outputs_2 = feed_forward_hidden_states_2 = None
        hidden_states_14 = add_16 + hidden_states_9
        add_16 = hidden_states_9 = None
        hidden_states_15 = torch.nn.functional.layer_norm(
            hidden_states_14,
            (2048,),
            l_self_modules_layers_modules_3_modules_input_layernorm_parameters_weight_,
            l_self_modules_layers_modules_3_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_3_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_layers_modules_3_modules_input_layernorm_parameters_bias_
        ) = None
        linear_18 = torch._C._nn.linear(
            hidden_states_15,
            l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_10 = linear_18.view((1, 2, -1, 64))
        linear_18 = None
        query_states_6 = view_10.transpose(1, 2)
        view_10 = None
        linear_19 = torch._C._nn.linear(
            hidden_states_15,
            l_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_11 = linear_19.view((1, 2, -1, 64))
        linear_19 = None
        key_states_6 = view_11.transpose(1, 2)
        view_11 = None
        linear_20 = torch._C._nn.linear(
            hidden_states_15,
            l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_12 = linear_20.view((1, 2, -1, 64))
        linear_20 = None
        value_states_3 = view_12.transpose(1, 2)
        view_12 = None
        query_rot_3 = query_states_6[(Ellipsis, slice(None, 32, None))]
        query_pass_3 = query_states_6[(Ellipsis, slice(32, None, None))]
        query_states_6 = None
        key_rot_3 = key_states_6[(Ellipsis, slice(None, 32, None))]
        key_pass_3 = key_states_6[(Ellipsis, slice(32, None, None))]
        key_states_6 = None
        cos_6 = cos_2.unsqueeze(1)
        sin_6 = sin_2.unsqueeze(1)
        mul_27 = query_rot_3 * cos_6
        x1_6 = query_rot_3[(Ellipsis, slice(None, 16, None))]
        x2_6 = query_rot_3[(Ellipsis, slice(16, None, None))]
        query_rot_3 = None
        neg_6 = -x2_6
        x2_6 = None
        cat_13 = torch.cat((neg_6, x1_6), dim=-1)
        neg_6 = x1_6 = None
        mul_28 = cat_13 * sin_6
        cat_13 = None
        q_embed_3 = mul_27 + mul_28
        mul_27 = mul_28 = None
        mul_29 = key_rot_3 * cos_6
        cos_6 = None
        x1_7 = key_rot_3[(Ellipsis, slice(None, 16, None))]
        x2_7 = key_rot_3[(Ellipsis, slice(16, None, None))]
        key_rot_3 = None
        neg_7 = -x2_7
        x2_7 = None
        cat_14 = torch.cat((neg_7, x1_7), dim=-1)
        neg_7 = x1_7 = None
        mul_30 = cat_14 * sin_6
        cat_14 = sin_6 = None
        k_embed_3 = mul_29 + mul_30
        mul_29 = mul_30 = None
        query_states_7 = torch.cat((q_embed_3, query_pass_3), dim=-1)
        q_embed_3 = query_pass_3 = None
        key_states_7 = torch.cat((k_embed_3, key_pass_3), dim=-1)
        k_embed_3 = key_pass_3 = None
        attention_mask_4 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_3 = query_states_7.contiguous()
        query_states_7 = None
        key_3 = key_states_7.contiguous()
        value_3 = value_states_3.contiguous()
        attn_output_12 = torch._C._nn.scaled_dot_product_attention(
            query_3,
            key_3,
            value_3,
            attn_mask=attention_mask_4,
            dropout_p=0.0,
            scale=0.125,
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
            l_self_modules_layers_modules_3_modules_self_attn_modules_dense_parameters_weight_,
            l_self_modules_layers_modules_3_modules_self_attn_modules_dense_parameters_bias_,
        )
        attn_output_14 = l_self_modules_layers_modules_3_modules_self_attn_modules_dense_parameters_weight_ = l_self_modules_layers_modules_3_modules_self_attn_modules_dense_parameters_bias_ = (None)
        attn_outputs_3 = torch.nn.functional.dropout(attn_output_15, 0.0, False, False)
        attn_output_15 = None
        hidden_states_16 = torch._C._nn.linear(
            hidden_states_15,
            l_self_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_15 = (
            l_self_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_31 = 0.5 * hidden_states_16
        pow_4 = torch.pow(hidden_states_16, 3.0)
        mul_32 = 0.044715 * pow_4
        pow_4 = None
        add_20 = hidden_states_16 + mul_32
        hidden_states_16 = mul_32 = None
        mul_33 = 0.7978845608028654 * add_20
        add_20 = None
        tanh_3 = torch.tanh(mul_33)
        mul_33 = None
        add_21 = 1.0 + tanh_3
        tanh_3 = None
        hidden_states_17 = mul_31 * add_21
        mul_31 = add_21 = None
        hidden_states_18 = torch._C._nn.linear(
            hidden_states_17,
            l_self_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_17 = (
            l_self_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        feed_forward_hidden_states_3 = torch.nn.functional.dropout(
            hidden_states_18, 0.0, False, False
        )
        hidden_states_18 = None
        add_22 = attn_outputs_3 + feed_forward_hidden_states_3
        attn_outputs_3 = feed_forward_hidden_states_3 = None
        hidden_states_19 = add_22 + hidden_states_14
        add_22 = hidden_states_14 = None
        hidden_states_20 = torch.nn.functional.layer_norm(
            hidden_states_19,
            (2048,),
            l_self_modules_layers_modules_4_modules_input_layernorm_parameters_weight_,
            l_self_modules_layers_modules_4_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_4_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_layers_modules_4_modules_input_layernorm_parameters_bias_
        ) = None
        linear_24 = torch._C._nn.linear(
            hidden_states_20,
            l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_13 = linear_24.view((1, 2, -1, 64))
        linear_24 = None
        query_states_8 = view_13.transpose(1, 2)
        view_13 = None
        linear_25 = torch._C._nn.linear(
            hidden_states_20,
            l_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_14 = linear_25.view((1, 2, -1, 64))
        linear_25 = None
        key_states_8 = view_14.transpose(1, 2)
        view_14 = None
        linear_26 = torch._C._nn.linear(
            hidden_states_20,
            l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_15 = linear_26.view((1, 2, -1, 64))
        linear_26 = None
        value_states_4 = view_15.transpose(1, 2)
        view_15 = None
        query_rot_4 = query_states_8[(Ellipsis, slice(None, 32, None))]
        query_pass_4 = query_states_8[(Ellipsis, slice(32, None, None))]
        query_states_8 = None
        key_rot_4 = key_states_8[(Ellipsis, slice(None, 32, None))]
        key_pass_4 = key_states_8[(Ellipsis, slice(32, None, None))]
        key_states_8 = None
        cos_7 = cos_2.unsqueeze(1)
        sin_7 = sin_2.unsqueeze(1)
        mul_35 = query_rot_4 * cos_7
        x1_8 = query_rot_4[(Ellipsis, slice(None, 16, None))]
        x2_8 = query_rot_4[(Ellipsis, slice(16, None, None))]
        query_rot_4 = None
        neg_8 = -x2_8
        x2_8 = None
        cat_17 = torch.cat((neg_8, x1_8), dim=-1)
        neg_8 = x1_8 = None
        mul_36 = cat_17 * sin_7
        cat_17 = None
        q_embed_4 = mul_35 + mul_36
        mul_35 = mul_36 = None
        mul_37 = key_rot_4 * cos_7
        cos_7 = None
        x1_9 = key_rot_4[(Ellipsis, slice(None, 16, None))]
        x2_9 = key_rot_4[(Ellipsis, slice(16, None, None))]
        key_rot_4 = None
        neg_9 = -x2_9
        x2_9 = None
        cat_18 = torch.cat((neg_9, x1_9), dim=-1)
        neg_9 = x1_9 = None
        mul_38 = cat_18 * sin_7
        cat_18 = sin_7 = None
        k_embed_4 = mul_37 + mul_38
        mul_37 = mul_38 = None
        query_states_9 = torch.cat((q_embed_4, query_pass_4), dim=-1)
        q_embed_4 = query_pass_4 = None
        key_states_9 = torch.cat((k_embed_4, key_pass_4), dim=-1)
        k_embed_4 = key_pass_4 = None
        attention_mask_5 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_4 = query_states_9.contiguous()
        query_states_9 = None
        key_4 = key_states_9.contiguous()
        value_4 = value_states_4.contiguous()
        attn_output_16 = torch._C._nn.scaled_dot_product_attention(
            query_4,
            key_4,
            value_4,
            attn_mask=attention_mask_5,
            dropout_p=0.0,
            scale=0.125,
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
            l_self_modules_layers_modules_4_modules_self_attn_modules_dense_parameters_weight_,
            l_self_modules_layers_modules_4_modules_self_attn_modules_dense_parameters_bias_,
        )
        attn_output_18 = l_self_modules_layers_modules_4_modules_self_attn_modules_dense_parameters_weight_ = l_self_modules_layers_modules_4_modules_self_attn_modules_dense_parameters_bias_ = (None)
        attn_outputs_4 = torch.nn.functional.dropout(attn_output_19, 0.0, False, False)
        attn_output_19 = None
        hidden_states_21 = torch._C._nn.linear(
            hidden_states_20,
            l_self_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_20 = (
            l_self_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_39 = 0.5 * hidden_states_21
        pow_5 = torch.pow(hidden_states_21, 3.0)
        mul_40 = 0.044715 * pow_5
        pow_5 = None
        add_26 = hidden_states_21 + mul_40
        hidden_states_21 = mul_40 = None
        mul_41 = 0.7978845608028654 * add_26
        add_26 = None
        tanh_4 = torch.tanh(mul_41)
        mul_41 = None
        add_27 = 1.0 + tanh_4
        tanh_4 = None
        hidden_states_22 = mul_39 * add_27
        mul_39 = add_27 = None
        hidden_states_23 = torch._C._nn.linear(
            hidden_states_22,
            l_self_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_22 = (
            l_self_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        feed_forward_hidden_states_4 = torch.nn.functional.dropout(
            hidden_states_23, 0.0, False, False
        )
        hidden_states_23 = None
        add_28 = attn_outputs_4 + feed_forward_hidden_states_4
        attn_outputs_4 = feed_forward_hidden_states_4 = None
        hidden_states_24 = add_28 + hidden_states_19
        add_28 = hidden_states_19 = None
        hidden_states_25 = torch.nn.functional.layer_norm(
            hidden_states_24,
            (2048,),
            l_self_modules_layers_modules_5_modules_input_layernorm_parameters_weight_,
            l_self_modules_layers_modules_5_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_5_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_layers_modules_5_modules_input_layernorm_parameters_bias_
        ) = None
        linear_30 = torch._C._nn.linear(
            hidden_states_25,
            l_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_16 = linear_30.view((1, 2, -1, 64))
        linear_30 = None
        query_states_10 = view_16.transpose(1, 2)
        view_16 = None
        linear_31 = torch._C._nn.linear(
            hidden_states_25,
            l_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_17 = linear_31.view((1, 2, -1, 64))
        linear_31 = None
        key_states_10 = view_17.transpose(1, 2)
        view_17 = None
        linear_32 = torch._C._nn.linear(
            hidden_states_25,
            l_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_18 = linear_32.view((1, 2, -1, 64))
        linear_32 = None
        value_states_5 = view_18.transpose(1, 2)
        view_18 = None
        query_rot_5 = query_states_10[(Ellipsis, slice(None, 32, None))]
        query_pass_5 = query_states_10[(Ellipsis, slice(32, None, None))]
        query_states_10 = None
        key_rot_5 = key_states_10[(Ellipsis, slice(None, 32, None))]
        key_pass_5 = key_states_10[(Ellipsis, slice(32, None, None))]
        key_states_10 = None
        cos_8 = cos_2.unsqueeze(1)
        sin_8 = sin_2.unsqueeze(1)
        mul_43 = query_rot_5 * cos_8
        x1_10 = query_rot_5[(Ellipsis, slice(None, 16, None))]
        x2_10 = query_rot_5[(Ellipsis, slice(16, None, None))]
        query_rot_5 = None
        neg_10 = -x2_10
        x2_10 = None
        cat_21 = torch.cat((neg_10, x1_10), dim=-1)
        neg_10 = x1_10 = None
        mul_44 = cat_21 * sin_8
        cat_21 = None
        q_embed_5 = mul_43 + mul_44
        mul_43 = mul_44 = None
        mul_45 = key_rot_5 * cos_8
        cos_8 = None
        x1_11 = key_rot_5[(Ellipsis, slice(None, 16, None))]
        x2_11 = key_rot_5[(Ellipsis, slice(16, None, None))]
        key_rot_5 = None
        neg_11 = -x2_11
        x2_11 = None
        cat_22 = torch.cat((neg_11, x1_11), dim=-1)
        neg_11 = x1_11 = None
        mul_46 = cat_22 * sin_8
        cat_22 = sin_8 = None
        k_embed_5 = mul_45 + mul_46
        mul_45 = mul_46 = None
        query_states_11 = torch.cat((q_embed_5, query_pass_5), dim=-1)
        q_embed_5 = query_pass_5 = None
        key_states_11 = torch.cat((k_embed_5, key_pass_5), dim=-1)
        k_embed_5 = key_pass_5 = None
        attention_mask_6 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_5 = query_states_11.contiguous()
        query_states_11 = None
        key_5 = key_states_11.contiguous()
        value_5 = value_states_5.contiguous()
        attn_output_20 = torch._C._nn.scaled_dot_product_attention(
            query_5,
            key_5,
            value_5,
            attn_mask=attention_mask_6,
            dropout_p=0.0,
            scale=0.125,
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
            l_self_modules_layers_modules_5_modules_self_attn_modules_dense_parameters_weight_,
            l_self_modules_layers_modules_5_modules_self_attn_modules_dense_parameters_bias_,
        )
        attn_output_22 = l_self_modules_layers_modules_5_modules_self_attn_modules_dense_parameters_weight_ = l_self_modules_layers_modules_5_modules_self_attn_modules_dense_parameters_bias_ = (None)
        attn_outputs_5 = torch.nn.functional.dropout(attn_output_23, 0.0, False, False)
        attn_output_23 = None
        hidden_states_26 = torch._C._nn.linear(
            hidden_states_25,
            l_self_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_25 = (
            l_self_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_47 = 0.5 * hidden_states_26
        pow_6 = torch.pow(hidden_states_26, 3.0)
        mul_48 = 0.044715 * pow_6
        pow_6 = None
        add_32 = hidden_states_26 + mul_48
        hidden_states_26 = mul_48 = None
        mul_49 = 0.7978845608028654 * add_32
        add_32 = None
        tanh_5 = torch.tanh(mul_49)
        mul_49 = None
        add_33 = 1.0 + tanh_5
        tanh_5 = None
        hidden_states_27 = mul_47 * add_33
        mul_47 = add_33 = None
        hidden_states_28 = torch._C._nn.linear(
            hidden_states_27,
            l_self_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_27 = (
            l_self_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        feed_forward_hidden_states_5 = torch.nn.functional.dropout(
            hidden_states_28, 0.0, False, False
        )
        hidden_states_28 = None
        add_34 = attn_outputs_5 + feed_forward_hidden_states_5
        attn_outputs_5 = feed_forward_hidden_states_5 = None
        hidden_states_29 = add_34 + hidden_states_24
        add_34 = hidden_states_24 = None
        hidden_states_30 = torch.nn.functional.layer_norm(
            hidden_states_29,
            (2048,),
            l_self_modules_layers_modules_6_modules_input_layernorm_parameters_weight_,
            l_self_modules_layers_modules_6_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_6_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_layers_modules_6_modules_input_layernorm_parameters_bias_
        ) = None
        linear_36 = torch._C._nn.linear(
            hidden_states_30,
            l_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_19 = linear_36.view((1, 2, -1, 64))
        linear_36 = None
        query_states_12 = view_19.transpose(1, 2)
        view_19 = None
        linear_37 = torch._C._nn.linear(
            hidden_states_30,
            l_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_20 = linear_37.view((1, 2, -1, 64))
        linear_37 = None
        key_states_12 = view_20.transpose(1, 2)
        view_20 = None
        linear_38 = torch._C._nn.linear(
            hidden_states_30,
            l_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_21 = linear_38.view((1, 2, -1, 64))
        linear_38 = None
        value_states_6 = view_21.transpose(1, 2)
        view_21 = None
        query_rot_6 = query_states_12[(Ellipsis, slice(None, 32, None))]
        query_pass_6 = query_states_12[(Ellipsis, slice(32, None, None))]
        query_states_12 = None
        key_rot_6 = key_states_12[(Ellipsis, slice(None, 32, None))]
        key_pass_6 = key_states_12[(Ellipsis, slice(32, None, None))]
        key_states_12 = None
        cos_9 = cos_2.unsqueeze(1)
        sin_9 = sin_2.unsqueeze(1)
        mul_51 = query_rot_6 * cos_9
        x1_12 = query_rot_6[(Ellipsis, slice(None, 16, None))]
        x2_12 = query_rot_6[(Ellipsis, slice(16, None, None))]
        query_rot_6 = None
        neg_12 = -x2_12
        x2_12 = None
        cat_25 = torch.cat((neg_12, x1_12), dim=-1)
        neg_12 = x1_12 = None
        mul_52 = cat_25 * sin_9
        cat_25 = None
        q_embed_6 = mul_51 + mul_52
        mul_51 = mul_52 = None
        mul_53 = key_rot_6 * cos_9
        cos_9 = None
        x1_13 = key_rot_6[(Ellipsis, slice(None, 16, None))]
        x2_13 = key_rot_6[(Ellipsis, slice(16, None, None))]
        key_rot_6 = None
        neg_13 = -x2_13
        x2_13 = None
        cat_26 = torch.cat((neg_13, x1_13), dim=-1)
        neg_13 = x1_13 = None
        mul_54 = cat_26 * sin_9
        cat_26 = sin_9 = None
        k_embed_6 = mul_53 + mul_54
        mul_53 = mul_54 = None
        query_states_13 = torch.cat((q_embed_6, query_pass_6), dim=-1)
        q_embed_6 = query_pass_6 = None
        key_states_13 = torch.cat((k_embed_6, key_pass_6), dim=-1)
        k_embed_6 = key_pass_6 = None
        attention_mask_7 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_6 = query_states_13.contiguous()
        query_states_13 = None
        key_6 = key_states_13.contiguous()
        value_6 = value_states_6.contiguous()
        attn_output_24 = torch._C._nn.scaled_dot_product_attention(
            query_6,
            key_6,
            value_6,
            attn_mask=attention_mask_7,
            dropout_p=0.0,
            scale=0.125,
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
            l_self_modules_layers_modules_6_modules_self_attn_modules_dense_parameters_weight_,
            l_self_modules_layers_modules_6_modules_self_attn_modules_dense_parameters_bias_,
        )
        attn_output_26 = l_self_modules_layers_modules_6_modules_self_attn_modules_dense_parameters_weight_ = l_self_modules_layers_modules_6_modules_self_attn_modules_dense_parameters_bias_ = (None)
        attn_outputs_6 = torch.nn.functional.dropout(attn_output_27, 0.0, False, False)
        attn_output_27 = None
        hidden_states_31 = torch._C._nn.linear(
            hidden_states_30,
            l_self_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_30 = (
            l_self_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_55 = 0.5 * hidden_states_31
        pow_7 = torch.pow(hidden_states_31, 3.0)
        mul_56 = 0.044715 * pow_7
        pow_7 = None
        add_38 = hidden_states_31 + mul_56
        hidden_states_31 = mul_56 = None
        mul_57 = 0.7978845608028654 * add_38
        add_38 = None
        tanh_6 = torch.tanh(mul_57)
        mul_57 = None
        add_39 = 1.0 + tanh_6
        tanh_6 = None
        hidden_states_32 = mul_55 * add_39
        mul_55 = add_39 = None
        hidden_states_33 = torch._C._nn.linear(
            hidden_states_32,
            l_self_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_32 = (
            l_self_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        feed_forward_hidden_states_6 = torch.nn.functional.dropout(
            hidden_states_33, 0.0, False, False
        )
        hidden_states_33 = None
        add_40 = attn_outputs_6 + feed_forward_hidden_states_6
        attn_outputs_6 = feed_forward_hidden_states_6 = None
        hidden_states_34 = add_40 + hidden_states_29
        add_40 = hidden_states_29 = None
        hidden_states_35 = torch.nn.functional.layer_norm(
            hidden_states_34,
            (2048,),
            l_self_modules_layers_modules_7_modules_input_layernorm_parameters_weight_,
            l_self_modules_layers_modules_7_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_7_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_layers_modules_7_modules_input_layernorm_parameters_bias_
        ) = None
        linear_42 = torch._C._nn.linear(
            hidden_states_35,
            l_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_22 = linear_42.view((1, 2, -1, 64))
        linear_42 = None
        query_states_14 = view_22.transpose(1, 2)
        view_22 = None
        linear_43 = torch._C._nn.linear(
            hidden_states_35,
            l_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_23 = linear_43.view((1, 2, -1, 64))
        linear_43 = None
        key_states_14 = view_23.transpose(1, 2)
        view_23 = None
        linear_44 = torch._C._nn.linear(
            hidden_states_35,
            l_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_24 = linear_44.view((1, 2, -1, 64))
        linear_44 = None
        value_states_7 = view_24.transpose(1, 2)
        view_24 = None
        query_rot_7 = query_states_14[(Ellipsis, slice(None, 32, None))]
        query_pass_7 = query_states_14[(Ellipsis, slice(32, None, None))]
        query_states_14 = None
        key_rot_7 = key_states_14[(Ellipsis, slice(None, 32, None))]
        key_pass_7 = key_states_14[(Ellipsis, slice(32, None, None))]
        key_states_14 = None
        cos_10 = cos_2.unsqueeze(1)
        sin_10 = sin_2.unsqueeze(1)
        mul_59 = query_rot_7 * cos_10
        x1_14 = query_rot_7[(Ellipsis, slice(None, 16, None))]
        x2_14 = query_rot_7[(Ellipsis, slice(16, None, None))]
        query_rot_7 = None
        neg_14 = -x2_14
        x2_14 = None
        cat_29 = torch.cat((neg_14, x1_14), dim=-1)
        neg_14 = x1_14 = None
        mul_60 = cat_29 * sin_10
        cat_29 = None
        q_embed_7 = mul_59 + mul_60
        mul_59 = mul_60 = None
        mul_61 = key_rot_7 * cos_10
        cos_10 = None
        x1_15 = key_rot_7[(Ellipsis, slice(None, 16, None))]
        x2_15 = key_rot_7[(Ellipsis, slice(16, None, None))]
        key_rot_7 = None
        neg_15 = -x2_15
        x2_15 = None
        cat_30 = torch.cat((neg_15, x1_15), dim=-1)
        neg_15 = x1_15 = None
        mul_62 = cat_30 * sin_10
        cat_30 = sin_10 = None
        k_embed_7 = mul_61 + mul_62
        mul_61 = mul_62 = None
        query_states_15 = torch.cat((q_embed_7, query_pass_7), dim=-1)
        q_embed_7 = query_pass_7 = None
        key_states_15 = torch.cat((k_embed_7, key_pass_7), dim=-1)
        k_embed_7 = key_pass_7 = None
        attention_mask_8 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_7 = query_states_15.contiguous()
        query_states_15 = None
        key_7 = key_states_15.contiguous()
        value_7 = value_states_7.contiguous()
        attn_output_28 = torch._C._nn.scaled_dot_product_attention(
            query_7,
            key_7,
            value_7,
            attn_mask=attention_mask_8,
            dropout_p=0.0,
            scale=0.125,
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
            l_self_modules_layers_modules_7_modules_self_attn_modules_dense_parameters_weight_,
            l_self_modules_layers_modules_7_modules_self_attn_modules_dense_parameters_bias_,
        )
        attn_output_30 = l_self_modules_layers_modules_7_modules_self_attn_modules_dense_parameters_weight_ = l_self_modules_layers_modules_7_modules_self_attn_modules_dense_parameters_bias_ = (None)
        attn_outputs_7 = torch.nn.functional.dropout(attn_output_31, 0.0, False, False)
        attn_output_31 = None
        hidden_states_36 = torch._C._nn.linear(
            hidden_states_35,
            l_self_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_35 = (
            l_self_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_63 = 0.5 * hidden_states_36
        pow_8 = torch.pow(hidden_states_36, 3.0)
        mul_64 = 0.044715 * pow_8
        pow_8 = None
        add_44 = hidden_states_36 + mul_64
        hidden_states_36 = mul_64 = None
        mul_65 = 0.7978845608028654 * add_44
        add_44 = None
        tanh_7 = torch.tanh(mul_65)
        mul_65 = None
        add_45 = 1.0 + tanh_7
        tanh_7 = None
        hidden_states_37 = mul_63 * add_45
        mul_63 = add_45 = None
        hidden_states_38 = torch._C._nn.linear(
            hidden_states_37,
            l_self_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_37 = (
            l_self_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        feed_forward_hidden_states_7 = torch.nn.functional.dropout(
            hidden_states_38, 0.0, False, False
        )
        hidden_states_38 = None
        add_46 = attn_outputs_7 + feed_forward_hidden_states_7
        attn_outputs_7 = feed_forward_hidden_states_7 = None
        hidden_states_39 = add_46 + hidden_states_34
        add_46 = hidden_states_34 = None
        hidden_states_40 = torch.nn.functional.layer_norm(
            hidden_states_39,
            (2048,),
            l_self_modules_layers_modules_8_modules_input_layernorm_parameters_weight_,
            l_self_modules_layers_modules_8_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_8_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_layers_modules_8_modules_input_layernorm_parameters_bias_
        ) = None
        linear_48 = torch._C._nn.linear(
            hidden_states_40,
            l_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_25 = linear_48.view((1, 2, -1, 64))
        linear_48 = None
        query_states_16 = view_25.transpose(1, 2)
        view_25 = None
        linear_49 = torch._C._nn.linear(
            hidden_states_40,
            l_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_26 = linear_49.view((1, 2, -1, 64))
        linear_49 = None
        key_states_16 = view_26.transpose(1, 2)
        view_26 = None
        linear_50 = torch._C._nn.linear(
            hidden_states_40,
            l_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_27 = linear_50.view((1, 2, -1, 64))
        linear_50 = None
        value_states_8 = view_27.transpose(1, 2)
        view_27 = None
        query_rot_8 = query_states_16[(Ellipsis, slice(None, 32, None))]
        query_pass_8 = query_states_16[(Ellipsis, slice(32, None, None))]
        query_states_16 = None
        key_rot_8 = key_states_16[(Ellipsis, slice(None, 32, None))]
        key_pass_8 = key_states_16[(Ellipsis, slice(32, None, None))]
        key_states_16 = None
        cos_11 = cos_2.unsqueeze(1)
        sin_11 = sin_2.unsqueeze(1)
        mul_67 = query_rot_8 * cos_11
        x1_16 = query_rot_8[(Ellipsis, slice(None, 16, None))]
        x2_16 = query_rot_8[(Ellipsis, slice(16, None, None))]
        query_rot_8 = None
        neg_16 = -x2_16
        x2_16 = None
        cat_33 = torch.cat((neg_16, x1_16), dim=-1)
        neg_16 = x1_16 = None
        mul_68 = cat_33 * sin_11
        cat_33 = None
        q_embed_8 = mul_67 + mul_68
        mul_67 = mul_68 = None
        mul_69 = key_rot_8 * cos_11
        cos_11 = None
        x1_17 = key_rot_8[(Ellipsis, slice(None, 16, None))]
        x2_17 = key_rot_8[(Ellipsis, slice(16, None, None))]
        key_rot_8 = None
        neg_17 = -x2_17
        x2_17 = None
        cat_34 = torch.cat((neg_17, x1_17), dim=-1)
        neg_17 = x1_17 = None
        mul_70 = cat_34 * sin_11
        cat_34 = sin_11 = None
        k_embed_8 = mul_69 + mul_70
        mul_69 = mul_70 = None
        query_states_17 = torch.cat((q_embed_8, query_pass_8), dim=-1)
        q_embed_8 = query_pass_8 = None
        key_states_17 = torch.cat((k_embed_8, key_pass_8), dim=-1)
        k_embed_8 = key_pass_8 = None
        attention_mask_9 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_8 = query_states_17.contiguous()
        query_states_17 = None
        key_8 = key_states_17.contiguous()
        value_8 = value_states_8.contiguous()
        attn_output_32 = torch._C._nn.scaled_dot_product_attention(
            query_8,
            key_8,
            value_8,
            attn_mask=attention_mask_9,
            dropout_p=0.0,
            scale=0.125,
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
            l_self_modules_layers_modules_8_modules_self_attn_modules_dense_parameters_weight_,
            l_self_modules_layers_modules_8_modules_self_attn_modules_dense_parameters_bias_,
        )
        attn_output_34 = l_self_modules_layers_modules_8_modules_self_attn_modules_dense_parameters_weight_ = l_self_modules_layers_modules_8_modules_self_attn_modules_dense_parameters_bias_ = (None)
        attn_outputs_8 = torch.nn.functional.dropout(attn_output_35, 0.0, False, False)
        attn_output_35 = None
        hidden_states_41 = torch._C._nn.linear(
            hidden_states_40,
            l_self_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_40 = (
            l_self_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_71 = 0.5 * hidden_states_41
        pow_9 = torch.pow(hidden_states_41, 3.0)
        mul_72 = 0.044715 * pow_9
        pow_9 = None
        add_50 = hidden_states_41 + mul_72
        hidden_states_41 = mul_72 = None
        mul_73 = 0.7978845608028654 * add_50
        add_50 = None
        tanh_8 = torch.tanh(mul_73)
        mul_73 = None
        add_51 = 1.0 + tanh_8
        tanh_8 = None
        hidden_states_42 = mul_71 * add_51
        mul_71 = add_51 = None
        hidden_states_43 = torch._C._nn.linear(
            hidden_states_42,
            l_self_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_42 = (
            l_self_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        feed_forward_hidden_states_8 = torch.nn.functional.dropout(
            hidden_states_43, 0.0, False, False
        )
        hidden_states_43 = None
        add_52 = attn_outputs_8 + feed_forward_hidden_states_8
        attn_outputs_8 = feed_forward_hidden_states_8 = None
        hidden_states_44 = add_52 + hidden_states_39
        add_52 = hidden_states_39 = None
        hidden_states_45 = torch.nn.functional.layer_norm(
            hidden_states_44,
            (2048,),
            l_self_modules_layers_modules_9_modules_input_layernorm_parameters_weight_,
            l_self_modules_layers_modules_9_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_9_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_layers_modules_9_modules_input_layernorm_parameters_bias_
        ) = None
        linear_54 = torch._C._nn.linear(
            hidden_states_45,
            l_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_28 = linear_54.view((1, 2, -1, 64))
        linear_54 = None
        query_states_18 = view_28.transpose(1, 2)
        view_28 = None
        linear_55 = torch._C._nn.linear(
            hidden_states_45,
            l_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_29 = linear_55.view((1, 2, -1, 64))
        linear_55 = None
        key_states_18 = view_29.transpose(1, 2)
        view_29 = None
        linear_56 = torch._C._nn.linear(
            hidden_states_45,
            l_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_30 = linear_56.view((1, 2, -1, 64))
        linear_56 = None
        value_states_9 = view_30.transpose(1, 2)
        view_30 = None
        query_rot_9 = query_states_18[(Ellipsis, slice(None, 32, None))]
        query_pass_9 = query_states_18[(Ellipsis, slice(32, None, None))]
        query_states_18 = None
        key_rot_9 = key_states_18[(Ellipsis, slice(None, 32, None))]
        key_pass_9 = key_states_18[(Ellipsis, slice(32, None, None))]
        key_states_18 = None
        cos_12 = cos_2.unsqueeze(1)
        sin_12 = sin_2.unsqueeze(1)
        mul_75 = query_rot_9 * cos_12
        x1_18 = query_rot_9[(Ellipsis, slice(None, 16, None))]
        x2_18 = query_rot_9[(Ellipsis, slice(16, None, None))]
        query_rot_9 = None
        neg_18 = -x2_18
        x2_18 = None
        cat_37 = torch.cat((neg_18, x1_18), dim=-1)
        neg_18 = x1_18 = None
        mul_76 = cat_37 * sin_12
        cat_37 = None
        q_embed_9 = mul_75 + mul_76
        mul_75 = mul_76 = None
        mul_77 = key_rot_9 * cos_12
        cos_12 = None
        x1_19 = key_rot_9[(Ellipsis, slice(None, 16, None))]
        x2_19 = key_rot_9[(Ellipsis, slice(16, None, None))]
        key_rot_9 = None
        neg_19 = -x2_19
        x2_19 = None
        cat_38 = torch.cat((neg_19, x1_19), dim=-1)
        neg_19 = x1_19 = None
        mul_78 = cat_38 * sin_12
        cat_38 = sin_12 = None
        k_embed_9 = mul_77 + mul_78
        mul_77 = mul_78 = None
        query_states_19 = torch.cat((q_embed_9, query_pass_9), dim=-1)
        q_embed_9 = query_pass_9 = None
        key_states_19 = torch.cat((k_embed_9, key_pass_9), dim=-1)
        k_embed_9 = key_pass_9 = None
        attention_mask_10 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_9 = query_states_19.contiguous()
        query_states_19 = None
        key_9 = key_states_19.contiguous()
        value_9 = value_states_9.contiguous()
        attn_output_36 = torch._C._nn.scaled_dot_product_attention(
            query_9,
            key_9,
            value_9,
            attn_mask=attention_mask_10,
            dropout_p=0.0,
            scale=0.125,
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
            l_self_modules_layers_modules_9_modules_self_attn_modules_dense_parameters_weight_,
            l_self_modules_layers_modules_9_modules_self_attn_modules_dense_parameters_bias_,
        )
        attn_output_38 = l_self_modules_layers_modules_9_modules_self_attn_modules_dense_parameters_weight_ = l_self_modules_layers_modules_9_modules_self_attn_modules_dense_parameters_bias_ = (None)
        attn_outputs_9 = torch.nn.functional.dropout(attn_output_39, 0.0, False, False)
        attn_output_39 = None
        hidden_states_46 = torch._C._nn.linear(
            hidden_states_45,
            l_self_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_45 = (
            l_self_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_79 = 0.5 * hidden_states_46
        pow_10 = torch.pow(hidden_states_46, 3.0)
        mul_80 = 0.044715 * pow_10
        pow_10 = None
        add_56 = hidden_states_46 + mul_80
        hidden_states_46 = mul_80 = None
        mul_81 = 0.7978845608028654 * add_56
        add_56 = None
        tanh_9 = torch.tanh(mul_81)
        mul_81 = None
        add_57 = 1.0 + tanh_9
        tanh_9 = None
        hidden_states_47 = mul_79 * add_57
        mul_79 = add_57 = None
        hidden_states_48 = torch._C._nn.linear(
            hidden_states_47,
            l_self_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_47 = (
            l_self_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        feed_forward_hidden_states_9 = torch.nn.functional.dropout(
            hidden_states_48, 0.0, False, False
        )
        hidden_states_48 = None
        add_58 = attn_outputs_9 + feed_forward_hidden_states_9
        attn_outputs_9 = feed_forward_hidden_states_9 = None
        hidden_states_49 = add_58 + hidden_states_44
        add_58 = hidden_states_44 = None
        hidden_states_50 = torch.nn.functional.layer_norm(
            hidden_states_49,
            (2048,),
            l_self_modules_layers_modules_10_modules_input_layernorm_parameters_weight_,
            l_self_modules_layers_modules_10_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_10_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_layers_modules_10_modules_input_layernorm_parameters_bias_
        ) = None
        linear_60 = torch._C._nn.linear(
            hidden_states_50,
            l_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_31 = linear_60.view((1, 2, -1, 64))
        linear_60 = None
        query_states_20 = view_31.transpose(1, 2)
        view_31 = None
        linear_61 = torch._C._nn.linear(
            hidden_states_50,
            l_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_32 = linear_61.view((1, 2, -1, 64))
        linear_61 = None
        key_states_20 = view_32.transpose(1, 2)
        view_32 = None
        linear_62 = torch._C._nn.linear(
            hidden_states_50,
            l_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_33 = linear_62.view((1, 2, -1, 64))
        linear_62 = None
        value_states_10 = view_33.transpose(1, 2)
        view_33 = None
        query_rot_10 = query_states_20[(Ellipsis, slice(None, 32, None))]
        query_pass_10 = query_states_20[(Ellipsis, slice(32, None, None))]
        query_states_20 = None
        key_rot_10 = key_states_20[(Ellipsis, slice(None, 32, None))]
        key_pass_10 = key_states_20[(Ellipsis, slice(32, None, None))]
        key_states_20 = None
        cos_13 = cos_2.unsqueeze(1)
        sin_13 = sin_2.unsqueeze(1)
        mul_83 = query_rot_10 * cos_13
        x1_20 = query_rot_10[(Ellipsis, slice(None, 16, None))]
        x2_20 = query_rot_10[(Ellipsis, slice(16, None, None))]
        query_rot_10 = None
        neg_20 = -x2_20
        x2_20 = None
        cat_41 = torch.cat((neg_20, x1_20), dim=-1)
        neg_20 = x1_20 = None
        mul_84 = cat_41 * sin_13
        cat_41 = None
        q_embed_10 = mul_83 + mul_84
        mul_83 = mul_84 = None
        mul_85 = key_rot_10 * cos_13
        cos_13 = None
        x1_21 = key_rot_10[(Ellipsis, slice(None, 16, None))]
        x2_21 = key_rot_10[(Ellipsis, slice(16, None, None))]
        key_rot_10 = None
        neg_21 = -x2_21
        x2_21 = None
        cat_42 = torch.cat((neg_21, x1_21), dim=-1)
        neg_21 = x1_21 = None
        mul_86 = cat_42 * sin_13
        cat_42 = sin_13 = None
        k_embed_10 = mul_85 + mul_86
        mul_85 = mul_86 = None
        query_states_21 = torch.cat((q_embed_10, query_pass_10), dim=-1)
        q_embed_10 = query_pass_10 = None
        key_states_21 = torch.cat((k_embed_10, key_pass_10), dim=-1)
        k_embed_10 = key_pass_10 = None
        attention_mask_11 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_10 = query_states_21.contiguous()
        query_states_21 = None
        key_10 = key_states_21.contiguous()
        value_10 = value_states_10.contiguous()
        attn_output_40 = torch._C._nn.scaled_dot_product_attention(
            query_10,
            key_10,
            value_10,
            attn_mask=attention_mask_11,
            dropout_p=0.0,
            scale=0.125,
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
            l_self_modules_layers_modules_10_modules_self_attn_modules_dense_parameters_weight_,
            l_self_modules_layers_modules_10_modules_self_attn_modules_dense_parameters_bias_,
        )
        attn_output_42 = l_self_modules_layers_modules_10_modules_self_attn_modules_dense_parameters_weight_ = l_self_modules_layers_modules_10_modules_self_attn_modules_dense_parameters_bias_ = (None)
        attn_outputs_10 = torch.nn.functional.dropout(attn_output_43, 0.0, False, False)
        attn_output_43 = None
        hidden_states_51 = torch._C._nn.linear(
            hidden_states_50,
            l_self_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_50 = (
            l_self_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_87 = 0.5 * hidden_states_51
        pow_11 = torch.pow(hidden_states_51, 3.0)
        mul_88 = 0.044715 * pow_11
        pow_11 = None
        add_62 = hidden_states_51 + mul_88
        hidden_states_51 = mul_88 = None
        mul_89 = 0.7978845608028654 * add_62
        add_62 = None
        tanh_10 = torch.tanh(mul_89)
        mul_89 = None
        add_63 = 1.0 + tanh_10
        tanh_10 = None
        hidden_states_52 = mul_87 * add_63
        mul_87 = add_63 = None
        hidden_states_53 = torch._C._nn.linear(
            hidden_states_52,
            l_self_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_52 = (
            l_self_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        feed_forward_hidden_states_10 = torch.nn.functional.dropout(
            hidden_states_53, 0.0, False, False
        )
        hidden_states_53 = None
        add_64 = attn_outputs_10 + feed_forward_hidden_states_10
        attn_outputs_10 = feed_forward_hidden_states_10 = None
        hidden_states_54 = add_64 + hidden_states_49
        add_64 = hidden_states_49 = None
        hidden_states_55 = torch.nn.functional.layer_norm(
            hidden_states_54,
            (2048,),
            l_self_modules_layers_modules_11_modules_input_layernorm_parameters_weight_,
            l_self_modules_layers_modules_11_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_11_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_layers_modules_11_modules_input_layernorm_parameters_bias_
        ) = None
        linear_66 = torch._C._nn.linear(
            hidden_states_55,
            l_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_34 = linear_66.view((1, 2, -1, 64))
        linear_66 = None
        query_states_22 = view_34.transpose(1, 2)
        view_34 = None
        linear_67 = torch._C._nn.linear(
            hidden_states_55,
            l_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_35 = linear_67.view((1, 2, -1, 64))
        linear_67 = None
        key_states_22 = view_35.transpose(1, 2)
        view_35 = None
        linear_68 = torch._C._nn.linear(
            hidden_states_55,
            l_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_36 = linear_68.view((1, 2, -1, 64))
        linear_68 = None
        value_states_11 = view_36.transpose(1, 2)
        view_36 = None
        query_rot_11 = query_states_22[(Ellipsis, slice(None, 32, None))]
        query_pass_11 = query_states_22[(Ellipsis, slice(32, None, None))]
        query_states_22 = None
        key_rot_11 = key_states_22[(Ellipsis, slice(None, 32, None))]
        key_pass_11 = key_states_22[(Ellipsis, slice(32, None, None))]
        key_states_22 = None
        cos_14 = cos_2.unsqueeze(1)
        sin_14 = sin_2.unsqueeze(1)
        mul_91 = query_rot_11 * cos_14
        x1_22 = query_rot_11[(Ellipsis, slice(None, 16, None))]
        x2_22 = query_rot_11[(Ellipsis, slice(16, None, None))]
        query_rot_11 = None
        neg_22 = -x2_22
        x2_22 = None
        cat_45 = torch.cat((neg_22, x1_22), dim=-1)
        neg_22 = x1_22 = None
        mul_92 = cat_45 * sin_14
        cat_45 = None
        q_embed_11 = mul_91 + mul_92
        mul_91 = mul_92 = None
        mul_93 = key_rot_11 * cos_14
        cos_14 = None
        x1_23 = key_rot_11[(Ellipsis, slice(None, 16, None))]
        x2_23 = key_rot_11[(Ellipsis, slice(16, None, None))]
        key_rot_11 = None
        neg_23 = -x2_23
        x2_23 = None
        cat_46 = torch.cat((neg_23, x1_23), dim=-1)
        neg_23 = x1_23 = None
        mul_94 = cat_46 * sin_14
        cat_46 = sin_14 = None
        k_embed_11 = mul_93 + mul_94
        mul_93 = mul_94 = None
        query_states_23 = torch.cat((q_embed_11, query_pass_11), dim=-1)
        q_embed_11 = query_pass_11 = None
        key_states_23 = torch.cat((k_embed_11, key_pass_11), dim=-1)
        k_embed_11 = key_pass_11 = None
        attention_mask_12 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_11 = query_states_23.contiguous()
        query_states_23 = None
        key_11 = key_states_23.contiguous()
        value_11 = value_states_11.contiguous()
        attn_output_44 = torch._C._nn.scaled_dot_product_attention(
            query_11,
            key_11,
            value_11,
            attn_mask=attention_mask_12,
            dropout_p=0.0,
            scale=0.125,
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
            l_self_modules_layers_modules_11_modules_self_attn_modules_dense_parameters_weight_,
            l_self_modules_layers_modules_11_modules_self_attn_modules_dense_parameters_bias_,
        )
        attn_output_46 = l_self_modules_layers_modules_11_modules_self_attn_modules_dense_parameters_weight_ = l_self_modules_layers_modules_11_modules_self_attn_modules_dense_parameters_bias_ = (None)
        attn_outputs_11 = torch.nn.functional.dropout(attn_output_47, 0.0, False, False)
        attn_output_47 = None
        hidden_states_56 = torch._C._nn.linear(
            hidden_states_55,
            l_self_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_55 = (
            l_self_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_95 = 0.5 * hidden_states_56
        pow_12 = torch.pow(hidden_states_56, 3.0)
        mul_96 = 0.044715 * pow_12
        pow_12 = None
        add_68 = hidden_states_56 + mul_96
        hidden_states_56 = mul_96 = None
        mul_97 = 0.7978845608028654 * add_68
        add_68 = None
        tanh_11 = torch.tanh(mul_97)
        mul_97 = None
        add_69 = 1.0 + tanh_11
        tanh_11 = None
        hidden_states_57 = mul_95 * add_69
        mul_95 = add_69 = None
        hidden_states_58 = torch._C._nn.linear(
            hidden_states_57,
            l_self_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_57 = (
            l_self_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        feed_forward_hidden_states_11 = torch.nn.functional.dropout(
            hidden_states_58, 0.0, False, False
        )
        hidden_states_58 = None
        add_70 = attn_outputs_11 + feed_forward_hidden_states_11
        attn_outputs_11 = feed_forward_hidden_states_11 = None
        hidden_states_59 = add_70 + hidden_states_54
        add_70 = hidden_states_54 = None
        hidden_states_60 = torch.nn.functional.layer_norm(
            hidden_states_59,
            (2048,),
            l_self_modules_layers_modules_12_modules_input_layernorm_parameters_weight_,
            l_self_modules_layers_modules_12_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_12_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_layers_modules_12_modules_input_layernorm_parameters_bias_
        ) = None
        linear_72 = torch._C._nn.linear(
            hidden_states_60,
            l_self_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_37 = linear_72.view((1, 2, -1, 64))
        linear_72 = None
        query_states_24 = view_37.transpose(1, 2)
        view_37 = None
        linear_73 = torch._C._nn.linear(
            hidden_states_60,
            l_self_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_38 = linear_73.view((1, 2, -1, 64))
        linear_73 = None
        key_states_24 = view_38.transpose(1, 2)
        view_38 = None
        linear_74 = torch._C._nn.linear(
            hidden_states_60,
            l_self_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_39 = linear_74.view((1, 2, -1, 64))
        linear_74 = None
        value_states_12 = view_39.transpose(1, 2)
        view_39 = None
        query_rot_12 = query_states_24[(Ellipsis, slice(None, 32, None))]
        query_pass_12 = query_states_24[(Ellipsis, slice(32, None, None))]
        query_states_24 = None
        key_rot_12 = key_states_24[(Ellipsis, slice(None, 32, None))]
        key_pass_12 = key_states_24[(Ellipsis, slice(32, None, None))]
        key_states_24 = None
        cos_15 = cos_2.unsqueeze(1)
        sin_15 = sin_2.unsqueeze(1)
        mul_99 = query_rot_12 * cos_15
        x1_24 = query_rot_12[(Ellipsis, slice(None, 16, None))]
        x2_24 = query_rot_12[(Ellipsis, slice(16, None, None))]
        query_rot_12 = None
        neg_24 = -x2_24
        x2_24 = None
        cat_49 = torch.cat((neg_24, x1_24), dim=-1)
        neg_24 = x1_24 = None
        mul_100 = cat_49 * sin_15
        cat_49 = None
        q_embed_12 = mul_99 + mul_100
        mul_99 = mul_100 = None
        mul_101 = key_rot_12 * cos_15
        cos_15 = None
        x1_25 = key_rot_12[(Ellipsis, slice(None, 16, None))]
        x2_25 = key_rot_12[(Ellipsis, slice(16, None, None))]
        key_rot_12 = None
        neg_25 = -x2_25
        x2_25 = None
        cat_50 = torch.cat((neg_25, x1_25), dim=-1)
        neg_25 = x1_25 = None
        mul_102 = cat_50 * sin_15
        cat_50 = sin_15 = None
        k_embed_12 = mul_101 + mul_102
        mul_101 = mul_102 = None
        query_states_25 = torch.cat((q_embed_12, query_pass_12), dim=-1)
        q_embed_12 = query_pass_12 = None
        key_states_25 = torch.cat((k_embed_12, key_pass_12), dim=-1)
        k_embed_12 = key_pass_12 = None
        attention_mask_13 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_12 = query_states_25.contiguous()
        query_states_25 = None
        key_12 = key_states_25.contiguous()
        value_12 = value_states_12.contiguous()
        attn_output_48 = torch._C._nn.scaled_dot_product_attention(
            query_12,
            key_12,
            value_12,
            attn_mask=attention_mask_13,
            dropout_p=0.0,
            scale=0.125,
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
            l_self_modules_layers_modules_12_modules_self_attn_modules_dense_parameters_weight_,
            l_self_modules_layers_modules_12_modules_self_attn_modules_dense_parameters_bias_,
        )
        attn_output_50 = l_self_modules_layers_modules_12_modules_self_attn_modules_dense_parameters_weight_ = l_self_modules_layers_modules_12_modules_self_attn_modules_dense_parameters_bias_ = (None)
        attn_outputs_12 = torch.nn.functional.dropout(attn_output_51, 0.0, False, False)
        attn_output_51 = None
        hidden_states_61 = torch._C._nn.linear(
            hidden_states_60,
            l_self_modules_layers_modules_12_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_12_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_60 = (
            l_self_modules_layers_modules_12_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layers_modules_12_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_103 = 0.5 * hidden_states_61
        pow_13 = torch.pow(hidden_states_61, 3.0)
        mul_104 = 0.044715 * pow_13
        pow_13 = None
        add_74 = hidden_states_61 + mul_104
        hidden_states_61 = mul_104 = None
        mul_105 = 0.7978845608028654 * add_74
        add_74 = None
        tanh_12 = torch.tanh(mul_105)
        mul_105 = None
        add_75 = 1.0 + tanh_12
        tanh_12 = None
        hidden_states_62 = mul_103 * add_75
        mul_103 = add_75 = None
        hidden_states_63 = torch._C._nn.linear(
            hidden_states_62,
            l_self_modules_layers_modules_12_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_12_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_62 = (
            l_self_modules_layers_modules_12_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layers_modules_12_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        feed_forward_hidden_states_12 = torch.nn.functional.dropout(
            hidden_states_63, 0.0, False, False
        )
        hidden_states_63 = None
        add_76 = attn_outputs_12 + feed_forward_hidden_states_12
        attn_outputs_12 = feed_forward_hidden_states_12 = None
        hidden_states_64 = add_76 + hidden_states_59
        add_76 = hidden_states_59 = None
        hidden_states_65 = torch.nn.functional.layer_norm(
            hidden_states_64,
            (2048,),
            l_self_modules_layers_modules_13_modules_input_layernorm_parameters_weight_,
            l_self_modules_layers_modules_13_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_13_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_layers_modules_13_modules_input_layernorm_parameters_bias_
        ) = None
        linear_78 = torch._C._nn.linear(
            hidden_states_65,
            l_self_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_40 = linear_78.view((1, 2, -1, 64))
        linear_78 = None
        query_states_26 = view_40.transpose(1, 2)
        view_40 = None
        linear_79 = torch._C._nn.linear(
            hidden_states_65,
            l_self_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_41 = linear_79.view((1, 2, -1, 64))
        linear_79 = None
        key_states_26 = view_41.transpose(1, 2)
        view_41 = None
        linear_80 = torch._C._nn.linear(
            hidden_states_65,
            l_self_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_42 = linear_80.view((1, 2, -1, 64))
        linear_80 = None
        value_states_13 = view_42.transpose(1, 2)
        view_42 = None
        query_rot_13 = query_states_26[(Ellipsis, slice(None, 32, None))]
        query_pass_13 = query_states_26[(Ellipsis, slice(32, None, None))]
        query_states_26 = None
        key_rot_13 = key_states_26[(Ellipsis, slice(None, 32, None))]
        key_pass_13 = key_states_26[(Ellipsis, slice(32, None, None))]
        key_states_26 = None
        cos_16 = cos_2.unsqueeze(1)
        sin_16 = sin_2.unsqueeze(1)
        mul_107 = query_rot_13 * cos_16
        x1_26 = query_rot_13[(Ellipsis, slice(None, 16, None))]
        x2_26 = query_rot_13[(Ellipsis, slice(16, None, None))]
        query_rot_13 = None
        neg_26 = -x2_26
        x2_26 = None
        cat_53 = torch.cat((neg_26, x1_26), dim=-1)
        neg_26 = x1_26 = None
        mul_108 = cat_53 * sin_16
        cat_53 = None
        q_embed_13 = mul_107 + mul_108
        mul_107 = mul_108 = None
        mul_109 = key_rot_13 * cos_16
        cos_16 = None
        x1_27 = key_rot_13[(Ellipsis, slice(None, 16, None))]
        x2_27 = key_rot_13[(Ellipsis, slice(16, None, None))]
        key_rot_13 = None
        neg_27 = -x2_27
        x2_27 = None
        cat_54 = torch.cat((neg_27, x1_27), dim=-1)
        neg_27 = x1_27 = None
        mul_110 = cat_54 * sin_16
        cat_54 = sin_16 = None
        k_embed_13 = mul_109 + mul_110
        mul_109 = mul_110 = None
        query_states_27 = torch.cat((q_embed_13, query_pass_13), dim=-1)
        q_embed_13 = query_pass_13 = None
        key_states_27 = torch.cat((k_embed_13, key_pass_13), dim=-1)
        k_embed_13 = key_pass_13 = None
        attention_mask_14 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_13 = query_states_27.contiguous()
        query_states_27 = None
        key_13 = key_states_27.contiguous()
        value_13 = value_states_13.contiguous()
        attn_output_52 = torch._C._nn.scaled_dot_product_attention(
            query_13,
            key_13,
            value_13,
            attn_mask=attention_mask_14,
            dropout_p=0.0,
            scale=0.125,
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
            l_self_modules_layers_modules_13_modules_self_attn_modules_dense_parameters_weight_,
            l_self_modules_layers_modules_13_modules_self_attn_modules_dense_parameters_bias_,
        )
        attn_output_54 = l_self_modules_layers_modules_13_modules_self_attn_modules_dense_parameters_weight_ = l_self_modules_layers_modules_13_modules_self_attn_modules_dense_parameters_bias_ = (None)
        attn_outputs_13 = torch.nn.functional.dropout(attn_output_55, 0.0, False, False)
        attn_output_55 = None
        hidden_states_66 = torch._C._nn.linear(
            hidden_states_65,
            l_self_modules_layers_modules_13_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_13_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_65 = (
            l_self_modules_layers_modules_13_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layers_modules_13_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_111 = 0.5 * hidden_states_66
        pow_14 = torch.pow(hidden_states_66, 3.0)
        mul_112 = 0.044715 * pow_14
        pow_14 = None
        add_80 = hidden_states_66 + mul_112
        hidden_states_66 = mul_112 = None
        mul_113 = 0.7978845608028654 * add_80
        add_80 = None
        tanh_13 = torch.tanh(mul_113)
        mul_113 = None
        add_81 = 1.0 + tanh_13
        tanh_13 = None
        hidden_states_67 = mul_111 * add_81
        mul_111 = add_81 = None
        hidden_states_68 = torch._C._nn.linear(
            hidden_states_67,
            l_self_modules_layers_modules_13_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_13_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_67 = (
            l_self_modules_layers_modules_13_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layers_modules_13_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        feed_forward_hidden_states_13 = torch.nn.functional.dropout(
            hidden_states_68, 0.0, False, False
        )
        hidden_states_68 = None
        add_82 = attn_outputs_13 + feed_forward_hidden_states_13
        attn_outputs_13 = feed_forward_hidden_states_13 = None
        hidden_states_69 = add_82 + hidden_states_64
        add_82 = hidden_states_64 = None
        hidden_states_70 = torch.nn.functional.layer_norm(
            hidden_states_69,
            (2048,),
            l_self_modules_layers_modules_14_modules_input_layernorm_parameters_weight_,
            l_self_modules_layers_modules_14_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_14_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_layers_modules_14_modules_input_layernorm_parameters_bias_
        ) = None
        linear_84 = torch._C._nn.linear(
            hidden_states_70,
            l_self_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_43 = linear_84.view((1, 2, -1, 64))
        linear_84 = None
        query_states_28 = view_43.transpose(1, 2)
        view_43 = None
        linear_85 = torch._C._nn.linear(
            hidden_states_70,
            l_self_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_44 = linear_85.view((1, 2, -1, 64))
        linear_85 = None
        key_states_28 = view_44.transpose(1, 2)
        view_44 = None
        linear_86 = torch._C._nn.linear(
            hidden_states_70,
            l_self_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_45 = linear_86.view((1, 2, -1, 64))
        linear_86 = None
        value_states_14 = view_45.transpose(1, 2)
        view_45 = None
        query_rot_14 = query_states_28[(Ellipsis, slice(None, 32, None))]
        query_pass_14 = query_states_28[(Ellipsis, slice(32, None, None))]
        query_states_28 = None
        key_rot_14 = key_states_28[(Ellipsis, slice(None, 32, None))]
        key_pass_14 = key_states_28[(Ellipsis, slice(32, None, None))]
        key_states_28 = None
        cos_17 = cos_2.unsqueeze(1)
        sin_17 = sin_2.unsqueeze(1)
        mul_115 = query_rot_14 * cos_17
        x1_28 = query_rot_14[(Ellipsis, slice(None, 16, None))]
        x2_28 = query_rot_14[(Ellipsis, slice(16, None, None))]
        query_rot_14 = None
        neg_28 = -x2_28
        x2_28 = None
        cat_57 = torch.cat((neg_28, x1_28), dim=-1)
        neg_28 = x1_28 = None
        mul_116 = cat_57 * sin_17
        cat_57 = None
        q_embed_14 = mul_115 + mul_116
        mul_115 = mul_116 = None
        mul_117 = key_rot_14 * cos_17
        cos_17 = None
        x1_29 = key_rot_14[(Ellipsis, slice(None, 16, None))]
        x2_29 = key_rot_14[(Ellipsis, slice(16, None, None))]
        key_rot_14 = None
        neg_29 = -x2_29
        x2_29 = None
        cat_58 = torch.cat((neg_29, x1_29), dim=-1)
        neg_29 = x1_29 = None
        mul_118 = cat_58 * sin_17
        cat_58 = sin_17 = None
        k_embed_14 = mul_117 + mul_118
        mul_117 = mul_118 = None
        query_states_29 = torch.cat((q_embed_14, query_pass_14), dim=-1)
        q_embed_14 = query_pass_14 = None
        key_states_29 = torch.cat((k_embed_14, key_pass_14), dim=-1)
        k_embed_14 = key_pass_14 = None
        attention_mask_15 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_14 = query_states_29.contiguous()
        query_states_29 = None
        key_14 = key_states_29.contiguous()
        value_14 = value_states_14.contiguous()
        attn_output_56 = torch._C._nn.scaled_dot_product_attention(
            query_14,
            key_14,
            value_14,
            attn_mask=attention_mask_15,
            dropout_p=0.0,
            scale=0.125,
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
            l_self_modules_layers_modules_14_modules_self_attn_modules_dense_parameters_weight_,
            l_self_modules_layers_modules_14_modules_self_attn_modules_dense_parameters_bias_,
        )
        attn_output_58 = l_self_modules_layers_modules_14_modules_self_attn_modules_dense_parameters_weight_ = l_self_modules_layers_modules_14_modules_self_attn_modules_dense_parameters_bias_ = (None)
        attn_outputs_14 = torch.nn.functional.dropout(attn_output_59, 0.0, False, False)
        attn_output_59 = None
        hidden_states_71 = torch._C._nn.linear(
            hidden_states_70,
            l_self_modules_layers_modules_14_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_14_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_70 = (
            l_self_modules_layers_modules_14_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layers_modules_14_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_119 = 0.5 * hidden_states_71
        pow_15 = torch.pow(hidden_states_71, 3.0)
        mul_120 = 0.044715 * pow_15
        pow_15 = None
        add_86 = hidden_states_71 + mul_120
        hidden_states_71 = mul_120 = None
        mul_121 = 0.7978845608028654 * add_86
        add_86 = None
        tanh_14 = torch.tanh(mul_121)
        mul_121 = None
        add_87 = 1.0 + tanh_14
        tanh_14 = None
        hidden_states_72 = mul_119 * add_87
        mul_119 = add_87 = None
        hidden_states_73 = torch._C._nn.linear(
            hidden_states_72,
            l_self_modules_layers_modules_14_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_14_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_72 = (
            l_self_modules_layers_modules_14_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layers_modules_14_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        feed_forward_hidden_states_14 = torch.nn.functional.dropout(
            hidden_states_73, 0.0, False, False
        )
        hidden_states_73 = None
        add_88 = attn_outputs_14 + feed_forward_hidden_states_14
        attn_outputs_14 = feed_forward_hidden_states_14 = None
        hidden_states_74 = add_88 + hidden_states_69
        add_88 = hidden_states_69 = None
        hidden_states_75 = torch.nn.functional.layer_norm(
            hidden_states_74,
            (2048,),
            l_self_modules_layers_modules_15_modules_input_layernorm_parameters_weight_,
            l_self_modules_layers_modules_15_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_15_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_layers_modules_15_modules_input_layernorm_parameters_bias_
        ) = None
        linear_90 = torch._C._nn.linear(
            hidden_states_75,
            l_self_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_46 = linear_90.view((1, 2, -1, 64))
        linear_90 = None
        query_states_30 = view_46.transpose(1, 2)
        view_46 = None
        linear_91 = torch._C._nn.linear(
            hidden_states_75,
            l_self_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_47 = linear_91.view((1, 2, -1, 64))
        linear_91 = None
        key_states_30 = view_47.transpose(1, 2)
        view_47 = None
        linear_92 = torch._C._nn.linear(
            hidden_states_75,
            l_self_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_48 = linear_92.view((1, 2, -1, 64))
        linear_92 = None
        value_states_15 = view_48.transpose(1, 2)
        view_48 = None
        query_rot_15 = query_states_30[(Ellipsis, slice(None, 32, None))]
        query_pass_15 = query_states_30[(Ellipsis, slice(32, None, None))]
        query_states_30 = None
        key_rot_15 = key_states_30[(Ellipsis, slice(None, 32, None))]
        key_pass_15 = key_states_30[(Ellipsis, slice(32, None, None))]
        key_states_30 = None
        cos_18 = cos_2.unsqueeze(1)
        sin_18 = sin_2.unsqueeze(1)
        mul_123 = query_rot_15 * cos_18
        x1_30 = query_rot_15[(Ellipsis, slice(None, 16, None))]
        x2_30 = query_rot_15[(Ellipsis, slice(16, None, None))]
        query_rot_15 = None
        neg_30 = -x2_30
        x2_30 = None
        cat_61 = torch.cat((neg_30, x1_30), dim=-1)
        neg_30 = x1_30 = None
        mul_124 = cat_61 * sin_18
        cat_61 = None
        q_embed_15 = mul_123 + mul_124
        mul_123 = mul_124 = None
        mul_125 = key_rot_15 * cos_18
        cos_18 = None
        x1_31 = key_rot_15[(Ellipsis, slice(None, 16, None))]
        x2_31 = key_rot_15[(Ellipsis, slice(16, None, None))]
        key_rot_15 = None
        neg_31 = -x2_31
        x2_31 = None
        cat_62 = torch.cat((neg_31, x1_31), dim=-1)
        neg_31 = x1_31 = None
        mul_126 = cat_62 * sin_18
        cat_62 = sin_18 = None
        k_embed_15 = mul_125 + mul_126
        mul_125 = mul_126 = None
        query_states_31 = torch.cat((q_embed_15, query_pass_15), dim=-1)
        q_embed_15 = query_pass_15 = None
        key_states_31 = torch.cat((k_embed_15, key_pass_15), dim=-1)
        k_embed_15 = key_pass_15 = None
        attention_mask_16 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_15 = query_states_31.contiguous()
        query_states_31 = None
        key_15 = key_states_31.contiguous()
        value_15 = value_states_15.contiguous()
        attn_output_60 = torch._C._nn.scaled_dot_product_attention(
            query_15,
            key_15,
            value_15,
            attn_mask=attention_mask_16,
            dropout_p=0.0,
            scale=0.125,
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
            l_self_modules_layers_modules_15_modules_self_attn_modules_dense_parameters_weight_,
            l_self_modules_layers_modules_15_modules_self_attn_modules_dense_parameters_bias_,
        )
        attn_output_62 = l_self_modules_layers_modules_15_modules_self_attn_modules_dense_parameters_weight_ = l_self_modules_layers_modules_15_modules_self_attn_modules_dense_parameters_bias_ = (None)
        attn_outputs_15 = torch.nn.functional.dropout(attn_output_63, 0.0, False, False)
        attn_output_63 = None
        hidden_states_76 = torch._C._nn.linear(
            hidden_states_75,
            l_self_modules_layers_modules_15_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_15_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_75 = (
            l_self_modules_layers_modules_15_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layers_modules_15_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_127 = 0.5 * hidden_states_76
        pow_16 = torch.pow(hidden_states_76, 3.0)
        mul_128 = 0.044715 * pow_16
        pow_16 = None
        add_92 = hidden_states_76 + mul_128
        hidden_states_76 = mul_128 = None
        mul_129 = 0.7978845608028654 * add_92
        add_92 = None
        tanh_15 = torch.tanh(mul_129)
        mul_129 = None
        add_93 = 1.0 + tanh_15
        tanh_15 = None
        hidden_states_77 = mul_127 * add_93
        mul_127 = add_93 = None
        hidden_states_78 = torch._C._nn.linear(
            hidden_states_77,
            l_self_modules_layers_modules_15_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_15_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_77 = (
            l_self_modules_layers_modules_15_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layers_modules_15_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        feed_forward_hidden_states_15 = torch.nn.functional.dropout(
            hidden_states_78, 0.0, False, False
        )
        hidden_states_78 = None
        add_94 = attn_outputs_15 + feed_forward_hidden_states_15
        attn_outputs_15 = feed_forward_hidden_states_15 = None
        hidden_states_79 = add_94 + hidden_states_74
        add_94 = hidden_states_74 = None
        hidden_states_80 = torch.nn.functional.layer_norm(
            hidden_states_79,
            (2048,),
            l_self_modules_layers_modules_16_modules_input_layernorm_parameters_weight_,
            l_self_modules_layers_modules_16_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_16_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_layers_modules_16_modules_input_layernorm_parameters_bias_
        ) = None
        linear_96 = torch._C._nn.linear(
            hidden_states_80,
            l_self_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_49 = linear_96.view((1, 2, -1, 64))
        linear_96 = None
        query_states_32 = view_49.transpose(1, 2)
        view_49 = None
        linear_97 = torch._C._nn.linear(
            hidden_states_80,
            l_self_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_50 = linear_97.view((1, 2, -1, 64))
        linear_97 = None
        key_states_32 = view_50.transpose(1, 2)
        view_50 = None
        linear_98 = torch._C._nn.linear(
            hidden_states_80,
            l_self_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_51 = linear_98.view((1, 2, -1, 64))
        linear_98 = None
        value_states_16 = view_51.transpose(1, 2)
        view_51 = None
        query_rot_16 = query_states_32[(Ellipsis, slice(None, 32, None))]
        query_pass_16 = query_states_32[(Ellipsis, slice(32, None, None))]
        query_states_32 = None
        key_rot_16 = key_states_32[(Ellipsis, slice(None, 32, None))]
        key_pass_16 = key_states_32[(Ellipsis, slice(32, None, None))]
        key_states_32 = None
        cos_19 = cos_2.unsqueeze(1)
        sin_19 = sin_2.unsqueeze(1)
        mul_131 = query_rot_16 * cos_19
        x1_32 = query_rot_16[(Ellipsis, slice(None, 16, None))]
        x2_32 = query_rot_16[(Ellipsis, slice(16, None, None))]
        query_rot_16 = None
        neg_32 = -x2_32
        x2_32 = None
        cat_65 = torch.cat((neg_32, x1_32), dim=-1)
        neg_32 = x1_32 = None
        mul_132 = cat_65 * sin_19
        cat_65 = None
        q_embed_16 = mul_131 + mul_132
        mul_131 = mul_132 = None
        mul_133 = key_rot_16 * cos_19
        cos_19 = None
        x1_33 = key_rot_16[(Ellipsis, slice(None, 16, None))]
        x2_33 = key_rot_16[(Ellipsis, slice(16, None, None))]
        key_rot_16 = None
        neg_33 = -x2_33
        x2_33 = None
        cat_66 = torch.cat((neg_33, x1_33), dim=-1)
        neg_33 = x1_33 = None
        mul_134 = cat_66 * sin_19
        cat_66 = sin_19 = None
        k_embed_16 = mul_133 + mul_134
        mul_133 = mul_134 = None
        query_states_33 = torch.cat((q_embed_16, query_pass_16), dim=-1)
        q_embed_16 = query_pass_16 = None
        key_states_33 = torch.cat((k_embed_16, key_pass_16), dim=-1)
        k_embed_16 = key_pass_16 = None
        attention_mask_17 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_16 = query_states_33.contiguous()
        query_states_33 = None
        key_16 = key_states_33.contiguous()
        value_16 = value_states_16.contiguous()
        attn_output_64 = torch._C._nn.scaled_dot_product_attention(
            query_16,
            key_16,
            value_16,
            attn_mask=attention_mask_17,
            dropout_p=0.0,
            scale=0.125,
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
            l_self_modules_layers_modules_16_modules_self_attn_modules_dense_parameters_weight_,
            l_self_modules_layers_modules_16_modules_self_attn_modules_dense_parameters_bias_,
        )
        attn_output_66 = l_self_modules_layers_modules_16_modules_self_attn_modules_dense_parameters_weight_ = l_self_modules_layers_modules_16_modules_self_attn_modules_dense_parameters_bias_ = (None)
        attn_outputs_16 = torch.nn.functional.dropout(attn_output_67, 0.0, False, False)
        attn_output_67 = None
        hidden_states_81 = torch._C._nn.linear(
            hidden_states_80,
            l_self_modules_layers_modules_16_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_16_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_80 = (
            l_self_modules_layers_modules_16_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layers_modules_16_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_135 = 0.5 * hidden_states_81
        pow_17 = torch.pow(hidden_states_81, 3.0)
        mul_136 = 0.044715 * pow_17
        pow_17 = None
        add_98 = hidden_states_81 + mul_136
        hidden_states_81 = mul_136 = None
        mul_137 = 0.7978845608028654 * add_98
        add_98 = None
        tanh_16 = torch.tanh(mul_137)
        mul_137 = None
        add_99 = 1.0 + tanh_16
        tanh_16 = None
        hidden_states_82 = mul_135 * add_99
        mul_135 = add_99 = None
        hidden_states_83 = torch._C._nn.linear(
            hidden_states_82,
            l_self_modules_layers_modules_16_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_16_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_82 = (
            l_self_modules_layers_modules_16_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layers_modules_16_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        feed_forward_hidden_states_16 = torch.nn.functional.dropout(
            hidden_states_83, 0.0, False, False
        )
        hidden_states_83 = None
        add_100 = attn_outputs_16 + feed_forward_hidden_states_16
        attn_outputs_16 = feed_forward_hidden_states_16 = None
        hidden_states_84 = add_100 + hidden_states_79
        add_100 = hidden_states_79 = None
        hidden_states_85 = torch.nn.functional.layer_norm(
            hidden_states_84,
            (2048,),
            l_self_modules_layers_modules_17_modules_input_layernorm_parameters_weight_,
            l_self_modules_layers_modules_17_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_17_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_layers_modules_17_modules_input_layernorm_parameters_bias_
        ) = None
        linear_102 = torch._C._nn.linear(
            hidden_states_85,
            l_self_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_52 = linear_102.view((1, 2, -1, 64))
        linear_102 = None
        query_states_34 = view_52.transpose(1, 2)
        view_52 = None
        linear_103 = torch._C._nn.linear(
            hidden_states_85,
            l_self_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_53 = linear_103.view((1, 2, -1, 64))
        linear_103 = None
        key_states_34 = view_53.transpose(1, 2)
        view_53 = None
        linear_104 = torch._C._nn.linear(
            hidden_states_85,
            l_self_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_54 = linear_104.view((1, 2, -1, 64))
        linear_104 = None
        value_states_17 = view_54.transpose(1, 2)
        view_54 = None
        query_rot_17 = query_states_34[(Ellipsis, slice(None, 32, None))]
        query_pass_17 = query_states_34[(Ellipsis, slice(32, None, None))]
        query_states_34 = None
        key_rot_17 = key_states_34[(Ellipsis, slice(None, 32, None))]
        key_pass_17 = key_states_34[(Ellipsis, slice(32, None, None))]
        key_states_34 = None
        cos_20 = cos_2.unsqueeze(1)
        sin_20 = sin_2.unsqueeze(1)
        mul_139 = query_rot_17 * cos_20
        x1_34 = query_rot_17[(Ellipsis, slice(None, 16, None))]
        x2_34 = query_rot_17[(Ellipsis, slice(16, None, None))]
        query_rot_17 = None
        neg_34 = -x2_34
        x2_34 = None
        cat_69 = torch.cat((neg_34, x1_34), dim=-1)
        neg_34 = x1_34 = None
        mul_140 = cat_69 * sin_20
        cat_69 = None
        q_embed_17 = mul_139 + mul_140
        mul_139 = mul_140 = None
        mul_141 = key_rot_17 * cos_20
        cos_20 = None
        x1_35 = key_rot_17[(Ellipsis, slice(None, 16, None))]
        x2_35 = key_rot_17[(Ellipsis, slice(16, None, None))]
        key_rot_17 = None
        neg_35 = -x2_35
        x2_35 = None
        cat_70 = torch.cat((neg_35, x1_35), dim=-1)
        neg_35 = x1_35 = None
        mul_142 = cat_70 * sin_20
        cat_70 = sin_20 = None
        k_embed_17 = mul_141 + mul_142
        mul_141 = mul_142 = None
        query_states_35 = torch.cat((q_embed_17, query_pass_17), dim=-1)
        q_embed_17 = query_pass_17 = None
        key_states_35 = torch.cat((k_embed_17, key_pass_17), dim=-1)
        k_embed_17 = key_pass_17 = None
        attention_mask_18 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_17 = query_states_35.contiguous()
        query_states_35 = None
        key_17 = key_states_35.contiguous()
        value_17 = value_states_17.contiguous()
        attn_output_68 = torch._C._nn.scaled_dot_product_attention(
            query_17,
            key_17,
            value_17,
            attn_mask=attention_mask_18,
            dropout_p=0.0,
            scale=0.125,
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
            l_self_modules_layers_modules_17_modules_self_attn_modules_dense_parameters_weight_,
            l_self_modules_layers_modules_17_modules_self_attn_modules_dense_parameters_bias_,
        )
        attn_output_70 = l_self_modules_layers_modules_17_modules_self_attn_modules_dense_parameters_weight_ = l_self_modules_layers_modules_17_modules_self_attn_modules_dense_parameters_bias_ = (None)
        attn_outputs_17 = torch.nn.functional.dropout(attn_output_71, 0.0, False, False)
        attn_output_71 = None
        hidden_states_86 = torch._C._nn.linear(
            hidden_states_85,
            l_self_modules_layers_modules_17_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_17_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_85 = (
            l_self_modules_layers_modules_17_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layers_modules_17_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_143 = 0.5 * hidden_states_86
        pow_18 = torch.pow(hidden_states_86, 3.0)
        mul_144 = 0.044715 * pow_18
        pow_18 = None
        add_104 = hidden_states_86 + mul_144
        hidden_states_86 = mul_144 = None
        mul_145 = 0.7978845608028654 * add_104
        add_104 = None
        tanh_17 = torch.tanh(mul_145)
        mul_145 = None
        add_105 = 1.0 + tanh_17
        tanh_17 = None
        hidden_states_87 = mul_143 * add_105
        mul_143 = add_105 = None
        hidden_states_88 = torch._C._nn.linear(
            hidden_states_87,
            l_self_modules_layers_modules_17_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_17_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_87 = (
            l_self_modules_layers_modules_17_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layers_modules_17_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        feed_forward_hidden_states_17 = torch.nn.functional.dropout(
            hidden_states_88, 0.0, False, False
        )
        hidden_states_88 = None
        add_106 = attn_outputs_17 + feed_forward_hidden_states_17
        attn_outputs_17 = feed_forward_hidden_states_17 = None
        hidden_states_89 = add_106 + hidden_states_84
        add_106 = hidden_states_84 = None
        hidden_states_90 = torch.nn.functional.layer_norm(
            hidden_states_89,
            (2048,),
            l_self_modules_layers_modules_18_modules_input_layernorm_parameters_weight_,
            l_self_modules_layers_modules_18_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_18_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_layers_modules_18_modules_input_layernorm_parameters_bias_
        ) = None
        linear_108 = torch._C._nn.linear(
            hidden_states_90,
            l_self_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_55 = linear_108.view((1, 2, -1, 64))
        linear_108 = None
        query_states_36 = view_55.transpose(1, 2)
        view_55 = None
        linear_109 = torch._C._nn.linear(
            hidden_states_90,
            l_self_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_56 = linear_109.view((1, 2, -1, 64))
        linear_109 = None
        key_states_36 = view_56.transpose(1, 2)
        view_56 = None
        linear_110 = torch._C._nn.linear(
            hidden_states_90,
            l_self_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_57 = linear_110.view((1, 2, -1, 64))
        linear_110 = None
        value_states_18 = view_57.transpose(1, 2)
        view_57 = None
        query_rot_18 = query_states_36[(Ellipsis, slice(None, 32, None))]
        query_pass_18 = query_states_36[(Ellipsis, slice(32, None, None))]
        query_states_36 = None
        key_rot_18 = key_states_36[(Ellipsis, slice(None, 32, None))]
        key_pass_18 = key_states_36[(Ellipsis, slice(32, None, None))]
        key_states_36 = None
        cos_21 = cos_2.unsqueeze(1)
        sin_21 = sin_2.unsqueeze(1)
        mul_147 = query_rot_18 * cos_21
        x1_36 = query_rot_18[(Ellipsis, slice(None, 16, None))]
        x2_36 = query_rot_18[(Ellipsis, slice(16, None, None))]
        query_rot_18 = None
        neg_36 = -x2_36
        x2_36 = None
        cat_73 = torch.cat((neg_36, x1_36), dim=-1)
        neg_36 = x1_36 = None
        mul_148 = cat_73 * sin_21
        cat_73 = None
        q_embed_18 = mul_147 + mul_148
        mul_147 = mul_148 = None
        mul_149 = key_rot_18 * cos_21
        cos_21 = None
        x1_37 = key_rot_18[(Ellipsis, slice(None, 16, None))]
        x2_37 = key_rot_18[(Ellipsis, slice(16, None, None))]
        key_rot_18 = None
        neg_37 = -x2_37
        x2_37 = None
        cat_74 = torch.cat((neg_37, x1_37), dim=-1)
        neg_37 = x1_37 = None
        mul_150 = cat_74 * sin_21
        cat_74 = sin_21 = None
        k_embed_18 = mul_149 + mul_150
        mul_149 = mul_150 = None
        query_states_37 = torch.cat((q_embed_18, query_pass_18), dim=-1)
        q_embed_18 = query_pass_18 = None
        key_states_37 = torch.cat((k_embed_18, key_pass_18), dim=-1)
        k_embed_18 = key_pass_18 = None
        attention_mask_19 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_18 = query_states_37.contiguous()
        query_states_37 = None
        key_18 = key_states_37.contiguous()
        value_18 = value_states_18.contiguous()
        attn_output_72 = torch._C._nn.scaled_dot_product_attention(
            query_18,
            key_18,
            value_18,
            attn_mask=attention_mask_19,
            dropout_p=0.0,
            scale=0.125,
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
            l_self_modules_layers_modules_18_modules_self_attn_modules_dense_parameters_weight_,
            l_self_modules_layers_modules_18_modules_self_attn_modules_dense_parameters_bias_,
        )
        attn_output_74 = l_self_modules_layers_modules_18_modules_self_attn_modules_dense_parameters_weight_ = l_self_modules_layers_modules_18_modules_self_attn_modules_dense_parameters_bias_ = (None)
        attn_outputs_18 = torch.nn.functional.dropout(attn_output_75, 0.0, False, False)
        attn_output_75 = None
        hidden_states_91 = torch._C._nn.linear(
            hidden_states_90,
            l_self_modules_layers_modules_18_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_18_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_90 = (
            l_self_modules_layers_modules_18_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layers_modules_18_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_151 = 0.5 * hidden_states_91
        pow_19 = torch.pow(hidden_states_91, 3.0)
        mul_152 = 0.044715 * pow_19
        pow_19 = None
        add_110 = hidden_states_91 + mul_152
        hidden_states_91 = mul_152 = None
        mul_153 = 0.7978845608028654 * add_110
        add_110 = None
        tanh_18 = torch.tanh(mul_153)
        mul_153 = None
        add_111 = 1.0 + tanh_18
        tanh_18 = None
        hidden_states_92 = mul_151 * add_111
        mul_151 = add_111 = None
        hidden_states_93 = torch._C._nn.linear(
            hidden_states_92,
            l_self_modules_layers_modules_18_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_18_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_92 = (
            l_self_modules_layers_modules_18_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layers_modules_18_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        feed_forward_hidden_states_18 = torch.nn.functional.dropout(
            hidden_states_93, 0.0, False, False
        )
        hidden_states_93 = None
        add_112 = attn_outputs_18 + feed_forward_hidden_states_18
        attn_outputs_18 = feed_forward_hidden_states_18 = None
        hidden_states_94 = add_112 + hidden_states_89
        add_112 = hidden_states_89 = None
        hidden_states_95 = torch.nn.functional.layer_norm(
            hidden_states_94,
            (2048,),
            l_self_modules_layers_modules_19_modules_input_layernorm_parameters_weight_,
            l_self_modules_layers_modules_19_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_19_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_layers_modules_19_modules_input_layernorm_parameters_bias_
        ) = None
        linear_114 = torch._C._nn.linear(
            hidden_states_95,
            l_self_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_58 = linear_114.view((1, 2, -1, 64))
        linear_114 = None
        query_states_38 = view_58.transpose(1, 2)
        view_58 = None
        linear_115 = torch._C._nn.linear(
            hidden_states_95,
            l_self_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_59 = linear_115.view((1, 2, -1, 64))
        linear_115 = None
        key_states_38 = view_59.transpose(1, 2)
        view_59 = None
        linear_116 = torch._C._nn.linear(
            hidden_states_95,
            l_self_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_60 = linear_116.view((1, 2, -1, 64))
        linear_116 = None
        value_states_19 = view_60.transpose(1, 2)
        view_60 = None
        query_rot_19 = query_states_38[(Ellipsis, slice(None, 32, None))]
        query_pass_19 = query_states_38[(Ellipsis, slice(32, None, None))]
        query_states_38 = None
        key_rot_19 = key_states_38[(Ellipsis, slice(None, 32, None))]
        key_pass_19 = key_states_38[(Ellipsis, slice(32, None, None))]
        key_states_38 = None
        cos_22 = cos_2.unsqueeze(1)
        sin_22 = sin_2.unsqueeze(1)
        mul_155 = query_rot_19 * cos_22
        x1_38 = query_rot_19[(Ellipsis, slice(None, 16, None))]
        x2_38 = query_rot_19[(Ellipsis, slice(16, None, None))]
        query_rot_19 = None
        neg_38 = -x2_38
        x2_38 = None
        cat_77 = torch.cat((neg_38, x1_38), dim=-1)
        neg_38 = x1_38 = None
        mul_156 = cat_77 * sin_22
        cat_77 = None
        q_embed_19 = mul_155 + mul_156
        mul_155 = mul_156 = None
        mul_157 = key_rot_19 * cos_22
        cos_22 = None
        x1_39 = key_rot_19[(Ellipsis, slice(None, 16, None))]
        x2_39 = key_rot_19[(Ellipsis, slice(16, None, None))]
        key_rot_19 = None
        neg_39 = -x2_39
        x2_39 = None
        cat_78 = torch.cat((neg_39, x1_39), dim=-1)
        neg_39 = x1_39 = None
        mul_158 = cat_78 * sin_22
        cat_78 = sin_22 = None
        k_embed_19 = mul_157 + mul_158
        mul_157 = mul_158 = None
        query_states_39 = torch.cat((q_embed_19, query_pass_19), dim=-1)
        q_embed_19 = query_pass_19 = None
        key_states_39 = torch.cat((k_embed_19, key_pass_19), dim=-1)
        k_embed_19 = key_pass_19 = None
        attention_mask_20 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_19 = query_states_39.contiguous()
        query_states_39 = None
        key_19 = key_states_39.contiguous()
        value_19 = value_states_19.contiguous()
        attn_output_76 = torch._C._nn.scaled_dot_product_attention(
            query_19,
            key_19,
            value_19,
            attn_mask=attention_mask_20,
            dropout_p=0.0,
            scale=0.125,
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
            l_self_modules_layers_modules_19_modules_self_attn_modules_dense_parameters_weight_,
            l_self_modules_layers_modules_19_modules_self_attn_modules_dense_parameters_bias_,
        )
        attn_output_78 = l_self_modules_layers_modules_19_modules_self_attn_modules_dense_parameters_weight_ = l_self_modules_layers_modules_19_modules_self_attn_modules_dense_parameters_bias_ = (None)
        attn_outputs_19 = torch.nn.functional.dropout(attn_output_79, 0.0, False, False)
        attn_output_79 = None
        hidden_states_96 = torch._C._nn.linear(
            hidden_states_95,
            l_self_modules_layers_modules_19_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_19_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_95 = (
            l_self_modules_layers_modules_19_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layers_modules_19_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_159 = 0.5 * hidden_states_96
        pow_20 = torch.pow(hidden_states_96, 3.0)
        mul_160 = 0.044715 * pow_20
        pow_20 = None
        add_116 = hidden_states_96 + mul_160
        hidden_states_96 = mul_160 = None
        mul_161 = 0.7978845608028654 * add_116
        add_116 = None
        tanh_19 = torch.tanh(mul_161)
        mul_161 = None
        add_117 = 1.0 + tanh_19
        tanh_19 = None
        hidden_states_97 = mul_159 * add_117
        mul_159 = add_117 = None
        hidden_states_98 = torch._C._nn.linear(
            hidden_states_97,
            l_self_modules_layers_modules_19_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_19_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_97 = (
            l_self_modules_layers_modules_19_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layers_modules_19_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        feed_forward_hidden_states_19 = torch.nn.functional.dropout(
            hidden_states_98, 0.0, False, False
        )
        hidden_states_98 = None
        add_118 = attn_outputs_19 + feed_forward_hidden_states_19
        attn_outputs_19 = feed_forward_hidden_states_19 = None
        hidden_states_99 = add_118 + hidden_states_94
        add_118 = hidden_states_94 = None
        hidden_states_100 = torch.nn.functional.layer_norm(
            hidden_states_99,
            (2048,),
            l_self_modules_layers_modules_20_modules_input_layernorm_parameters_weight_,
            l_self_modules_layers_modules_20_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_20_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_layers_modules_20_modules_input_layernorm_parameters_bias_
        ) = None
        linear_120 = torch._C._nn.linear(
            hidden_states_100,
            l_self_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_61 = linear_120.view((1, 2, -1, 64))
        linear_120 = None
        query_states_40 = view_61.transpose(1, 2)
        view_61 = None
        linear_121 = torch._C._nn.linear(
            hidden_states_100,
            l_self_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_62 = linear_121.view((1, 2, -1, 64))
        linear_121 = None
        key_states_40 = view_62.transpose(1, 2)
        view_62 = None
        linear_122 = torch._C._nn.linear(
            hidden_states_100,
            l_self_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_63 = linear_122.view((1, 2, -1, 64))
        linear_122 = None
        value_states_20 = view_63.transpose(1, 2)
        view_63 = None
        query_rot_20 = query_states_40[(Ellipsis, slice(None, 32, None))]
        query_pass_20 = query_states_40[(Ellipsis, slice(32, None, None))]
        query_states_40 = None
        key_rot_20 = key_states_40[(Ellipsis, slice(None, 32, None))]
        key_pass_20 = key_states_40[(Ellipsis, slice(32, None, None))]
        key_states_40 = None
        cos_23 = cos_2.unsqueeze(1)
        sin_23 = sin_2.unsqueeze(1)
        mul_163 = query_rot_20 * cos_23
        x1_40 = query_rot_20[(Ellipsis, slice(None, 16, None))]
        x2_40 = query_rot_20[(Ellipsis, slice(16, None, None))]
        query_rot_20 = None
        neg_40 = -x2_40
        x2_40 = None
        cat_81 = torch.cat((neg_40, x1_40), dim=-1)
        neg_40 = x1_40 = None
        mul_164 = cat_81 * sin_23
        cat_81 = None
        q_embed_20 = mul_163 + mul_164
        mul_163 = mul_164 = None
        mul_165 = key_rot_20 * cos_23
        cos_23 = None
        x1_41 = key_rot_20[(Ellipsis, slice(None, 16, None))]
        x2_41 = key_rot_20[(Ellipsis, slice(16, None, None))]
        key_rot_20 = None
        neg_41 = -x2_41
        x2_41 = None
        cat_82 = torch.cat((neg_41, x1_41), dim=-1)
        neg_41 = x1_41 = None
        mul_166 = cat_82 * sin_23
        cat_82 = sin_23 = None
        k_embed_20 = mul_165 + mul_166
        mul_165 = mul_166 = None
        query_states_41 = torch.cat((q_embed_20, query_pass_20), dim=-1)
        q_embed_20 = query_pass_20 = None
        key_states_41 = torch.cat((k_embed_20, key_pass_20), dim=-1)
        k_embed_20 = key_pass_20 = None
        attention_mask_21 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_20 = query_states_41.contiguous()
        query_states_41 = None
        key_20 = key_states_41.contiguous()
        value_20 = value_states_20.contiguous()
        attn_output_80 = torch._C._nn.scaled_dot_product_attention(
            query_20,
            key_20,
            value_20,
            attn_mask=attention_mask_21,
            dropout_p=0.0,
            scale=0.125,
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
            l_self_modules_layers_modules_20_modules_self_attn_modules_dense_parameters_weight_,
            l_self_modules_layers_modules_20_modules_self_attn_modules_dense_parameters_bias_,
        )
        attn_output_82 = l_self_modules_layers_modules_20_modules_self_attn_modules_dense_parameters_weight_ = l_self_modules_layers_modules_20_modules_self_attn_modules_dense_parameters_bias_ = (None)
        attn_outputs_20 = torch.nn.functional.dropout(attn_output_83, 0.0, False, False)
        attn_output_83 = None
        hidden_states_101 = torch._C._nn.linear(
            hidden_states_100,
            l_self_modules_layers_modules_20_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_20_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_100 = (
            l_self_modules_layers_modules_20_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layers_modules_20_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_167 = 0.5 * hidden_states_101
        pow_21 = torch.pow(hidden_states_101, 3.0)
        mul_168 = 0.044715 * pow_21
        pow_21 = None
        add_122 = hidden_states_101 + mul_168
        hidden_states_101 = mul_168 = None
        mul_169 = 0.7978845608028654 * add_122
        add_122 = None
        tanh_20 = torch.tanh(mul_169)
        mul_169 = None
        add_123 = 1.0 + tanh_20
        tanh_20 = None
        hidden_states_102 = mul_167 * add_123
        mul_167 = add_123 = None
        hidden_states_103 = torch._C._nn.linear(
            hidden_states_102,
            l_self_modules_layers_modules_20_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_20_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_102 = (
            l_self_modules_layers_modules_20_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layers_modules_20_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        feed_forward_hidden_states_20 = torch.nn.functional.dropout(
            hidden_states_103, 0.0, False, False
        )
        hidden_states_103 = None
        add_124 = attn_outputs_20 + feed_forward_hidden_states_20
        attn_outputs_20 = feed_forward_hidden_states_20 = None
        hidden_states_104 = add_124 + hidden_states_99
        add_124 = hidden_states_99 = None
        hidden_states_105 = torch.nn.functional.layer_norm(
            hidden_states_104,
            (2048,),
            l_self_modules_layers_modules_21_modules_input_layernorm_parameters_weight_,
            l_self_modules_layers_modules_21_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_21_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_layers_modules_21_modules_input_layernorm_parameters_bias_
        ) = None
        linear_126 = torch._C._nn.linear(
            hidden_states_105,
            l_self_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_64 = linear_126.view((1, 2, -1, 64))
        linear_126 = None
        query_states_42 = view_64.transpose(1, 2)
        view_64 = None
        linear_127 = torch._C._nn.linear(
            hidden_states_105,
            l_self_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_65 = linear_127.view((1, 2, -1, 64))
        linear_127 = None
        key_states_42 = view_65.transpose(1, 2)
        view_65 = None
        linear_128 = torch._C._nn.linear(
            hidden_states_105,
            l_self_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_66 = linear_128.view((1, 2, -1, 64))
        linear_128 = None
        value_states_21 = view_66.transpose(1, 2)
        view_66 = None
        query_rot_21 = query_states_42[(Ellipsis, slice(None, 32, None))]
        query_pass_21 = query_states_42[(Ellipsis, slice(32, None, None))]
        query_states_42 = None
        key_rot_21 = key_states_42[(Ellipsis, slice(None, 32, None))]
        key_pass_21 = key_states_42[(Ellipsis, slice(32, None, None))]
        key_states_42 = None
        cos_24 = cos_2.unsqueeze(1)
        sin_24 = sin_2.unsqueeze(1)
        mul_171 = query_rot_21 * cos_24
        x1_42 = query_rot_21[(Ellipsis, slice(None, 16, None))]
        x2_42 = query_rot_21[(Ellipsis, slice(16, None, None))]
        query_rot_21 = None
        neg_42 = -x2_42
        x2_42 = None
        cat_85 = torch.cat((neg_42, x1_42), dim=-1)
        neg_42 = x1_42 = None
        mul_172 = cat_85 * sin_24
        cat_85 = None
        q_embed_21 = mul_171 + mul_172
        mul_171 = mul_172 = None
        mul_173 = key_rot_21 * cos_24
        cos_24 = None
        x1_43 = key_rot_21[(Ellipsis, slice(None, 16, None))]
        x2_43 = key_rot_21[(Ellipsis, slice(16, None, None))]
        key_rot_21 = None
        neg_43 = -x2_43
        x2_43 = None
        cat_86 = torch.cat((neg_43, x1_43), dim=-1)
        neg_43 = x1_43 = None
        mul_174 = cat_86 * sin_24
        cat_86 = sin_24 = None
        k_embed_21 = mul_173 + mul_174
        mul_173 = mul_174 = None
        query_states_43 = torch.cat((q_embed_21, query_pass_21), dim=-1)
        q_embed_21 = query_pass_21 = None
        key_states_43 = torch.cat((k_embed_21, key_pass_21), dim=-1)
        k_embed_21 = key_pass_21 = None
        attention_mask_22 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_21 = query_states_43.contiguous()
        query_states_43 = None
        key_21 = key_states_43.contiguous()
        value_21 = value_states_21.contiguous()
        attn_output_84 = torch._C._nn.scaled_dot_product_attention(
            query_21,
            key_21,
            value_21,
            attn_mask=attention_mask_22,
            dropout_p=0.0,
            scale=0.125,
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
            l_self_modules_layers_modules_21_modules_self_attn_modules_dense_parameters_weight_,
            l_self_modules_layers_modules_21_modules_self_attn_modules_dense_parameters_bias_,
        )
        attn_output_86 = l_self_modules_layers_modules_21_modules_self_attn_modules_dense_parameters_weight_ = l_self_modules_layers_modules_21_modules_self_attn_modules_dense_parameters_bias_ = (None)
        attn_outputs_21 = torch.nn.functional.dropout(attn_output_87, 0.0, False, False)
        attn_output_87 = None
        hidden_states_106 = torch._C._nn.linear(
            hidden_states_105,
            l_self_modules_layers_modules_21_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_21_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_105 = (
            l_self_modules_layers_modules_21_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layers_modules_21_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_175 = 0.5 * hidden_states_106
        pow_22 = torch.pow(hidden_states_106, 3.0)
        mul_176 = 0.044715 * pow_22
        pow_22 = None
        add_128 = hidden_states_106 + mul_176
        hidden_states_106 = mul_176 = None
        mul_177 = 0.7978845608028654 * add_128
        add_128 = None
        tanh_21 = torch.tanh(mul_177)
        mul_177 = None
        add_129 = 1.0 + tanh_21
        tanh_21 = None
        hidden_states_107 = mul_175 * add_129
        mul_175 = add_129 = None
        hidden_states_108 = torch._C._nn.linear(
            hidden_states_107,
            l_self_modules_layers_modules_21_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_21_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_107 = (
            l_self_modules_layers_modules_21_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layers_modules_21_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        feed_forward_hidden_states_21 = torch.nn.functional.dropout(
            hidden_states_108, 0.0, False, False
        )
        hidden_states_108 = None
        add_130 = attn_outputs_21 + feed_forward_hidden_states_21
        attn_outputs_21 = feed_forward_hidden_states_21 = None
        hidden_states_109 = add_130 + hidden_states_104
        add_130 = hidden_states_104 = None
        hidden_states_110 = torch.nn.functional.layer_norm(
            hidden_states_109,
            (2048,),
            l_self_modules_layers_modules_22_modules_input_layernorm_parameters_weight_,
            l_self_modules_layers_modules_22_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_22_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_layers_modules_22_modules_input_layernorm_parameters_bias_
        ) = None
        linear_132 = torch._C._nn.linear(
            hidden_states_110,
            l_self_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_67 = linear_132.view((1, 2, -1, 64))
        linear_132 = None
        query_states_44 = view_67.transpose(1, 2)
        view_67 = None
        linear_133 = torch._C._nn.linear(
            hidden_states_110,
            l_self_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_68 = linear_133.view((1, 2, -1, 64))
        linear_133 = None
        key_states_44 = view_68.transpose(1, 2)
        view_68 = None
        linear_134 = torch._C._nn.linear(
            hidden_states_110,
            l_self_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_69 = linear_134.view((1, 2, -1, 64))
        linear_134 = None
        value_states_22 = view_69.transpose(1, 2)
        view_69 = None
        query_rot_22 = query_states_44[(Ellipsis, slice(None, 32, None))]
        query_pass_22 = query_states_44[(Ellipsis, slice(32, None, None))]
        query_states_44 = None
        key_rot_22 = key_states_44[(Ellipsis, slice(None, 32, None))]
        key_pass_22 = key_states_44[(Ellipsis, slice(32, None, None))]
        key_states_44 = None
        cos_25 = cos_2.unsqueeze(1)
        sin_25 = sin_2.unsqueeze(1)
        mul_179 = query_rot_22 * cos_25
        x1_44 = query_rot_22[(Ellipsis, slice(None, 16, None))]
        x2_44 = query_rot_22[(Ellipsis, slice(16, None, None))]
        query_rot_22 = None
        neg_44 = -x2_44
        x2_44 = None
        cat_89 = torch.cat((neg_44, x1_44), dim=-1)
        neg_44 = x1_44 = None
        mul_180 = cat_89 * sin_25
        cat_89 = None
        q_embed_22 = mul_179 + mul_180
        mul_179 = mul_180 = None
        mul_181 = key_rot_22 * cos_25
        cos_25 = None
        x1_45 = key_rot_22[(Ellipsis, slice(None, 16, None))]
        x2_45 = key_rot_22[(Ellipsis, slice(16, None, None))]
        key_rot_22 = None
        neg_45 = -x2_45
        x2_45 = None
        cat_90 = torch.cat((neg_45, x1_45), dim=-1)
        neg_45 = x1_45 = None
        mul_182 = cat_90 * sin_25
        cat_90 = sin_25 = None
        k_embed_22 = mul_181 + mul_182
        mul_181 = mul_182 = None
        query_states_45 = torch.cat((q_embed_22, query_pass_22), dim=-1)
        q_embed_22 = query_pass_22 = None
        key_states_45 = torch.cat((k_embed_22, key_pass_22), dim=-1)
        k_embed_22 = key_pass_22 = None
        attention_mask_23 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_22 = query_states_45.contiguous()
        query_states_45 = None
        key_22 = key_states_45.contiguous()
        value_22 = value_states_22.contiguous()
        attn_output_88 = torch._C._nn.scaled_dot_product_attention(
            query_22,
            key_22,
            value_22,
            attn_mask=attention_mask_23,
            dropout_p=0.0,
            scale=0.125,
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
            l_self_modules_layers_modules_22_modules_self_attn_modules_dense_parameters_weight_,
            l_self_modules_layers_modules_22_modules_self_attn_modules_dense_parameters_bias_,
        )
        attn_output_90 = l_self_modules_layers_modules_22_modules_self_attn_modules_dense_parameters_weight_ = l_self_modules_layers_modules_22_modules_self_attn_modules_dense_parameters_bias_ = (None)
        attn_outputs_22 = torch.nn.functional.dropout(attn_output_91, 0.0, False, False)
        attn_output_91 = None
        hidden_states_111 = torch._C._nn.linear(
            hidden_states_110,
            l_self_modules_layers_modules_22_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_22_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_110 = (
            l_self_modules_layers_modules_22_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layers_modules_22_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_183 = 0.5 * hidden_states_111
        pow_23 = torch.pow(hidden_states_111, 3.0)
        mul_184 = 0.044715 * pow_23
        pow_23 = None
        add_134 = hidden_states_111 + mul_184
        hidden_states_111 = mul_184 = None
        mul_185 = 0.7978845608028654 * add_134
        add_134 = None
        tanh_22 = torch.tanh(mul_185)
        mul_185 = None
        add_135 = 1.0 + tanh_22
        tanh_22 = None
        hidden_states_112 = mul_183 * add_135
        mul_183 = add_135 = None
        hidden_states_113 = torch._C._nn.linear(
            hidden_states_112,
            l_self_modules_layers_modules_22_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_22_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_112 = (
            l_self_modules_layers_modules_22_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layers_modules_22_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        feed_forward_hidden_states_22 = torch.nn.functional.dropout(
            hidden_states_113, 0.0, False, False
        )
        hidden_states_113 = None
        add_136 = attn_outputs_22 + feed_forward_hidden_states_22
        attn_outputs_22 = feed_forward_hidden_states_22 = None
        hidden_states_114 = add_136 + hidden_states_109
        add_136 = hidden_states_109 = None
        hidden_states_115 = torch.nn.functional.layer_norm(
            hidden_states_114,
            (2048,),
            l_self_modules_layers_modules_23_modules_input_layernorm_parameters_weight_,
            l_self_modules_layers_modules_23_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_23_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_layers_modules_23_modules_input_layernorm_parameters_bias_
        ) = None
        linear_138 = torch._C._nn.linear(
            hidden_states_115,
            l_self_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_70 = linear_138.view((1, 2, -1, 64))
        linear_138 = None
        query_states_46 = view_70.transpose(1, 2)
        view_70 = None
        linear_139 = torch._C._nn.linear(
            hidden_states_115,
            l_self_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_71 = linear_139.view((1, 2, -1, 64))
        linear_139 = None
        key_states_46 = view_71.transpose(1, 2)
        view_71 = None
        linear_140 = torch._C._nn.linear(
            hidden_states_115,
            l_self_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_72 = linear_140.view((1, 2, -1, 64))
        linear_140 = None
        value_states_23 = view_72.transpose(1, 2)
        view_72 = None
        query_rot_23 = query_states_46[(Ellipsis, slice(None, 32, None))]
        query_pass_23 = query_states_46[(Ellipsis, slice(32, None, None))]
        query_states_46 = None
        key_rot_23 = key_states_46[(Ellipsis, slice(None, 32, None))]
        key_pass_23 = key_states_46[(Ellipsis, slice(32, None, None))]
        key_states_46 = None
        cos_26 = cos_2.unsqueeze(1)
        cos_2 = None
        sin_26 = sin_2.unsqueeze(1)
        sin_2 = None
        mul_187 = query_rot_23 * cos_26
        x1_46 = query_rot_23[(Ellipsis, slice(None, 16, None))]
        x2_46 = query_rot_23[(Ellipsis, slice(16, None, None))]
        query_rot_23 = None
        neg_46 = -x2_46
        x2_46 = None
        cat_93 = torch.cat((neg_46, x1_46), dim=-1)
        neg_46 = x1_46 = None
        mul_188 = cat_93 * sin_26
        cat_93 = None
        q_embed_23 = mul_187 + mul_188
        mul_187 = mul_188 = None
        mul_189 = key_rot_23 * cos_26
        cos_26 = None
        x1_47 = key_rot_23[(Ellipsis, slice(None, 16, None))]
        x2_47 = key_rot_23[(Ellipsis, slice(16, None, None))]
        key_rot_23 = None
        neg_47 = -x2_47
        x2_47 = None
        cat_94 = torch.cat((neg_47, x1_47), dim=-1)
        neg_47 = x1_47 = None
        mul_190 = cat_94 * sin_26
        cat_94 = sin_26 = None
        k_embed_23 = mul_189 + mul_190
        mul_189 = mul_190 = None
        query_states_47 = torch.cat((q_embed_23, query_pass_23), dim=-1)
        q_embed_23 = query_pass_23 = None
        key_states_47 = torch.cat((k_embed_23, key_pass_23), dim=-1)
        k_embed_23 = key_pass_23 = None
        attention_mask_24 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        causal_mask_2 = None
        query_23 = query_states_47.contiguous()
        query_states_47 = None
        key_23 = key_states_47.contiguous()
        value_23 = value_states_23.contiguous()
        attn_output_92 = torch._C._nn.scaled_dot_product_attention(
            query_23,
            key_23,
            value_23,
            attn_mask=attention_mask_24,
            dropout_p=0.0,
            scale=0.125,
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
            l_self_modules_layers_modules_23_modules_self_attn_modules_dense_parameters_weight_,
            l_self_modules_layers_modules_23_modules_self_attn_modules_dense_parameters_bias_,
        )
        attn_output_94 = l_self_modules_layers_modules_23_modules_self_attn_modules_dense_parameters_weight_ = l_self_modules_layers_modules_23_modules_self_attn_modules_dense_parameters_bias_ = (None)
        attn_outputs_23 = torch.nn.functional.dropout(attn_output_95, 0.0, False, False)
        attn_output_95 = None
        hidden_states_116 = torch._C._nn.linear(
            hidden_states_115,
            l_self_modules_layers_modules_23_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_23_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_115 = (
            l_self_modules_layers_modules_23_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layers_modules_23_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_191 = 0.5 * hidden_states_116
        pow_24 = torch.pow(hidden_states_116, 3.0)
        mul_192 = 0.044715 * pow_24
        pow_24 = None
        add_140 = hidden_states_116 + mul_192
        hidden_states_116 = mul_192 = None
        mul_193 = 0.7978845608028654 * add_140
        add_140 = None
        tanh_23 = torch.tanh(mul_193)
        mul_193 = None
        add_141 = 1.0 + tanh_23
        tanh_23 = None
        hidden_states_117 = mul_191 * add_141
        mul_191 = add_141 = None
        hidden_states_118 = torch._C._nn.linear(
            hidden_states_117,
            l_self_modules_layers_modules_23_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_23_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_117 = (
            l_self_modules_layers_modules_23_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layers_modules_23_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        feed_forward_hidden_states_23 = torch.nn.functional.dropout(
            hidden_states_118, 0.0, False, False
        )
        hidden_states_118 = None
        add_142 = attn_outputs_23 + feed_forward_hidden_states_23
        attn_outputs_23 = feed_forward_hidden_states_23 = None
        hidden_states_119 = add_142 + hidden_states_114
        add_142 = hidden_states_114 = None
        hidden_states_120 = torch.nn.functional.layer_norm(
            hidden_states_119,
            (2048,),
            l_self_modules_final_layernorm_parameters_weight_,
            l_self_modules_final_layernorm_parameters_bias_,
            1e-05,
        )
        hidden_states_119 = (
            l_self_modules_final_layernorm_parameters_weight_
        ) = l_self_modules_final_layernorm_parameters_bias_ = None
        return (
            value_states,
            key_states_1,
            value_states_1,
            key_states_3,
            value_states_2,
            key_states_5,
            value_states_3,
            key_states_7,
            value_states_4,
            key_states_9,
            value_states_5,
            key_states_11,
            value_states_6,
            key_states_13,
            value_states_7,
            key_states_15,
            value_states_8,
            key_states_17,
            value_states_9,
            key_states_19,
            value_states_10,
            key_states_21,
            value_states_11,
            key_states_23,
            value_states_12,
            key_states_25,
            value_states_13,
            key_states_27,
            value_states_14,
            key_states_29,
            value_states_15,
            key_states_31,
            value_states_16,
            key_states_33,
            value_states_17,
            key_states_35,
            value_states_18,
            key_states_37,
            value_states_19,
            key_states_39,
            value_states_20,
            key_states_41,
            value_states_21,
            key_states_43,
            value_states_22,
            key_states_45,
            value_states_23,
            key_states_47,
            hidden_states_120,
        )
