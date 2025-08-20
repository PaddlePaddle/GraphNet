import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_inputs_embeds_: torch.Tensor,
        L_encoder_hidden_states_: torch.Tensor,
        L_encoder_attention_mask_: torch.Tensor,
        L_self_modules_block_modules_0_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_relative_attention_bias_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_0_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_0_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_0_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_0_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_0_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_0_modules_layer_modules_2_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_0_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_0_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_2_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_2_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_2_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_4_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_4_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_4_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_4_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_4_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_4_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_4_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_4_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_4_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_4_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_4_modules_layer_modules_2_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_4_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_4_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_5_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_5_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_5_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_5_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_5_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_5_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_5_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_5_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_5_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_5_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_5_modules_layer_modules_2_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_5_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_5_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_6_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_6_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_6_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_6_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_6_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_6_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_6_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_6_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_6_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_6_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_6_modules_layer_modules_2_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_6_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_6_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_7_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_7_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_7_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_7_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_7_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_7_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_7_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_7_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_7_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_7_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_7_modules_layer_modules_2_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_7_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_7_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_8_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_8_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_8_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_8_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_8_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_8_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_8_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_8_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_8_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_8_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_8_modules_layer_modules_2_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_8_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_8_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_9_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_9_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_9_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_9_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_9_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_9_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_9_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_9_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_9_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_9_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_9_modules_layer_modules_2_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_9_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_9_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_10_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_10_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_10_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_10_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_10_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_10_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_10_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_10_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_10_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_10_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_10_modules_layer_modules_2_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_10_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_10_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_11_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_11_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_11_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_11_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_11_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_11_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_11_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_11_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_11_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_11_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_11_modules_layer_modules_2_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_11_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_11_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_12_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_12_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_12_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_12_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_12_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_12_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_12_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_12_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_12_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_12_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_12_modules_layer_modules_2_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_12_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_12_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_13_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_13_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_13_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_13_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_13_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_13_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_13_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_13_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_13_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_13_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_13_modules_layer_modules_2_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_13_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_13_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_14_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_14_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_14_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_14_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_14_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_14_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_14_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_14_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_14_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_14_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_14_modules_layer_modules_2_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_14_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_14_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_15_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_15_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_15_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_15_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_15_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_15_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_15_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_15_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_15_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_15_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_15_modules_layer_modules_2_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_15_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_15_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_16_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_16_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_16_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_16_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_16_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_16_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_16_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_16_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_16_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_16_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_16_modules_layer_modules_2_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_16_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_16_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_17_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_17_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_17_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_17_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_17_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_17_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_17_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_17_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_17_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_17_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_17_modules_layer_modules_2_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_17_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_17_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_18_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_18_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_18_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_18_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_18_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_18_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_18_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_18_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_18_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_18_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_18_modules_layer_modules_2_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_18_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_18_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_19_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_19_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_19_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_19_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_19_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_19_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_19_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_19_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_19_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_19_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_19_modules_layer_modules_2_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_19_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_19_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
    ):
        l_inputs_embeds_ = L_inputs_embeds_
        l_encoder_hidden_states_ = L_encoder_hidden_states_
        l_encoder_attention_mask_ = L_encoder_attention_mask_
        l_self_modules_block_modules_0_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_0_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_0_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_0_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_0_modules_layer_modules_0_modules_self_attention_modules_relative_attention_bias_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_relative_attention_bias_parameters_weight_
        l_self_modules_block_modules_0_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_0_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_0_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_0_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_0_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_0_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_0_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_2_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_0_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_0_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_2_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_2_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_2_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_4_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_4_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_4_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_4_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_4_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_4_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_4_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_4_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_4_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_4_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_4_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_4_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_4_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_4_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_4_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_4_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_4_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_4_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_4_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_4_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_4_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_4_modules_layer_modules_2_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_4_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_4_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_4_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_4_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_5_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_5_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_5_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_5_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_5_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_5_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_5_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_5_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_5_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_5_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_5_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_5_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_5_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_5_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_5_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_5_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_5_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_5_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_5_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_5_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_5_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_5_modules_layer_modules_2_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_5_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_5_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_5_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_5_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_6_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_6_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_6_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_6_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_6_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_6_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_6_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_6_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_6_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_6_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_6_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_6_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_6_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_6_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_6_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_6_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_6_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_6_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_6_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_6_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_6_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_6_modules_layer_modules_2_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_6_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_6_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_6_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_6_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_7_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_7_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_7_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_7_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_7_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_7_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_7_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_7_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_7_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_7_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_7_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_7_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_7_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_7_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_7_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_7_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_7_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_7_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_7_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_7_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_7_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_7_modules_layer_modules_2_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_7_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_7_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_7_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_7_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_8_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_8_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_8_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_8_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_8_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_8_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_8_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_8_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_8_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_8_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_8_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_8_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_8_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_8_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_8_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_8_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_8_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_8_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_8_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_8_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_8_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_8_modules_layer_modules_2_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_8_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_8_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_8_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_8_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_9_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_9_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_9_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_9_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_9_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_9_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_9_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_9_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_9_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_9_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_9_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_9_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_9_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_9_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_9_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_9_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_9_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_9_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_9_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_9_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_9_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_9_modules_layer_modules_2_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_9_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_9_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_9_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_9_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_10_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_10_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_10_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_10_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_10_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_10_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_10_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_10_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_10_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_10_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_10_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_10_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_10_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_10_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_10_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_10_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_10_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_10_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_10_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_10_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_10_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_10_modules_layer_modules_2_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_10_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_10_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_10_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_10_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_11_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_11_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_11_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_11_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_11_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_11_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_11_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_11_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_11_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_11_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_11_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_11_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_11_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_11_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_11_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_11_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_11_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_11_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_11_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_11_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_11_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_11_modules_layer_modules_2_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_11_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_11_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_11_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_11_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_12_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_12_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_12_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_12_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_12_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_12_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_12_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_12_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_12_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_12_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_12_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_12_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_12_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_12_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_12_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_12_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_12_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_12_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_12_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_12_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_12_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_12_modules_layer_modules_2_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_12_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_12_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_12_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_12_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_13_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_13_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_13_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_13_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_13_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_13_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_13_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_13_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_13_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_13_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_13_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_13_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_13_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_13_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_13_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_13_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_13_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_13_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_13_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_13_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_13_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_13_modules_layer_modules_2_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_13_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_13_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_13_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_13_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_14_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_14_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_14_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_14_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_14_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_14_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_14_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_14_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_14_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_14_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_14_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_14_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_14_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_14_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_14_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_14_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_14_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_14_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_14_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_14_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_14_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_14_modules_layer_modules_2_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_14_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_14_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_14_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_14_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_15_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_15_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_15_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_15_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_15_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_15_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_15_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_15_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_15_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_15_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_15_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_15_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_15_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_15_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_15_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_15_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_15_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_15_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_15_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_15_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_15_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_15_modules_layer_modules_2_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_15_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_15_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_15_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_15_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_16_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_16_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_16_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_16_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_16_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_16_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_16_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_16_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_16_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_16_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_16_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_16_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_16_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_16_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_16_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_16_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_16_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_16_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_16_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_16_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_16_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_16_modules_layer_modules_2_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_16_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_16_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_16_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_16_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_17_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_17_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_17_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_17_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_17_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_17_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_17_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_17_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_17_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_17_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_17_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_17_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_17_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_17_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_17_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_17_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_17_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_17_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_17_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_17_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_17_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_17_modules_layer_modules_2_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_17_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_17_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_17_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_17_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_18_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_18_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_18_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_18_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_18_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_18_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_18_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_18_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_18_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_18_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_18_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_18_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_18_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_18_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_18_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_18_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_18_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_18_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_18_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_18_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_18_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_18_modules_layer_modules_2_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_18_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_18_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_18_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_18_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_19_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_19_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_19_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_19_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_19_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_19_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_19_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_19_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_19_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_19_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_19_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_19_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_19_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_19_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_19_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_19_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_19_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_19_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_19_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_19_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_19_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_19_modules_layer_modules_2_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_19_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_19_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_19_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_19_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_final_layer_norm_parameters_weight_
        )
        cache_position = torch.arange(0, 23, device=device(type="cuda", index=0))
        causal_mask = torch.full(
            (23, 24),
            fill_value=-3.4028234663852886e38,
            dtype=torch.float32,
            device=device(type="cuda", index=0),
        )
        causal_mask_1 = torch.triu(causal_mask, diagonal=1)
        causal_mask = None
        arange_1 = torch.arange(24, device=device(type="cuda", index=0))
        reshape = cache_position.reshape(-1, 1)
        gt = arange_1 > reshape
        arange_1 = reshape = None
        causal_mask_1 *= gt
        causal_mask_2 = causal_mask_1
        causal_mask_1 = gt = None
        getitem = causal_mask_2[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        causal_mask_2 = None
        causal_mask_3 = getitem.expand(1, 1, -1, -1)
        getitem = None
        encoder_extended_attention_mask = l_encoder_attention_mask_[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        l_encoder_attention_mask_ = None
        encoder_extended_attention_mask_1 = encoder_extended_attention_mask.to(
            dtype=torch.float32
        )
        encoder_extended_attention_mask = None
        sub = 1.0 - encoder_extended_attention_mask_1
        encoder_extended_attention_mask_1 = None
        encoder_extended_attention_mask_2 = sub * -3.4028234663852886e38
        sub = None
        hidden_states = torch.nn.functional.dropout(l_inputs_embeds_, 0.1, False, False)
        l_inputs_embeds_ = None
        to_1 = hidden_states.to(torch.float32)
        pow_1 = to_1.pow(2)
        to_1 = None
        variance = pow_1.mean(-1, keepdim=True)
        pow_1 = None
        add = variance + 1e-06
        variance = None
        rsqrt = torch.rsqrt(add)
        add = None
        hidden_states_1 = hidden_states * rsqrt
        rsqrt = None
        normed_hidden_states = (
            l_self_modules_block_modules_0_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_1
        )
        l_self_modules_block_modules_0_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_1
        ) = None
        query_states = torch._C._nn.linear(
            normed_hidden_states,
            l_self_modules_block_modules_0_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_0_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view = query_states.view(1, -1, 8, 64)
        query_states = None
        query_states_1 = view.transpose(1, 2)
        view = None
        key_states = torch._C._nn.linear(
            normed_hidden_states,
            l_self_modules_block_modules_0_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_0_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states = torch._C._nn.linear(
            normed_hidden_states,
            l_self_modules_block_modules_0_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states = l_self_modules_block_modules_0_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_1 = key_states.view(1, -1, 8, 64)
        key_states = None
        key_states_1 = view_1.transpose(1, 2)
        view_1 = None
        view_2 = value_states.view(1, -1, 8, 64)
        value_states = None
        value_states_1 = view_2.transpose(1, 2)
        view_2 = None
        transpose_3 = key_states_1.transpose(3, 2)
        scores = torch.matmul(query_states_1, transpose_3)
        query_states_1 = transpose_3 = None
        getitem_2 = cache_position[-1]
        real_seq_length = getitem_2 + 1
        getitem_2 = real_seq_length = None
        getitem_3 = cache_position[(slice(None, None, None), None)]
        context_position = getitem_3.to(device(type="cuda", index=0))
        getitem_3 = None
        arange_2 = torch.arange(
            23, dtype=torch.int64, device=device(type="cuda", index=0)
        )
        memory_position = arange_2[(None, slice(None, None, None))]
        arange_2 = None
        relative_position = memory_position - context_position
        memory_position = context_position = None
        zeros_like = torch.zeros_like(relative_position)
        min_1 = torch.min(relative_position, zeros_like)
        relative_position = zeros_like = None
        relative_position_1 = -min_1
        min_1 = None
        is_small = relative_position_1 < 16
        float_1 = relative_position_1.float()
        truediv = float_1 / 16
        float_1 = None
        log = torch.log(truediv)
        truediv = None
        truediv_1 = log / 2.0794415416798357
        log = None
        mul_3 = truediv_1 * 16
        truediv_1 = None
        to_3 = mul_3.to(torch.int64)
        mul_3 = None
        relative_position_if_large = 16 + to_3
        to_3 = None
        full_like = torch.full_like(relative_position_if_large, 31)
        relative_position_if_large_1 = torch.min(relative_position_if_large, full_like)
        relative_position_if_large = full_like = None
        where = torch.where(is_small, relative_position_1, relative_position_if_large_1)
        is_small = relative_position_1 = relative_position_if_large_1 = None
        relative_buckets = 0 + where
        where = None
        values = torch.nn.functional.embedding(
            relative_buckets,
            l_self_modules_block_modules_0_modules_layer_modules_0_modules_self_attention_modules_relative_attention_bias_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        relative_buckets = l_self_modules_block_modules_0_modules_layer_modules_0_modules_self_attention_modules_relative_attention_bias_parameters_weight_ = (None)
        permute = values.permute([2, 0, 1])
        values = None
        values_1 = permute.unsqueeze(0)
        permute = None
        position_bias = values_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(-23, None, None),
                slice(None, None, None),
            )
        ]
        values_1 = None
        causal_mask_4 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 23, None),
            )
        ]
        causal_mask_3 = None
        position_bias_1 = position_bias + causal_mask_4
        position_bias = causal_mask_4 = None
        scores += position_bias_1
        scores_1 = scores
        scores = None
        float_2 = scores_1.float()
        softmax = torch.nn.functional.softmax(float_2, dim=-1)
        float_2 = None
        attn_weights = softmax.type_as(scores_1)
        softmax = scores_1 = None
        attn_weights_1 = torch.nn.functional.dropout(
            attn_weights, p=0.1, training=False
        )
        attn_weights = None
        attn_output = torch.matmul(attn_weights_1, value_states_1)
        attn_weights_1 = None
        transpose_4 = attn_output.transpose(1, 2)
        attn_output = None
        attn_output_1 = transpose_4.contiguous()
        transpose_4 = None
        attn_output_2 = attn_output_1.view(1, -1, 512)
        attn_output_1 = None
        attn_output_3 = torch._C._nn.linear(
            attn_output_2,
            l_self_modules_block_modules_0_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_2 = l_self_modules_block_modules_0_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_2 = torch.nn.functional.dropout(attn_output_3, 0.1, False, False)
        attn_output_3 = None
        hidden_states_2 = hidden_states + dropout_2
        hidden_states = dropout_2 = None
        getitem_7 = cache_position[-1]
        real_seq_length_1 = getitem_7 + 1
        getitem_7 = real_seq_length_1 = None
        to_4 = hidden_states_2.to(torch.float32)
        pow_2 = to_4.pow(2)
        to_4 = None
        variance_1 = pow_2.mean(-1, keepdim=True)
        pow_2 = None
        add_7 = variance_1 + 1e-06
        variance_1 = None
        rsqrt_1 = torch.rsqrt(add_7)
        add_7 = None
        hidden_states_3 = hidden_states_2 * rsqrt_1
        rsqrt_1 = None
        normed_hidden_states_1 = (
            l_self_modules_block_modules_0_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_3
        )
        l_self_modules_block_modules_0_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_3
        ) = None
        query_states_2 = torch._C._nn.linear(
            normed_hidden_states_1,
            l_self_modules_block_modules_0_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_,
            None,
        )
        normed_hidden_states_1 = l_self_modules_block_modules_0_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_ = (None)
        view_4 = query_states_2.view(1, -1, 8, 64)
        query_states_2 = None
        query_states_3 = view_4.transpose(1, 2)
        view_4 = None
        key_states_2 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_block_modules_0_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_0_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_2 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_block_modules_0_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_0_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_ = (
            None
        )
        view_5 = key_states_2.view(1, -1, 8, 64)
        key_states_2 = None
        key_states_3 = view_5.transpose(1, 2)
        view_5 = None
        view_6 = value_states_2.view(1, -1, 8, 64)
        value_states_2 = None
        value_states_3 = view_6.transpose(1, 2)
        view_6 = None
        transpose_8 = key_states_3.transpose(3, 2)
        scores_2 = torch.matmul(query_states_3, transpose_8)
        query_states_3 = transpose_8 = None
        position_bias_2 = torch.zeros(
            (1, 8, 23, 23), device=device(type="cuda", index=0), dtype=torch.float32
        )
        causal_mask_5 = encoder_extended_attention_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 23, None),
            )
        ]
        encoder_extended_attention_mask_2 = None
        position_bias_3 = position_bias_2 + causal_mask_5
        position_bias_2 = causal_mask_5 = None
        scores_2 += position_bias_3
        scores_3 = scores_2
        scores_2 = None
        float_3 = scores_3.float()
        softmax_1 = torch.nn.functional.softmax(float_3, dim=-1)
        float_3 = None
        attn_weights_2 = softmax_1.type_as(scores_3)
        softmax_1 = scores_3 = None
        attn_weights_3 = torch.nn.functional.dropout(
            attn_weights_2, p=0.1, training=False
        )
        attn_weights_2 = None
        attn_output_4 = torch.matmul(attn_weights_3, value_states_3)
        attn_weights_3 = None
        transpose_9 = attn_output_4.transpose(1, 2)
        attn_output_4 = None
        attn_output_5 = transpose_9.contiguous()
        transpose_9 = None
        attn_output_6 = attn_output_5.view(1, -1, 512)
        attn_output_5 = None
        attn_output_7 = torch._C._nn.linear(
            attn_output_6,
            l_self_modules_block_modules_0_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_6 = l_self_modules_block_modules_0_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_ = (None)
        dropout_4 = torch.nn.functional.dropout(attn_output_7, 0.1, False, False)
        attn_output_7 = None
        layer_output = hidden_states_2 + dropout_4
        hidden_states_2 = dropout_4 = None
        to_5 = layer_output.to(torch.float32)
        pow_3 = to_5.pow(2)
        to_5 = None
        variance_2 = pow_3.mean(-1, keepdim=True)
        pow_3 = None
        add_10 = variance_2 + 1e-06
        variance_2 = None
        rsqrt_2 = torch.rsqrt(add_10)
        add_10 = None
        hidden_states_4 = layer_output * rsqrt_2
        rsqrt_2 = None
        forwarded_states = (
            l_self_modules_block_modules_0_modules_layer_modules_2_modules_layer_norm_parameters_weight_
            * hidden_states_4
        )
        l_self_modules_block_modules_0_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = (
            hidden_states_4
        ) = None
        hidden_states_5 = torch._C._nn.linear(
            forwarded_states,
            l_self_modules_block_modules_0_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states = l_self_modules_block_modules_0_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_6 = torch.nn.functional.relu(hidden_states_5, inplace=False)
        hidden_states_5 = None
        hidden_states_7 = torch.nn.functional.dropout(
            hidden_states_6, 0.1, False, False
        )
        hidden_states_6 = None
        hidden_states_8 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_block_modules_0_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_7 = l_self_modules_block_modules_0_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_6 = torch.nn.functional.dropout(hidden_states_8, 0.1, False, False)
        hidden_states_8 = None
        hidden_states_9 = layer_output + dropout_6
        layer_output = dropout_6 = None
        to_6 = hidden_states_9.to(torch.float32)
        pow_4 = to_6.pow(2)
        to_6 = None
        variance_3 = pow_4.mean(-1, keepdim=True)
        pow_4 = None
        add_12 = variance_3 + 1e-06
        variance_3 = None
        rsqrt_3 = torch.rsqrt(add_12)
        add_12 = None
        hidden_states_10 = hidden_states_9 * rsqrt_3
        rsqrt_3 = None
        normed_hidden_states_2 = (
            l_self_modules_block_modules_1_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_10
        )
        l_self_modules_block_modules_1_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_10
        ) = None
        query_states_4 = torch._C._nn.linear(
            normed_hidden_states_2,
            l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_8 = query_states_4.view(1, -1, 8, 64)
        query_states_4 = None
        query_states_5 = view_8.transpose(1, 2)
        view_8 = None
        key_states_4 = torch._C._nn.linear(
            normed_hidden_states_2,
            l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_4 = torch._C._nn.linear(
            normed_hidden_states_2,
            l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_2 = l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_9 = key_states_4.view(1, -1, 8, 64)
        key_states_4 = None
        key_states_5 = view_9.transpose(1, 2)
        view_9 = None
        view_10 = value_states_4.view(1, -1, 8, 64)
        value_states_4 = None
        value_states_5 = view_10.transpose(1, 2)
        view_10 = None
        transpose_13 = key_states_5.transpose(3, 2)
        scores_4 = torch.matmul(query_states_5, transpose_13)
        query_states_5 = transpose_13 = None
        scores_4 += position_bias_1
        scores_5 = scores_4
        scores_4 = None
        float_4 = scores_5.float()
        softmax_2 = torch.nn.functional.softmax(float_4, dim=-1)
        float_4 = None
        attn_weights_4 = softmax_2.type_as(scores_5)
        softmax_2 = scores_5 = None
        attn_weights_5 = torch.nn.functional.dropout(
            attn_weights_4, p=0.1, training=False
        )
        attn_weights_4 = None
        attn_output_8 = torch.matmul(attn_weights_5, value_states_5)
        attn_weights_5 = None
        transpose_14 = attn_output_8.transpose(1, 2)
        attn_output_8 = None
        attn_output_9 = transpose_14.contiguous()
        transpose_14 = None
        attn_output_10 = attn_output_9.view(1, -1, 512)
        attn_output_9 = None
        attn_output_11 = torch._C._nn.linear(
            attn_output_10,
            l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_10 = l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_8 = torch.nn.functional.dropout(attn_output_11, 0.1, False, False)
        attn_output_11 = None
        hidden_states_11 = hidden_states_9 + dropout_8
        hidden_states_9 = dropout_8 = None
        getitem_9 = cache_position[-1]
        add_14 = getitem_9 + 1
        getitem_9 = add_14 = None
        to_7 = hidden_states_11.to(torch.float32)
        pow_5 = to_7.pow(2)
        to_7 = None
        variance_4 = pow_5.mean(-1, keepdim=True)
        pow_5 = None
        add_15 = variance_4 + 1e-06
        variance_4 = None
        rsqrt_4 = torch.rsqrt(add_15)
        add_15 = None
        hidden_states_12 = hidden_states_11 * rsqrt_4
        rsqrt_4 = None
        normed_hidden_states_3 = (
            l_self_modules_block_modules_1_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_12
        )
        l_self_modules_block_modules_1_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_12
        ) = None
        query_states_6 = torch._C._nn.linear(
            normed_hidden_states_3,
            l_self_modules_block_modules_1_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_,
            None,
        )
        normed_hidden_states_3 = l_self_modules_block_modules_1_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_ = (None)
        view_12 = query_states_6.view(1, -1, 8, 64)
        query_states_6 = None
        query_states_7 = view_12.transpose(1, 2)
        view_12 = None
        key_states_6 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_block_modules_1_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_1_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_6 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_block_modules_1_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_1_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_ = (
            None
        )
        view_13 = key_states_6.view(1, -1, 8, 64)
        key_states_6 = None
        key_states_7 = view_13.transpose(1, 2)
        view_13 = None
        view_14 = value_states_6.view(1, -1, 8, 64)
        value_states_6 = None
        value_states_7 = view_14.transpose(1, 2)
        view_14 = None
        transpose_18 = key_states_7.transpose(3, 2)
        scores_6 = torch.matmul(query_states_7, transpose_18)
        query_states_7 = transpose_18 = None
        scores_6 += position_bias_3
        scores_7 = scores_6
        scores_6 = None
        float_5 = scores_7.float()
        softmax_3 = torch.nn.functional.softmax(float_5, dim=-1)
        float_5 = None
        attn_weights_6 = softmax_3.type_as(scores_7)
        softmax_3 = scores_7 = None
        attn_weights_7 = torch.nn.functional.dropout(
            attn_weights_6, p=0.1, training=False
        )
        attn_weights_6 = None
        attn_output_12 = torch.matmul(attn_weights_7, value_states_7)
        attn_weights_7 = None
        transpose_19 = attn_output_12.transpose(1, 2)
        attn_output_12 = None
        attn_output_13 = transpose_19.contiguous()
        transpose_19 = None
        attn_output_14 = attn_output_13.view(1, -1, 512)
        attn_output_13 = None
        attn_output_15 = torch._C._nn.linear(
            attn_output_14,
            l_self_modules_block_modules_1_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_14 = l_self_modules_block_modules_1_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_ = (None)
        dropout_10 = torch.nn.functional.dropout(attn_output_15, 0.1, False, False)
        attn_output_15 = None
        layer_output_1 = hidden_states_11 + dropout_10
        hidden_states_11 = dropout_10 = None
        to_8 = layer_output_1.to(torch.float32)
        pow_6 = to_8.pow(2)
        to_8 = None
        variance_5 = pow_6.mean(-1, keepdim=True)
        pow_6 = None
        add_17 = variance_5 + 1e-06
        variance_5 = None
        rsqrt_5 = torch.rsqrt(add_17)
        add_17 = None
        hidden_states_13 = layer_output_1 * rsqrt_5
        rsqrt_5 = None
        forwarded_states_1 = (
            l_self_modules_block_modules_1_modules_layer_modules_2_modules_layer_norm_parameters_weight_
            * hidden_states_13
        )
        l_self_modules_block_modules_1_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = (
            hidden_states_13
        ) = None
        hidden_states_14 = torch._C._nn.linear(
            forwarded_states_1,
            l_self_modules_block_modules_1_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_1 = l_self_modules_block_modules_1_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_15 = torch.nn.functional.relu(hidden_states_14, inplace=False)
        hidden_states_14 = None
        hidden_states_16 = torch.nn.functional.dropout(
            hidden_states_15, 0.1, False, False
        )
        hidden_states_15 = None
        hidden_states_17 = torch._C._nn.linear(
            hidden_states_16,
            l_self_modules_block_modules_1_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_16 = l_self_modules_block_modules_1_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_12 = torch.nn.functional.dropout(hidden_states_17, 0.1, False, False)
        hidden_states_17 = None
        hidden_states_18 = layer_output_1 + dropout_12
        layer_output_1 = dropout_12 = None
        to_9 = hidden_states_18.to(torch.float32)
        pow_7 = to_9.pow(2)
        to_9 = None
        variance_6 = pow_7.mean(-1, keepdim=True)
        pow_7 = None
        add_19 = variance_6 + 1e-06
        variance_6 = None
        rsqrt_6 = torch.rsqrt(add_19)
        add_19 = None
        hidden_states_19 = hidden_states_18 * rsqrt_6
        rsqrt_6 = None
        normed_hidden_states_4 = (
            l_self_modules_block_modules_2_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_19
        )
        l_self_modules_block_modules_2_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_19
        ) = None
        query_states_8 = torch._C._nn.linear(
            normed_hidden_states_4,
            l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_16 = query_states_8.view(1, -1, 8, 64)
        query_states_8 = None
        query_states_9 = view_16.transpose(1, 2)
        view_16 = None
        key_states_8 = torch._C._nn.linear(
            normed_hidden_states_4,
            l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_8 = torch._C._nn.linear(
            normed_hidden_states_4,
            l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_4 = l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_17 = key_states_8.view(1, -1, 8, 64)
        key_states_8 = None
        key_states_9 = view_17.transpose(1, 2)
        view_17 = None
        view_18 = value_states_8.view(1, -1, 8, 64)
        value_states_8 = None
        value_states_9 = view_18.transpose(1, 2)
        view_18 = None
        transpose_23 = key_states_9.transpose(3, 2)
        scores_8 = torch.matmul(query_states_9, transpose_23)
        query_states_9 = transpose_23 = None
        scores_8 += position_bias_1
        scores_9 = scores_8
        scores_8 = None
        float_6 = scores_9.float()
        softmax_4 = torch.nn.functional.softmax(float_6, dim=-1)
        float_6 = None
        attn_weights_8 = softmax_4.type_as(scores_9)
        softmax_4 = scores_9 = None
        attn_weights_9 = torch.nn.functional.dropout(
            attn_weights_8, p=0.1, training=False
        )
        attn_weights_8 = None
        attn_output_16 = torch.matmul(attn_weights_9, value_states_9)
        attn_weights_9 = None
        transpose_24 = attn_output_16.transpose(1, 2)
        attn_output_16 = None
        attn_output_17 = transpose_24.contiguous()
        transpose_24 = None
        attn_output_18 = attn_output_17.view(1, -1, 512)
        attn_output_17 = None
        attn_output_19 = torch._C._nn.linear(
            attn_output_18,
            l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_18 = l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_14 = torch.nn.functional.dropout(attn_output_19, 0.1, False, False)
        attn_output_19 = None
        hidden_states_20 = hidden_states_18 + dropout_14
        hidden_states_18 = dropout_14 = None
        getitem_10 = cache_position[-1]
        add_21 = getitem_10 + 1
        getitem_10 = add_21 = None
        to_10 = hidden_states_20.to(torch.float32)
        pow_8 = to_10.pow(2)
        to_10 = None
        variance_7 = pow_8.mean(-1, keepdim=True)
        pow_8 = None
        add_22 = variance_7 + 1e-06
        variance_7 = None
        rsqrt_7 = torch.rsqrt(add_22)
        add_22 = None
        hidden_states_21 = hidden_states_20 * rsqrt_7
        rsqrt_7 = None
        normed_hidden_states_5 = (
            l_self_modules_block_modules_2_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_21
        )
        l_self_modules_block_modules_2_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_21
        ) = None
        query_states_10 = torch._C._nn.linear(
            normed_hidden_states_5,
            l_self_modules_block_modules_2_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_,
            None,
        )
        normed_hidden_states_5 = l_self_modules_block_modules_2_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_ = (None)
        view_20 = query_states_10.view(1, -1, 8, 64)
        query_states_10 = None
        query_states_11 = view_20.transpose(1, 2)
        view_20 = None
        key_states_10 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_block_modules_2_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_2_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_10 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_block_modules_2_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_2_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_ = (
            None
        )
        view_21 = key_states_10.view(1, -1, 8, 64)
        key_states_10 = None
        key_states_11 = view_21.transpose(1, 2)
        view_21 = None
        view_22 = value_states_10.view(1, -1, 8, 64)
        value_states_10 = None
        value_states_11 = view_22.transpose(1, 2)
        view_22 = None
        transpose_28 = key_states_11.transpose(3, 2)
        scores_10 = torch.matmul(query_states_11, transpose_28)
        query_states_11 = transpose_28 = None
        scores_10 += position_bias_3
        scores_11 = scores_10
        scores_10 = None
        float_7 = scores_11.float()
        softmax_5 = torch.nn.functional.softmax(float_7, dim=-1)
        float_7 = None
        attn_weights_10 = softmax_5.type_as(scores_11)
        softmax_5 = scores_11 = None
        attn_weights_11 = torch.nn.functional.dropout(
            attn_weights_10, p=0.1, training=False
        )
        attn_weights_10 = None
        attn_output_20 = torch.matmul(attn_weights_11, value_states_11)
        attn_weights_11 = None
        transpose_29 = attn_output_20.transpose(1, 2)
        attn_output_20 = None
        attn_output_21 = transpose_29.contiguous()
        transpose_29 = None
        attn_output_22 = attn_output_21.view(1, -1, 512)
        attn_output_21 = None
        attn_output_23 = torch._C._nn.linear(
            attn_output_22,
            l_self_modules_block_modules_2_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_22 = l_self_modules_block_modules_2_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_ = (None)
        dropout_16 = torch.nn.functional.dropout(attn_output_23, 0.1, False, False)
        attn_output_23 = None
        layer_output_2 = hidden_states_20 + dropout_16
        hidden_states_20 = dropout_16 = None
        to_11 = layer_output_2.to(torch.float32)
        pow_9 = to_11.pow(2)
        to_11 = None
        variance_8 = pow_9.mean(-1, keepdim=True)
        pow_9 = None
        add_24 = variance_8 + 1e-06
        variance_8 = None
        rsqrt_8 = torch.rsqrt(add_24)
        add_24 = None
        hidden_states_22 = layer_output_2 * rsqrt_8
        rsqrt_8 = None
        forwarded_states_2 = (
            l_self_modules_block_modules_2_modules_layer_modules_2_modules_layer_norm_parameters_weight_
            * hidden_states_22
        )
        l_self_modules_block_modules_2_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = (
            hidden_states_22
        ) = None
        hidden_states_23 = torch._C._nn.linear(
            forwarded_states_2,
            l_self_modules_block_modules_2_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_2 = l_self_modules_block_modules_2_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_24 = torch.nn.functional.relu(hidden_states_23, inplace=False)
        hidden_states_23 = None
        hidden_states_25 = torch.nn.functional.dropout(
            hidden_states_24, 0.1, False, False
        )
        hidden_states_24 = None
        hidden_states_26 = torch._C._nn.linear(
            hidden_states_25,
            l_self_modules_block_modules_2_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_25 = l_self_modules_block_modules_2_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_18 = torch.nn.functional.dropout(hidden_states_26, 0.1, False, False)
        hidden_states_26 = None
        hidden_states_27 = layer_output_2 + dropout_18
        layer_output_2 = dropout_18 = None
        to_12 = hidden_states_27.to(torch.float32)
        pow_10 = to_12.pow(2)
        to_12 = None
        variance_9 = pow_10.mean(-1, keepdim=True)
        pow_10 = None
        add_26 = variance_9 + 1e-06
        variance_9 = None
        rsqrt_9 = torch.rsqrt(add_26)
        add_26 = None
        hidden_states_28 = hidden_states_27 * rsqrt_9
        rsqrt_9 = None
        normed_hidden_states_6 = (
            l_self_modules_block_modules_3_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_28
        )
        l_self_modules_block_modules_3_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_28
        ) = None
        query_states_12 = torch._C._nn.linear(
            normed_hidden_states_6,
            l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_24 = query_states_12.view(1, -1, 8, 64)
        query_states_12 = None
        query_states_13 = view_24.transpose(1, 2)
        view_24 = None
        key_states_12 = torch._C._nn.linear(
            normed_hidden_states_6,
            l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_12 = torch._C._nn.linear(
            normed_hidden_states_6,
            l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_6 = l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_25 = key_states_12.view(1, -1, 8, 64)
        key_states_12 = None
        key_states_13 = view_25.transpose(1, 2)
        view_25 = None
        view_26 = value_states_12.view(1, -1, 8, 64)
        value_states_12 = None
        value_states_13 = view_26.transpose(1, 2)
        view_26 = None
        transpose_33 = key_states_13.transpose(3, 2)
        scores_12 = torch.matmul(query_states_13, transpose_33)
        query_states_13 = transpose_33 = None
        scores_12 += position_bias_1
        scores_13 = scores_12
        scores_12 = None
        float_8 = scores_13.float()
        softmax_6 = torch.nn.functional.softmax(float_8, dim=-1)
        float_8 = None
        attn_weights_12 = softmax_6.type_as(scores_13)
        softmax_6 = scores_13 = None
        attn_weights_13 = torch.nn.functional.dropout(
            attn_weights_12, p=0.1, training=False
        )
        attn_weights_12 = None
        attn_output_24 = torch.matmul(attn_weights_13, value_states_13)
        attn_weights_13 = None
        transpose_34 = attn_output_24.transpose(1, 2)
        attn_output_24 = None
        attn_output_25 = transpose_34.contiguous()
        transpose_34 = None
        attn_output_26 = attn_output_25.view(1, -1, 512)
        attn_output_25 = None
        attn_output_27 = torch._C._nn.linear(
            attn_output_26,
            l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_26 = l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_20 = torch.nn.functional.dropout(attn_output_27, 0.1, False, False)
        attn_output_27 = None
        hidden_states_29 = hidden_states_27 + dropout_20
        hidden_states_27 = dropout_20 = None
        getitem_11 = cache_position[-1]
        add_28 = getitem_11 + 1
        getitem_11 = add_28 = None
        to_13 = hidden_states_29.to(torch.float32)
        pow_11 = to_13.pow(2)
        to_13 = None
        variance_10 = pow_11.mean(-1, keepdim=True)
        pow_11 = None
        add_29 = variance_10 + 1e-06
        variance_10 = None
        rsqrt_10 = torch.rsqrt(add_29)
        add_29 = None
        hidden_states_30 = hidden_states_29 * rsqrt_10
        rsqrt_10 = None
        normed_hidden_states_7 = (
            l_self_modules_block_modules_3_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_30
        )
        l_self_modules_block_modules_3_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_30
        ) = None
        query_states_14 = torch._C._nn.linear(
            normed_hidden_states_7,
            l_self_modules_block_modules_3_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_,
            None,
        )
        normed_hidden_states_7 = l_self_modules_block_modules_3_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_ = (None)
        view_28 = query_states_14.view(1, -1, 8, 64)
        query_states_14 = None
        query_states_15 = view_28.transpose(1, 2)
        view_28 = None
        key_states_14 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_block_modules_3_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_3_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_14 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_block_modules_3_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_3_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_ = (
            None
        )
        view_29 = key_states_14.view(1, -1, 8, 64)
        key_states_14 = None
        key_states_15 = view_29.transpose(1, 2)
        view_29 = None
        view_30 = value_states_14.view(1, -1, 8, 64)
        value_states_14 = None
        value_states_15 = view_30.transpose(1, 2)
        view_30 = None
        transpose_38 = key_states_15.transpose(3, 2)
        scores_14 = torch.matmul(query_states_15, transpose_38)
        query_states_15 = transpose_38 = None
        scores_14 += position_bias_3
        scores_15 = scores_14
        scores_14 = None
        float_9 = scores_15.float()
        softmax_7 = torch.nn.functional.softmax(float_9, dim=-1)
        float_9 = None
        attn_weights_14 = softmax_7.type_as(scores_15)
        softmax_7 = scores_15 = None
        attn_weights_15 = torch.nn.functional.dropout(
            attn_weights_14, p=0.1, training=False
        )
        attn_weights_14 = None
        attn_output_28 = torch.matmul(attn_weights_15, value_states_15)
        attn_weights_15 = None
        transpose_39 = attn_output_28.transpose(1, 2)
        attn_output_28 = None
        attn_output_29 = transpose_39.contiguous()
        transpose_39 = None
        attn_output_30 = attn_output_29.view(1, -1, 512)
        attn_output_29 = None
        attn_output_31 = torch._C._nn.linear(
            attn_output_30,
            l_self_modules_block_modules_3_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_30 = l_self_modules_block_modules_3_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_ = (None)
        dropout_22 = torch.nn.functional.dropout(attn_output_31, 0.1, False, False)
        attn_output_31 = None
        layer_output_3 = hidden_states_29 + dropout_22
        hidden_states_29 = dropout_22 = None
        to_14 = layer_output_3.to(torch.float32)
        pow_12 = to_14.pow(2)
        to_14 = None
        variance_11 = pow_12.mean(-1, keepdim=True)
        pow_12 = None
        add_31 = variance_11 + 1e-06
        variance_11 = None
        rsqrt_11 = torch.rsqrt(add_31)
        add_31 = None
        hidden_states_31 = layer_output_3 * rsqrt_11
        rsqrt_11 = None
        forwarded_states_3 = (
            l_self_modules_block_modules_3_modules_layer_modules_2_modules_layer_norm_parameters_weight_
            * hidden_states_31
        )
        l_self_modules_block_modules_3_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = (
            hidden_states_31
        ) = None
        hidden_states_32 = torch._C._nn.linear(
            forwarded_states_3,
            l_self_modules_block_modules_3_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_3 = l_self_modules_block_modules_3_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_33 = torch.nn.functional.relu(hidden_states_32, inplace=False)
        hidden_states_32 = None
        hidden_states_34 = torch.nn.functional.dropout(
            hidden_states_33, 0.1, False, False
        )
        hidden_states_33 = None
        hidden_states_35 = torch._C._nn.linear(
            hidden_states_34,
            l_self_modules_block_modules_3_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_34 = l_self_modules_block_modules_3_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_24 = torch.nn.functional.dropout(hidden_states_35, 0.1, False, False)
        hidden_states_35 = None
        hidden_states_36 = layer_output_3 + dropout_24
        layer_output_3 = dropout_24 = None
        to_15 = hidden_states_36.to(torch.float32)
        pow_13 = to_15.pow(2)
        to_15 = None
        variance_12 = pow_13.mean(-1, keepdim=True)
        pow_13 = None
        add_33 = variance_12 + 1e-06
        variance_12 = None
        rsqrt_12 = torch.rsqrt(add_33)
        add_33 = None
        hidden_states_37 = hidden_states_36 * rsqrt_12
        rsqrt_12 = None
        normed_hidden_states_8 = (
            l_self_modules_block_modules_4_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_37
        )
        l_self_modules_block_modules_4_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_37
        ) = None
        query_states_16 = torch._C._nn.linear(
            normed_hidden_states_8,
            l_self_modules_block_modules_4_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_4_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_32 = query_states_16.view(1, -1, 8, 64)
        query_states_16 = None
        query_states_17 = view_32.transpose(1, 2)
        view_32 = None
        key_states_16 = torch._C._nn.linear(
            normed_hidden_states_8,
            l_self_modules_block_modules_4_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_4_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_16 = torch._C._nn.linear(
            normed_hidden_states_8,
            l_self_modules_block_modules_4_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_8 = l_self_modules_block_modules_4_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_33 = key_states_16.view(1, -1, 8, 64)
        key_states_16 = None
        key_states_17 = view_33.transpose(1, 2)
        view_33 = None
        view_34 = value_states_16.view(1, -1, 8, 64)
        value_states_16 = None
        value_states_17 = view_34.transpose(1, 2)
        view_34 = None
        transpose_43 = key_states_17.transpose(3, 2)
        scores_16 = torch.matmul(query_states_17, transpose_43)
        query_states_17 = transpose_43 = None
        scores_16 += position_bias_1
        scores_17 = scores_16
        scores_16 = None
        float_10 = scores_17.float()
        softmax_8 = torch.nn.functional.softmax(float_10, dim=-1)
        float_10 = None
        attn_weights_16 = softmax_8.type_as(scores_17)
        softmax_8 = scores_17 = None
        attn_weights_17 = torch.nn.functional.dropout(
            attn_weights_16, p=0.1, training=False
        )
        attn_weights_16 = None
        attn_output_32 = torch.matmul(attn_weights_17, value_states_17)
        attn_weights_17 = None
        transpose_44 = attn_output_32.transpose(1, 2)
        attn_output_32 = None
        attn_output_33 = transpose_44.contiguous()
        transpose_44 = None
        attn_output_34 = attn_output_33.view(1, -1, 512)
        attn_output_33 = None
        attn_output_35 = torch._C._nn.linear(
            attn_output_34,
            l_self_modules_block_modules_4_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_34 = l_self_modules_block_modules_4_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_26 = torch.nn.functional.dropout(attn_output_35, 0.1, False, False)
        attn_output_35 = None
        hidden_states_38 = hidden_states_36 + dropout_26
        hidden_states_36 = dropout_26 = None
        getitem_12 = cache_position[-1]
        add_35 = getitem_12 + 1
        getitem_12 = add_35 = None
        to_16 = hidden_states_38.to(torch.float32)
        pow_14 = to_16.pow(2)
        to_16 = None
        variance_13 = pow_14.mean(-1, keepdim=True)
        pow_14 = None
        add_36 = variance_13 + 1e-06
        variance_13 = None
        rsqrt_13 = torch.rsqrt(add_36)
        add_36 = None
        hidden_states_39 = hidden_states_38 * rsqrt_13
        rsqrt_13 = None
        normed_hidden_states_9 = (
            l_self_modules_block_modules_4_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_39
        )
        l_self_modules_block_modules_4_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_39
        ) = None
        query_states_18 = torch._C._nn.linear(
            normed_hidden_states_9,
            l_self_modules_block_modules_4_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_,
            None,
        )
        normed_hidden_states_9 = l_self_modules_block_modules_4_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_ = (None)
        view_36 = query_states_18.view(1, -1, 8, 64)
        query_states_18 = None
        query_states_19 = view_36.transpose(1, 2)
        view_36 = None
        key_states_18 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_block_modules_4_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_4_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_18 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_block_modules_4_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_4_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_ = (
            None
        )
        view_37 = key_states_18.view(1, -1, 8, 64)
        key_states_18 = None
        key_states_19 = view_37.transpose(1, 2)
        view_37 = None
        view_38 = value_states_18.view(1, -1, 8, 64)
        value_states_18 = None
        value_states_19 = view_38.transpose(1, 2)
        view_38 = None
        transpose_48 = key_states_19.transpose(3, 2)
        scores_18 = torch.matmul(query_states_19, transpose_48)
        query_states_19 = transpose_48 = None
        scores_18 += position_bias_3
        scores_19 = scores_18
        scores_18 = None
        float_11 = scores_19.float()
        softmax_9 = torch.nn.functional.softmax(float_11, dim=-1)
        float_11 = None
        attn_weights_18 = softmax_9.type_as(scores_19)
        softmax_9 = scores_19 = None
        attn_weights_19 = torch.nn.functional.dropout(
            attn_weights_18, p=0.1, training=False
        )
        attn_weights_18 = None
        attn_output_36 = torch.matmul(attn_weights_19, value_states_19)
        attn_weights_19 = None
        transpose_49 = attn_output_36.transpose(1, 2)
        attn_output_36 = None
        attn_output_37 = transpose_49.contiguous()
        transpose_49 = None
        attn_output_38 = attn_output_37.view(1, -1, 512)
        attn_output_37 = None
        attn_output_39 = torch._C._nn.linear(
            attn_output_38,
            l_self_modules_block_modules_4_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_38 = l_self_modules_block_modules_4_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_ = (None)
        dropout_28 = torch.nn.functional.dropout(attn_output_39, 0.1, False, False)
        attn_output_39 = None
        layer_output_4 = hidden_states_38 + dropout_28
        hidden_states_38 = dropout_28 = None
        to_17 = layer_output_4.to(torch.float32)
        pow_15 = to_17.pow(2)
        to_17 = None
        variance_14 = pow_15.mean(-1, keepdim=True)
        pow_15 = None
        add_38 = variance_14 + 1e-06
        variance_14 = None
        rsqrt_14 = torch.rsqrt(add_38)
        add_38 = None
        hidden_states_40 = layer_output_4 * rsqrt_14
        rsqrt_14 = None
        forwarded_states_4 = (
            l_self_modules_block_modules_4_modules_layer_modules_2_modules_layer_norm_parameters_weight_
            * hidden_states_40
        )
        l_self_modules_block_modules_4_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = (
            hidden_states_40
        ) = None
        hidden_states_41 = torch._C._nn.linear(
            forwarded_states_4,
            l_self_modules_block_modules_4_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_4 = l_self_modules_block_modules_4_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_42 = torch.nn.functional.relu(hidden_states_41, inplace=False)
        hidden_states_41 = None
        hidden_states_43 = torch.nn.functional.dropout(
            hidden_states_42, 0.1, False, False
        )
        hidden_states_42 = None
        hidden_states_44 = torch._C._nn.linear(
            hidden_states_43,
            l_self_modules_block_modules_4_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_43 = l_self_modules_block_modules_4_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_30 = torch.nn.functional.dropout(hidden_states_44, 0.1, False, False)
        hidden_states_44 = None
        hidden_states_45 = layer_output_4 + dropout_30
        layer_output_4 = dropout_30 = None
        to_18 = hidden_states_45.to(torch.float32)
        pow_16 = to_18.pow(2)
        to_18 = None
        variance_15 = pow_16.mean(-1, keepdim=True)
        pow_16 = None
        add_40 = variance_15 + 1e-06
        variance_15 = None
        rsqrt_15 = torch.rsqrt(add_40)
        add_40 = None
        hidden_states_46 = hidden_states_45 * rsqrt_15
        rsqrt_15 = None
        normed_hidden_states_10 = (
            l_self_modules_block_modules_5_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_46
        )
        l_self_modules_block_modules_5_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_46
        ) = None
        query_states_20 = torch._C._nn.linear(
            normed_hidden_states_10,
            l_self_modules_block_modules_5_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_5_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_40 = query_states_20.view(1, -1, 8, 64)
        query_states_20 = None
        query_states_21 = view_40.transpose(1, 2)
        view_40 = None
        key_states_20 = torch._C._nn.linear(
            normed_hidden_states_10,
            l_self_modules_block_modules_5_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_5_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_20 = torch._C._nn.linear(
            normed_hidden_states_10,
            l_self_modules_block_modules_5_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_10 = l_self_modules_block_modules_5_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_41 = key_states_20.view(1, -1, 8, 64)
        key_states_20 = None
        key_states_21 = view_41.transpose(1, 2)
        view_41 = None
        view_42 = value_states_20.view(1, -1, 8, 64)
        value_states_20 = None
        value_states_21 = view_42.transpose(1, 2)
        view_42 = None
        transpose_53 = key_states_21.transpose(3, 2)
        scores_20 = torch.matmul(query_states_21, transpose_53)
        query_states_21 = transpose_53 = None
        scores_20 += position_bias_1
        scores_21 = scores_20
        scores_20 = None
        float_12 = scores_21.float()
        softmax_10 = torch.nn.functional.softmax(float_12, dim=-1)
        float_12 = None
        attn_weights_20 = softmax_10.type_as(scores_21)
        softmax_10 = scores_21 = None
        attn_weights_21 = torch.nn.functional.dropout(
            attn_weights_20, p=0.1, training=False
        )
        attn_weights_20 = None
        attn_output_40 = torch.matmul(attn_weights_21, value_states_21)
        attn_weights_21 = None
        transpose_54 = attn_output_40.transpose(1, 2)
        attn_output_40 = None
        attn_output_41 = transpose_54.contiguous()
        transpose_54 = None
        attn_output_42 = attn_output_41.view(1, -1, 512)
        attn_output_41 = None
        attn_output_43 = torch._C._nn.linear(
            attn_output_42,
            l_self_modules_block_modules_5_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_42 = l_self_modules_block_modules_5_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_32 = torch.nn.functional.dropout(attn_output_43, 0.1, False, False)
        attn_output_43 = None
        hidden_states_47 = hidden_states_45 + dropout_32
        hidden_states_45 = dropout_32 = None
        getitem_13 = cache_position[-1]
        add_42 = getitem_13 + 1
        getitem_13 = add_42 = None
        to_19 = hidden_states_47.to(torch.float32)
        pow_17 = to_19.pow(2)
        to_19 = None
        variance_16 = pow_17.mean(-1, keepdim=True)
        pow_17 = None
        add_43 = variance_16 + 1e-06
        variance_16 = None
        rsqrt_16 = torch.rsqrt(add_43)
        add_43 = None
        hidden_states_48 = hidden_states_47 * rsqrt_16
        rsqrt_16 = None
        normed_hidden_states_11 = (
            l_self_modules_block_modules_5_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_48
        )
        l_self_modules_block_modules_5_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_48
        ) = None
        query_states_22 = torch._C._nn.linear(
            normed_hidden_states_11,
            l_self_modules_block_modules_5_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_,
            None,
        )
        normed_hidden_states_11 = l_self_modules_block_modules_5_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_ = (None)
        view_44 = query_states_22.view(1, -1, 8, 64)
        query_states_22 = None
        query_states_23 = view_44.transpose(1, 2)
        view_44 = None
        key_states_22 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_block_modules_5_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_5_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_22 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_block_modules_5_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_5_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_ = (
            None
        )
        view_45 = key_states_22.view(1, -1, 8, 64)
        key_states_22 = None
        key_states_23 = view_45.transpose(1, 2)
        view_45 = None
        view_46 = value_states_22.view(1, -1, 8, 64)
        value_states_22 = None
        value_states_23 = view_46.transpose(1, 2)
        view_46 = None
        transpose_58 = key_states_23.transpose(3, 2)
        scores_22 = torch.matmul(query_states_23, transpose_58)
        query_states_23 = transpose_58 = None
        scores_22 += position_bias_3
        scores_23 = scores_22
        scores_22 = None
        float_13 = scores_23.float()
        softmax_11 = torch.nn.functional.softmax(float_13, dim=-1)
        float_13 = None
        attn_weights_22 = softmax_11.type_as(scores_23)
        softmax_11 = scores_23 = None
        attn_weights_23 = torch.nn.functional.dropout(
            attn_weights_22, p=0.1, training=False
        )
        attn_weights_22 = None
        attn_output_44 = torch.matmul(attn_weights_23, value_states_23)
        attn_weights_23 = None
        transpose_59 = attn_output_44.transpose(1, 2)
        attn_output_44 = None
        attn_output_45 = transpose_59.contiguous()
        transpose_59 = None
        attn_output_46 = attn_output_45.view(1, -1, 512)
        attn_output_45 = None
        attn_output_47 = torch._C._nn.linear(
            attn_output_46,
            l_self_modules_block_modules_5_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_46 = l_self_modules_block_modules_5_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_ = (None)
        dropout_34 = torch.nn.functional.dropout(attn_output_47, 0.1, False, False)
        attn_output_47 = None
        layer_output_5 = hidden_states_47 + dropout_34
        hidden_states_47 = dropout_34 = None
        to_20 = layer_output_5.to(torch.float32)
        pow_18 = to_20.pow(2)
        to_20 = None
        variance_17 = pow_18.mean(-1, keepdim=True)
        pow_18 = None
        add_45 = variance_17 + 1e-06
        variance_17 = None
        rsqrt_17 = torch.rsqrt(add_45)
        add_45 = None
        hidden_states_49 = layer_output_5 * rsqrt_17
        rsqrt_17 = None
        forwarded_states_5 = (
            l_self_modules_block_modules_5_modules_layer_modules_2_modules_layer_norm_parameters_weight_
            * hidden_states_49
        )
        l_self_modules_block_modules_5_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = (
            hidden_states_49
        ) = None
        hidden_states_50 = torch._C._nn.linear(
            forwarded_states_5,
            l_self_modules_block_modules_5_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_5 = l_self_modules_block_modules_5_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_51 = torch.nn.functional.relu(hidden_states_50, inplace=False)
        hidden_states_50 = None
        hidden_states_52 = torch.nn.functional.dropout(
            hidden_states_51, 0.1, False, False
        )
        hidden_states_51 = None
        hidden_states_53 = torch._C._nn.linear(
            hidden_states_52,
            l_self_modules_block_modules_5_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_52 = l_self_modules_block_modules_5_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_36 = torch.nn.functional.dropout(hidden_states_53, 0.1, False, False)
        hidden_states_53 = None
        hidden_states_54 = layer_output_5 + dropout_36
        layer_output_5 = dropout_36 = None
        to_21 = hidden_states_54.to(torch.float32)
        pow_19 = to_21.pow(2)
        to_21 = None
        variance_18 = pow_19.mean(-1, keepdim=True)
        pow_19 = None
        add_47 = variance_18 + 1e-06
        variance_18 = None
        rsqrt_18 = torch.rsqrt(add_47)
        add_47 = None
        hidden_states_55 = hidden_states_54 * rsqrt_18
        rsqrt_18 = None
        normed_hidden_states_12 = (
            l_self_modules_block_modules_6_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_55
        )
        l_self_modules_block_modules_6_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_55
        ) = None
        query_states_24 = torch._C._nn.linear(
            normed_hidden_states_12,
            l_self_modules_block_modules_6_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_6_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_48 = query_states_24.view(1, -1, 8, 64)
        query_states_24 = None
        query_states_25 = view_48.transpose(1, 2)
        view_48 = None
        key_states_24 = torch._C._nn.linear(
            normed_hidden_states_12,
            l_self_modules_block_modules_6_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_6_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_24 = torch._C._nn.linear(
            normed_hidden_states_12,
            l_self_modules_block_modules_6_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_12 = l_self_modules_block_modules_6_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_49 = key_states_24.view(1, -1, 8, 64)
        key_states_24 = None
        key_states_25 = view_49.transpose(1, 2)
        view_49 = None
        view_50 = value_states_24.view(1, -1, 8, 64)
        value_states_24 = None
        value_states_25 = view_50.transpose(1, 2)
        view_50 = None
        transpose_63 = key_states_25.transpose(3, 2)
        scores_24 = torch.matmul(query_states_25, transpose_63)
        query_states_25 = transpose_63 = None
        scores_24 += position_bias_1
        scores_25 = scores_24
        scores_24 = None
        float_14 = scores_25.float()
        softmax_12 = torch.nn.functional.softmax(float_14, dim=-1)
        float_14 = None
        attn_weights_24 = softmax_12.type_as(scores_25)
        softmax_12 = scores_25 = None
        attn_weights_25 = torch.nn.functional.dropout(
            attn_weights_24, p=0.1, training=False
        )
        attn_weights_24 = None
        attn_output_48 = torch.matmul(attn_weights_25, value_states_25)
        attn_weights_25 = None
        transpose_64 = attn_output_48.transpose(1, 2)
        attn_output_48 = None
        attn_output_49 = transpose_64.contiguous()
        transpose_64 = None
        attn_output_50 = attn_output_49.view(1, -1, 512)
        attn_output_49 = None
        attn_output_51 = torch._C._nn.linear(
            attn_output_50,
            l_self_modules_block_modules_6_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_50 = l_self_modules_block_modules_6_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_38 = torch.nn.functional.dropout(attn_output_51, 0.1, False, False)
        attn_output_51 = None
        hidden_states_56 = hidden_states_54 + dropout_38
        hidden_states_54 = dropout_38 = None
        getitem_14 = cache_position[-1]
        add_49 = getitem_14 + 1
        getitem_14 = add_49 = None
        to_22 = hidden_states_56.to(torch.float32)
        pow_20 = to_22.pow(2)
        to_22 = None
        variance_19 = pow_20.mean(-1, keepdim=True)
        pow_20 = None
        add_50 = variance_19 + 1e-06
        variance_19 = None
        rsqrt_19 = torch.rsqrt(add_50)
        add_50 = None
        hidden_states_57 = hidden_states_56 * rsqrt_19
        rsqrt_19 = None
        normed_hidden_states_13 = (
            l_self_modules_block_modules_6_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_57
        )
        l_self_modules_block_modules_6_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_57
        ) = None
        query_states_26 = torch._C._nn.linear(
            normed_hidden_states_13,
            l_self_modules_block_modules_6_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_,
            None,
        )
        normed_hidden_states_13 = l_self_modules_block_modules_6_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_ = (None)
        view_52 = query_states_26.view(1, -1, 8, 64)
        query_states_26 = None
        query_states_27 = view_52.transpose(1, 2)
        view_52 = None
        key_states_26 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_block_modules_6_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_6_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_26 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_block_modules_6_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_6_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_ = (
            None
        )
        view_53 = key_states_26.view(1, -1, 8, 64)
        key_states_26 = None
        key_states_27 = view_53.transpose(1, 2)
        view_53 = None
        view_54 = value_states_26.view(1, -1, 8, 64)
        value_states_26 = None
        value_states_27 = view_54.transpose(1, 2)
        view_54 = None
        transpose_68 = key_states_27.transpose(3, 2)
        scores_26 = torch.matmul(query_states_27, transpose_68)
        query_states_27 = transpose_68 = None
        scores_26 += position_bias_3
        scores_27 = scores_26
        scores_26 = None
        float_15 = scores_27.float()
        softmax_13 = torch.nn.functional.softmax(float_15, dim=-1)
        float_15 = None
        attn_weights_26 = softmax_13.type_as(scores_27)
        softmax_13 = scores_27 = None
        attn_weights_27 = torch.nn.functional.dropout(
            attn_weights_26, p=0.1, training=False
        )
        attn_weights_26 = None
        attn_output_52 = torch.matmul(attn_weights_27, value_states_27)
        attn_weights_27 = None
        transpose_69 = attn_output_52.transpose(1, 2)
        attn_output_52 = None
        attn_output_53 = transpose_69.contiguous()
        transpose_69 = None
        attn_output_54 = attn_output_53.view(1, -1, 512)
        attn_output_53 = None
        attn_output_55 = torch._C._nn.linear(
            attn_output_54,
            l_self_modules_block_modules_6_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_54 = l_self_modules_block_modules_6_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_ = (None)
        dropout_40 = torch.nn.functional.dropout(attn_output_55, 0.1, False, False)
        attn_output_55 = None
        layer_output_6 = hidden_states_56 + dropout_40
        hidden_states_56 = dropout_40 = None
        to_23 = layer_output_6.to(torch.float32)
        pow_21 = to_23.pow(2)
        to_23 = None
        variance_20 = pow_21.mean(-1, keepdim=True)
        pow_21 = None
        add_52 = variance_20 + 1e-06
        variance_20 = None
        rsqrt_20 = torch.rsqrt(add_52)
        add_52 = None
        hidden_states_58 = layer_output_6 * rsqrt_20
        rsqrt_20 = None
        forwarded_states_6 = (
            l_self_modules_block_modules_6_modules_layer_modules_2_modules_layer_norm_parameters_weight_
            * hidden_states_58
        )
        l_self_modules_block_modules_6_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = (
            hidden_states_58
        ) = None
        hidden_states_59 = torch._C._nn.linear(
            forwarded_states_6,
            l_self_modules_block_modules_6_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_6 = l_self_modules_block_modules_6_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_60 = torch.nn.functional.relu(hidden_states_59, inplace=False)
        hidden_states_59 = None
        hidden_states_61 = torch.nn.functional.dropout(
            hidden_states_60, 0.1, False, False
        )
        hidden_states_60 = None
        hidden_states_62 = torch._C._nn.linear(
            hidden_states_61,
            l_self_modules_block_modules_6_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_61 = l_self_modules_block_modules_6_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_42 = torch.nn.functional.dropout(hidden_states_62, 0.1, False, False)
        hidden_states_62 = None
        hidden_states_63 = layer_output_6 + dropout_42
        layer_output_6 = dropout_42 = None
        to_24 = hidden_states_63.to(torch.float32)
        pow_22 = to_24.pow(2)
        to_24 = None
        variance_21 = pow_22.mean(-1, keepdim=True)
        pow_22 = None
        add_54 = variance_21 + 1e-06
        variance_21 = None
        rsqrt_21 = torch.rsqrt(add_54)
        add_54 = None
        hidden_states_64 = hidden_states_63 * rsqrt_21
        rsqrt_21 = None
        normed_hidden_states_14 = (
            l_self_modules_block_modules_7_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_64
        )
        l_self_modules_block_modules_7_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_64
        ) = None
        query_states_28 = torch._C._nn.linear(
            normed_hidden_states_14,
            l_self_modules_block_modules_7_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_7_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_56 = query_states_28.view(1, -1, 8, 64)
        query_states_28 = None
        query_states_29 = view_56.transpose(1, 2)
        view_56 = None
        key_states_28 = torch._C._nn.linear(
            normed_hidden_states_14,
            l_self_modules_block_modules_7_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_7_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_28 = torch._C._nn.linear(
            normed_hidden_states_14,
            l_self_modules_block_modules_7_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_14 = l_self_modules_block_modules_7_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_57 = key_states_28.view(1, -1, 8, 64)
        key_states_28 = None
        key_states_29 = view_57.transpose(1, 2)
        view_57 = None
        view_58 = value_states_28.view(1, -1, 8, 64)
        value_states_28 = None
        value_states_29 = view_58.transpose(1, 2)
        view_58 = None
        transpose_73 = key_states_29.transpose(3, 2)
        scores_28 = torch.matmul(query_states_29, transpose_73)
        query_states_29 = transpose_73 = None
        scores_28 += position_bias_1
        scores_29 = scores_28
        scores_28 = None
        float_16 = scores_29.float()
        softmax_14 = torch.nn.functional.softmax(float_16, dim=-1)
        float_16 = None
        attn_weights_28 = softmax_14.type_as(scores_29)
        softmax_14 = scores_29 = None
        attn_weights_29 = torch.nn.functional.dropout(
            attn_weights_28, p=0.1, training=False
        )
        attn_weights_28 = None
        attn_output_56 = torch.matmul(attn_weights_29, value_states_29)
        attn_weights_29 = None
        transpose_74 = attn_output_56.transpose(1, 2)
        attn_output_56 = None
        attn_output_57 = transpose_74.contiguous()
        transpose_74 = None
        attn_output_58 = attn_output_57.view(1, -1, 512)
        attn_output_57 = None
        attn_output_59 = torch._C._nn.linear(
            attn_output_58,
            l_self_modules_block_modules_7_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_58 = l_self_modules_block_modules_7_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_44 = torch.nn.functional.dropout(attn_output_59, 0.1, False, False)
        attn_output_59 = None
        hidden_states_65 = hidden_states_63 + dropout_44
        hidden_states_63 = dropout_44 = None
        getitem_15 = cache_position[-1]
        add_56 = getitem_15 + 1
        getitem_15 = add_56 = None
        to_25 = hidden_states_65.to(torch.float32)
        pow_23 = to_25.pow(2)
        to_25 = None
        variance_22 = pow_23.mean(-1, keepdim=True)
        pow_23 = None
        add_57 = variance_22 + 1e-06
        variance_22 = None
        rsqrt_22 = torch.rsqrt(add_57)
        add_57 = None
        hidden_states_66 = hidden_states_65 * rsqrt_22
        rsqrt_22 = None
        normed_hidden_states_15 = (
            l_self_modules_block_modules_7_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_66
        )
        l_self_modules_block_modules_7_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_66
        ) = None
        query_states_30 = torch._C._nn.linear(
            normed_hidden_states_15,
            l_self_modules_block_modules_7_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_,
            None,
        )
        normed_hidden_states_15 = l_self_modules_block_modules_7_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_ = (None)
        view_60 = query_states_30.view(1, -1, 8, 64)
        query_states_30 = None
        query_states_31 = view_60.transpose(1, 2)
        view_60 = None
        key_states_30 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_block_modules_7_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_7_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_30 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_block_modules_7_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_7_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_ = (
            None
        )
        view_61 = key_states_30.view(1, -1, 8, 64)
        key_states_30 = None
        key_states_31 = view_61.transpose(1, 2)
        view_61 = None
        view_62 = value_states_30.view(1, -1, 8, 64)
        value_states_30 = None
        value_states_31 = view_62.transpose(1, 2)
        view_62 = None
        transpose_78 = key_states_31.transpose(3, 2)
        scores_30 = torch.matmul(query_states_31, transpose_78)
        query_states_31 = transpose_78 = None
        scores_30 += position_bias_3
        scores_31 = scores_30
        scores_30 = None
        float_17 = scores_31.float()
        softmax_15 = torch.nn.functional.softmax(float_17, dim=-1)
        float_17 = None
        attn_weights_30 = softmax_15.type_as(scores_31)
        softmax_15 = scores_31 = None
        attn_weights_31 = torch.nn.functional.dropout(
            attn_weights_30, p=0.1, training=False
        )
        attn_weights_30 = None
        attn_output_60 = torch.matmul(attn_weights_31, value_states_31)
        attn_weights_31 = None
        transpose_79 = attn_output_60.transpose(1, 2)
        attn_output_60 = None
        attn_output_61 = transpose_79.contiguous()
        transpose_79 = None
        attn_output_62 = attn_output_61.view(1, -1, 512)
        attn_output_61 = None
        attn_output_63 = torch._C._nn.linear(
            attn_output_62,
            l_self_modules_block_modules_7_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_62 = l_self_modules_block_modules_7_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_ = (None)
        dropout_46 = torch.nn.functional.dropout(attn_output_63, 0.1, False, False)
        attn_output_63 = None
        layer_output_7 = hidden_states_65 + dropout_46
        hidden_states_65 = dropout_46 = None
        to_26 = layer_output_7.to(torch.float32)
        pow_24 = to_26.pow(2)
        to_26 = None
        variance_23 = pow_24.mean(-1, keepdim=True)
        pow_24 = None
        add_59 = variance_23 + 1e-06
        variance_23 = None
        rsqrt_23 = torch.rsqrt(add_59)
        add_59 = None
        hidden_states_67 = layer_output_7 * rsqrt_23
        rsqrt_23 = None
        forwarded_states_7 = (
            l_self_modules_block_modules_7_modules_layer_modules_2_modules_layer_norm_parameters_weight_
            * hidden_states_67
        )
        l_self_modules_block_modules_7_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = (
            hidden_states_67
        ) = None
        hidden_states_68 = torch._C._nn.linear(
            forwarded_states_7,
            l_self_modules_block_modules_7_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_7 = l_self_modules_block_modules_7_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_69 = torch.nn.functional.relu(hidden_states_68, inplace=False)
        hidden_states_68 = None
        hidden_states_70 = torch.nn.functional.dropout(
            hidden_states_69, 0.1, False, False
        )
        hidden_states_69 = None
        hidden_states_71 = torch._C._nn.linear(
            hidden_states_70,
            l_self_modules_block_modules_7_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_70 = l_self_modules_block_modules_7_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_48 = torch.nn.functional.dropout(hidden_states_71, 0.1, False, False)
        hidden_states_71 = None
        hidden_states_72 = layer_output_7 + dropout_48
        layer_output_7 = dropout_48 = None
        to_27 = hidden_states_72.to(torch.float32)
        pow_25 = to_27.pow(2)
        to_27 = None
        variance_24 = pow_25.mean(-1, keepdim=True)
        pow_25 = None
        add_61 = variance_24 + 1e-06
        variance_24 = None
        rsqrt_24 = torch.rsqrt(add_61)
        add_61 = None
        hidden_states_73 = hidden_states_72 * rsqrt_24
        rsqrt_24 = None
        normed_hidden_states_16 = (
            l_self_modules_block_modules_8_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_73
        )
        l_self_modules_block_modules_8_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_73
        ) = None
        query_states_32 = torch._C._nn.linear(
            normed_hidden_states_16,
            l_self_modules_block_modules_8_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_8_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_64 = query_states_32.view(1, -1, 8, 64)
        query_states_32 = None
        query_states_33 = view_64.transpose(1, 2)
        view_64 = None
        key_states_32 = torch._C._nn.linear(
            normed_hidden_states_16,
            l_self_modules_block_modules_8_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_8_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_32 = torch._C._nn.linear(
            normed_hidden_states_16,
            l_self_modules_block_modules_8_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_16 = l_self_modules_block_modules_8_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_65 = key_states_32.view(1, -1, 8, 64)
        key_states_32 = None
        key_states_33 = view_65.transpose(1, 2)
        view_65 = None
        view_66 = value_states_32.view(1, -1, 8, 64)
        value_states_32 = None
        value_states_33 = view_66.transpose(1, 2)
        view_66 = None
        transpose_83 = key_states_33.transpose(3, 2)
        scores_32 = torch.matmul(query_states_33, transpose_83)
        query_states_33 = transpose_83 = None
        scores_32 += position_bias_1
        scores_33 = scores_32
        scores_32 = None
        float_18 = scores_33.float()
        softmax_16 = torch.nn.functional.softmax(float_18, dim=-1)
        float_18 = None
        attn_weights_32 = softmax_16.type_as(scores_33)
        softmax_16 = scores_33 = None
        attn_weights_33 = torch.nn.functional.dropout(
            attn_weights_32, p=0.1, training=False
        )
        attn_weights_32 = None
        attn_output_64 = torch.matmul(attn_weights_33, value_states_33)
        attn_weights_33 = None
        transpose_84 = attn_output_64.transpose(1, 2)
        attn_output_64 = None
        attn_output_65 = transpose_84.contiguous()
        transpose_84 = None
        attn_output_66 = attn_output_65.view(1, -1, 512)
        attn_output_65 = None
        attn_output_67 = torch._C._nn.linear(
            attn_output_66,
            l_self_modules_block_modules_8_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_66 = l_self_modules_block_modules_8_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_50 = torch.nn.functional.dropout(attn_output_67, 0.1, False, False)
        attn_output_67 = None
        hidden_states_74 = hidden_states_72 + dropout_50
        hidden_states_72 = dropout_50 = None
        getitem_16 = cache_position[-1]
        add_63 = getitem_16 + 1
        getitem_16 = add_63 = None
        to_28 = hidden_states_74.to(torch.float32)
        pow_26 = to_28.pow(2)
        to_28 = None
        variance_25 = pow_26.mean(-1, keepdim=True)
        pow_26 = None
        add_64 = variance_25 + 1e-06
        variance_25 = None
        rsqrt_25 = torch.rsqrt(add_64)
        add_64 = None
        hidden_states_75 = hidden_states_74 * rsqrt_25
        rsqrt_25 = None
        normed_hidden_states_17 = (
            l_self_modules_block_modules_8_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_75
        )
        l_self_modules_block_modules_8_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_75
        ) = None
        query_states_34 = torch._C._nn.linear(
            normed_hidden_states_17,
            l_self_modules_block_modules_8_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_,
            None,
        )
        normed_hidden_states_17 = l_self_modules_block_modules_8_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_ = (None)
        view_68 = query_states_34.view(1, -1, 8, 64)
        query_states_34 = None
        query_states_35 = view_68.transpose(1, 2)
        view_68 = None
        key_states_34 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_block_modules_8_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_8_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_34 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_block_modules_8_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_8_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_ = (
            None
        )
        view_69 = key_states_34.view(1, -1, 8, 64)
        key_states_34 = None
        key_states_35 = view_69.transpose(1, 2)
        view_69 = None
        view_70 = value_states_34.view(1, -1, 8, 64)
        value_states_34 = None
        value_states_35 = view_70.transpose(1, 2)
        view_70 = None
        transpose_88 = key_states_35.transpose(3, 2)
        scores_34 = torch.matmul(query_states_35, transpose_88)
        query_states_35 = transpose_88 = None
        scores_34 += position_bias_3
        scores_35 = scores_34
        scores_34 = None
        float_19 = scores_35.float()
        softmax_17 = torch.nn.functional.softmax(float_19, dim=-1)
        float_19 = None
        attn_weights_34 = softmax_17.type_as(scores_35)
        softmax_17 = scores_35 = None
        attn_weights_35 = torch.nn.functional.dropout(
            attn_weights_34, p=0.1, training=False
        )
        attn_weights_34 = None
        attn_output_68 = torch.matmul(attn_weights_35, value_states_35)
        attn_weights_35 = None
        transpose_89 = attn_output_68.transpose(1, 2)
        attn_output_68 = None
        attn_output_69 = transpose_89.contiguous()
        transpose_89 = None
        attn_output_70 = attn_output_69.view(1, -1, 512)
        attn_output_69 = None
        attn_output_71 = torch._C._nn.linear(
            attn_output_70,
            l_self_modules_block_modules_8_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_70 = l_self_modules_block_modules_8_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_ = (None)
        dropout_52 = torch.nn.functional.dropout(attn_output_71, 0.1, False, False)
        attn_output_71 = None
        layer_output_8 = hidden_states_74 + dropout_52
        hidden_states_74 = dropout_52 = None
        to_29 = layer_output_8.to(torch.float32)
        pow_27 = to_29.pow(2)
        to_29 = None
        variance_26 = pow_27.mean(-1, keepdim=True)
        pow_27 = None
        add_66 = variance_26 + 1e-06
        variance_26 = None
        rsqrt_26 = torch.rsqrt(add_66)
        add_66 = None
        hidden_states_76 = layer_output_8 * rsqrt_26
        rsqrt_26 = None
        forwarded_states_8 = (
            l_self_modules_block_modules_8_modules_layer_modules_2_modules_layer_norm_parameters_weight_
            * hidden_states_76
        )
        l_self_modules_block_modules_8_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = (
            hidden_states_76
        ) = None
        hidden_states_77 = torch._C._nn.linear(
            forwarded_states_8,
            l_self_modules_block_modules_8_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_8 = l_self_modules_block_modules_8_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_78 = torch.nn.functional.relu(hidden_states_77, inplace=False)
        hidden_states_77 = None
        hidden_states_79 = torch.nn.functional.dropout(
            hidden_states_78, 0.1, False, False
        )
        hidden_states_78 = None
        hidden_states_80 = torch._C._nn.linear(
            hidden_states_79,
            l_self_modules_block_modules_8_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_79 = l_self_modules_block_modules_8_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_54 = torch.nn.functional.dropout(hidden_states_80, 0.1, False, False)
        hidden_states_80 = None
        hidden_states_81 = layer_output_8 + dropout_54
        layer_output_8 = dropout_54 = None
        to_30 = hidden_states_81.to(torch.float32)
        pow_28 = to_30.pow(2)
        to_30 = None
        variance_27 = pow_28.mean(-1, keepdim=True)
        pow_28 = None
        add_68 = variance_27 + 1e-06
        variance_27 = None
        rsqrt_27 = torch.rsqrt(add_68)
        add_68 = None
        hidden_states_82 = hidden_states_81 * rsqrt_27
        rsqrt_27 = None
        normed_hidden_states_18 = (
            l_self_modules_block_modules_9_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_82
        )
        l_self_modules_block_modules_9_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_82
        ) = None
        query_states_36 = torch._C._nn.linear(
            normed_hidden_states_18,
            l_self_modules_block_modules_9_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_9_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_72 = query_states_36.view(1, -1, 8, 64)
        query_states_36 = None
        query_states_37 = view_72.transpose(1, 2)
        view_72 = None
        key_states_36 = torch._C._nn.linear(
            normed_hidden_states_18,
            l_self_modules_block_modules_9_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_9_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_36 = torch._C._nn.linear(
            normed_hidden_states_18,
            l_self_modules_block_modules_9_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_18 = l_self_modules_block_modules_9_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_73 = key_states_36.view(1, -1, 8, 64)
        key_states_36 = None
        key_states_37 = view_73.transpose(1, 2)
        view_73 = None
        view_74 = value_states_36.view(1, -1, 8, 64)
        value_states_36 = None
        value_states_37 = view_74.transpose(1, 2)
        view_74 = None
        transpose_93 = key_states_37.transpose(3, 2)
        scores_36 = torch.matmul(query_states_37, transpose_93)
        query_states_37 = transpose_93 = None
        scores_36 += position_bias_1
        scores_37 = scores_36
        scores_36 = None
        float_20 = scores_37.float()
        softmax_18 = torch.nn.functional.softmax(float_20, dim=-1)
        float_20 = None
        attn_weights_36 = softmax_18.type_as(scores_37)
        softmax_18 = scores_37 = None
        attn_weights_37 = torch.nn.functional.dropout(
            attn_weights_36, p=0.1, training=False
        )
        attn_weights_36 = None
        attn_output_72 = torch.matmul(attn_weights_37, value_states_37)
        attn_weights_37 = None
        transpose_94 = attn_output_72.transpose(1, 2)
        attn_output_72 = None
        attn_output_73 = transpose_94.contiguous()
        transpose_94 = None
        attn_output_74 = attn_output_73.view(1, -1, 512)
        attn_output_73 = None
        attn_output_75 = torch._C._nn.linear(
            attn_output_74,
            l_self_modules_block_modules_9_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_74 = l_self_modules_block_modules_9_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_56 = torch.nn.functional.dropout(attn_output_75, 0.1, False, False)
        attn_output_75 = None
        hidden_states_83 = hidden_states_81 + dropout_56
        hidden_states_81 = dropout_56 = None
        getitem_17 = cache_position[-1]
        add_70 = getitem_17 + 1
        getitem_17 = add_70 = None
        to_31 = hidden_states_83.to(torch.float32)
        pow_29 = to_31.pow(2)
        to_31 = None
        variance_28 = pow_29.mean(-1, keepdim=True)
        pow_29 = None
        add_71 = variance_28 + 1e-06
        variance_28 = None
        rsqrt_28 = torch.rsqrt(add_71)
        add_71 = None
        hidden_states_84 = hidden_states_83 * rsqrt_28
        rsqrt_28 = None
        normed_hidden_states_19 = (
            l_self_modules_block_modules_9_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_84
        )
        l_self_modules_block_modules_9_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_84
        ) = None
        query_states_38 = torch._C._nn.linear(
            normed_hidden_states_19,
            l_self_modules_block_modules_9_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_,
            None,
        )
        normed_hidden_states_19 = l_self_modules_block_modules_9_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_ = (None)
        view_76 = query_states_38.view(1, -1, 8, 64)
        query_states_38 = None
        query_states_39 = view_76.transpose(1, 2)
        view_76 = None
        key_states_38 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_block_modules_9_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_9_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_38 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_block_modules_9_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_9_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_ = (
            None
        )
        view_77 = key_states_38.view(1, -1, 8, 64)
        key_states_38 = None
        key_states_39 = view_77.transpose(1, 2)
        view_77 = None
        view_78 = value_states_38.view(1, -1, 8, 64)
        value_states_38 = None
        value_states_39 = view_78.transpose(1, 2)
        view_78 = None
        transpose_98 = key_states_39.transpose(3, 2)
        scores_38 = torch.matmul(query_states_39, transpose_98)
        query_states_39 = transpose_98 = None
        scores_38 += position_bias_3
        scores_39 = scores_38
        scores_38 = None
        float_21 = scores_39.float()
        softmax_19 = torch.nn.functional.softmax(float_21, dim=-1)
        float_21 = None
        attn_weights_38 = softmax_19.type_as(scores_39)
        softmax_19 = scores_39 = None
        attn_weights_39 = torch.nn.functional.dropout(
            attn_weights_38, p=0.1, training=False
        )
        attn_weights_38 = None
        attn_output_76 = torch.matmul(attn_weights_39, value_states_39)
        attn_weights_39 = None
        transpose_99 = attn_output_76.transpose(1, 2)
        attn_output_76 = None
        attn_output_77 = transpose_99.contiguous()
        transpose_99 = None
        attn_output_78 = attn_output_77.view(1, -1, 512)
        attn_output_77 = None
        attn_output_79 = torch._C._nn.linear(
            attn_output_78,
            l_self_modules_block_modules_9_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_78 = l_self_modules_block_modules_9_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_ = (None)
        dropout_58 = torch.nn.functional.dropout(attn_output_79, 0.1, False, False)
        attn_output_79 = None
        layer_output_9 = hidden_states_83 + dropout_58
        hidden_states_83 = dropout_58 = None
        to_32 = layer_output_9.to(torch.float32)
        pow_30 = to_32.pow(2)
        to_32 = None
        variance_29 = pow_30.mean(-1, keepdim=True)
        pow_30 = None
        add_73 = variance_29 + 1e-06
        variance_29 = None
        rsqrt_29 = torch.rsqrt(add_73)
        add_73 = None
        hidden_states_85 = layer_output_9 * rsqrt_29
        rsqrt_29 = None
        forwarded_states_9 = (
            l_self_modules_block_modules_9_modules_layer_modules_2_modules_layer_norm_parameters_weight_
            * hidden_states_85
        )
        l_self_modules_block_modules_9_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = (
            hidden_states_85
        ) = None
        hidden_states_86 = torch._C._nn.linear(
            forwarded_states_9,
            l_self_modules_block_modules_9_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_9 = l_self_modules_block_modules_9_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_87 = torch.nn.functional.relu(hidden_states_86, inplace=False)
        hidden_states_86 = None
        hidden_states_88 = torch.nn.functional.dropout(
            hidden_states_87, 0.1, False, False
        )
        hidden_states_87 = None
        hidden_states_89 = torch._C._nn.linear(
            hidden_states_88,
            l_self_modules_block_modules_9_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_88 = l_self_modules_block_modules_9_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_60 = torch.nn.functional.dropout(hidden_states_89, 0.1, False, False)
        hidden_states_89 = None
        hidden_states_90 = layer_output_9 + dropout_60
        layer_output_9 = dropout_60 = None
        to_33 = hidden_states_90.to(torch.float32)
        pow_31 = to_33.pow(2)
        to_33 = None
        variance_30 = pow_31.mean(-1, keepdim=True)
        pow_31 = None
        add_75 = variance_30 + 1e-06
        variance_30 = None
        rsqrt_30 = torch.rsqrt(add_75)
        add_75 = None
        hidden_states_91 = hidden_states_90 * rsqrt_30
        rsqrt_30 = None
        normed_hidden_states_20 = (
            l_self_modules_block_modules_10_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_91
        )
        l_self_modules_block_modules_10_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_91
        ) = None
        query_states_40 = torch._C._nn.linear(
            normed_hidden_states_20,
            l_self_modules_block_modules_10_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_10_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_80 = query_states_40.view(1, -1, 8, 64)
        query_states_40 = None
        query_states_41 = view_80.transpose(1, 2)
        view_80 = None
        key_states_40 = torch._C._nn.linear(
            normed_hidden_states_20,
            l_self_modules_block_modules_10_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_10_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_40 = torch._C._nn.linear(
            normed_hidden_states_20,
            l_self_modules_block_modules_10_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_20 = l_self_modules_block_modules_10_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_81 = key_states_40.view(1, -1, 8, 64)
        key_states_40 = None
        key_states_41 = view_81.transpose(1, 2)
        view_81 = None
        view_82 = value_states_40.view(1, -1, 8, 64)
        value_states_40 = None
        value_states_41 = view_82.transpose(1, 2)
        view_82 = None
        transpose_103 = key_states_41.transpose(3, 2)
        scores_40 = torch.matmul(query_states_41, transpose_103)
        query_states_41 = transpose_103 = None
        scores_40 += position_bias_1
        scores_41 = scores_40
        scores_40 = None
        float_22 = scores_41.float()
        softmax_20 = torch.nn.functional.softmax(float_22, dim=-1)
        float_22 = None
        attn_weights_40 = softmax_20.type_as(scores_41)
        softmax_20 = scores_41 = None
        attn_weights_41 = torch.nn.functional.dropout(
            attn_weights_40, p=0.1, training=False
        )
        attn_weights_40 = None
        attn_output_80 = torch.matmul(attn_weights_41, value_states_41)
        attn_weights_41 = None
        transpose_104 = attn_output_80.transpose(1, 2)
        attn_output_80 = None
        attn_output_81 = transpose_104.contiguous()
        transpose_104 = None
        attn_output_82 = attn_output_81.view(1, -1, 512)
        attn_output_81 = None
        attn_output_83 = torch._C._nn.linear(
            attn_output_82,
            l_self_modules_block_modules_10_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_82 = l_self_modules_block_modules_10_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_62 = torch.nn.functional.dropout(attn_output_83, 0.1, False, False)
        attn_output_83 = None
        hidden_states_92 = hidden_states_90 + dropout_62
        hidden_states_90 = dropout_62 = None
        getitem_18 = cache_position[-1]
        add_77 = getitem_18 + 1
        getitem_18 = add_77 = None
        to_34 = hidden_states_92.to(torch.float32)
        pow_32 = to_34.pow(2)
        to_34 = None
        variance_31 = pow_32.mean(-1, keepdim=True)
        pow_32 = None
        add_78 = variance_31 + 1e-06
        variance_31 = None
        rsqrt_31 = torch.rsqrt(add_78)
        add_78 = None
        hidden_states_93 = hidden_states_92 * rsqrt_31
        rsqrt_31 = None
        normed_hidden_states_21 = (
            l_self_modules_block_modules_10_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_93
        )
        l_self_modules_block_modules_10_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_93
        ) = None
        query_states_42 = torch._C._nn.linear(
            normed_hidden_states_21,
            l_self_modules_block_modules_10_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_,
            None,
        )
        normed_hidden_states_21 = l_self_modules_block_modules_10_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_ = (None)
        view_84 = query_states_42.view(1, -1, 8, 64)
        query_states_42 = None
        query_states_43 = view_84.transpose(1, 2)
        view_84 = None
        key_states_42 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_block_modules_10_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_10_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_42 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_block_modules_10_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_10_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_ = (
            None
        )
        view_85 = key_states_42.view(1, -1, 8, 64)
        key_states_42 = None
        key_states_43 = view_85.transpose(1, 2)
        view_85 = None
        view_86 = value_states_42.view(1, -1, 8, 64)
        value_states_42 = None
        value_states_43 = view_86.transpose(1, 2)
        view_86 = None
        transpose_108 = key_states_43.transpose(3, 2)
        scores_42 = torch.matmul(query_states_43, transpose_108)
        query_states_43 = transpose_108 = None
        scores_42 += position_bias_3
        scores_43 = scores_42
        scores_42 = None
        float_23 = scores_43.float()
        softmax_21 = torch.nn.functional.softmax(float_23, dim=-1)
        float_23 = None
        attn_weights_42 = softmax_21.type_as(scores_43)
        softmax_21 = scores_43 = None
        attn_weights_43 = torch.nn.functional.dropout(
            attn_weights_42, p=0.1, training=False
        )
        attn_weights_42 = None
        attn_output_84 = torch.matmul(attn_weights_43, value_states_43)
        attn_weights_43 = None
        transpose_109 = attn_output_84.transpose(1, 2)
        attn_output_84 = None
        attn_output_85 = transpose_109.contiguous()
        transpose_109 = None
        attn_output_86 = attn_output_85.view(1, -1, 512)
        attn_output_85 = None
        attn_output_87 = torch._C._nn.linear(
            attn_output_86,
            l_self_modules_block_modules_10_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_86 = l_self_modules_block_modules_10_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_ = (None)
        dropout_64 = torch.nn.functional.dropout(attn_output_87, 0.1, False, False)
        attn_output_87 = None
        layer_output_10 = hidden_states_92 + dropout_64
        hidden_states_92 = dropout_64 = None
        to_35 = layer_output_10.to(torch.float32)
        pow_33 = to_35.pow(2)
        to_35 = None
        variance_32 = pow_33.mean(-1, keepdim=True)
        pow_33 = None
        add_80 = variance_32 + 1e-06
        variance_32 = None
        rsqrt_32 = torch.rsqrt(add_80)
        add_80 = None
        hidden_states_94 = layer_output_10 * rsqrt_32
        rsqrt_32 = None
        forwarded_states_10 = (
            l_self_modules_block_modules_10_modules_layer_modules_2_modules_layer_norm_parameters_weight_
            * hidden_states_94
        )
        l_self_modules_block_modules_10_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = (
            hidden_states_94
        ) = None
        hidden_states_95 = torch._C._nn.linear(
            forwarded_states_10,
            l_self_modules_block_modules_10_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_10 = l_self_modules_block_modules_10_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_96 = torch.nn.functional.relu(hidden_states_95, inplace=False)
        hidden_states_95 = None
        hidden_states_97 = torch.nn.functional.dropout(
            hidden_states_96, 0.1, False, False
        )
        hidden_states_96 = None
        hidden_states_98 = torch._C._nn.linear(
            hidden_states_97,
            l_self_modules_block_modules_10_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_97 = l_self_modules_block_modules_10_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_66 = torch.nn.functional.dropout(hidden_states_98, 0.1, False, False)
        hidden_states_98 = None
        hidden_states_99 = layer_output_10 + dropout_66
        layer_output_10 = dropout_66 = None
        to_36 = hidden_states_99.to(torch.float32)
        pow_34 = to_36.pow(2)
        to_36 = None
        variance_33 = pow_34.mean(-1, keepdim=True)
        pow_34 = None
        add_82 = variance_33 + 1e-06
        variance_33 = None
        rsqrt_33 = torch.rsqrt(add_82)
        add_82 = None
        hidden_states_100 = hidden_states_99 * rsqrt_33
        rsqrt_33 = None
        normed_hidden_states_22 = (
            l_self_modules_block_modules_11_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_100
        )
        l_self_modules_block_modules_11_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_100
        ) = None
        query_states_44 = torch._C._nn.linear(
            normed_hidden_states_22,
            l_self_modules_block_modules_11_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_11_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_88 = query_states_44.view(1, -1, 8, 64)
        query_states_44 = None
        query_states_45 = view_88.transpose(1, 2)
        view_88 = None
        key_states_44 = torch._C._nn.linear(
            normed_hidden_states_22,
            l_self_modules_block_modules_11_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_11_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_44 = torch._C._nn.linear(
            normed_hidden_states_22,
            l_self_modules_block_modules_11_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_22 = l_self_modules_block_modules_11_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_89 = key_states_44.view(1, -1, 8, 64)
        key_states_44 = None
        key_states_45 = view_89.transpose(1, 2)
        view_89 = None
        view_90 = value_states_44.view(1, -1, 8, 64)
        value_states_44 = None
        value_states_45 = view_90.transpose(1, 2)
        view_90 = None
        transpose_113 = key_states_45.transpose(3, 2)
        scores_44 = torch.matmul(query_states_45, transpose_113)
        query_states_45 = transpose_113 = None
        scores_44 += position_bias_1
        scores_45 = scores_44
        scores_44 = None
        float_24 = scores_45.float()
        softmax_22 = torch.nn.functional.softmax(float_24, dim=-1)
        float_24 = None
        attn_weights_44 = softmax_22.type_as(scores_45)
        softmax_22 = scores_45 = None
        attn_weights_45 = torch.nn.functional.dropout(
            attn_weights_44, p=0.1, training=False
        )
        attn_weights_44 = None
        attn_output_88 = torch.matmul(attn_weights_45, value_states_45)
        attn_weights_45 = None
        transpose_114 = attn_output_88.transpose(1, 2)
        attn_output_88 = None
        attn_output_89 = transpose_114.contiguous()
        transpose_114 = None
        attn_output_90 = attn_output_89.view(1, -1, 512)
        attn_output_89 = None
        attn_output_91 = torch._C._nn.linear(
            attn_output_90,
            l_self_modules_block_modules_11_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_90 = l_self_modules_block_modules_11_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_68 = torch.nn.functional.dropout(attn_output_91, 0.1, False, False)
        attn_output_91 = None
        hidden_states_101 = hidden_states_99 + dropout_68
        hidden_states_99 = dropout_68 = None
        getitem_19 = cache_position[-1]
        add_84 = getitem_19 + 1
        getitem_19 = add_84 = None
        to_37 = hidden_states_101.to(torch.float32)
        pow_35 = to_37.pow(2)
        to_37 = None
        variance_34 = pow_35.mean(-1, keepdim=True)
        pow_35 = None
        add_85 = variance_34 + 1e-06
        variance_34 = None
        rsqrt_34 = torch.rsqrt(add_85)
        add_85 = None
        hidden_states_102 = hidden_states_101 * rsqrt_34
        rsqrt_34 = None
        normed_hidden_states_23 = (
            l_self_modules_block_modules_11_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_102
        )
        l_self_modules_block_modules_11_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_102
        ) = None
        query_states_46 = torch._C._nn.linear(
            normed_hidden_states_23,
            l_self_modules_block_modules_11_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_,
            None,
        )
        normed_hidden_states_23 = l_self_modules_block_modules_11_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_ = (None)
        view_92 = query_states_46.view(1, -1, 8, 64)
        query_states_46 = None
        query_states_47 = view_92.transpose(1, 2)
        view_92 = None
        key_states_46 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_block_modules_11_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_11_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_46 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_block_modules_11_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_11_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_ = (
            None
        )
        view_93 = key_states_46.view(1, -1, 8, 64)
        key_states_46 = None
        key_states_47 = view_93.transpose(1, 2)
        view_93 = None
        view_94 = value_states_46.view(1, -1, 8, 64)
        value_states_46 = None
        value_states_47 = view_94.transpose(1, 2)
        view_94 = None
        transpose_118 = key_states_47.transpose(3, 2)
        scores_46 = torch.matmul(query_states_47, transpose_118)
        query_states_47 = transpose_118 = None
        scores_46 += position_bias_3
        scores_47 = scores_46
        scores_46 = None
        float_25 = scores_47.float()
        softmax_23 = torch.nn.functional.softmax(float_25, dim=-1)
        float_25 = None
        attn_weights_46 = softmax_23.type_as(scores_47)
        softmax_23 = scores_47 = None
        attn_weights_47 = torch.nn.functional.dropout(
            attn_weights_46, p=0.1, training=False
        )
        attn_weights_46 = None
        attn_output_92 = torch.matmul(attn_weights_47, value_states_47)
        attn_weights_47 = None
        transpose_119 = attn_output_92.transpose(1, 2)
        attn_output_92 = None
        attn_output_93 = transpose_119.contiguous()
        transpose_119 = None
        attn_output_94 = attn_output_93.view(1, -1, 512)
        attn_output_93 = None
        attn_output_95 = torch._C._nn.linear(
            attn_output_94,
            l_self_modules_block_modules_11_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_94 = l_self_modules_block_modules_11_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_ = (None)
        dropout_70 = torch.nn.functional.dropout(attn_output_95, 0.1, False, False)
        attn_output_95 = None
        layer_output_11 = hidden_states_101 + dropout_70
        hidden_states_101 = dropout_70 = None
        to_38 = layer_output_11.to(torch.float32)
        pow_36 = to_38.pow(2)
        to_38 = None
        variance_35 = pow_36.mean(-1, keepdim=True)
        pow_36 = None
        add_87 = variance_35 + 1e-06
        variance_35 = None
        rsqrt_35 = torch.rsqrt(add_87)
        add_87 = None
        hidden_states_103 = layer_output_11 * rsqrt_35
        rsqrt_35 = None
        forwarded_states_11 = (
            l_self_modules_block_modules_11_modules_layer_modules_2_modules_layer_norm_parameters_weight_
            * hidden_states_103
        )
        l_self_modules_block_modules_11_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = (
            hidden_states_103
        ) = None
        hidden_states_104 = torch._C._nn.linear(
            forwarded_states_11,
            l_self_modules_block_modules_11_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_11 = l_self_modules_block_modules_11_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_105 = torch.nn.functional.relu(hidden_states_104, inplace=False)
        hidden_states_104 = None
        hidden_states_106 = torch.nn.functional.dropout(
            hidden_states_105, 0.1, False, False
        )
        hidden_states_105 = None
        hidden_states_107 = torch._C._nn.linear(
            hidden_states_106,
            l_self_modules_block_modules_11_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_106 = l_self_modules_block_modules_11_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_72 = torch.nn.functional.dropout(hidden_states_107, 0.1, False, False)
        hidden_states_107 = None
        hidden_states_108 = layer_output_11 + dropout_72
        layer_output_11 = dropout_72 = None
        to_39 = hidden_states_108.to(torch.float32)
        pow_37 = to_39.pow(2)
        to_39 = None
        variance_36 = pow_37.mean(-1, keepdim=True)
        pow_37 = None
        add_89 = variance_36 + 1e-06
        variance_36 = None
        rsqrt_36 = torch.rsqrt(add_89)
        add_89 = None
        hidden_states_109 = hidden_states_108 * rsqrt_36
        rsqrt_36 = None
        normed_hidden_states_24 = (
            l_self_modules_block_modules_12_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_109
        )
        l_self_modules_block_modules_12_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_109
        ) = None
        query_states_48 = torch._C._nn.linear(
            normed_hidden_states_24,
            l_self_modules_block_modules_12_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_12_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_96 = query_states_48.view(1, -1, 8, 64)
        query_states_48 = None
        query_states_49 = view_96.transpose(1, 2)
        view_96 = None
        key_states_48 = torch._C._nn.linear(
            normed_hidden_states_24,
            l_self_modules_block_modules_12_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_12_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_48 = torch._C._nn.linear(
            normed_hidden_states_24,
            l_self_modules_block_modules_12_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_24 = l_self_modules_block_modules_12_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_97 = key_states_48.view(1, -1, 8, 64)
        key_states_48 = None
        key_states_49 = view_97.transpose(1, 2)
        view_97 = None
        view_98 = value_states_48.view(1, -1, 8, 64)
        value_states_48 = None
        value_states_49 = view_98.transpose(1, 2)
        view_98 = None
        transpose_123 = key_states_49.transpose(3, 2)
        scores_48 = torch.matmul(query_states_49, transpose_123)
        query_states_49 = transpose_123 = None
        scores_48 += position_bias_1
        scores_49 = scores_48
        scores_48 = None
        float_26 = scores_49.float()
        softmax_24 = torch.nn.functional.softmax(float_26, dim=-1)
        float_26 = None
        attn_weights_48 = softmax_24.type_as(scores_49)
        softmax_24 = scores_49 = None
        attn_weights_49 = torch.nn.functional.dropout(
            attn_weights_48, p=0.1, training=False
        )
        attn_weights_48 = None
        attn_output_96 = torch.matmul(attn_weights_49, value_states_49)
        attn_weights_49 = None
        transpose_124 = attn_output_96.transpose(1, 2)
        attn_output_96 = None
        attn_output_97 = transpose_124.contiguous()
        transpose_124 = None
        attn_output_98 = attn_output_97.view(1, -1, 512)
        attn_output_97 = None
        attn_output_99 = torch._C._nn.linear(
            attn_output_98,
            l_self_modules_block_modules_12_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_98 = l_self_modules_block_modules_12_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_74 = torch.nn.functional.dropout(attn_output_99, 0.1, False, False)
        attn_output_99 = None
        hidden_states_110 = hidden_states_108 + dropout_74
        hidden_states_108 = dropout_74 = None
        getitem_20 = cache_position[-1]
        add_91 = getitem_20 + 1
        getitem_20 = add_91 = None
        to_40 = hidden_states_110.to(torch.float32)
        pow_38 = to_40.pow(2)
        to_40 = None
        variance_37 = pow_38.mean(-1, keepdim=True)
        pow_38 = None
        add_92 = variance_37 + 1e-06
        variance_37 = None
        rsqrt_37 = torch.rsqrt(add_92)
        add_92 = None
        hidden_states_111 = hidden_states_110 * rsqrt_37
        rsqrt_37 = None
        normed_hidden_states_25 = (
            l_self_modules_block_modules_12_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_111
        )
        l_self_modules_block_modules_12_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_111
        ) = None
        query_states_50 = torch._C._nn.linear(
            normed_hidden_states_25,
            l_self_modules_block_modules_12_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_,
            None,
        )
        normed_hidden_states_25 = l_self_modules_block_modules_12_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_ = (None)
        view_100 = query_states_50.view(1, -1, 8, 64)
        query_states_50 = None
        query_states_51 = view_100.transpose(1, 2)
        view_100 = None
        key_states_50 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_block_modules_12_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_12_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_50 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_block_modules_12_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_12_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_ = (
            None
        )
        view_101 = key_states_50.view(1, -1, 8, 64)
        key_states_50 = None
        key_states_51 = view_101.transpose(1, 2)
        view_101 = None
        view_102 = value_states_50.view(1, -1, 8, 64)
        value_states_50 = None
        value_states_51 = view_102.transpose(1, 2)
        view_102 = None
        transpose_128 = key_states_51.transpose(3, 2)
        scores_50 = torch.matmul(query_states_51, transpose_128)
        query_states_51 = transpose_128 = None
        scores_50 += position_bias_3
        scores_51 = scores_50
        scores_50 = None
        float_27 = scores_51.float()
        softmax_25 = torch.nn.functional.softmax(float_27, dim=-1)
        float_27 = None
        attn_weights_50 = softmax_25.type_as(scores_51)
        softmax_25 = scores_51 = None
        attn_weights_51 = torch.nn.functional.dropout(
            attn_weights_50, p=0.1, training=False
        )
        attn_weights_50 = None
        attn_output_100 = torch.matmul(attn_weights_51, value_states_51)
        attn_weights_51 = None
        transpose_129 = attn_output_100.transpose(1, 2)
        attn_output_100 = None
        attn_output_101 = transpose_129.contiguous()
        transpose_129 = None
        attn_output_102 = attn_output_101.view(1, -1, 512)
        attn_output_101 = None
        attn_output_103 = torch._C._nn.linear(
            attn_output_102,
            l_self_modules_block_modules_12_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_102 = l_self_modules_block_modules_12_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_ = (None)
        dropout_76 = torch.nn.functional.dropout(attn_output_103, 0.1, False, False)
        attn_output_103 = None
        layer_output_12 = hidden_states_110 + dropout_76
        hidden_states_110 = dropout_76 = None
        to_41 = layer_output_12.to(torch.float32)
        pow_39 = to_41.pow(2)
        to_41 = None
        variance_38 = pow_39.mean(-1, keepdim=True)
        pow_39 = None
        add_94 = variance_38 + 1e-06
        variance_38 = None
        rsqrt_38 = torch.rsqrt(add_94)
        add_94 = None
        hidden_states_112 = layer_output_12 * rsqrt_38
        rsqrt_38 = None
        forwarded_states_12 = (
            l_self_modules_block_modules_12_modules_layer_modules_2_modules_layer_norm_parameters_weight_
            * hidden_states_112
        )
        l_self_modules_block_modules_12_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = (
            hidden_states_112
        ) = None
        hidden_states_113 = torch._C._nn.linear(
            forwarded_states_12,
            l_self_modules_block_modules_12_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_12 = l_self_modules_block_modules_12_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_114 = torch.nn.functional.relu(hidden_states_113, inplace=False)
        hidden_states_113 = None
        hidden_states_115 = torch.nn.functional.dropout(
            hidden_states_114, 0.1, False, False
        )
        hidden_states_114 = None
        hidden_states_116 = torch._C._nn.linear(
            hidden_states_115,
            l_self_modules_block_modules_12_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_115 = l_self_modules_block_modules_12_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_78 = torch.nn.functional.dropout(hidden_states_116, 0.1, False, False)
        hidden_states_116 = None
        hidden_states_117 = layer_output_12 + dropout_78
        layer_output_12 = dropout_78 = None
        to_42 = hidden_states_117.to(torch.float32)
        pow_40 = to_42.pow(2)
        to_42 = None
        variance_39 = pow_40.mean(-1, keepdim=True)
        pow_40 = None
        add_96 = variance_39 + 1e-06
        variance_39 = None
        rsqrt_39 = torch.rsqrt(add_96)
        add_96 = None
        hidden_states_118 = hidden_states_117 * rsqrt_39
        rsqrt_39 = None
        normed_hidden_states_26 = (
            l_self_modules_block_modules_13_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_118
        )
        l_self_modules_block_modules_13_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_118
        ) = None
        query_states_52 = torch._C._nn.linear(
            normed_hidden_states_26,
            l_self_modules_block_modules_13_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_13_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_104 = query_states_52.view(1, -1, 8, 64)
        query_states_52 = None
        query_states_53 = view_104.transpose(1, 2)
        view_104 = None
        key_states_52 = torch._C._nn.linear(
            normed_hidden_states_26,
            l_self_modules_block_modules_13_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_13_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_52 = torch._C._nn.linear(
            normed_hidden_states_26,
            l_self_modules_block_modules_13_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_26 = l_self_modules_block_modules_13_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_105 = key_states_52.view(1, -1, 8, 64)
        key_states_52 = None
        key_states_53 = view_105.transpose(1, 2)
        view_105 = None
        view_106 = value_states_52.view(1, -1, 8, 64)
        value_states_52 = None
        value_states_53 = view_106.transpose(1, 2)
        view_106 = None
        transpose_133 = key_states_53.transpose(3, 2)
        scores_52 = torch.matmul(query_states_53, transpose_133)
        query_states_53 = transpose_133 = None
        scores_52 += position_bias_1
        scores_53 = scores_52
        scores_52 = None
        float_28 = scores_53.float()
        softmax_26 = torch.nn.functional.softmax(float_28, dim=-1)
        float_28 = None
        attn_weights_52 = softmax_26.type_as(scores_53)
        softmax_26 = scores_53 = None
        attn_weights_53 = torch.nn.functional.dropout(
            attn_weights_52, p=0.1, training=False
        )
        attn_weights_52 = None
        attn_output_104 = torch.matmul(attn_weights_53, value_states_53)
        attn_weights_53 = None
        transpose_134 = attn_output_104.transpose(1, 2)
        attn_output_104 = None
        attn_output_105 = transpose_134.contiguous()
        transpose_134 = None
        attn_output_106 = attn_output_105.view(1, -1, 512)
        attn_output_105 = None
        attn_output_107 = torch._C._nn.linear(
            attn_output_106,
            l_self_modules_block_modules_13_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_106 = l_self_modules_block_modules_13_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_80 = torch.nn.functional.dropout(attn_output_107, 0.1, False, False)
        attn_output_107 = None
        hidden_states_119 = hidden_states_117 + dropout_80
        hidden_states_117 = dropout_80 = None
        getitem_21 = cache_position[-1]
        add_98 = getitem_21 + 1
        getitem_21 = add_98 = None
        to_43 = hidden_states_119.to(torch.float32)
        pow_41 = to_43.pow(2)
        to_43 = None
        variance_40 = pow_41.mean(-1, keepdim=True)
        pow_41 = None
        add_99 = variance_40 + 1e-06
        variance_40 = None
        rsqrt_40 = torch.rsqrt(add_99)
        add_99 = None
        hidden_states_120 = hidden_states_119 * rsqrt_40
        rsqrt_40 = None
        normed_hidden_states_27 = (
            l_self_modules_block_modules_13_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_120
        )
        l_self_modules_block_modules_13_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_120
        ) = None
        query_states_54 = torch._C._nn.linear(
            normed_hidden_states_27,
            l_self_modules_block_modules_13_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_,
            None,
        )
        normed_hidden_states_27 = l_self_modules_block_modules_13_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_ = (None)
        view_108 = query_states_54.view(1, -1, 8, 64)
        query_states_54 = None
        query_states_55 = view_108.transpose(1, 2)
        view_108 = None
        key_states_54 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_block_modules_13_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_13_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_54 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_block_modules_13_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_13_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_ = (
            None
        )
        view_109 = key_states_54.view(1, -1, 8, 64)
        key_states_54 = None
        key_states_55 = view_109.transpose(1, 2)
        view_109 = None
        view_110 = value_states_54.view(1, -1, 8, 64)
        value_states_54 = None
        value_states_55 = view_110.transpose(1, 2)
        view_110 = None
        transpose_138 = key_states_55.transpose(3, 2)
        scores_54 = torch.matmul(query_states_55, transpose_138)
        query_states_55 = transpose_138 = None
        scores_54 += position_bias_3
        scores_55 = scores_54
        scores_54 = None
        float_29 = scores_55.float()
        softmax_27 = torch.nn.functional.softmax(float_29, dim=-1)
        float_29 = None
        attn_weights_54 = softmax_27.type_as(scores_55)
        softmax_27 = scores_55 = None
        attn_weights_55 = torch.nn.functional.dropout(
            attn_weights_54, p=0.1, training=False
        )
        attn_weights_54 = None
        attn_output_108 = torch.matmul(attn_weights_55, value_states_55)
        attn_weights_55 = None
        transpose_139 = attn_output_108.transpose(1, 2)
        attn_output_108 = None
        attn_output_109 = transpose_139.contiguous()
        transpose_139 = None
        attn_output_110 = attn_output_109.view(1, -1, 512)
        attn_output_109 = None
        attn_output_111 = torch._C._nn.linear(
            attn_output_110,
            l_self_modules_block_modules_13_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_110 = l_self_modules_block_modules_13_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_ = (None)
        dropout_82 = torch.nn.functional.dropout(attn_output_111, 0.1, False, False)
        attn_output_111 = None
        layer_output_13 = hidden_states_119 + dropout_82
        hidden_states_119 = dropout_82 = None
        to_44 = layer_output_13.to(torch.float32)
        pow_42 = to_44.pow(2)
        to_44 = None
        variance_41 = pow_42.mean(-1, keepdim=True)
        pow_42 = None
        add_101 = variance_41 + 1e-06
        variance_41 = None
        rsqrt_41 = torch.rsqrt(add_101)
        add_101 = None
        hidden_states_121 = layer_output_13 * rsqrt_41
        rsqrt_41 = None
        forwarded_states_13 = (
            l_self_modules_block_modules_13_modules_layer_modules_2_modules_layer_norm_parameters_weight_
            * hidden_states_121
        )
        l_self_modules_block_modules_13_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = (
            hidden_states_121
        ) = None
        hidden_states_122 = torch._C._nn.linear(
            forwarded_states_13,
            l_self_modules_block_modules_13_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_13 = l_self_modules_block_modules_13_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_123 = torch.nn.functional.relu(hidden_states_122, inplace=False)
        hidden_states_122 = None
        hidden_states_124 = torch.nn.functional.dropout(
            hidden_states_123, 0.1, False, False
        )
        hidden_states_123 = None
        hidden_states_125 = torch._C._nn.linear(
            hidden_states_124,
            l_self_modules_block_modules_13_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_124 = l_self_modules_block_modules_13_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_84 = torch.nn.functional.dropout(hidden_states_125, 0.1, False, False)
        hidden_states_125 = None
        hidden_states_126 = layer_output_13 + dropout_84
        layer_output_13 = dropout_84 = None
        to_45 = hidden_states_126.to(torch.float32)
        pow_43 = to_45.pow(2)
        to_45 = None
        variance_42 = pow_43.mean(-1, keepdim=True)
        pow_43 = None
        add_103 = variance_42 + 1e-06
        variance_42 = None
        rsqrt_42 = torch.rsqrt(add_103)
        add_103 = None
        hidden_states_127 = hidden_states_126 * rsqrt_42
        rsqrt_42 = None
        normed_hidden_states_28 = (
            l_self_modules_block_modules_14_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_127
        )
        l_self_modules_block_modules_14_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_127
        ) = None
        query_states_56 = torch._C._nn.linear(
            normed_hidden_states_28,
            l_self_modules_block_modules_14_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_14_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_112 = query_states_56.view(1, -1, 8, 64)
        query_states_56 = None
        query_states_57 = view_112.transpose(1, 2)
        view_112 = None
        key_states_56 = torch._C._nn.linear(
            normed_hidden_states_28,
            l_self_modules_block_modules_14_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_14_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_56 = torch._C._nn.linear(
            normed_hidden_states_28,
            l_self_modules_block_modules_14_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_28 = l_self_modules_block_modules_14_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_113 = key_states_56.view(1, -1, 8, 64)
        key_states_56 = None
        key_states_57 = view_113.transpose(1, 2)
        view_113 = None
        view_114 = value_states_56.view(1, -1, 8, 64)
        value_states_56 = None
        value_states_57 = view_114.transpose(1, 2)
        view_114 = None
        transpose_143 = key_states_57.transpose(3, 2)
        scores_56 = torch.matmul(query_states_57, transpose_143)
        query_states_57 = transpose_143 = None
        scores_56 += position_bias_1
        scores_57 = scores_56
        scores_56 = None
        float_30 = scores_57.float()
        softmax_28 = torch.nn.functional.softmax(float_30, dim=-1)
        float_30 = None
        attn_weights_56 = softmax_28.type_as(scores_57)
        softmax_28 = scores_57 = None
        attn_weights_57 = torch.nn.functional.dropout(
            attn_weights_56, p=0.1, training=False
        )
        attn_weights_56 = None
        attn_output_112 = torch.matmul(attn_weights_57, value_states_57)
        attn_weights_57 = None
        transpose_144 = attn_output_112.transpose(1, 2)
        attn_output_112 = None
        attn_output_113 = transpose_144.contiguous()
        transpose_144 = None
        attn_output_114 = attn_output_113.view(1, -1, 512)
        attn_output_113 = None
        attn_output_115 = torch._C._nn.linear(
            attn_output_114,
            l_self_modules_block_modules_14_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_114 = l_self_modules_block_modules_14_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_86 = torch.nn.functional.dropout(attn_output_115, 0.1, False, False)
        attn_output_115 = None
        hidden_states_128 = hidden_states_126 + dropout_86
        hidden_states_126 = dropout_86 = None
        getitem_22 = cache_position[-1]
        add_105 = getitem_22 + 1
        getitem_22 = add_105 = None
        to_46 = hidden_states_128.to(torch.float32)
        pow_44 = to_46.pow(2)
        to_46 = None
        variance_43 = pow_44.mean(-1, keepdim=True)
        pow_44 = None
        add_106 = variance_43 + 1e-06
        variance_43 = None
        rsqrt_43 = torch.rsqrt(add_106)
        add_106 = None
        hidden_states_129 = hidden_states_128 * rsqrt_43
        rsqrt_43 = None
        normed_hidden_states_29 = (
            l_self_modules_block_modules_14_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_129
        )
        l_self_modules_block_modules_14_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_129
        ) = None
        query_states_58 = torch._C._nn.linear(
            normed_hidden_states_29,
            l_self_modules_block_modules_14_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_,
            None,
        )
        normed_hidden_states_29 = l_self_modules_block_modules_14_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_ = (None)
        view_116 = query_states_58.view(1, -1, 8, 64)
        query_states_58 = None
        query_states_59 = view_116.transpose(1, 2)
        view_116 = None
        key_states_58 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_block_modules_14_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_14_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_58 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_block_modules_14_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_14_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_ = (
            None
        )
        view_117 = key_states_58.view(1, -1, 8, 64)
        key_states_58 = None
        key_states_59 = view_117.transpose(1, 2)
        view_117 = None
        view_118 = value_states_58.view(1, -1, 8, 64)
        value_states_58 = None
        value_states_59 = view_118.transpose(1, 2)
        view_118 = None
        transpose_148 = key_states_59.transpose(3, 2)
        scores_58 = torch.matmul(query_states_59, transpose_148)
        query_states_59 = transpose_148 = None
        scores_58 += position_bias_3
        scores_59 = scores_58
        scores_58 = None
        float_31 = scores_59.float()
        softmax_29 = torch.nn.functional.softmax(float_31, dim=-1)
        float_31 = None
        attn_weights_58 = softmax_29.type_as(scores_59)
        softmax_29 = scores_59 = None
        attn_weights_59 = torch.nn.functional.dropout(
            attn_weights_58, p=0.1, training=False
        )
        attn_weights_58 = None
        attn_output_116 = torch.matmul(attn_weights_59, value_states_59)
        attn_weights_59 = None
        transpose_149 = attn_output_116.transpose(1, 2)
        attn_output_116 = None
        attn_output_117 = transpose_149.contiguous()
        transpose_149 = None
        attn_output_118 = attn_output_117.view(1, -1, 512)
        attn_output_117 = None
        attn_output_119 = torch._C._nn.linear(
            attn_output_118,
            l_self_modules_block_modules_14_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_118 = l_self_modules_block_modules_14_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_ = (None)
        dropout_88 = torch.nn.functional.dropout(attn_output_119, 0.1, False, False)
        attn_output_119 = None
        layer_output_14 = hidden_states_128 + dropout_88
        hidden_states_128 = dropout_88 = None
        to_47 = layer_output_14.to(torch.float32)
        pow_45 = to_47.pow(2)
        to_47 = None
        variance_44 = pow_45.mean(-1, keepdim=True)
        pow_45 = None
        add_108 = variance_44 + 1e-06
        variance_44 = None
        rsqrt_44 = torch.rsqrt(add_108)
        add_108 = None
        hidden_states_130 = layer_output_14 * rsqrt_44
        rsqrt_44 = None
        forwarded_states_14 = (
            l_self_modules_block_modules_14_modules_layer_modules_2_modules_layer_norm_parameters_weight_
            * hidden_states_130
        )
        l_self_modules_block_modules_14_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = (
            hidden_states_130
        ) = None
        hidden_states_131 = torch._C._nn.linear(
            forwarded_states_14,
            l_self_modules_block_modules_14_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_14 = l_self_modules_block_modules_14_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_132 = torch.nn.functional.relu(hidden_states_131, inplace=False)
        hidden_states_131 = None
        hidden_states_133 = torch.nn.functional.dropout(
            hidden_states_132, 0.1, False, False
        )
        hidden_states_132 = None
        hidden_states_134 = torch._C._nn.linear(
            hidden_states_133,
            l_self_modules_block_modules_14_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_133 = l_self_modules_block_modules_14_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_90 = torch.nn.functional.dropout(hidden_states_134, 0.1, False, False)
        hidden_states_134 = None
        hidden_states_135 = layer_output_14 + dropout_90
        layer_output_14 = dropout_90 = None
        to_48 = hidden_states_135.to(torch.float32)
        pow_46 = to_48.pow(2)
        to_48 = None
        variance_45 = pow_46.mean(-1, keepdim=True)
        pow_46 = None
        add_110 = variance_45 + 1e-06
        variance_45 = None
        rsqrt_45 = torch.rsqrt(add_110)
        add_110 = None
        hidden_states_136 = hidden_states_135 * rsqrt_45
        rsqrt_45 = None
        normed_hidden_states_30 = (
            l_self_modules_block_modules_15_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_136
        )
        l_self_modules_block_modules_15_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_136
        ) = None
        query_states_60 = torch._C._nn.linear(
            normed_hidden_states_30,
            l_self_modules_block_modules_15_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_15_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_120 = query_states_60.view(1, -1, 8, 64)
        query_states_60 = None
        query_states_61 = view_120.transpose(1, 2)
        view_120 = None
        key_states_60 = torch._C._nn.linear(
            normed_hidden_states_30,
            l_self_modules_block_modules_15_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_15_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_60 = torch._C._nn.linear(
            normed_hidden_states_30,
            l_self_modules_block_modules_15_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_30 = l_self_modules_block_modules_15_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_121 = key_states_60.view(1, -1, 8, 64)
        key_states_60 = None
        key_states_61 = view_121.transpose(1, 2)
        view_121 = None
        view_122 = value_states_60.view(1, -1, 8, 64)
        value_states_60 = None
        value_states_61 = view_122.transpose(1, 2)
        view_122 = None
        transpose_153 = key_states_61.transpose(3, 2)
        scores_60 = torch.matmul(query_states_61, transpose_153)
        query_states_61 = transpose_153 = None
        scores_60 += position_bias_1
        scores_61 = scores_60
        scores_60 = None
        float_32 = scores_61.float()
        softmax_30 = torch.nn.functional.softmax(float_32, dim=-1)
        float_32 = None
        attn_weights_60 = softmax_30.type_as(scores_61)
        softmax_30 = scores_61 = None
        attn_weights_61 = torch.nn.functional.dropout(
            attn_weights_60, p=0.1, training=False
        )
        attn_weights_60 = None
        attn_output_120 = torch.matmul(attn_weights_61, value_states_61)
        attn_weights_61 = None
        transpose_154 = attn_output_120.transpose(1, 2)
        attn_output_120 = None
        attn_output_121 = transpose_154.contiguous()
        transpose_154 = None
        attn_output_122 = attn_output_121.view(1, -1, 512)
        attn_output_121 = None
        attn_output_123 = torch._C._nn.linear(
            attn_output_122,
            l_self_modules_block_modules_15_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_122 = l_self_modules_block_modules_15_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_92 = torch.nn.functional.dropout(attn_output_123, 0.1, False, False)
        attn_output_123 = None
        hidden_states_137 = hidden_states_135 + dropout_92
        hidden_states_135 = dropout_92 = None
        getitem_23 = cache_position[-1]
        add_112 = getitem_23 + 1
        getitem_23 = add_112 = None
        to_49 = hidden_states_137.to(torch.float32)
        pow_47 = to_49.pow(2)
        to_49 = None
        variance_46 = pow_47.mean(-1, keepdim=True)
        pow_47 = None
        add_113 = variance_46 + 1e-06
        variance_46 = None
        rsqrt_46 = torch.rsqrt(add_113)
        add_113 = None
        hidden_states_138 = hidden_states_137 * rsqrt_46
        rsqrt_46 = None
        normed_hidden_states_31 = (
            l_self_modules_block_modules_15_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_138
        )
        l_self_modules_block_modules_15_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_138
        ) = None
        query_states_62 = torch._C._nn.linear(
            normed_hidden_states_31,
            l_self_modules_block_modules_15_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_,
            None,
        )
        normed_hidden_states_31 = l_self_modules_block_modules_15_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_ = (None)
        view_124 = query_states_62.view(1, -1, 8, 64)
        query_states_62 = None
        query_states_63 = view_124.transpose(1, 2)
        view_124 = None
        key_states_62 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_block_modules_15_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_15_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_62 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_block_modules_15_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_15_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_ = (
            None
        )
        view_125 = key_states_62.view(1, -1, 8, 64)
        key_states_62 = None
        key_states_63 = view_125.transpose(1, 2)
        view_125 = None
        view_126 = value_states_62.view(1, -1, 8, 64)
        value_states_62 = None
        value_states_63 = view_126.transpose(1, 2)
        view_126 = None
        transpose_158 = key_states_63.transpose(3, 2)
        scores_62 = torch.matmul(query_states_63, transpose_158)
        query_states_63 = transpose_158 = None
        scores_62 += position_bias_3
        scores_63 = scores_62
        scores_62 = None
        float_33 = scores_63.float()
        softmax_31 = torch.nn.functional.softmax(float_33, dim=-1)
        float_33 = None
        attn_weights_62 = softmax_31.type_as(scores_63)
        softmax_31 = scores_63 = None
        attn_weights_63 = torch.nn.functional.dropout(
            attn_weights_62, p=0.1, training=False
        )
        attn_weights_62 = None
        attn_output_124 = torch.matmul(attn_weights_63, value_states_63)
        attn_weights_63 = None
        transpose_159 = attn_output_124.transpose(1, 2)
        attn_output_124 = None
        attn_output_125 = transpose_159.contiguous()
        transpose_159 = None
        attn_output_126 = attn_output_125.view(1, -1, 512)
        attn_output_125 = None
        attn_output_127 = torch._C._nn.linear(
            attn_output_126,
            l_self_modules_block_modules_15_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_126 = l_self_modules_block_modules_15_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_ = (None)
        dropout_94 = torch.nn.functional.dropout(attn_output_127, 0.1, False, False)
        attn_output_127 = None
        layer_output_15 = hidden_states_137 + dropout_94
        hidden_states_137 = dropout_94 = None
        to_50 = layer_output_15.to(torch.float32)
        pow_48 = to_50.pow(2)
        to_50 = None
        variance_47 = pow_48.mean(-1, keepdim=True)
        pow_48 = None
        add_115 = variance_47 + 1e-06
        variance_47 = None
        rsqrt_47 = torch.rsqrt(add_115)
        add_115 = None
        hidden_states_139 = layer_output_15 * rsqrt_47
        rsqrt_47 = None
        forwarded_states_15 = (
            l_self_modules_block_modules_15_modules_layer_modules_2_modules_layer_norm_parameters_weight_
            * hidden_states_139
        )
        l_self_modules_block_modules_15_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = (
            hidden_states_139
        ) = None
        hidden_states_140 = torch._C._nn.linear(
            forwarded_states_15,
            l_self_modules_block_modules_15_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_15 = l_self_modules_block_modules_15_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_141 = torch.nn.functional.relu(hidden_states_140, inplace=False)
        hidden_states_140 = None
        hidden_states_142 = torch.nn.functional.dropout(
            hidden_states_141, 0.1, False, False
        )
        hidden_states_141 = None
        hidden_states_143 = torch._C._nn.linear(
            hidden_states_142,
            l_self_modules_block_modules_15_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_142 = l_self_modules_block_modules_15_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_96 = torch.nn.functional.dropout(hidden_states_143, 0.1, False, False)
        hidden_states_143 = None
        hidden_states_144 = layer_output_15 + dropout_96
        layer_output_15 = dropout_96 = None
        to_51 = hidden_states_144.to(torch.float32)
        pow_49 = to_51.pow(2)
        to_51 = None
        variance_48 = pow_49.mean(-1, keepdim=True)
        pow_49 = None
        add_117 = variance_48 + 1e-06
        variance_48 = None
        rsqrt_48 = torch.rsqrt(add_117)
        add_117 = None
        hidden_states_145 = hidden_states_144 * rsqrt_48
        rsqrt_48 = None
        normed_hidden_states_32 = (
            l_self_modules_block_modules_16_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_145
        )
        l_self_modules_block_modules_16_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_145
        ) = None
        query_states_64 = torch._C._nn.linear(
            normed_hidden_states_32,
            l_self_modules_block_modules_16_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_16_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_128 = query_states_64.view(1, -1, 8, 64)
        query_states_64 = None
        query_states_65 = view_128.transpose(1, 2)
        view_128 = None
        key_states_64 = torch._C._nn.linear(
            normed_hidden_states_32,
            l_self_modules_block_modules_16_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_16_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_64 = torch._C._nn.linear(
            normed_hidden_states_32,
            l_self_modules_block_modules_16_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_32 = l_self_modules_block_modules_16_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_129 = key_states_64.view(1, -1, 8, 64)
        key_states_64 = None
        key_states_65 = view_129.transpose(1, 2)
        view_129 = None
        view_130 = value_states_64.view(1, -1, 8, 64)
        value_states_64 = None
        value_states_65 = view_130.transpose(1, 2)
        view_130 = None
        transpose_163 = key_states_65.transpose(3, 2)
        scores_64 = torch.matmul(query_states_65, transpose_163)
        query_states_65 = transpose_163 = None
        scores_64 += position_bias_1
        scores_65 = scores_64
        scores_64 = None
        float_34 = scores_65.float()
        softmax_32 = torch.nn.functional.softmax(float_34, dim=-1)
        float_34 = None
        attn_weights_64 = softmax_32.type_as(scores_65)
        softmax_32 = scores_65 = None
        attn_weights_65 = torch.nn.functional.dropout(
            attn_weights_64, p=0.1, training=False
        )
        attn_weights_64 = None
        attn_output_128 = torch.matmul(attn_weights_65, value_states_65)
        attn_weights_65 = None
        transpose_164 = attn_output_128.transpose(1, 2)
        attn_output_128 = None
        attn_output_129 = transpose_164.contiguous()
        transpose_164 = None
        attn_output_130 = attn_output_129.view(1, -1, 512)
        attn_output_129 = None
        attn_output_131 = torch._C._nn.linear(
            attn_output_130,
            l_self_modules_block_modules_16_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_130 = l_self_modules_block_modules_16_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_98 = torch.nn.functional.dropout(attn_output_131, 0.1, False, False)
        attn_output_131 = None
        hidden_states_146 = hidden_states_144 + dropout_98
        hidden_states_144 = dropout_98 = None
        getitem_24 = cache_position[-1]
        add_119 = getitem_24 + 1
        getitem_24 = add_119 = None
        to_52 = hidden_states_146.to(torch.float32)
        pow_50 = to_52.pow(2)
        to_52 = None
        variance_49 = pow_50.mean(-1, keepdim=True)
        pow_50 = None
        add_120 = variance_49 + 1e-06
        variance_49 = None
        rsqrt_49 = torch.rsqrt(add_120)
        add_120 = None
        hidden_states_147 = hidden_states_146 * rsqrt_49
        rsqrt_49 = None
        normed_hidden_states_33 = (
            l_self_modules_block_modules_16_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_147
        )
        l_self_modules_block_modules_16_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_147
        ) = None
        query_states_66 = torch._C._nn.linear(
            normed_hidden_states_33,
            l_self_modules_block_modules_16_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_,
            None,
        )
        normed_hidden_states_33 = l_self_modules_block_modules_16_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_ = (None)
        view_132 = query_states_66.view(1, -1, 8, 64)
        query_states_66 = None
        query_states_67 = view_132.transpose(1, 2)
        view_132 = None
        key_states_66 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_block_modules_16_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_16_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_66 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_block_modules_16_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_16_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_ = (
            None
        )
        view_133 = key_states_66.view(1, -1, 8, 64)
        key_states_66 = None
        key_states_67 = view_133.transpose(1, 2)
        view_133 = None
        view_134 = value_states_66.view(1, -1, 8, 64)
        value_states_66 = None
        value_states_67 = view_134.transpose(1, 2)
        view_134 = None
        transpose_168 = key_states_67.transpose(3, 2)
        scores_66 = torch.matmul(query_states_67, transpose_168)
        query_states_67 = transpose_168 = None
        scores_66 += position_bias_3
        scores_67 = scores_66
        scores_66 = None
        float_35 = scores_67.float()
        softmax_33 = torch.nn.functional.softmax(float_35, dim=-1)
        float_35 = None
        attn_weights_66 = softmax_33.type_as(scores_67)
        softmax_33 = scores_67 = None
        attn_weights_67 = torch.nn.functional.dropout(
            attn_weights_66, p=0.1, training=False
        )
        attn_weights_66 = None
        attn_output_132 = torch.matmul(attn_weights_67, value_states_67)
        attn_weights_67 = None
        transpose_169 = attn_output_132.transpose(1, 2)
        attn_output_132 = None
        attn_output_133 = transpose_169.contiguous()
        transpose_169 = None
        attn_output_134 = attn_output_133.view(1, -1, 512)
        attn_output_133 = None
        attn_output_135 = torch._C._nn.linear(
            attn_output_134,
            l_self_modules_block_modules_16_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_134 = l_self_modules_block_modules_16_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_ = (None)
        dropout_100 = torch.nn.functional.dropout(attn_output_135, 0.1, False, False)
        attn_output_135 = None
        layer_output_16 = hidden_states_146 + dropout_100
        hidden_states_146 = dropout_100 = None
        to_53 = layer_output_16.to(torch.float32)
        pow_51 = to_53.pow(2)
        to_53 = None
        variance_50 = pow_51.mean(-1, keepdim=True)
        pow_51 = None
        add_122 = variance_50 + 1e-06
        variance_50 = None
        rsqrt_50 = torch.rsqrt(add_122)
        add_122 = None
        hidden_states_148 = layer_output_16 * rsqrt_50
        rsqrt_50 = None
        forwarded_states_16 = (
            l_self_modules_block_modules_16_modules_layer_modules_2_modules_layer_norm_parameters_weight_
            * hidden_states_148
        )
        l_self_modules_block_modules_16_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = (
            hidden_states_148
        ) = None
        hidden_states_149 = torch._C._nn.linear(
            forwarded_states_16,
            l_self_modules_block_modules_16_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_16 = l_self_modules_block_modules_16_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_150 = torch.nn.functional.relu(hidden_states_149, inplace=False)
        hidden_states_149 = None
        hidden_states_151 = torch.nn.functional.dropout(
            hidden_states_150, 0.1, False, False
        )
        hidden_states_150 = None
        hidden_states_152 = torch._C._nn.linear(
            hidden_states_151,
            l_self_modules_block_modules_16_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_151 = l_self_modules_block_modules_16_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_102 = torch.nn.functional.dropout(hidden_states_152, 0.1, False, False)
        hidden_states_152 = None
        hidden_states_153 = layer_output_16 + dropout_102
        layer_output_16 = dropout_102 = None
        to_54 = hidden_states_153.to(torch.float32)
        pow_52 = to_54.pow(2)
        to_54 = None
        variance_51 = pow_52.mean(-1, keepdim=True)
        pow_52 = None
        add_124 = variance_51 + 1e-06
        variance_51 = None
        rsqrt_51 = torch.rsqrt(add_124)
        add_124 = None
        hidden_states_154 = hidden_states_153 * rsqrt_51
        rsqrt_51 = None
        normed_hidden_states_34 = (
            l_self_modules_block_modules_17_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_154
        )
        l_self_modules_block_modules_17_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_154
        ) = None
        query_states_68 = torch._C._nn.linear(
            normed_hidden_states_34,
            l_self_modules_block_modules_17_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_17_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_136 = query_states_68.view(1, -1, 8, 64)
        query_states_68 = None
        query_states_69 = view_136.transpose(1, 2)
        view_136 = None
        key_states_68 = torch._C._nn.linear(
            normed_hidden_states_34,
            l_self_modules_block_modules_17_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_17_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_68 = torch._C._nn.linear(
            normed_hidden_states_34,
            l_self_modules_block_modules_17_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_34 = l_self_modules_block_modules_17_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_137 = key_states_68.view(1, -1, 8, 64)
        key_states_68 = None
        key_states_69 = view_137.transpose(1, 2)
        view_137 = None
        view_138 = value_states_68.view(1, -1, 8, 64)
        value_states_68 = None
        value_states_69 = view_138.transpose(1, 2)
        view_138 = None
        transpose_173 = key_states_69.transpose(3, 2)
        scores_68 = torch.matmul(query_states_69, transpose_173)
        query_states_69 = transpose_173 = None
        scores_68 += position_bias_1
        scores_69 = scores_68
        scores_68 = None
        float_36 = scores_69.float()
        softmax_34 = torch.nn.functional.softmax(float_36, dim=-1)
        float_36 = None
        attn_weights_68 = softmax_34.type_as(scores_69)
        softmax_34 = scores_69 = None
        attn_weights_69 = torch.nn.functional.dropout(
            attn_weights_68, p=0.1, training=False
        )
        attn_weights_68 = None
        attn_output_136 = torch.matmul(attn_weights_69, value_states_69)
        attn_weights_69 = None
        transpose_174 = attn_output_136.transpose(1, 2)
        attn_output_136 = None
        attn_output_137 = transpose_174.contiguous()
        transpose_174 = None
        attn_output_138 = attn_output_137.view(1, -1, 512)
        attn_output_137 = None
        attn_output_139 = torch._C._nn.linear(
            attn_output_138,
            l_self_modules_block_modules_17_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_138 = l_self_modules_block_modules_17_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_104 = torch.nn.functional.dropout(attn_output_139, 0.1, False, False)
        attn_output_139 = None
        hidden_states_155 = hidden_states_153 + dropout_104
        hidden_states_153 = dropout_104 = None
        getitem_25 = cache_position[-1]
        add_126 = getitem_25 + 1
        getitem_25 = add_126 = None
        to_55 = hidden_states_155.to(torch.float32)
        pow_53 = to_55.pow(2)
        to_55 = None
        variance_52 = pow_53.mean(-1, keepdim=True)
        pow_53 = None
        add_127 = variance_52 + 1e-06
        variance_52 = None
        rsqrt_52 = torch.rsqrt(add_127)
        add_127 = None
        hidden_states_156 = hidden_states_155 * rsqrt_52
        rsqrt_52 = None
        normed_hidden_states_35 = (
            l_self_modules_block_modules_17_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_156
        )
        l_self_modules_block_modules_17_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_156
        ) = None
        query_states_70 = torch._C._nn.linear(
            normed_hidden_states_35,
            l_self_modules_block_modules_17_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_,
            None,
        )
        normed_hidden_states_35 = l_self_modules_block_modules_17_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_ = (None)
        view_140 = query_states_70.view(1, -1, 8, 64)
        query_states_70 = None
        query_states_71 = view_140.transpose(1, 2)
        view_140 = None
        key_states_70 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_block_modules_17_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_17_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_70 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_block_modules_17_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_17_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_ = (
            None
        )
        view_141 = key_states_70.view(1, -1, 8, 64)
        key_states_70 = None
        key_states_71 = view_141.transpose(1, 2)
        view_141 = None
        view_142 = value_states_70.view(1, -1, 8, 64)
        value_states_70 = None
        value_states_71 = view_142.transpose(1, 2)
        view_142 = None
        transpose_178 = key_states_71.transpose(3, 2)
        scores_70 = torch.matmul(query_states_71, transpose_178)
        query_states_71 = transpose_178 = None
        scores_70 += position_bias_3
        scores_71 = scores_70
        scores_70 = None
        float_37 = scores_71.float()
        softmax_35 = torch.nn.functional.softmax(float_37, dim=-1)
        float_37 = None
        attn_weights_70 = softmax_35.type_as(scores_71)
        softmax_35 = scores_71 = None
        attn_weights_71 = torch.nn.functional.dropout(
            attn_weights_70, p=0.1, training=False
        )
        attn_weights_70 = None
        attn_output_140 = torch.matmul(attn_weights_71, value_states_71)
        attn_weights_71 = None
        transpose_179 = attn_output_140.transpose(1, 2)
        attn_output_140 = None
        attn_output_141 = transpose_179.contiguous()
        transpose_179 = None
        attn_output_142 = attn_output_141.view(1, -1, 512)
        attn_output_141 = None
        attn_output_143 = torch._C._nn.linear(
            attn_output_142,
            l_self_modules_block_modules_17_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_142 = l_self_modules_block_modules_17_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_ = (None)
        dropout_106 = torch.nn.functional.dropout(attn_output_143, 0.1, False, False)
        attn_output_143 = None
        layer_output_17 = hidden_states_155 + dropout_106
        hidden_states_155 = dropout_106 = None
        to_56 = layer_output_17.to(torch.float32)
        pow_54 = to_56.pow(2)
        to_56 = None
        variance_53 = pow_54.mean(-1, keepdim=True)
        pow_54 = None
        add_129 = variance_53 + 1e-06
        variance_53 = None
        rsqrt_53 = torch.rsqrt(add_129)
        add_129 = None
        hidden_states_157 = layer_output_17 * rsqrt_53
        rsqrt_53 = None
        forwarded_states_17 = (
            l_self_modules_block_modules_17_modules_layer_modules_2_modules_layer_norm_parameters_weight_
            * hidden_states_157
        )
        l_self_modules_block_modules_17_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = (
            hidden_states_157
        ) = None
        hidden_states_158 = torch._C._nn.linear(
            forwarded_states_17,
            l_self_modules_block_modules_17_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_17 = l_self_modules_block_modules_17_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_159 = torch.nn.functional.relu(hidden_states_158, inplace=False)
        hidden_states_158 = None
        hidden_states_160 = torch.nn.functional.dropout(
            hidden_states_159, 0.1, False, False
        )
        hidden_states_159 = None
        hidden_states_161 = torch._C._nn.linear(
            hidden_states_160,
            l_self_modules_block_modules_17_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_160 = l_self_modules_block_modules_17_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_108 = torch.nn.functional.dropout(hidden_states_161, 0.1, False, False)
        hidden_states_161 = None
        hidden_states_162 = layer_output_17 + dropout_108
        layer_output_17 = dropout_108 = None
        to_57 = hidden_states_162.to(torch.float32)
        pow_55 = to_57.pow(2)
        to_57 = None
        variance_54 = pow_55.mean(-1, keepdim=True)
        pow_55 = None
        add_131 = variance_54 + 1e-06
        variance_54 = None
        rsqrt_54 = torch.rsqrt(add_131)
        add_131 = None
        hidden_states_163 = hidden_states_162 * rsqrt_54
        rsqrt_54 = None
        normed_hidden_states_36 = (
            l_self_modules_block_modules_18_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_163
        )
        l_self_modules_block_modules_18_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_163
        ) = None
        query_states_72 = torch._C._nn.linear(
            normed_hidden_states_36,
            l_self_modules_block_modules_18_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_18_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_144 = query_states_72.view(1, -1, 8, 64)
        query_states_72 = None
        query_states_73 = view_144.transpose(1, 2)
        view_144 = None
        key_states_72 = torch._C._nn.linear(
            normed_hidden_states_36,
            l_self_modules_block_modules_18_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_18_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_72 = torch._C._nn.linear(
            normed_hidden_states_36,
            l_self_modules_block_modules_18_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_36 = l_self_modules_block_modules_18_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_145 = key_states_72.view(1, -1, 8, 64)
        key_states_72 = None
        key_states_73 = view_145.transpose(1, 2)
        view_145 = None
        view_146 = value_states_72.view(1, -1, 8, 64)
        value_states_72 = None
        value_states_73 = view_146.transpose(1, 2)
        view_146 = None
        transpose_183 = key_states_73.transpose(3, 2)
        scores_72 = torch.matmul(query_states_73, transpose_183)
        query_states_73 = transpose_183 = None
        scores_72 += position_bias_1
        scores_73 = scores_72
        scores_72 = None
        float_38 = scores_73.float()
        softmax_36 = torch.nn.functional.softmax(float_38, dim=-1)
        float_38 = None
        attn_weights_72 = softmax_36.type_as(scores_73)
        softmax_36 = scores_73 = None
        attn_weights_73 = torch.nn.functional.dropout(
            attn_weights_72, p=0.1, training=False
        )
        attn_weights_72 = None
        attn_output_144 = torch.matmul(attn_weights_73, value_states_73)
        attn_weights_73 = None
        transpose_184 = attn_output_144.transpose(1, 2)
        attn_output_144 = None
        attn_output_145 = transpose_184.contiguous()
        transpose_184 = None
        attn_output_146 = attn_output_145.view(1, -1, 512)
        attn_output_145 = None
        attn_output_147 = torch._C._nn.linear(
            attn_output_146,
            l_self_modules_block_modules_18_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_146 = l_self_modules_block_modules_18_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_110 = torch.nn.functional.dropout(attn_output_147, 0.1, False, False)
        attn_output_147 = None
        hidden_states_164 = hidden_states_162 + dropout_110
        hidden_states_162 = dropout_110 = None
        getitem_26 = cache_position[-1]
        add_133 = getitem_26 + 1
        getitem_26 = add_133 = None
        to_58 = hidden_states_164.to(torch.float32)
        pow_56 = to_58.pow(2)
        to_58 = None
        variance_55 = pow_56.mean(-1, keepdim=True)
        pow_56 = None
        add_134 = variance_55 + 1e-06
        variance_55 = None
        rsqrt_55 = torch.rsqrt(add_134)
        add_134 = None
        hidden_states_165 = hidden_states_164 * rsqrt_55
        rsqrt_55 = None
        normed_hidden_states_37 = (
            l_self_modules_block_modules_18_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_165
        )
        l_self_modules_block_modules_18_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_165
        ) = None
        query_states_74 = torch._C._nn.linear(
            normed_hidden_states_37,
            l_self_modules_block_modules_18_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_,
            None,
        )
        normed_hidden_states_37 = l_self_modules_block_modules_18_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_ = (None)
        view_148 = query_states_74.view(1, -1, 8, 64)
        query_states_74 = None
        query_states_75 = view_148.transpose(1, 2)
        view_148 = None
        key_states_74 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_block_modules_18_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_18_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_74 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_block_modules_18_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_18_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_ = (
            None
        )
        view_149 = key_states_74.view(1, -1, 8, 64)
        key_states_74 = None
        key_states_75 = view_149.transpose(1, 2)
        view_149 = None
        view_150 = value_states_74.view(1, -1, 8, 64)
        value_states_74 = None
        value_states_75 = view_150.transpose(1, 2)
        view_150 = None
        transpose_188 = key_states_75.transpose(3, 2)
        scores_74 = torch.matmul(query_states_75, transpose_188)
        query_states_75 = transpose_188 = None
        scores_74 += position_bias_3
        scores_75 = scores_74
        scores_74 = None
        float_39 = scores_75.float()
        softmax_37 = torch.nn.functional.softmax(float_39, dim=-1)
        float_39 = None
        attn_weights_74 = softmax_37.type_as(scores_75)
        softmax_37 = scores_75 = None
        attn_weights_75 = torch.nn.functional.dropout(
            attn_weights_74, p=0.1, training=False
        )
        attn_weights_74 = None
        attn_output_148 = torch.matmul(attn_weights_75, value_states_75)
        attn_weights_75 = None
        transpose_189 = attn_output_148.transpose(1, 2)
        attn_output_148 = None
        attn_output_149 = transpose_189.contiguous()
        transpose_189 = None
        attn_output_150 = attn_output_149.view(1, -1, 512)
        attn_output_149 = None
        attn_output_151 = torch._C._nn.linear(
            attn_output_150,
            l_self_modules_block_modules_18_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_150 = l_self_modules_block_modules_18_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_ = (None)
        dropout_112 = torch.nn.functional.dropout(attn_output_151, 0.1, False, False)
        attn_output_151 = None
        layer_output_18 = hidden_states_164 + dropout_112
        hidden_states_164 = dropout_112 = None
        to_59 = layer_output_18.to(torch.float32)
        pow_57 = to_59.pow(2)
        to_59 = None
        variance_56 = pow_57.mean(-1, keepdim=True)
        pow_57 = None
        add_136 = variance_56 + 1e-06
        variance_56 = None
        rsqrt_56 = torch.rsqrt(add_136)
        add_136 = None
        hidden_states_166 = layer_output_18 * rsqrt_56
        rsqrt_56 = None
        forwarded_states_18 = (
            l_self_modules_block_modules_18_modules_layer_modules_2_modules_layer_norm_parameters_weight_
            * hidden_states_166
        )
        l_self_modules_block_modules_18_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = (
            hidden_states_166
        ) = None
        hidden_states_167 = torch._C._nn.linear(
            forwarded_states_18,
            l_self_modules_block_modules_18_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_18 = l_self_modules_block_modules_18_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_168 = torch.nn.functional.relu(hidden_states_167, inplace=False)
        hidden_states_167 = None
        hidden_states_169 = torch.nn.functional.dropout(
            hidden_states_168, 0.1, False, False
        )
        hidden_states_168 = None
        hidden_states_170 = torch._C._nn.linear(
            hidden_states_169,
            l_self_modules_block_modules_18_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_169 = l_self_modules_block_modules_18_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_114 = torch.nn.functional.dropout(hidden_states_170, 0.1, False, False)
        hidden_states_170 = None
        hidden_states_171 = layer_output_18 + dropout_114
        layer_output_18 = dropout_114 = None
        to_60 = hidden_states_171.to(torch.float32)
        pow_58 = to_60.pow(2)
        to_60 = None
        variance_57 = pow_58.mean(-1, keepdim=True)
        pow_58 = None
        add_138 = variance_57 + 1e-06
        variance_57 = None
        rsqrt_57 = torch.rsqrt(add_138)
        add_138 = None
        hidden_states_172 = hidden_states_171 * rsqrt_57
        rsqrt_57 = None
        normed_hidden_states_38 = (
            l_self_modules_block_modules_19_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_172
        )
        l_self_modules_block_modules_19_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_172
        ) = None
        query_states_76 = torch._C._nn.linear(
            normed_hidden_states_38,
            l_self_modules_block_modules_19_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_19_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_152 = query_states_76.view(1, -1, 8, 64)
        query_states_76 = None
        query_states_77 = view_152.transpose(1, 2)
        view_152 = None
        key_states_76 = torch._C._nn.linear(
            normed_hidden_states_38,
            l_self_modules_block_modules_19_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_19_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_76 = torch._C._nn.linear(
            normed_hidden_states_38,
            l_self_modules_block_modules_19_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_38 = l_self_modules_block_modules_19_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_153 = key_states_76.view(1, -1, 8, 64)
        key_states_76 = None
        key_states_77 = view_153.transpose(1, 2)
        view_153 = None
        view_154 = value_states_76.view(1, -1, 8, 64)
        value_states_76 = None
        value_states_77 = view_154.transpose(1, 2)
        view_154 = None
        transpose_193 = key_states_77.transpose(3, 2)
        scores_76 = torch.matmul(query_states_77, transpose_193)
        query_states_77 = transpose_193 = None
        scores_76 += position_bias_1
        scores_77 = scores_76
        scores_76 = position_bias_1 = None
        float_40 = scores_77.float()
        softmax_38 = torch.nn.functional.softmax(float_40, dim=-1)
        float_40 = None
        attn_weights_76 = softmax_38.type_as(scores_77)
        softmax_38 = scores_77 = None
        attn_weights_77 = torch.nn.functional.dropout(
            attn_weights_76, p=0.1, training=False
        )
        attn_weights_76 = None
        attn_output_152 = torch.matmul(attn_weights_77, value_states_77)
        attn_weights_77 = None
        transpose_194 = attn_output_152.transpose(1, 2)
        attn_output_152 = None
        attn_output_153 = transpose_194.contiguous()
        transpose_194 = None
        attn_output_154 = attn_output_153.view(1, -1, 512)
        attn_output_153 = None
        attn_output_155 = torch._C._nn.linear(
            attn_output_154,
            l_self_modules_block_modules_19_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_154 = l_self_modules_block_modules_19_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_116 = torch.nn.functional.dropout(attn_output_155, 0.1, False, False)
        attn_output_155 = None
        hidden_states_173 = hidden_states_171 + dropout_116
        hidden_states_171 = dropout_116 = None
        getitem_27 = cache_position[-1]
        cache_position = None
        add_140 = getitem_27 + 1
        getitem_27 = add_140 = None
        to_61 = hidden_states_173.to(torch.float32)
        pow_59 = to_61.pow(2)
        to_61 = None
        variance_58 = pow_59.mean(-1, keepdim=True)
        pow_59 = None
        add_141 = variance_58 + 1e-06
        variance_58 = None
        rsqrt_58 = torch.rsqrt(add_141)
        add_141 = None
        hidden_states_174 = hidden_states_173 * rsqrt_58
        rsqrt_58 = None
        normed_hidden_states_39 = (
            l_self_modules_block_modules_19_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_174
        )
        l_self_modules_block_modules_19_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_174
        ) = None
        query_states_78 = torch._C._nn.linear(
            normed_hidden_states_39,
            l_self_modules_block_modules_19_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_,
            None,
        )
        normed_hidden_states_39 = l_self_modules_block_modules_19_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_ = (None)
        view_156 = query_states_78.view(1, -1, 8, 64)
        query_states_78 = None
        query_states_79 = view_156.transpose(1, 2)
        view_156 = None
        key_states_78 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_block_modules_19_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_19_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_78 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_block_modules_19_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_,
            None,
        )
        l_encoder_hidden_states_ = l_self_modules_block_modules_19_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_ = (None)
        view_157 = key_states_78.view(1, -1, 8, 64)
        key_states_78 = None
        key_states_79 = view_157.transpose(1, 2)
        view_157 = None
        view_158 = value_states_78.view(1, -1, 8, 64)
        value_states_78 = None
        value_states_79 = view_158.transpose(1, 2)
        view_158 = None
        transpose_198 = key_states_79.transpose(3, 2)
        scores_78 = torch.matmul(query_states_79, transpose_198)
        query_states_79 = transpose_198 = None
        scores_78 += position_bias_3
        scores_79 = scores_78
        scores_78 = position_bias_3 = None
        float_41 = scores_79.float()
        softmax_39 = torch.nn.functional.softmax(float_41, dim=-1)
        float_41 = None
        attn_weights_78 = softmax_39.type_as(scores_79)
        softmax_39 = scores_79 = None
        attn_weights_79 = torch.nn.functional.dropout(
            attn_weights_78, p=0.1, training=False
        )
        attn_weights_78 = None
        attn_output_156 = torch.matmul(attn_weights_79, value_states_79)
        attn_weights_79 = None
        transpose_199 = attn_output_156.transpose(1, 2)
        attn_output_156 = None
        attn_output_157 = transpose_199.contiguous()
        transpose_199 = None
        attn_output_158 = attn_output_157.view(1, -1, 512)
        attn_output_157 = None
        attn_output_159 = torch._C._nn.linear(
            attn_output_158,
            l_self_modules_block_modules_19_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_158 = l_self_modules_block_modules_19_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_ = (None)
        dropout_118 = torch.nn.functional.dropout(attn_output_159, 0.1, False, False)
        attn_output_159 = None
        layer_output_19 = hidden_states_173 + dropout_118
        hidden_states_173 = dropout_118 = None
        to_62 = layer_output_19.to(torch.float32)
        pow_60 = to_62.pow(2)
        to_62 = None
        variance_59 = pow_60.mean(-1, keepdim=True)
        pow_60 = None
        add_143 = variance_59 + 1e-06
        variance_59 = None
        rsqrt_59 = torch.rsqrt(add_143)
        add_143 = None
        hidden_states_175 = layer_output_19 * rsqrt_59
        rsqrt_59 = None
        forwarded_states_19 = (
            l_self_modules_block_modules_19_modules_layer_modules_2_modules_layer_norm_parameters_weight_
            * hidden_states_175
        )
        l_self_modules_block_modules_19_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = (
            hidden_states_175
        ) = None
        hidden_states_176 = torch._C._nn.linear(
            forwarded_states_19,
            l_self_modules_block_modules_19_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_19 = l_self_modules_block_modules_19_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_177 = torch.nn.functional.relu(hidden_states_176, inplace=False)
        hidden_states_176 = None
        hidden_states_178 = torch.nn.functional.dropout(
            hidden_states_177, 0.1, False, False
        )
        hidden_states_177 = None
        hidden_states_179 = torch._C._nn.linear(
            hidden_states_178,
            l_self_modules_block_modules_19_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_178 = l_self_modules_block_modules_19_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_120 = torch.nn.functional.dropout(hidden_states_179, 0.1, False, False)
        hidden_states_179 = None
        hidden_states_180 = layer_output_19 + dropout_120
        layer_output_19 = dropout_120 = None
        to_63 = hidden_states_180.to(torch.float32)
        pow_61 = to_63.pow(2)
        to_63 = None
        variance_60 = pow_61.mean(-1, keepdim=True)
        pow_61 = None
        add_145 = variance_60 + 1e-06
        variance_60 = None
        rsqrt_60 = torch.rsqrt(add_145)
        add_145 = None
        hidden_states_181 = hidden_states_180 * rsqrt_60
        hidden_states_180 = rsqrt_60 = None
        hidden_states_182 = (
            l_self_modules_final_layer_norm_parameters_weight_ * hidden_states_181
        )
        l_self_modules_final_layer_norm_parameters_weight_ = hidden_states_181 = None
        hidden_states_183 = torch.nn.functional.dropout(
            hidden_states_182, 0.1, False, False
        )
        hidden_states_182 = None
        return (
            value_states_1,
            key_states_1,
            value_states_3,
            key_states_3,
            value_states_5,
            key_states_5,
            value_states_7,
            key_states_7,
            value_states_9,
            key_states_9,
            value_states_11,
            key_states_11,
            value_states_13,
            key_states_13,
            value_states_15,
            key_states_15,
            value_states_17,
            key_states_17,
            value_states_19,
            key_states_19,
            value_states_21,
            key_states_21,
            value_states_23,
            key_states_23,
            value_states_25,
            key_states_25,
            value_states_27,
            key_states_27,
            value_states_29,
            key_states_29,
            value_states_31,
            key_states_31,
            value_states_33,
            key_states_33,
            value_states_35,
            key_states_35,
            value_states_37,
            key_states_37,
            value_states_39,
            key_states_39,
            value_states_41,
            key_states_41,
            value_states_43,
            key_states_43,
            value_states_45,
            key_states_45,
            value_states_47,
            key_states_47,
            value_states_49,
            key_states_49,
            value_states_51,
            key_states_51,
            value_states_53,
            key_states_53,
            value_states_55,
            key_states_55,
            value_states_57,
            key_states_57,
            value_states_59,
            key_states_59,
            value_states_61,
            key_states_61,
            value_states_63,
            key_states_63,
            value_states_65,
            key_states_65,
            value_states_67,
            key_states_67,
            value_states_69,
            key_states_69,
            value_states_71,
            key_states_71,
            value_states_73,
            key_states_73,
            value_states_75,
            key_states_75,
            value_states_77,
            key_states_77,
            value_states_79,
            key_states_79,
            hidden_states_183,
        )
