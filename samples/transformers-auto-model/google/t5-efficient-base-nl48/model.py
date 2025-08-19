import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_input_ids_: torch.Tensor,
        L_self_modules_embed_tokens_parameters_weight_: torch.nn.parameter.Parameter,
        L_attention_mask_: torch.Tensor,
        L_self_modules_block_modules_0_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_relative_attention_bias_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_0_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_0_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_0_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_4_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_4_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_4_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_4_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_4_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_4_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_4_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_4_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_5_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_5_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_5_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_5_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_5_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_5_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_5_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_5_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_6_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_6_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_6_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_6_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_6_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_6_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_6_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_6_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_7_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_7_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_7_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_7_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_7_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_7_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_7_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_7_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_8_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_8_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_8_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_8_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_8_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_8_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_8_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_8_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_9_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_9_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_9_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_9_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_9_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_9_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_9_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_9_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_10_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_10_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_10_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_10_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_10_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_10_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_10_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_10_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_11_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_11_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_11_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_11_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_11_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_11_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_11_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_11_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_12_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_12_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_12_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_12_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_12_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_12_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_12_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_12_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_13_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_13_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_13_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_13_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_13_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_13_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_13_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_13_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_14_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_14_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_14_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_14_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_14_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_14_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_14_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_14_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_15_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_15_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_15_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_15_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_15_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_15_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_15_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_15_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_16_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_16_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_16_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_16_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_16_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_16_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_16_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_16_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_17_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_17_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_17_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_17_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_17_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_17_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_17_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_17_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_18_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_18_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_18_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_18_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_18_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_18_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_18_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_18_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_19_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_19_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_19_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_19_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_19_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_19_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_19_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_19_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_20_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_20_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_20_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_20_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_20_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_20_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_20_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_20_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_21_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_21_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_21_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_21_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_21_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_21_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_21_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_21_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_22_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_22_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_22_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_22_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_22_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_22_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_22_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_22_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_23_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_23_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_23_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_23_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_23_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_23_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_23_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_23_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_24_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_24_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_24_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_24_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_24_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_24_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_24_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_24_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_25_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_25_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_25_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_25_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_25_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_25_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_25_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_25_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_26_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_26_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_26_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_26_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_26_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_26_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_26_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_26_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_27_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_27_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_27_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_27_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_27_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_27_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_27_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_27_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_28_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_28_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_28_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_28_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_28_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_28_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_28_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_28_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_29_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_29_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_29_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_29_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_29_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_29_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_29_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_29_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_30_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_30_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_30_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_30_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_30_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_30_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_30_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_30_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_31_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_31_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_31_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_31_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_31_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_31_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_31_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_31_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_32_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_32_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_32_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_32_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_32_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_32_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_32_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_32_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_33_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_33_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_33_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_33_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_33_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_33_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_33_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_33_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_34_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_34_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_34_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_34_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_34_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_34_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_34_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_34_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_35_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_35_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_35_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_35_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_35_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_35_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_35_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_35_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_36_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_36_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_36_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_36_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_36_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_36_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_36_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_36_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_37_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_37_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_37_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_37_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_37_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_37_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_37_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_37_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_38_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_38_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_38_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_38_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_38_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_38_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_38_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_38_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_39_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_39_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_39_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_39_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_39_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_39_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_39_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_39_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_40_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_40_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_40_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_40_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_40_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_40_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_40_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_40_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_41_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_41_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_41_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_41_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_41_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_41_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_41_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_41_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_42_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_42_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_42_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_42_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_42_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_42_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_42_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_42_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_43_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_43_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_43_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_43_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_43_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_43_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_43_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_43_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_44_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_44_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_44_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_44_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_44_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_44_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_44_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_44_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_45_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_45_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_45_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_45_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_45_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_45_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_45_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_45_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_46_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_46_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_46_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_46_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_46_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_46_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_46_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_46_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_47_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_47_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_47_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_47_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_47_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_47_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_47_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_47_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
    ):
        l_input_ids_ = L_input_ids_
        l_self_modules_embed_tokens_parameters_weight_ = (
            L_self_modules_embed_tokens_parameters_weight_
        )
        l_attention_mask_ = L_attention_mask_
        l_self_modules_block_modules_0_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_0_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_0_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_0_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_0_modules_layer_modules_0_modules_self_attention_modules_relative_attention_bias_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_relative_attention_bias_parameters_weight_
        l_self_modules_block_modules_0_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_0_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_0_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_0_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_4_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_4_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_4_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_4_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_4_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_4_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_4_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_4_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_4_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_4_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_4_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_4_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_4_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_4_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_4_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_4_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_5_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_5_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_5_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_5_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_5_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_5_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_5_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_5_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_5_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_5_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_5_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_5_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_5_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_5_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_5_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_5_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_6_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_6_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_6_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_6_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_6_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_6_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_6_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_6_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_6_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_6_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_6_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_6_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_6_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_6_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_6_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_6_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_7_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_7_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_7_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_7_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_7_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_7_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_7_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_7_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_7_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_7_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_7_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_7_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_7_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_7_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_7_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_7_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_8_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_8_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_8_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_8_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_8_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_8_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_8_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_8_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_8_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_8_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_8_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_8_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_8_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_8_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_8_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_8_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_9_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_9_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_9_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_9_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_9_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_9_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_9_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_9_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_9_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_9_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_9_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_9_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_9_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_9_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_9_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_9_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_10_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_10_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_10_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_10_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_10_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_10_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_10_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_10_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_10_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_10_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_10_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_10_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_10_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_10_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_10_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_10_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_11_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_11_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_11_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_11_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_11_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_11_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_11_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_11_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_11_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_11_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_11_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_11_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_11_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_11_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_11_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_11_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_12_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_12_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_12_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_12_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_12_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_12_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_12_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_12_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_12_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_12_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_12_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_12_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_12_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_12_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_12_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_12_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_13_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_13_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_13_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_13_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_13_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_13_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_13_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_13_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_13_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_13_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_13_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_13_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_13_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_13_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_13_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_13_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_14_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_14_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_14_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_14_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_14_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_14_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_14_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_14_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_14_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_14_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_14_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_14_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_14_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_14_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_14_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_14_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_15_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_15_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_15_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_15_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_15_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_15_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_15_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_15_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_15_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_15_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_15_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_15_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_15_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_15_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_15_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_15_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_16_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_16_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_16_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_16_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_16_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_16_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_16_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_16_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_16_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_16_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_16_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_16_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_16_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_16_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_16_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_16_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_17_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_17_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_17_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_17_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_17_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_17_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_17_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_17_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_17_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_17_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_17_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_17_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_17_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_17_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_17_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_17_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_18_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_18_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_18_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_18_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_18_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_18_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_18_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_18_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_18_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_18_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_18_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_18_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_18_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_18_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_18_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_18_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_19_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_19_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_19_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_19_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_19_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_19_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_19_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_19_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_19_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_19_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_19_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_19_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_19_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_19_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_19_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_19_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_20_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_20_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_20_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_20_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_20_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_20_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_20_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_20_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_20_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_20_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_20_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_20_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_20_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_20_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_20_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_20_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_21_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_21_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_21_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_21_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_21_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_21_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_21_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_21_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_21_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_21_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_21_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_21_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_21_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_21_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_21_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_21_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_22_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_22_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_22_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_22_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_22_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_22_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_22_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_22_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_22_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_22_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_22_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_22_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_22_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_22_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_22_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_22_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_23_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_23_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_23_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_23_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_23_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_23_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_23_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_23_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_23_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_23_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_23_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_23_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_23_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_23_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_23_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_23_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_24_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_24_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_24_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_24_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_24_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_24_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_24_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_24_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_24_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_24_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_24_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_24_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_24_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_24_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_24_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_24_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_25_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_25_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_25_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_25_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_25_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_25_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_25_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_25_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_25_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_25_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_25_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_25_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_25_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_25_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_25_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_25_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_26_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_26_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_26_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_26_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_26_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_26_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_26_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_26_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_26_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_26_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_26_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_26_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_26_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_26_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_26_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_26_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_27_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_27_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_27_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_27_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_27_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_27_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_27_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_27_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_27_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_27_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_27_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_27_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_27_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_27_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_27_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_27_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_28_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_28_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_28_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_28_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_28_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_28_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_28_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_28_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_28_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_28_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_28_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_28_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_28_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_28_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_28_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_28_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_29_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_29_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_29_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_29_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_29_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_29_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_29_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_29_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_29_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_29_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_29_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_29_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_29_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_29_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_29_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_29_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_30_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_30_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_30_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_30_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_30_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_30_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_30_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_30_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_30_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_30_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_30_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_30_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_30_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_30_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_30_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_30_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_31_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_31_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_31_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_31_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_31_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_31_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_31_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_31_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_31_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_31_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_31_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_31_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_31_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_31_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_31_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_31_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_32_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_32_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_32_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_32_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_32_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_32_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_32_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_32_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_32_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_32_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_32_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_32_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_32_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_32_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_32_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_32_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_33_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_33_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_33_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_33_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_33_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_33_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_33_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_33_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_33_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_33_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_33_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_33_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_33_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_33_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_33_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_33_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_34_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_34_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_34_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_34_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_34_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_34_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_34_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_34_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_34_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_34_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_34_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_34_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_34_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_34_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_34_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_34_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_35_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_35_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_35_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_35_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_35_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_35_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_35_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_35_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_35_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_35_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_35_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_35_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_35_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_35_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_35_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_35_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_36_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_36_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_36_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_36_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_36_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_36_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_36_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_36_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_36_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_36_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_36_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_36_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_36_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_36_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_36_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_36_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_37_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_37_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_37_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_37_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_37_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_37_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_37_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_37_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_37_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_37_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_37_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_37_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_37_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_37_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_37_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_37_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_38_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_38_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_38_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_38_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_38_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_38_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_38_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_38_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_38_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_38_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_38_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_38_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_38_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_38_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_38_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_38_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_39_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_39_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_39_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_39_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_39_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_39_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_39_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_39_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_39_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_39_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_39_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_39_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_39_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_39_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_39_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_39_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_40_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_40_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_40_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_40_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_40_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_40_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_40_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_40_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_40_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_40_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_40_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_40_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_40_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_40_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_40_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_40_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_41_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_41_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_41_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_41_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_41_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_41_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_41_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_41_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_41_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_41_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_41_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_41_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_41_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_41_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_41_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_41_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_42_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_42_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_42_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_42_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_42_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_42_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_42_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_42_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_42_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_42_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_42_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_42_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_42_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_42_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_42_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_42_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_43_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_43_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_43_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_43_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_43_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_43_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_43_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_43_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_43_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_43_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_43_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_43_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_43_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_43_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_43_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_43_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_44_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_44_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_44_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_44_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_44_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_44_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_44_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_44_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_44_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_44_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_44_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_44_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_44_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_44_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_44_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_44_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_45_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_45_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_45_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_45_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_45_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_45_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_45_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_45_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_45_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_45_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_45_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_45_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_45_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_45_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_45_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_45_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_46_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_46_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_46_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_46_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_46_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_46_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_46_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_46_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_46_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_46_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_46_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_46_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_46_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_46_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_46_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_46_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_47_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_47_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_47_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_47_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_47_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_47_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_47_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_47_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_47_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_47_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_47_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_47_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_47_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_47_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_47_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_47_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_final_layer_norm_parameters_weight_
        )
        input_ids = l_input_ids_.view(-1, 12)
        l_input_ids_ = None
        inputs_embeds = torch.nn.functional.embedding(
            input_ids,
            l_self_modules_embed_tokens_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        input_ids = l_self_modules_embed_tokens_parameters_weight_ = None
        cache_position = torch.arange(0, 12, device=device(type="cuda", index=0))
        causal_mask = l_attention_mask_[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        l_attention_mask_ = None
        causal_mask_1 = causal_mask.to(dtype=torch.float32)
        causal_mask = None
        sub = 1.0 - causal_mask_1
        causal_mask_1 = None
        causal_mask_2 = sub * -3.4028234663852886e38
        sub = None
        hidden_states = torch.nn.functional.dropout(inputs_embeds, 0.1, False, False)
        inputs_embeds = None
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
        view_1 = query_states.view(1, -1, 12, 64)
        query_states = None
        query_states_1 = view_1.transpose(1, 2)
        view_1 = None
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
        view_2 = key_states.view(1, -1, 12, 64)
        key_states = None
        key_states_1 = view_2.transpose(1, 2)
        view_2 = None
        view_3 = value_states.view(1, -1, 12, 64)
        value_states = None
        value_states_1 = view_3.transpose(1, 2)
        view_3 = None
        transpose_3 = key_states_1.transpose(3, 2)
        key_states_1 = None
        scores = torch.matmul(query_states_1, transpose_3)
        query_states_1 = transpose_3 = None
        getitem_1 = cache_position[-1]
        real_seq_length = getitem_1 + 1
        getitem_1 = real_seq_length = None
        getitem_2 = cache_position[(slice(None, None, None), None)]
        cache_position = None
        context_position = getitem_2.to(device(type="cuda", index=0))
        getitem_2 = None
        arange_1 = torch.arange(
            12, dtype=torch.int64, device=device(type="cuda", index=0)
        )
        memory_position = arange_1[(None, slice(None, None, None))]
        arange_1 = None
        relative_position = memory_position - context_position
        memory_position = context_position = None
        gt = relative_position > 0
        to_3 = gt.to(torch.int64)
        gt = None
        mul_3 = to_3 * 16
        to_3 = None
        relative_buckets = 0 + mul_3
        mul_3 = None
        relative_position_1 = torch.abs(relative_position)
        relative_position = None
        is_small = relative_position_1 < 8
        float_1 = relative_position_1.float()
        truediv = float_1 / 8
        float_1 = None
        log = torch.log(truediv)
        truediv = None
        truediv_1 = log / 2.772588722239781
        log = None
        mul_4 = truediv_1 * 8
        truediv_1 = None
        to_4 = mul_4.to(torch.int64)
        mul_4 = None
        relative_position_if_large = 8 + to_4
        to_4 = None
        full_like = torch.full_like(relative_position_if_large, 15)
        relative_position_if_large_1 = torch.min(relative_position_if_large, full_like)
        relative_position_if_large = full_like = None
        where = torch.where(is_small, relative_position_1, relative_position_if_large_1)
        is_small = relative_position_1 = relative_position_if_large_1 = None
        relative_buckets += where
        relative_buckets_1 = relative_buckets
        relative_buckets = where = None
        values = torch.nn.functional.embedding(
            relative_buckets_1,
            l_self_modules_block_modules_0_modules_layer_modules_0_modules_self_attention_modules_relative_attention_bias_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        relative_buckets_1 = l_self_modules_block_modules_0_modules_layer_modules_0_modules_self_attention_modules_relative_attention_bias_parameters_weight_ = (None)
        permute = values.permute([2, 0, 1])
        values = None
        values_1 = permute.unsqueeze(0)
        permute = None
        position_bias = values_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(-12, None, None),
                slice(None, None, None),
            )
        ]
        values_1 = None
        causal_mask_3 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 12, None),
            )
        ]
        causal_mask_2 = None
        position_bias_1 = position_bias + causal_mask_3
        position_bias = causal_mask_3 = None
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
        attn_weights_1 = value_states_1 = None
        transpose_4 = attn_output.transpose(1, 2)
        attn_output = None
        attn_output_1 = transpose_4.contiguous()
        transpose_4 = None
        attn_output_2 = attn_output_1.view(1, -1, 768)
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
        to_5 = hidden_states_2.to(torch.float32)
        pow_2 = to_5.pow(2)
        to_5 = None
        variance_1 = pow_2.mean(-1, keepdim=True)
        pow_2 = None
        add_6 = variance_1 + 1e-06
        variance_1 = None
        rsqrt_1 = torch.rsqrt(add_6)
        add_6 = None
        hidden_states_3 = hidden_states_2 * rsqrt_1
        rsqrt_1 = None
        forwarded_states = (
            l_self_modules_block_modules_0_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_3
        )
        l_self_modules_block_modules_0_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_3
        ) = None
        hidden_states_4 = torch._C._nn.linear(
            forwarded_states,
            l_self_modules_block_modules_0_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states = l_self_modules_block_modules_0_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_5 = torch.nn.functional.relu(hidden_states_4, inplace=False)
        hidden_states_4 = None
        hidden_states_6 = torch.nn.functional.dropout(
            hidden_states_5, 0.1, False, False
        )
        hidden_states_5 = None
        hidden_states_7 = torch._C._nn.linear(
            hidden_states_6,
            l_self_modules_block_modules_0_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_6 = l_self_modules_block_modules_0_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_4 = torch.nn.functional.dropout(hidden_states_7, 0.1, False, False)
        hidden_states_7 = None
        hidden_states_8 = hidden_states_2 + dropout_4
        hidden_states_2 = dropout_4 = None
        to_6 = hidden_states_8.to(torch.float32)
        pow_3 = to_6.pow(2)
        to_6 = None
        variance_2 = pow_3.mean(-1, keepdim=True)
        pow_3 = None
        add_8 = variance_2 + 1e-06
        variance_2 = None
        rsqrt_2 = torch.rsqrt(add_8)
        add_8 = None
        hidden_states_9 = hidden_states_8 * rsqrt_2
        rsqrt_2 = None
        normed_hidden_states_1 = (
            l_self_modules_block_modules_1_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_9
        )
        l_self_modules_block_modules_1_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_9
        ) = None
        query_states_2 = torch._C._nn.linear(
            normed_hidden_states_1,
            l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_5 = query_states_2.view(1, -1, 12, 64)
        query_states_2 = None
        query_states_3 = view_5.transpose(1, 2)
        view_5 = None
        key_states_2 = torch._C._nn.linear(
            normed_hidden_states_1,
            l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_2 = torch._C._nn.linear(
            normed_hidden_states_1,
            l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_1 = l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_6 = key_states_2.view(1, -1, 12, 64)
        key_states_2 = None
        key_states_3 = view_6.transpose(1, 2)
        view_6 = None
        view_7 = value_states_2.view(1, -1, 12, 64)
        value_states_2 = None
        value_states_3 = view_7.transpose(1, 2)
        view_7 = None
        transpose_8 = key_states_3.transpose(3, 2)
        key_states_3 = None
        scores_2 = torch.matmul(query_states_3, transpose_8)
        query_states_3 = transpose_8 = None
        scores_2 += position_bias_1
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
        attn_weights_3 = value_states_3 = None
        transpose_9 = attn_output_4.transpose(1, 2)
        attn_output_4 = None
        attn_output_5 = transpose_9.contiguous()
        transpose_9 = None
        attn_output_6 = attn_output_5.view(1, -1, 768)
        attn_output_5 = None
        attn_output_7 = torch._C._nn.linear(
            attn_output_6,
            l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_6 = l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_6 = torch.nn.functional.dropout(attn_output_7, 0.1, False, False)
        attn_output_7 = None
        hidden_states_10 = hidden_states_8 + dropout_6
        hidden_states_8 = dropout_6 = None
        to_7 = hidden_states_10.to(torch.float32)
        pow_4 = to_7.pow(2)
        to_7 = None
        variance_3 = pow_4.mean(-1, keepdim=True)
        pow_4 = None
        add_10 = variance_3 + 1e-06
        variance_3 = None
        rsqrt_3 = torch.rsqrt(add_10)
        add_10 = None
        hidden_states_11 = hidden_states_10 * rsqrt_3
        rsqrt_3 = None
        forwarded_states_1 = (
            l_self_modules_block_modules_1_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_11
        )
        l_self_modules_block_modules_1_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_11
        ) = None
        hidden_states_12 = torch._C._nn.linear(
            forwarded_states_1,
            l_self_modules_block_modules_1_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_1 = l_self_modules_block_modules_1_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_13 = torch.nn.functional.relu(hidden_states_12, inplace=False)
        hidden_states_12 = None
        hidden_states_14 = torch.nn.functional.dropout(
            hidden_states_13, 0.1, False, False
        )
        hidden_states_13 = None
        hidden_states_15 = torch._C._nn.linear(
            hidden_states_14,
            l_self_modules_block_modules_1_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_14 = l_self_modules_block_modules_1_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_8 = torch.nn.functional.dropout(hidden_states_15, 0.1, False, False)
        hidden_states_15 = None
        hidden_states_16 = hidden_states_10 + dropout_8
        hidden_states_10 = dropout_8 = None
        to_8 = hidden_states_16.to(torch.float32)
        pow_5 = to_8.pow(2)
        to_8 = None
        variance_4 = pow_5.mean(-1, keepdim=True)
        pow_5 = None
        add_12 = variance_4 + 1e-06
        variance_4 = None
        rsqrt_4 = torch.rsqrt(add_12)
        add_12 = None
        hidden_states_17 = hidden_states_16 * rsqrt_4
        rsqrt_4 = None
        normed_hidden_states_2 = (
            l_self_modules_block_modules_2_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_17
        )
        l_self_modules_block_modules_2_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_17
        ) = None
        query_states_4 = torch._C._nn.linear(
            normed_hidden_states_2,
            l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_9 = query_states_4.view(1, -1, 12, 64)
        query_states_4 = None
        query_states_5 = view_9.transpose(1, 2)
        view_9 = None
        key_states_4 = torch._C._nn.linear(
            normed_hidden_states_2,
            l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_4 = torch._C._nn.linear(
            normed_hidden_states_2,
            l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_2 = l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_10 = key_states_4.view(1, -1, 12, 64)
        key_states_4 = None
        key_states_5 = view_10.transpose(1, 2)
        view_10 = None
        view_11 = value_states_4.view(1, -1, 12, 64)
        value_states_4 = None
        value_states_5 = view_11.transpose(1, 2)
        view_11 = None
        transpose_13 = key_states_5.transpose(3, 2)
        key_states_5 = None
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
        attn_weights_5 = value_states_5 = None
        transpose_14 = attn_output_8.transpose(1, 2)
        attn_output_8 = None
        attn_output_9 = transpose_14.contiguous()
        transpose_14 = None
        attn_output_10 = attn_output_9.view(1, -1, 768)
        attn_output_9 = None
        attn_output_11 = torch._C._nn.linear(
            attn_output_10,
            l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_10 = l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_10 = torch.nn.functional.dropout(attn_output_11, 0.1, False, False)
        attn_output_11 = None
        hidden_states_18 = hidden_states_16 + dropout_10
        hidden_states_16 = dropout_10 = None
        to_9 = hidden_states_18.to(torch.float32)
        pow_6 = to_9.pow(2)
        to_9 = None
        variance_5 = pow_6.mean(-1, keepdim=True)
        pow_6 = None
        add_14 = variance_5 + 1e-06
        variance_5 = None
        rsqrt_5 = torch.rsqrt(add_14)
        add_14 = None
        hidden_states_19 = hidden_states_18 * rsqrt_5
        rsqrt_5 = None
        forwarded_states_2 = (
            l_self_modules_block_modules_2_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_19
        )
        l_self_modules_block_modules_2_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_19
        ) = None
        hidden_states_20 = torch._C._nn.linear(
            forwarded_states_2,
            l_self_modules_block_modules_2_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_2 = l_self_modules_block_modules_2_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_21 = torch.nn.functional.relu(hidden_states_20, inplace=False)
        hidden_states_20 = None
        hidden_states_22 = torch.nn.functional.dropout(
            hidden_states_21, 0.1, False, False
        )
        hidden_states_21 = None
        hidden_states_23 = torch._C._nn.linear(
            hidden_states_22,
            l_self_modules_block_modules_2_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_22 = l_self_modules_block_modules_2_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_12 = torch.nn.functional.dropout(hidden_states_23, 0.1, False, False)
        hidden_states_23 = None
        hidden_states_24 = hidden_states_18 + dropout_12
        hidden_states_18 = dropout_12 = None
        to_10 = hidden_states_24.to(torch.float32)
        pow_7 = to_10.pow(2)
        to_10 = None
        variance_6 = pow_7.mean(-1, keepdim=True)
        pow_7 = None
        add_16 = variance_6 + 1e-06
        variance_6 = None
        rsqrt_6 = torch.rsqrt(add_16)
        add_16 = None
        hidden_states_25 = hidden_states_24 * rsqrt_6
        rsqrt_6 = None
        normed_hidden_states_3 = (
            l_self_modules_block_modules_3_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_25
        )
        l_self_modules_block_modules_3_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_25
        ) = None
        query_states_6 = torch._C._nn.linear(
            normed_hidden_states_3,
            l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_13 = query_states_6.view(1, -1, 12, 64)
        query_states_6 = None
        query_states_7 = view_13.transpose(1, 2)
        view_13 = None
        key_states_6 = torch._C._nn.linear(
            normed_hidden_states_3,
            l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_6 = torch._C._nn.linear(
            normed_hidden_states_3,
            l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_3 = l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_14 = key_states_6.view(1, -1, 12, 64)
        key_states_6 = None
        key_states_7 = view_14.transpose(1, 2)
        view_14 = None
        view_15 = value_states_6.view(1, -1, 12, 64)
        value_states_6 = None
        value_states_7 = view_15.transpose(1, 2)
        view_15 = None
        transpose_18 = key_states_7.transpose(3, 2)
        key_states_7 = None
        scores_6 = torch.matmul(query_states_7, transpose_18)
        query_states_7 = transpose_18 = None
        scores_6 += position_bias_1
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
        attn_weights_7 = value_states_7 = None
        transpose_19 = attn_output_12.transpose(1, 2)
        attn_output_12 = None
        attn_output_13 = transpose_19.contiguous()
        transpose_19 = None
        attn_output_14 = attn_output_13.view(1, -1, 768)
        attn_output_13 = None
        attn_output_15 = torch._C._nn.linear(
            attn_output_14,
            l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_14 = l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_14 = torch.nn.functional.dropout(attn_output_15, 0.1, False, False)
        attn_output_15 = None
        hidden_states_26 = hidden_states_24 + dropout_14
        hidden_states_24 = dropout_14 = None
        to_11 = hidden_states_26.to(torch.float32)
        pow_8 = to_11.pow(2)
        to_11 = None
        variance_7 = pow_8.mean(-1, keepdim=True)
        pow_8 = None
        add_18 = variance_7 + 1e-06
        variance_7 = None
        rsqrt_7 = torch.rsqrt(add_18)
        add_18 = None
        hidden_states_27 = hidden_states_26 * rsqrt_7
        rsqrt_7 = None
        forwarded_states_3 = (
            l_self_modules_block_modules_3_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_27
        )
        l_self_modules_block_modules_3_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_27
        ) = None
        hidden_states_28 = torch._C._nn.linear(
            forwarded_states_3,
            l_self_modules_block_modules_3_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_3 = l_self_modules_block_modules_3_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_29 = torch.nn.functional.relu(hidden_states_28, inplace=False)
        hidden_states_28 = None
        hidden_states_30 = torch.nn.functional.dropout(
            hidden_states_29, 0.1, False, False
        )
        hidden_states_29 = None
        hidden_states_31 = torch._C._nn.linear(
            hidden_states_30,
            l_self_modules_block_modules_3_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_30 = l_self_modules_block_modules_3_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_16 = torch.nn.functional.dropout(hidden_states_31, 0.1, False, False)
        hidden_states_31 = None
        hidden_states_32 = hidden_states_26 + dropout_16
        hidden_states_26 = dropout_16 = None
        to_12 = hidden_states_32.to(torch.float32)
        pow_9 = to_12.pow(2)
        to_12 = None
        variance_8 = pow_9.mean(-1, keepdim=True)
        pow_9 = None
        add_20 = variance_8 + 1e-06
        variance_8 = None
        rsqrt_8 = torch.rsqrt(add_20)
        add_20 = None
        hidden_states_33 = hidden_states_32 * rsqrt_8
        rsqrt_8 = None
        normed_hidden_states_4 = (
            l_self_modules_block_modules_4_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_33
        )
        l_self_modules_block_modules_4_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_33
        ) = None
        query_states_8 = torch._C._nn.linear(
            normed_hidden_states_4,
            l_self_modules_block_modules_4_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_4_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_17 = query_states_8.view(1, -1, 12, 64)
        query_states_8 = None
        query_states_9 = view_17.transpose(1, 2)
        view_17 = None
        key_states_8 = torch._C._nn.linear(
            normed_hidden_states_4,
            l_self_modules_block_modules_4_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_4_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_8 = torch._C._nn.linear(
            normed_hidden_states_4,
            l_self_modules_block_modules_4_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_4 = l_self_modules_block_modules_4_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_18 = key_states_8.view(1, -1, 12, 64)
        key_states_8 = None
        key_states_9 = view_18.transpose(1, 2)
        view_18 = None
        view_19 = value_states_8.view(1, -1, 12, 64)
        value_states_8 = None
        value_states_9 = view_19.transpose(1, 2)
        view_19 = None
        transpose_23 = key_states_9.transpose(3, 2)
        key_states_9 = None
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
        attn_weights_9 = value_states_9 = None
        transpose_24 = attn_output_16.transpose(1, 2)
        attn_output_16 = None
        attn_output_17 = transpose_24.contiguous()
        transpose_24 = None
        attn_output_18 = attn_output_17.view(1, -1, 768)
        attn_output_17 = None
        attn_output_19 = torch._C._nn.linear(
            attn_output_18,
            l_self_modules_block_modules_4_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_18 = l_self_modules_block_modules_4_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_18 = torch.nn.functional.dropout(attn_output_19, 0.1, False, False)
        attn_output_19 = None
        hidden_states_34 = hidden_states_32 + dropout_18
        hidden_states_32 = dropout_18 = None
        to_13 = hidden_states_34.to(torch.float32)
        pow_10 = to_13.pow(2)
        to_13 = None
        variance_9 = pow_10.mean(-1, keepdim=True)
        pow_10 = None
        add_22 = variance_9 + 1e-06
        variance_9 = None
        rsqrt_9 = torch.rsqrt(add_22)
        add_22 = None
        hidden_states_35 = hidden_states_34 * rsqrt_9
        rsqrt_9 = None
        forwarded_states_4 = (
            l_self_modules_block_modules_4_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_35
        )
        l_self_modules_block_modules_4_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_35
        ) = None
        hidden_states_36 = torch._C._nn.linear(
            forwarded_states_4,
            l_self_modules_block_modules_4_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_4 = l_self_modules_block_modules_4_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_37 = torch.nn.functional.relu(hidden_states_36, inplace=False)
        hidden_states_36 = None
        hidden_states_38 = torch.nn.functional.dropout(
            hidden_states_37, 0.1, False, False
        )
        hidden_states_37 = None
        hidden_states_39 = torch._C._nn.linear(
            hidden_states_38,
            l_self_modules_block_modules_4_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_38 = l_self_modules_block_modules_4_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_20 = torch.nn.functional.dropout(hidden_states_39, 0.1, False, False)
        hidden_states_39 = None
        hidden_states_40 = hidden_states_34 + dropout_20
        hidden_states_34 = dropout_20 = None
        to_14 = hidden_states_40.to(torch.float32)
        pow_11 = to_14.pow(2)
        to_14 = None
        variance_10 = pow_11.mean(-1, keepdim=True)
        pow_11 = None
        add_24 = variance_10 + 1e-06
        variance_10 = None
        rsqrt_10 = torch.rsqrt(add_24)
        add_24 = None
        hidden_states_41 = hidden_states_40 * rsqrt_10
        rsqrt_10 = None
        normed_hidden_states_5 = (
            l_self_modules_block_modules_5_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_41
        )
        l_self_modules_block_modules_5_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_41
        ) = None
        query_states_10 = torch._C._nn.linear(
            normed_hidden_states_5,
            l_self_modules_block_modules_5_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_5_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_21 = query_states_10.view(1, -1, 12, 64)
        query_states_10 = None
        query_states_11 = view_21.transpose(1, 2)
        view_21 = None
        key_states_10 = torch._C._nn.linear(
            normed_hidden_states_5,
            l_self_modules_block_modules_5_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_5_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_10 = torch._C._nn.linear(
            normed_hidden_states_5,
            l_self_modules_block_modules_5_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_5 = l_self_modules_block_modules_5_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_22 = key_states_10.view(1, -1, 12, 64)
        key_states_10 = None
        key_states_11 = view_22.transpose(1, 2)
        view_22 = None
        view_23 = value_states_10.view(1, -1, 12, 64)
        value_states_10 = None
        value_states_11 = view_23.transpose(1, 2)
        view_23 = None
        transpose_28 = key_states_11.transpose(3, 2)
        key_states_11 = None
        scores_10 = torch.matmul(query_states_11, transpose_28)
        query_states_11 = transpose_28 = None
        scores_10 += position_bias_1
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
        attn_weights_11 = value_states_11 = None
        transpose_29 = attn_output_20.transpose(1, 2)
        attn_output_20 = None
        attn_output_21 = transpose_29.contiguous()
        transpose_29 = None
        attn_output_22 = attn_output_21.view(1, -1, 768)
        attn_output_21 = None
        attn_output_23 = torch._C._nn.linear(
            attn_output_22,
            l_self_modules_block_modules_5_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_22 = l_self_modules_block_modules_5_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_22 = torch.nn.functional.dropout(attn_output_23, 0.1, False, False)
        attn_output_23 = None
        hidden_states_42 = hidden_states_40 + dropout_22
        hidden_states_40 = dropout_22 = None
        to_15 = hidden_states_42.to(torch.float32)
        pow_12 = to_15.pow(2)
        to_15 = None
        variance_11 = pow_12.mean(-1, keepdim=True)
        pow_12 = None
        add_26 = variance_11 + 1e-06
        variance_11 = None
        rsqrt_11 = torch.rsqrt(add_26)
        add_26 = None
        hidden_states_43 = hidden_states_42 * rsqrt_11
        rsqrt_11 = None
        forwarded_states_5 = (
            l_self_modules_block_modules_5_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_43
        )
        l_self_modules_block_modules_5_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_43
        ) = None
        hidden_states_44 = torch._C._nn.linear(
            forwarded_states_5,
            l_self_modules_block_modules_5_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_5 = l_self_modules_block_modules_5_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_45 = torch.nn.functional.relu(hidden_states_44, inplace=False)
        hidden_states_44 = None
        hidden_states_46 = torch.nn.functional.dropout(
            hidden_states_45, 0.1, False, False
        )
        hidden_states_45 = None
        hidden_states_47 = torch._C._nn.linear(
            hidden_states_46,
            l_self_modules_block_modules_5_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_46 = l_self_modules_block_modules_5_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_24 = torch.nn.functional.dropout(hidden_states_47, 0.1, False, False)
        hidden_states_47 = None
        hidden_states_48 = hidden_states_42 + dropout_24
        hidden_states_42 = dropout_24 = None
        to_16 = hidden_states_48.to(torch.float32)
        pow_13 = to_16.pow(2)
        to_16 = None
        variance_12 = pow_13.mean(-1, keepdim=True)
        pow_13 = None
        add_28 = variance_12 + 1e-06
        variance_12 = None
        rsqrt_12 = torch.rsqrt(add_28)
        add_28 = None
        hidden_states_49 = hidden_states_48 * rsqrt_12
        rsqrt_12 = None
        normed_hidden_states_6 = (
            l_self_modules_block_modules_6_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_49
        )
        l_self_modules_block_modules_6_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_49
        ) = None
        query_states_12 = torch._C._nn.linear(
            normed_hidden_states_6,
            l_self_modules_block_modules_6_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_6_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_25 = query_states_12.view(1, -1, 12, 64)
        query_states_12 = None
        query_states_13 = view_25.transpose(1, 2)
        view_25 = None
        key_states_12 = torch._C._nn.linear(
            normed_hidden_states_6,
            l_self_modules_block_modules_6_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_6_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_12 = torch._C._nn.linear(
            normed_hidden_states_6,
            l_self_modules_block_modules_6_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_6 = l_self_modules_block_modules_6_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_26 = key_states_12.view(1, -1, 12, 64)
        key_states_12 = None
        key_states_13 = view_26.transpose(1, 2)
        view_26 = None
        view_27 = value_states_12.view(1, -1, 12, 64)
        value_states_12 = None
        value_states_13 = view_27.transpose(1, 2)
        view_27 = None
        transpose_33 = key_states_13.transpose(3, 2)
        key_states_13 = None
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
        attn_weights_13 = value_states_13 = None
        transpose_34 = attn_output_24.transpose(1, 2)
        attn_output_24 = None
        attn_output_25 = transpose_34.contiguous()
        transpose_34 = None
        attn_output_26 = attn_output_25.view(1, -1, 768)
        attn_output_25 = None
        attn_output_27 = torch._C._nn.linear(
            attn_output_26,
            l_self_modules_block_modules_6_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_26 = l_self_modules_block_modules_6_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_26 = torch.nn.functional.dropout(attn_output_27, 0.1, False, False)
        attn_output_27 = None
        hidden_states_50 = hidden_states_48 + dropout_26
        hidden_states_48 = dropout_26 = None
        to_17 = hidden_states_50.to(torch.float32)
        pow_14 = to_17.pow(2)
        to_17 = None
        variance_13 = pow_14.mean(-1, keepdim=True)
        pow_14 = None
        add_30 = variance_13 + 1e-06
        variance_13 = None
        rsqrt_13 = torch.rsqrt(add_30)
        add_30 = None
        hidden_states_51 = hidden_states_50 * rsqrt_13
        rsqrt_13 = None
        forwarded_states_6 = (
            l_self_modules_block_modules_6_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_51
        )
        l_self_modules_block_modules_6_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_51
        ) = None
        hidden_states_52 = torch._C._nn.linear(
            forwarded_states_6,
            l_self_modules_block_modules_6_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_6 = l_self_modules_block_modules_6_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_53 = torch.nn.functional.relu(hidden_states_52, inplace=False)
        hidden_states_52 = None
        hidden_states_54 = torch.nn.functional.dropout(
            hidden_states_53, 0.1, False, False
        )
        hidden_states_53 = None
        hidden_states_55 = torch._C._nn.linear(
            hidden_states_54,
            l_self_modules_block_modules_6_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_54 = l_self_modules_block_modules_6_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_28 = torch.nn.functional.dropout(hidden_states_55, 0.1, False, False)
        hidden_states_55 = None
        hidden_states_56 = hidden_states_50 + dropout_28
        hidden_states_50 = dropout_28 = None
        to_18 = hidden_states_56.to(torch.float32)
        pow_15 = to_18.pow(2)
        to_18 = None
        variance_14 = pow_15.mean(-1, keepdim=True)
        pow_15 = None
        add_32 = variance_14 + 1e-06
        variance_14 = None
        rsqrt_14 = torch.rsqrt(add_32)
        add_32 = None
        hidden_states_57 = hidden_states_56 * rsqrt_14
        rsqrt_14 = None
        normed_hidden_states_7 = (
            l_self_modules_block_modules_7_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_57
        )
        l_self_modules_block_modules_7_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_57
        ) = None
        query_states_14 = torch._C._nn.linear(
            normed_hidden_states_7,
            l_self_modules_block_modules_7_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_7_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_29 = query_states_14.view(1, -1, 12, 64)
        query_states_14 = None
        query_states_15 = view_29.transpose(1, 2)
        view_29 = None
        key_states_14 = torch._C._nn.linear(
            normed_hidden_states_7,
            l_self_modules_block_modules_7_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_7_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_14 = torch._C._nn.linear(
            normed_hidden_states_7,
            l_self_modules_block_modules_7_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_7 = l_self_modules_block_modules_7_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_30 = key_states_14.view(1, -1, 12, 64)
        key_states_14 = None
        key_states_15 = view_30.transpose(1, 2)
        view_30 = None
        view_31 = value_states_14.view(1, -1, 12, 64)
        value_states_14 = None
        value_states_15 = view_31.transpose(1, 2)
        view_31 = None
        transpose_38 = key_states_15.transpose(3, 2)
        key_states_15 = None
        scores_14 = torch.matmul(query_states_15, transpose_38)
        query_states_15 = transpose_38 = None
        scores_14 += position_bias_1
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
        attn_weights_15 = value_states_15 = None
        transpose_39 = attn_output_28.transpose(1, 2)
        attn_output_28 = None
        attn_output_29 = transpose_39.contiguous()
        transpose_39 = None
        attn_output_30 = attn_output_29.view(1, -1, 768)
        attn_output_29 = None
        attn_output_31 = torch._C._nn.linear(
            attn_output_30,
            l_self_modules_block_modules_7_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_30 = l_self_modules_block_modules_7_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_30 = torch.nn.functional.dropout(attn_output_31, 0.1, False, False)
        attn_output_31 = None
        hidden_states_58 = hidden_states_56 + dropout_30
        hidden_states_56 = dropout_30 = None
        to_19 = hidden_states_58.to(torch.float32)
        pow_16 = to_19.pow(2)
        to_19 = None
        variance_15 = pow_16.mean(-1, keepdim=True)
        pow_16 = None
        add_34 = variance_15 + 1e-06
        variance_15 = None
        rsqrt_15 = torch.rsqrt(add_34)
        add_34 = None
        hidden_states_59 = hidden_states_58 * rsqrt_15
        rsqrt_15 = None
        forwarded_states_7 = (
            l_self_modules_block_modules_7_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_59
        )
        l_self_modules_block_modules_7_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_59
        ) = None
        hidden_states_60 = torch._C._nn.linear(
            forwarded_states_7,
            l_self_modules_block_modules_7_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_7 = l_self_modules_block_modules_7_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_61 = torch.nn.functional.relu(hidden_states_60, inplace=False)
        hidden_states_60 = None
        hidden_states_62 = torch.nn.functional.dropout(
            hidden_states_61, 0.1, False, False
        )
        hidden_states_61 = None
        hidden_states_63 = torch._C._nn.linear(
            hidden_states_62,
            l_self_modules_block_modules_7_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_62 = l_self_modules_block_modules_7_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_32 = torch.nn.functional.dropout(hidden_states_63, 0.1, False, False)
        hidden_states_63 = None
        hidden_states_64 = hidden_states_58 + dropout_32
        hidden_states_58 = dropout_32 = None
        to_20 = hidden_states_64.to(torch.float32)
        pow_17 = to_20.pow(2)
        to_20 = None
        variance_16 = pow_17.mean(-1, keepdim=True)
        pow_17 = None
        add_36 = variance_16 + 1e-06
        variance_16 = None
        rsqrt_16 = torch.rsqrt(add_36)
        add_36 = None
        hidden_states_65 = hidden_states_64 * rsqrt_16
        rsqrt_16 = None
        normed_hidden_states_8 = (
            l_self_modules_block_modules_8_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_65
        )
        l_self_modules_block_modules_8_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_65
        ) = None
        query_states_16 = torch._C._nn.linear(
            normed_hidden_states_8,
            l_self_modules_block_modules_8_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_8_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_33 = query_states_16.view(1, -1, 12, 64)
        query_states_16 = None
        query_states_17 = view_33.transpose(1, 2)
        view_33 = None
        key_states_16 = torch._C._nn.linear(
            normed_hidden_states_8,
            l_self_modules_block_modules_8_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_8_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_16 = torch._C._nn.linear(
            normed_hidden_states_8,
            l_self_modules_block_modules_8_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_8 = l_self_modules_block_modules_8_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_34 = key_states_16.view(1, -1, 12, 64)
        key_states_16 = None
        key_states_17 = view_34.transpose(1, 2)
        view_34 = None
        view_35 = value_states_16.view(1, -1, 12, 64)
        value_states_16 = None
        value_states_17 = view_35.transpose(1, 2)
        view_35 = None
        transpose_43 = key_states_17.transpose(3, 2)
        key_states_17 = None
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
        attn_weights_17 = value_states_17 = None
        transpose_44 = attn_output_32.transpose(1, 2)
        attn_output_32 = None
        attn_output_33 = transpose_44.contiguous()
        transpose_44 = None
        attn_output_34 = attn_output_33.view(1, -1, 768)
        attn_output_33 = None
        attn_output_35 = torch._C._nn.linear(
            attn_output_34,
            l_self_modules_block_modules_8_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_34 = l_self_modules_block_modules_8_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_34 = torch.nn.functional.dropout(attn_output_35, 0.1, False, False)
        attn_output_35 = None
        hidden_states_66 = hidden_states_64 + dropout_34
        hidden_states_64 = dropout_34 = None
        to_21 = hidden_states_66.to(torch.float32)
        pow_18 = to_21.pow(2)
        to_21 = None
        variance_17 = pow_18.mean(-1, keepdim=True)
        pow_18 = None
        add_38 = variance_17 + 1e-06
        variance_17 = None
        rsqrt_17 = torch.rsqrt(add_38)
        add_38 = None
        hidden_states_67 = hidden_states_66 * rsqrt_17
        rsqrt_17 = None
        forwarded_states_8 = (
            l_self_modules_block_modules_8_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_67
        )
        l_self_modules_block_modules_8_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_67
        ) = None
        hidden_states_68 = torch._C._nn.linear(
            forwarded_states_8,
            l_self_modules_block_modules_8_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_8 = l_self_modules_block_modules_8_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_69 = torch.nn.functional.relu(hidden_states_68, inplace=False)
        hidden_states_68 = None
        hidden_states_70 = torch.nn.functional.dropout(
            hidden_states_69, 0.1, False, False
        )
        hidden_states_69 = None
        hidden_states_71 = torch._C._nn.linear(
            hidden_states_70,
            l_self_modules_block_modules_8_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_70 = l_self_modules_block_modules_8_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_36 = torch.nn.functional.dropout(hidden_states_71, 0.1, False, False)
        hidden_states_71 = None
        hidden_states_72 = hidden_states_66 + dropout_36
        hidden_states_66 = dropout_36 = None
        to_22 = hidden_states_72.to(torch.float32)
        pow_19 = to_22.pow(2)
        to_22 = None
        variance_18 = pow_19.mean(-1, keepdim=True)
        pow_19 = None
        add_40 = variance_18 + 1e-06
        variance_18 = None
        rsqrt_18 = torch.rsqrt(add_40)
        add_40 = None
        hidden_states_73 = hidden_states_72 * rsqrt_18
        rsqrt_18 = None
        normed_hidden_states_9 = (
            l_self_modules_block_modules_9_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_73
        )
        l_self_modules_block_modules_9_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_73
        ) = None
        query_states_18 = torch._C._nn.linear(
            normed_hidden_states_9,
            l_self_modules_block_modules_9_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_9_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_37 = query_states_18.view(1, -1, 12, 64)
        query_states_18 = None
        query_states_19 = view_37.transpose(1, 2)
        view_37 = None
        key_states_18 = torch._C._nn.linear(
            normed_hidden_states_9,
            l_self_modules_block_modules_9_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_9_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_18 = torch._C._nn.linear(
            normed_hidden_states_9,
            l_self_modules_block_modules_9_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_9 = l_self_modules_block_modules_9_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_38 = key_states_18.view(1, -1, 12, 64)
        key_states_18 = None
        key_states_19 = view_38.transpose(1, 2)
        view_38 = None
        view_39 = value_states_18.view(1, -1, 12, 64)
        value_states_18 = None
        value_states_19 = view_39.transpose(1, 2)
        view_39 = None
        transpose_48 = key_states_19.transpose(3, 2)
        key_states_19 = None
        scores_18 = torch.matmul(query_states_19, transpose_48)
        query_states_19 = transpose_48 = None
        scores_18 += position_bias_1
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
        attn_weights_19 = value_states_19 = None
        transpose_49 = attn_output_36.transpose(1, 2)
        attn_output_36 = None
        attn_output_37 = transpose_49.contiguous()
        transpose_49 = None
        attn_output_38 = attn_output_37.view(1, -1, 768)
        attn_output_37 = None
        attn_output_39 = torch._C._nn.linear(
            attn_output_38,
            l_self_modules_block_modules_9_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_38 = l_self_modules_block_modules_9_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_38 = torch.nn.functional.dropout(attn_output_39, 0.1, False, False)
        attn_output_39 = None
        hidden_states_74 = hidden_states_72 + dropout_38
        hidden_states_72 = dropout_38 = None
        to_23 = hidden_states_74.to(torch.float32)
        pow_20 = to_23.pow(2)
        to_23 = None
        variance_19 = pow_20.mean(-1, keepdim=True)
        pow_20 = None
        add_42 = variance_19 + 1e-06
        variance_19 = None
        rsqrt_19 = torch.rsqrt(add_42)
        add_42 = None
        hidden_states_75 = hidden_states_74 * rsqrt_19
        rsqrt_19 = None
        forwarded_states_9 = (
            l_self_modules_block_modules_9_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_75
        )
        l_self_modules_block_modules_9_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_75
        ) = None
        hidden_states_76 = torch._C._nn.linear(
            forwarded_states_9,
            l_self_modules_block_modules_9_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_9 = l_self_modules_block_modules_9_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_77 = torch.nn.functional.relu(hidden_states_76, inplace=False)
        hidden_states_76 = None
        hidden_states_78 = torch.nn.functional.dropout(
            hidden_states_77, 0.1, False, False
        )
        hidden_states_77 = None
        hidden_states_79 = torch._C._nn.linear(
            hidden_states_78,
            l_self_modules_block_modules_9_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_78 = l_self_modules_block_modules_9_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_40 = torch.nn.functional.dropout(hidden_states_79, 0.1, False, False)
        hidden_states_79 = None
        hidden_states_80 = hidden_states_74 + dropout_40
        hidden_states_74 = dropout_40 = None
        to_24 = hidden_states_80.to(torch.float32)
        pow_21 = to_24.pow(2)
        to_24 = None
        variance_20 = pow_21.mean(-1, keepdim=True)
        pow_21 = None
        add_44 = variance_20 + 1e-06
        variance_20 = None
        rsqrt_20 = torch.rsqrt(add_44)
        add_44 = None
        hidden_states_81 = hidden_states_80 * rsqrt_20
        rsqrt_20 = None
        normed_hidden_states_10 = (
            l_self_modules_block_modules_10_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_81
        )
        l_self_modules_block_modules_10_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_81
        ) = None
        query_states_20 = torch._C._nn.linear(
            normed_hidden_states_10,
            l_self_modules_block_modules_10_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_10_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_41 = query_states_20.view(1, -1, 12, 64)
        query_states_20 = None
        query_states_21 = view_41.transpose(1, 2)
        view_41 = None
        key_states_20 = torch._C._nn.linear(
            normed_hidden_states_10,
            l_self_modules_block_modules_10_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_10_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_20 = torch._C._nn.linear(
            normed_hidden_states_10,
            l_self_modules_block_modules_10_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_10 = l_self_modules_block_modules_10_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_42 = key_states_20.view(1, -1, 12, 64)
        key_states_20 = None
        key_states_21 = view_42.transpose(1, 2)
        view_42 = None
        view_43 = value_states_20.view(1, -1, 12, 64)
        value_states_20 = None
        value_states_21 = view_43.transpose(1, 2)
        view_43 = None
        transpose_53 = key_states_21.transpose(3, 2)
        key_states_21 = None
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
        attn_weights_21 = value_states_21 = None
        transpose_54 = attn_output_40.transpose(1, 2)
        attn_output_40 = None
        attn_output_41 = transpose_54.contiguous()
        transpose_54 = None
        attn_output_42 = attn_output_41.view(1, -1, 768)
        attn_output_41 = None
        attn_output_43 = torch._C._nn.linear(
            attn_output_42,
            l_self_modules_block_modules_10_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_42 = l_self_modules_block_modules_10_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_42 = torch.nn.functional.dropout(attn_output_43, 0.1, False, False)
        attn_output_43 = None
        hidden_states_82 = hidden_states_80 + dropout_42
        hidden_states_80 = dropout_42 = None
        to_25 = hidden_states_82.to(torch.float32)
        pow_22 = to_25.pow(2)
        to_25 = None
        variance_21 = pow_22.mean(-1, keepdim=True)
        pow_22 = None
        add_46 = variance_21 + 1e-06
        variance_21 = None
        rsqrt_21 = torch.rsqrt(add_46)
        add_46 = None
        hidden_states_83 = hidden_states_82 * rsqrt_21
        rsqrt_21 = None
        forwarded_states_10 = (
            l_self_modules_block_modules_10_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_83
        )
        l_self_modules_block_modules_10_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_83
        ) = None
        hidden_states_84 = torch._C._nn.linear(
            forwarded_states_10,
            l_self_modules_block_modules_10_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_10 = l_self_modules_block_modules_10_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_85 = torch.nn.functional.relu(hidden_states_84, inplace=False)
        hidden_states_84 = None
        hidden_states_86 = torch.nn.functional.dropout(
            hidden_states_85, 0.1, False, False
        )
        hidden_states_85 = None
        hidden_states_87 = torch._C._nn.linear(
            hidden_states_86,
            l_self_modules_block_modules_10_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_86 = l_self_modules_block_modules_10_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_44 = torch.nn.functional.dropout(hidden_states_87, 0.1, False, False)
        hidden_states_87 = None
        hidden_states_88 = hidden_states_82 + dropout_44
        hidden_states_82 = dropout_44 = None
        to_26 = hidden_states_88.to(torch.float32)
        pow_23 = to_26.pow(2)
        to_26 = None
        variance_22 = pow_23.mean(-1, keepdim=True)
        pow_23 = None
        add_48 = variance_22 + 1e-06
        variance_22 = None
        rsqrt_22 = torch.rsqrt(add_48)
        add_48 = None
        hidden_states_89 = hidden_states_88 * rsqrt_22
        rsqrt_22 = None
        normed_hidden_states_11 = (
            l_self_modules_block_modules_11_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_89
        )
        l_self_modules_block_modules_11_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_89
        ) = None
        query_states_22 = torch._C._nn.linear(
            normed_hidden_states_11,
            l_self_modules_block_modules_11_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_11_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_45 = query_states_22.view(1, -1, 12, 64)
        query_states_22 = None
        query_states_23 = view_45.transpose(1, 2)
        view_45 = None
        key_states_22 = torch._C._nn.linear(
            normed_hidden_states_11,
            l_self_modules_block_modules_11_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_11_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_22 = torch._C._nn.linear(
            normed_hidden_states_11,
            l_self_modules_block_modules_11_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_11 = l_self_modules_block_modules_11_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_46 = key_states_22.view(1, -1, 12, 64)
        key_states_22 = None
        key_states_23 = view_46.transpose(1, 2)
        view_46 = None
        view_47 = value_states_22.view(1, -1, 12, 64)
        value_states_22 = None
        value_states_23 = view_47.transpose(1, 2)
        view_47 = None
        transpose_58 = key_states_23.transpose(3, 2)
        key_states_23 = None
        scores_22 = torch.matmul(query_states_23, transpose_58)
        query_states_23 = transpose_58 = None
        scores_22 += position_bias_1
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
        attn_weights_23 = value_states_23 = None
        transpose_59 = attn_output_44.transpose(1, 2)
        attn_output_44 = None
        attn_output_45 = transpose_59.contiguous()
        transpose_59 = None
        attn_output_46 = attn_output_45.view(1, -1, 768)
        attn_output_45 = None
        attn_output_47 = torch._C._nn.linear(
            attn_output_46,
            l_self_modules_block_modules_11_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_46 = l_self_modules_block_modules_11_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_46 = torch.nn.functional.dropout(attn_output_47, 0.1, False, False)
        attn_output_47 = None
        hidden_states_90 = hidden_states_88 + dropout_46
        hidden_states_88 = dropout_46 = None
        to_27 = hidden_states_90.to(torch.float32)
        pow_24 = to_27.pow(2)
        to_27 = None
        variance_23 = pow_24.mean(-1, keepdim=True)
        pow_24 = None
        add_50 = variance_23 + 1e-06
        variance_23 = None
        rsqrt_23 = torch.rsqrt(add_50)
        add_50 = None
        hidden_states_91 = hidden_states_90 * rsqrt_23
        rsqrt_23 = None
        forwarded_states_11 = (
            l_self_modules_block_modules_11_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_91
        )
        l_self_modules_block_modules_11_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_91
        ) = None
        hidden_states_92 = torch._C._nn.linear(
            forwarded_states_11,
            l_self_modules_block_modules_11_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_11 = l_self_modules_block_modules_11_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_93 = torch.nn.functional.relu(hidden_states_92, inplace=False)
        hidden_states_92 = None
        hidden_states_94 = torch.nn.functional.dropout(
            hidden_states_93, 0.1, False, False
        )
        hidden_states_93 = None
        hidden_states_95 = torch._C._nn.linear(
            hidden_states_94,
            l_self_modules_block_modules_11_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_94 = l_self_modules_block_modules_11_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_48 = torch.nn.functional.dropout(hidden_states_95, 0.1, False, False)
        hidden_states_95 = None
        hidden_states_96 = hidden_states_90 + dropout_48
        hidden_states_90 = dropout_48 = None
        to_28 = hidden_states_96.to(torch.float32)
        pow_25 = to_28.pow(2)
        to_28 = None
        variance_24 = pow_25.mean(-1, keepdim=True)
        pow_25 = None
        add_52 = variance_24 + 1e-06
        variance_24 = None
        rsqrt_24 = torch.rsqrt(add_52)
        add_52 = None
        hidden_states_97 = hidden_states_96 * rsqrt_24
        rsqrt_24 = None
        normed_hidden_states_12 = (
            l_self_modules_block_modules_12_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_97
        )
        l_self_modules_block_modules_12_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_97
        ) = None
        query_states_24 = torch._C._nn.linear(
            normed_hidden_states_12,
            l_self_modules_block_modules_12_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_12_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_49 = query_states_24.view(1, -1, 12, 64)
        query_states_24 = None
        query_states_25 = view_49.transpose(1, 2)
        view_49 = None
        key_states_24 = torch._C._nn.linear(
            normed_hidden_states_12,
            l_self_modules_block_modules_12_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_12_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_24 = torch._C._nn.linear(
            normed_hidden_states_12,
            l_self_modules_block_modules_12_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_12 = l_self_modules_block_modules_12_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_50 = key_states_24.view(1, -1, 12, 64)
        key_states_24 = None
        key_states_25 = view_50.transpose(1, 2)
        view_50 = None
        view_51 = value_states_24.view(1, -1, 12, 64)
        value_states_24 = None
        value_states_25 = view_51.transpose(1, 2)
        view_51 = None
        transpose_63 = key_states_25.transpose(3, 2)
        key_states_25 = None
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
        attn_weights_25 = value_states_25 = None
        transpose_64 = attn_output_48.transpose(1, 2)
        attn_output_48 = None
        attn_output_49 = transpose_64.contiguous()
        transpose_64 = None
        attn_output_50 = attn_output_49.view(1, -1, 768)
        attn_output_49 = None
        attn_output_51 = torch._C._nn.linear(
            attn_output_50,
            l_self_modules_block_modules_12_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_50 = l_self_modules_block_modules_12_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_50 = torch.nn.functional.dropout(attn_output_51, 0.1, False, False)
        attn_output_51 = None
        hidden_states_98 = hidden_states_96 + dropout_50
        hidden_states_96 = dropout_50 = None
        to_29 = hidden_states_98.to(torch.float32)
        pow_26 = to_29.pow(2)
        to_29 = None
        variance_25 = pow_26.mean(-1, keepdim=True)
        pow_26 = None
        add_54 = variance_25 + 1e-06
        variance_25 = None
        rsqrt_25 = torch.rsqrt(add_54)
        add_54 = None
        hidden_states_99 = hidden_states_98 * rsqrt_25
        rsqrt_25 = None
        forwarded_states_12 = (
            l_self_modules_block_modules_12_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_99
        )
        l_self_modules_block_modules_12_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_99
        ) = None
        hidden_states_100 = torch._C._nn.linear(
            forwarded_states_12,
            l_self_modules_block_modules_12_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_12 = l_self_modules_block_modules_12_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_101 = torch.nn.functional.relu(hidden_states_100, inplace=False)
        hidden_states_100 = None
        hidden_states_102 = torch.nn.functional.dropout(
            hidden_states_101, 0.1, False, False
        )
        hidden_states_101 = None
        hidden_states_103 = torch._C._nn.linear(
            hidden_states_102,
            l_self_modules_block_modules_12_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_102 = l_self_modules_block_modules_12_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_52 = torch.nn.functional.dropout(hidden_states_103, 0.1, False, False)
        hidden_states_103 = None
        hidden_states_104 = hidden_states_98 + dropout_52
        hidden_states_98 = dropout_52 = None
        to_30 = hidden_states_104.to(torch.float32)
        pow_27 = to_30.pow(2)
        to_30 = None
        variance_26 = pow_27.mean(-1, keepdim=True)
        pow_27 = None
        add_56 = variance_26 + 1e-06
        variance_26 = None
        rsqrt_26 = torch.rsqrt(add_56)
        add_56 = None
        hidden_states_105 = hidden_states_104 * rsqrt_26
        rsqrt_26 = None
        normed_hidden_states_13 = (
            l_self_modules_block_modules_13_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_105
        )
        l_self_modules_block_modules_13_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_105
        ) = None
        query_states_26 = torch._C._nn.linear(
            normed_hidden_states_13,
            l_self_modules_block_modules_13_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_13_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_53 = query_states_26.view(1, -1, 12, 64)
        query_states_26 = None
        query_states_27 = view_53.transpose(1, 2)
        view_53 = None
        key_states_26 = torch._C._nn.linear(
            normed_hidden_states_13,
            l_self_modules_block_modules_13_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_13_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_26 = torch._C._nn.linear(
            normed_hidden_states_13,
            l_self_modules_block_modules_13_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_13 = l_self_modules_block_modules_13_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_54 = key_states_26.view(1, -1, 12, 64)
        key_states_26 = None
        key_states_27 = view_54.transpose(1, 2)
        view_54 = None
        view_55 = value_states_26.view(1, -1, 12, 64)
        value_states_26 = None
        value_states_27 = view_55.transpose(1, 2)
        view_55 = None
        transpose_68 = key_states_27.transpose(3, 2)
        key_states_27 = None
        scores_26 = torch.matmul(query_states_27, transpose_68)
        query_states_27 = transpose_68 = None
        scores_26 += position_bias_1
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
        attn_weights_27 = value_states_27 = None
        transpose_69 = attn_output_52.transpose(1, 2)
        attn_output_52 = None
        attn_output_53 = transpose_69.contiguous()
        transpose_69 = None
        attn_output_54 = attn_output_53.view(1, -1, 768)
        attn_output_53 = None
        attn_output_55 = torch._C._nn.linear(
            attn_output_54,
            l_self_modules_block_modules_13_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_54 = l_self_modules_block_modules_13_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_54 = torch.nn.functional.dropout(attn_output_55, 0.1, False, False)
        attn_output_55 = None
        hidden_states_106 = hidden_states_104 + dropout_54
        hidden_states_104 = dropout_54 = None
        to_31 = hidden_states_106.to(torch.float32)
        pow_28 = to_31.pow(2)
        to_31 = None
        variance_27 = pow_28.mean(-1, keepdim=True)
        pow_28 = None
        add_58 = variance_27 + 1e-06
        variance_27 = None
        rsqrt_27 = torch.rsqrt(add_58)
        add_58 = None
        hidden_states_107 = hidden_states_106 * rsqrt_27
        rsqrt_27 = None
        forwarded_states_13 = (
            l_self_modules_block_modules_13_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_107
        )
        l_self_modules_block_modules_13_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_107
        ) = None
        hidden_states_108 = torch._C._nn.linear(
            forwarded_states_13,
            l_self_modules_block_modules_13_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_13 = l_self_modules_block_modules_13_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_109 = torch.nn.functional.relu(hidden_states_108, inplace=False)
        hidden_states_108 = None
        hidden_states_110 = torch.nn.functional.dropout(
            hidden_states_109, 0.1, False, False
        )
        hidden_states_109 = None
        hidden_states_111 = torch._C._nn.linear(
            hidden_states_110,
            l_self_modules_block_modules_13_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_110 = l_self_modules_block_modules_13_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_56 = torch.nn.functional.dropout(hidden_states_111, 0.1, False, False)
        hidden_states_111 = None
        hidden_states_112 = hidden_states_106 + dropout_56
        hidden_states_106 = dropout_56 = None
        to_32 = hidden_states_112.to(torch.float32)
        pow_29 = to_32.pow(2)
        to_32 = None
        variance_28 = pow_29.mean(-1, keepdim=True)
        pow_29 = None
        add_60 = variance_28 + 1e-06
        variance_28 = None
        rsqrt_28 = torch.rsqrt(add_60)
        add_60 = None
        hidden_states_113 = hidden_states_112 * rsqrt_28
        rsqrt_28 = None
        normed_hidden_states_14 = (
            l_self_modules_block_modules_14_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_113
        )
        l_self_modules_block_modules_14_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_113
        ) = None
        query_states_28 = torch._C._nn.linear(
            normed_hidden_states_14,
            l_self_modules_block_modules_14_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_14_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_57 = query_states_28.view(1, -1, 12, 64)
        query_states_28 = None
        query_states_29 = view_57.transpose(1, 2)
        view_57 = None
        key_states_28 = torch._C._nn.linear(
            normed_hidden_states_14,
            l_self_modules_block_modules_14_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_14_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_28 = torch._C._nn.linear(
            normed_hidden_states_14,
            l_self_modules_block_modules_14_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_14 = l_self_modules_block_modules_14_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_58 = key_states_28.view(1, -1, 12, 64)
        key_states_28 = None
        key_states_29 = view_58.transpose(1, 2)
        view_58 = None
        view_59 = value_states_28.view(1, -1, 12, 64)
        value_states_28 = None
        value_states_29 = view_59.transpose(1, 2)
        view_59 = None
        transpose_73 = key_states_29.transpose(3, 2)
        key_states_29 = None
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
        attn_weights_29 = value_states_29 = None
        transpose_74 = attn_output_56.transpose(1, 2)
        attn_output_56 = None
        attn_output_57 = transpose_74.contiguous()
        transpose_74 = None
        attn_output_58 = attn_output_57.view(1, -1, 768)
        attn_output_57 = None
        attn_output_59 = torch._C._nn.linear(
            attn_output_58,
            l_self_modules_block_modules_14_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_58 = l_self_modules_block_modules_14_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_58 = torch.nn.functional.dropout(attn_output_59, 0.1, False, False)
        attn_output_59 = None
        hidden_states_114 = hidden_states_112 + dropout_58
        hidden_states_112 = dropout_58 = None
        to_33 = hidden_states_114.to(torch.float32)
        pow_30 = to_33.pow(2)
        to_33 = None
        variance_29 = pow_30.mean(-1, keepdim=True)
        pow_30 = None
        add_62 = variance_29 + 1e-06
        variance_29 = None
        rsqrt_29 = torch.rsqrt(add_62)
        add_62 = None
        hidden_states_115 = hidden_states_114 * rsqrt_29
        rsqrt_29 = None
        forwarded_states_14 = (
            l_self_modules_block_modules_14_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_115
        )
        l_self_modules_block_modules_14_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_115
        ) = None
        hidden_states_116 = torch._C._nn.linear(
            forwarded_states_14,
            l_self_modules_block_modules_14_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_14 = l_self_modules_block_modules_14_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_117 = torch.nn.functional.relu(hidden_states_116, inplace=False)
        hidden_states_116 = None
        hidden_states_118 = torch.nn.functional.dropout(
            hidden_states_117, 0.1, False, False
        )
        hidden_states_117 = None
        hidden_states_119 = torch._C._nn.linear(
            hidden_states_118,
            l_self_modules_block_modules_14_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_118 = l_self_modules_block_modules_14_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_60 = torch.nn.functional.dropout(hidden_states_119, 0.1, False, False)
        hidden_states_119 = None
        hidden_states_120 = hidden_states_114 + dropout_60
        hidden_states_114 = dropout_60 = None
        to_34 = hidden_states_120.to(torch.float32)
        pow_31 = to_34.pow(2)
        to_34 = None
        variance_30 = pow_31.mean(-1, keepdim=True)
        pow_31 = None
        add_64 = variance_30 + 1e-06
        variance_30 = None
        rsqrt_30 = torch.rsqrt(add_64)
        add_64 = None
        hidden_states_121 = hidden_states_120 * rsqrt_30
        rsqrt_30 = None
        normed_hidden_states_15 = (
            l_self_modules_block_modules_15_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_121
        )
        l_self_modules_block_modules_15_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_121
        ) = None
        query_states_30 = torch._C._nn.linear(
            normed_hidden_states_15,
            l_self_modules_block_modules_15_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_15_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_61 = query_states_30.view(1, -1, 12, 64)
        query_states_30 = None
        query_states_31 = view_61.transpose(1, 2)
        view_61 = None
        key_states_30 = torch._C._nn.linear(
            normed_hidden_states_15,
            l_self_modules_block_modules_15_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_15_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_30 = torch._C._nn.linear(
            normed_hidden_states_15,
            l_self_modules_block_modules_15_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_15 = l_self_modules_block_modules_15_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_62 = key_states_30.view(1, -1, 12, 64)
        key_states_30 = None
        key_states_31 = view_62.transpose(1, 2)
        view_62 = None
        view_63 = value_states_30.view(1, -1, 12, 64)
        value_states_30 = None
        value_states_31 = view_63.transpose(1, 2)
        view_63 = None
        transpose_78 = key_states_31.transpose(3, 2)
        key_states_31 = None
        scores_30 = torch.matmul(query_states_31, transpose_78)
        query_states_31 = transpose_78 = None
        scores_30 += position_bias_1
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
        attn_weights_31 = value_states_31 = None
        transpose_79 = attn_output_60.transpose(1, 2)
        attn_output_60 = None
        attn_output_61 = transpose_79.contiguous()
        transpose_79 = None
        attn_output_62 = attn_output_61.view(1, -1, 768)
        attn_output_61 = None
        attn_output_63 = torch._C._nn.linear(
            attn_output_62,
            l_self_modules_block_modules_15_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_62 = l_self_modules_block_modules_15_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_62 = torch.nn.functional.dropout(attn_output_63, 0.1, False, False)
        attn_output_63 = None
        hidden_states_122 = hidden_states_120 + dropout_62
        hidden_states_120 = dropout_62 = None
        to_35 = hidden_states_122.to(torch.float32)
        pow_32 = to_35.pow(2)
        to_35 = None
        variance_31 = pow_32.mean(-1, keepdim=True)
        pow_32 = None
        add_66 = variance_31 + 1e-06
        variance_31 = None
        rsqrt_31 = torch.rsqrt(add_66)
        add_66 = None
        hidden_states_123 = hidden_states_122 * rsqrt_31
        rsqrt_31 = None
        forwarded_states_15 = (
            l_self_modules_block_modules_15_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_123
        )
        l_self_modules_block_modules_15_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_123
        ) = None
        hidden_states_124 = torch._C._nn.linear(
            forwarded_states_15,
            l_self_modules_block_modules_15_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_15 = l_self_modules_block_modules_15_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_125 = torch.nn.functional.relu(hidden_states_124, inplace=False)
        hidden_states_124 = None
        hidden_states_126 = torch.nn.functional.dropout(
            hidden_states_125, 0.1, False, False
        )
        hidden_states_125 = None
        hidden_states_127 = torch._C._nn.linear(
            hidden_states_126,
            l_self_modules_block_modules_15_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_126 = l_self_modules_block_modules_15_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_64 = torch.nn.functional.dropout(hidden_states_127, 0.1, False, False)
        hidden_states_127 = None
        hidden_states_128 = hidden_states_122 + dropout_64
        hidden_states_122 = dropout_64 = None
        to_36 = hidden_states_128.to(torch.float32)
        pow_33 = to_36.pow(2)
        to_36 = None
        variance_32 = pow_33.mean(-1, keepdim=True)
        pow_33 = None
        add_68 = variance_32 + 1e-06
        variance_32 = None
        rsqrt_32 = torch.rsqrt(add_68)
        add_68 = None
        hidden_states_129 = hidden_states_128 * rsqrt_32
        rsqrt_32 = None
        normed_hidden_states_16 = (
            l_self_modules_block_modules_16_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_129
        )
        l_self_modules_block_modules_16_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_129
        ) = None
        query_states_32 = torch._C._nn.linear(
            normed_hidden_states_16,
            l_self_modules_block_modules_16_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_16_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_65 = query_states_32.view(1, -1, 12, 64)
        query_states_32 = None
        query_states_33 = view_65.transpose(1, 2)
        view_65 = None
        key_states_32 = torch._C._nn.linear(
            normed_hidden_states_16,
            l_self_modules_block_modules_16_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_16_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_32 = torch._C._nn.linear(
            normed_hidden_states_16,
            l_self_modules_block_modules_16_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_16 = l_self_modules_block_modules_16_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_66 = key_states_32.view(1, -1, 12, 64)
        key_states_32 = None
        key_states_33 = view_66.transpose(1, 2)
        view_66 = None
        view_67 = value_states_32.view(1, -1, 12, 64)
        value_states_32 = None
        value_states_33 = view_67.transpose(1, 2)
        view_67 = None
        transpose_83 = key_states_33.transpose(3, 2)
        key_states_33 = None
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
        attn_weights_33 = value_states_33 = None
        transpose_84 = attn_output_64.transpose(1, 2)
        attn_output_64 = None
        attn_output_65 = transpose_84.contiguous()
        transpose_84 = None
        attn_output_66 = attn_output_65.view(1, -1, 768)
        attn_output_65 = None
        attn_output_67 = torch._C._nn.linear(
            attn_output_66,
            l_self_modules_block_modules_16_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_66 = l_self_modules_block_modules_16_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_66 = torch.nn.functional.dropout(attn_output_67, 0.1, False, False)
        attn_output_67 = None
        hidden_states_130 = hidden_states_128 + dropout_66
        hidden_states_128 = dropout_66 = None
        to_37 = hidden_states_130.to(torch.float32)
        pow_34 = to_37.pow(2)
        to_37 = None
        variance_33 = pow_34.mean(-1, keepdim=True)
        pow_34 = None
        add_70 = variance_33 + 1e-06
        variance_33 = None
        rsqrt_33 = torch.rsqrt(add_70)
        add_70 = None
        hidden_states_131 = hidden_states_130 * rsqrt_33
        rsqrt_33 = None
        forwarded_states_16 = (
            l_self_modules_block_modules_16_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_131
        )
        l_self_modules_block_modules_16_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_131
        ) = None
        hidden_states_132 = torch._C._nn.linear(
            forwarded_states_16,
            l_self_modules_block_modules_16_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_16 = l_self_modules_block_modules_16_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_133 = torch.nn.functional.relu(hidden_states_132, inplace=False)
        hidden_states_132 = None
        hidden_states_134 = torch.nn.functional.dropout(
            hidden_states_133, 0.1, False, False
        )
        hidden_states_133 = None
        hidden_states_135 = torch._C._nn.linear(
            hidden_states_134,
            l_self_modules_block_modules_16_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_134 = l_self_modules_block_modules_16_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_68 = torch.nn.functional.dropout(hidden_states_135, 0.1, False, False)
        hidden_states_135 = None
        hidden_states_136 = hidden_states_130 + dropout_68
        hidden_states_130 = dropout_68 = None
        to_38 = hidden_states_136.to(torch.float32)
        pow_35 = to_38.pow(2)
        to_38 = None
        variance_34 = pow_35.mean(-1, keepdim=True)
        pow_35 = None
        add_72 = variance_34 + 1e-06
        variance_34 = None
        rsqrt_34 = torch.rsqrt(add_72)
        add_72 = None
        hidden_states_137 = hidden_states_136 * rsqrt_34
        rsqrt_34 = None
        normed_hidden_states_17 = (
            l_self_modules_block_modules_17_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_137
        )
        l_self_modules_block_modules_17_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_137
        ) = None
        query_states_34 = torch._C._nn.linear(
            normed_hidden_states_17,
            l_self_modules_block_modules_17_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_17_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_69 = query_states_34.view(1, -1, 12, 64)
        query_states_34 = None
        query_states_35 = view_69.transpose(1, 2)
        view_69 = None
        key_states_34 = torch._C._nn.linear(
            normed_hidden_states_17,
            l_self_modules_block_modules_17_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_17_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_34 = torch._C._nn.linear(
            normed_hidden_states_17,
            l_self_modules_block_modules_17_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_17 = l_self_modules_block_modules_17_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_70 = key_states_34.view(1, -1, 12, 64)
        key_states_34 = None
        key_states_35 = view_70.transpose(1, 2)
        view_70 = None
        view_71 = value_states_34.view(1, -1, 12, 64)
        value_states_34 = None
        value_states_35 = view_71.transpose(1, 2)
        view_71 = None
        transpose_88 = key_states_35.transpose(3, 2)
        key_states_35 = None
        scores_34 = torch.matmul(query_states_35, transpose_88)
        query_states_35 = transpose_88 = None
        scores_34 += position_bias_1
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
        attn_weights_35 = value_states_35 = None
        transpose_89 = attn_output_68.transpose(1, 2)
        attn_output_68 = None
        attn_output_69 = transpose_89.contiguous()
        transpose_89 = None
        attn_output_70 = attn_output_69.view(1, -1, 768)
        attn_output_69 = None
        attn_output_71 = torch._C._nn.linear(
            attn_output_70,
            l_self_modules_block_modules_17_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_70 = l_self_modules_block_modules_17_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_70 = torch.nn.functional.dropout(attn_output_71, 0.1, False, False)
        attn_output_71 = None
        hidden_states_138 = hidden_states_136 + dropout_70
        hidden_states_136 = dropout_70 = None
        to_39 = hidden_states_138.to(torch.float32)
        pow_36 = to_39.pow(2)
        to_39 = None
        variance_35 = pow_36.mean(-1, keepdim=True)
        pow_36 = None
        add_74 = variance_35 + 1e-06
        variance_35 = None
        rsqrt_35 = torch.rsqrt(add_74)
        add_74 = None
        hidden_states_139 = hidden_states_138 * rsqrt_35
        rsqrt_35 = None
        forwarded_states_17 = (
            l_self_modules_block_modules_17_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_139
        )
        l_self_modules_block_modules_17_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_139
        ) = None
        hidden_states_140 = torch._C._nn.linear(
            forwarded_states_17,
            l_self_modules_block_modules_17_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_17 = l_self_modules_block_modules_17_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_141 = torch.nn.functional.relu(hidden_states_140, inplace=False)
        hidden_states_140 = None
        hidden_states_142 = torch.nn.functional.dropout(
            hidden_states_141, 0.1, False, False
        )
        hidden_states_141 = None
        hidden_states_143 = torch._C._nn.linear(
            hidden_states_142,
            l_self_modules_block_modules_17_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_142 = l_self_modules_block_modules_17_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_72 = torch.nn.functional.dropout(hidden_states_143, 0.1, False, False)
        hidden_states_143 = None
        hidden_states_144 = hidden_states_138 + dropout_72
        hidden_states_138 = dropout_72 = None
        to_40 = hidden_states_144.to(torch.float32)
        pow_37 = to_40.pow(2)
        to_40 = None
        variance_36 = pow_37.mean(-1, keepdim=True)
        pow_37 = None
        add_76 = variance_36 + 1e-06
        variance_36 = None
        rsqrt_36 = torch.rsqrt(add_76)
        add_76 = None
        hidden_states_145 = hidden_states_144 * rsqrt_36
        rsqrt_36 = None
        normed_hidden_states_18 = (
            l_self_modules_block_modules_18_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_145
        )
        l_self_modules_block_modules_18_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_145
        ) = None
        query_states_36 = torch._C._nn.linear(
            normed_hidden_states_18,
            l_self_modules_block_modules_18_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_18_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_73 = query_states_36.view(1, -1, 12, 64)
        query_states_36 = None
        query_states_37 = view_73.transpose(1, 2)
        view_73 = None
        key_states_36 = torch._C._nn.linear(
            normed_hidden_states_18,
            l_self_modules_block_modules_18_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_18_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_36 = torch._C._nn.linear(
            normed_hidden_states_18,
            l_self_modules_block_modules_18_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_18 = l_self_modules_block_modules_18_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_74 = key_states_36.view(1, -1, 12, 64)
        key_states_36 = None
        key_states_37 = view_74.transpose(1, 2)
        view_74 = None
        view_75 = value_states_36.view(1, -1, 12, 64)
        value_states_36 = None
        value_states_37 = view_75.transpose(1, 2)
        view_75 = None
        transpose_93 = key_states_37.transpose(3, 2)
        key_states_37 = None
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
        attn_weights_37 = value_states_37 = None
        transpose_94 = attn_output_72.transpose(1, 2)
        attn_output_72 = None
        attn_output_73 = transpose_94.contiguous()
        transpose_94 = None
        attn_output_74 = attn_output_73.view(1, -1, 768)
        attn_output_73 = None
        attn_output_75 = torch._C._nn.linear(
            attn_output_74,
            l_self_modules_block_modules_18_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_74 = l_self_modules_block_modules_18_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_74 = torch.nn.functional.dropout(attn_output_75, 0.1, False, False)
        attn_output_75 = None
        hidden_states_146 = hidden_states_144 + dropout_74
        hidden_states_144 = dropout_74 = None
        to_41 = hidden_states_146.to(torch.float32)
        pow_38 = to_41.pow(2)
        to_41 = None
        variance_37 = pow_38.mean(-1, keepdim=True)
        pow_38 = None
        add_78 = variance_37 + 1e-06
        variance_37 = None
        rsqrt_37 = torch.rsqrt(add_78)
        add_78 = None
        hidden_states_147 = hidden_states_146 * rsqrt_37
        rsqrt_37 = None
        forwarded_states_18 = (
            l_self_modules_block_modules_18_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_147
        )
        l_self_modules_block_modules_18_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_147
        ) = None
        hidden_states_148 = torch._C._nn.linear(
            forwarded_states_18,
            l_self_modules_block_modules_18_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_18 = l_self_modules_block_modules_18_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_149 = torch.nn.functional.relu(hidden_states_148, inplace=False)
        hidden_states_148 = None
        hidden_states_150 = torch.nn.functional.dropout(
            hidden_states_149, 0.1, False, False
        )
        hidden_states_149 = None
        hidden_states_151 = torch._C._nn.linear(
            hidden_states_150,
            l_self_modules_block_modules_18_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_150 = l_self_modules_block_modules_18_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_76 = torch.nn.functional.dropout(hidden_states_151, 0.1, False, False)
        hidden_states_151 = None
        hidden_states_152 = hidden_states_146 + dropout_76
        hidden_states_146 = dropout_76 = None
        to_42 = hidden_states_152.to(torch.float32)
        pow_39 = to_42.pow(2)
        to_42 = None
        variance_38 = pow_39.mean(-1, keepdim=True)
        pow_39 = None
        add_80 = variance_38 + 1e-06
        variance_38 = None
        rsqrt_38 = torch.rsqrt(add_80)
        add_80 = None
        hidden_states_153 = hidden_states_152 * rsqrt_38
        rsqrt_38 = None
        normed_hidden_states_19 = (
            l_self_modules_block_modules_19_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_153
        )
        l_self_modules_block_modules_19_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_153
        ) = None
        query_states_38 = torch._C._nn.linear(
            normed_hidden_states_19,
            l_self_modules_block_modules_19_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_19_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_77 = query_states_38.view(1, -1, 12, 64)
        query_states_38 = None
        query_states_39 = view_77.transpose(1, 2)
        view_77 = None
        key_states_38 = torch._C._nn.linear(
            normed_hidden_states_19,
            l_self_modules_block_modules_19_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_19_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_38 = torch._C._nn.linear(
            normed_hidden_states_19,
            l_self_modules_block_modules_19_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_19 = l_self_modules_block_modules_19_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_78 = key_states_38.view(1, -1, 12, 64)
        key_states_38 = None
        key_states_39 = view_78.transpose(1, 2)
        view_78 = None
        view_79 = value_states_38.view(1, -1, 12, 64)
        value_states_38 = None
        value_states_39 = view_79.transpose(1, 2)
        view_79 = None
        transpose_98 = key_states_39.transpose(3, 2)
        key_states_39 = None
        scores_38 = torch.matmul(query_states_39, transpose_98)
        query_states_39 = transpose_98 = None
        scores_38 += position_bias_1
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
        attn_weights_39 = value_states_39 = None
        transpose_99 = attn_output_76.transpose(1, 2)
        attn_output_76 = None
        attn_output_77 = transpose_99.contiguous()
        transpose_99 = None
        attn_output_78 = attn_output_77.view(1, -1, 768)
        attn_output_77 = None
        attn_output_79 = torch._C._nn.linear(
            attn_output_78,
            l_self_modules_block_modules_19_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_78 = l_self_modules_block_modules_19_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_78 = torch.nn.functional.dropout(attn_output_79, 0.1, False, False)
        attn_output_79 = None
        hidden_states_154 = hidden_states_152 + dropout_78
        hidden_states_152 = dropout_78 = None
        to_43 = hidden_states_154.to(torch.float32)
        pow_40 = to_43.pow(2)
        to_43 = None
        variance_39 = pow_40.mean(-1, keepdim=True)
        pow_40 = None
        add_82 = variance_39 + 1e-06
        variance_39 = None
        rsqrt_39 = torch.rsqrt(add_82)
        add_82 = None
        hidden_states_155 = hidden_states_154 * rsqrt_39
        rsqrt_39 = None
        forwarded_states_19 = (
            l_self_modules_block_modules_19_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_155
        )
        l_self_modules_block_modules_19_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_155
        ) = None
        hidden_states_156 = torch._C._nn.linear(
            forwarded_states_19,
            l_self_modules_block_modules_19_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_19 = l_self_modules_block_modules_19_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_157 = torch.nn.functional.relu(hidden_states_156, inplace=False)
        hidden_states_156 = None
        hidden_states_158 = torch.nn.functional.dropout(
            hidden_states_157, 0.1, False, False
        )
        hidden_states_157 = None
        hidden_states_159 = torch._C._nn.linear(
            hidden_states_158,
            l_self_modules_block_modules_19_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_158 = l_self_modules_block_modules_19_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_80 = torch.nn.functional.dropout(hidden_states_159, 0.1, False, False)
        hidden_states_159 = None
        hidden_states_160 = hidden_states_154 + dropout_80
        hidden_states_154 = dropout_80 = None
        to_44 = hidden_states_160.to(torch.float32)
        pow_41 = to_44.pow(2)
        to_44 = None
        variance_40 = pow_41.mean(-1, keepdim=True)
        pow_41 = None
        add_84 = variance_40 + 1e-06
        variance_40 = None
        rsqrt_40 = torch.rsqrt(add_84)
        add_84 = None
        hidden_states_161 = hidden_states_160 * rsqrt_40
        rsqrt_40 = None
        normed_hidden_states_20 = (
            l_self_modules_block_modules_20_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_161
        )
        l_self_modules_block_modules_20_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_161
        ) = None
        query_states_40 = torch._C._nn.linear(
            normed_hidden_states_20,
            l_self_modules_block_modules_20_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_20_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_81 = query_states_40.view(1, -1, 12, 64)
        query_states_40 = None
        query_states_41 = view_81.transpose(1, 2)
        view_81 = None
        key_states_40 = torch._C._nn.linear(
            normed_hidden_states_20,
            l_self_modules_block_modules_20_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_20_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_40 = torch._C._nn.linear(
            normed_hidden_states_20,
            l_self_modules_block_modules_20_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_20 = l_self_modules_block_modules_20_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_82 = key_states_40.view(1, -1, 12, 64)
        key_states_40 = None
        key_states_41 = view_82.transpose(1, 2)
        view_82 = None
        view_83 = value_states_40.view(1, -1, 12, 64)
        value_states_40 = None
        value_states_41 = view_83.transpose(1, 2)
        view_83 = None
        transpose_103 = key_states_41.transpose(3, 2)
        key_states_41 = None
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
        attn_weights_41 = value_states_41 = None
        transpose_104 = attn_output_80.transpose(1, 2)
        attn_output_80 = None
        attn_output_81 = transpose_104.contiguous()
        transpose_104 = None
        attn_output_82 = attn_output_81.view(1, -1, 768)
        attn_output_81 = None
        attn_output_83 = torch._C._nn.linear(
            attn_output_82,
            l_self_modules_block_modules_20_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_82 = l_self_modules_block_modules_20_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_82 = torch.nn.functional.dropout(attn_output_83, 0.1, False, False)
        attn_output_83 = None
        hidden_states_162 = hidden_states_160 + dropout_82
        hidden_states_160 = dropout_82 = None
        to_45 = hidden_states_162.to(torch.float32)
        pow_42 = to_45.pow(2)
        to_45 = None
        variance_41 = pow_42.mean(-1, keepdim=True)
        pow_42 = None
        add_86 = variance_41 + 1e-06
        variance_41 = None
        rsqrt_41 = torch.rsqrt(add_86)
        add_86 = None
        hidden_states_163 = hidden_states_162 * rsqrt_41
        rsqrt_41 = None
        forwarded_states_20 = (
            l_self_modules_block_modules_20_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_163
        )
        l_self_modules_block_modules_20_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_163
        ) = None
        hidden_states_164 = torch._C._nn.linear(
            forwarded_states_20,
            l_self_modules_block_modules_20_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_20 = l_self_modules_block_modules_20_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_165 = torch.nn.functional.relu(hidden_states_164, inplace=False)
        hidden_states_164 = None
        hidden_states_166 = torch.nn.functional.dropout(
            hidden_states_165, 0.1, False, False
        )
        hidden_states_165 = None
        hidden_states_167 = torch._C._nn.linear(
            hidden_states_166,
            l_self_modules_block_modules_20_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_166 = l_self_modules_block_modules_20_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_84 = torch.nn.functional.dropout(hidden_states_167, 0.1, False, False)
        hidden_states_167 = None
        hidden_states_168 = hidden_states_162 + dropout_84
        hidden_states_162 = dropout_84 = None
        to_46 = hidden_states_168.to(torch.float32)
        pow_43 = to_46.pow(2)
        to_46 = None
        variance_42 = pow_43.mean(-1, keepdim=True)
        pow_43 = None
        add_88 = variance_42 + 1e-06
        variance_42 = None
        rsqrt_42 = torch.rsqrt(add_88)
        add_88 = None
        hidden_states_169 = hidden_states_168 * rsqrt_42
        rsqrt_42 = None
        normed_hidden_states_21 = (
            l_self_modules_block_modules_21_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_169
        )
        l_self_modules_block_modules_21_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_169
        ) = None
        query_states_42 = torch._C._nn.linear(
            normed_hidden_states_21,
            l_self_modules_block_modules_21_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_21_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_85 = query_states_42.view(1, -1, 12, 64)
        query_states_42 = None
        query_states_43 = view_85.transpose(1, 2)
        view_85 = None
        key_states_42 = torch._C._nn.linear(
            normed_hidden_states_21,
            l_self_modules_block_modules_21_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_21_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_42 = torch._C._nn.linear(
            normed_hidden_states_21,
            l_self_modules_block_modules_21_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_21 = l_self_modules_block_modules_21_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_86 = key_states_42.view(1, -1, 12, 64)
        key_states_42 = None
        key_states_43 = view_86.transpose(1, 2)
        view_86 = None
        view_87 = value_states_42.view(1, -1, 12, 64)
        value_states_42 = None
        value_states_43 = view_87.transpose(1, 2)
        view_87 = None
        transpose_108 = key_states_43.transpose(3, 2)
        key_states_43 = None
        scores_42 = torch.matmul(query_states_43, transpose_108)
        query_states_43 = transpose_108 = None
        scores_42 += position_bias_1
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
        attn_weights_43 = value_states_43 = None
        transpose_109 = attn_output_84.transpose(1, 2)
        attn_output_84 = None
        attn_output_85 = transpose_109.contiguous()
        transpose_109 = None
        attn_output_86 = attn_output_85.view(1, -1, 768)
        attn_output_85 = None
        attn_output_87 = torch._C._nn.linear(
            attn_output_86,
            l_self_modules_block_modules_21_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_86 = l_self_modules_block_modules_21_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_86 = torch.nn.functional.dropout(attn_output_87, 0.1, False, False)
        attn_output_87 = None
        hidden_states_170 = hidden_states_168 + dropout_86
        hidden_states_168 = dropout_86 = None
        to_47 = hidden_states_170.to(torch.float32)
        pow_44 = to_47.pow(2)
        to_47 = None
        variance_43 = pow_44.mean(-1, keepdim=True)
        pow_44 = None
        add_90 = variance_43 + 1e-06
        variance_43 = None
        rsqrt_43 = torch.rsqrt(add_90)
        add_90 = None
        hidden_states_171 = hidden_states_170 * rsqrt_43
        rsqrt_43 = None
        forwarded_states_21 = (
            l_self_modules_block_modules_21_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_171
        )
        l_self_modules_block_modules_21_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_171
        ) = None
        hidden_states_172 = torch._C._nn.linear(
            forwarded_states_21,
            l_self_modules_block_modules_21_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_21 = l_self_modules_block_modules_21_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_173 = torch.nn.functional.relu(hidden_states_172, inplace=False)
        hidden_states_172 = None
        hidden_states_174 = torch.nn.functional.dropout(
            hidden_states_173, 0.1, False, False
        )
        hidden_states_173 = None
        hidden_states_175 = torch._C._nn.linear(
            hidden_states_174,
            l_self_modules_block_modules_21_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_174 = l_self_modules_block_modules_21_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_88 = torch.nn.functional.dropout(hidden_states_175, 0.1, False, False)
        hidden_states_175 = None
        hidden_states_176 = hidden_states_170 + dropout_88
        hidden_states_170 = dropout_88 = None
        to_48 = hidden_states_176.to(torch.float32)
        pow_45 = to_48.pow(2)
        to_48 = None
        variance_44 = pow_45.mean(-1, keepdim=True)
        pow_45 = None
        add_92 = variance_44 + 1e-06
        variance_44 = None
        rsqrt_44 = torch.rsqrt(add_92)
        add_92 = None
        hidden_states_177 = hidden_states_176 * rsqrt_44
        rsqrt_44 = None
        normed_hidden_states_22 = (
            l_self_modules_block_modules_22_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_177
        )
        l_self_modules_block_modules_22_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_177
        ) = None
        query_states_44 = torch._C._nn.linear(
            normed_hidden_states_22,
            l_self_modules_block_modules_22_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_22_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_89 = query_states_44.view(1, -1, 12, 64)
        query_states_44 = None
        query_states_45 = view_89.transpose(1, 2)
        view_89 = None
        key_states_44 = torch._C._nn.linear(
            normed_hidden_states_22,
            l_self_modules_block_modules_22_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_22_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_44 = torch._C._nn.linear(
            normed_hidden_states_22,
            l_self_modules_block_modules_22_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_22 = l_self_modules_block_modules_22_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_90 = key_states_44.view(1, -1, 12, 64)
        key_states_44 = None
        key_states_45 = view_90.transpose(1, 2)
        view_90 = None
        view_91 = value_states_44.view(1, -1, 12, 64)
        value_states_44 = None
        value_states_45 = view_91.transpose(1, 2)
        view_91 = None
        transpose_113 = key_states_45.transpose(3, 2)
        key_states_45 = None
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
        attn_weights_45 = value_states_45 = None
        transpose_114 = attn_output_88.transpose(1, 2)
        attn_output_88 = None
        attn_output_89 = transpose_114.contiguous()
        transpose_114 = None
        attn_output_90 = attn_output_89.view(1, -1, 768)
        attn_output_89 = None
        attn_output_91 = torch._C._nn.linear(
            attn_output_90,
            l_self_modules_block_modules_22_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_90 = l_self_modules_block_modules_22_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_90 = torch.nn.functional.dropout(attn_output_91, 0.1, False, False)
        attn_output_91 = None
        hidden_states_178 = hidden_states_176 + dropout_90
        hidden_states_176 = dropout_90 = None
        to_49 = hidden_states_178.to(torch.float32)
        pow_46 = to_49.pow(2)
        to_49 = None
        variance_45 = pow_46.mean(-1, keepdim=True)
        pow_46 = None
        add_94 = variance_45 + 1e-06
        variance_45 = None
        rsqrt_45 = torch.rsqrt(add_94)
        add_94 = None
        hidden_states_179 = hidden_states_178 * rsqrt_45
        rsqrt_45 = None
        forwarded_states_22 = (
            l_self_modules_block_modules_22_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_179
        )
        l_self_modules_block_modules_22_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_179
        ) = None
        hidden_states_180 = torch._C._nn.linear(
            forwarded_states_22,
            l_self_modules_block_modules_22_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_22 = l_self_modules_block_modules_22_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_181 = torch.nn.functional.relu(hidden_states_180, inplace=False)
        hidden_states_180 = None
        hidden_states_182 = torch.nn.functional.dropout(
            hidden_states_181, 0.1, False, False
        )
        hidden_states_181 = None
        hidden_states_183 = torch._C._nn.linear(
            hidden_states_182,
            l_self_modules_block_modules_22_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_182 = l_self_modules_block_modules_22_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_92 = torch.nn.functional.dropout(hidden_states_183, 0.1, False, False)
        hidden_states_183 = None
        hidden_states_184 = hidden_states_178 + dropout_92
        hidden_states_178 = dropout_92 = None
        to_50 = hidden_states_184.to(torch.float32)
        pow_47 = to_50.pow(2)
        to_50 = None
        variance_46 = pow_47.mean(-1, keepdim=True)
        pow_47 = None
        add_96 = variance_46 + 1e-06
        variance_46 = None
        rsqrt_46 = torch.rsqrt(add_96)
        add_96 = None
        hidden_states_185 = hidden_states_184 * rsqrt_46
        rsqrt_46 = None
        normed_hidden_states_23 = (
            l_self_modules_block_modules_23_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_185
        )
        l_self_modules_block_modules_23_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_185
        ) = None
        query_states_46 = torch._C._nn.linear(
            normed_hidden_states_23,
            l_self_modules_block_modules_23_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_23_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_93 = query_states_46.view(1, -1, 12, 64)
        query_states_46 = None
        query_states_47 = view_93.transpose(1, 2)
        view_93 = None
        key_states_46 = torch._C._nn.linear(
            normed_hidden_states_23,
            l_self_modules_block_modules_23_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_23_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_46 = torch._C._nn.linear(
            normed_hidden_states_23,
            l_self_modules_block_modules_23_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_23 = l_self_modules_block_modules_23_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_94 = key_states_46.view(1, -1, 12, 64)
        key_states_46 = None
        key_states_47 = view_94.transpose(1, 2)
        view_94 = None
        view_95 = value_states_46.view(1, -1, 12, 64)
        value_states_46 = None
        value_states_47 = view_95.transpose(1, 2)
        view_95 = None
        transpose_118 = key_states_47.transpose(3, 2)
        key_states_47 = None
        scores_46 = torch.matmul(query_states_47, transpose_118)
        query_states_47 = transpose_118 = None
        scores_46 += position_bias_1
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
        attn_weights_47 = value_states_47 = None
        transpose_119 = attn_output_92.transpose(1, 2)
        attn_output_92 = None
        attn_output_93 = transpose_119.contiguous()
        transpose_119 = None
        attn_output_94 = attn_output_93.view(1, -1, 768)
        attn_output_93 = None
        attn_output_95 = torch._C._nn.linear(
            attn_output_94,
            l_self_modules_block_modules_23_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_94 = l_self_modules_block_modules_23_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_94 = torch.nn.functional.dropout(attn_output_95, 0.1, False, False)
        attn_output_95 = None
        hidden_states_186 = hidden_states_184 + dropout_94
        hidden_states_184 = dropout_94 = None
        to_51 = hidden_states_186.to(torch.float32)
        pow_48 = to_51.pow(2)
        to_51 = None
        variance_47 = pow_48.mean(-1, keepdim=True)
        pow_48 = None
        add_98 = variance_47 + 1e-06
        variance_47 = None
        rsqrt_47 = torch.rsqrt(add_98)
        add_98 = None
        hidden_states_187 = hidden_states_186 * rsqrt_47
        rsqrt_47 = None
        forwarded_states_23 = (
            l_self_modules_block_modules_23_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_187
        )
        l_self_modules_block_modules_23_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_187
        ) = None
        hidden_states_188 = torch._C._nn.linear(
            forwarded_states_23,
            l_self_modules_block_modules_23_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_23 = l_self_modules_block_modules_23_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_189 = torch.nn.functional.relu(hidden_states_188, inplace=False)
        hidden_states_188 = None
        hidden_states_190 = torch.nn.functional.dropout(
            hidden_states_189, 0.1, False, False
        )
        hidden_states_189 = None
        hidden_states_191 = torch._C._nn.linear(
            hidden_states_190,
            l_self_modules_block_modules_23_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_190 = l_self_modules_block_modules_23_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_96 = torch.nn.functional.dropout(hidden_states_191, 0.1, False, False)
        hidden_states_191 = None
        hidden_states_192 = hidden_states_186 + dropout_96
        hidden_states_186 = dropout_96 = None
        to_52 = hidden_states_192.to(torch.float32)
        pow_49 = to_52.pow(2)
        to_52 = None
        variance_48 = pow_49.mean(-1, keepdim=True)
        pow_49 = None
        add_100 = variance_48 + 1e-06
        variance_48 = None
        rsqrt_48 = torch.rsqrt(add_100)
        add_100 = None
        hidden_states_193 = hidden_states_192 * rsqrt_48
        rsqrt_48 = None
        normed_hidden_states_24 = (
            l_self_modules_block_modules_24_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_193
        )
        l_self_modules_block_modules_24_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_193
        ) = None
        query_states_48 = torch._C._nn.linear(
            normed_hidden_states_24,
            l_self_modules_block_modules_24_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_24_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_97 = query_states_48.view(1, -1, 12, 64)
        query_states_48 = None
        query_states_49 = view_97.transpose(1, 2)
        view_97 = None
        key_states_48 = torch._C._nn.linear(
            normed_hidden_states_24,
            l_self_modules_block_modules_24_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_24_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_48 = torch._C._nn.linear(
            normed_hidden_states_24,
            l_self_modules_block_modules_24_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_24 = l_self_modules_block_modules_24_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_98 = key_states_48.view(1, -1, 12, 64)
        key_states_48 = None
        key_states_49 = view_98.transpose(1, 2)
        view_98 = None
        view_99 = value_states_48.view(1, -1, 12, 64)
        value_states_48 = None
        value_states_49 = view_99.transpose(1, 2)
        view_99 = None
        transpose_123 = key_states_49.transpose(3, 2)
        key_states_49 = None
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
        attn_weights_49 = value_states_49 = None
        transpose_124 = attn_output_96.transpose(1, 2)
        attn_output_96 = None
        attn_output_97 = transpose_124.contiguous()
        transpose_124 = None
        attn_output_98 = attn_output_97.view(1, -1, 768)
        attn_output_97 = None
        attn_output_99 = torch._C._nn.linear(
            attn_output_98,
            l_self_modules_block_modules_24_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_98 = l_self_modules_block_modules_24_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_98 = torch.nn.functional.dropout(attn_output_99, 0.1, False, False)
        attn_output_99 = None
        hidden_states_194 = hidden_states_192 + dropout_98
        hidden_states_192 = dropout_98 = None
        to_53 = hidden_states_194.to(torch.float32)
        pow_50 = to_53.pow(2)
        to_53 = None
        variance_49 = pow_50.mean(-1, keepdim=True)
        pow_50 = None
        add_102 = variance_49 + 1e-06
        variance_49 = None
        rsqrt_49 = torch.rsqrt(add_102)
        add_102 = None
        hidden_states_195 = hidden_states_194 * rsqrt_49
        rsqrt_49 = None
        forwarded_states_24 = (
            l_self_modules_block_modules_24_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_195
        )
        l_self_modules_block_modules_24_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_195
        ) = None
        hidden_states_196 = torch._C._nn.linear(
            forwarded_states_24,
            l_self_modules_block_modules_24_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_24 = l_self_modules_block_modules_24_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_197 = torch.nn.functional.relu(hidden_states_196, inplace=False)
        hidden_states_196 = None
        hidden_states_198 = torch.nn.functional.dropout(
            hidden_states_197, 0.1, False, False
        )
        hidden_states_197 = None
        hidden_states_199 = torch._C._nn.linear(
            hidden_states_198,
            l_self_modules_block_modules_24_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_198 = l_self_modules_block_modules_24_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_100 = torch.nn.functional.dropout(hidden_states_199, 0.1, False, False)
        hidden_states_199 = None
        hidden_states_200 = hidden_states_194 + dropout_100
        hidden_states_194 = dropout_100 = None
        to_54 = hidden_states_200.to(torch.float32)
        pow_51 = to_54.pow(2)
        to_54 = None
        variance_50 = pow_51.mean(-1, keepdim=True)
        pow_51 = None
        add_104 = variance_50 + 1e-06
        variance_50 = None
        rsqrt_50 = torch.rsqrt(add_104)
        add_104 = None
        hidden_states_201 = hidden_states_200 * rsqrt_50
        rsqrt_50 = None
        normed_hidden_states_25 = (
            l_self_modules_block_modules_25_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_201
        )
        l_self_modules_block_modules_25_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_201
        ) = None
        query_states_50 = torch._C._nn.linear(
            normed_hidden_states_25,
            l_self_modules_block_modules_25_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_25_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_101 = query_states_50.view(1, -1, 12, 64)
        query_states_50 = None
        query_states_51 = view_101.transpose(1, 2)
        view_101 = None
        key_states_50 = torch._C._nn.linear(
            normed_hidden_states_25,
            l_self_modules_block_modules_25_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_25_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_50 = torch._C._nn.linear(
            normed_hidden_states_25,
            l_self_modules_block_modules_25_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_25 = l_self_modules_block_modules_25_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_102 = key_states_50.view(1, -1, 12, 64)
        key_states_50 = None
        key_states_51 = view_102.transpose(1, 2)
        view_102 = None
        view_103 = value_states_50.view(1, -1, 12, 64)
        value_states_50 = None
        value_states_51 = view_103.transpose(1, 2)
        view_103 = None
        transpose_128 = key_states_51.transpose(3, 2)
        key_states_51 = None
        scores_50 = torch.matmul(query_states_51, transpose_128)
        query_states_51 = transpose_128 = None
        scores_50 += position_bias_1
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
        attn_weights_51 = value_states_51 = None
        transpose_129 = attn_output_100.transpose(1, 2)
        attn_output_100 = None
        attn_output_101 = transpose_129.contiguous()
        transpose_129 = None
        attn_output_102 = attn_output_101.view(1, -1, 768)
        attn_output_101 = None
        attn_output_103 = torch._C._nn.linear(
            attn_output_102,
            l_self_modules_block_modules_25_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_102 = l_self_modules_block_modules_25_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_102 = torch.nn.functional.dropout(attn_output_103, 0.1, False, False)
        attn_output_103 = None
        hidden_states_202 = hidden_states_200 + dropout_102
        hidden_states_200 = dropout_102 = None
        to_55 = hidden_states_202.to(torch.float32)
        pow_52 = to_55.pow(2)
        to_55 = None
        variance_51 = pow_52.mean(-1, keepdim=True)
        pow_52 = None
        add_106 = variance_51 + 1e-06
        variance_51 = None
        rsqrt_51 = torch.rsqrt(add_106)
        add_106 = None
        hidden_states_203 = hidden_states_202 * rsqrt_51
        rsqrt_51 = None
        forwarded_states_25 = (
            l_self_modules_block_modules_25_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_203
        )
        l_self_modules_block_modules_25_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_203
        ) = None
        hidden_states_204 = torch._C._nn.linear(
            forwarded_states_25,
            l_self_modules_block_modules_25_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_25 = l_self_modules_block_modules_25_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_205 = torch.nn.functional.relu(hidden_states_204, inplace=False)
        hidden_states_204 = None
        hidden_states_206 = torch.nn.functional.dropout(
            hidden_states_205, 0.1, False, False
        )
        hidden_states_205 = None
        hidden_states_207 = torch._C._nn.linear(
            hidden_states_206,
            l_self_modules_block_modules_25_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_206 = l_self_modules_block_modules_25_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_104 = torch.nn.functional.dropout(hidden_states_207, 0.1, False, False)
        hidden_states_207 = None
        hidden_states_208 = hidden_states_202 + dropout_104
        hidden_states_202 = dropout_104 = None
        to_56 = hidden_states_208.to(torch.float32)
        pow_53 = to_56.pow(2)
        to_56 = None
        variance_52 = pow_53.mean(-1, keepdim=True)
        pow_53 = None
        add_108 = variance_52 + 1e-06
        variance_52 = None
        rsqrt_52 = torch.rsqrt(add_108)
        add_108 = None
        hidden_states_209 = hidden_states_208 * rsqrt_52
        rsqrt_52 = None
        normed_hidden_states_26 = (
            l_self_modules_block_modules_26_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_209
        )
        l_self_modules_block_modules_26_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_209
        ) = None
        query_states_52 = torch._C._nn.linear(
            normed_hidden_states_26,
            l_self_modules_block_modules_26_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_26_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_105 = query_states_52.view(1, -1, 12, 64)
        query_states_52 = None
        query_states_53 = view_105.transpose(1, 2)
        view_105 = None
        key_states_52 = torch._C._nn.linear(
            normed_hidden_states_26,
            l_self_modules_block_modules_26_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_26_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_52 = torch._C._nn.linear(
            normed_hidden_states_26,
            l_self_modules_block_modules_26_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_26 = l_self_modules_block_modules_26_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_106 = key_states_52.view(1, -1, 12, 64)
        key_states_52 = None
        key_states_53 = view_106.transpose(1, 2)
        view_106 = None
        view_107 = value_states_52.view(1, -1, 12, 64)
        value_states_52 = None
        value_states_53 = view_107.transpose(1, 2)
        view_107 = None
        transpose_133 = key_states_53.transpose(3, 2)
        key_states_53 = None
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
        attn_weights_53 = value_states_53 = None
        transpose_134 = attn_output_104.transpose(1, 2)
        attn_output_104 = None
        attn_output_105 = transpose_134.contiguous()
        transpose_134 = None
        attn_output_106 = attn_output_105.view(1, -1, 768)
        attn_output_105 = None
        attn_output_107 = torch._C._nn.linear(
            attn_output_106,
            l_self_modules_block_modules_26_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_106 = l_self_modules_block_modules_26_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_106 = torch.nn.functional.dropout(attn_output_107, 0.1, False, False)
        attn_output_107 = None
        hidden_states_210 = hidden_states_208 + dropout_106
        hidden_states_208 = dropout_106 = None
        to_57 = hidden_states_210.to(torch.float32)
        pow_54 = to_57.pow(2)
        to_57 = None
        variance_53 = pow_54.mean(-1, keepdim=True)
        pow_54 = None
        add_110 = variance_53 + 1e-06
        variance_53 = None
        rsqrt_53 = torch.rsqrt(add_110)
        add_110 = None
        hidden_states_211 = hidden_states_210 * rsqrt_53
        rsqrt_53 = None
        forwarded_states_26 = (
            l_self_modules_block_modules_26_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_211
        )
        l_self_modules_block_modules_26_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_211
        ) = None
        hidden_states_212 = torch._C._nn.linear(
            forwarded_states_26,
            l_self_modules_block_modules_26_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_26 = l_self_modules_block_modules_26_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_213 = torch.nn.functional.relu(hidden_states_212, inplace=False)
        hidden_states_212 = None
        hidden_states_214 = torch.nn.functional.dropout(
            hidden_states_213, 0.1, False, False
        )
        hidden_states_213 = None
        hidden_states_215 = torch._C._nn.linear(
            hidden_states_214,
            l_self_modules_block_modules_26_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_214 = l_self_modules_block_modules_26_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_108 = torch.nn.functional.dropout(hidden_states_215, 0.1, False, False)
        hidden_states_215 = None
        hidden_states_216 = hidden_states_210 + dropout_108
        hidden_states_210 = dropout_108 = None
        to_58 = hidden_states_216.to(torch.float32)
        pow_55 = to_58.pow(2)
        to_58 = None
        variance_54 = pow_55.mean(-1, keepdim=True)
        pow_55 = None
        add_112 = variance_54 + 1e-06
        variance_54 = None
        rsqrt_54 = torch.rsqrt(add_112)
        add_112 = None
        hidden_states_217 = hidden_states_216 * rsqrt_54
        rsqrt_54 = None
        normed_hidden_states_27 = (
            l_self_modules_block_modules_27_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_217
        )
        l_self_modules_block_modules_27_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_217
        ) = None
        query_states_54 = torch._C._nn.linear(
            normed_hidden_states_27,
            l_self_modules_block_modules_27_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_27_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_109 = query_states_54.view(1, -1, 12, 64)
        query_states_54 = None
        query_states_55 = view_109.transpose(1, 2)
        view_109 = None
        key_states_54 = torch._C._nn.linear(
            normed_hidden_states_27,
            l_self_modules_block_modules_27_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_27_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_54 = torch._C._nn.linear(
            normed_hidden_states_27,
            l_self_modules_block_modules_27_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_27 = l_self_modules_block_modules_27_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_110 = key_states_54.view(1, -1, 12, 64)
        key_states_54 = None
        key_states_55 = view_110.transpose(1, 2)
        view_110 = None
        view_111 = value_states_54.view(1, -1, 12, 64)
        value_states_54 = None
        value_states_55 = view_111.transpose(1, 2)
        view_111 = None
        transpose_138 = key_states_55.transpose(3, 2)
        key_states_55 = None
        scores_54 = torch.matmul(query_states_55, transpose_138)
        query_states_55 = transpose_138 = None
        scores_54 += position_bias_1
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
        attn_weights_55 = value_states_55 = None
        transpose_139 = attn_output_108.transpose(1, 2)
        attn_output_108 = None
        attn_output_109 = transpose_139.contiguous()
        transpose_139 = None
        attn_output_110 = attn_output_109.view(1, -1, 768)
        attn_output_109 = None
        attn_output_111 = torch._C._nn.linear(
            attn_output_110,
            l_self_modules_block_modules_27_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_110 = l_self_modules_block_modules_27_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_110 = torch.nn.functional.dropout(attn_output_111, 0.1, False, False)
        attn_output_111 = None
        hidden_states_218 = hidden_states_216 + dropout_110
        hidden_states_216 = dropout_110 = None
        to_59 = hidden_states_218.to(torch.float32)
        pow_56 = to_59.pow(2)
        to_59 = None
        variance_55 = pow_56.mean(-1, keepdim=True)
        pow_56 = None
        add_114 = variance_55 + 1e-06
        variance_55 = None
        rsqrt_55 = torch.rsqrt(add_114)
        add_114 = None
        hidden_states_219 = hidden_states_218 * rsqrt_55
        rsqrt_55 = None
        forwarded_states_27 = (
            l_self_modules_block_modules_27_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_219
        )
        l_self_modules_block_modules_27_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_219
        ) = None
        hidden_states_220 = torch._C._nn.linear(
            forwarded_states_27,
            l_self_modules_block_modules_27_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_27 = l_self_modules_block_modules_27_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_221 = torch.nn.functional.relu(hidden_states_220, inplace=False)
        hidden_states_220 = None
        hidden_states_222 = torch.nn.functional.dropout(
            hidden_states_221, 0.1, False, False
        )
        hidden_states_221 = None
        hidden_states_223 = torch._C._nn.linear(
            hidden_states_222,
            l_self_modules_block_modules_27_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_222 = l_self_modules_block_modules_27_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_112 = torch.nn.functional.dropout(hidden_states_223, 0.1, False, False)
        hidden_states_223 = None
        hidden_states_224 = hidden_states_218 + dropout_112
        hidden_states_218 = dropout_112 = None
        to_60 = hidden_states_224.to(torch.float32)
        pow_57 = to_60.pow(2)
        to_60 = None
        variance_56 = pow_57.mean(-1, keepdim=True)
        pow_57 = None
        add_116 = variance_56 + 1e-06
        variance_56 = None
        rsqrt_56 = torch.rsqrt(add_116)
        add_116 = None
        hidden_states_225 = hidden_states_224 * rsqrt_56
        rsqrt_56 = None
        normed_hidden_states_28 = (
            l_self_modules_block_modules_28_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_225
        )
        l_self_modules_block_modules_28_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_225
        ) = None
        query_states_56 = torch._C._nn.linear(
            normed_hidden_states_28,
            l_self_modules_block_modules_28_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_28_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_113 = query_states_56.view(1, -1, 12, 64)
        query_states_56 = None
        query_states_57 = view_113.transpose(1, 2)
        view_113 = None
        key_states_56 = torch._C._nn.linear(
            normed_hidden_states_28,
            l_self_modules_block_modules_28_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_28_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_56 = torch._C._nn.linear(
            normed_hidden_states_28,
            l_self_modules_block_modules_28_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_28 = l_self_modules_block_modules_28_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_114 = key_states_56.view(1, -1, 12, 64)
        key_states_56 = None
        key_states_57 = view_114.transpose(1, 2)
        view_114 = None
        view_115 = value_states_56.view(1, -1, 12, 64)
        value_states_56 = None
        value_states_57 = view_115.transpose(1, 2)
        view_115 = None
        transpose_143 = key_states_57.transpose(3, 2)
        key_states_57 = None
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
        attn_weights_57 = value_states_57 = None
        transpose_144 = attn_output_112.transpose(1, 2)
        attn_output_112 = None
        attn_output_113 = transpose_144.contiguous()
        transpose_144 = None
        attn_output_114 = attn_output_113.view(1, -1, 768)
        attn_output_113 = None
        attn_output_115 = torch._C._nn.linear(
            attn_output_114,
            l_self_modules_block_modules_28_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_114 = l_self_modules_block_modules_28_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_114 = torch.nn.functional.dropout(attn_output_115, 0.1, False, False)
        attn_output_115 = None
        hidden_states_226 = hidden_states_224 + dropout_114
        hidden_states_224 = dropout_114 = None
        to_61 = hidden_states_226.to(torch.float32)
        pow_58 = to_61.pow(2)
        to_61 = None
        variance_57 = pow_58.mean(-1, keepdim=True)
        pow_58 = None
        add_118 = variance_57 + 1e-06
        variance_57 = None
        rsqrt_57 = torch.rsqrt(add_118)
        add_118 = None
        hidden_states_227 = hidden_states_226 * rsqrt_57
        rsqrt_57 = None
        forwarded_states_28 = (
            l_self_modules_block_modules_28_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_227
        )
        l_self_modules_block_modules_28_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_227
        ) = None
        hidden_states_228 = torch._C._nn.linear(
            forwarded_states_28,
            l_self_modules_block_modules_28_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_28 = l_self_modules_block_modules_28_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_229 = torch.nn.functional.relu(hidden_states_228, inplace=False)
        hidden_states_228 = None
        hidden_states_230 = torch.nn.functional.dropout(
            hidden_states_229, 0.1, False, False
        )
        hidden_states_229 = None
        hidden_states_231 = torch._C._nn.linear(
            hidden_states_230,
            l_self_modules_block_modules_28_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_230 = l_self_modules_block_modules_28_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_116 = torch.nn.functional.dropout(hidden_states_231, 0.1, False, False)
        hidden_states_231 = None
        hidden_states_232 = hidden_states_226 + dropout_116
        hidden_states_226 = dropout_116 = None
        to_62 = hidden_states_232.to(torch.float32)
        pow_59 = to_62.pow(2)
        to_62 = None
        variance_58 = pow_59.mean(-1, keepdim=True)
        pow_59 = None
        add_120 = variance_58 + 1e-06
        variance_58 = None
        rsqrt_58 = torch.rsqrt(add_120)
        add_120 = None
        hidden_states_233 = hidden_states_232 * rsqrt_58
        rsqrt_58 = None
        normed_hidden_states_29 = (
            l_self_modules_block_modules_29_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_233
        )
        l_self_modules_block_modules_29_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_233
        ) = None
        query_states_58 = torch._C._nn.linear(
            normed_hidden_states_29,
            l_self_modules_block_modules_29_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_29_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_117 = query_states_58.view(1, -1, 12, 64)
        query_states_58 = None
        query_states_59 = view_117.transpose(1, 2)
        view_117 = None
        key_states_58 = torch._C._nn.linear(
            normed_hidden_states_29,
            l_self_modules_block_modules_29_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_29_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_58 = torch._C._nn.linear(
            normed_hidden_states_29,
            l_self_modules_block_modules_29_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_29 = l_self_modules_block_modules_29_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_118 = key_states_58.view(1, -1, 12, 64)
        key_states_58 = None
        key_states_59 = view_118.transpose(1, 2)
        view_118 = None
        view_119 = value_states_58.view(1, -1, 12, 64)
        value_states_58 = None
        value_states_59 = view_119.transpose(1, 2)
        view_119 = None
        transpose_148 = key_states_59.transpose(3, 2)
        key_states_59 = None
        scores_58 = torch.matmul(query_states_59, transpose_148)
        query_states_59 = transpose_148 = None
        scores_58 += position_bias_1
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
        attn_weights_59 = value_states_59 = None
        transpose_149 = attn_output_116.transpose(1, 2)
        attn_output_116 = None
        attn_output_117 = transpose_149.contiguous()
        transpose_149 = None
        attn_output_118 = attn_output_117.view(1, -1, 768)
        attn_output_117 = None
        attn_output_119 = torch._C._nn.linear(
            attn_output_118,
            l_self_modules_block_modules_29_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_118 = l_self_modules_block_modules_29_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_118 = torch.nn.functional.dropout(attn_output_119, 0.1, False, False)
        attn_output_119 = None
        hidden_states_234 = hidden_states_232 + dropout_118
        hidden_states_232 = dropout_118 = None
        to_63 = hidden_states_234.to(torch.float32)
        pow_60 = to_63.pow(2)
        to_63 = None
        variance_59 = pow_60.mean(-1, keepdim=True)
        pow_60 = None
        add_122 = variance_59 + 1e-06
        variance_59 = None
        rsqrt_59 = torch.rsqrt(add_122)
        add_122 = None
        hidden_states_235 = hidden_states_234 * rsqrt_59
        rsqrt_59 = None
        forwarded_states_29 = (
            l_self_modules_block_modules_29_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_235
        )
        l_self_modules_block_modules_29_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_235
        ) = None
        hidden_states_236 = torch._C._nn.linear(
            forwarded_states_29,
            l_self_modules_block_modules_29_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_29 = l_self_modules_block_modules_29_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_237 = torch.nn.functional.relu(hidden_states_236, inplace=False)
        hidden_states_236 = None
        hidden_states_238 = torch.nn.functional.dropout(
            hidden_states_237, 0.1, False, False
        )
        hidden_states_237 = None
        hidden_states_239 = torch._C._nn.linear(
            hidden_states_238,
            l_self_modules_block_modules_29_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_238 = l_self_modules_block_modules_29_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_120 = torch.nn.functional.dropout(hidden_states_239, 0.1, False, False)
        hidden_states_239 = None
        hidden_states_240 = hidden_states_234 + dropout_120
        hidden_states_234 = dropout_120 = None
        to_64 = hidden_states_240.to(torch.float32)
        pow_61 = to_64.pow(2)
        to_64 = None
        variance_60 = pow_61.mean(-1, keepdim=True)
        pow_61 = None
        add_124 = variance_60 + 1e-06
        variance_60 = None
        rsqrt_60 = torch.rsqrt(add_124)
        add_124 = None
        hidden_states_241 = hidden_states_240 * rsqrt_60
        rsqrt_60 = None
        normed_hidden_states_30 = (
            l_self_modules_block_modules_30_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_241
        )
        l_self_modules_block_modules_30_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_241
        ) = None
        query_states_60 = torch._C._nn.linear(
            normed_hidden_states_30,
            l_self_modules_block_modules_30_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_30_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_121 = query_states_60.view(1, -1, 12, 64)
        query_states_60 = None
        query_states_61 = view_121.transpose(1, 2)
        view_121 = None
        key_states_60 = torch._C._nn.linear(
            normed_hidden_states_30,
            l_self_modules_block_modules_30_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_30_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_60 = torch._C._nn.linear(
            normed_hidden_states_30,
            l_self_modules_block_modules_30_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_30 = l_self_modules_block_modules_30_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_122 = key_states_60.view(1, -1, 12, 64)
        key_states_60 = None
        key_states_61 = view_122.transpose(1, 2)
        view_122 = None
        view_123 = value_states_60.view(1, -1, 12, 64)
        value_states_60 = None
        value_states_61 = view_123.transpose(1, 2)
        view_123 = None
        transpose_153 = key_states_61.transpose(3, 2)
        key_states_61 = None
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
        attn_weights_61 = value_states_61 = None
        transpose_154 = attn_output_120.transpose(1, 2)
        attn_output_120 = None
        attn_output_121 = transpose_154.contiguous()
        transpose_154 = None
        attn_output_122 = attn_output_121.view(1, -1, 768)
        attn_output_121 = None
        attn_output_123 = torch._C._nn.linear(
            attn_output_122,
            l_self_modules_block_modules_30_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_122 = l_self_modules_block_modules_30_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_122 = torch.nn.functional.dropout(attn_output_123, 0.1, False, False)
        attn_output_123 = None
        hidden_states_242 = hidden_states_240 + dropout_122
        hidden_states_240 = dropout_122 = None
        to_65 = hidden_states_242.to(torch.float32)
        pow_62 = to_65.pow(2)
        to_65 = None
        variance_61 = pow_62.mean(-1, keepdim=True)
        pow_62 = None
        add_126 = variance_61 + 1e-06
        variance_61 = None
        rsqrt_61 = torch.rsqrt(add_126)
        add_126 = None
        hidden_states_243 = hidden_states_242 * rsqrt_61
        rsqrt_61 = None
        forwarded_states_30 = (
            l_self_modules_block_modules_30_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_243
        )
        l_self_modules_block_modules_30_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_243
        ) = None
        hidden_states_244 = torch._C._nn.linear(
            forwarded_states_30,
            l_self_modules_block_modules_30_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_30 = l_self_modules_block_modules_30_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_245 = torch.nn.functional.relu(hidden_states_244, inplace=False)
        hidden_states_244 = None
        hidden_states_246 = torch.nn.functional.dropout(
            hidden_states_245, 0.1, False, False
        )
        hidden_states_245 = None
        hidden_states_247 = torch._C._nn.linear(
            hidden_states_246,
            l_self_modules_block_modules_30_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_246 = l_self_modules_block_modules_30_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_124 = torch.nn.functional.dropout(hidden_states_247, 0.1, False, False)
        hidden_states_247 = None
        hidden_states_248 = hidden_states_242 + dropout_124
        hidden_states_242 = dropout_124 = None
        to_66 = hidden_states_248.to(torch.float32)
        pow_63 = to_66.pow(2)
        to_66 = None
        variance_62 = pow_63.mean(-1, keepdim=True)
        pow_63 = None
        add_128 = variance_62 + 1e-06
        variance_62 = None
        rsqrt_62 = torch.rsqrt(add_128)
        add_128 = None
        hidden_states_249 = hidden_states_248 * rsqrt_62
        rsqrt_62 = None
        normed_hidden_states_31 = (
            l_self_modules_block_modules_31_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_249
        )
        l_self_modules_block_modules_31_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_249
        ) = None
        query_states_62 = torch._C._nn.linear(
            normed_hidden_states_31,
            l_self_modules_block_modules_31_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_31_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_125 = query_states_62.view(1, -1, 12, 64)
        query_states_62 = None
        query_states_63 = view_125.transpose(1, 2)
        view_125 = None
        key_states_62 = torch._C._nn.linear(
            normed_hidden_states_31,
            l_self_modules_block_modules_31_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_31_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_62 = torch._C._nn.linear(
            normed_hidden_states_31,
            l_self_modules_block_modules_31_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_31 = l_self_modules_block_modules_31_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_126 = key_states_62.view(1, -1, 12, 64)
        key_states_62 = None
        key_states_63 = view_126.transpose(1, 2)
        view_126 = None
        view_127 = value_states_62.view(1, -1, 12, 64)
        value_states_62 = None
        value_states_63 = view_127.transpose(1, 2)
        view_127 = None
        transpose_158 = key_states_63.transpose(3, 2)
        key_states_63 = None
        scores_62 = torch.matmul(query_states_63, transpose_158)
        query_states_63 = transpose_158 = None
        scores_62 += position_bias_1
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
        attn_weights_63 = value_states_63 = None
        transpose_159 = attn_output_124.transpose(1, 2)
        attn_output_124 = None
        attn_output_125 = transpose_159.contiguous()
        transpose_159 = None
        attn_output_126 = attn_output_125.view(1, -1, 768)
        attn_output_125 = None
        attn_output_127 = torch._C._nn.linear(
            attn_output_126,
            l_self_modules_block_modules_31_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_126 = l_self_modules_block_modules_31_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_126 = torch.nn.functional.dropout(attn_output_127, 0.1, False, False)
        attn_output_127 = None
        hidden_states_250 = hidden_states_248 + dropout_126
        hidden_states_248 = dropout_126 = None
        to_67 = hidden_states_250.to(torch.float32)
        pow_64 = to_67.pow(2)
        to_67 = None
        variance_63 = pow_64.mean(-1, keepdim=True)
        pow_64 = None
        add_130 = variance_63 + 1e-06
        variance_63 = None
        rsqrt_63 = torch.rsqrt(add_130)
        add_130 = None
        hidden_states_251 = hidden_states_250 * rsqrt_63
        rsqrt_63 = None
        forwarded_states_31 = (
            l_self_modules_block_modules_31_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_251
        )
        l_self_modules_block_modules_31_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_251
        ) = None
        hidden_states_252 = torch._C._nn.linear(
            forwarded_states_31,
            l_self_modules_block_modules_31_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_31 = l_self_modules_block_modules_31_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_253 = torch.nn.functional.relu(hidden_states_252, inplace=False)
        hidden_states_252 = None
        hidden_states_254 = torch.nn.functional.dropout(
            hidden_states_253, 0.1, False, False
        )
        hidden_states_253 = None
        hidden_states_255 = torch._C._nn.linear(
            hidden_states_254,
            l_self_modules_block_modules_31_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_254 = l_self_modules_block_modules_31_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_128 = torch.nn.functional.dropout(hidden_states_255, 0.1, False, False)
        hidden_states_255 = None
        hidden_states_256 = hidden_states_250 + dropout_128
        hidden_states_250 = dropout_128 = None
        to_68 = hidden_states_256.to(torch.float32)
        pow_65 = to_68.pow(2)
        to_68 = None
        variance_64 = pow_65.mean(-1, keepdim=True)
        pow_65 = None
        add_132 = variance_64 + 1e-06
        variance_64 = None
        rsqrt_64 = torch.rsqrt(add_132)
        add_132 = None
        hidden_states_257 = hidden_states_256 * rsqrt_64
        rsqrt_64 = None
        normed_hidden_states_32 = (
            l_self_modules_block_modules_32_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_257
        )
        l_self_modules_block_modules_32_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_257
        ) = None
        query_states_64 = torch._C._nn.linear(
            normed_hidden_states_32,
            l_self_modules_block_modules_32_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_32_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_129 = query_states_64.view(1, -1, 12, 64)
        query_states_64 = None
        query_states_65 = view_129.transpose(1, 2)
        view_129 = None
        key_states_64 = torch._C._nn.linear(
            normed_hidden_states_32,
            l_self_modules_block_modules_32_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_32_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_64 = torch._C._nn.linear(
            normed_hidden_states_32,
            l_self_modules_block_modules_32_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_32 = l_self_modules_block_modules_32_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_130 = key_states_64.view(1, -1, 12, 64)
        key_states_64 = None
        key_states_65 = view_130.transpose(1, 2)
        view_130 = None
        view_131 = value_states_64.view(1, -1, 12, 64)
        value_states_64 = None
        value_states_65 = view_131.transpose(1, 2)
        view_131 = None
        transpose_163 = key_states_65.transpose(3, 2)
        key_states_65 = None
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
        attn_weights_65 = value_states_65 = None
        transpose_164 = attn_output_128.transpose(1, 2)
        attn_output_128 = None
        attn_output_129 = transpose_164.contiguous()
        transpose_164 = None
        attn_output_130 = attn_output_129.view(1, -1, 768)
        attn_output_129 = None
        attn_output_131 = torch._C._nn.linear(
            attn_output_130,
            l_self_modules_block_modules_32_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_130 = l_self_modules_block_modules_32_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_130 = torch.nn.functional.dropout(attn_output_131, 0.1, False, False)
        attn_output_131 = None
        hidden_states_258 = hidden_states_256 + dropout_130
        hidden_states_256 = dropout_130 = None
        to_69 = hidden_states_258.to(torch.float32)
        pow_66 = to_69.pow(2)
        to_69 = None
        variance_65 = pow_66.mean(-1, keepdim=True)
        pow_66 = None
        add_134 = variance_65 + 1e-06
        variance_65 = None
        rsqrt_65 = torch.rsqrt(add_134)
        add_134 = None
        hidden_states_259 = hidden_states_258 * rsqrt_65
        rsqrt_65 = None
        forwarded_states_32 = (
            l_self_modules_block_modules_32_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_259
        )
        l_self_modules_block_modules_32_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_259
        ) = None
        hidden_states_260 = torch._C._nn.linear(
            forwarded_states_32,
            l_self_modules_block_modules_32_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_32 = l_self_modules_block_modules_32_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_261 = torch.nn.functional.relu(hidden_states_260, inplace=False)
        hidden_states_260 = None
        hidden_states_262 = torch.nn.functional.dropout(
            hidden_states_261, 0.1, False, False
        )
        hidden_states_261 = None
        hidden_states_263 = torch._C._nn.linear(
            hidden_states_262,
            l_self_modules_block_modules_32_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_262 = l_self_modules_block_modules_32_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_132 = torch.nn.functional.dropout(hidden_states_263, 0.1, False, False)
        hidden_states_263 = None
        hidden_states_264 = hidden_states_258 + dropout_132
        hidden_states_258 = dropout_132 = None
        to_70 = hidden_states_264.to(torch.float32)
        pow_67 = to_70.pow(2)
        to_70 = None
        variance_66 = pow_67.mean(-1, keepdim=True)
        pow_67 = None
        add_136 = variance_66 + 1e-06
        variance_66 = None
        rsqrt_66 = torch.rsqrt(add_136)
        add_136 = None
        hidden_states_265 = hidden_states_264 * rsqrt_66
        rsqrt_66 = None
        normed_hidden_states_33 = (
            l_self_modules_block_modules_33_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_265
        )
        l_self_modules_block_modules_33_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_265
        ) = None
        query_states_66 = torch._C._nn.linear(
            normed_hidden_states_33,
            l_self_modules_block_modules_33_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_33_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_133 = query_states_66.view(1, -1, 12, 64)
        query_states_66 = None
        query_states_67 = view_133.transpose(1, 2)
        view_133 = None
        key_states_66 = torch._C._nn.linear(
            normed_hidden_states_33,
            l_self_modules_block_modules_33_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_33_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_66 = torch._C._nn.linear(
            normed_hidden_states_33,
            l_self_modules_block_modules_33_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_33 = l_self_modules_block_modules_33_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_134 = key_states_66.view(1, -1, 12, 64)
        key_states_66 = None
        key_states_67 = view_134.transpose(1, 2)
        view_134 = None
        view_135 = value_states_66.view(1, -1, 12, 64)
        value_states_66 = None
        value_states_67 = view_135.transpose(1, 2)
        view_135 = None
        transpose_168 = key_states_67.transpose(3, 2)
        key_states_67 = None
        scores_66 = torch.matmul(query_states_67, transpose_168)
        query_states_67 = transpose_168 = None
        scores_66 += position_bias_1
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
        attn_weights_67 = value_states_67 = None
        transpose_169 = attn_output_132.transpose(1, 2)
        attn_output_132 = None
        attn_output_133 = transpose_169.contiguous()
        transpose_169 = None
        attn_output_134 = attn_output_133.view(1, -1, 768)
        attn_output_133 = None
        attn_output_135 = torch._C._nn.linear(
            attn_output_134,
            l_self_modules_block_modules_33_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_134 = l_self_modules_block_modules_33_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_134 = torch.nn.functional.dropout(attn_output_135, 0.1, False, False)
        attn_output_135 = None
        hidden_states_266 = hidden_states_264 + dropout_134
        hidden_states_264 = dropout_134 = None
        to_71 = hidden_states_266.to(torch.float32)
        pow_68 = to_71.pow(2)
        to_71 = None
        variance_67 = pow_68.mean(-1, keepdim=True)
        pow_68 = None
        add_138 = variance_67 + 1e-06
        variance_67 = None
        rsqrt_67 = torch.rsqrt(add_138)
        add_138 = None
        hidden_states_267 = hidden_states_266 * rsqrt_67
        rsqrt_67 = None
        forwarded_states_33 = (
            l_self_modules_block_modules_33_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_267
        )
        l_self_modules_block_modules_33_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_267
        ) = None
        hidden_states_268 = torch._C._nn.linear(
            forwarded_states_33,
            l_self_modules_block_modules_33_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_33 = l_self_modules_block_modules_33_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_269 = torch.nn.functional.relu(hidden_states_268, inplace=False)
        hidden_states_268 = None
        hidden_states_270 = torch.nn.functional.dropout(
            hidden_states_269, 0.1, False, False
        )
        hidden_states_269 = None
        hidden_states_271 = torch._C._nn.linear(
            hidden_states_270,
            l_self_modules_block_modules_33_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_270 = l_self_modules_block_modules_33_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_136 = torch.nn.functional.dropout(hidden_states_271, 0.1, False, False)
        hidden_states_271 = None
        hidden_states_272 = hidden_states_266 + dropout_136
        hidden_states_266 = dropout_136 = None
        to_72 = hidden_states_272.to(torch.float32)
        pow_69 = to_72.pow(2)
        to_72 = None
        variance_68 = pow_69.mean(-1, keepdim=True)
        pow_69 = None
        add_140 = variance_68 + 1e-06
        variance_68 = None
        rsqrt_68 = torch.rsqrt(add_140)
        add_140 = None
        hidden_states_273 = hidden_states_272 * rsqrt_68
        rsqrt_68 = None
        normed_hidden_states_34 = (
            l_self_modules_block_modules_34_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_273
        )
        l_self_modules_block_modules_34_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_273
        ) = None
        query_states_68 = torch._C._nn.linear(
            normed_hidden_states_34,
            l_self_modules_block_modules_34_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_34_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_137 = query_states_68.view(1, -1, 12, 64)
        query_states_68 = None
        query_states_69 = view_137.transpose(1, 2)
        view_137 = None
        key_states_68 = torch._C._nn.linear(
            normed_hidden_states_34,
            l_self_modules_block_modules_34_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_34_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_68 = torch._C._nn.linear(
            normed_hidden_states_34,
            l_self_modules_block_modules_34_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_34 = l_self_modules_block_modules_34_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_138 = key_states_68.view(1, -1, 12, 64)
        key_states_68 = None
        key_states_69 = view_138.transpose(1, 2)
        view_138 = None
        view_139 = value_states_68.view(1, -1, 12, 64)
        value_states_68 = None
        value_states_69 = view_139.transpose(1, 2)
        view_139 = None
        transpose_173 = key_states_69.transpose(3, 2)
        key_states_69 = None
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
        attn_weights_69 = value_states_69 = None
        transpose_174 = attn_output_136.transpose(1, 2)
        attn_output_136 = None
        attn_output_137 = transpose_174.contiguous()
        transpose_174 = None
        attn_output_138 = attn_output_137.view(1, -1, 768)
        attn_output_137 = None
        attn_output_139 = torch._C._nn.linear(
            attn_output_138,
            l_self_modules_block_modules_34_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_138 = l_self_modules_block_modules_34_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_138 = torch.nn.functional.dropout(attn_output_139, 0.1, False, False)
        attn_output_139 = None
        hidden_states_274 = hidden_states_272 + dropout_138
        hidden_states_272 = dropout_138 = None
        to_73 = hidden_states_274.to(torch.float32)
        pow_70 = to_73.pow(2)
        to_73 = None
        variance_69 = pow_70.mean(-1, keepdim=True)
        pow_70 = None
        add_142 = variance_69 + 1e-06
        variance_69 = None
        rsqrt_69 = torch.rsqrt(add_142)
        add_142 = None
        hidden_states_275 = hidden_states_274 * rsqrt_69
        rsqrt_69 = None
        forwarded_states_34 = (
            l_self_modules_block_modules_34_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_275
        )
        l_self_modules_block_modules_34_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_275
        ) = None
        hidden_states_276 = torch._C._nn.linear(
            forwarded_states_34,
            l_self_modules_block_modules_34_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_34 = l_self_modules_block_modules_34_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_277 = torch.nn.functional.relu(hidden_states_276, inplace=False)
        hidden_states_276 = None
        hidden_states_278 = torch.nn.functional.dropout(
            hidden_states_277, 0.1, False, False
        )
        hidden_states_277 = None
        hidden_states_279 = torch._C._nn.linear(
            hidden_states_278,
            l_self_modules_block_modules_34_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_278 = l_self_modules_block_modules_34_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_140 = torch.nn.functional.dropout(hidden_states_279, 0.1, False, False)
        hidden_states_279 = None
        hidden_states_280 = hidden_states_274 + dropout_140
        hidden_states_274 = dropout_140 = None
        to_74 = hidden_states_280.to(torch.float32)
        pow_71 = to_74.pow(2)
        to_74 = None
        variance_70 = pow_71.mean(-1, keepdim=True)
        pow_71 = None
        add_144 = variance_70 + 1e-06
        variance_70 = None
        rsqrt_70 = torch.rsqrt(add_144)
        add_144 = None
        hidden_states_281 = hidden_states_280 * rsqrt_70
        rsqrt_70 = None
        normed_hidden_states_35 = (
            l_self_modules_block_modules_35_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_281
        )
        l_self_modules_block_modules_35_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_281
        ) = None
        query_states_70 = torch._C._nn.linear(
            normed_hidden_states_35,
            l_self_modules_block_modules_35_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_35_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_141 = query_states_70.view(1, -1, 12, 64)
        query_states_70 = None
        query_states_71 = view_141.transpose(1, 2)
        view_141 = None
        key_states_70 = torch._C._nn.linear(
            normed_hidden_states_35,
            l_self_modules_block_modules_35_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_35_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_70 = torch._C._nn.linear(
            normed_hidden_states_35,
            l_self_modules_block_modules_35_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_35 = l_self_modules_block_modules_35_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_142 = key_states_70.view(1, -1, 12, 64)
        key_states_70 = None
        key_states_71 = view_142.transpose(1, 2)
        view_142 = None
        view_143 = value_states_70.view(1, -1, 12, 64)
        value_states_70 = None
        value_states_71 = view_143.transpose(1, 2)
        view_143 = None
        transpose_178 = key_states_71.transpose(3, 2)
        key_states_71 = None
        scores_70 = torch.matmul(query_states_71, transpose_178)
        query_states_71 = transpose_178 = None
        scores_70 += position_bias_1
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
        attn_weights_71 = value_states_71 = None
        transpose_179 = attn_output_140.transpose(1, 2)
        attn_output_140 = None
        attn_output_141 = transpose_179.contiguous()
        transpose_179 = None
        attn_output_142 = attn_output_141.view(1, -1, 768)
        attn_output_141 = None
        attn_output_143 = torch._C._nn.linear(
            attn_output_142,
            l_self_modules_block_modules_35_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_142 = l_self_modules_block_modules_35_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_142 = torch.nn.functional.dropout(attn_output_143, 0.1, False, False)
        attn_output_143 = None
        hidden_states_282 = hidden_states_280 + dropout_142
        hidden_states_280 = dropout_142 = None
        to_75 = hidden_states_282.to(torch.float32)
        pow_72 = to_75.pow(2)
        to_75 = None
        variance_71 = pow_72.mean(-1, keepdim=True)
        pow_72 = None
        add_146 = variance_71 + 1e-06
        variance_71 = None
        rsqrt_71 = torch.rsqrt(add_146)
        add_146 = None
        hidden_states_283 = hidden_states_282 * rsqrt_71
        rsqrt_71 = None
        forwarded_states_35 = (
            l_self_modules_block_modules_35_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_283
        )
        l_self_modules_block_modules_35_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_283
        ) = None
        hidden_states_284 = torch._C._nn.linear(
            forwarded_states_35,
            l_self_modules_block_modules_35_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_35 = l_self_modules_block_modules_35_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_285 = torch.nn.functional.relu(hidden_states_284, inplace=False)
        hidden_states_284 = None
        hidden_states_286 = torch.nn.functional.dropout(
            hidden_states_285, 0.1, False, False
        )
        hidden_states_285 = None
        hidden_states_287 = torch._C._nn.linear(
            hidden_states_286,
            l_self_modules_block_modules_35_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_286 = l_self_modules_block_modules_35_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_144 = torch.nn.functional.dropout(hidden_states_287, 0.1, False, False)
        hidden_states_287 = None
        hidden_states_288 = hidden_states_282 + dropout_144
        hidden_states_282 = dropout_144 = None
        to_76 = hidden_states_288.to(torch.float32)
        pow_73 = to_76.pow(2)
        to_76 = None
        variance_72 = pow_73.mean(-1, keepdim=True)
        pow_73 = None
        add_148 = variance_72 + 1e-06
        variance_72 = None
        rsqrt_72 = torch.rsqrt(add_148)
        add_148 = None
        hidden_states_289 = hidden_states_288 * rsqrt_72
        rsqrt_72 = None
        normed_hidden_states_36 = (
            l_self_modules_block_modules_36_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_289
        )
        l_self_modules_block_modules_36_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_289
        ) = None
        query_states_72 = torch._C._nn.linear(
            normed_hidden_states_36,
            l_self_modules_block_modules_36_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_36_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_145 = query_states_72.view(1, -1, 12, 64)
        query_states_72 = None
        query_states_73 = view_145.transpose(1, 2)
        view_145 = None
        key_states_72 = torch._C._nn.linear(
            normed_hidden_states_36,
            l_self_modules_block_modules_36_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_36_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_72 = torch._C._nn.linear(
            normed_hidden_states_36,
            l_self_modules_block_modules_36_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_36 = l_self_modules_block_modules_36_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_146 = key_states_72.view(1, -1, 12, 64)
        key_states_72 = None
        key_states_73 = view_146.transpose(1, 2)
        view_146 = None
        view_147 = value_states_72.view(1, -1, 12, 64)
        value_states_72 = None
        value_states_73 = view_147.transpose(1, 2)
        view_147 = None
        transpose_183 = key_states_73.transpose(3, 2)
        key_states_73 = None
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
        attn_weights_73 = value_states_73 = None
        transpose_184 = attn_output_144.transpose(1, 2)
        attn_output_144 = None
        attn_output_145 = transpose_184.contiguous()
        transpose_184 = None
        attn_output_146 = attn_output_145.view(1, -1, 768)
        attn_output_145 = None
        attn_output_147 = torch._C._nn.linear(
            attn_output_146,
            l_self_modules_block_modules_36_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_146 = l_self_modules_block_modules_36_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_146 = torch.nn.functional.dropout(attn_output_147, 0.1, False, False)
        attn_output_147 = None
        hidden_states_290 = hidden_states_288 + dropout_146
        hidden_states_288 = dropout_146 = None
        to_77 = hidden_states_290.to(torch.float32)
        pow_74 = to_77.pow(2)
        to_77 = None
        variance_73 = pow_74.mean(-1, keepdim=True)
        pow_74 = None
        add_150 = variance_73 + 1e-06
        variance_73 = None
        rsqrt_73 = torch.rsqrt(add_150)
        add_150 = None
        hidden_states_291 = hidden_states_290 * rsqrt_73
        rsqrt_73 = None
        forwarded_states_36 = (
            l_self_modules_block_modules_36_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_291
        )
        l_self_modules_block_modules_36_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_291
        ) = None
        hidden_states_292 = torch._C._nn.linear(
            forwarded_states_36,
            l_self_modules_block_modules_36_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_36 = l_self_modules_block_modules_36_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_293 = torch.nn.functional.relu(hidden_states_292, inplace=False)
        hidden_states_292 = None
        hidden_states_294 = torch.nn.functional.dropout(
            hidden_states_293, 0.1, False, False
        )
        hidden_states_293 = None
        hidden_states_295 = torch._C._nn.linear(
            hidden_states_294,
            l_self_modules_block_modules_36_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_294 = l_self_modules_block_modules_36_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_148 = torch.nn.functional.dropout(hidden_states_295, 0.1, False, False)
        hidden_states_295 = None
        hidden_states_296 = hidden_states_290 + dropout_148
        hidden_states_290 = dropout_148 = None
        to_78 = hidden_states_296.to(torch.float32)
        pow_75 = to_78.pow(2)
        to_78 = None
        variance_74 = pow_75.mean(-1, keepdim=True)
        pow_75 = None
        add_152 = variance_74 + 1e-06
        variance_74 = None
        rsqrt_74 = torch.rsqrt(add_152)
        add_152 = None
        hidden_states_297 = hidden_states_296 * rsqrt_74
        rsqrt_74 = None
        normed_hidden_states_37 = (
            l_self_modules_block_modules_37_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_297
        )
        l_self_modules_block_modules_37_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_297
        ) = None
        query_states_74 = torch._C._nn.linear(
            normed_hidden_states_37,
            l_self_modules_block_modules_37_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_37_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_149 = query_states_74.view(1, -1, 12, 64)
        query_states_74 = None
        query_states_75 = view_149.transpose(1, 2)
        view_149 = None
        key_states_74 = torch._C._nn.linear(
            normed_hidden_states_37,
            l_self_modules_block_modules_37_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_37_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_74 = torch._C._nn.linear(
            normed_hidden_states_37,
            l_self_modules_block_modules_37_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_37 = l_self_modules_block_modules_37_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_150 = key_states_74.view(1, -1, 12, 64)
        key_states_74 = None
        key_states_75 = view_150.transpose(1, 2)
        view_150 = None
        view_151 = value_states_74.view(1, -1, 12, 64)
        value_states_74 = None
        value_states_75 = view_151.transpose(1, 2)
        view_151 = None
        transpose_188 = key_states_75.transpose(3, 2)
        key_states_75 = None
        scores_74 = torch.matmul(query_states_75, transpose_188)
        query_states_75 = transpose_188 = None
        scores_74 += position_bias_1
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
        attn_weights_75 = value_states_75 = None
        transpose_189 = attn_output_148.transpose(1, 2)
        attn_output_148 = None
        attn_output_149 = transpose_189.contiguous()
        transpose_189 = None
        attn_output_150 = attn_output_149.view(1, -1, 768)
        attn_output_149 = None
        attn_output_151 = torch._C._nn.linear(
            attn_output_150,
            l_self_modules_block_modules_37_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_150 = l_self_modules_block_modules_37_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_150 = torch.nn.functional.dropout(attn_output_151, 0.1, False, False)
        attn_output_151 = None
        hidden_states_298 = hidden_states_296 + dropout_150
        hidden_states_296 = dropout_150 = None
        to_79 = hidden_states_298.to(torch.float32)
        pow_76 = to_79.pow(2)
        to_79 = None
        variance_75 = pow_76.mean(-1, keepdim=True)
        pow_76 = None
        add_154 = variance_75 + 1e-06
        variance_75 = None
        rsqrt_75 = torch.rsqrt(add_154)
        add_154 = None
        hidden_states_299 = hidden_states_298 * rsqrt_75
        rsqrt_75 = None
        forwarded_states_37 = (
            l_self_modules_block_modules_37_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_299
        )
        l_self_modules_block_modules_37_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_299
        ) = None
        hidden_states_300 = torch._C._nn.linear(
            forwarded_states_37,
            l_self_modules_block_modules_37_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_37 = l_self_modules_block_modules_37_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_301 = torch.nn.functional.relu(hidden_states_300, inplace=False)
        hidden_states_300 = None
        hidden_states_302 = torch.nn.functional.dropout(
            hidden_states_301, 0.1, False, False
        )
        hidden_states_301 = None
        hidden_states_303 = torch._C._nn.linear(
            hidden_states_302,
            l_self_modules_block_modules_37_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_302 = l_self_modules_block_modules_37_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_152 = torch.nn.functional.dropout(hidden_states_303, 0.1, False, False)
        hidden_states_303 = None
        hidden_states_304 = hidden_states_298 + dropout_152
        hidden_states_298 = dropout_152 = None
        to_80 = hidden_states_304.to(torch.float32)
        pow_77 = to_80.pow(2)
        to_80 = None
        variance_76 = pow_77.mean(-1, keepdim=True)
        pow_77 = None
        add_156 = variance_76 + 1e-06
        variance_76 = None
        rsqrt_76 = torch.rsqrt(add_156)
        add_156 = None
        hidden_states_305 = hidden_states_304 * rsqrt_76
        rsqrt_76 = None
        normed_hidden_states_38 = (
            l_self_modules_block_modules_38_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_305
        )
        l_self_modules_block_modules_38_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_305
        ) = None
        query_states_76 = torch._C._nn.linear(
            normed_hidden_states_38,
            l_self_modules_block_modules_38_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_38_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_153 = query_states_76.view(1, -1, 12, 64)
        query_states_76 = None
        query_states_77 = view_153.transpose(1, 2)
        view_153 = None
        key_states_76 = torch._C._nn.linear(
            normed_hidden_states_38,
            l_self_modules_block_modules_38_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_38_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_76 = torch._C._nn.linear(
            normed_hidden_states_38,
            l_self_modules_block_modules_38_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_38 = l_self_modules_block_modules_38_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_154 = key_states_76.view(1, -1, 12, 64)
        key_states_76 = None
        key_states_77 = view_154.transpose(1, 2)
        view_154 = None
        view_155 = value_states_76.view(1, -1, 12, 64)
        value_states_76 = None
        value_states_77 = view_155.transpose(1, 2)
        view_155 = None
        transpose_193 = key_states_77.transpose(3, 2)
        key_states_77 = None
        scores_76 = torch.matmul(query_states_77, transpose_193)
        query_states_77 = transpose_193 = None
        scores_76 += position_bias_1
        scores_77 = scores_76
        scores_76 = None
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
        attn_weights_77 = value_states_77 = None
        transpose_194 = attn_output_152.transpose(1, 2)
        attn_output_152 = None
        attn_output_153 = transpose_194.contiguous()
        transpose_194 = None
        attn_output_154 = attn_output_153.view(1, -1, 768)
        attn_output_153 = None
        attn_output_155 = torch._C._nn.linear(
            attn_output_154,
            l_self_modules_block_modules_38_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_154 = l_self_modules_block_modules_38_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_154 = torch.nn.functional.dropout(attn_output_155, 0.1, False, False)
        attn_output_155 = None
        hidden_states_306 = hidden_states_304 + dropout_154
        hidden_states_304 = dropout_154 = None
        to_81 = hidden_states_306.to(torch.float32)
        pow_78 = to_81.pow(2)
        to_81 = None
        variance_77 = pow_78.mean(-1, keepdim=True)
        pow_78 = None
        add_158 = variance_77 + 1e-06
        variance_77 = None
        rsqrt_77 = torch.rsqrt(add_158)
        add_158 = None
        hidden_states_307 = hidden_states_306 * rsqrt_77
        rsqrt_77 = None
        forwarded_states_38 = (
            l_self_modules_block_modules_38_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_307
        )
        l_self_modules_block_modules_38_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_307
        ) = None
        hidden_states_308 = torch._C._nn.linear(
            forwarded_states_38,
            l_self_modules_block_modules_38_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_38 = l_self_modules_block_modules_38_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_309 = torch.nn.functional.relu(hidden_states_308, inplace=False)
        hidden_states_308 = None
        hidden_states_310 = torch.nn.functional.dropout(
            hidden_states_309, 0.1, False, False
        )
        hidden_states_309 = None
        hidden_states_311 = torch._C._nn.linear(
            hidden_states_310,
            l_self_modules_block_modules_38_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_310 = l_self_modules_block_modules_38_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_156 = torch.nn.functional.dropout(hidden_states_311, 0.1, False, False)
        hidden_states_311 = None
        hidden_states_312 = hidden_states_306 + dropout_156
        hidden_states_306 = dropout_156 = None
        to_82 = hidden_states_312.to(torch.float32)
        pow_79 = to_82.pow(2)
        to_82 = None
        variance_78 = pow_79.mean(-1, keepdim=True)
        pow_79 = None
        add_160 = variance_78 + 1e-06
        variance_78 = None
        rsqrt_78 = torch.rsqrt(add_160)
        add_160 = None
        hidden_states_313 = hidden_states_312 * rsqrt_78
        rsqrt_78 = None
        normed_hidden_states_39 = (
            l_self_modules_block_modules_39_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_313
        )
        l_self_modules_block_modules_39_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_313
        ) = None
        query_states_78 = torch._C._nn.linear(
            normed_hidden_states_39,
            l_self_modules_block_modules_39_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_39_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_157 = query_states_78.view(1, -1, 12, 64)
        query_states_78 = None
        query_states_79 = view_157.transpose(1, 2)
        view_157 = None
        key_states_78 = torch._C._nn.linear(
            normed_hidden_states_39,
            l_self_modules_block_modules_39_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_39_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_78 = torch._C._nn.linear(
            normed_hidden_states_39,
            l_self_modules_block_modules_39_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_39 = l_self_modules_block_modules_39_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_158 = key_states_78.view(1, -1, 12, 64)
        key_states_78 = None
        key_states_79 = view_158.transpose(1, 2)
        view_158 = None
        view_159 = value_states_78.view(1, -1, 12, 64)
        value_states_78 = None
        value_states_79 = view_159.transpose(1, 2)
        view_159 = None
        transpose_198 = key_states_79.transpose(3, 2)
        key_states_79 = None
        scores_78 = torch.matmul(query_states_79, transpose_198)
        query_states_79 = transpose_198 = None
        scores_78 += position_bias_1
        scores_79 = scores_78
        scores_78 = None
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
        attn_weights_79 = value_states_79 = None
        transpose_199 = attn_output_156.transpose(1, 2)
        attn_output_156 = None
        attn_output_157 = transpose_199.contiguous()
        transpose_199 = None
        attn_output_158 = attn_output_157.view(1, -1, 768)
        attn_output_157 = None
        attn_output_159 = torch._C._nn.linear(
            attn_output_158,
            l_self_modules_block_modules_39_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_158 = l_self_modules_block_modules_39_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_158 = torch.nn.functional.dropout(attn_output_159, 0.1, False, False)
        attn_output_159 = None
        hidden_states_314 = hidden_states_312 + dropout_158
        hidden_states_312 = dropout_158 = None
        to_83 = hidden_states_314.to(torch.float32)
        pow_80 = to_83.pow(2)
        to_83 = None
        variance_79 = pow_80.mean(-1, keepdim=True)
        pow_80 = None
        add_162 = variance_79 + 1e-06
        variance_79 = None
        rsqrt_79 = torch.rsqrt(add_162)
        add_162 = None
        hidden_states_315 = hidden_states_314 * rsqrt_79
        rsqrt_79 = None
        forwarded_states_39 = (
            l_self_modules_block_modules_39_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_315
        )
        l_self_modules_block_modules_39_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_315
        ) = None
        hidden_states_316 = torch._C._nn.linear(
            forwarded_states_39,
            l_self_modules_block_modules_39_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_39 = l_self_modules_block_modules_39_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_317 = torch.nn.functional.relu(hidden_states_316, inplace=False)
        hidden_states_316 = None
        hidden_states_318 = torch.nn.functional.dropout(
            hidden_states_317, 0.1, False, False
        )
        hidden_states_317 = None
        hidden_states_319 = torch._C._nn.linear(
            hidden_states_318,
            l_self_modules_block_modules_39_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_318 = l_self_modules_block_modules_39_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_160 = torch.nn.functional.dropout(hidden_states_319, 0.1, False, False)
        hidden_states_319 = None
        hidden_states_320 = hidden_states_314 + dropout_160
        hidden_states_314 = dropout_160 = None
        to_84 = hidden_states_320.to(torch.float32)
        pow_81 = to_84.pow(2)
        to_84 = None
        variance_80 = pow_81.mean(-1, keepdim=True)
        pow_81 = None
        add_164 = variance_80 + 1e-06
        variance_80 = None
        rsqrt_80 = torch.rsqrt(add_164)
        add_164 = None
        hidden_states_321 = hidden_states_320 * rsqrt_80
        rsqrt_80 = None
        normed_hidden_states_40 = (
            l_self_modules_block_modules_40_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_321
        )
        l_self_modules_block_modules_40_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_321
        ) = None
        query_states_80 = torch._C._nn.linear(
            normed_hidden_states_40,
            l_self_modules_block_modules_40_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_40_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_161 = query_states_80.view(1, -1, 12, 64)
        query_states_80 = None
        query_states_81 = view_161.transpose(1, 2)
        view_161 = None
        key_states_80 = torch._C._nn.linear(
            normed_hidden_states_40,
            l_self_modules_block_modules_40_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_40_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_80 = torch._C._nn.linear(
            normed_hidden_states_40,
            l_self_modules_block_modules_40_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_40 = l_self_modules_block_modules_40_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_162 = key_states_80.view(1, -1, 12, 64)
        key_states_80 = None
        key_states_81 = view_162.transpose(1, 2)
        view_162 = None
        view_163 = value_states_80.view(1, -1, 12, 64)
        value_states_80 = None
        value_states_81 = view_163.transpose(1, 2)
        view_163 = None
        transpose_203 = key_states_81.transpose(3, 2)
        key_states_81 = None
        scores_80 = torch.matmul(query_states_81, transpose_203)
        query_states_81 = transpose_203 = None
        scores_80 += position_bias_1
        scores_81 = scores_80
        scores_80 = None
        float_42 = scores_81.float()
        softmax_40 = torch.nn.functional.softmax(float_42, dim=-1)
        float_42 = None
        attn_weights_80 = softmax_40.type_as(scores_81)
        softmax_40 = scores_81 = None
        attn_weights_81 = torch.nn.functional.dropout(
            attn_weights_80, p=0.1, training=False
        )
        attn_weights_80 = None
        attn_output_160 = torch.matmul(attn_weights_81, value_states_81)
        attn_weights_81 = value_states_81 = None
        transpose_204 = attn_output_160.transpose(1, 2)
        attn_output_160 = None
        attn_output_161 = transpose_204.contiguous()
        transpose_204 = None
        attn_output_162 = attn_output_161.view(1, -1, 768)
        attn_output_161 = None
        attn_output_163 = torch._C._nn.linear(
            attn_output_162,
            l_self_modules_block_modules_40_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_162 = l_self_modules_block_modules_40_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_162 = torch.nn.functional.dropout(attn_output_163, 0.1, False, False)
        attn_output_163 = None
        hidden_states_322 = hidden_states_320 + dropout_162
        hidden_states_320 = dropout_162 = None
        to_85 = hidden_states_322.to(torch.float32)
        pow_82 = to_85.pow(2)
        to_85 = None
        variance_81 = pow_82.mean(-1, keepdim=True)
        pow_82 = None
        add_166 = variance_81 + 1e-06
        variance_81 = None
        rsqrt_81 = torch.rsqrt(add_166)
        add_166 = None
        hidden_states_323 = hidden_states_322 * rsqrt_81
        rsqrt_81 = None
        forwarded_states_40 = (
            l_self_modules_block_modules_40_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_323
        )
        l_self_modules_block_modules_40_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_323
        ) = None
        hidden_states_324 = torch._C._nn.linear(
            forwarded_states_40,
            l_self_modules_block_modules_40_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_40 = l_self_modules_block_modules_40_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_325 = torch.nn.functional.relu(hidden_states_324, inplace=False)
        hidden_states_324 = None
        hidden_states_326 = torch.nn.functional.dropout(
            hidden_states_325, 0.1, False, False
        )
        hidden_states_325 = None
        hidden_states_327 = torch._C._nn.linear(
            hidden_states_326,
            l_self_modules_block_modules_40_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_326 = l_self_modules_block_modules_40_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_164 = torch.nn.functional.dropout(hidden_states_327, 0.1, False, False)
        hidden_states_327 = None
        hidden_states_328 = hidden_states_322 + dropout_164
        hidden_states_322 = dropout_164 = None
        to_86 = hidden_states_328.to(torch.float32)
        pow_83 = to_86.pow(2)
        to_86 = None
        variance_82 = pow_83.mean(-1, keepdim=True)
        pow_83 = None
        add_168 = variance_82 + 1e-06
        variance_82 = None
        rsqrt_82 = torch.rsqrt(add_168)
        add_168 = None
        hidden_states_329 = hidden_states_328 * rsqrt_82
        rsqrt_82 = None
        normed_hidden_states_41 = (
            l_self_modules_block_modules_41_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_329
        )
        l_self_modules_block_modules_41_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_329
        ) = None
        query_states_82 = torch._C._nn.linear(
            normed_hidden_states_41,
            l_self_modules_block_modules_41_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_41_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_165 = query_states_82.view(1, -1, 12, 64)
        query_states_82 = None
        query_states_83 = view_165.transpose(1, 2)
        view_165 = None
        key_states_82 = torch._C._nn.linear(
            normed_hidden_states_41,
            l_self_modules_block_modules_41_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_41_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_82 = torch._C._nn.linear(
            normed_hidden_states_41,
            l_self_modules_block_modules_41_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_41 = l_self_modules_block_modules_41_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_166 = key_states_82.view(1, -1, 12, 64)
        key_states_82 = None
        key_states_83 = view_166.transpose(1, 2)
        view_166 = None
        view_167 = value_states_82.view(1, -1, 12, 64)
        value_states_82 = None
        value_states_83 = view_167.transpose(1, 2)
        view_167 = None
        transpose_208 = key_states_83.transpose(3, 2)
        key_states_83 = None
        scores_82 = torch.matmul(query_states_83, transpose_208)
        query_states_83 = transpose_208 = None
        scores_82 += position_bias_1
        scores_83 = scores_82
        scores_82 = None
        float_43 = scores_83.float()
        softmax_41 = torch.nn.functional.softmax(float_43, dim=-1)
        float_43 = None
        attn_weights_82 = softmax_41.type_as(scores_83)
        softmax_41 = scores_83 = None
        attn_weights_83 = torch.nn.functional.dropout(
            attn_weights_82, p=0.1, training=False
        )
        attn_weights_82 = None
        attn_output_164 = torch.matmul(attn_weights_83, value_states_83)
        attn_weights_83 = value_states_83 = None
        transpose_209 = attn_output_164.transpose(1, 2)
        attn_output_164 = None
        attn_output_165 = transpose_209.contiguous()
        transpose_209 = None
        attn_output_166 = attn_output_165.view(1, -1, 768)
        attn_output_165 = None
        attn_output_167 = torch._C._nn.linear(
            attn_output_166,
            l_self_modules_block_modules_41_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_166 = l_self_modules_block_modules_41_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_166 = torch.nn.functional.dropout(attn_output_167, 0.1, False, False)
        attn_output_167 = None
        hidden_states_330 = hidden_states_328 + dropout_166
        hidden_states_328 = dropout_166 = None
        to_87 = hidden_states_330.to(torch.float32)
        pow_84 = to_87.pow(2)
        to_87 = None
        variance_83 = pow_84.mean(-1, keepdim=True)
        pow_84 = None
        add_170 = variance_83 + 1e-06
        variance_83 = None
        rsqrt_83 = torch.rsqrt(add_170)
        add_170 = None
        hidden_states_331 = hidden_states_330 * rsqrt_83
        rsqrt_83 = None
        forwarded_states_41 = (
            l_self_modules_block_modules_41_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_331
        )
        l_self_modules_block_modules_41_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_331
        ) = None
        hidden_states_332 = torch._C._nn.linear(
            forwarded_states_41,
            l_self_modules_block_modules_41_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_41 = l_self_modules_block_modules_41_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_333 = torch.nn.functional.relu(hidden_states_332, inplace=False)
        hidden_states_332 = None
        hidden_states_334 = torch.nn.functional.dropout(
            hidden_states_333, 0.1, False, False
        )
        hidden_states_333 = None
        hidden_states_335 = torch._C._nn.linear(
            hidden_states_334,
            l_self_modules_block_modules_41_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_334 = l_self_modules_block_modules_41_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_168 = torch.nn.functional.dropout(hidden_states_335, 0.1, False, False)
        hidden_states_335 = None
        hidden_states_336 = hidden_states_330 + dropout_168
        hidden_states_330 = dropout_168 = None
        to_88 = hidden_states_336.to(torch.float32)
        pow_85 = to_88.pow(2)
        to_88 = None
        variance_84 = pow_85.mean(-1, keepdim=True)
        pow_85 = None
        add_172 = variance_84 + 1e-06
        variance_84 = None
        rsqrt_84 = torch.rsqrt(add_172)
        add_172 = None
        hidden_states_337 = hidden_states_336 * rsqrt_84
        rsqrt_84 = None
        normed_hidden_states_42 = (
            l_self_modules_block_modules_42_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_337
        )
        l_self_modules_block_modules_42_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_337
        ) = None
        query_states_84 = torch._C._nn.linear(
            normed_hidden_states_42,
            l_self_modules_block_modules_42_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_42_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_169 = query_states_84.view(1, -1, 12, 64)
        query_states_84 = None
        query_states_85 = view_169.transpose(1, 2)
        view_169 = None
        key_states_84 = torch._C._nn.linear(
            normed_hidden_states_42,
            l_self_modules_block_modules_42_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_42_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_84 = torch._C._nn.linear(
            normed_hidden_states_42,
            l_self_modules_block_modules_42_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_42 = l_self_modules_block_modules_42_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_170 = key_states_84.view(1, -1, 12, 64)
        key_states_84 = None
        key_states_85 = view_170.transpose(1, 2)
        view_170 = None
        view_171 = value_states_84.view(1, -1, 12, 64)
        value_states_84 = None
        value_states_85 = view_171.transpose(1, 2)
        view_171 = None
        transpose_213 = key_states_85.transpose(3, 2)
        key_states_85 = None
        scores_84 = torch.matmul(query_states_85, transpose_213)
        query_states_85 = transpose_213 = None
        scores_84 += position_bias_1
        scores_85 = scores_84
        scores_84 = None
        float_44 = scores_85.float()
        softmax_42 = torch.nn.functional.softmax(float_44, dim=-1)
        float_44 = None
        attn_weights_84 = softmax_42.type_as(scores_85)
        softmax_42 = scores_85 = None
        attn_weights_85 = torch.nn.functional.dropout(
            attn_weights_84, p=0.1, training=False
        )
        attn_weights_84 = None
        attn_output_168 = torch.matmul(attn_weights_85, value_states_85)
        attn_weights_85 = value_states_85 = None
        transpose_214 = attn_output_168.transpose(1, 2)
        attn_output_168 = None
        attn_output_169 = transpose_214.contiguous()
        transpose_214 = None
        attn_output_170 = attn_output_169.view(1, -1, 768)
        attn_output_169 = None
        attn_output_171 = torch._C._nn.linear(
            attn_output_170,
            l_self_modules_block_modules_42_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_170 = l_self_modules_block_modules_42_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_170 = torch.nn.functional.dropout(attn_output_171, 0.1, False, False)
        attn_output_171 = None
        hidden_states_338 = hidden_states_336 + dropout_170
        hidden_states_336 = dropout_170 = None
        to_89 = hidden_states_338.to(torch.float32)
        pow_86 = to_89.pow(2)
        to_89 = None
        variance_85 = pow_86.mean(-1, keepdim=True)
        pow_86 = None
        add_174 = variance_85 + 1e-06
        variance_85 = None
        rsqrt_85 = torch.rsqrt(add_174)
        add_174 = None
        hidden_states_339 = hidden_states_338 * rsqrt_85
        rsqrt_85 = None
        forwarded_states_42 = (
            l_self_modules_block_modules_42_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_339
        )
        l_self_modules_block_modules_42_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_339
        ) = None
        hidden_states_340 = torch._C._nn.linear(
            forwarded_states_42,
            l_self_modules_block_modules_42_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_42 = l_self_modules_block_modules_42_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_341 = torch.nn.functional.relu(hidden_states_340, inplace=False)
        hidden_states_340 = None
        hidden_states_342 = torch.nn.functional.dropout(
            hidden_states_341, 0.1, False, False
        )
        hidden_states_341 = None
        hidden_states_343 = torch._C._nn.linear(
            hidden_states_342,
            l_self_modules_block_modules_42_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_342 = l_self_modules_block_modules_42_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_172 = torch.nn.functional.dropout(hidden_states_343, 0.1, False, False)
        hidden_states_343 = None
        hidden_states_344 = hidden_states_338 + dropout_172
        hidden_states_338 = dropout_172 = None
        to_90 = hidden_states_344.to(torch.float32)
        pow_87 = to_90.pow(2)
        to_90 = None
        variance_86 = pow_87.mean(-1, keepdim=True)
        pow_87 = None
        add_176 = variance_86 + 1e-06
        variance_86 = None
        rsqrt_86 = torch.rsqrt(add_176)
        add_176 = None
        hidden_states_345 = hidden_states_344 * rsqrt_86
        rsqrt_86 = None
        normed_hidden_states_43 = (
            l_self_modules_block_modules_43_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_345
        )
        l_self_modules_block_modules_43_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_345
        ) = None
        query_states_86 = torch._C._nn.linear(
            normed_hidden_states_43,
            l_self_modules_block_modules_43_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_43_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_173 = query_states_86.view(1, -1, 12, 64)
        query_states_86 = None
        query_states_87 = view_173.transpose(1, 2)
        view_173 = None
        key_states_86 = torch._C._nn.linear(
            normed_hidden_states_43,
            l_self_modules_block_modules_43_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_43_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_86 = torch._C._nn.linear(
            normed_hidden_states_43,
            l_self_modules_block_modules_43_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_43 = l_self_modules_block_modules_43_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_174 = key_states_86.view(1, -1, 12, 64)
        key_states_86 = None
        key_states_87 = view_174.transpose(1, 2)
        view_174 = None
        view_175 = value_states_86.view(1, -1, 12, 64)
        value_states_86 = None
        value_states_87 = view_175.transpose(1, 2)
        view_175 = None
        transpose_218 = key_states_87.transpose(3, 2)
        key_states_87 = None
        scores_86 = torch.matmul(query_states_87, transpose_218)
        query_states_87 = transpose_218 = None
        scores_86 += position_bias_1
        scores_87 = scores_86
        scores_86 = None
        float_45 = scores_87.float()
        softmax_43 = torch.nn.functional.softmax(float_45, dim=-1)
        float_45 = None
        attn_weights_86 = softmax_43.type_as(scores_87)
        softmax_43 = scores_87 = None
        attn_weights_87 = torch.nn.functional.dropout(
            attn_weights_86, p=0.1, training=False
        )
        attn_weights_86 = None
        attn_output_172 = torch.matmul(attn_weights_87, value_states_87)
        attn_weights_87 = value_states_87 = None
        transpose_219 = attn_output_172.transpose(1, 2)
        attn_output_172 = None
        attn_output_173 = transpose_219.contiguous()
        transpose_219 = None
        attn_output_174 = attn_output_173.view(1, -1, 768)
        attn_output_173 = None
        attn_output_175 = torch._C._nn.linear(
            attn_output_174,
            l_self_modules_block_modules_43_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_174 = l_self_modules_block_modules_43_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_174 = torch.nn.functional.dropout(attn_output_175, 0.1, False, False)
        attn_output_175 = None
        hidden_states_346 = hidden_states_344 + dropout_174
        hidden_states_344 = dropout_174 = None
        to_91 = hidden_states_346.to(torch.float32)
        pow_88 = to_91.pow(2)
        to_91 = None
        variance_87 = pow_88.mean(-1, keepdim=True)
        pow_88 = None
        add_178 = variance_87 + 1e-06
        variance_87 = None
        rsqrt_87 = torch.rsqrt(add_178)
        add_178 = None
        hidden_states_347 = hidden_states_346 * rsqrt_87
        rsqrt_87 = None
        forwarded_states_43 = (
            l_self_modules_block_modules_43_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_347
        )
        l_self_modules_block_modules_43_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_347
        ) = None
        hidden_states_348 = torch._C._nn.linear(
            forwarded_states_43,
            l_self_modules_block_modules_43_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_43 = l_self_modules_block_modules_43_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_349 = torch.nn.functional.relu(hidden_states_348, inplace=False)
        hidden_states_348 = None
        hidden_states_350 = torch.nn.functional.dropout(
            hidden_states_349, 0.1, False, False
        )
        hidden_states_349 = None
        hidden_states_351 = torch._C._nn.linear(
            hidden_states_350,
            l_self_modules_block_modules_43_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_350 = l_self_modules_block_modules_43_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_176 = torch.nn.functional.dropout(hidden_states_351, 0.1, False, False)
        hidden_states_351 = None
        hidden_states_352 = hidden_states_346 + dropout_176
        hidden_states_346 = dropout_176 = None
        to_92 = hidden_states_352.to(torch.float32)
        pow_89 = to_92.pow(2)
        to_92 = None
        variance_88 = pow_89.mean(-1, keepdim=True)
        pow_89 = None
        add_180 = variance_88 + 1e-06
        variance_88 = None
        rsqrt_88 = torch.rsqrt(add_180)
        add_180 = None
        hidden_states_353 = hidden_states_352 * rsqrt_88
        rsqrt_88 = None
        normed_hidden_states_44 = (
            l_self_modules_block_modules_44_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_353
        )
        l_self_modules_block_modules_44_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_353
        ) = None
        query_states_88 = torch._C._nn.linear(
            normed_hidden_states_44,
            l_self_modules_block_modules_44_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_44_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_177 = query_states_88.view(1, -1, 12, 64)
        query_states_88 = None
        query_states_89 = view_177.transpose(1, 2)
        view_177 = None
        key_states_88 = torch._C._nn.linear(
            normed_hidden_states_44,
            l_self_modules_block_modules_44_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_44_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_88 = torch._C._nn.linear(
            normed_hidden_states_44,
            l_self_modules_block_modules_44_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_44 = l_self_modules_block_modules_44_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_178 = key_states_88.view(1, -1, 12, 64)
        key_states_88 = None
        key_states_89 = view_178.transpose(1, 2)
        view_178 = None
        view_179 = value_states_88.view(1, -1, 12, 64)
        value_states_88 = None
        value_states_89 = view_179.transpose(1, 2)
        view_179 = None
        transpose_223 = key_states_89.transpose(3, 2)
        key_states_89 = None
        scores_88 = torch.matmul(query_states_89, transpose_223)
        query_states_89 = transpose_223 = None
        scores_88 += position_bias_1
        scores_89 = scores_88
        scores_88 = None
        float_46 = scores_89.float()
        softmax_44 = torch.nn.functional.softmax(float_46, dim=-1)
        float_46 = None
        attn_weights_88 = softmax_44.type_as(scores_89)
        softmax_44 = scores_89 = None
        attn_weights_89 = torch.nn.functional.dropout(
            attn_weights_88, p=0.1, training=False
        )
        attn_weights_88 = None
        attn_output_176 = torch.matmul(attn_weights_89, value_states_89)
        attn_weights_89 = value_states_89 = None
        transpose_224 = attn_output_176.transpose(1, 2)
        attn_output_176 = None
        attn_output_177 = transpose_224.contiguous()
        transpose_224 = None
        attn_output_178 = attn_output_177.view(1, -1, 768)
        attn_output_177 = None
        attn_output_179 = torch._C._nn.linear(
            attn_output_178,
            l_self_modules_block_modules_44_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_178 = l_self_modules_block_modules_44_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_178 = torch.nn.functional.dropout(attn_output_179, 0.1, False, False)
        attn_output_179 = None
        hidden_states_354 = hidden_states_352 + dropout_178
        hidden_states_352 = dropout_178 = None
        to_93 = hidden_states_354.to(torch.float32)
        pow_90 = to_93.pow(2)
        to_93 = None
        variance_89 = pow_90.mean(-1, keepdim=True)
        pow_90 = None
        add_182 = variance_89 + 1e-06
        variance_89 = None
        rsqrt_89 = torch.rsqrt(add_182)
        add_182 = None
        hidden_states_355 = hidden_states_354 * rsqrt_89
        rsqrt_89 = None
        forwarded_states_44 = (
            l_self_modules_block_modules_44_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_355
        )
        l_self_modules_block_modules_44_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_355
        ) = None
        hidden_states_356 = torch._C._nn.linear(
            forwarded_states_44,
            l_self_modules_block_modules_44_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_44 = l_self_modules_block_modules_44_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_357 = torch.nn.functional.relu(hidden_states_356, inplace=False)
        hidden_states_356 = None
        hidden_states_358 = torch.nn.functional.dropout(
            hidden_states_357, 0.1, False, False
        )
        hidden_states_357 = None
        hidden_states_359 = torch._C._nn.linear(
            hidden_states_358,
            l_self_modules_block_modules_44_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_358 = l_self_modules_block_modules_44_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_180 = torch.nn.functional.dropout(hidden_states_359, 0.1, False, False)
        hidden_states_359 = None
        hidden_states_360 = hidden_states_354 + dropout_180
        hidden_states_354 = dropout_180 = None
        to_94 = hidden_states_360.to(torch.float32)
        pow_91 = to_94.pow(2)
        to_94 = None
        variance_90 = pow_91.mean(-1, keepdim=True)
        pow_91 = None
        add_184 = variance_90 + 1e-06
        variance_90 = None
        rsqrt_90 = torch.rsqrt(add_184)
        add_184 = None
        hidden_states_361 = hidden_states_360 * rsqrt_90
        rsqrt_90 = None
        normed_hidden_states_45 = (
            l_self_modules_block_modules_45_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_361
        )
        l_self_modules_block_modules_45_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_361
        ) = None
        query_states_90 = torch._C._nn.linear(
            normed_hidden_states_45,
            l_self_modules_block_modules_45_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_45_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_181 = query_states_90.view(1, -1, 12, 64)
        query_states_90 = None
        query_states_91 = view_181.transpose(1, 2)
        view_181 = None
        key_states_90 = torch._C._nn.linear(
            normed_hidden_states_45,
            l_self_modules_block_modules_45_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_45_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_90 = torch._C._nn.linear(
            normed_hidden_states_45,
            l_self_modules_block_modules_45_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_45 = l_self_modules_block_modules_45_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_182 = key_states_90.view(1, -1, 12, 64)
        key_states_90 = None
        key_states_91 = view_182.transpose(1, 2)
        view_182 = None
        view_183 = value_states_90.view(1, -1, 12, 64)
        value_states_90 = None
        value_states_91 = view_183.transpose(1, 2)
        view_183 = None
        transpose_228 = key_states_91.transpose(3, 2)
        key_states_91 = None
        scores_90 = torch.matmul(query_states_91, transpose_228)
        query_states_91 = transpose_228 = None
        scores_90 += position_bias_1
        scores_91 = scores_90
        scores_90 = None
        float_47 = scores_91.float()
        softmax_45 = torch.nn.functional.softmax(float_47, dim=-1)
        float_47 = None
        attn_weights_90 = softmax_45.type_as(scores_91)
        softmax_45 = scores_91 = None
        attn_weights_91 = torch.nn.functional.dropout(
            attn_weights_90, p=0.1, training=False
        )
        attn_weights_90 = None
        attn_output_180 = torch.matmul(attn_weights_91, value_states_91)
        attn_weights_91 = value_states_91 = None
        transpose_229 = attn_output_180.transpose(1, 2)
        attn_output_180 = None
        attn_output_181 = transpose_229.contiguous()
        transpose_229 = None
        attn_output_182 = attn_output_181.view(1, -1, 768)
        attn_output_181 = None
        attn_output_183 = torch._C._nn.linear(
            attn_output_182,
            l_self_modules_block_modules_45_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_182 = l_self_modules_block_modules_45_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_182 = torch.nn.functional.dropout(attn_output_183, 0.1, False, False)
        attn_output_183 = None
        hidden_states_362 = hidden_states_360 + dropout_182
        hidden_states_360 = dropout_182 = None
        to_95 = hidden_states_362.to(torch.float32)
        pow_92 = to_95.pow(2)
        to_95 = None
        variance_91 = pow_92.mean(-1, keepdim=True)
        pow_92 = None
        add_186 = variance_91 + 1e-06
        variance_91 = None
        rsqrt_91 = torch.rsqrt(add_186)
        add_186 = None
        hidden_states_363 = hidden_states_362 * rsqrt_91
        rsqrt_91 = None
        forwarded_states_45 = (
            l_self_modules_block_modules_45_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_363
        )
        l_self_modules_block_modules_45_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_363
        ) = None
        hidden_states_364 = torch._C._nn.linear(
            forwarded_states_45,
            l_self_modules_block_modules_45_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_45 = l_self_modules_block_modules_45_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_365 = torch.nn.functional.relu(hidden_states_364, inplace=False)
        hidden_states_364 = None
        hidden_states_366 = torch.nn.functional.dropout(
            hidden_states_365, 0.1, False, False
        )
        hidden_states_365 = None
        hidden_states_367 = torch._C._nn.linear(
            hidden_states_366,
            l_self_modules_block_modules_45_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_366 = l_self_modules_block_modules_45_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_184 = torch.nn.functional.dropout(hidden_states_367, 0.1, False, False)
        hidden_states_367 = None
        hidden_states_368 = hidden_states_362 + dropout_184
        hidden_states_362 = dropout_184 = None
        to_96 = hidden_states_368.to(torch.float32)
        pow_93 = to_96.pow(2)
        to_96 = None
        variance_92 = pow_93.mean(-1, keepdim=True)
        pow_93 = None
        add_188 = variance_92 + 1e-06
        variance_92 = None
        rsqrt_92 = torch.rsqrt(add_188)
        add_188 = None
        hidden_states_369 = hidden_states_368 * rsqrt_92
        rsqrt_92 = None
        normed_hidden_states_46 = (
            l_self_modules_block_modules_46_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_369
        )
        l_self_modules_block_modules_46_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_369
        ) = None
        query_states_92 = torch._C._nn.linear(
            normed_hidden_states_46,
            l_self_modules_block_modules_46_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_46_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_185 = query_states_92.view(1, -1, 12, 64)
        query_states_92 = None
        query_states_93 = view_185.transpose(1, 2)
        view_185 = None
        key_states_92 = torch._C._nn.linear(
            normed_hidden_states_46,
            l_self_modules_block_modules_46_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_46_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_92 = torch._C._nn.linear(
            normed_hidden_states_46,
            l_self_modules_block_modules_46_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_46 = l_self_modules_block_modules_46_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_186 = key_states_92.view(1, -1, 12, 64)
        key_states_92 = None
        key_states_93 = view_186.transpose(1, 2)
        view_186 = None
        view_187 = value_states_92.view(1, -1, 12, 64)
        value_states_92 = None
        value_states_93 = view_187.transpose(1, 2)
        view_187 = None
        transpose_233 = key_states_93.transpose(3, 2)
        key_states_93 = None
        scores_92 = torch.matmul(query_states_93, transpose_233)
        query_states_93 = transpose_233 = None
        scores_92 += position_bias_1
        scores_93 = scores_92
        scores_92 = None
        float_48 = scores_93.float()
        softmax_46 = torch.nn.functional.softmax(float_48, dim=-1)
        float_48 = None
        attn_weights_92 = softmax_46.type_as(scores_93)
        softmax_46 = scores_93 = None
        attn_weights_93 = torch.nn.functional.dropout(
            attn_weights_92, p=0.1, training=False
        )
        attn_weights_92 = None
        attn_output_184 = torch.matmul(attn_weights_93, value_states_93)
        attn_weights_93 = value_states_93 = None
        transpose_234 = attn_output_184.transpose(1, 2)
        attn_output_184 = None
        attn_output_185 = transpose_234.contiguous()
        transpose_234 = None
        attn_output_186 = attn_output_185.view(1, -1, 768)
        attn_output_185 = None
        attn_output_187 = torch._C._nn.linear(
            attn_output_186,
            l_self_modules_block_modules_46_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_186 = l_self_modules_block_modules_46_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_186 = torch.nn.functional.dropout(attn_output_187, 0.1, False, False)
        attn_output_187 = None
        hidden_states_370 = hidden_states_368 + dropout_186
        hidden_states_368 = dropout_186 = None
        to_97 = hidden_states_370.to(torch.float32)
        pow_94 = to_97.pow(2)
        to_97 = None
        variance_93 = pow_94.mean(-1, keepdim=True)
        pow_94 = None
        add_190 = variance_93 + 1e-06
        variance_93 = None
        rsqrt_93 = torch.rsqrt(add_190)
        add_190 = None
        hidden_states_371 = hidden_states_370 * rsqrt_93
        rsqrt_93 = None
        forwarded_states_46 = (
            l_self_modules_block_modules_46_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_371
        )
        l_self_modules_block_modules_46_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_371
        ) = None
        hidden_states_372 = torch._C._nn.linear(
            forwarded_states_46,
            l_self_modules_block_modules_46_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_46 = l_self_modules_block_modules_46_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_373 = torch.nn.functional.relu(hidden_states_372, inplace=False)
        hidden_states_372 = None
        hidden_states_374 = torch.nn.functional.dropout(
            hidden_states_373, 0.1, False, False
        )
        hidden_states_373 = None
        hidden_states_375 = torch._C._nn.linear(
            hidden_states_374,
            l_self_modules_block_modules_46_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_374 = l_self_modules_block_modules_46_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_188 = torch.nn.functional.dropout(hidden_states_375, 0.1, False, False)
        hidden_states_375 = None
        hidden_states_376 = hidden_states_370 + dropout_188
        hidden_states_370 = dropout_188 = None
        to_98 = hidden_states_376.to(torch.float32)
        pow_95 = to_98.pow(2)
        to_98 = None
        variance_94 = pow_95.mean(-1, keepdim=True)
        pow_95 = None
        add_192 = variance_94 + 1e-06
        variance_94 = None
        rsqrt_94 = torch.rsqrt(add_192)
        add_192 = None
        hidden_states_377 = hidden_states_376 * rsqrt_94
        rsqrt_94 = None
        normed_hidden_states_47 = (
            l_self_modules_block_modules_47_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_377
        )
        l_self_modules_block_modules_47_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_377
        ) = None
        query_states_94 = torch._C._nn.linear(
            normed_hidden_states_47,
            l_self_modules_block_modules_47_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_47_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_189 = query_states_94.view(1, -1, 12, 64)
        query_states_94 = None
        query_states_95 = view_189.transpose(1, 2)
        view_189 = None
        key_states_94 = torch._C._nn.linear(
            normed_hidden_states_47,
            l_self_modules_block_modules_47_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_47_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_94 = torch._C._nn.linear(
            normed_hidden_states_47,
            l_self_modules_block_modules_47_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_47 = l_self_modules_block_modules_47_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_190 = key_states_94.view(1, -1, 12, 64)
        key_states_94 = None
        key_states_95 = view_190.transpose(1, 2)
        view_190 = None
        view_191 = value_states_94.view(1, -1, 12, 64)
        value_states_94 = None
        value_states_95 = view_191.transpose(1, 2)
        view_191 = None
        transpose_238 = key_states_95.transpose(3, 2)
        key_states_95 = None
        scores_94 = torch.matmul(query_states_95, transpose_238)
        query_states_95 = transpose_238 = None
        scores_94 += position_bias_1
        scores_95 = scores_94
        scores_94 = position_bias_1 = None
        float_49 = scores_95.float()
        softmax_47 = torch.nn.functional.softmax(float_49, dim=-1)
        float_49 = None
        attn_weights_94 = softmax_47.type_as(scores_95)
        softmax_47 = scores_95 = None
        attn_weights_95 = torch.nn.functional.dropout(
            attn_weights_94, p=0.1, training=False
        )
        attn_weights_94 = None
        attn_output_188 = torch.matmul(attn_weights_95, value_states_95)
        attn_weights_95 = value_states_95 = None
        transpose_239 = attn_output_188.transpose(1, 2)
        attn_output_188 = None
        attn_output_189 = transpose_239.contiguous()
        transpose_239 = None
        attn_output_190 = attn_output_189.view(1, -1, 768)
        attn_output_189 = None
        attn_output_191 = torch._C._nn.linear(
            attn_output_190,
            l_self_modules_block_modules_47_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_190 = l_self_modules_block_modules_47_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_190 = torch.nn.functional.dropout(attn_output_191, 0.1, False, False)
        attn_output_191 = None
        hidden_states_378 = hidden_states_376 + dropout_190
        hidden_states_376 = dropout_190 = None
        to_99 = hidden_states_378.to(torch.float32)
        pow_96 = to_99.pow(2)
        to_99 = None
        variance_95 = pow_96.mean(-1, keepdim=True)
        pow_96 = None
        add_194 = variance_95 + 1e-06
        variance_95 = None
        rsqrt_95 = torch.rsqrt(add_194)
        add_194 = None
        hidden_states_379 = hidden_states_378 * rsqrt_95
        rsqrt_95 = None
        forwarded_states_47 = (
            l_self_modules_block_modules_47_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_379
        )
        l_self_modules_block_modules_47_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_379
        ) = None
        hidden_states_380 = torch._C._nn.linear(
            forwarded_states_47,
            l_self_modules_block_modules_47_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_47 = l_self_modules_block_modules_47_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_381 = torch.nn.functional.relu(hidden_states_380, inplace=False)
        hidden_states_380 = None
        hidden_states_382 = torch.nn.functional.dropout(
            hidden_states_381, 0.1, False, False
        )
        hidden_states_381 = None
        hidden_states_383 = torch._C._nn.linear(
            hidden_states_382,
            l_self_modules_block_modules_47_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_382 = l_self_modules_block_modules_47_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_192 = torch.nn.functional.dropout(hidden_states_383, 0.1, False, False)
        hidden_states_383 = None
        hidden_states_384 = hidden_states_378 + dropout_192
        hidden_states_378 = dropout_192 = None
        to_100 = hidden_states_384.to(torch.float32)
        pow_97 = to_100.pow(2)
        to_100 = None
        variance_96 = pow_97.mean(-1, keepdim=True)
        pow_97 = None
        add_196 = variance_96 + 1e-06
        variance_96 = None
        rsqrt_96 = torch.rsqrt(add_196)
        add_196 = None
        hidden_states_385 = hidden_states_384 * rsqrt_96
        hidden_states_384 = rsqrt_96 = None
        hidden_states_386 = (
            l_self_modules_final_layer_norm_parameters_weight_ * hidden_states_385
        )
        l_self_modules_final_layer_norm_parameters_weight_ = hidden_states_385 = None
        hidden_states_387 = torch.nn.functional.dropout(
            hidden_states_386, 0.1, False, False
        )
        hidden_states_386 = None
        return (hidden_states_387,)
