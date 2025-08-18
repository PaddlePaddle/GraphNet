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
        L_self_modules_block_modules_0_modules_layer_modules_1_modules_DenseReluDense_modules_wi_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_0_modules_layer_modules_1_modules_DenseReluDense_modules_wi_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_0_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_1_modules_DenseReluDense_modules_wi_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_1_modules_DenseReluDense_modules_wi_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_1_modules_DenseReluDense_modules_wi_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_1_modules_DenseReluDense_modules_wi_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_1_modules_DenseReluDense_modules_wi_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_1_modules_DenseReluDense_modules_wi_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_4_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_4_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_4_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_4_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_4_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_4_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_4_modules_layer_modules_1_modules_DenseReluDense_modules_wi_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_4_modules_layer_modules_1_modules_DenseReluDense_modules_wi_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_4_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_5_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_5_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_5_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_5_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_5_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_5_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_5_modules_layer_modules_1_modules_DenseReluDense_modules_wi_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_5_modules_layer_modules_1_modules_DenseReluDense_modules_wi_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_5_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_6_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_6_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_6_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_6_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_6_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_6_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_6_modules_layer_modules_1_modules_DenseReluDense_modules_wi_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_6_modules_layer_modules_1_modules_DenseReluDense_modules_wi_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_6_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_7_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_7_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_7_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_7_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_7_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_7_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_7_modules_layer_modules_1_modules_DenseReluDense_modules_wi_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_7_modules_layer_modules_1_modules_DenseReluDense_modules_wi_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_7_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
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
        l_self_modules_block_modules_0_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_0_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_1_modules_DenseReluDense_modules_wi_0_parameters_weight_
        l_self_modules_block_modules_0_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_1_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_1_modules_DenseReluDense_modules_wi_1_parameters_weight_
        l_self_modules_block_modules_0_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_0_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_1_modules_DenseReluDense_modules_wi_0_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_1_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_1_modules_DenseReluDense_modules_wi_1_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_0_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_1_modules_DenseReluDense_modules_wi_0_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_1_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_1_modules_DenseReluDense_modules_wi_1_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_0_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_1_modules_DenseReluDense_modules_wi_0_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_1_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_1_modules_DenseReluDense_modules_wi_1_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_4_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_4_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_4_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_4_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_4_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_4_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_4_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_4_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_4_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_4_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_4_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_4_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_4_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_0_parameters_weight_ = L_self_modules_block_modules_4_modules_layer_modules_1_modules_DenseReluDense_modules_wi_0_parameters_weight_
        l_self_modules_block_modules_4_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_1_parameters_weight_ = L_self_modules_block_modules_4_modules_layer_modules_1_modules_DenseReluDense_modules_wi_1_parameters_weight_
        l_self_modules_block_modules_4_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_4_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_5_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_5_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_5_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_5_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_5_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_5_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_5_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_5_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_5_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_5_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_5_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_5_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_5_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_0_parameters_weight_ = L_self_modules_block_modules_5_modules_layer_modules_1_modules_DenseReluDense_modules_wi_0_parameters_weight_
        l_self_modules_block_modules_5_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_1_parameters_weight_ = L_self_modules_block_modules_5_modules_layer_modules_1_modules_DenseReluDense_modules_wi_1_parameters_weight_
        l_self_modules_block_modules_5_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_5_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_6_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_6_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_6_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_6_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_6_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_6_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_6_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_6_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_6_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_6_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_6_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_6_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_6_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_0_parameters_weight_ = L_self_modules_block_modules_6_modules_layer_modules_1_modules_DenseReluDense_modules_wi_0_parameters_weight_
        l_self_modules_block_modules_6_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_1_parameters_weight_ = L_self_modules_block_modules_6_modules_layer_modules_1_modules_DenseReluDense_modules_wi_1_parameters_weight_
        l_self_modules_block_modules_6_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_6_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_7_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_7_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_7_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_7_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_7_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_7_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_7_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_7_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_7_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_7_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_7_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_7_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_7_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_0_parameters_weight_ = L_self_modules_block_modules_7_modules_layer_modules_1_modules_DenseReluDense_modules_wi_0_parameters_weight_
        l_self_modules_block_modules_7_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_1_parameters_weight_ = L_self_modules_block_modules_7_modules_layer_modules_1_modules_DenseReluDense_modules_wi_1_parameters_weight_
        l_self_modules_block_modules_7_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_7_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
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
        view_1 = query_states.view(1, -1, 6, 64)
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
        view_2 = key_states.view(1, -1, 6, 64)
        key_states = None
        key_states_1 = view_2.transpose(1, 2)
        view_2 = None
        view_3 = value_states.view(1, -1, 6, 64)
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
        attn_output_2 = attn_output_1.view(1, -1, 384)
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
        linear_4 = torch._C._nn.linear(
            forwarded_states,
            l_self_modules_block_modules_0_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_0_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_0_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_0_parameters_weight_ = (
            None
        )
        mul_7 = 0.5 * linear_4
        pow_3 = torch.pow(linear_4, 3.0)
        mul_8 = 0.044715 * pow_3
        pow_3 = None
        add_7 = linear_4 + mul_8
        linear_4 = mul_8 = None
        mul_9 = 0.7978845608028654 * add_7
        add_7 = None
        tanh = torch.tanh(mul_9)
        mul_9 = None
        add_8 = 1.0 + tanh
        tanh = None
        hidden_gelu = mul_7 * add_8
        mul_7 = add_8 = None
        hidden_linear = torch._C._nn.linear(
            forwarded_states,
            l_self_modules_block_modules_0_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_1_parameters_weight_,
            None,
        )
        forwarded_states = l_self_modules_block_modules_0_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_1_parameters_weight_ = (None)
        hidden_states_4 = hidden_gelu * hidden_linear
        hidden_gelu = hidden_linear = None
        hidden_states_5 = torch.nn.functional.dropout(
            hidden_states_4, 0.1, False, False
        )
        hidden_states_4 = None
        hidden_states_6 = torch._C._nn.linear(
            hidden_states_5,
            l_self_modules_block_modules_0_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_5 = l_self_modules_block_modules_0_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_4 = torch.nn.functional.dropout(hidden_states_6, 0.1, False, False)
        hidden_states_6 = None
        hidden_states_7 = hidden_states_2 + dropout_4
        hidden_states_2 = dropout_4 = None
        to_6 = hidden_states_7.to(torch.float32)
        pow_4 = to_6.pow(2)
        to_6 = None
        variance_2 = pow_4.mean(-1, keepdim=True)
        pow_4 = None
        add_10 = variance_2 + 1e-06
        variance_2 = None
        rsqrt_2 = torch.rsqrt(add_10)
        add_10 = None
        hidden_states_8 = hidden_states_7 * rsqrt_2
        rsqrt_2 = None
        normed_hidden_states_1 = (
            l_self_modules_block_modules_1_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_8
        )
        l_self_modules_block_modules_1_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_8
        ) = None
        query_states_2 = torch._C._nn.linear(
            normed_hidden_states_1,
            l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_5 = query_states_2.view(1, -1, 6, 64)
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
        view_6 = key_states_2.view(1, -1, 6, 64)
        key_states_2 = None
        key_states_3 = view_6.transpose(1, 2)
        view_6 = None
        view_7 = value_states_2.view(1, -1, 6, 64)
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
        attn_output_6 = attn_output_5.view(1, -1, 384)
        attn_output_5 = None
        attn_output_7 = torch._C._nn.linear(
            attn_output_6,
            l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_6 = l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_6 = torch.nn.functional.dropout(attn_output_7, 0.1, False, False)
        attn_output_7 = None
        hidden_states_9 = hidden_states_7 + dropout_6
        hidden_states_7 = dropout_6 = None
        to_7 = hidden_states_9.to(torch.float32)
        pow_5 = to_7.pow(2)
        to_7 = None
        variance_3 = pow_5.mean(-1, keepdim=True)
        pow_5 = None
        add_12 = variance_3 + 1e-06
        variance_3 = None
        rsqrt_3 = torch.rsqrt(add_12)
        add_12 = None
        hidden_states_10 = hidden_states_9 * rsqrt_3
        rsqrt_3 = None
        forwarded_states_1 = (
            l_self_modules_block_modules_1_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_10
        )
        l_self_modules_block_modules_1_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_10
        ) = None
        linear_11 = torch._C._nn.linear(
            forwarded_states_1,
            l_self_modules_block_modules_1_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_0_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_1_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_0_parameters_weight_ = (
            None
        )
        mul_16 = 0.5 * linear_11
        pow_6 = torch.pow(linear_11, 3.0)
        mul_17 = 0.044715 * pow_6
        pow_6 = None
        add_13 = linear_11 + mul_17
        linear_11 = mul_17 = None
        mul_18 = 0.7978845608028654 * add_13
        add_13 = None
        tanh_1 = torch.tanh(mul_18)
        mul_18 = None
        add_14 = 1.0 + tanh_1
        tanh_1 = None
        hidden_gelu_1 = mul_16 * add_14
        mul_16 = add_14 = None
        hidden_linear_1 = torch._C._nn.linear(
            forwarded_states_1,
            l_self_modules_block_modules_1_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_1_parameters_weight_,
            None,
        )
        forwarded_states_1 = l_self_modules_block_modules_1_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_1_parameters_weight_ = (None)
        hidden_states_11 = hidden_gelu_1 * hidden_linear_1
        hidden_gelu_1 = hidden_linear_1 = None
        hidden_states_12 = torch.nn.functional.dropout(
            hidden_states_11, 0.1, False, False
        )
        hidden_states_11 = None
        hidden_states_13 = torch._C._nn.linear(
            hidden_states_12,
            l_self_modules_block_modules_1_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_12 = l_self_modules_block_modules_1_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_8 = torch.nn.functional.dropout(hidden_states_13, 0.1, False, False)
        hidden_states_13 = None
        hidden_states_14 = hidden_states_9 + dropout_8
        hidden_states_9 = dropout_8 = None
        to_8 = hidden_states_14.to(torch.float32)
        pow_7 = to_8.pow(2)
        to_8 = None
        variance_4 = pow_7.mean(-1, keepdim=True)
        pow_7 = None
        add_16 = variance_4 + 1e-06
        variance_4 = None
        rsqrt_4 = torch.rsqrt(add_16)
        add_16 = None
        hidden_states_15 = hidden_states_14 * rsqrt_4
        rsqrt_4 = None
        normed_hidden_states_2 = (
            l_self_modules_block_modules_2_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_15
        )
        l_self_modules_block_modules_2_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_15
        ) = None
        query_states_4 = torch._C._nn.linear(
            normed_hidden_states_2,
            l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_9 = query_states_4.view(1, -1, 6, 64)
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
        view_10 = key_states_4.view(1, -1, 6, 64)
        key_states_4 = None
        key_states_5 = view_10.transpose(1, 2)
        view_10 = None
        view_11 = value_states_4.view(1, -1, 6, 64)
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
        attn_output_10 = attn_output_9.view(1, -1, 384)
        attn_output_9 = None
        attn_output_11 = torch._C._nn.linear(
            attn_output_10,
            l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_10 = l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_10 = torch.nn.functional.dropout(attn_output_11, 0.1, False, False)
        attn_output_11 = None
        hidden_states_16 = hidden_states_14 + dropout_10
        hidden_states_14 = dropout_10 = None
        to_9 = hidden_states_16.to(torch.float32)
        pow_8 = to_9.pow(2)
        to_9 = None
        variance_5 = pow_8.mean(-1, keepdim=True)
        pow_8 = None
        add_18 = variance_5 + 1e-06
        variance_5 = None
        rsqrt_5 = torch.rsqrt(add_18)
        add_18 = None
        hidden_states_17 = hidden_states_16 * rsqrt_5
        rsqrt_5 = None
        forwarded_states_2 = (
            l_self_modules_block_modules_2_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_17
        )
        l_self_modules_block_modules_2_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_17
        ) = None
        linear_18 = torch._C._nn.linear(
            forwarded_states_2,
            l_self_modules_block_modules_2_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_0_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_2_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_0_parameters_weight_ = (
            None
        )
        mul_25 = 0.5 * linear_18
        pow_9 = torch.pow(linear_18, 3.0)
        mul_26 = 0.044715 * pow_9
        pow_9 = None
        add_19 = linear_18 + mul_26
        linear_18 = mul_26 = None
        mul_27 = 0.7978845608028654 * add_19
        add_19 = None
        tanh_2 = torch.tanh(mul_27)
        mul_27 = None
        add_20 = 1.0 + tanh_2
        tanh_2 = None
        hidden_gelu_2 = mul_25 * add_20
        mul_25 = add_20 = None
        hidden_linear_2 = torch._C._nn.linear(
            forwarded_states_2,
            l_self_modules_block_modules_2_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_1_parameters_weight_,
            None,
        )
        forwarded_states_2 = l_self_modules_block_modules_2_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_1_parameters_weight_ = (None)
        hidden_states_18 = hidden_gelu_2 * hidden_linear_2
        hidden_gelu_2 = hidden_linear_2 = None
        hidden_states_19 = torch.nn.functional.dropout(
            hidden_states_18, 0.1, False, False
        )
        hidden_states_18 = None
        hidden_states_20 = torch._C._nn.linear(
            hidden_states_19,
            l_self_modules_block_modules_2_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_19 = l_self_modules_block_modules_2_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_12 = torch.nn.functional.dropout(hidden_states_20, 0.1, False, False)
        hidden_states_20 = None
        hidden_states_21 = hidden_states_16 + dropout_12
        hidden_states_16 = dropout_12 = None
        to_10 = hidden_states_21.to(torch.float32)
        pow_10 = to_10.pow(2)
        to_10 = None
        variance_6 = pow_10.mean(-1, keepdim=True)
        pow_10 = None
        add_22 = variance_6 + 1e-06
        variance_6 = None
        rsqrt_6 = torch.rsqrt(add_22)
        add_22 = None
        hidden_states_22 = hidden_states_21 * rsqrt_6
        rsqrt_6 = None
        normed_hidden_states_3 = (
            l_self_modules_block_modules_3_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_22
        )
        l_self_modules_block_modules_3_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_22
        ) = None
        query_states_6 = torch._C._nn.linear(
            normed_hidden_states_3,
            l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_13 = query_states_6.view(1, -1, 6, 64)
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
        view_14 = key_states_6.view(1, -1, 6, 64)
        key_states_6 = None
        key_states_7 = view_14.transpose(1, 2)
        view_14 = None
        view_15 = value_states_6.view(1, -1, 6, 64)
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
        attn_output_14 = attn_output_13.view(1, -1, 384)
        attn_output_13 = None
        attn_output_15 = torch._C._nn.linear(
            attn_output_14,
            l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_14 = l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_14 = torch.nn.functional.dropout(attn_output_15, 0.1, False, False)
        attn_output_15 = None
        hidden_states_23 = hidden_states_21 + dropout_14
        hidden_states_21 = dropout_14 = None
        to_11 = hidden_states_23.to(torch.float32)
        pow_11 = to_11.pow(2)
        to_11 = None
        variance_7 = pow_11.mean(-1, keepdim=True)
        pow_11 = None
        add_24 = variance_7 + 1e-06
        variance_7 = None
        rsqrt_7 = torch.rsqrt(add_24)
        add_24 = None
        hidden_states_24 = hidden_states_23 * rsqrt_7
        rsqrt_7 = None
        forwarded_states_3 = (
            l_self_modules_block_modules_3_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_24
        )
        l_self_modules_block_modules_3_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_24
        ) = None
        linear_25 = torch._C._nn.linear(
            forwarded_states_3,
            l_self_modules_block_modules_3_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_0_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_3_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_0_parameters_weight_ = (
            None
        )
        mul_34 = 0.5 * linear_25
        pow_12 = torch.pow(linear_25, 3.0)
        mul_35 = 0.044715 * pow_12
        pow_12 = None
        add_25 = linear_25 + mul_35
        linear_25 = mul_35 = None
        mul_36 = 0.7978845608028654 * add_25
        add_25 = None
        tanh_3 = torch.tanh(mul_36)
        mul_36 = None
        add_26 = 1.0 + tanh_3
        tanh_3 = None
        hidden_gelu_3 = mul_34 * add_26
        mul_34 = add_26 = None
        hidden_linear_3 = torch._C._nn.linear(
            forwarded_states_3,
            l_self_modules_block_modules_3_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_1_parameters_weight_,
            None,
        )
        forwarded_states_3 = l_self_modules_block_modules_3_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_1_parameters_weight_ = (None)
        hidden_states_25 = hidden_gelu_3 * hidden_linear_3
        hidden_gelu_3 = hidden_linear_3 = None
        hidden_states_26 = torch.nn.functional.dropout(
            hidden_states_25, 0.1, False, False
        )
        hidden_states_25 = None
        hidden_states_27 = torch._C._nn.linear(
            hidden_states_26,
            l_self_modules_block_modules_3_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_26 = l_self_modules_block_modules_3_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_16 = torch.nn.functional.dropout(hidden_states_27, 0.1, False, False)
        hidden_states_27 = None
        hidden_states_28 = hidden_states_23 + dropout_16
        hidden_states_23 = dropout_16 = None
        to_12 = hidden_states_28.to(torch.float32)
        pow_13 = to_12.pow(2)
        to_12 = None
        variance_8 = pow_13.mean(-1, keepdim=True)
        pow_13 = None
        add_28 = variance_8 + 1e-06
        variance_8 = None
        rsqrt_8 = torch.rsqrt(add_28)
        add_28 = None
        hidden_states_29 = hidden_states_28 * rsqrt_8
        rsqrt_8 = None
        normed_hidden_states_4 = (
            l_self_modules_block_modules_4_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_29
        )
        l_self_modules_block_modules_4_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_29
        ) = None
        query_states_8 = torch._C._nn.linear(
            normed_hidden_states_4,
            l_self_modules_block_modules_4_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_4_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_17 = query_states_8.view(1, -1, 6, 64)
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
        view_18 = key_states_8.view(1, -1, 6, 64)
        key_states_8 = None
        key_states_9 = view_18.transpose(1, 2)
        view_18 = None
        view_19 = value_states_8.view(1, -1, 6, 64)
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
        attn_output_18 = attn_output_17.view(1, -1, 384)
        attn_output_17 = None
        attn_output_19 = torch._C._nn.linear(
            attn_output_18,
            l_self_modules_block_modules_4_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_18 = l_self_modules_block_modules_4_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_18 = torch.nn.functional.dropout(attn_output_19, 0.1, False, False)
        attn_output_19 = None
        hidden_states_30 = hidden_states_28 + dropout_18
        hidden_states_28 = dropout_18 = None
        to_13 = hidden_states_30.to(torch.float32)
        pow_14 = to_13.pow(2)
        to_13 = None
        variance_9 = pow_14.mean(-1, keepdim=True)
        pow_14 = None
        add_30 = variance_9 + 1e-06
        variance_9 = None
        rsqrt_9 = torch.rsqrt(add_30)
        add_30 = None
        hidden_states_31 = hidden_states_30 * rsqrt_9
        rsqrt_9 = None
        forwarded_states_4 = (
            l_self_modules_block_modules_4_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_31
        )
        l_self_modules_block_modules_4_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_31
        ) = None
        linear_32 = torch._C._nn.linear(
            forwarded_states_4,
            l_self_modules_block_modules_4_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_0_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_4_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_0_parameters_weight_ = (
            None
        )
        mul_43 = 0.5 * linear_32
        pow_15 = torch.pow(linear_32, 3.0)
        mul_44 = 0.044715 * pow_15
        pow_15 = None
        add_31 = linear_32 + mul_44
        linear_32 = mul_44 = None
        mul_45 = 0.7978845608028654 * add_31
        add_31 = None
        tanh_4 = torch.tanh(mul_45)
        mul_45 = None
        add_32 = 1.0 + tanh_4
        tanh_4 = None
        hidden_gelu_4 = mul_43 * add_32
        mul_43 = add_32 = None
        hidden_linear_4 = torch._C._nn.linear(
            forwarded_states_4,
            l_self_modules_block_modules_4_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_1_parameters_weight_,
            None,
        )
        forwarded_states_4 = l_self_modules_block_modules_4_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_1_parameters_weight_ = (None)
        hidden_states_32 = hidden_gelu_4 * hidden_linear_4
        hidden_gelu_4 = hidden_linear_4 = None
        hidden_states_33 = torch.nn.functional.dropout(
            hidden_states_32, 0.1, False, False
        )
        hidden_states_32 = None
        hidden_states_34 = torch._C._nn.linear(
            hidden_states_33,
            l_self_modules_block_modules_4_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_33 = l_self_modules_block_modules_4_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_20 = torch.nn.functional.dropout(hidden_states_34, 0.1, False, False)
        hidden_states_34 = None
        hidden_states_35 = hidden_states_30 + dropout_20
        hidden_states_30 = dropout_20 = None
        to_14 = hidden_states_35.to(torch.float32)
        pow_16 = to_14.pow(2)
        to_14 = None
        variance_10 = pow_16.mean(-1, keepdim=True)
        pow_16 = None
        add_34 = variance_10 + 1e-06
        variance_10 = None
        rsqrt_10 = torch.rsqrt(add_34)
        add_34 = None
        hidden_states_36 = hidden_states_35 * rsqrt_10
        rsqrt_10 = None
        normed_hidden_states_5 = (
            l_self_modules_block_modules_5_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_36
        )
        l_self_modules_block_modules_5_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_36
        ) = None
        query_states_10 = torch._C._nn.linear(
            normed_hidden_states_5,
            l_self_modules_block_modules_5_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_5_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_21 = query_states_10.view(1, -1, 6, 64)
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
        view_22 = key_states_10.view(1, -1, 6, 64)
        key_states_10 = None
        key_states_11 = view_22.transpose(1, 2)
        view_22 = None
        view_23 = value_states_10.view(1, -1, 6, 64)
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
        attn_output_22 = attn_output_21.view(1, -1, 384)
        attn_output_21 = None
        attn_output_23 = torch._C._nn.linear(
            attn_output_22,
            l_self_modules_block_modules_5_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_22 = l_self_modules_block_modules_5_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_22 = torch.nn.functional.dropout(attn_output_23, 0.1, False, False)
        attn_output_23 = None
        hidden_states_37 = hidden_states_35 + dropout_22
        hidden_states_35 = dropout_22 = None
        to_15 = hidden_states_37.to(torch.float32)
        pow_17 = to_15.pow(2)
        to_15 = None
        variance_11 = pow_17.mean(-1, keepdim=True)
        pow_17 = None
        add_36 = variance_11 + 1e-06
        variance_11 = None
        rsqrt_11 = torch.rsqrt(add_36)
        add_36 = None
        hidden_states_38 = hidden_states_37 * rsqrt_11
        rsqrt_11 = None
        forwarded_states_5 = (
            l_self_modules_block_modules_5_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_38
        )
        l_self_modules_block_modules_5_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_38
        ) = None
        linear_39 = torch._C._nn.linear(
            forwarded_states_5,
            l_self_modules_block_modules_5_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_0_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_5_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_0_parameters_weight_ = (
            None
        )
        mul_52 = 0.5 * linear_39
        pow_18 = torch.pow(linear_39, 3.0)
        mul_53 = 0.044715 * pow_18
        pow_18 = None
        add_37 = linear_39 + mul_53
        linear_39 = mul_53 = None
        mul_54 = 0.7978845608028654 * add_37
        add_37 = None
        tanh_5 = torch.tanh(mul_54)
        mul_54 = None
        add_38 = 1.0 + tanh_5
        tanh_5 = None
        hidden_gelu_5 = mul_52 * add_38
        mul_52 = add_38 = None
        hidden_linear_5 = torch._C._nn.linear(
            forwarded_states_5,
            l_self_modules_block_modules_5_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_1_parameters_weight_,
            None,
        )
        forwarded_states_5 = l_self_modules_block_modules_5_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_1_parameters_weight_ = (None)
        hidden_states_39 = hidden_gelu_5 * hidden_linear_5
        hidden_gelu_5 = hidden_linear_5 = None
        hidden_states_40 = torch.nn.functional.dropout(
            hidden_states_39, 0.1, False, False
        )
        hidden_states_39 = None
        hidden_states_41 = torch._C._nn.linear(
            hidden_states_40,
            l_self_modules_block_modules_5_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_40 = l_self_modules_block_modules_5_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_24 = torch.nn.functional.dropout(hidden_states_41, 0.1, False, False)
        hidden_states_41 = None
        hidden_states_42 = hidden_states_37 + dropout_24
        hidden_states_37 = dropout_24 = None
        to_16 = hidden_states_42.to(torch.float32)
        pow_19 = to_16.pow(2)
        to_16 = None
        variance_12 = pow_19.mean(-1, keepdim=True)
        pow_19 = None
        add_40 = variance_12 + 1e-06
        variance_12 = None
        rsqrt_12 = torch.rsqrt(add_40)
        add_40 = None
        hidden_states_43 = hidden_states_42 * rsqrt_12
        rsqrt_12 = None
        normed_hidden_states_6 = (
            l_self_modules_block_modules_6_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_43
        )
        l_self_modules_block_modules_6_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_43
        ) = None
        query_states_12 = torch._C._nn.linear(
            normed_hidden_states_6,
            l_self_modules_block_modules_6_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_6_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_25 = query_states_12.view(1, -1, 6, 64)
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
        view_26 = key_states_12.view(1, -1, 6, 64)
        key_states_12 = None
        key_states_13 = view_26.transpose(1, 2)
        view_26 = None
        view_27 = value_states_12.view(1, -1, 6, 64)
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
        attn_output_26 = attn_output_25.view(1, -1, 384)
        attn_output_25 = None
        attn_output_27 = torch._C._nn.linear(
            attn_output_26,
            l_self_modules_block_modules_6_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_26 = l_self_modules_block_modules_6_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_26 = torch.nn.functional.dropout(attn_output_27, 0.1, False, False)
        attn_output_27 = None
        hidden_states_44 = hidden_states_42 + dropout_26
        hidden_states_42 = dropout_26 = None
        to_17 = hidden_states_44.to(torch.float32)
        pow_20 = to_17.pow(2)
        to_17 = None
        variance_13 = pow_20.mean(-1, keepdim=True)
        pow_20 = None
        add_42 = variance_13 + 1e-06
        variance_13 = None
        rsqrt_13 = torch.rsqrt(add_42)
        add_42 = None
        hidden_states_45 = hidden_states_44 * rsqrt_13
        rsqrt_13 = None
        forwarded_states_6 = (
            l_self_modules_block_modules_6_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_45
        )
        l_self_modules_block_modules_6_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_45
        ) = None
        linear_46 = torch._C._nn.linear(
            forwarded_states_6,
            l_self_modules_block_modules_6_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_0_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_6_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_0_parameters_weight_ = (
            None
        )
        mul_61 = 0.5 * linear_46
        pow_21 = torch.pow(linear_46, 3.0)
        mul_62 = 0.044715 * pow_21
        pow_21 = None
        add_43 = linear_46 + mul_62
        linear_46 = mul_62 = None
        mul_63 = 0.7978845608028654 * add_43
        add_43 = None
        tanh_6 = torch.tanh(mul_63)
        mul_63 = None
        add_44 = 1.0 + tanh_6
        tanh_6 = None
        hidden_gelu_6 = mul_61 * add_44
        mul_61 = add_44 = None
        hidden_linear_6 = torch._C._nn.linear(
            forwarded_states_6,
            l_self_modules_block_modules_6_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_1_parameters_weight_,
            None,
        )
        forwarded_states_6 = l_self_modules_block_modules_6_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_1_parameters_weight_ = (None)
        hidden_states_46 = hidden_gelu_6 * hidden_linear_6
        hidden_gelu_6 = hidden_linear_6 = None
        hidden_states_47 = torch.nn.functional.dropout(
            hidden_states_46, 0.1, False, False
        )
        hidden_states_46 = None
        hidden_states_48 = torch._C._nn.linear(
            hidden_states_47,
            l_self_modules_block_modules_6_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_47 = l_self_modules_block_modules_6_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_28 = torch.nn.functional.dropout(hidden_states_48, 0.1, False, False)
        hidden_states_48 = None
        hidden_states_49 = hidden_states_44 + dropout_28
        hidden_states_44 = dropout_28 = None
        to_18 = hidden_states_49.to(torch.float32)
        pow_22 = to_18.pow(2)
        to_18 = None
        variance_14 = pow_22.mean(-1, keepdim=True)
        pow_22 = None
        add_46 = variance_14 + 1e-06
        variance_14 = None
        rsqrt_14 = torch.rsqrt(add_46)
        add_46 = None
        hidden_states_50 = hidden_states_49 * rsqrt_14
        rsqrt_14 = None
        normed_hidden_states_7 = (
            l_self_modules_block_modules_7_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_50
        )
        l_self_modules_block_modules_7_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_50
        ) = None
        query_states_14 = torch._C._nn.linear(
            normed_hidden_states_7,
            l_self_modules_block_modules_7_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_7_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_29 = query_states_14.view(1, -1, 6, 64)
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
        view_30 = key_states_14.view(1, -1, 6, 64)
        key_states_14 = None
        key_states_15 = view_30.transpose(1, 2)
        view_30 = None
        view_31 = value_states_14.view(1, -1, 6, 64)
        value_states_14 = None
        value_states_15 = view_31.transpose(1, 2)
        view_31 = None
        transpose_38 = key_states_15.transpose(3, 2)
        key_states_15 = None
        scores_14 = torch.matmul(query_states_15, transpose_38)
        query_states_15 = transpose_38 = None
        scores_14 += position_bias_1
        scores_15 = scores_14
        scores_14 = position_bias_1 = None
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
        attn_output_30 = attn_output_29.view(1, -1, 384)
        attn_output_29 = None
        attn_output_31 = torch._C._nn.linear(
            attn_output_30,
            l_self_modules_block_modules_7_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_30 = l_self_modules_block_modules_7_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_30 = torch.nn.functional.dropout(attn_output_31, 0.1, False, False)
        attn_output_31 = None
        hidden_states_51 = hidden_states_49 + dropout_30
        hidden_states_49 = dropout_30 = None
        to_19 = hidden_states_51.to(torch.float32)
        pow_23 = to_19.pow(2)
        to_19 = None
        variance_15 = pow_23.mean(-1, keepdim=True)
        pow_23 = None
        add_48 = variance_15 + 1e-06
        variance_15 = None
        rsqrt_15 = torch.rsqrt(add_48)
        add_48 = None
        hidden_states_52 = hidden_states_51 * rsqrt_15
        rsqrt_15 = None
        forwarded_states_7 = (
            l_self_modules_block_modules_7_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_52
        )
        l_self_modules_block_modules_7_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_52
        ) = None
        linear_53 = torch._C._nn.linear(
            forwarded_states_7,
            l_self_modules_block_modules_7_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_0_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_7_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_0_parameters_weight_ = (
            None
        )
        mul_70 = 0.5 * linear_53
        pow_24 = torch.pow(linear_53, 3.0)
        mul_71 = 0.044715 * pow_24
        pow_24 = None
        add_49 = linear_53 + mul_71
        linear_53 = mul_71 = None
        mul_72 = 0.7978845608028654 * add_49
        add_49 = None
        tanh_7 = torch.tanh(mul_72)
        mul_72 = None
        add_50 = 1.0 + tanh_7
        tanh_7 = None
        hidden_gelu_7 = mul_70 * add_50
        mul_70 = add_50 = None
        hidden_linear_7 = torch._C._nn.linear(
            forwarded_states_7,
            l_self_modules_block_modules_7_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_1_parameters_weight_,
            None,
        )
        forwarded_states_7 = l_self_modules_block_modules_7_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_1_parameters_weight_ = (None)
        hidden_states_53 = hidden_gelu_7 * hidden_linear_7
        hidden_gelu_7 = hidden_linear_7 = None
        hidden_states_54 = torch.nn.functional.dropout(
            hidden_states_53, 0.1, False, False
        )
        hidden_states_53 = None
        hidden_states_55 = torch._C._nn.linear(
            hidden_states_54,
            l_self_modules_block_modules_7_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_54 = l_self_modules_block_modules_7_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_32 = torch.nn.functional.dropout(hidden_states_55, 0.1, False, False)
        hidden_states_55 = None
        hidden_states_56 = hidden_states_51 + dropout_32
        hidden_states_51 = dropout_32 = None
        to_20 = hidden_states_56.to(torch.float32)
        pow_25 = to_20.pow(2)
        to_20 = None
        variance_16 = pow_25.mean(-1, keepdim=True)
        pow_25 = None
        add_52 = variance_16 + 1e-06
        variance_16 = None
        rsqrt_16 = torch.rsqrt(add_52)
        add_52 = None
        hidden_states_57 = hidden_states_56 * rsqrt_16
        hidden_states_56 = rsqrt_16 = None
        hidden_states_58 = (
            l_self_modules_final_layer_norm_parameters_weight_ * hidden_states_57
        )
        l_self_modules_final_layer_norm_parameters_weight_ = hidden_states_57 = None
        hidden_states_59 = torch.nn.functional.dropout(
            hidden_states_58, 0.1, False, False
        )
        hidden_states_58 = None
        return (hidden_states_59,)
