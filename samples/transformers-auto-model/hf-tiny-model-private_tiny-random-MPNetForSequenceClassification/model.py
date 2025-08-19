import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_input_ids_: torch.Tensor,
        L_attention_mask_: torch.Tensor,
        L_self_modules_embeddings_modules_word_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_position_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_relative_attention_bias_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_q_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_k_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_v_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_o_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_q_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_k_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_v_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_o_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_q_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_k_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_v_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_o_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_q_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_k_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_v_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_o_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_q_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_k_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_v_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_o_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_pooler_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_pooler_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_input_ids_ = L_input_ids_
        l_attention_mask_ = L_attention_mask_
        l_self_modules_embeddings_modules_word_embeddings_parameters_weight_ = (
            L_self_modules_embeddings_modules_word_embeddings_parameters_weight_
        )
        l_self_modules_embeddings_modules_position_embeddings_parameters_weight_ = (
            L_self_modules_embeddings_modules_position_embeddings_parameters_weight_
        )
        l_self_modules_embeddings_modules_layer_norm_parameters_weight_ = (
            L_self_modules_embeddings_modules_LayerNorm_parameters_weight_
        )
        l_self_modules_embeddings_modules_layer_norm_parameters_bias_ = (
            L_self_modules_embeddings_modules_LayerNorm_parameters_bias_
        )
        l_self_modules_encoder_modules_relative_attention_bias_parameters_weight_ = (
            L_self_modules_encoder_modules_relative_attention_bias_parameters_weight_
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_q_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_q_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_q_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_q_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_k_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_k_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_k_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_k_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_v_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_v_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_v_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_v_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_o_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_o_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_o_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_o_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_q_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_q_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_q_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_q_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_k_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_k_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_k_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_k_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_v_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_v_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_v_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_v_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_o_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_o_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_o_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_o_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_q_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_q_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_q_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_q_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_k_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_k_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_k_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_k_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_v_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_v_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_v_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_v_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_o_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_o_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_o_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_o_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_q_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_q_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_q_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_q_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_k_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_k_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_k_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_k_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_v_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_v_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_v_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_v_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_o_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_o_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_o_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_o_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_q_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_q_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_q_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_q_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_k_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_k_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_k_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_k_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_v_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_v_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_v_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_v_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_o_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_o_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_o_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_o_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_pooler_modules_dense_parameters_weight_ = (
            L_self_modules_pooler_modules_dense_parameters_weight_
        )
        l_self_modules_pooler_modules_dense_parameters_bias_ = (
            L_self_modules_pooler_modules_dense_parameters_bias_
        )
        extended_attention_mask = l_attention_mask_[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        l_attention_mask_ = None
        extended_attention_mask_1 = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = None
        sub = 1.0 - extended_attention_mask_1
        extended_attention_mask_1 = None
        extended_attention_mask_2 = sub * -3.4028234663852886e38
        sub = None
        ne = l_input_ids_.ne(1)
        mask = ne.int()
        ne = None
        cumsum = torch.cumsum(mask, dim=1)
        type_as = cumsum.type_as(mask)
        cumsum = None
        incremental_indices = type_as * mask
        type_as = mask = None
        long = incremental_indices.long()
        incremental_indices = None
        position_ids = long + 1
        long = None
        inputs_embeds = torch.nn.functional.embedding(
            l_input_ids_,
            l_self_modules_embeddings_modules_word_embeddings_parameters_weight_,
            1,
            None,
            2.0,
            False,
            False,
        )
        l_input_ids_ = (
            l_self_modules_embeddings_modules_word_embeddings_parameters_weight_
        ) = None
        position_embeddings = torch.nn.functional.embedding(
            position_ids,
            l_self_modules_embeddings_modules_position_embeddings_parameters_weight_,
            1,
            None,
            2.0,
            False,
            False,
        )
        position_ids = (
            l_self_modules_embeddings_modules_position_embeddings_parameters_weight_
        ) = None
        embeddings = inputs_embeds + position_embeddings
        inputs_embeds = position_embeddings = None
        embeddings_1 = torch.nn.functional.layer_norm(
            embeddings,
            (64,),
            l_self_modules_embeddings_modules_layer_norm_parameters_weight_,
            l_self_modules_embeddings_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        embeddings = (
            l_self_modules_embeddings_modules_layer_norm_parameters_weight_
        ) = l_self_modules_embeddings_modules_layer_norm_parameters_bias_ = None
        embeddings_2 = torch.nn.functional.dropout(embeddings_1, 0.1, False, False)
        embeddings_1 = None
        arange = torch.arange(45, dtype=torch.int64)
        context_position = arange[(slice(None, None, None), None)]
        arange = None
        arange_1 = torch.arange(45, dtype=torch.int64)
        memory_position = arange_1[(None, slice(None, None, None))]
        arange_1 = None
        relative_position = memory_position - context_position
        memory_position = context_position = None
        n = -relative_position
        relative_position = None
        lt = n < 0
        to_1 = lt.to(torch.int64)
        lt = None
        mul_2 = to_1 * 16
        to_1 = None
        ret = 0 + mul_2
        mul_2 = None
        n_1 = torch.abs(n)
        n = None
        is_small = n_1 < 8
        float_1 = n_1.float()
        truediv = float_1 / 8
        float_1 = None
        log = torch.log(truediv)
        truediv = None
        truediv_1 = log / 2.772588722239781
        log = None
        mul_3 = truediv_1 * 8
        truediv_1 = None
        to_2 = mul_3.to(torch.int64)
        mul_3 = None
        val_if_large = 8 + to_2
        to_2 = None
        full_like = torch.full_like(val_if_large, 15)
        val_if_large_1 = torch.min(val_if_large, full_like)
        val_if_large = full_like = None
        where = torch.where(is_small, n_1, val_if_large_1)
        is_small = n_1 = val_if_large_1 = None
        ret += where
        ret_1 = ret
        ret = where = None
        rp_bucket = ret_1.to(device(type="cuda", index=0))
        ret_1 = None
        values = torch.nn.functional.embedding(
            rp_bucket,
            l_self_modules_encoder_modules_relative_attention_bias_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        rp_bucket = (
            l_self_modules_encoder_modules_relative_attention_bias_parameters_weight_
        ) = None
        permute = values.permute([2, 0, 1])
        values = None
        values_1 = permute.unsqueeze(0)
        permute = None
        expand = values_1.expand((1, -1, 45, 45))
        values_1 = None
        values_2 = expand.contiguous()
        expand = None
        linear = torch._C._nn.linear(
            embeddings_2,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_q_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_q_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_q_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_q_parameters_bias_ = (None)
        view = linear.view(1, -1, 4, 16)
        linear = None
        q = view.transpose(1, 2)
        view = None
        linear_1 = torch._C._nn.linear(
            embeddings_2,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_k_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_k_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_k_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_k_parameters_bias_ = (None)
        view_1 = linear_1.view(1, -1, 4, 16)
        linear_1 = None
        k = view_1.transpose(1, 2)
        view_1 = None
        linear_2 = torch._C._nn.linear(
            embeddings_2,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_v_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_v_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_v_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_v_parameters_bias_ = (None)
        view_2 = linear_2.view(1, -1, 4, 16)
        linear_2 = None
        v = view_2.transpose(1, 2)
        view_2 = None
        transpose_3 = k.transpose(-1, -2)
        k = None
        attention_scores = torch.matmul(q, transpose_3)
        q = transpose_3 = None
        attention_scores_1 = attention_scores / 4.0
        attention_scores = None
        attention_scores_1 += values_2
        attention_scores_2 = attention_scores_1
        attention_scores_1 = None
        attention_scores_3 = attention_scores_2 + extended_attention_mask_2
        attention_scores_2 = None
        attention_probs = torch.nn.functional.softmax(attention_scores_3, dim=-1)
        attention_scores_3 = None
        attention_probs_1 = torch.nn.functional.dropout(
            attention_probs, 0.1, False, False
        )
        attention_probs = None
        c = torch.matmul(attention_probs_1, v)
        attention_probs_1 = v = None
        permute_1 = c.permute(0, 2, 1, 3)
        c = None
        c_1 = permute_1.contiguous()
        permute_1 = None
        c_2 = c_1.view(1, 45, 64)
        c_1 = None
        o = torch._C._nn.linear(
            c_2,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_o_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_o_parameters_bias_,
        )
        c_2 = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_o_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_o_parameters_bias_ = (None)
        dropout_2 = torch.nn.functional.dropout(o, 0.1, False, False)
        o = None
        add_5 = dropout_2 + embeddings_2
        dropout_2 = embeddings_2 = None
        attention_output = torch.nn.functional.layer_norm(
            add_5,
            (64,),
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_5 = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_layer_norm_parameters_bias_ = (None)
        hidden_states = torch._C._nn.linear(
            attention_output,
            l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_1 = torch._C._nn.gelu(hidden_states)
        hidden_states = None
        hidden_states_2 = torch._C._nn.linear(
            hidden_states_1,
            l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_1 = l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_3 = torch.nn.functional.dropout(
            hidden_states_2, 0.1, False, False
        )
        hidden_states_2 = None
        add_6 = hidden_states_3 + attention_output
        hidden_states_3 = attention_output = None
        hidden_states_4 = torch.nn.functional.layer_norm(
            add_6,
            (64,),
            l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_6 = l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_bias_ = (None)
        linear_6 = torch._C._nn.linear(
            hidden_states_4,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_q_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_q_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_q_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_q_parameters_bias_ = (None)
        view_4 = linear_6.view(1, -1, 4, 16)
        linear_6 = None
        q_1 = view_4.transpose(1, 2)
        view_4 = None
        linear_7 = torch._C._nn.linear(
            hidden_states_4,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_k_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_k_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_k_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_k_parameters_bias_ = (None)
        view_5 = linear_7.view(1, -1, 4, 16)
        linear_7 = None
        k_1 = view_5.transpose(1, 2)
        view_5 = None
        linear_8 = torch._C._nn.linear(
            hidden_states_4,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_v_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_v_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_v_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_v_parameters_bias_ = (None)
        view_6 = linear_8.view(1, -1, 4, 16)
        linear_8 = None
        v_1 = view_6.transpose(1, 2)
        view_6 = None
        transpose_7 = k_1.transpose(-1, -2)
        k_1 = None
        attention_scores_4 = torch.matmul(q_1, transpose_7)
        q_1 = transpose_7 = None
        attention_scores_5 = attention_scores_4 / 4.0
        attention_scores_4 = None
        attention_scores_5 += values_2
        attention_scores_6 = attention_scores_5
        attention_scores_5 = None
        attention_scores_7 = attention_scores_6 + extended_attention_mask_2
        attention_scores_6 = None
        attention_probs_2 = torch.nn.functional.softmax(attention_scores_7, dim=-1)
        attention_scores_7 = None
        attention_probs_3 = torch.nn.functional.dropout(
            attention_probs_2, 0.1, False, False
        )
        attention_probs_2 = None
        c_3 = torch.matmul(attention_probs_3, v_1)
        attention_probs_3 = v_1 = None
        permute_2 = c_3.permute(0, 2, 1, 3)
        c_3 = None
        c_4 = permute_2.contiguous()
        permute_2 = None
        c_5 = c_4.view(1, 45, 64)
        c_4 = None
        o_1 = torch._C._nn.linear(
            c_5,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_o_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_o_parameters_bias_,
        )
        c_5 = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_o_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_o_parameters_bias_ = (None)
        dropout_5 = torch.nn.functional.dropout(o_1, 0.1, False, False)
        o_1 = None
        add_8 = dropout_5 + hidden_states_4
        dropout_5 = hidden_states_4 = None
        attention_output_1 = torch.nn.functional.layer_norm(
            add_8,
            (64,),
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_8 = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_5 = torch._C._nn.linear(
            attention_output_1,
            l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_6 = torch._C._nn.gelu(hidden_states_5)
        hidden_states_5 = None
        hidden_states_7 = torch._C._nn.linear(
            hidden_states_6,
            l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_6 = l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_8 = torch.nn.functional.dropout(
            hidden_states_7, 0.1, False, False
        )
        hidden_states_7 = None
        add_9 = hidden_states_8 + attention_output_1
        hidden_states_8 = attention_output_1 = None
        hidden_states_9 = torch.nn.functional.layer_norm(
            add_9,
            (64,),
            l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_9 = l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_bias_ = (None)
        linear_12 = torch._C._nn.linear(
            hidden_states_9,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_q_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_q_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_q_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_q_parameters_bias_ = (None)
        view_8 = linear_12.view(1, -1, 4, 16)
        linear_12 = None
        q_2 = view_8.transpose(1, 2)
        view_8 = None
        linear_13 = torch._C._nn.linear(
            hidden_states_9,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_k_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_k_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_k_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_k_parameters_bias_ = (None)
        view_9 = linear_13.view(1, -1, 4, 16)
        linear_13 = None
        k_2 = view_9.transpose(1, 2)
        view_9 = None
        linear_14 = torch._C._nn.linear(
            hidden_states_9,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_v_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_v_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_v_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_v_parameters_bias_ = (None)
        view_10 = linear_14.view(1, -1, 4, 16)
        linear_14 = None
        v_2 = view_10.transpose(1, 2)
        view_10 = None
        transpose_11 = k_2.transpose(-1, -2)
        k_2 = None
        attention_scores_8 = torch.matmul(q_2, transpose_11)
        q_2 = transpose_11 = None
        attention_scores_9 = attention_scores_8 / 4.0
        attention_scores_8 = None
        attention_scores_9 += values_2
        attention_scores_10 = attention_scores_9
        attention_scores_9 = None
        attention_scores_11 = attention_scores_10 + extended_attention_mask_2
        attention_scores_10 = None
        attention_probs_4 = torch.nn.functional.softmax(attention_scores_11, dim=-1)
        attention_scores_11 = None
        attention_probs_5 = torch.nn.functional.dropout(
            attention_probs_4, 0.1, False, False
        )
        attention_probs_4 = None
        c_6 = torch.matmul(attention_probs_5, v_2)
        attention_probs_5 = v_2 = None
        permute_3 = c_6.permute(0, 2, 1, 3)
        c_6 = None
        c_7 = permute_3.contiguous()
        permute_3 = None
        c_8 = c_7.view(1, 45, 64)
        c_7 = None
        o_2 = torch._C._nn.linear(
            c_8,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_o_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_o_parameters_bias_,
        )
        c_8 = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_o_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_o_parameters_bias_ = (None)
        dropout_8 = torch.nn.functional.dropout(o_2, 0.1, False, False)
        o_2 = None
        add_11 = dropout_8 + hidden_states_9
        dropout_8 = hidden_states_9 = None
        attention_output_2 = torch.nn.functional.layer_norm(
            add_11,
            (64,),
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_11 = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_10 = torch._C._nn.linear(
            attention_output_2,
            l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_11 = torch._C._nn.gelu(hidden_states_10)
        hidden_states_10 = None
        hidden_states_12 = torch._C._nn.linear(
            hidden_states_11,
            l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_11 = l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_13 = torch.nn.functional.dropout(
            hidden_states_12, 0.1, False, False
        )
        hidden_states_12 = None
        add_12 = hidden_states_13 + attention_output_2
        hidden_states_13 = attention_output_2 = None
        hidden_states_14 = torch.nn.functional.layer_norm(
            add_12,
            (64,),
            l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_12 = l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_bias_ = (None)
        linear_18 = torch._C._nn.linear(
            hidden_states_14,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_q_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_q_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_q_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_q_parameters_bias_ = (None)
        view_12 = linear_18.view(1, -1, 4, 16)
        linear_18 = None
        q_3 = view_12.transpose(1, 2)
        view_12 = None
        linear_19 = torch._C._nn.linear(
            hidden_states_14,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_k_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_k_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_k_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_k_parameters_bias_ = (None)
        view_13 = linear_19.view(1, -1, 4, 16)
        linear_19 = None
        k_3 = view_13.transpose(1, 2)
        view_13 = None
        linear_20 = torch._C._nn.linear(
            hidden_states_14,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_v_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_v_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_v_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_v_parameters_bias_ = (None)
        view_14 = linear_20.view(1, -1, 4, 16)
        linear_20 = None
        v_3 = view_14.transpose(1, 2)
        view_14 = None
        transpose_15 = k_3.transpose(-1, -2)
        k_3 = None
        attention_scores_12 = torch.matmul(q_3, transpose_15)
        q_3 = transpose_15 = None
        attention_scores_13 = attention_scores_12 / 4.0
        attention_scores_12 = None
        attention_scores_13 += values_2
        attention_scores_14 = attention_scores_13
        attention_scores_13 = None
        attention_scores_15 = attention_scores_14 + extended_attention_mask_2
        attention_scores_14 = None
        attention_probs_6 = torch.nn.functional.softmax(attention_scores_15, dim=-1)
        attention_scores_15 = None
        attention_probs_7 = torch.nn.functional.dropout(
            attention_probs_6, 0.1, False, False
        )
        attention_probs_6 = None
        c_9 = torch.matmul(attention_probs_7, v_3)
        attention_probs_7 = v_3 = None
        permute_4 = c_9.permute(0, 2, 1, 3)
        c_9 = None
        c_10 = permute_4.contiguous()
        permute_4 = None
        c_11 = c_10.view(1, 45, 64)
        c_10 = None
        o_3 = torch._C._nn.linear(
            c_11,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_o_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_o_parameters_bias_,
        )
        c_11 = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_o_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_o_parameters_bias_ = (None)
        dropout_11 = torch.nn.functional.dropout(o_3, 0.1, False, False)
        o_3 = None
        add_14 = dropout_11 + hidden_states_14
        dropout_11 = hidden_states_14 = None
        attention_output_3 = torch.nn.functional.layer_norm(
            add_14,
            (64,),
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_14 = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_15 = torch._C._nn.linear(
            attention_output_3,
            l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_16 = torch._C._nn.gelu(hidden_states_15)
        hidden_states_15 = None
        hidden_states_17 = torch._C._nn.linear(
            hidden_states_16,
            l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_16 = l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_18 = torch.nn.functional.dropout(
            hidden_states_17, 0.1, False, False
        )
        hidden_states_17 = None
        add_15 = hidden_states_18 + attention_output_3
        hidden_states_18 = attention_output_3 = None
        hidden_states_19 = torch.nn.functional.layer_norm(
            add_15,
            (64,),
            l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_15 = l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_bias_ = (None)
        linear_24 = torch._C._nn.linear(
            hidden_states_19,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_q_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_q_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_q_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_q_parameters_bias_ = (None)
        view_16 = linear_24.view(1, -1, 4, 16)
        linear_24 = None
        q_4 = view_16.transpose(1, 2)
        view_16 = None
        linear_25 = torch._C._nn.linear(
            hidden_states_19,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_k_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_k_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_k_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_k_parameters_bias_ = (None)
        view_17 = linear_25.view(1, -1, 4, 16)
        linear_25 = None
        k_4 = view_17.transpose(1, 2)
        view_17 = None
        linear_26 = torch._C._nn.linear(
            hidden_states_19,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_v_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_v_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_v_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_v_parameters_bias_ = (None)
        view_18 = linear_26.view(1, -1, 4, 16)
        linear_26 = None
        v_4 = view_18.transpose(1, 2)
        view_18 = None
        transpose_19 = k_4.transpose(-1, -2)
        k_4 = None
        attention_scores_16 = torch.matmul(q_4, transpose_19)
        q_4 = transpose_19 = None
        attention_scores_17 = attention_scores_16 / 4.0
        attention_scores_16 = None
        attention_scores_17 += values_2
        attention_scores_18 = attention_scores_17
        attention_scores_17 = values_2 = None
        attention_scores_19 = attention_scores_18 + extended_attention_mask_2
        attention_scores_18 = extended_attention_mask_2 = None
        attention_probs_8 = torch.nn.functional.softmax(attention_scores_19, dim=-1)
        attention_scores_19 = None
        attention_probs_9 = torch.nn.functional.dropout(
            attention_probs_8, 0.1, False, False
        )
        attention_probs_8 = None
        c_12 = torch.matmul(attention_probs_9, v_4)
        attention_probs_9 = v_4 = None
        permute_5 = c_12.permute(0, 2, 1, 3)
        c_12 = None
        c_13 = permute_5.contiguous()
        permute_5 = None
        c_14 = c_13.view(1, 45, 64)
        c_13 = None
        o_4 = torch._C._nn.linear(
            c_14,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_o_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_o_parameters_bias_,
        )
        c_14 = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_o_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_o_parameters_bias_ = (None)
        dropout_14 = torch.nn.functional.dropout(o_4, 0.1, False, False)
        o_4 = None
        add_17 = dropout_14 + hidden_states_19
        dropout_14 = hidden_states_19 = None
        attention_output_4 = torch.nn.functional.layer_norm(
            add_17,
            (64,),
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_17 = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_20 = torch._C._nn.linear(
            attention_output_4,
            l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_21 = torch._C._nn.gelu(hidden_states_20)
        hidden_states_20 = None
        hidden_states_22 = torch._C._nn.linear(
            hidden_states_21,
            l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_21 = l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_23 = torch.nn.functional.dropout(
            hidden_states_22, 0.1, False, False
        )
        hidden_states_22 = None
        add_18 = hidden_states_23 + attention_output_4
        hidden_states_23 = attention_output_4 = None
        hidden_states_24 = torch.nn.functional.layer_norm(
            add_18,
            (64,),
            l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_18 = l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_bias_ = (None)
        first_token_tensor = hidden_states_24[(slice(None, None, None), 0)]
        pooled_output = torch._C._nn.linear(
            first_token_tensor,
            l_self_modules_pooler_modules_dense_parameters_weight_,
            l_self_modules_pooler_modules_dense_parameters_bias_,
        )
        first_token_tensor = (
            l_self_modules_pooler_modules_dense_parameters_weight_
        ) = l_self_modules_pooler_modules_dense_parameters_bias_ = None
        pooled_output_1 = torch.tanh(pooled_output)
        pooled_output = None
        return (hidden_states_24, pooled_output_1)
