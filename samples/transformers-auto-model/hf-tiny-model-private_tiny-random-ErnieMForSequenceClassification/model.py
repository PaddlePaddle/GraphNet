import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_input_ids_: torch.Tensor,
        L_self_modules_embeddings_modules_word_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_position_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_0_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_0_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_0_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_0_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_1_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_1_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_1_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_1_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_2_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_2_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_2_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_2_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_3_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_3_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_3_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_3_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_3_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_3_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_4_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_4_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_4_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_4_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_4_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_4_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_pooler_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_pooler_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_input_ids_ = L_input_ids_
        l_self_modules_embeddings_modules_word_embeddings_parameters_weight_ = (
            L_self_modules_embeddings_modules_word_embeddings_parameters_weight_
        )
        l_self_modules_embeddings_modules_position_embeddings_parameters_weight_ = (
            L_self_modules_embeddings_modules_position_embeddings_parameters_weight_
        )
        l_self_modules_embeddings_modules_layer_norm_parameters_weight_ = (
            L_self_modules_embeddings_modules_layer_norm_parameters_weight_
        )
        l_self_modules_embeddings_modules_layer_norm_parameters_bias_ = (
            L_self_modules_embeddings_modules_layer_norm_parameters_bias_
        )
        l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_0_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_0_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_0_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_0_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_0_modules_linear1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_0_modules_linear1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_0_modules_linear1_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_0_modules_linear1_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_0_modules_linear2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_0_modules_linear2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_0_modules_linear2_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_0_modules_linear2_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_0_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_0_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_0_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_0_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_1_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_1_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_1_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_1_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_1_modules_linear1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_1_modules_linear1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_1_modules_linear1_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_1_modules_linear1_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_1_modules_linear2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_1_modules_linear2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_1_modules_linear2_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_1_modules_linear2_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_1_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_1_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_1_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_1_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_2_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_2_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_2_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_2_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_2_modules_linear1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_2_modules_linear1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_2_modules_linear1_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_2_modules_linear1_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_2_modules_linear2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_2_modules_linear2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_2_modules_linear2_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_2_modules_linear2_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_2_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_2_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_2_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_2_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_3_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_3_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_3_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_3_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_3_modules_linear1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_3_modules_linear1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_3_modules_linear1_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_3_modules_linear1_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_3_modules_linear2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_3_modules_linear2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_3_modules_linear2_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_3_modules_linear2_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_3_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_3_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_3_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_3_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_4_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_4_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_4_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_4_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_4_modules_linear1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_4_modules_linear1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_4_modules_linear1_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_4_modules_linear1_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_4_modules_linear2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_4_modules_linear2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_4_modules_linear2_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_4_modules_linear2_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_4_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_4_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_4_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_4_modules_norm2_parameters_bias_
        l_self_modules_pooler_modules_dense_parameters_weight_ = (
            L_self_modules_pooler_modules_dense_parameters_weight_
        )
        l_self_modules_pooler_modules_dense_parameters_bias_ = (
            L_self_modules_pooler_modules_dense_parameters_bias_
        )
        eq = l_input_ids_.__eq__(1)
        attention_mask = eq.to(torch.float32)
        eq = None
        attention_mask *= -3.4028234663852886e38
        attention_mask_1 = attention_mask
        attention_mask = None
        unsqueeze = attention_mask_1.unsqueeze(1)
        attention_mask_1 = None
        extended_attention_mask = unsqueeze.unsqueeze(1)
        unsqueeze = None
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
        ones = torch.ones(
            (1, 15), dtype=torch.int64, device=device(type="cuda", index=0)
        )
        seq_length = torch.cumsum(ones, dim=1)
        position_ids = seq_length - ones
        seq_length = ones = None
        position_ids += 2
        position_ids_1 = position_ids
        position_ids = None
        position_embeddings = torch.nn.functional.embedding(
            position_ids_1,
            l_self_modules_embeddings_modules_position_embeddings_parameters_weight_,
            1,
            None,
            2.0,
            False,
            False,
        )
        position_ids_1 = (
            l_self_modules_embeddings_modules_position_embeddings_parameters_weight_
        ) = None
        embeddings = inputs_embeds + position_embeddings
        inputs_embeds = position_embeddings = None
        embeddings_1 = torch.nn.functional.layer_norm(
            embeddings,
            (32,),
            l_self_modules_embeddings_modules_layer_norm_parameters_weight_,
            l_self_modules_embeddings_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        embeddings = (
            l_self_modules_embeddings_modules_layer_norm_parameters_weight_
        ) = l_self_modules_embeddings_modules_layer_norm_parameters_bias_ = None
        embeddings_2 = torch.nn.functional.dropout(embeddings_1, 0.1, False, False)
        embeddings_1 = None
        mixed_query_layer = torch._C._nn.linear(
            embeddings_2,
            l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        linear_1 = torch._C._nn.linear(
            embeddings_2,
            l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        x = linear_1.view((1, 15, 4, 8))
        linear_1 = None
        key_layer = x.permute(0, 2, 1, 3)
        x = None
        linear_2 = torch._C._nn.linear(
            embeddings_2,
            l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        x_1 = linear_2.view((1, 15, 4, 8))
        linear_2 = None
        value_layer = x_1.permute(0, 2, 1, 3)
        x_1 = None
        x_2 = mixed_query_layer.view((1, 15, 4, 8))
        mixed_query_layer = None
        query_layer = x_2.permute(0, 2, 1, 3)
        x_2 = None
        transpose = key_layer.transpose(-1, -2)
        key_layer = None
        attention_scores = torch.matmul(query_layer, transpose)
        query_layer = transpose = None
        attention_scores_1 = attention_scores / 2.8284271247461903
        attention_scores = None
        attention_scores_2 = attention_scores_1 + extended_attention_mask
        attention_scores_1 = None
        attention_probs = torch.nn.functional.softmax(attention_scores_2, dim=-1)
        attention_scores_2 = None
        attention_probs_1 = torch.nn.functional.dropout(
            attention_probs, 0.1, False, False
        )
        attention_probs = None
        context_layer = torch.matmul(attention_probs_1, value_layer)
        attention_probs_1 = value_layer = None
        permute_3 = context_layer.permute(0, 2, 1, 3)
        context_layer = None
        context_layer_1 = permute_3.contiguous()
        permute_3 = None
        context_layer_2 = context_layer_1.view((1, 15, 32))
        context_layer_1 = None
        attention_output = torch._C._nn.linear(
            context_layer_2,
            l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        context_layer_2 = l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        dropout_2 = torch.nn.functional.dropout(attention_output, 0.1, False, False)
        attention_output = None
        hidden_states = embeddings_2 + dropout_2
        embeddings_2 = dropout_2 = None
        hidden_states_1 = torch.nn.functional.layer_norm(
            hidden_states,
            (32,),
            l_self_modules_encoder_modules_layers_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        hidden_states = l_self_modules_encoder_modules_layers_modules_0_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_0_modules_norm1_parameters_bias_ = (None)
        hidden_states_2 = torch._C._nn.linear(
            hidden_states_1,
            l_self_modules_encoder_modules_layers_modules_0_modules_linear1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_0_modules_linear1_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_0_modules_linear1_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_0_modules_linear1_parameters_bias_ = (None)
        hidden_states_3 = torch._C._nn.gelu(hidden_states_2)
        hidden_states_2 = None
        hidden_states_4 = torch.nn.functional.dropout(
            hidden_states_3, 0.0, False, False
        )
        hidden_states_3 = None
        hidden_states_5 = torch._C._nn.linear(
            hidden_states_4,
            l_self_modules_encoder_modules_layers_modules_0_modules_linear2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_0_modules_linear2_parameters_bias_,
        )
        hidden_states_4 = l_self_modules_encoder_modules_layers_modules_0_modules_linear2_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_0_modules_linear2_parameters_bias_ = (None)
        dropout_4 = torch.nn.functional.dropout(hidden_states_5, 0.1, False, False)
        hidden_states_5 = None
        hidden_states_6 = hidden_states_1 + dropout_4
        hidden_states_1 = dropout_4 = None
        hidden_states_7 = torch.nn.functional.layer_norm(
            hidden_states_6,
            (32,),
            l_self_modules_encoder_modules_layers_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        hidden_states_6 = l_self_modules_encoder_modules_layers_modules_0_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_0_modules_norm2_parameters_bias_ = (None)
        mixed_query_layer_1 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        linear_7 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        x_3 = linear_7.view((1, 15, 4, 8))
        linear_7 = None
        key_layer_1 = x_3.permute(0, 2, 1, 3)
        x_3 = None
        linear_8 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        x_4 = linear_8.view((1, 15, 4, 8))
        linear_8 = None
        value_layer_1 = x_4.permute(0, 2, 1, 3)
        x_4 = None
        x_5 = mixed_query_layer_1.view((1, 15, 4, 8))
        mixed_query_layer_1 = None
        query_layer_1 = x_5.permute(0, 2, 1, 3)
        x_5 = None
        transpose_1 = key_layer_1.transpose(-1, -2)
        key_layer_1 = None
        attention_scores_3 = torch.matmul(query_layer_1, transpose_1)
        query_layer_1 = transpose_1 = None
        attention_scores_4 = attention_scores_3 / 2.8284271247461903
        attention_scores_3 = None
        attention_scores_5 = attention_scores_4 + extended_attention_mask
        attention_scores_4 = None
        attention_probs_2 = torch.nn.functional.softmax(attention_scores_5, dim=-1)
        attention_scores_5 = None
        attention_probs_3 = torch.nn.functional.dropout(
            attention_probs_2, 0.1, False, False
        )
        attention_probs_2 = None
        context_layer_3 = torch.matmul(attention_probs_3, value_layer_1)
        attention_probs_3 = value_layer_1 = None
        permute_7 = context_layer_3.permute(0, 2, 1, 3)
        context_layer_3 = None
        context_layer_4 = permute_7.contiguous()
        permute_7 = None
        context_layer_5 = context_layer_4.view((1, 15, 32))
        context_layer_4 = None
        attention_output_1 = torch._C._nn.linear(
            context_layer_5,
            l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        context_layer_5 = l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        dropout_6 = torch.nn.functional.dropout(attention_output_1, 0.1, False, False)
        attention_output_1 = None
        hidden_states_8 = hidden_states_7 + dropout_6
        hidden_states_7 = dropout_6 = None
        hidden_states_9 = torch.nn.functional.layer_norm(
            hidden_states_8,
            (32,),
            l_self_modules_encoder_modules_layers_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        hidden_states_8 = l_self_modules_encoder_modules_layers_modules_1_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_1_modules_norm1_parameters_bias_ = (None)
        hidden_states_10 = torch._C._nn.linear(
            hidden_states_9,
            l_self_modules_encoder_modules_layers_modules_1_modules_linear1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_1_modules_linear1_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_1_modules_linear1_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_1_modules_linear1_parameters_bias_ = (None)
        hidden_states_11 = torch._C._nn.gelu(hidden_states_10)
        hidden_states_10 = None
        hidden_states_12 = torch.nn.functional.dropout(
            hidden_states_11, 0.0, False, False
        )
        hidden_states_11 = None
        hidden_states_13 = torch._C._nn.linear(
            hidden_states_12,
            l_self_modules_encoder_modules_layers_modules_1_modules_linear2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_1_modules_linear2_parameters_bias_,
        )
        hidden_states_12 = l_self_modules_encoder_modules_layers_modules_1_modules_linear2_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_1_modules_linear2_parameters_bias_ = (None)
        dropout_8 = torch.nn.functional.dropout(hidden_states_13, 0.1, False, False)
        hidden_states_13 = None
        hidden_states_14 = hidden_states_9 + dropout_8
        hidden_states_9 = dropout_8 = None
        hidden_states_15 = torch.nn.functional.layer_norm(
            hidden_states_14,
            (32,),
            l_self_modules_encoder_modules_layers_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        hidden_states_14 = l_self_modules_encoder_modules_layers_modules_1_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_1_modules_norm2_parameters_bias_ = (None)
        mixed_query_layer_2 = torch._C._nn.linear(
            hidden_states_15,
            l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        linear_13 = torch._C._nn.linear(
            hidden_states_15,
            l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        x_6 = linear_13.view((1, 15, 4, 8))
        linear_13 = None
        key_layer_2 = x_6.permute(0, 2, 1, 3)
        x_6 = None
        linear_14 = torch._C._nn.linear(
            hidden_states_15,
            l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        x_7 = linear_14.view((1, 15, 4, 8))
        linear_14 = None
        value_layer_2 = x_7.permute(0, 2, 1, 3)
        x_7 = None
        x_8 = mixed_query_layer_2.view((1, 15, 4, 8))
        mixed_query_layer_2 = None
        query_layer_2 = x_8.permute(0, 2, 1, 3)
        x_8 = None
        transpose_2 = key_layer_2.transpose(-1, -2)
        key_layer_2 = None
        attention_scores_6 = torch.matmul(query_layer_2, transpose_2)
        query_layer_2 = transpose_2 = None
        attention_scores_7 = attention_scores_6 / 2.8284271247461903
        attention_scores_6 = None
        attention_scores_8 = attention_scores_7 + extended_attention_mask
        attention_scores_7 = None
        attention_probs_4 = torch.nn.functional.softmax(attention_scores_8, dim=-1)
        attention_scores_8 = None
        attention_probs_5 = torch.nn.functional.dropout(
            attention_probs_4, 0.1, False, False
        )
        attention_probs_4 = None
        context_layer_6 = torch.matmul(attention_probs_5, value_layer_2)
        attention_probs_5 = value_layer_2 = None
        permute_11 = context_layer_6.permute(0, 2, 1, 3)
        context_layer_6 = None
        context_layer_7 = permute_11.contiguous()
        permute_11 = None
        context_layer_8 = context_layer_7.view((1, 15, 32))
        context_layer_7 = None
        attention_output_2 = torch._C._nn.linear(
            context_layer_8,
            l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        context_layer_8 = l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        dropout_10 = torch.nn.functional.dropout(attention_output_2, 0.1, False, False)
        attention_output_2 = None
        hidden_states_16 = hidden_states_15 + dropout_10
        hidden_states_15 = dropout_10 = None
        hidden_states_17 = torch.nn.functional.layer_norm(
            hidden_states_16,
            (32,),
            l_self_modules_encoder_modules_layers_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_2_modules_norm1_parameters_bias_,
            1e-05,
        )
        hidden_states_16 = l_self_modules_encoder_modules_layers_modules_2_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_2_modules_norm1_parameters_bias_ = (None)
        hidden_states_18 = torch._C._nn.linear(
            hidden_states_17,
            l_self_modules_encoder_modules_layers_modules_2_modules_linear1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_2_modules_linear1_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_2_modules_linear1_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_2_modules_linear1_parameters_bias_ = (None)
        hidden_states_19 = torch._C._nn.gelu(hidden_states_18)
        hidden_states_18 = None
        hidden_states_20 = torch.nn.functional.dropout(
            hidden_states_19, 0.0, False, False
        )
        hidden_states_19 = None
        hidden_states_21 = torch._C._nn.linear(
            hidden_states_20,
            l_self_modules_encoder_modules_layers_modules_2_modules_linear2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_2_modules_linear2_parameters_bias_,
        )
        hidden_states_20 = l_self_modules_encoder_modules_layers_modules_2_modules_linear2_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_2_modules_linear2_parameters_bias_ = (None)
        dropout_12 = torch.nn.functional.dropout(hidden_states_21, 0.1, False, False)
        hidden_states_21 = None
        hidden_states_22 = hidden_states_17 + dropout_12
        hidden_states_17 = dropout_12 = None
        hidden_states_23 = torch.nn.functional.layer_norm(
            hidden_states_22,
            (32,),
            l_self_modules_encoder_modules_layers_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_2_modules_norm2_parameters_bias_,
            1e-05,
        )
        hidden_states_22 = l_self_modules_encoder_modules_layers_modules_2_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_2_modules_norm2_parameters_bias_ = (None)
        mixed_query_layer_3 = torch._C._nn.linear(
            hidden_states_23,
            l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        linear_19 = torch._C._nn.linear(
            hidden_states_23,
            l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        x_9 = linear_19.view((1, 15, 4, 8))
        linear_19 = None
        key_layer_3 = x_9.permute(0, 2, 1, 3)
        x_9 = None
        linear_20 = torch._C._nn.linear(
            hidden_states_23,
            l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        x_10 = linear_20.view((1, 15, 4, 8))
        linear_20 = None
        value_layer_3 = x_10.permute(0, 2, 1, 3)
        x_10 = None
        x_11 = mixed_query_layer_3.view((1, 15, 4, 8))
        mixed_query_layer_3 = None
        query_layer_3 = x_11.permute(0, 2, 1, 3)
        x_11 = None
        transpose_3 = key_layer_3.transpose(-1, -2)
        key_layer_3 = None
        attention_scores_9 = torch.matmul(query_layer_3, transpose_3)
        query_layer_3 = transpose_3 = None
        attention_scores_10 = attention_scores_9 / 2.8284271247461903
        attention_scores_9 = None
        attention_scores_11 = attention_scores_10 + extended_attention_mask
        attention_scores_10 = None
        attention_probs_6 = torch.nn.functional.softmax(attention_scores_11, dim=-1)
        attention_scores_11 = None
        attention_probs_7 = torch.nn.functional.dropout(
            attention_probs_6, 0.1, False, False
        )
        attention_probs_6 = None
        context_layer_9 = torch.matmul(attention_probs_7, value_layer_3)
        attention_probs_7 = value_layer_3 = None
        permute_15 = context_layer_9.permute(0, 2, 1, 3)
        context_layer_9 = None
        context_layer_10 = permute_15.contiguous()
        permute_15 = None
        context_layer_11 = context_layer_10.view((1, 15, 32))
        context_layer_10 = None
        attention_output_3 = torch._C._nn.linear(
            context_layer_11,
            l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        context_layer_11 = l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        dropout_14 = torch.nn.functional.dropout(attention_output_3, 0.1, False, False)
        attention_output_3 = None
        hidden_states_24 = hidden_states_23 + dropout_14
        hidden_states_23 = dropout_14 = None
        hidden_states_25 = torch.nn.functional.layer_norm(
            hidden_states_24,
            (32,),
            l_self_modules_encoder_modules_layers_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_3_modules_norm1_parameters_bias_,
            1e-05,
        )
        hidden_states_24 = l_self_modules_encoder_modules_layers_modules_3_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_3_modules_norm1_parameters_bias_ = (None)
        hidden_states_26 = torch._C._nn.linear(
            hidden_states_25,
            l_self_modules_encoder_modules_layers_modules_3_modules_linear1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_3_modules_linear1_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_3_modules_linear1_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_3_modules_linear1_parameters_bias_ = (None)
        hidden_states_27 = torch._C._nn.gelu(hidden_states_26)
        hidden_states_26 = None
        hidden_states_28 = torch.nn.functional.dropout(
            hidden_states_27, 0.0, False, False
        )
        hidden_states_27 = None
        hidden_states_29 = torch._C._nn.linear(
            hidden_states_28,
            l_self_modules_encoder_modules_layers_modules_3_modules_linear2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_3_modules_linear2_parameters_bias_,
        )
        hidden_states_28 = l_self_modules_encoder_modules_layers_modules_3_modules_linear2_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_3_modules_linear2_parameters_bias_ = (None)
        dropout_16 = torch.nn.functional.dropout(hidden_states_29, 0.1, False, False)
        hidden_states_29 = None
        hidden_states_30 = hidden_states_25 + dropout_16
        hidden_states_25 = dropout_16 = None
        hidden_states_31 = torch.nn.functional.layer_norm(
            hidden_states_30,
            (32,),
            l_self_modules_encoder_modules_layers_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_3_modules_norm2_parameters_bias_,
            1e-05,
        )
        hidden_states_30 = l_self_modules_encoder_modules_layers_modules_3_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_3_modules_norm2_parameters_bias_ = (None)
        mixed_query_layer_4 = torch._C._nn.linear(
            hidden_states_31,
            l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        linear_25 = torch._C._nn.linear(
            hidden_states_31,
            l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        x_12 = linear_25.view((1, 15, 4, 8))
        linear_25 = None
        key_layer_4 = x_12.permute(0, 2, 1, 3)
        x_12 = None
        linear_26 = torch._C._nn.linear(
            hidden_states_31,
            l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        x_13 = linear_26.view((1, 15, 4, 8))
        linear_26 = None
        value_layer_4 = x_13.permute(0, 2, 1, 3)
        x_13 = None
        x_14 = mixed_query_layer_4.view((1, 15, 4, 8))
        mixed_query_layer_4 = None
        query_layer_4 = x_14.permute(0, 2, 1, 3)
        x_14 = None
        transpose_4 = key_layer_4.transpose(-1, -2)
        key_layer_4 = None
        attention_scores_12 = torch.matmul(query_layer_4, transpose_4)
        query_layer_4 = transpose_4 = None
        attention_scores_13 = attention_scores_12 / 2.8284271247461903
        attention_scores_12 = None
        attention_scores_14 = attention_scores_13 + extended_attention_mask
        attention_scores_13 = extended_attention_mask = None
        attention_probs_8 = torch.nn.functional.softmax(attention_scores_14, dim=-1)
        attention_scores_14 = None
        attention_probs_9 = torch.nn.functional.dropout(
            attention_probs_8, 0.1, False, False
        )
        attention_probs_8 = None
        context_layer_12 = torch.matmul(attention_probs_9, value_layer_4)
        attention_probs_9 = value_layer_4 = None
        permute_19 = context_layer_12.permute(0, 2, 1, 3)
        context_layer_12 = None
        context_layer_13 = permute_19.contiguous()
        permute_19 = None
        context_layer_14 = context_layer_13.view((1, 15, 32))
        context_layer_13 = None
        attention_output_4 = torch._C._nn.linear(
            context_layer_14,
            l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        context_layer_14 = l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        dropout_18 = torch.nn.functional.dropout(attention_output_4, 0.1, False, False)
        attention_output_4 = None
        hidden_states_32 = hidden_states_31 + dropout_18
        hidden_states_31 = dropout_18 = None
        hidden_states_33 = torch.nn.functional.layer_norm(
            hidden_states_32,
            (32,),
            l_self_modules_encoder_modules_layers_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_4_modules_norm1_parameters_bias_,
            1e-05,
        )
        hidden_states_32 = l_self_modules_encoder_modules_layers_modules_4_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_4_modules_norm1_parameters_bias_ = (None)
        hidden_states_34 = torch._C._nn.linear(
            hidden_states_33,
            l_self_modules_encoder_modules_layers_modules_4_modules_linear1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_4_modules_linear1_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_4_modules_linear1_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_4_modules_linear1_parameters_bias_ = (None)
        hidden_states_35 = torch._C._nn.gelu(hidden_states_34)
        hidden_states_34 = None
        hidden_states_36 = torch.nn.functional.dropout(
            hidden_states_35, 0.0, False, False
        )
        hidden_states_35 = None
        hidden_states_37 = torch._C._nn.linear(
            hidden_states_36,
            l_self_modules_encoder_modules_layers_modules_4_modules_linear2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_4_modules_linear2_parameters_bias_,
        )
        hidden_states_36 = l_self_modules_encoder_modules_layers_modules_4_modules_linear2_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_4_modules_linear2_parameters_bias_ = (None)
        dropout_20 = torch.nn.functional.dropout(hidden_states_37, 0.1, False, False)
        hidden_states_37 = None
        hidden_states_38 = hidden_states_33 + dropout_20
        hidden_states_33 = dropout_20 = None
        hidden_states_39 = torch.nn.functional.layer_norm(
            hidden_states_38,
            (32,),
            l_self_modules_encoder_modules_layers_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_4_modules_norm2_parameters_bias_,
            1e-05,
        )
        hidden_states_38 = l_self_modules_encoder_modules_layers_modules_4_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_4_modules_norm2_parameters_bias_ = (None)
        first_token_tensor = hidden_states_39[(slice(None, None, None), 0)]
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
        return (hidden_states_39, pooled_output_1)
