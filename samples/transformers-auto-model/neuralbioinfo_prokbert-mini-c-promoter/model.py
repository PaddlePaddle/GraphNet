import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_input_ids_: torch.Tensor,
        L_attention_mask_: torch.Tensor,
        L_token_type_ids_: torch.Tensor,
        L_self_modules_embeddings_modules_word_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_token_type_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_ln_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_ln_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_distance_embedding_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_ln_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_ln_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_ln_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_ln_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_distance_embedding_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_ln_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_ln_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_ln_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_ln_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_distance_embedding_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_ln_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_ln_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_ln_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_ln_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_distance_embedding_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_ln_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_ln_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_ln_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_ln_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_distance_embedding_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_ln_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_ln_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_ln_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_ln_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_distance_embedding_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_ln_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_ln_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_ln_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_ln_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_pooler_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_pooler_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_input_ids_ = L_input_ids_
        l_attention_mask_ = L_attention_mask_
        l_token_type_ids_ = L_token_type_ids_
        l_self_modules_embeddings_modules_word_embeddings_parameters_weight_ = (
            L_self_modules_embeddings_modules_word_embeddings_parameters_weight_
        )
        l_self_modules_embeddings_modules_token_type_embeddings_parameters_weight_ = (
            L_self_modules_embeddings_modules_token_type_embeddings_parameters_weight_
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_ln_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_ln_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_ln_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_ln_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_distance_embedding_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_distance_embedding_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_ln_parameters_weight_ = (
            L_self_modules_encoder_modules_layer_modules_0_modules_ln_parameters_weight_
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_ln_parameters_bias_ = (
            L_self_modules_encoder_modules_layer_modules_0_modules_ln_parameters_bias_
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_ln_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_ln_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_ln_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_ln_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_distance_embedding_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_distance_embedding_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_ln_parameters_weight_ = (
            L_self_modules_encoder_modules_layer_modules_1_modules_ln_parameters_weight_
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_ln_parameters_bias_ = (
            L_self_modules_encoder_modules_layer_modules_1_modules_ln_parameters_bias_
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_ln_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_ln_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_ln_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_ln_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_distance_embedding_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_distance_embedding_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_ln_parameters_weight_ = (
            L_self_modules_encoder_modules_layer_modules_2_modules_ln_parameters_weight_
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_ln_parameters_bias_ = (
            L_self_modules_encoder_modules_layer_modules_2_modules_ln_parameters_bias_
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_ln_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_ln_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_ln_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_ln_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_distance_embedding_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_distance_embedding_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_ln_parameters_weight_ = (
            L_self_modules_encoder_modules_layer_modules_3_modules_ln_parameters_weight_
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_ln_parameters_bias_ = (
            L_self_modules_encoder_modules_layer_modules_3_modules_ln_parameters_bias_
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_ln_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_ln_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_ln_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_ln_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_distance_embedding_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_distance_embedding_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_ln_parameters_weight_ = (
            L_self_modules_encoder_modules_layer_modules_4_modules_ln_parameters_weight_
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_ln_parameters_bias_ = (
            L_self_modules_encoder_modules_layer_modules_4_modules_ln_parameters_bias_
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_ln_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_ln_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_ln_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_ln_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_distance_embedding_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_distance_embedding_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_ln_parameters_weight_ = (
            L_self_modules_encoder_modules_layer_modules_5_modules_ln_parameters_weight_
        )
        l_self_modules_encoder_modules_layer_modules_5_modules_ln_parameters_bias_ = (
            L_self_modules_encoder_modules_layer_modules_5_modules_ln_parameters_bias_
        )
        l_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_ln_parameters_weight_ = (
            L_self_modules_encoder_modules_ln_parameters_weight_
        )
        l_self_modules_encoder_modules_ln_parameters_bias_ = (
            L_self_modules_encoder_modules_ln_parameters_bias_
        )
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
        inputs_embeds = torch.nn.functional.embedding(
            l_input_ids_,
            l_self_modules_embeddings_modules_word_embeddings_parameters_weight_,
            0,
            None,
            2.0,
            False,
            False,
        )
        l_input_ids_ = (
            l_self_modules_embeddings_modules_word_embeddings_parameters_weight_
        ) = None
        token_type_embeddings = torch.nn.functional.embedding(
            l_token_type_ids_,
            l_self_modules_embeddings_modules_token_type_embeddings_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        l_token_type_ids_ = (
            l_self_modules_embeddings_modules_token_type_embeddings_parameters_weight_
        ) = None
        embeddings = inputs_embeds + token_type_embeddings
        inputs_embeds = token_type_embeddings = None
        embeddings_1 = torch.nn.functional.dropout(embeddings, 0.1, False, False)
        embeddings = None
        ln_outputs = torch.nn.functional.layer_norm(
            embeddings_1,
            (384,),
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_ln_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_ln_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_ln_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_ln_parameters_bias_ = (None)
        query_layer = torch._C._nn.linear(
            ln_outputs,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view = query_layer.view(1, -1, 6, 64)
        query_layer = None
        query_layer_1 = view.transpose(1, 2)
        view = None
        key_layer = torch._C._nn.linear(
            ln_outputs,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_1 = key_layer.view(1, -1, 6, 64)
        key_layer = None
        key_layer_1 = view_1.transpose(1, 2)
        view_1 = None
        value_layer = torch._C._nn.linear(
            ln_outputs,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        ln_outputs = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_2 = value_layer.view(1, -1, 6, 64)
        value_layer = None
        value_layer_1 = view_2.transpose(1, 2)
        view_2 = None
        transpose_3 = key_layer_1.transpose(-1, -2)
        attention_scores = torch.matmul(query_layer_1, transpose_3)
        transpose_3 = None
        arange = torch.arange(
            51, dtype=torch.int64, device=device(type="cuda", index=0)
        )
        position_ids_l = arange.view(-1, 1)
        arange = None
        arange_1 = torch.arange(
            51, dtype=torch.int64, device=device(type="cuda", index=0)
        )
        position_ids_r = arange_1.view(1, -1)
        arange_1 = None
        distance = position_ids_l - position_ids_r
        position_ids_l = position_ids_r = None
        add_1 = distance + 2048
        distance = None
        sub_2 = add_1 - 1
        add_1 = None
        positional_embedding = torch.nn.functional.embedding(
            sub_2,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_distance_embedding_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        sub_2 = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_distance_embedding_parameters_weight_ = (None)
        positional_embedding_1 = positional_embedding.to(dtype=torch.float32)
        positional_embedding = None
        relative_position_scores_query = torch.functional.einsum(
            "bhld,lrd->bhlr", query_layer_1, positional_embedding_1
        )
        query_layer_1 = None
        relative_position_scores_key = torch.functional.einsum(
            "bhrd,lrd->bhlr", key_layer_1, positional_embedding_1
        )
        key_layer_1 = positional_embedding_1 = None
        add_2 = attention_scores + relative_position_scores_query
        attention_scores = relative_position_scores_query = None
        attention_scores_1 = add_2 + relative_position_scores_key
        add_2 = relative_position_scores_key = None
        attention_scores_2 = attention_scores_1 / 8.0
        attention_scores_1 = None
        attention_scores_3 = attention_scores_2 + extended_attention_mask_2
        attention_scores_2 = None
        attention_probs = torch.nn.functional.softmax(attention_scores_3, dim=-1)
        attention_scores_3 = None
        attention_probs_1 = torch.nn.functional.dropout(
            attention_probs, 0.1, False, False
        )
        attention_probs = None
        context_layer = torch.matmul(attention_probs_1, value_layer_1)
        attention_probs_1 = value_layer_1 = None
        permute = context_layer.permute(0, 2, 1, 3)
        context_layer = None
        context_layer_1 = permute.contiguous()
        permute = None
        context_layer_2 = context_layer_1.view((1, 51, 384))
        context_layer_1 = None
        hidden_states = torch._C._nn.linear(
            context_layer_2,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_2 = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_1 = torch.nn.functional.dropout(hidden_states, 0.1, False, False)
        hidden_states = None
        attention_output = embeddings_1 + hidden_states_1
        embeddings_1 = hidden_states_1 = None
        ln_output = torch.nn.functional.layer_norm(
            attention_output,
            (384,),
            l_self_modules_encoder_modules_layer_modules_0_modules_ln_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_ln_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_ln_parameters_weight_ = (
            l_self_modules_encoder_modules_layer_modules_0_modules_ln_parameters_bias_
        ) = None
        hidden_states_2 = torch._C._nn.linear(
            ln_output,
            l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_,
        )
        ln_output = l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_3 = torch._C._nn.gelu(hidden_states_2)
        hidden_states_2 = None
        hidden_states_4 = torch._C._nn.linear(
            hidden_states_3,
            l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_3 = l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_5 = torch.nn.functional.dropout(
            hidden_states_4, 0.1, False, False
        )
        hidden_states_4 = None
        layer_output = attention_output + hidden_states_5
        attention_output = hidden_states_5 = None
        ln_outputs_1 = torch.nn.functional.layer_norm(
            layer_output,
            (384,),
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_ln_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_ln_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_ln_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_ln_parameters_bias_ = (None)
        query_layer_2 = torch._C._nn.linear(
            ln_outputs_1,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_6 = query_layer_2.view(1, -1, 6, 64)
        query_layer_2 = None
        query_layer_3 = view_6.transpose(1, 2)
        view_6 = None
        key_layer_2 = torch._C._nn.linear(
            ln_outputs_1,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_7 = key_layer_2.view(1, -1, 6, 64)
        key_layer_2 = None
        key_layer_3 = view_7.transpose(1, 2)
        view_7 = None
        value_layer_2 = torch._C._nn.linear(
            ln_outputs_1,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        ln_outputs_1 = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_8 = value_layer_2.view(1, -1, 6, 64)
        value_layer_2 = None
        value_layer_3 = view_8.transpose(1, 2)
        view_8 = None
        transpose_7 = key_layer_3.transpose(-1, -2)
        attention_scores_4 = torch.matmul(query_layer_3, transpose_7)
        transpose_7 = None
        arange_2 = torch.arange(
            51, dtype=torch.int64, device=device(type="cuda", index=0)
        )
        position_ids_l_1 = arange_2.view(-1, 1)
        arange_2 = None
        arange_3 = torch.arange(
            51, dtype=torch.int64, device=device(type="cuda", index=0)
        )
        position_ids_r_1 = arange_3.view(1, -1)
        arange_3 = None
        distance_1 = position_ids_l_1 - position_ids_r_1
        position_ids_l_1 = position_ids_r_1 = None
        add_7 = distance_1 + 2048
        distance_1 = None
        sub_4 = add_7 - 1
        add_7 = None
        positional_embedding_2 = torch.nn.functional.embedding(
            sub_4,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_distance_embedding_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        sub_4 = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_distance_embedding_parameters_weight_ = (None)
        positional_embedding_3 = positional_embedding_2.to(dtype=torch.float32)
        positional_embedding_2 = None
        relative_position_scores_query_1 = torch.functional.einsum(
            "bhld,lrd->bhlr", query_layer_3, positional_embedding_3
        )
        query_layer_3 = None
        relative_position_scores_key_1 = torch.functional.einsum(
            "bhrd,lrd->bhlr", key_layer_3, positional_embedding_3
        )
        key_layer_3 = positional_embedding_3 = None
        add_8 = attention_scores_4 + relative_position_scores_query_1
        attention_scores_4 = relative_position_scores_query_1 = None
        attention_scores_5 = add_8 + relative_position_scores_key_1
        add_8 = relative_position_scores_key_1 = None
        attention_scores_6 = attention_scores_5 / 8.0
        attention_scores_5 = None
        attention_scores_7 = attention_scores_6 + extended_attention_mask_2
        attention_scores_6 = None
        attention_probs_2 = torch.nn.functional.softmax(attention_scores_7, dim=-1)
        attention_scores_7 = None
        attention_probs_3 = torch.nn.functional.dropout(
            attention_probs_2, 0.1, False, False
        )
        attention_probs_2 = None
        context_layer_3 = torch.matmul(attention_probs_3, value_layer_3)
        attention_probs_3 = value_layer_3 = None
        permute_1 = context_layer_3.permute(0, 2, 1, 3)
        context_layer_3 = None
        context_layer_4 = permute_1.contiguous()
        permute_1 = None
        context_layer_5 = context_layer_4.view((1, 51, 384))
        context_layer_4 = None
        hidden_states_6 = torch._C._nn.linear(
            context_layer_5,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_5 = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_7 = torch.nn.functional.dropout(
            hidden_states_6, 0.1, False, False
        )
        hidden_states_6 = None
        attention_output_1 = layer_output + hidden_states_7
        layer_output = hidden_states_7 = None
        ln_output_1 = torch.nn.functional.layer_norm(
            attention_output_1,
            (384,),
            l_self_modules_encoder_modules_layer_modules_1_modules_ln_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_ln_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_ln_parameters_weight_ = (
            l_self_modules_encoder_modules_layer_modules_1_modules_ln_parameters_bias_
        ) = None
        hidden_states_8 = torch._C._nn.linear(
            ln_output_1,
            l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_,
        )
        ln_output_1 = l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_9 = torch._C._nn.gelu(hidden_states_8)
        hidden_states_8 = None
        hidden_states_10 = torch._C._nn.linear(
            hidden_states_9,
            l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_9 = l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_11 = torch.nn.functional.dropout(
            hidden_states_10, 0.1, False, False
        )
        hidden_states_10 = None
        layer_output_1 = attention_output_1 + hidden_states_11
        attention_output_1 = hidden_states_11 = None
        ln_outputs_2 = torch.nn.functional.layer_norm(
            layer_output_1,
            (384,),
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_ln_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_ln_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_ln_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_ln_parameters_bias_ = (None)
        query_layer_4 = torch._C._nn.linear(
            ln_outputs_2,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_12 = query_layer_4.view(1, -1, 6, 64)
        query_layer_4 = None
        query_layer_5 = view_12.transpose(1, 2)
        view_12 = None
        key_layer_4 = torch._C._nn.linear(
            ln_outputs_2,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_13 = key_layer_4.view(1, -1, 6, 64)
        key_layer_4 = None
        key_layer_5 = view_13.transpose(1, 2)
        view_13 = None
        value_layer_4 = torch._C._nn.linear(
            ln_outputs_2,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        ln_outputs_2 = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_14 = value_layer_4.view(1, -1, 6, 64)
        value_layer_4 = None
        value_layer_5 = view_14.transpose(1, 2)
        view_14 = None
        transpose_11 = key_layer_5.transpose(-1, -2)
        attention_scores_8 = torch.matmul(query_layer_5, transpose_11)
        transpose_11 = None
        arange_4 = torch.arange(
            51, dtype=torch.int64, device=device(type="cuda", index=0)
        )
        position_ids_l_2 = arange_4.view(-1, 1)
        arange_4 = None
        arange_5 = torch.arange(
            51, dtype=torch.int64, device=device(type="cuda", index=0)
        )
        position_ids_r_2 = arange_5.view(1, -1)
        arange_5 = None
        distance_2 = position_ids_l_2 - position_ids_r_2
        position_ids_l_2 = position_ids_r_2 = None
        add_13 = distance_2 + 2048
        distance_2 = None
        sub_6 = add_13 - 1
        add_13 = None
        positional_embedding_4 = torch.nn.functional.embedding(
            sub_6,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_distance_embedding_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        sub_6 = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_distance_embedding_parameters_weight_ = (None)
        positional_embedding_5 = positional_embedding_4.to(dtype=torch.float32)
        positional_embedding_4 = None
        relative_position_scores_query_2 = torch.functional.einsum(
            "bhld,lrd->bhlr", query_layer_5, positional_embedding_5
        )
        query_layer_5 = None
        relative_position_scores_key_2 = torch.functional.einsum(
            "bhrd,lrd->bhlr", key_layer_5, positional_embedding_5
        )
        key_layer_5 = positional_embedding_5 = None
        add_14 = attention_scores_8 + relative_position_scores_query_2
        attention_scores_8 = relative_position_scores_query_2 = None
        attention_scores_9 = add_14 + relative_position_scores_key_2
        add_14 = relative_position_scores_key_2 = None
        attention_scores_10 = attention_scores_9 / 8.0
        attention_scores_9 = None
        attention_scores_11 = attention_scores_10 + extended_attention_mask_2
        attention_scores_10 = None
        attention_probs_4 = torch.nn.functional.softmax(attention_scores_11, dim=-1)
        attention_scores_11 = None
        attention_probs_5 = torch.nn.functional.dropout(
            attention_probs_4, 0.1, False, False
        )
        attention_probs_4 = None
        context_layer_6 = torch.matmul(attention_probs_5, value_layer_5)
        attention_probs_5 = value_layer_5 = None
        permute_2 = context_layer_6.permute(0, 2, 1, 3)
        context_layer_6 = None
        context_layer_7 = permute_2.contiguous()
        permute_2 = None
        context_layer_8 = context_layer_7.view((1, 51, 384))
        context_layer_7 = None
        hidden_states_12 = torch._C._nn.linear(
            context_layer_8,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_8 = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_13 = torch.nn.functional.dropout(
            hidden_states_12, 0.1, False, False
        )
        hidden_states_12 = None
        attention_output_2 = layer_output_1 + hidden_states_13
        layer_output_1 = hidden_states_13 = None
        ln_output_2 = torch.nn.functional.layer_norm(
            attention_output_2,
            (384,),
            l_self_modules_encoder_modules_layer_modules_2_modules_ln_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_ln_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_ln_parameters_weight_ = (
            l_self_modules_encoder_modules_layer_modules_2_modules_ln_parameters_bias_
        ) = None
        hidden_states_14 = torch._C._nn.linear(
            ln_output_2,
            l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_,
        )
        ln_output_2 = l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_15 = torch._C._nn.gelu(hidden_states_14)
        hidden_states_14 = None
        hidden_states_16 = torch._C._nn.linear(
            hidden_states_15,
            l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_15 = l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_17 = torch.nn.functional.dropout(
            hidden_states_16, 0.1, False, False
        )
        hidden_states_16 = None
        layer_output_2 = attention_output_2 + hidden_states_17
        attention_output_2 = hidden_states_17 = None
        ln_outputs_3 = torch.nn.functional.layer_norm(
            layer_output_2,
            (384,),
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_ln_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_ln_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_ln_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_ln_parameters_bias_ = (None)
        query_layer_6 = torch._C._nn.linear(
            ln_outputs_3,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_18 = query_layer_6.view(1, -1, 6, 64)
        query_layer_6 = None
        query_layer_7 = view_18.transpose(1, 2)
        view_18 = None
        key_layer_6 = torch._C._nn.linear(
            ln_outputs_3,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_19 = key_layer_6.view(1, -1, 6, 64)
        key_layer_6 = None
        key_layer_7 = view_19.transpose(1, 2)
        view_19 = None
        value_layer_6 = torch._C._nn.linear(
            ln_outputs_3,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        ln_outputs_3 = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_20 = value_layer_6.view(1, -1, 6, 64)
        value_layer_6 = None
        value_layer_7 = view_20.transpose(1, 2)
        view_20 = None
        transpose_15 = key_layer_7.transpose(-1, -2)
        attention_scores_12 = torch.matmul(query_layer_7, transpose_15)
        transpose_15 = None
        arange_6 = torch.arange(
            51, dtype=torch.int64, device=device(type="cuda", index=0)
        )
        position_ids_l_3 = arange_6.view(-1, 1)
        arange_6 = None
        arange_7 = torch.arange(
            51, dtype=torch.int64, device=device(type="cuda", index=0)
        )
        position_ids_r_3 = arange_7.view(1, -1)
        arange_7 = None
        distance_3 = position_ids_l_3 - position_ids_r_3
        position_ids_l_3 = position_ids_r_3 = None
        add_19 = distance_3 + 2048
        distance_3 = None
        sub_8 = add_19 - 1
        add_19 = None
        positional_embedding_6 = torch.nn.functional.embedding(
            sub_8,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_distance_embedding_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        sub_8 = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_distance_embedding_parameters_weight_ = (None)
        positional_embedding_7 = positional_embedding_6.to(dtype=torch.float32)
        positional_embedding_6 = None
        relative_position_scores_query_3 = torch.functional.einsum(
            "bhld,lrd->bhlr", query_layer_7, positional_embedding_7
        )
        query_layer_7 = None
        relative_position_scores_key_3 = torch.functional.einsum(
            "bhrd,lrd->bhlr", key_layer_7, positional_embedding_7
        )
        key_layer_7 = positional_embedding_7 = None
        add_20 = attention_scores_12 + relative_position_scores_query_3
        attention_scores_12 = relative_position_scores_query_3 = None
        attention_scores_13 = add_20 + relative_position_scores_key_3
        add_20 = relative_position_scores_key_3 = None
        attention_scores_14 = attention_scores_13 / 8.0
        attention_scores_13 = None
        attention_scores_15 = attention_scores_14 + extended_attention_mask_2
        attention_scores_14 = None
        attention_probs_6 = torch.nn.functional.softmax(attention_scores_15, dim=-1)
        attention_scores_15 = None
        attention_probs_7 = torch.nn.functional.dropout(
            attention_probs_6, 0.1, False, False
        )
        attention_probs_6 = None
        context_layer_9 = torch.matmul(attention_probs_7, value_layer_7)
        attention_probs_7 = value_layer_7 = None
        permute_3 = context_layer_9.permute(0, 2, 1, 3)
        context_layer_9 = None
        context_layer_10 = permute_3.contiguous()
        permute_3 = None
        context_layer_11 = context_layer_10.view((1, 51, 384))
        context_layer_10 = None
        hidden_states_18 = torch._C._nn.linear(
            context_layer_11,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_11 = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_19 = torch.nn.functional.dropout(
            hidden_states_18, 0.1, False, False
        )
        hidden_states_18 = None
        attention_output_3 = layer_output_2 + hidden_states_19
        layer_output_2 = hidden_states_19 = None
        ln_output_3 = torch.nn.functional.layer_norm(
            attention_output_3,
            (384,),
            l_self_modules_encoder_modules_layer_modules_3_modules_ln_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_ln_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_ln_parameters_weight_ = (
            l_self_modules_encoder_modules_layer_modules_3_modules_ln_parameters_bias_
        ) = None
        hidden_states_20 = torch._C._nn.linear(
            ln_output_3,
            l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_,
        )
        ln_output_3 = l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_21 = torch._C._nn.gelu(hidden_states_20)
        hidden_states_20 = None
        hidden_states_22 = torch._C._nn.linear(
            hidden_states_21,
            l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_21 = l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_23 = torch.nn.functional.dropout(
            hidden_states_22, 0.1, False, False
        )
        hidden_states_22 = None
        layer_output_3 = attention_output_3 + hidden_states_23
        attention_output_3 = hidden_states_23 = None
        ln_outputs_4 = torch.nn.functional.layer_norm(
            layer_output_3,
            (384,),
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_ln_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_ln_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_ln_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_ln_parameters_bias_ = (None)
        query_layer_8 = torch._C._nn.linear(
            ln_outputs_4,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_24 = query_layer_8.view(1, -1, 6, 64)
        query_layer_8 = None
        query_layer_9 = view_24.transpose(1, 2)
        view_24 = None
        key_layer_8 = torch._C._nn.linear(
            ln_outputs_4,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_25 = key_layer_8.view(1, -1, 6, 64)
        key_layer_8 = None
        key_layer_9 = view_25.transpose(1, 2)
        view_25 = None
        value_layer_8 = torch._C._nn.linear(
            ln_outputs_4,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        ln_outputs_4 = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_26 = value_layer_8.view(1, -1, 6, 64)
        value_layer_8 = None
        value_layer_9 = view_26.transpose(1, 2)
        view_26 = None
        transpose_19 = key_layer_9.transpose(-1, -2)
        attention_scores_16 = torch.matmul(query_layer_9, transpose_19)
        transpose_19 = None
        arange_8 = torch.arange(
            51, dtype=torch.int64, device=device(type="cuda", index=0)
        )
        position_ids_l_4 = arange_8.view(-1, 1)
        arange_8 = None
        arange_9 = torch.arange(
            51, dtype=torch.int64, device=device(type="cuda", index=0)
        )
        position_ids_r_4 = arange_9.view(1, -1)
        arange_9 = None
        distance_4 = position_ids_l_4 - position_ids_r_4
        position_ids_l_4 = position_ids_r_4 = None
        add_25 = distance_4 + 2048
        distance_4 = None
        sub_10 = add_25 - 1
        add_25 = None
        positional_embedding_8 = torch.nn.functional.embedding(
            sub_10,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_distance_embedding_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        sub_10 = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_distance_embedding_parameters_weight_ = (None)
        positional_embedding_9 = positional_embedding_8.to(dtype=torch.float32)
        positional_embedding_8 = None
        relative_position_scores_query_4 = torch.functional.einsum(
            "bhld,lrd->bhlr", query_layer_9, positional_embedding_9
        )
        query_layer_9 = None
        relative_position_scores_key_4 = torch.functional.einsum(
            "bhrd,lrd->bhlr", key_layer_9, positional_embedding_9
        )
        key_layer_9 = positional_embedding_9 = None
        add_26 = attention_scores_16 + relative_position_scores_query_4
        attention_scores_16 = relative_position_scores_query_4 = None
        attention_scores_17 = add_26 + relative_position_scores_key_4
        add_26 = relative_position_scores_key_4 = None
        attention_scores_18 = attention_scores_17 / 8.0
        attention_scores_17 = None
        attention_scores_19 = attention_scores_18 + extended_attention_mask_2
        attention_scores_18 = None
        attention_probs_8 = torch.nn.functional.softmax(attention_scores_19, dim=-1)
        attention_scores_19 = None
        attention_probs_9 = torch.nn.functional.dropout(
            attention_probs_8, 0.1, False, False
        )
        attention_probs_8 = None
        context_layer_12 = torch.matmul(attention_probs_9, value_layer_9)
        attention_probs_9 = value_layer_9 = None
        permute_4 = context_layer_12.permute(0, 2, 1, 3)
        context_layer_12 = None
        context_layer_13 = permute_4.contiguous()
        permute_4 = None
        context_layer_14 = context_layer_13.view((1, 51, 384))
        context_layer_13 = None
        hidden_states_24 = torch._C._nn.linear(
            context_layer_14,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_14 = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_25 = torch.nn.functional.dropout(
            hidden_states_24, 0.1, False, False
        )
        hidden_states_24 = None
        attention_output_4 = layer_output_3 + hidden_states_25
        layer_output_3 = hidden_states_25 = None
        ln_output_4 = torch.nn.functional.layer_norm(
            attention_output_4,
            (384,),
            l_self_modules_encoder_modules_layer_modules_4_modules_ln_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_ln_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_ln_parameters_weight_ = (
            l_self_modules_encoder_modules_layer_modules_4_modules_ln_parameters_bias_
        ) = None
        hidden_states_26 = torch._C._nn.linear(
            ln_output_4,
            l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_,
        )
        ln_output_4 = l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_27 = torch._C._nn.gelu(hidden_states_26)
        hidden_states_26 = None
        hidden_states_28 = torch._C._nn.linear(
            hidden_states_27,
            l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_27 = l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_29 = torch.nn.functional.dropout(
            hidden_states_28, 0.1, False, False
        )
        hidden_states_28 = None
        layer_output_4 = attention_output_4 + hidden_states_29
        attention_output_4 = hidden_states_29 = None
        ln_outputs_5 = torch.nn.functional.layer_norm(
            layer_output_4,
            (384,),
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_ln_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_ln_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_ln_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_ln_parameters_bias_ = (None)
        query_layer_10 = torch._C._nn.linear(
            ln_outputs_5,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_30 = query_layer_10.view(1, -1, 6, 64)
        query_layer_10 = None
        query_layer_11 = view_30.transpose(1, 2)
        view_30 = None
        key_layer_10 = torch._C._nn.linear(
            ln_outputs_5,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_31 = key_layer_10.view(1, -1, 6, 64)
        key_layer_10 = None
        key_layer_11 = view_31.transpose(1, 2)
        view_31 = None
        value_layer_10 = torch._C._nn.linear(
            ln_outputs_5,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        ln_outputs_5 = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_32 = value_layer_10.view(1, -1, 6, 64)
        value_layer_10 = None
        value_layer_11 = view_32.transpose(1, 2)
        view_32 = None
        transpose_23 = key_layer_11.transpose(-1, -2)
        attention_scores_20 = torch.matmul(query_layer_11, transpose_23)
        transpose_23 = None
        arange_10 = torch.arange(
            51, dtype=torch.int64, device=device(type="cuda", index=0)
        )
        position_ids_l_5 = arange_10.view(-1, 1)
        arange_10 = None
        arange_11 = torch.arange(
            51, dtype=torch.int64, device=device(type="cuda", index=0)
        )
        position_ids_r_5 = arange_11.view(1, -1)
        arange_11 = None
        distance_5 = position_ids_l_5 - position_ids_r_5
        position_ids_l_5 = position_ids_r_5 = None
        add_31 = distance_5 + 2048
        distance_5 = None
        sub_12 = add_31 - 1
        add_31 = None
        positional_embedding_10 = torch.nn.functional.embedding(
            sub_12,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_distance_embedding_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        sub_12 = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_distance_embedding_parameters_weight_ = (None)
        positional_embedding_11 = positional_embedding_10.to(dtype=torch.float32)
        positional_embedding_10 = None
        relative_position_scores_query_5 = torch.functional.einsum(
            "bhld,lrd->bhlr", query_layer_11, positional_embedding_11
        )
        query_layer_11 = None
        relative_position_scores_key_5 = torch.functional.einsum(
            "bhrd,lrd->bhlr", key_layer_11, positional_embedding_11
        )
        key_layer_11 = positional_embedding_11 = None
        add_32 = attention_scores_20 + relative_position_scores_query_5
        attention_scores_20 = relative_position_scores_query_5 = None
        attention_scores_21 = add_32 + relative_position_scores_key_5
        add_32 = relative_position_scores_key_5 = None
        attention_scores_22 = attention_scores_21 / 8.0
        attention_scores_21 = None
        attention_scores_23 = attention_scores_22 + extended_attention_mask_2
        attention_scores_22 = extended_attention_mask_2 = None
        attention_probs_10 = torch.nn.functional.softmax(attention_scores_23, dim=-1)
        attention_scores_23 = None
        attention_probs_11 = torch.nn.functional.dropout(
            attention_probs_10, 0.1, False, False
        )
        attention_probs_10 = None
        context_layer_15 = torch.matmul(attention_probs_11, value_layer_11)
        attention_probs_11 = value_layer_11 = None
        permute_5 = context_layer_15.permute(0, 2, 1, 3)
        context_layer_15 = None
        context_layer_16 = permute_5.contiguous()
        permute_5 = None
        context_layer_17 = context_layer_16.view((1, 51, 384))
        context_layer_16 = None
        hidden_states_30 = torch._C._nn.linear(
            context_layer_17,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_17 = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_31 = torch.nn.functional.dropout(
            hidden_states_30, 0.1, False, False
        )
        hidden_states_30 = None
        attention_output_5 = layer_output_4 + hidden_states_31
        layer_output_4 = hidden_states_31 = None
        ln_output_5 = torch.nn.functional.layer_norm(
            attention_output_5,
            (384,),
            l_self_modules_encoder_modules_layer_modules_5_modules_ln_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_ln_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_5_modules_ln_parameters_weight_ = (
            l_self_modules_encoder_modules_layer_modules_5_modules_ln_parameters_bias_
        ) = None
        hidden_states_32 = torch._C._nn.linear(
            ln_output_5,
            l_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_,
        )
        ln_output_5 = l_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_33 = torch._C._nn.gelu(hidden_states_32)
        hidden_states_32 = None
        hidden_states_34 = torch._C._nn.linear(
            hidden_states_33,
            l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_33 = l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_35 = torch.nn.functional.dropout(
            hidden_states_34, 0.1, False, False
        )
        hidden_states_34 = None
        layer_output_5 = attention_output_5 + hidden_states_35
        attention_output_5 = hidden_states_35 = None
        hidden_states_36 = torch.nn.functional.layer_norm(
            layer_output_5,
            (384,),
            l_self_modules_encoder_modules_ln_parameters_weight_,
            l_self_modules_encoder_modules_ln_parameters_bias_,
            1e-12,
        )
        layer_output_5 = (
            l_self_modules_encoder_modules_ln_parameters_weight_
        ) = l_self_modules_encoder_modules_ln_parameters_bias_ = None
        first_token_tensor = hidden_states_36[(slice(None, None, None), 0)]
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
        return (hidden_states_36, pooled_output_1)
