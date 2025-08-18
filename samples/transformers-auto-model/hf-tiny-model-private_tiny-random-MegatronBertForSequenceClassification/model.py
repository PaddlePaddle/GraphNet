import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_input_ids_: torch.Tensor,
        L_attention_mask_: torch.Tensor,
        L_token_type_ids_: torch.Tensor,
        L_self_modules_embeddings_modules_word_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_buffers_position_ids_: torch.Tensor,
        L_self_modules_embeddings_modules_token_type_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_position_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_ln_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_ln_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_ln_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_ln_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_embeddings_buffers_position_ids_ = (
            L_self_modules_embeddings_buffers_position_ids_
        )
        l_self_modules_embeddings_modules_token_type_embeddings_parameters_weight_ = (
            L_self_modules_embeddings_modules_token_type_embeddings_parameters_weight_
        )
        l_self_modules_embeddings_modules_position_embeddings_parameters_weight_ = (
            L_self_modules_embeddings_modules_position_embeddings_parameters_weight_
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_ln_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_ln_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_ln_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_ln_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_
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
        position_ids = l_self_modules_embeddings_buffers_position_ids_[
            (slice(None, None, None), slice(0, 45, None))
        ]
        l_self_modules_embeddings_buffers_position_ids_ = None
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
        position_embeddings = torch.nn.functional.embedding(
            position_ids,
            l_self_modules_embeddings_modules_position_embeddings_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        position_ids = (
            l_self_modules_embeddings_modules_position_embeddings_parameters_weight_
        ) = None
        embeddings += position_embeddings
        embeddings_1 = embeddings
        embeddings = position_embeddings = None
        embeddings_2 = torch.nn.functional.dropout(embeddings_1, 0.1, False, False)
        embeddings_1 = None
        ln_outputs = torch.nn.functional.layer_norm(
            embeddings_2,
            (64,),
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
        view = query_layer.view(1, -1, 4, 16)
        query_layer = None
        query_layer_1 = view.transpose(1, 2)
        view = None
        key_layer = torch._C._nn.linear(
            ln_outputs,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_1 = key_layer.view(1, -1, 4, 16)
        key_layer = None
        key_layer_1 = view_1.transpose(1, 2)
        view_1 = None
        value_layer = torch._C._nn.linear(
            ln_outputs,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        ln_outputs = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_2 = value_layer.view(1, -1, 4, 16)
        value_layer = None
        value_layer_1 = view_2.transpose(1, 2)
        view_2 = None
        transpose_3 = key_layer_1.transpose(-1, -2)
        key_layer_1 = None
        attention_scores = torch.matmul(query_layer_1, transpose_3)
        query_layer_1 = transpose_3 = None
        attention_scores_1 = attention_scores / 4.0
        attention_scores = None
        attention_scores_2 = attention_scores_1 + extended_attention_mask_2
        attention_scores_1 = None
        attention_probs = torch.nn.functional.softmax(attention_scores_2, dim=-1)
        attention_scores_2 = None
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
        context_layer_2 = context_layer_1.view((1, 45, 64))
        context_layer_1 = None
        hidden_states = torch._C._nn.linear(
            context_layer_2,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_2 = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_1 = torch.nn.functional.dropout(hidden_states, 0.1, False, False)
        hidden_states = None
        attention_output = embeddings_2 + hidden_states_1
        embeddings_2 = hidden_states_1 = None
        ln_output = torch.nn.functional.layer_norm(
            attention_output,
            (64,),
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
            (64,),
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
        view_4 = query_layer_2.view(1, -1, 4, 16)
        query_layer_2 = None
        query_layer_3 = view_4.transpose(1, 2)
        view_4 = None
        key_layer_2 = torch._C._nn.linear(
            ln_outputs_1,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_5 = key_layer_2.view(1, -1, 4, 16)
        key_layer_2 = None
        key_layer_3 = view_5.transpose(1, 2)
        view_5 = None
        value_layer_2 = torch._C._nn.linear(
            ln_outputs_1,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        ln_outputs_1 = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_6 = value_layer_2.view(1, -1, 4, 16)
        value_layer_2 = None
        value_layer_3 = view_6.transpose(1, 2)
        view_6 = None
        transpose_7 = key_layer_3.transpose(-1, -2)
        key_layer_3 = None
        attention_scores_3 = torch.matmul(query_layer_3, transpose_7)
        query_layer_3 = transpose_7 = None
        attention_scores_4 = attention_scores_3 / 4.0
        attention_scores_3 = None
        attention_scores_5 = attention_scores_4 + extended_attention_mask_2
        attention_scores_4 = None
        attention_probs_2 = torch.nn.functional.softmax(attention_scores_5, dim=-1)
        attention_scores_5 = None
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
        context_layer_5 = context_layer_4.view((1, 45, 64))
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
            (64,),
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
            (64,),
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
        view_8 = query_layer_4.view(1, -1, 4, 16)
        query_layer_4 = None
        query_layer_5 = view_8.transpose(1, 2)
        view_8 = None
        key_layer_4 = torch._C._nn.linear(
            ln_outputs_2,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_9 = key_layer_4.view(1, -1, 4, 16)
        key_layer_4 = None
        key_layer_5 = view_9.transpose(1, 2)
        view_9 = None
        value_layer_4 = torch._C._nn.linear(
            ln_outputs_2,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        ln_outputs_2 = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_10 = value_layer_4.view(1, -1, 4, 16)
        value_layer_4 = None
        value_layer_5 = view_10.transpose(1, 2)
        view_10 = None
        transpose_11 = key_layer_5.transpose(-1, -2)
        key_layer_5 = None
        attention_scores_6 = torch.matmul(query_layer_5, transpose_11)
        query_layer_5 = transpose_11 = None
        attention_scores_7 = attention_scores_6 / 4.0
        attention_scores_6 = None
        attention_scores_8 = attention_scores_7 + extended_attention_mask_2
        attention_scores_7 = None
        attention_probs_4 = torch.nn.functional.softmax(attention_scores_8, dim=-1)
        attention_scores_8 = None
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
        context_layer_8 = context_layer_7.view((1, 45, 64))
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
            (64,),
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
            (64,),
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
        view_12 = query_layer_6.view(1, -1, 4, 16)
        query_layer_6 = None
        query_layer_7 = view_12.transpose(1, 2)
        view_12 = None
        key_layer_6 = torch._C._nn.linear(
            ln_outputs_3,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_13 = key_layer_6.view(1, -1, 4, 16)
        key_layer_6 = None
        key_layer_7 = view_13.transpose(1, 2)
        view_13 = None
        value_layer_6 = torch._C._nn.linear(
            ln_outputs_3,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        ln_outputs_3 = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_14 = value_layer_6.view(1, -1, 4, 16)
        value_layer_6 = None
        value_layer_7 = view_14.transpose(1, 2)
        view_14 = None
        transpose_15 = key_layer_7.transpose(-1, -2)
        key_layer_7 = None
        attention_scores_9 = torch.matmul(query_layer_7, transpose_15)
        query_layer_7 = transpose_15 = None
        attention_scores_10 = attention_scores_9 / 4.0
        attention_scores_9 = None
        attention_scores_11 = attention_scores_10 + extended_attention_mask_2
        attention_scores_10 = None
        attention_probs_6 = torch.nn.functional.softmax(attention_scores_11, dim=-1)
        attention_scores_11 = None
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
        context_layer_11 = context_layer_10.view((1, 45, 64))
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
            (64,),
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
            (64,),
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
        view_16 = query_layer_8.view(1, -1, 4, 16)
        query_layer_8 = None
        query_layer_9 = view_16.transpose(1, 2)
        view_16 = None
        key_layer_8 = torch._C._nn.linear(
            ln_outputs_4,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_17 = key_layer_8.view(1, -1, 4, 16)
        key_layer_8 = None
        key_layer_9 = view_17.transpose(1, 2)
        view_17 = None
        value_layer_8 = torch._C._nn.linear(
            ln_outputs_4,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        ln_outputs_4 = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_18 = value_layer_8.view(1, -1, 4, 16)
        value_layer_8 = None
        value_layer_9 = view_18.transpose(1, 2)
        view_18 = None
        transpose_19 = key_layer_9.transpose(-1, -2)
        key_layer_9 = None
        attention_scores_12 = torch.matmul(query_layer_9, transpose_19)
        query_layer_9 = transpose_19 = None
        attention_scores_13 = attention_scores_12 / 4.0
        attention_scores_12 = None
        attention_scores_14 = attention_scores_13 + extended_attention_mask_2
        attention_scores_13 = extended_attention_mask_2 = None
        attention_probs_8 = torch.nn.functional.softmax(attention_scores_14, dim=-1)
        attention_scores_14 = None
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
        context_layer_14 = context_layer_13.view((1, 45, 64))
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
            (64,),
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
        hidden_states_30 = torch.nn.functional.layer_norm(
            layer_output_4,
            (64,),
            l_self_modules_encoder_modules_ln_parameters_weight_,
            l_self_modules_encoder_modules_ln_parameters_bias_,
            1e-12,
        )
        layer_output_4 = (
            l_self_modules_encoder_modules_ln_parameters_weight_
        ) = l_self_modules_encoder_modules_ln_parameters_bias_ = None
        first_token_tensor = hidden_states_30[(slice(None, None, None), 0)]
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
        return (hidden_states_30, pooled_output_1)
