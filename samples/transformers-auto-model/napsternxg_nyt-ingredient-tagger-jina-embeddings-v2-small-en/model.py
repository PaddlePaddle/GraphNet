import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_input_ids_: torch.Tensor,
        L_attention_mask_: torch.Tensor,
        L_token_type_ids_: torch.Tensor,
        L_self_modules_embeddings_modules_word_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_token_type_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_buffers_alibi_: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_gated_layers_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_wo_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_gated_layers_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_wo_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_gated_layers_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_wo_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_gated_layers_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_wo_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_embeddings_modules_layer_norm_parameters_weight_ = (
            L_self_modules_embeddings_modules_LayerNorm_parameters_weight_
        )
        l_self_modules_embeddings_modules_layer_norm_parameters_bias_ = (
            L_self_modules_embeddings_modules_LayerNorm_parameters_bias_
        )
        l_self_modules_encoder_buffers_alibi_ = L_self_modules_encoder_buffers_alibi_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_gated_layers_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_gated_layers_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_wo_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_wo_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_wo_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_wo_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_layernorm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_layernorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_layernorm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_layernorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_gated_layers_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_gated_layers_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_wo_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_wo_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_wo_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_wo_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_layernorm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_layernorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_layernorm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_layernorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_gated_layers_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_gated_layers_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_wo_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_wo_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_wo_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_wo_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_layernorm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_layernorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_layernorm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_layernorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_gated_layers_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_gated_layers_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_wo_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_wo_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_wo_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_wo_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_layernorm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_layernorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_layernorm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_layernorm_parameters_bias_
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
        embeddings_1 = torch.nn.functional.layer_norm(
            embeddings,
            (512,),
            l_self_modules_embeddings_modules_layer_norm_parameters_weight_,
            l_self_modules_embeddings_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        embeddings = (
            l_self_modules_embeddings_modules_layer_norm_parameters_weight_
        ) = l_self_modules_embeddings_modules_layer_norm_parameters_bias_ = None
        embeddings_2 = torch.nn.functional.dropout(embeddings_1, 0.1, False, False)
        embeddings_1 = None
        alibi_bias = l_self_modules_encoder_buffers_alibi_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 11, None),
                slice(None, 11, None),
            )
        ]
        l_self_modules_encoder_buffers_alibi_ = None
        mixed_query_layer = torch._C._nn.linear(
            embeddings_2,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        linear_1 = torch._C._nn.linear(
            embeddings_2,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        x = linear_1.view((1, 11, 8, 64))
        linear_1 = None
        key_layer = x.permute(0, 2, 1, 3)
        x = None
        linear_2 = torch._C._nn.linear(
            embeddings_2,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        x_1 = linear_2.view((1, 11, 8, 64))
        linear_2 = None
        value_layer = x_1.permute(0, 2, 1, 3)
        x_1 = None
        x_2 = mixed_query_layer.view((1, 11, 8, 64))
        mixed_query_layer = None
        query_layer = x_2.permute(0, 2, 1, 3)
        x_2 = None
        transpose = key_layer.transpose(-1, -2)
        key_layer = None
        attention_scores = torch.matmul(query_layer, transpose)
        query_layer = transpose = None
        attention_scores_1 = attention_scores / 8.0
        attention_scores = None
        attention_scores_2 = attention_scores_1 + extended_attention_mask_2
        attention_scores_1 = None
        add_2 = attention_scores_2 + alibi_bias
        attention_scores_2 = None
        attention_probs = torch.nn.functional.softmax(add_2, dim=-1)
        add_2 = None
        attention_probs_1 = torch.nn.functional.dropout(
            attention_probs, 0.0, False, False
        )
        attention_probs = None
        context_layer = torch.matmul(attention_probs_1, value_layer)
        attention_probs_1 = value_layer = None
        permute_3 = context_layer.permute(0, 2, 1, 3)
        context_layer = None
        context_layer_1 = permute_3.contiguous()
        permute_3 = None
        context_layer_2 = context_layer_1.view((1, 11, 512))
        context_layer_1 = None
        hidden_states = torch._C._nn.linear(
            context_layer_2,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_2 = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_1 = torch.nn.functional.dropout(hidden_states, 0.1, False, False)
        hidden_states = None
        add_3 = hidden_states_1 + embeddings_2
        hidden_states_1 = embeddings_2 = None
        hidden_states_2 = torch.nn.functional.layer_norm(
            add_3,
            (512,),
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_3 = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_3 = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_gated_layers_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_gated_layers_parameters_weight_ = (
            None
        )
        gated = hidden_states_3[
            (slice(None, None, None), slice(None, None, None), slice(None, 2048, None))
        ]
        non_gated = hidden_states_3[
            (slice(None, None, None), slice(None, None, None), slice(2048, None, None))
        ]
        hidden_states_3 = None
        gelu = torch._C._nn.gelu(gated, approximate="none")
        gated = None
        hidden_states_4 = gelu * non_gated
        gelu = non_gated = None
        hidden_states_5 = torch.nn.functional.dropout(
            hidden_states_4, 0.1, False, False
        )
        hidden_states_4 = None
        hidden_states_6 = torch._C._nn.linear(
            hidden_states_5,
            l_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_wo_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_wo_parameters_bias_,
        )
        hidden_states_5 = l_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_wo_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_wo_parameters_bias_ = (None)
        add_4 = hidden_states_6 + hidden_states_2
        hidden_states_6 = hidden_states_2 = None
        hidden_states_7 = torch.nn.functional.layer_norm(
            add_4,
            (512,),
            l_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_layernorm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_layernorm_parameters_bias_,
            1e-12,
        )
        add_4 = l_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_layernorm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_layernorm_parameters_bias_ = (None)
        mixed_query_layer_1 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        linear_7 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        x_3 = linear_7.view((1, 11, 8, 64))
        linear_7 = None
        key_layer_1 = x_3.permute(0, 2, 1, 3)
        x_3 = None
        linear_8 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        x_4 = linear_8.view((1, 11, 8, 64))
        linear_8 = None
        value_layer_1 = x_4.permute(0, 2, 1, 3)
        x_4 = None
        x_5 = mixed_query_layer_1.view((1, 11, 8, 64))
        mixed_query_layer_1 = None
        query_layer_1 = x_5.permute(0, 2, 1, 3)
        x_5 = None
        transpose_1 = key_layer_1.transpose(-1, -2)
        key_layer_1 = None
        attention_scores_3 = torch.matmul(query_layer_1, transpose_1)
        query_layer_1 = transpose_1 = None
        attention_scores_4 = attention_scores_3 / 8.0
        attention_scores_3 = None
        attention_scores_5 = attention_scores_4 + extended_attention_mask_2
        attention_scores_4 = None
        add_6 = attention_scores_5 + alibi_bias
        attention_scores_5 = None
        attention_probs_2 = torch.nn.functional.softmax(add_6, dim=-1)
        add_6 = None
        attention_probs_3 = torch.nn.functional.dropout(
            attention_probs_2, 0.0, False, False
        )
        attention_probs_2 = None
        context_layer_3 = torch.matmul(attention_probs_3, value_layer_1)
        attention_probs_3 = value_layer_1 = None
        permute_7 = context_layer_3.permute(0, 2, 1, 3)
        context_layer_3 = None
        context_layer_4 = permute_7.contiguous()
        permute_7 = None
        context_layer_5 = context_layer_4.view((1, 11, 512))
        context_layer_4 = None
        hidden_states_8 = torch._C._nn.linear(
            context_layer_5,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_5 = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_9 = torch.nn.functional.dropout(
            hidden_states_8, 0.1, False, False
        )
        hidden_states_8 = None
        add_7 = hidden_states_9 + hidden_states_7
        hidden_states_9 = hidden_states_7 = None
        hidden_states_10 = torch.nn.functional.layer_norm(
            add_7,
            (512,),
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_7 = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_11 = torch._C._nn.linear(
            hidden_states_10,
            l_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_gated_layers_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_gated_layers_parameters_weight_ = (
            None
        )
        gated_1 = hidden_states_11[
            (slice(None, None, None), slice(None, None, None), slice(None, 2048, None))
        ]
        non_gated_1 = hidden_states_11[
            (slice(None, None, None), slice(None, None, None), slice(2048, None, None))
        ]
        hidden_states_11 = None
        gelu_1 = torch._C._nn.gelu(gated_1, approximate="none")
        gated_1 = None
        hidden_states_12 = gelu_1 * non_gated_1
        gelu_1 = non_gated_1 = None
        hidden_states_13 = torch.nn.functional.dropout(
            hidden_states_12, 0.1, False, False
        )
        hidden_states_12 = None
        hidden_states_14 = torch._C._nn.linear(
            hidden_states_13,
            l_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_wo_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_wo_parameters_bias_,
        )
        hidden_states_13 = l_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_wo_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_wo_parameters_bias_ = (None)
        add_8 = hidden_states_14 + hidden_states_10
        hidden_states_14 = hidden_states_10 = None
        hidden_states_15 = torch.nn.functional.layer_norm(
            add_8,
            (512,),
            l_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_layernorm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_layernorm_parameters_bias_,
            1e-12,
        )
        add_8 = l_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_layernorm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_layernorm_parameters_bias_ = (None)
        mixed_query_layer_2 = torch._C._nn.linear(
            hidden_states_15,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        linear_13 = torch._C._nn.linear(
            hidden_states_15,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        x_6 = linear_13.view((1, 11, 8, 64))
        linear_13 = None
        key_layer_2 = x_6.permute(0, 2, 1, 3)
        x_6 = None
        linear_14 = torch._C._nn.linear(
            hidden_states_15,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        x_7 = linear_14.view((1, 11, 8, 64))
        linear_14 = None
        value_layer_2 = x_7.permute(0, 2, 1, 3)
        x_7 = None
        x_8 = mixed_query_layer_2.view((1, 11, 8, 64))
        mixed_query_layer_2 = None
        query_layer_2 = x_8.permute(0, 2, 1, 3)
        x_8 = None
        transpose_2 = key_layer_2.transpose(-1, -2)
        key_layer_2 = None
        attention_scores_6 = torch.matmul(query_layer_2, transpose_2)
        query_layer_2 = transpose_2 = None
        attention_scores_7 = attention_scores_6 / 8.0
        attention_scores_6 = None
        attention_scores_8 = attention_scores_7 + extended_attention_mask_2
        attention_scores_7 = None
        add_10 = attention_scores_8 + alibi_bias
        attention_scores_8 = None
        attention_probs_4 = torch.nn.functional.softmax(add_10, dim=-1)
        add_10 = None
        attention_probs_5 = torch.nn.functional.dropout(
            attention_probs_4, 0.0, False, False
        )
        attention_probs_4 = None
        context_layer_6 = torch.matmul(attention_probs_5, value_layer_2)
        attention_probs_5 = value_layer_2 = None
        permute_11 = context_layer_6.permute(0, 2, 1, 3)
        context_layer_6 = None
        context_layer_7 = permute_11.contiguous()
        permute_11 = None
        context_layer_8 = context_layer_7.view((1, 11, 512))
        context_layer_7 = None
        hidden_states_16 = torch._C._nn.linear(
            context_layer_8,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_8 = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_17 = torch.nn.functional.dropout(
            hidden_states_16, 0.1, False, False
        )
        hidden_states_16 = None
        add_11 = hidden_states_17 + hidden_states_15
        hidden_states_17 = hidden_states_15 = None
        hidden_states_18 = torch.nn.functional.layer_norm(
            add_11,
            (512,),
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_11 = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_19 = torch._C._nn.linear(
            hidden_states_18,
            l_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_gated_layers_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_gated_layers_parameters_weight_ = (
            None
        )
        gated_2 = hidden_states_19[
            (slice(None, None, None), slice(None, None, None), slice(None, 2048, None))
        ]
        non_gated_2 = hidden_states_19[
            (slice(None, None, None), slice(None, None, None), slice(2048, None, None))
        ]
        hidden_states_19 = None
        gelu_2 = torch._C._nn.gelu(gated_2, approximate="none")
        gated_2 = None
        hidden_states_20 = gelu_2 * non_gated_2
        gelu_2 = non_gated_2 = None
        hidden_states_21 = torch.nn.functional.dropout(
            hidden_states_20, 0.1, False, False
        )
        hidden_states_20 = None
        hidden_states_22 = torch._C._nn.linear(
            hidden_states_21,
            l_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_wo_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_wo_parameters_bias_,
        )
        hidden_states_21 = l_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_wo_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_wo_parameters_bias_ = (None)
        add_12 = hidden_states_22 + hidden_states_18
        hidden_states_22 = hidden_states_18 = None
        hidden_states_23 = torch.nn.functional.layer_norm(
            add_12,
            (512,),
            l_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_layernorm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_layernorm_parameters_bias_,
            1e-12,
        )
        add_12 = l_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_layernorm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_layernorm_parameters_bias_ = (None)
        mixed_query_layer_3 = torch._C._nn.linear(
            hidden_states_23,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        linear_19 = torch._C._nn.linear(
            hidden_states_23,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        x_9 = linear_19.view((1, 11, 8, 64))
        linear_19 = None
        key_layer_3 = x_9.permute(0, 2, 1, 3)
        x_9 = None
        linear_20 = torch._C._nn.linear(
            hidden_states_23,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        x_10 = linear_20.view((1, 11, 8, 64))
        linear_20 = None
        value_layer_3 = x_10.permute(0, 2, 1, 3)
        x_10 = None
        x_11 = mixed_query_layer_3.view((1, 11, 8, 64))
        mixed_query_layer_3 = None
        query_layer_3 = x_11.permute(0, 2, 1, 3)
        x_11 = None
        transpose_3 = key_layer_3.transpose(-1, -2)
        key_layer_3 = None
        attention_scores_9 = torch.matmul(query_layer_3, transpose_3)
        query_layer_3 = transpose_3 = None
        attention_scores_10 = attention_scores_9 / 8.0
        attention_scores_9 = None
        attention_scores_11 = attention_scores_10 + extended_attention_mask_2
        attention_scores_10 = extended_attention_mask_2 = None
        add_14 = attention_scores_11 + alibi_bias
        attention_scores_11 = alibi_bias = None
        attention_probs_6 = torch.nn.functional.softmax(add_14, dim=-1)
        add_14 = None
        attention_probs_7 = torch.nn.functional.dropout(
            attention_probs_6, 0.0, False, False
        )
        attention_probs_6 = None
        context_layer_9 = torch.matmul(attention_probs_7, value_layer_3)
        attention_probs_7 = value_layer_3 = None
        permute_15 = context_layer_9.permute(0, 2, 1, 3)
        context_layer_9 = None
        context_layer_10 = permute_15.contiguous()
        permute_15 = None
        context_layer_11 = context_layer_10.view((1, 11, 512))
        context_layer_10 = None
        hidden_states_24 = torch._C._nn.linear(
            context_layer_11,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_11 = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_25 = torch.nn.functional.dropout(
            hidden_states_24, 0.1, False, False
        )
        hidden_states_24 = None
        add_15 = hidden_states_25 + hidden_states_23
        hidden_states_25 = hidden_states_23 = None
        hidden_states_26 = torch.nn.functional.layer_norm(
            add_15,
            (512,),
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_15 = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_27 = torch._C._nn.linear(
            hidden_states_26,
            l_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_gated_layers_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_gated_layers_parameters_weight_ = (
            None
        )
        gated_3 = hidden_states_27[
            (slice(None, None, None), slice(None, None, None), slice(None, 2048, None))
        ]
        non_gated_3 = hidden_states_27[
            (slice(None, None, None), slice(None, None, None), slice(2048, None, None))
        ]
        hidden_states_27 = None
        gelu_3 = torch._C._nn.gelu(gated_3, approximate="none")
        gated_3 = None
        hidden_states_28 = gelu_3 * non_gated_3
        gelu_3 = non_gated_3 = None
        hidden_states_29 = torch.nn.functional.dropout(
            hidden_states_28, 0.1, False, False
        )
        hidden_states_28 = None
        hidden_states_30 = torch._C._nn.linear(
            hidden_states_29,
            l_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_wo_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_wo_parameters_bias_,
        )
        hidden_states_29 = l_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_wo_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_wo_parameters_bias_ = (None)
        add_16 = hidden_states_30 + hidden_states_26
        hidden_states_30 = hidden_states_26 = None
        hidden_states_31 = torch.nn.functional.layer_norm(
            add_16,
            (512,),
            l_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_layernorm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_layernorm_parameters_bias_,
            1e-12,
        )
        add_16 = l_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_layernorm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_layernorm_parameters_bias_ = (None)
        first_token_tensor = hidden_states_31[(slice(None, None, None), 0)]
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
        return (hidden_states_31, pooled_output_1)
