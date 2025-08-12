import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_input_ids_: torch.Tensor,
        L_attention_mask_: torch.Tensor,
        L_token_type_ids_: torch.Tensor,
        L_self_modules_embeddings_buffers_position_ids_: torch.Tensor,
        L_self_modules_embeddings_modules_word_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_token_type_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_position_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_embedding_hidden_mapping_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_embedding_hidden_mapping_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_pooler_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_pooler_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_input_ids_ = L_input_ids_
        l_attention_mask_ = L_attention_mask_
        l_token_type_ids_ = L_token_type_ids_
        l_self_modules_embeddings_buffers_position_ids_ = (
            L_self_modules_embeddings_buffers_position_ids_
        )
        l_self_modules_embeddings_modules_word_embeddings_parameters_weight_ = (
            L_self_modules_embeddings_modules_word_embeddings_parameters_weight_
        )
        l_self_modules_embeddings_modules_token_type_embeddings_parameters_weight_ = (
            L_self_modules_embeddings_modules_token_type_embeddings_parameters_weight_
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
        l_self_modules_encoder_modules_embedding_hidden_mapping_in_parameters_weight_ = L_self_modules_encoder_modules_embedding_hidden_mapping_in_parameters_weight_
        l_self_modules_encoder_modules_embedding_hidden_mapping_in_parameters_bias_ = (
            L_self_modules_encoder_modules_embedding_hidden_mapping_in_parameters_bias_
        )
        l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_weight_ = L_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_weight_
        l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_bias_ = L_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_bias_
        l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_weight_ = L_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_weight_
        l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_bias_ = L_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_bias_
        l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_bias_
        l_self_modules_pooler_parameters_weight_ = (
            L_self_modules_pooler_parameters_weight_
        )
        l_self_modules_pooler_parameters_bias_ = L_self_modules_pooler_parameters_bias_
        position_ids = l_self_modules_embeddings_buffers_position_ids_[
            (slice(None, None, None), slice(0, 17, None))
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
        embeddings_2 = torch.nn.functional.layer_norm(
            embeddings_1,
            (128,),
            l_self_modules_embeddings_modules_layer_norm_parameters_weight_,
            l_self_modules_embeddings_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        embeddings_1 = (
            l_self_modules_embeddings_modules_layer_norm_parameters_weight_
        ) = l_self_modules_embeddings_modules_layer_norm_parameters_bias_ = None
        embeddings_3 = torch.nn.functional.dropout(embeddings_2, 0, False, False)
        embeddings_2 = None
        getitem_1 = l_attention_mask_[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        l_attention_mask_ = None
        expand = getitem_1.expand(1, 1, 17, 17)
        getitem_1 = None
        expanded_mask = expand.to(torch.float32)
        expand = None
        tensor = torch.tensor(1.0, dtype=torch.float32)
        inverted_mask = tensor - expanded_mask
        tensor = expanded_mask = None
        to_1 = inverted_mask.to(torch.bool)
        extended_attention_mask = inverted_mask.masked_fill(
            to_1, -3.4028234663852886e38
        )
        inverted_mask = to_1 = None
        hidden_states = torch._C._nn.linear(
            embeddings_3,
            l_self_modules_encoder_modules_embedding_hidden_mapping_in_parameters_weight_,
            l_self_modules_encoder_modules_embedding_hidden_mapping_in_parameters_bias_,
        )
        embeddings_3 = l_self_modules_encoder_modules_embedding_hidden_mapping_in_parameters_weight_ = (
            l_self_modules_encoder_modules_embedding_hidden_mapping_in_parameters_bias_
        ) = None
        linear_1 = torch._C._nn.linear(
            hidden_states,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_bias_,
        )
        view = linear_1.view(1, -1, 12, 64)
        linear_1 = None
        query_layer = view.transpose(1, 2)
        view = None
        linear_2 = torch._C._nn.linear(
            hidden_states,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_bias_,
        )
        view_1 = linear_2.view(1, -1, 12, 64)
        linear_2 = None
        key_layer = view_1.transpose(1, 2)
        view_1 = None
        linear_3 = torch._C._nn.linear(
            hidden_states,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_bias_,
        )
        view_2 = linear_3.view(1, -1, 12, 64)
        linear_3 = None
        value_layer = view_2.transpose(1, 2)
        view_2 = None
        attention_output = torch._C._nn.scaled_dot_product_attention(
            query=query_layer,
            key=key_layer,
            value=value_layer,
            attn_mask=extended_attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        query_layer = key_layer = value_layer = None
        attention_output_1 = attention_output.transpose(1, 2)
        attention_output = None
        attention_output_2 = attention_output_1.reshape(1, 17, 768)
        attention_output_1 = None
        projected_context_layer = torch._C._nn.linear(
            attention_output_2,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_bias_,
        )
        attention_output_2 = None
        projected_context_layer_dropout = torch.nn.functional.dropout(
            projected_context_layer, 0, False, False
        )
        projected_context_layer = None
        add_1 = hidden_states + projected_context_layer_dropout
        hidden_states = projected_context_layer_dropout = None
        layernormed_context_layer = torch.nn.functional.layer_norm(
            add_1,
            (768,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_1 = None
        ffn_output = torch._C._nn.linear(
            layernormed_context_layer,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_bias_,
        )
        ffn_output_1 = torch._C._nn.gelu(ffn_output)
        ffn_output = None
        ffn_output_2 = torch._C._nn.linear(
            ffn_output_1,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_bias_,
        )
        ffn_output_1 = None
        add_2 = ffn_output_2 + layernormed_context_layer
        ffn_output_2 = layernormed_context_layer = None
        hidden_states_1 = torch.nn.functional.layer_norm(
            add_2,
            (768,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_bias_,
            1e-12,
        )
        add_2 = None
        linear_7 = torch._C._nn.linear(
            hidden_states_1,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_bias_ = (None)
        view_3 = linear_7.view(1, -1, 12, 64)
        linear_7 = None
        query_layer_1 = view_3.transpose(1, 2)
        view_3 = None
        linear_8 = torch._C._nn.linear(
            hidden_states_1,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_bias_ = (None)
        view_4 = linear_8.view(1, -1, 12, 64)
        linear_8 = None
        key_layer_1 = view_4.transpose(1, 2)
        view_4 = None
        linear_9 = torch._C._nn.linear(
            hidden_states_1,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_bias_ = (None)
        view_5 = linear_9.view(1, -1, 12, 64)
        linear_9 = None
        value_layer_1 = view_5.transpose(1, 2)
        view_5 = None
        attention_output_3 = torch._C._nn.scaled_dot_product_attention(
            query=query_layer_1,
            key=key_layer_1,
            value=value_layer_1,
            attn_mask=extended_attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        query_layer_1 = key_layer_1 = value_layer_1 = extended_attention_mask = None
        attention_output_4 = attention_output_3.transpose(1, 2)
        attention_output_3 = None
        attention_output_5 = attention_output_4.reshape(1, 17, 768)
        attention_output_4 = None
        projected_context_layer_1 = torch._C._nn.linear(
            attention_output_5,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_bias_,
        )
        attention_output_5 = l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_bias_ = (None)
        projected_context_layer_dropout_1 = torch.nn.functional.dropout(
            projected_context_layer_1, 0, False, False
        )
        projected_context_layer_1 = None
        add_3 = hidden_states_1 + projected_context_layer_dropout_1
        hidden_states_1 = projected_context_layer_dropout_1 = None
        layernormed_context_layer_1 = torch.nn.functional.layer_norm(
            add_3,
            (768,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_3 = l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_bias_ = (None)
        ffn_output_3 = torch._C._nn.linear(
            layernormed_context_layer_1,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_bias_,
        )
        l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_weight_ = l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_bias_ = (None)
        ffn_output_4 = torch._C._nn.gelu(ffn_output_3)
        ffn_output_3 = None
        ffn_output_5 = torch._C._nn.linear(
            ffn_output_4,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_bias_,
        )
        ffn_output_4 = l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_weight_ = l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_bias_ = (None)
        add_4 = ffn_output_5 + layernormed_context_layer_1
        ffn_output_5 = layernormed_context_layer_1 = None
        hidden_states_2 = torch.nn.functional.layer_norm(
            add_4,
            (768,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_bias_,
            1e-12,
        )
        add_4 = l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_bias_ = (None)
        getitem_2 = hidden_states_2[(slice(None, None, None), 0)]
        linear_13 = torch._C._nn.linear(
            getitem_2,
            l_self_modules_pooler_parameters_weight_,
            l_self_modules_pooler_parameters_bias_,
        )
        getitem_2 = (
            l_self_modules_pooler_parameters_weight_
        ) = l_self_modules_pooler_parameters_bias_ = None
        pooled_output = torch.tanh(linear_13)
        linear_13 = None
        return (hidden_states_2, pooled_output)
