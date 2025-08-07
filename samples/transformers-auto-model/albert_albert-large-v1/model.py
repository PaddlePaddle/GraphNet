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
            (slice(None, None, None), slice(0, 12, None))
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
        embeddings_3 = torch.nn.functional.dropout(embeddings_2, 0.1, False, False)
        embeddings_2 = None
        getitem_1 = l_attention_mask_[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        l_attention_mask_ = None
        expand = getitem_1.expand(1, 1, 12, 12)
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
        view = linear_1.view(1, -1, 16, 64)
        linear_1 = None
        query_layer = view.transpose(1, 2)
        view = None
        linear_2 = torch._C._nn.linear(
            hidden_states,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_bias_,
        )
        view_1 = linear_2.view(1, -1, 16, 64)
        linear_2 = None
        key_layer = view_1.transpose(1, 2)
        view_1 = None
        linear_3 = torch._C._nn.linear(
            hidden_states,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_bias_,
        )
        view_2 = linear_3.view(1, -1, 16, 64)
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
        attention_output_2 = attention_output_1.reshape(1, 12, 1024)
        attention_output_1 = None
        projected_context_layer = torch._C._nn.linear(
            attention_output_2,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_bias_,
        )
        attention_output_2 = None
        projected_context_layer_dropout = torch.nn.functional.dropout(
            projected_context_layer, 0.1, False, False
        )
        projected_context_layer = None
        add_1 = hidden_states + projected_context_layer_dropout
        hidden_states = projected_context_layer_dropout = None
        layernormed_context_layer = torch.nn.functional.layer_norm(
            add_1,
            (1024,),
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
            (1024,),
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
        view_3 = linear_7.view(1, -1, 16, 64)
        linear_7 = None
        query_layer_1 = view_3.transpose(1, 2)
        view_3 = None
        linear_8 = torch._C._nn.linear(
            hidden_states_1,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_bias_,
        )
        view_4 = linear_8.view(1, -1, 16, 64)
        linear_8 = None
        key_layer_1 = view_4.transpose(1, 2)
        view_4 = None
        linear_9 = torch._C._nn.linear(
            hidden_states_1,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_bias_,
        )
        view_5 = linear_9.view(1, -1, 16, 64)
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
        query_layer_1 = key_layer_1 = value_layer_1 = None
        attention_output_4 = attention_output_3.transpose(1, 2)
        attention_output_3 = None
        attention_output_5 = attention_output_4.reshape(1, 12, 1024)
        attention_output_4 = None
        projected_context_layer_1 = torch._C._nn.linear(
            attention_output_5,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_bias_,
        )
        attention_output_5 = None
        projected_context_layer_dropout_1 = torch.nn.functional.dropout(
            projected_context_layer_1, 0.1, False, False
        )
        projected_context_layer_1 = None
        add_3 = hidden_states_1 + projected_context_layer_dropout_1
        hidden_states_1 = projected_context_layer_dropout_1 = None
        layernormed_context_layer_1 = torch.nn.functional.layer_norm(
            add_3,
            (1024,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_3 = None
        ffn_output_3 = torch._C._nn.linear(
            layernormed_context_layer_1,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_bias_,
        )
        ffn_output_4 = torch._C._nn.gelu(ffn_output_3)
        ffn_output_3 = None
        ffn_output_5 = torch._C._nn.linear(
            ffn_output_4,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_bias_,
        )
        ffn_output_4 = None
        add_4 = ffn_output_5 + layernormed_context_layer_1
        ffn_output_5 = layernormed_context_layer_1 = None
        hidden_states_2 = torch.nn.functional.layer_norm(
            add_4,
            (1024,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_bias_,
            1e-12,
        )
        add_4 = None
        linear_13 = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_bias_,
        )
        view_6 = linear_13.view(1, -1, 16, 64)
        linear_13 = None
        query_layer_2 = view_6.transpose(1, 2)
        view_6 = None
        linear_14 = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_bias_,
        )
        view_7 = linear_14.view(1, -1, 16, 64)
        linear_14 = None
        key_layer_2 = view_7.transpose(1, 2)
        view_7 = None
        linear_15 = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_bias_,
        )
        view_8 = linear_15.view(1, -1, 16, 64)
        linear_15 = None
        value_layer_2 = view_8.transpose(1, 2)
        view_8 = None
        attention_output_6 = torch._C._nn.scaled_dot_product_attention(
            query=query_layer_2,
            key=key_layer_2,
            value=value_layer_2,
            attn_mask=extended_attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        query_layer_2 = key_layer_2 = value_layer_2 = None
        attention_output_7 = attention_output_6.transpose(1, 2)
        attention_output_6 = None
        attention_output_8 = attention_output_7.reshape(1, 12, 1024)
        attention_output_7 = None
        projected_context_layer_2 = torch._C._nn.linear(
            attention_output_8,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_bias_,
        )
        attention_output_8 = None
        projected_context_layer_dropout_2 = torch.nn.functional.dropout(
            projected_context_layer_2, 0.1, False, False
        )
        projected_context_layer_2 = None
        add_5 = hidden_states_2 + projected_context_layer_dropout_2
        hidden_states_2 = projected_context_layer_dropout_2 = None
        layernormed_context_layer_2 = torch.nn.functional.layer_norm(
            add_5,
            (1024,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_5 = None
        ffn_output_6 = torch._C._nn.linear(
            layernormed_context_layer_2,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_bias_,
        )
        ffn_output_7 = torch._C._nn.gelu(ffn_output_6)
        ffn_output_6 = None
        ffn_output_8 = torch._C._nn.linear(
            ffn_output_7,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_bias_,
        )
        ffn_output_7 = None
        add_6 = ffn_output_8 + layernormed_context_layer_2
        ffn_output_8 = layernormed_context_layer_2 = None
        hidden_states_3 = torch.nn.functional.layer_norm(
            add_6,
            (1024,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_bias_,
            1e-12,
        )
        add_6 = None
        linear_19 = torch._C._nn.linear(
            hidden_states_3,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_bias_,
        )
        view_9 = linear_19.view(1, -1, 16, 64)
        linear_19 = None
        query_layer_3 = view_9.transpose(1, 2)
        view_9 = None
        linear_20 = torch._C._nn.linear(
            hidden_states_3,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_bias_,
        )
        view_10 = linear_20.view(1, -1, 16, 64)
        linear_20 = None
        key_layer_3 = view_10.transpose(1, 2)
        view_10 = None
        linear_21 = torch._C._nn.linear(
            hidden_states_3,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_bias_,
        )
        view_11 = linear_21.view(1, -1, 16, 64)
        linear_21 = None
        value_layer_3 = view_11.transpose(1, 2)
        view_11 = None
        attention_output_9 = torch._C._nn.scaled_dot_product_attention(
            query=query_layer_3,
            key=key_layer_3,
            value=value_layer_3,
            attn_mask=extended_attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        query_layer_3 = key_layer_3 = value_layer_3 = None
        attention_output_10 = attention_output_9.transpose(1, 2)
        attention_output_9 = None
        attention_output_11 = attention_output_10.reshape(1, 12, 1024)
        attention_output_10 = None
        projected_context_layer_3 = torch._C._nn.linear(
            attention_output_11,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_bias_,
        )
        attention_output_11 = None
        projected_context_layer_dropout_3 = torch.nn.functional.dropout(
            projected_context_layer_3, 0.1, False, False
        )
        projected_context_layer_3 = None
        add_7 = hidden_states_3 + projected_context_layer_dropout_3
        hidden_states_3 = projected_context_layer_dropout_3 = None
        layernormed_context_layer_3 = torch.nn.functional.layer_norm(
            add_7,
            (1024,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_7 = None
        ffn_output_9 = torch._C._nn.linear(
            layernormed_context_layer_3,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_bias_,
        )
        ffn_output_10 = torch._C._nn.gelu(ffn_output_9)
        ffn_output_9 = None
        ffn_output_11 = torch._C._nn.linear(
            ffn_output_10,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_bias_,
        )
        ffn_output_10 = None
        add_8 = ffn_output_11 + layernormed_context_layer_3
        ffn_output_11 = layernormed_context_layer_3 = None
        hidden_states_4 = torch.nn.functional.layer_norm(
            add_8,
            (1024,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_bias_,
            1e-12,
        )
        add_8 = None
        linear_25 = torch._C._nn.linear(
            hidden_states_4,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_bias_,
        )
        view_12 = linear_25.view(1, -1, 16, 64)
        linear_25 = None
        query_layer_4 = view_12.transpose(1, 2)
        view_12 = None
        linear_26 = torch._C._nn.linear(
            hidden_states_4,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_bias_,
        )
        view_13 = linear_26.view(1, -1, 16, 64)
        linear_26 = None
        key_layer_4 = view_13.transpose(1, 2)
        view_13 = None
        linear_27 = torch._C._nn.linear(
            hidden_states_4,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_bias_,
        )
        view_14 = linear_27.view(1, -1, 16, 64)
        linear_27 = None
        value_layer_4 = view_14.transpose(1, 2)
        view_14 = None
        attention_output_12 = torch._C._nn.scaled_dot_product_attention(
            query=query_layer_4,
            key=key_layer_4,
            value=value_layer_4,
            attn_mask=extended_attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        query_layer_4 = key_layer_4 = value_layer_4 = None
        attention_output_13 = attention_output_12.transpose(1, 2)
        attention_output_12 = None
        attention_output_14 = attention_output_13.reshape(1, 12, 1024)
        attention_output_13 = None
        projected_context_layer_4 = torch._C._nn.linear(
            attention_output_14,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_bias_,
        )
        attention_output_14 = None
        projected_context_layer_dropout_4 = torch.nn.functional.dropout(
            projected_context_layer_4, 0.1, False, False
        )
        projected_context_layer_4 = None
        add_9 = hidden_states_4 + projected_context_layer_dropout_4
        hidden_states_4 = projected_context_layer_dropout_4 = None
        layernormed_context_layer_4 = torch.nn.functional.layer_norm(
            add_9,
            (1024,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_9 = None
        ffn_output_12 = torch._C._nn.linear(
            layernormed_context_layer_4,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_bias_,
        )
        ffn_output_13 = torch._C._nn.gelu(ffn_output_12)
        ffn_output_12 = None
        ffn_output_14 = torch._C._nn.linear(
            ffn_output_13,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_bias_,
        )
        ffn_output_13 = None
        add_10 = ffn_output_14 + layernormed_context_layer_4
        ffn_output_14 = layernormed_context_layer_4 = None
        hidden_states_5 = torch.nn.functional.layer_norm(
            add_10,
            (1024,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_bias_,
            1e-12,
        )
        add_10 = None
        linear_31 = torch._C._nn.linear(
            hidden_states_5,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_bias_,
        )
        view_15 = linear_31.view(1, -1, 16, 64)
        linear_31 = None
        query_layer_5 = view_15.transpose(1, 2)
        view_15 = None
        linear_32 = torch._C._nn.linear(
            hidden_states_5,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_bias_,
        )
        view_16 = linear_32.view(1, -1, 16, 64)
        linear_32 = None
        key_layer_5 = view_16.transpose(1, 2)
        view_16 = None
        linear_33 = torch._C._nn.linear(
            hidden_states_5,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_bias_,
        )
        view_17 = linear_33.view(1, -1, 16, 64)
        linear_33 = None
        value_layer_5 = view_17.transpose(1, 2)
        view_17 = None
        attention_output_15 = torch._C._nn.scaled_dot_product_attention(
            query=query_layer_5,
            key=key_layer_5,
            value=value_layer_5,
            attn_mask=extended_attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        query_layer_5 = key_layer_5 = value_layer_5 = None
        attention_output_16 = attention_output_15.transpose(1, 2)
        attention_output_15 = None
        attention_output_17 = attention_output_16.reshape(1, 12, 1024)
        attention_output_16 = None
        projected_context_layer_5 = torch._C._nn.linear(
            attention_output_17,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_bias_,
        )
        attention_output_17 = None
        projected_context_layer_dropout_5 = torch.nn.functional.dropout(
            projected_context_layer_5, 0.1, False, False
        )
        projected_context_layer_5 = None
        add_11 = hidden_states_5 + projected_context_layer_dropout_5
        hidden_states_5 = projected_context_layer_dropout_5 = None
        layernormed_context_layer_5 = torch.nn.functional.layer_norm(
            add_11,
            (1024,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_11 = None
        ffn_output_15 = torch._C._nn.linear(
            layernormed_context_layer_5,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_bias_,
        )
        ffn_output_16 = torch._C._nn.gelu(ffn_output_15)
        ffn_output_15 = None
        ffn_output_17 = torch._C._nn.linear(
            ffn_output_16,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_bias_,
        )
        ffn_output_16 = None
        add_12 = ffn_output_17 + layernormed_context_layer_5
        ffn_output_17 = layernormed_context_layer_5 = None
        hidden_states_6 = torch.nn.functional.layer_norm(
            add_12,
            (1024,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_bias_,
            1e-12,
        )
        add_12 = None
        linear_37 = torch._C._nn.linear(
            hidden_states_6,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_bias_,
        )
        view_18 = linear_37.view(1, -1, 16, 64)
        linear_37 = None
        query_layer_6 = view_18.transpose(1, 2)
        view_18 = None
        linear_38 = torch._C._nn.linear(
            hidden_states_6,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_bias_,
        )
        view_19 = linear_38.view(1, -1, 16, 64)
        linear_38 = None
        key_layer_6 = view_19.transpose(1, 2)
        view_19 = None
        linear_39 = torch._C._nn.linear(
            hidden_states_6,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_bias_,
        )
        view_20 = linear_39.view(1, -1, 16, 64)
        linear_39 = None
        value_layer_6 = view_20.transpose(1, 2)
        view_20 = None
        attention_output_18 = torch._C._nn.scaled_dot_product_attention(
            query=query_layer_6,
            key=key_layer_6,
            value=value_layer_6,
            attn_mask=extended_attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        query_layer_6 = key_layer_6 = value_layer_6 = None
        attention_output_19 = attention_output_18.transpose(1, 2)
        attention_output_18 = None
        attention_output_20 = attention_output_19.reshape(1, 12, 1024)
        attention_output_19 = None
        projected_context_layer_6 = torch._C._nn.linear(
            attention_output_20,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_bias_,
        )
        attention_output_20 = None
        projected_context_layer_dropout_6 = torch.nn.functional.dropout(
            projected_context_layer_6, 0.1, False, False
        )
        projected_context_layer_6 = None
        add_13 = hidden_states_6 + projected_context_layer_dropout_6
        hidden_states_6 = projected_context_layer_dropout_6 = None
        layernormed_context_layer_6 = torch.nn.functional.layer_norm(
            add_13,
            (1024,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_13 = None
        ffn_output_18 = torch._C._nn.linear(
            layernormed_context_layer_6,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_bias_,
        )
        ffn_output_19 = torch._C._nn.gelu(ffn_output_18)
        ffn_output_18 = None
        ffn_output_20 = torch._C._nn.linear(
            ffn_output_19,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_bias_,
        )
        ffn_output_19 = None
        add_14 = ffn_output_20 + layernormed_context_layer_6
        ffn_output_20 = layernormed_context_layer_6 = None
        hidden_states_7 = torch.nn.functional.layer_norm(
            add_14,
            (1024,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_bias_,
            1e-12,
        )
        add_14 = None
        linear_43 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_bias_,
        )
        view_21 = linear_43.view(1, -1, 16, 64)
        linear_43 = None
        query_layer_7 = view_21.transpose(1, 2)
        view_21 = None
        linear_44 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_bias_,
        )
        view_22 = linear_44.view(1, -1, 16, 64)
        linear_44 = None
        key_layer_7 = view_22.transpose(1, 2)
        view_22 = None
        linear_45 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_bias_,
        )
        view_23 = linear_45.view(1, -1, 16, 64)
        linear_45 = None
        value_layer_7 = view_23.transpose(1, 2)
        view_23 = None
        attention_output_21 = torch._C._nn.scaled_dot_product_attention(
            query=query_layer_7,
            key=key_layer_7,
            value=value_layer_7,
            attn_mask=extended_attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        query_layer_7 = key_layer_7 = value_layer_7 = None
        attention_output_22 = attention_output_21.transpose(1, 2)
        attention_output_21 = None
        attention_output_23 = attention_output_22.reshape(1, 12, 1024)
        attention_output_22 = None
        projected_context_layer_7 = torch._C._nn.linear(
            attention_output_23,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_bias_,
        )
        attention_output_23 = None
        projected_context_layer_dropout_7 = torch.nn.functional.dropout(
            projected_context_layer_7, 0.1, False, False
        )
        projected_context_layer_7 = None
        add_15 = hidden_states_7 + projected_context_layer_dropout_7
        hidden_states_7 = projected_context_layer_dropout_7 = None
        layernormed_context_layer_7 = torch.nn.functional.layer_norm(
            add_15,
            (1024,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_15 = None
        ffn_output_21 = torch._C._nn.linear(
            layernormed_context_layer_7,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_bias_,
        )
        ffn_output_22 = torch._C._nn.gelu(ffn_output_21)
        ffn_output_21 = None
        ffn_output_23 = torch._C._nn.linear(
            ffn_output_22,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_bias_,
        )
        ffn_output_22 = None
        add_16 = ffn_output_23 + layernormed_context_layer_7
        ffn_output_23 = layernormed_context_layer_7 = None
        hidden_states_8 = torch.nn.functional.layer_norm(
            add_16,
            (1024,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_bias_,
            1e-12,
        )
        add_16 = None
        linear_49 = torch._C._nn.linear(
            hidden_states_8,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_bias_,
        )
        view_24 = linear_49.view(1, -1, 16, 64)
        linear_49 = None
        query_layer_8 = view_24.transpose(1, 2)
        view_24 = None
        linear_50 = torch._C._nn.linear(
            hidden_states_8,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_bias_,
        )
        view_25 = linear_50.view(1, -1, 16, 64)
        linear_50 = None
        key_layer_8 = view_25.transpose(1, 2)
        view_25 = None
        linear_51 = torch._C._nn.linear(
            hidden_states_8,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_bias_,
        )
        view_26 = linear_51.view(1, -1, 16, 64)
        linear_51 = None
        value_layer_8 = view_26.transpose(1, 2)
        view_26 = None
        attention_output_24 = torch._C._nn.scaled_dot_product_attention(
            query=query_layer_8,
            key=key_layer_8,
            value=value_layer_8,
            attn_mask=extended_attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        query_layer_8 = key_layer_8 = value_layer_8 = None
        attention_output_25 = attention_output_24.transpose(1, 2)
        attention_output_24 = None
        attention_output_26 = attention_output_25.reshape(1, 12, 1024)
        attention_output_25 = None
        projected_context_layer_8 = torch._C._nn.linear(
            attention_output_26,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_bias_,
        )
        attention_output_26 = None
        projected_context_layer_dropout_8 = torch.nn.functional.dropout(
            projected_context_layer_8, 0.1, False, False
        )
        projected_context_layer_8 = None
        add_17 = hidden_states_8 + projected_context_layer_dropout_8
        hidden_states_8 = projected_context_layer_dropout_8 = None
        layernormed_context_layer_8 = torch.nn.functional.layer_norm(
            add_17,
            (1024,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_17 = None
        ffn_output_24 = torch._C._nn.linear(
            layernormed_context_layer_8,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_bias_,
        )
        ffn_output_25 = torch._C._nn.gelu(ffn_output_24)
        ffn_output_24 = None
        ffn_output_26 = torch._C._nn.linear(
            ffn_output_25,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_bias_,
        )
        ffn_output_25 = None
        add_18 = ffn_output_26 + layernormed_context_layer_8
        ffn_output_26 = layernormed_context_layer_8 = None
        hidden_states_9 = torch.nn.functional.layer_norm(
            add_18,
            (1024,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_bias_,
            1e-12,
        )
        add_18 = None
        linear_55 = torch._C._nn.linear(
            hidden_states_9,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_bias_,
        )
        view_27 = linear_55.view(1, -1, 16, 64)
        linear_55 = None
        query_layer_9 = view_27.transpose(1, 2)
        view_27 = None
        linear_56 = torch._C._nn.linear(
            hidden_states_9,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_bias_,
        )
        view_28 = linear_56.view(1, -1, 16, 64)
        linear_56 = None
        key_layer_9 = view_28.transpose(1, 2)
        view_28 = None
        linear_57 = torch._C._nn.linear(
            hidden_states_9,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_bias_,
        )
        view_29 = linear_57.view(1, -1, 16, 64)
        linear_57 = None
        value_layer_9 = view_29.transpose(1, 2)
        view_29 = None
        attention_output_27 = torch._C._nn.scaled_dot_product_attention(
            query=query_layer_9,
            key=key_layer_9,
            value=value_layer_9,
            attn_mask=extended_attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        query_layer_9 = key_layer_9 = value_layer_9 = None
        attention_output_28 = attention_output_27.transpose(1, 2)
        attention_output_27 = None
        attention_output_29 = attention_output_28.reshape(1, 12, 1024)
        attention_output_28 = None
        projected_context_layer_9 = torch._C._nn.linear(
            attention_output_29,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_bias_,
        )
        attention_output_29 = None
        projected_context_layer_dropout_9 = torch.nn.functional.dropout(
            projected_context_layer_9, 0.1, False, False
        )
        projected_context_layer_9 = None
        add_19 = hidden_states_9 + projected_context_layer_dropout_9
        hidden_states_9 = projected_context_layer_dropout_9 = None
        layernormed_context_layer_9 = torch.nn.functional.layer_norm(
            add_19,
            (1024,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_19 = None
        ffn_output_27 = torch._C._nn.linear(
            layernormed_context_layer_9,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_bias_,
        )
        ffn_output_28 = torch._C._nn.gelu(ffn_output_27)
        ffn_output_27 = None
        ffn_output_29 = torch._C._nn.linear(
            ffn_output_28,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_bias_,
        )
        ffn_output_28 = None
        add_20 = ffn_output_29 + layernormed_context_layer_9
        ffn_output_29 = layernormed_context_layer_9 = None
        hidden_states_10 = torch.nn.functional.layer_norm(
            add_20,
            (1024,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_bias_,
            1e-12,
        )
        add_20 = None
        linear_61 = torch._C._nn.linear(
            hidden_states_10,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_bias_,
        )
        view_30 = linear_61.view(1, -1, 16, 64)
        linear_61 = None
        query_layer_10 = view_30.transpose(1, 2)
        view_30 = None
        linear_62 = torch._C._nn.linear(
            hidden_states_10,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_bias_,
        )
        view_31 = linear_62.view(1, -1, 16, 64)
        linear_62 = None
        key_layer_10 = view_31.transpose(1, 2)
        view_31 = None
        linear_63 = torch._C._nn.linear(
            hidden_states_10,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_bias_,
        )
        view_32 = linear_63.view(1, -1, 16, 64)
        linear_63 = None
        value_layer_10 = view_32.transpose(1, 2)
        view_32 = None
        attention_output_30 = torch._C._nn.scaled_dot_product_attention(
            query=query_layer_10,
            key=key_layer_10,
            value=value_layer_10,
            attn_mask=extended_attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        query_layer_10 = key_layer_10 = value_layer_10 = None
        attention_output_31 = attention_output_30.transpose(1, 2)
        attention_output_30 = None
        attention_output_32 = attention_output_31.reshape(1, 12, 1024)
        attention_output_31 = None
        projected_context_layer_10 = torch._C._nn.linear(
            attention_output_32,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_bias_,
        )
        attention_output_32 = None
        projected_context_layer_dropout_10 = torch.nn.functional.dropout(
            projected_context_layer_10, 0.1, False, False
        )
        projected_context_layer_10 = None
        add_21 = hidden_states_10 + projected_context_layer_dropout_10
        hidden_states_10 = projected_context_layer_dropout_10 = None
        layernormed_context_layer_10 = torch.nn.functional.layer_norm(
            add_21,
            (1024,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_21 = None
        ffn_output_30 = torch._C._nn.linear(
            layernormed_context_layer_10,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_bias_,
        )
        ffn_output_31 = torch._C._nn.gelu(ffn_output_30)
        ffn_output_30 = None
        ffn_output_32 = torch._C._nn.linear(
            ffn_output_31,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_bias_,
        )
        ffn_output_31 = None
        add_22 = ffn_output_32 + layernormed_context_layer_10
        ffn_output_32 = layernormed_context_layer_10 = None
        hidden_states_11 = torch.nn.functional.layer_norm(
            add_22,
            (1024,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_bias_,
            1e-12,
        )
        add_22 = None
        linear_67 = torch._C._nn.linear(
            hidden_states_11,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_bias_,
        )
        view_33 = linear_67.view(1, -1, 16, 64)
        linear_67 = None
        query_layer_11 = view_33.transpose(1, 2)
        view_33 = None
        linear_68 = torch._C._nn.linear(
            hidden_states_11,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_bias_,
        )
        view_34 = linear_68.view(1, -1, 16, 64)
        linear_68 = None
        key_layer_11 = view_34.transpose(1, 2)
        view_34 = None
        linear_69 = torch._C._nn.linear(
            hidden_states_11,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_bias_,
        )
        view_35 = linear_69.view(1, -1, 16, 64)
        linear_69 = None
        value_layer_11 = view_35.transpose(1, 2)
        view_35 = None
        attention_output_33 = torch._C._nn.scaled_dot_product_attention(
            query=query_layer_11,
            key=key_layer_11,
            value=value_layer_11,
            attn_mask=extended_attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        query_layer_11 = key_layer_11 = value_layer_11 = None
        attention_output_34 = attention_output_33.transpose(1, 2)
        attention_output_33 = None
        attention_output_35 = attention_output_34.reshape(1, 12, 1024)
        attention_output_34 = None
        projected_context_layer_11 = torch._C._nn.linear(
            attention_output_35,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_bias_,
        )
        attention_output_35 = None
        projected_context_layer_dropout_11 = torch.nn.functional.dropout(
            projected_context_layer_11, 0.1, False, False
        )
        projected_context_layer_11 = None
        add_23 = hidden_states_11 + projected_context_layer_dropout_11
        hidden_states_11 = projected_context_layer_dropout_11 = None
        layernormed_context_layer_11 = torch.nn.functional.layer_norm(
            add_23,
            (1024,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_23 = None
        ffn_output_33 = torch._C._nn.linear(
            layernormed_context_layer_11,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_bias_,
        )
        ffn_output_34 = torch._C._nn.gelu(ffn_output_33)
        ffn_output_33 = None
        ffn_output_35 = torch._C._nn.linear(
            ffn_output_34,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_bias_,
        )
        ffn_output_34 = None
        add_24 = ffn_output_35 + layernormed_context_layer_11
        ffn_output_35 = layernormed_context_layer_11 = None
        hidden_states_12 = torch.nn.functional.layer_norm(
            add_24,
            (1024,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_bias_,
            1e-12,
        )
        add_24 = None
        linear_73 = torch._C._nn.linear(
            hidden_states_12,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_bias_,
        )
        view_36 = linear_73.view(1, -1, 16, 64)
        linear_73 = None
        query_layer_12 = view_36.transpose(1, 2)
        view_36 = None
        linear_74 = torch._C._nn.linear(
            hidden_states_12,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_bias_,
        )
        view_37 = linear_74.view(1, -1, 16, 64)
        linear_74 = None
        key_layer_12 = view_37.transpose(1, 2)
        view_37 = None
        linear_75 = torch._C._nn.linear(
            hidden_states_12,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_bias_,
        )
        view_38 = linear_75.view(1, -1, 16, 64)
        linear_75 = None
        value_layer_12 = view_38.transpose(1, 2)
        view_38 = None
        attention_output_36 = torch._C._nn.scaled_dot_product_attention(
            query=query_layer_12,
            key=key_layer_12,
            value=value_layer_12,
            attn_mask=extended_attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        query_layer_12 = key_layer_12 = value_layer_12 = None
        attention_output_37 = attention_output_36.transpose(1, 2)
        attention_output_36 = None
        attention_output_38 = attention_output_37.reshape(1, 12, 1024)
        attention_output_37 = None
        projected_context_layer_12 = torch._C._nn.linear(
            attention_output_38,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_bias_,
        )
        attention_output_38 = None
        projected_context_layer_dropout_12 = torch.nn.functional.dropout(
            projected_context_layer_12, 0.1, False, False
        )
        projected_context_layer_12 = None
        add_25 = hidden_states_12 + projected_context_layer_dropout_12
        hidden_states_12 = projected_context_layer_dropout_12 = None
        layernormed_context_layer_12 = torch.nn.functional.layer_norm(
            add_25,
            (1024,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_25 = None
        ffn_output_36 = torch._C._nn.linear(
            layernormed_context_layer_12,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_bias_,
        )
        ffn_output_37 = torch._C._nn.gelu(ffn_output_36)
        ffn_output_36 = None
        ffn_output_38 = torch._C._nn.linear(
            ffn_output_37,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_bias_,
        )
        ffn_output_37 = None
        add_26 = ffn_output_38 + layernormed_context_layer_12
        ffn_output_38 = layernormed_context_layer_12 = None
        hidden_states_13 = torch.nn.functional.layer_norm(
            add_26,
            (1024,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_bias_,
            1e-12,
        )
        add_26 = None
        linear_79 = torch._C._nn.linear(
            hidden_states_13,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_bias_,
        )
        view_39 = linear_79.view(1, -1, 16, 64)
        linear_79 = None
        query_layer_13 = view_39.transpose(1, 2)
        view_39 = None
        linear_80 = torch._C._nn.linear(
            hidden_states_13,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_bias_,
        )
        view_40 = linear_80.view(1, -1, 16, 64)
        linear_80 = None
        key_layer_13 = view_40.transpose(1, 2)
        view_40 = None
        linear_81 = torch._C._nn.linear(
            hidden_states_13,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_bias_,
        )
        view_41 = linear_81.view(1, -1, 16, 64)
        linear_81 = None
        value_layer_13 = view_41.transpose(1, 2)
        view_41 = None
        attention_output_39 = torch._C._nn.scaled_dot_product_attention(
            query=query_layer_13,
            key=key_layer_13,
            value=value_layer_13,
            attn_mask=extended_attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        query_layer_13 = key_layer_13 = value_layer_13 = None
        attention_output_40 = attention_output_39.transpose(1, 2)
        attention_output_39 = None
        attention_output_41 = attention_output_40.reshape(1, 12, 1024)
        attention_output_40 = None
        projected_context_layer_13 = torch._C._nn.linear(
            attention_output_41,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_bias_,
        )
        attention_output_41 = None
        projected_context_layer_dropout_13 = torch.nn.functional.dropout(
            projected_context_layer_13, 0.1, False, False
        )
        projected_context_layer_13 = None
        add_27 = hidden_states_13 + projected_context_layer_dropout_13
        hidden_states_13 = projected_context_layer_dropout_13 = None
        layernormed_context_layer_13 = torch.nn.functional.layer_norm(
            add_27,
            (1024,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_27 = None
        ffn_output_39 = torch._C._nn.linear(
            layernormed_context_layer_13,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_bias_,
        )
        ffn_output_40 = torch._C._nn.gelu(ffn_output_39)
        ffn_output_39 = None
        ffn_output_41 = torch._C._nn.linear(
            ffn_output_40,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_bias_,
        )
        ffn_output_40 = None
        add_28 = ffn_output_41 + layernormed_context_layer_13
        ffn_output_41 = layernormed_context_layer_13 = None
        hidden_states_14 = torch.nn.functional.layer_norm(
            add_28,
            (1024,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_bias_,
            1e-12,
        )
        add_28 = None
        linear_85 = torch._C._nn.linear(
            hidden_states_14,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_bias_,
        )
        view_42 = linear_85.view(1, -1, 16, 64)
        linear_85 = None
        query_layer_14 = view_42.transpose(1, 2)
        view_42 = None
        linear_86 = torch._C._nn.linear(
            hidden_states_14,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_bias_,
        )
        view_43 = linear_86.view(1, -1, 16, 64)
        linear_86 = None
        key_layer_14 = view_43.transpose(1, 2)
        view_43 = None
        linear_87 = torch._C._nn.linear(
            hidden_states_14,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_bias_,
        )
        view_44 = linear_87.view(1, -1, 16, 64)
        linear_87 = None
        value_layer_14 = view_44.transpose(1, 2)
        view_44 = None
        attention_output_42 = torch._C._nn.scaled_dot_product_attention(
            query=query_layer_14,
            key=key_layer_14,
            value=value_layer_14,
            attn_mask=extended_attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        query_layer_14 = key_layer_14 = value_layer_14 = None
        attention_output_43 = attention_output_42.transpose(1, 2)
        attention_output_42 = None
        attention_output_44 = attention_output_43.reshape(1, 12, 1024)
        attention_output_43 = None
        projected_context_layer_14 = torch._C._nn.linear(
            attention_output_44,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_bias_,
        )
        attention_output_44 = None
        projected_context_layer_dropout_14 = torch.nn.functional.dropout(
            projected_context_layer_14, 0.1, False, False
        )
        projected_context_layer_14 = None
        add_29 = hidden_states_14 + projected_context_layer_dropout_14
        hidden_states_14 = projected_context_layer_dropout_14 = None
        layernormed_context_layer_14 = torch.nn.functional.layer_norm(
            add_29,
            (1024,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_29 = None
        ffn_output_42 = torch._C._nn.linear(
            layernormed_context_layer_14,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_bias_,
        )
        ffn_output_43 = torch._C._nn.gelu(ffn_output_42)
        ffn_output_42 = None
        ffn_output_44 = torch._C._nn.linear(
            ffn_output_43,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_bias_,
        )
        ffn_output_43 = None
        add_30 = ffn_output_44 + layernormed_context_layer_14
        ffn_output_44 = layernormed_context_layer_14 = None
        hidden_states_15 = torch.nn.functional.layer_norm(
            add_30,
            (1024,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_bias_,
            1e-12,
        )
        add_30 = None
        linear_91 = torch._C._nn.linear(
            hidden_states_15,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_bias_,
        )
        view_45 = linear_91.view(1, -1, 16, 64)
        linear_91 = None
        query_layer_15 = view_45.transpose(1, 2)
        view_45 = None
        linear_92 = torch._C._nn.linear(
            hidden_states_15,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_bias_,
        )
        view_46 = linear_92.view(1, -1, 16, 64)
        linear_92 = None
        key_layer_15 = view_46.transpose(1, 2)
        view_46 = None
        linear_93 = torch._C._nn.linear(
            hidden_states_15,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_bias_,
        )
        view_47 = linear_93.view(1, -1, 16, 64)
        linear_93 = None
        value_layer_15 = view_47.transpose(1, 2)
        view_47 = None
        attention_output_45 = torch._C._nn.scaled_dot_product_attention(
            query=query_layer_15,
            key=key_layer_15,
            value=value_layer_15,
            attn_mask=extended_attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        query_layer_15 = key_layer_15 = value_layer_15 = None
        attention_output_46 = attention_output_45.transpose(1, 2)
        attention_output_45 = None
        attention_output_47 = attention_output_46.reshape(1, 12, 1024)
        attention_output_46 = None
        projected_context_layer_15 = torch._C._nn.linear(
            attention_output_47,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_bias_,
        )
        attention_output_47 = None
        projected_context_layer_dropout_15 = torch.nn.functional.dropout(
            projected_context_layer_15, 0.1, False, False
        )
        projected_context_layer_15 = None
        add_31 = hidden_states_15 + projected_context_layer_dropout_15
        hidden_states_15 = projected_context_layer_dropout_15 = None
        layernormed_context_layer_15 = torch.nn.functional.layer_norm(
            add_31,
            (1024,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_31 = None
        ffn_output_45 = torch._C._nn.linear(
            layernormed_context_layer_15,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_bias_,
        )
        ffn_output_46 = torch._C._nn.gelu(ffn_output_45)
        ffn_output_45 = None
        ffn_output_47 = torch._C._nn.linear(
            ffn_output_46,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_bias_,
        )
        ffn_output_46 = None
        add_32 = ffn_output_47 + layernormed_context_layer_15
        ffn_output_47 = layernormed_context_layer_15 = None
        hidden_states_16 = torch.nn.functional.layer_norm(
            add_32,
            (1024,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_bias_,
            1e-12,
        )
        add_32 = None
        linear_97 = torch._C._nn.linear(
            hidden_states_16,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_bias_,
        )
        view_48 = linear_97.view(1, -1, 16, 64)
        linear_97 = None
        query_layer_16 = view_48.transpose(1, 2)
        view_48 = None
        linear_98 = torch._C._nn.linear(
            hidden_states_16,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_bias_,
        )
        view_49 = linear_98.view(1, -1, 16, 64)
        linear_98 = None
        key_layer_16 = view_49.transpose(1, 2)
        view_49 = None
        linear_99 = torch._C._nn.linear(
            hidden_states_16,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_bias_,
        )
        view_50 = linear_99.view(1, -1, 16, 64)
        linear_99 = None
        value_layer_16 = view_50.transpose(1, 2)
        view_50 = None
        attention_output_48 = torch._C._nn.scaled_dot_product_attention(
            query=query_layer_16,
            key=key_layer_16,
            value=value_layer_16,
            attn_mask=extended_attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        query_layer_16 = key_layer_16 = value_layer_16 = None
        attention_output_49 = attention_output_48.transpose(1, 2)
        attention_output_48 = None
        attention_output_50 = attention_output_49.reshape(1, 12, 1024)
        attention_output_49 = None
        projected_context_layer_16 = torch._C._nn.linear(
            attention_output_50,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_bias_,
        )
        attention_output_50 = None
        projected_context_layer_dropout_16 = torch.nn.functional.dropout(
            projected_context_layer_16, 0.1, False, False
        )
        projected_context_layer_16 = None
        add_33 = hidden_states_16 + projected_context_layer_dropout_16
        hidden_states_16 = projected_context_layer_dropout_16 = None
        layernormed_context_layer_16 = torch.nn.functional.layer_norm(
            add_33,
            (1024,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_33 = None
        ffn_output_48 = torch._C._nn.linear(
            layernormed_context_layer_16,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_bias_,
        )
        ffn_output_49 = torch._C._nn.gelu(ffn_output_48)
        ffn_output_48 = None
        ffn_output_50 = torch._C._nn.linear(
            ffn_output_49,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_bias_,
        )
        ffn_output_49 = None
        add_34 = ffn_output_50 + layernormed_context_layer_16
        ffn_output_50 = layernormed_context_layer_16 = None
        hidden_states_17 = torch.nn.functional.layer_norm(
            add_34,
            (1024,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_bias_,
            1e-12,
        )
        add_34 = None
        linear_103 = torch._C._nn.linear(
            hidden_states_17,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_bias_,
        )
        view_51 = linear_103.view(1, -1, 16, 64)
        linear_103 = None
        query_layer_17 = view_51.transpose(1, 2)
        view_51 = None
        linear_104 = torch._C._nn.linear(
            hidden_states_17,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_bias_,
        )
        view_52 = linear_104.view(1, -1, 16, 64)
        linear_104 = None
        key_layer_17 = view_52.transpose(1, 2)
        view_52 = None
        linear_105 = torch._C._nn.linear(
            hidden_states_17,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_bias_,
        )
        view_53 = linear_105.view(1, -1, 16, 64)
        linear_105 = None
        value_layer_17 = view_53.transpose(1, 2)
        view_53 = None
        attention_output_51 = torch._C._nn.scaled_dot_product_attention(
            query=query_layer_17,
            key=key_layer_17,
            value=value_layer_17,
            attn_mask=extended_attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        query_layer_17 = key_layer_17 = value_layer_17 = None
        attention_output_52 = attention_output_51.transpose(1, 2)
        attention_output_51 = None
        attention_output_53 = attention_output_52.reshape(1, 12, 1024)
        attention_output_52 = None
        projected_context_layer_17 = torch._C._nn.linear(
            attention_output_53,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_bias_,
        )
        attention_output_53 = None
        projected_context_layer_dropout_17 = torch.nn.functional.dropout(
            projected_context_layer_17, 0.1, False, False
        )
        projected_context_layer_17 = None
        add_35 = hidden_states_17 + projected_context_layer_dropout_17
        hidden_states_17 = projected_context_layer_dropout_17 = None
        layernormed_context_layer_17 = torch.nn.functional.layer_norm(
            add_35,
            (1024,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_35 = None
        ffn_output_51 = torch._C._nn.linear(
            layernormed_context_layer_17,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_bias_,
        )
        ffn_output_52 = torch._C._nn.gelu(ffn_output_51)
        ffn_output_51 = None
        ffn_output_53 = torch._C._nn.linear(
            ffn_output_52,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_bias_,
        )
        ffn_output_52 = None
        add_36 = ffn_output_53 + layernormed_context_layer_17
        ffn_output_53 = layernormed_context_layer_17 = None
        hidden_states_18 = torch.nn.functional.layer_norm(
            add_36,
            (1024,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_bias_,
            1e-12,
        )
        add_36 = None
        linear_109 = torch._C._nn.linear(
            hidden_states_18,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_bias_,
        )
        view_54 = linear_109.view(1, -1, 16, 64)
        linear_109 = None
        query_layer_18 = view_54.transpose(1, 2)
        view_54 = None
        linear_110 = torch._C._nn.linear(
            hidden_states_18,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_bias_,
        )
        view_55 = linear_110.view(1, -1, 16, 64)
        linear_110 = None
        key_layer_18 = view_55.transpose(1, 2)
        view_55 = None
        linear_111 = torch._C._nn.linear(
            hidden_states_18,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_bias_,
        )
        view_56 = linear_111.view(1, -1, 16, 64)
        linear_111 = None
        value_layer_18 = view_56.transpose(1, 2)
        view_56 = None
        attention_output_54 = torch._C._nn.scaled_dot_product_attention(
            query=query_layer_18,
            key=key_layer_18,
            value=value_layer_18,
            attn_mask=extended_attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        query_layer_18 = key_layer_18 = value_layer_18 = None
        attention_output_55 = attention_output_54.transpose(1, 2)
        attention_output_54 = None
        attention_output_56 = attention_output_55.reshape(1, 12, 1024)
        attention_output_55 = None
        projected_context_layer_18 = torch._C._nn.linear(
            attention_output_56,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_bias_,
        )
        attention_output_56 = None
        projected_context_layer_dropout_18 = torch.nn.functional.dropout(
            projected_context_layer_18, 0.1, False, False
        )
        projected_context_layer_18 = None
        add_37 = hidden_states_18 + projected_context_layer_dropout_18
        hidden_states_18 = projected_context_layer_dropout_18 = None
        layernormed_context_layer_18 = torch.nn.functional.layer_norm(
            add_37,
            (1024,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_37 = None
        ffn_output_54 = torch._C._nn.linear(
            layernormed_context_layer_18,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_bias_,
        )
        ffn_output_55 = torch._C._nn.gelu(ffn_output_54)
        ffn_output_54 = None
        ffn_output_56 = torch._C._nn.linear(
            ffn_output_55,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_bias_,
        )
        ffn_output_55 = None
        add_38 = ffn_output_56 + layernormed_context_layer_18
        ffn_output_56 = layernormed_context_layer_18 = None
        hidden_states_19 = torch.nn.functional.layer_norm(
            add_38,
            (1024,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_bias_,
            1e-12,
        )
        add_38 = None
        linear_115 = torch._C._nn.linear(
            hidden_states_19,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_bias_,
        )
        view_57 = linear_115.view(1, -1, 16, 64)
        linear_115 = None
        query_layer_19 = view_57.transpose(1, 2)
        view_57 = None
        linear_116 = torch._C._nn.linear(
            hidden_states_19,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_bias_,
        )
        view_58 = linear_116.view(1, -1, 16, 64)
        linear_116 = None
        key_layer_19 = view_58.transpose(1, 2)
        view_58 = None
        linear_117 = torch._C._nn.linear(
            hidden_states_19,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_bias_,
        )
        view_59 = linear_117.view(1, -1, 16, 64)
        linear_117 = None
        value_layer_19 = view_59.transpose(1, 2)
        view_59 = None
        attention_output_57 = torch._C._nn.scaled_dot_product_attention(
            query=query_layer_19,
            key=key_layer_19,
            value=value_layer_19,
            attn_mask=extended_attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        query_layer_19 = key_layer_19 = value_layer_19 = None
        attention_output_58 = attention_output_57.transpose(1, 2)
        attention_output_57 = None
        attention_output_59 = attention_output_58.reshape(1, 12, 1024)
        attention_output_58 = None
        projected_context_layer_19 = torch._C._nn.linear(
            attention_output_59,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_bias_,
        )
        attention_output_59 = None
        projected_context_layer_dropout_19 = torch.nn.functional.dropout(
            projected_context_layer_19, 0.1, False, False
        )
        projected_context_layer_19 = None
        add_39 = hidden_states_19 + projected_context_layer_dropout_19
        hidden_states_19 = projected_context_layer_dropout_19 = None
        layernormed_context_layer_19 = torch.nn.functional.layer_norm(
            add_39,
            (1024,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_39 = None
        ffn_output_57 = torch._C._nn.linear(
            layernormed_context_layer_19,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_bias_,
        )
        ffn_output_58 = torch._C._nn.gelu(ffn_output_57)
        ffn_output_57 = None
        ffn_output_59 = torch._C._nn.linear(
            ffn_output_58,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_bias_,
        )
        ffn_output_58 = None
        add_40 = ffn_output_59 + layernormed_context_layer_19
        ffn_output_59 = layernormed_context_layer_19 = None
        hidden_states_20 = torch.nn.functional.layer_norm(
            add_40,
            (1024,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_bias_,
            1e-12,
        )
        add_40 = None
        linear_121 = torch._C._nn.linear(
            hidden_states_20,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_bias_,
        )
        view_60 = linear_121.view(1, -1, 16, 64)
        linear_121 = None
        query_layer_20 = view_60.transpose(1, 2)
        view_60 = None
        linear_122 = torch._C._nn.linear(
            hidden_states_20,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_bias_,
        )
        view_61 = linear_122.view(1, -1, 16, 64)
        linear_122 = None
        key_layer_20 = view_61.transpose(1, 2)
        view_61 = None
        linear_123 = torch._C._nn.linear(
            hidden_states_20,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_bias_,
        )
        view_62 = linear_123.view(1, -1, 16, 64)
        linear_123 = None
        value_layer_20 = view_62.transpose(1, 2)
        view_62 = None
        attention_output_60 = torch._C._nn.scaled_dot_product_attention(
            query=query_layer_20,
            key=key_layer_20,
            value=value_layer_20,
            attn_mask=extended_attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        query_layer_20 = key_layer_20 = value_layer_20 = None
        attention_output_61 = attention_output_60.transpose(1, 2)
        attention_output_60 = None
        attention_output_62 = attention_output_61.reshape(1, 12, 1024)
        attention_output_61 = None
        projected_context_layer_20 = torch._C._nn.linear(
            attention_output_62,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_bias_,
        )
        attention_output_62 = None
        projected_context_layer_dropout_20 = torch.nn.functional.dropout(
            projected_context_layer_20, 0.1, False, False
        )
        projected_context_layer_20 = None
        add_41 = hidden_states_20 + projected_context_layer_dropout_20
        hidden_states_20 = projected_context_layer_dropout_20 = None
        layernormed_context_layer_20 = torch.nn.functional.layer_norm(
            add_41,
            (1024,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_41 = None
        ffn_output_60 = torch._C._nn.linear(
            layernormed_context_layer_20,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_bias_,
        )
        ffn_output_61 = torch._C._nn.gelu(ffn_output_60)
        ffn_output_60 = None
        ffn_output_62 = torch._C._nn.linear(
            ffn_output_61,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_bias_,
        )
        ffn_output_61 = None
        add_42 = ffn_output_62 + layernormed_context_layer_20
        ffn_output_62 = layernormed_context_layer_20 = None
        hidden_states_21 = torch.nn.functional.layer_norm(
            add_42,
            (1024,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_bias_,
            1e-12,
        )
        add_42 = None
        linear_127 = torch._C._nn.linear(
            hidden_states_21,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_bias_,
        )
        view_63 = linear_127.view(1, -1, 16, 64)
        linear_127 = None
        query_layer_21 = view_63.transpose(1, 2)
        view_63 = None
        linear_128 = torch._C._nn.linear(
            hidden_states_21,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_bias_,
        )
        view_64 = linear_128.view(1, -1, 16, 64)
        linear_128 = None
        key_layer_21 = view_64.transpose(1, 2)
        view_64 = None
        linear_129 = torch._C._nn.linear(
            hidden_states_21,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_bias_,
        )
        view_65 = linear_129.view(1, -1, 16, 64)
        linear_129 = None
        value_layer_21 = view_65.transpose(1, 2)
        view_65 = None
        attention_output_63 = torch._C._nn.scaled_dot_product_attention(
            query=query_layer_21,
            key=key_layer_21,
            value=value_layer_21,
            attn_mask=extended_attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        query_layer_21 = key_layer_21 = value_layer_21 = None
        attention_output_64 = attention_output_63.transpose(1, 2)
        attention_output_63 = None
        attention_output_65 = attention_output_64.reshape(1, 12, 1024)
        attention_output_64 = None
        projected_context_layer_21 = torch._C._nn.linear(
            attention_output_65,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_bias_,
        )
        attention_output_65 = None
        projected_context_layer_dropout_21 = torch.nn.functional.dropout(
            projected_context_layer_21, 0.1, False, False
        )
        projected_context_layer_21 = None
        add_43 = hidden_states_21 + projected_context_layer_dropout_21
        hidden_states_21 = projected_context_layer_dropout_21 = None
        layernormed_context_layer_21 = torch.nn.functional.layer_norm(
            add_43,
            (1024,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_43 = None
        ffn_output_63 = torch._C._nn.linear(
            layernormed_context_layer_21,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_bias_,
        )
        ffn_output_64 = torch._C._nn.gelu(ffn_output_63)
        ffn_output_63 = None
        ffn_output_65 = torch._C._nn.linear(
            ffn_output_64,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_bias_,
        )
        ffn_output_64 = None
        add_44 = ffn_output_65 + layernormed_context_layer_21
        ffn_output_65 = layernormed_context_layer_21 = None
        hidden_states_22 = torch.nn.functional.layer_norm(
            add_44,
            (1024,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_bias_,
            1e-12,
        )
        add_44 = None
        linear_133 = torch._C._nn.linear(
            hidden_states_22,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_bias_,
        )
        view_66 = linear_133.view(1, -1, 16, 64)
        linear_133 = None
        query_layer_22 = view_66.transpose(1, 2)
        view_66 = None
        linear_134 = torch._C._nn.linear(
            hidden_states_22,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_bias_,
        )
        view_67 = linear_134.view(1, -1, 16, 64)
        linear_134 = None
        key_layer_22 = view_67.transpose(1, 2)
        view_67 = None
        linear_135 = torch._C._nn.linear(
            hidden_states_22,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_bias_,
        )
        view_68 = linear_135.view(1, -1, 16, 64)
        linear_135 = None
        value_layer_22 = view_68.transpose(1, 2)
        view_68 = None
        attention_output_66 = torch._C._nn.scaled_dot_product_attention(
            query=query_layer_22,
            key=key_layer_22,
            value=value_layer_22,
            attn_mask=extended_attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        query_layer_22 = key_layer_22 = value_layer_22 = None
        attention_output_67 = attention_output_66.transpose(1, 2)
        attention_output_66 = None
        attention_output_68 = attention_output_67.reshape(1, 12, 1024)
        attention_output_67 = None
        projected_context_layer_22 = torch._C._nn.linear(
            attention_output_68,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_bias_,
        )
        attention_output_68 = None
        projected_context_layer_dropout_22 = torch.nn.functional.dropout(
            projected_context_layer_22, 0.1, False, False
        )
        projected_context_layer_22 = None
        add_45 = hidden_states_22 + projected_context_layer_dropout_22
        hidden_states_22 = projected_context_layer_dropout_22 = None
        layernormed_context_layer_22 = torch.nn.functional.layer_norm(
            add_45,
            (1024,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_45 = None
        ffn_output_66 = torch._C._nn.linear(
            layernormed_context_layer_22,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_bias_,
        )
        ffn_output_67 = torch._C._nn.gelu(ffn_output_66)
        ffn_output_66 = None
        ffn_output_68 = torch._C._nn.linear(
            ffn_output_67,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_bias_,
        )
        ffn_output_67 = None
        add_46 = ffn_output_68 + layernormed_context_layer_22
        ffn_output_68 = layernormed_context_layer_22 = None
        hidden_states_23 = torch.nn.functional.layer_norm(
            add_46,
            (1024,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_bias_,
            1e-12,
        )
        add_46 = None
        linear_139 = torch._C._nn.linear(
            hidden_states_23,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_bias_ = (None)
        view_69 = linear_139.view(1, -1, 16, 64)
        linear_139 = None
        query_layer_23 = view_69.transpose(1, 2)
        view_69 = None
        linear_140 = torch._C._nn.linear(
            hidden_states_23,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_bias_ = (None)
        view_70 = linear_140.view(1, -1, 16, 64)
        linear_140 = None
        key_layer_23 = view_70.transpose(1, 2)
        view_70 = None
        linear_141 = torch._C._nn.linear(
            hidden_states_23,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_bias_ = (None)
        view_71 = linear_141.view(1, -1, 16, 64)
        linear_141 = None
        value_layer_23 = view_71.transpose(1, 2)
        view_71 = None
        attention_output_69 = torch._C._nn.scaled_dot_product_attention(
            query=query_layer_23,
            key=key_layer_23,
            value=value_layer_23,
            attn_mask=extended_attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        query_layer_23 = key_layer_23 = value_layer_23 = extended_attention_mask = None
        attention_output_70 = attention_output_69.transpose(1, 2)
        attention_output_69 = None
        attention_output_71 = attention_output_70.reshape(1, 12, 1024)
        attention_output_70 = None
        projected_context_layer_23 = torch._C._nn.linear(
            attention_output_71,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_bias_,
        )
        attention_output_71 = l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_bias_ = (None)
        projected_context_layer_dropout_23 = torch.nn.functional.dropout(
            projected_context_layer_23, 0.1, False, False
        )
        projected_context_layer_23 = None
        add_47 = hidden_states_23 + projected_context_layer_dropout_23
        hidden_states_23 = projected_context_layer_dropout_23 = None
        layernormed_context_layer_23 = torch.nn.functional.layer_norm(
            add_47,
            (1024,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_47 = l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_bias_ = (None)
        ffn_output_69 = torch._C._nn.linear(
            layernormed_context_layer_23,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_bias_,
        )
        l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_weight_ = l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_bias_ = (None)
        ffn_output_70 = torch._C._nn.gelu(ffn_output_69)
        ffn_output_69 = None
        ffn_output_71 = torch._C._nn.linear(
            ffn_output_70,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_bias_,
        )
        ffn_output_70 = l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_weight_ = l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_bias_ = (None)
        add_48 = ffn_output_71 + layernormed_context_layer_23
        ffn_output_71 = layernormed_context_layer_23 = None
        hidden_states_24 = torch.nn.functional.layer_norm(
            add_48,
            (1024,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_bias_,
            1e-12,
        )
        add_48 = l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_bias_ = (None)
        getitem_2 = hidden_states_24[(slice(None, None, None), 0)]
        linear_145 = torch._C._nn.linear(
            getitem_2,
            l_self_modules_pooler_parameters_weight_,
            l_self_modules_pooler_parameters_bias_,
        )
        getitem_2 = (
            l_self_modules_pooler_parameters_weight_
        ) = l_self_modules_pooler_parameters_bias_ = None
        pooled_output = torch.tanh(linear_145)
        linear_145 = None
        return (hidden_states_24, pooled_output)
