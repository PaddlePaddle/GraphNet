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
        embeddings_3 = torch.nn.functional.dropout(embeddings_2, 0, False, False)
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
        view = linear_1.view(1, -1, 64, 64)
        linear_1 = None
        query_layer = view.transpose(1, 2)
        view = None
        linear_2 = torch._C._nn.linear(
            hidden_states,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_bias_,
        )
        view_1 = linear_2.view(1, -1, 64, 64)
        linear_2 = None
        key_layer = view_1.transpose(1, 2)
        view_1 = None
        linear_3 = torch._C._nn.linear(
            hidden_states,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_bias_,
        )
        view_2 = linear_3.view(1, -1, 64, 64)
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
        attention_output_2 = attention_output_1.reshape(1, 12, 4096)
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
            (4096,),
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
            (4096,),
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
        view_3 = linear_7.view(1, -1, 64, 64)
        linear_7 = None
        query_layer_1 = view_3.transpose(1, 2)
        view_3 = None
        linear_8 = torch._C._nn.linear(
            hidden_states_1,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_bias_,
        )
        view_4 = linear_8.view(1, -1, 64, 64)
        linear_8 = None
        key_layer_1 = view_4.transpose(1, 2)
        view_4 = None
        linear_9 = torch._C._nn.linear(
            hidden_states_1,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_bias_,
        )
        view_5 = linear_9.view(1, -1, 64, 64)
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
        attention_output_5 = attention_output_4.reshape(1, 12, 4096)
        attention_output_4 = None
        projected_context_layer_1 = torch._C._nn.linear(
            attention_output_5,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_bias_,
        )
        attention_output_5 = None
        projected_context_layer_dropout_1 = torch.nn.functional.dropout(
            projected_context_layer_1, 0, False, False
        )
        projected_context_layer_1 = None
        add_3 = hidden_states_1 + projected_context_layer_dropout_1
        hidden_states_1 = projected_context_layer_dropout_1 = None
        layernormed_context_layer_1 = torch.nn.functional.layer_norm(
            add_3,
            (4096,),
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
            (4096,),
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
        view_6 = linear_13.view(1, -1, 64, 64)
        linear_13 = None
        query_layer_2 = view_6.transpose(1, 2)
        view_6 = None
        linear_14 = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_bias_,
        )
        view_7 = linear_14.view(1, -1, 64, 64)
        linear_14 = None
        key_layer_2 = view_7.transpose(1, 2)
        view_7 = None
        linear_15 = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_bias_,
        )
        view_8 = linear_15.view(1, -1, 64, 64)
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
        attention_output_8 = attention_output_7.reshape(1, 12, 4096)
        attention_output_7 = None
        projected_context_layer_2 = torch._C._nn.linear(
            attention_output_8,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_bias_,
        )
        attention_output_8 = None
        projected_context_layer_dropout_2 = torch.nn.functional.dropout(
            projected_context_layer_2, 0, False, False
        )
        projected_context_layer_2 = None
        add_5 = hidden_states_2 + projected_context_layer_dropout_2
        hidden_states_2 = projected_context_layer_dropout_2 = None
        layernormed_context_layer_2 = torch.nn.functional.layer_norm(
            add_5,
            (4096,),
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
            (4096,),
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
        view_9 = linear_19.view(1, -1, 64, 64)
        linear_19 = None
        query_layer_3 = view_9.transpose(1, 2)
        view_9 = None
        linear_20 = torch._C._nn.linear(
            hidden_states_3,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_bias_,
        )
        view_10 = linear_20.view(1, -1, 64, 64)
        linear_20 = None
        key_layer_3 = view_10.transpose(1, 2)
        view_10 = None
        linear_21 = torch._C._nn.linear(
            hidden_states_3,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_bias_,
        )
        view_11 = linear_21.view(1, -1, 64, 64)
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
        attention_output_11 = attention_output_10.reshape(1, 12, 4096)
        attention_output_10 = None
        projected_context_layer_3 = torch._C._nn.linear(
            attention_output_11,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_bias_,
        )
        attention_output_11 = None
        projected_context_layer_dropout_3 = torch.nn.functional.dropout(
            projected_context_layer_3, 0, False, False
        )
        projected_context_layer_3 = None
        add_7 = hidden_states_3 + projected_context_layer_dropout_3
        hidden_states_3 = projected_context_layer_dropout_3 = None
        layernormed_context_layer_3 = torch.nn.functional.layer_norm(
            add_7,
            (4096,),
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
            (4096,),
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
        view_12 = linear_25.view(1, -1, 64, 64)
        linear_25 = None
        query_layer_4 = view_12.transpose(1, 2)
        view_12 = None
        linear_26 = torch._C._nn.linear(
            hidden_states_4,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_bias_,
        )
        view_13 = linear_26.view(1, -1, 64, 64)
        linear_26 = None
        key_layer_4 = view_13.transpose(1, 2)
        view_13 = None
        linear_27 = torch._C._nn.linear(
            hidden_states_4,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_bias_,
        )
        view_14 = linear_27.view(1, -1, 64, 64)
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
        attention_output_14 = attention_output_13.reshape(1, 12, 4096)
        attention_output_13 = None
        projected_context_layer_4 = torch._C._nn.linear(
            attention_output_14,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_bias_,
        )
        attention_output_14 = None
        projected_context_layer_dropout_4 = torch.nn.functional.dropout(
            projected_context_layer_4, 0, False, False
        )
        projected_context_layer_4 = None
        add_9 = hidden_states_4 + projected_context_layer_dropout_4
        hidden_states_4 = projected_context_layer_dropout_4 = None
        layernormed_context_layer_4 = torch.nn.functional.layer_norm(
            add_9,
            (4096,),
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
            (4096,),
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
        view_15 = linear_31.view(1, -1, 64, 64)
        linear_31 = None
        query_layer_5 = view_15.transpose(1, 2)
        view_15 = None
        linear_32 = torch._C._nn.linear(
            hidden_states_5,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_bias_,
        )
        view_16 = linear_32.view(1, -1, 64, 64)
        linear_32 = None
        key_layer_5 = view_16.transpose(1, 2)
        view_16 = None
        linear_33 = torch._C._nn.linear(
            hidden_states_5,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_bias_,
        )
        view_17 = linear_33.view(1, -1, 64, 64)
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
        attention_output_17 = attention_output_16.reshape(1, 12, 4096)
        attention_output_16 = None
        projected_context_layer_5 = torch._C._nn.linear(
            attention_output_17,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_bias_,
        )
        attention_output_17 = None
        projected_context_layer_dropout_5 = torch.nn.functional.dropout(
            projected_context_layer_5, 0, False, False
        )
        projected_context_layer_5 = None
        add_11 = hidden_states_5 + projected_context_layer_dropout_5
        hidden_states_5 = projected_context_layer_dropout_5 = None
        layernormed_context_layer_5 = torch.nn.functional.layer_norm(
            add_11,
            (4096,),
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
            (4096,),
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
        view_18 = linear_37.view(1, -1, 64, 64)
        linear_37 = None
        query_layer_6 = view_18.transpose(1, 2)
        view_18 = None
        linear_38 = torch._C._nn.linear(
            hidden_states_6,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_bias_,
        )
        view_19 = linear_38.view(1, -1, 64, 64)
        linear_38 = None
        key_layer_6 = view_19.transpose(1, 2)
        view_19 = None
        linear_39 = torch._C._nn.linear(
            hidden_states_6,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_bias_,
        )
        view_20 = linear_39.view(1, -1, 64, 64)
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
        attention_output_20 = attention_output_19.reshape(1, 12, 4096)
        attention_output_19 = None
        projected_context_layer_6 = torch._C._nn.linear(
            attention_output_20,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_bias_,
        )
        attention_output_20 = None
        projected_context_layer_dropout_6 = torch.nn.functional.dropout(
            projected_context_layer_6, 0, False, False
        )
        projected_context_layer_6 = None
        add_13 = hidden_states_6 + projected_context_layer_dropout_6
        hidden_states_6 = projected_context_layer_dropout_6 = None
        layernormed_context_layer_6 = torch.nn.functional.layer_norm(
            add_13,
            (4096,),
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
            (4096,),
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
        view_21 = linear_43.view(1, -1, 64, 64)
        linear_43 = None
        query_layer_7 = view_21.transpose(1, 2)
        view_21 = None
        linear_44 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_bias_,
        )
        view_22 = linear_44.view(1, -1, 64, 64)
        linear_44 = None
        key_layer_7 = view_22.transpose(1, 2)
        view_22 = None
        linear_45 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_bias_,
        )
        view_23 = linear_45.view(1, -1, 64, 64)
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
        attention_output_23 = attention_output_22.reshape(1, 12, 4096)
        attention_output_22 = None
        projected_context_layer_7 = torch._C._nn.linear(
            attention_output_23,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_bias_,
        )
        attention_output_23 = None
        projected_context_layer_dropout_7 = torch.nn.functional.dropout(
            projected_context_layer_7, 0, False, False
        )
        projected_context_layer_7 = None
        add_15 = hidden_states_7 + projected_context_layer_dropout_7
        hidden_states_7 = projected_context_layer_dropout_7 = None
        layernormed_context_layer_7 = torch.nn.functional.layer_norm(
            add_15,
            (4096,),
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
            (4096,),
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
        view_24 = linear_49.view(1, -1, 64, 64)
        linear_49 = None
        query_layer_8 = view_24.transpose(1, 2)
        view_24 = None
        linear_50 = torch._C._nn.linear(
            hidden_states_8,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_bias_,
        )
        view_25 = linear_50.view(1, -1, 64, 64)
        linear_50 = None
        key_layer_8 = view_25.transpose(1, 2)
        view_25 = None
        linear_51 = torch._C._nn.linear(
            hidden_states_8,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_bias_,
        )
        view_26 = linear_51.view(1, -1, 64, 64)
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
        attention_output_26 = attention_output_25.reshape(1, 12, 4096)
        attention_output_25 = None
        projected_context_layer_8 = torch._C._nn.linear(
            attention_output_26,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_bias_,
        )
        attention_output_26 = None
        projected_context_layer_dropout_8 = torch.nn.functional.dropout(
            projected_context_layer_8, 0, False, False
        )
        projected_context_layer_8 = None
        add_17 = hidden_states_8 + projected_context_layer_dropout_8
        hidden_states_8 = projected_context_layer_dropout_8 = None
        layernormed_context_layer_8 = torch.nn.functional.layer_norm(
            add_17,
            (4096,),
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
            (4096,),
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
        view_27 = linear_55.view(1, -1, 64, 64)
        linear_55 = None
        query_layer_9 = view_27.transpose(1, 2)
        view_27 = None
        linear_56 = torch._C._nn.linear(
            hidden_states_9,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_bias_,
        )
        view_28 = linear_56.view(1, -1, 64, 64)
        linear_56 = None
        key_layer_9 = view_28.transpose(1, 2)
        view_28 = None
        linear_57 = torch._C._nn.linear(
            hidden_states_9,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_bias_,
        )
        view_29 = linear_57.view(1, -1, 64, 64)
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
        attention_output_29 = attention_output_28.reshape(1, 12, 4096)
        attention_output_28 = None
        projected_context_layer_9 = torch._C._nn.linear(
            attention_output_29,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_bias_,
        )
        attention_output_29 = None
        projected_context_layer_dropout_9 = torch.nn.functional.dropout(
            projected_context_layer_9, 0, False, False
        )
        projected_context_layer_9 = None
        add_19 = hidden_states_9 + projected_context_layer_dropout_9
        hidden_states_9 = projected_context_layer_dropout_9 = None
        layernormed_context_layer_9 = torch.nn.functional.layer_norm(
            add_19,
            (4096,),
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
            (4096,),
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
        view_30 = linear_61.view(1, -1, 64, 64)
        linear_61 = None
        query_layer_10 = view_30.transpose(1, 2)
        view_30 = None
        linear_62 = torch._C._nn.linear(
            hidden_states_10,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_bias_,
        )
        view_31 = linear_62.view(1, -1, 64, 64)
        linear_62 = None
        key_layer_10 = view_31.transpose(1, 2)
        view_31 = None
        linear_63 = torch._C._nn.linear(
            hidden_states_10,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_bias_,
        )
        view_32 = linear_63.view(1, -1, 64, 64)
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
        attention_output_32 = attention_output_31.reshape(1, 12, 4096)
        attention_output_31 = None
        projected_context_layer_10 = torch._C._nn.linear(
            attention_output_32,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_bias_,
        )
        attention_output_32 = None
        projected_context_layer_dropout_10 = torch.nn.functional.dropout(
            projected_context_layer_10, 0, False, False
        )
        projected_context_layer_10 = None
        add_21 = hidden_states_10 + projected_context_layer_dropout_10
        hidden_states_10 = projected_context_layer_dropout_10 = None
        layernormed_context_layer_10 = torch.nn.functional.layer_norm(
            add_21,
            (4096,),
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
            (4096,),
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
        l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_bias_ = (None)
        view_33 = linear_67.view(1, -1, 64, 64)
        linear_67 = None
        query_layer_11 = view_33.transpose(1, 2)
        view_33 = None
        linear_68 = torch._C._nn.linear(
            hidden_states_11,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_bias_ = (None)
        view_34 = linear_68.view(1, -1, 64, 64)
        linear_68 = None
        key_layer_11 = view_34.transpose(1, 2)
        view_34 = None
        linear_69 = torch._C._nn.linear(
            hidden_states_11,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_bias_ = (None)
        view_35 = linear_69.view(1, -1, 64, 64)
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
        query_layer_11 = key_layer_11 = value_layer_11 = extended_attention_mask = None
        attention_output_34 = attention_output_33.transpose(1, 2)
        attention_output_33 = None
        attention_output_35 = attention_output_34.reshape(1, 12, 4096)
        attention_output_34 = None
        projected_context_layer_11 = torch._C._nn.linear(
            attention_output_35,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_bias_,
        )
        attention_output_35 = l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_bias_ = (None)
        projected_context_layer_dropout_11 = torch.nn.functional.dropout(
            projected_context_layer_11, 0, False, False
        )
        projected_context_layer_11 = None
        add_23 = hidden_states_11 + projected_context_layer_dropout_11
        hidden_states_11 = projected_context_layer_dropout_11 = None
        layernormed_context_layer_11 = torch.nn.functional.layer_norm(
            add_23,
            (4096,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_23 = l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_layer_norm_parameters_bias_ = (None)
        ffn_output_33 = torch._C._nn.linear(
            layernormed_context_layer_11,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_bias_,
        )
        l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_weight_ = l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_bias_ = (None)
        ffn_output_34 = torch._C._nn.gelu(ffn_output_33)
        ffn_output_33 = None
        ffn_output_35 = torch._C._nn.linear(
            ffn_output_34,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_bias_,
        )
        ffn_output_34 = l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_weight_ = l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_bias_ = (None)
        add_24 = ffn_output_35 + layernormed_context_layer_11
        ffn_output_35 = layernormed_context_layer_11 = None
        hidden_states_12 = torch.nn.functional.layer_norm(
            add_24,
            (4096,),
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_bias_,
            1e-12,
        )
        add_24 = l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_bias_ = (None)
        getitem_2 = hidden_states_12[(slice(None, None, None), 0)]
        linear_73 = torch._C._nn.linear(
            getitem_2,
            l_self_modules_pooler_parameters_weight_,
            l_self_modules_pooler_parameters_bias_,
        )
        getitem_2 = (
            l_self_modules_pooler_parameters_weight_
        ) = l_self_modules_pooler_parameters_bias_ = None
        pooled_output = torch.tanh(linear_73)
        linear_73 = None
        return (hidden_states_12, pooled_output)
