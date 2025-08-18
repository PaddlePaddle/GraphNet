import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_inputs_embeds_: torch.Tensor,
        L_encoder_hidden_states_: torch.Tensor,
        L_encoder_attention_mask_: torch.Tensor,
        L_self_modules_embed_positions_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layernorm_embedding_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layernorm_embedding_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_inputs_embeds_ = L_inputs_embeds_
        l_encoder_hidden_states_ = L_encoder_hidden_states_
        l_encoder_attention_mask_ = L_encoder_attention_mask_
        l_self_modules_embed_positions_parameters_weight_ = (
            L_self_modules_embed_positions_parameters_weight_
        )
        l_self_modules_layernorm_embedding_parameters_weight_ = (
            L_self_modules_layernorm_embedding_parameters_weight_
        )
        l_self_modules_layernorm_embedding_parameters_bias_ = (
            L_self_modules_layernorm_embedding_parameters_bias_
        )
        l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_0_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_0_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_0_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_0_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_0_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_0_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_0_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_0_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_
        )
        l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_1_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_1_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_1_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_1_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_1_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_1_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_1_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_1_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_
        )
        cache_position = torch.arange(0, 11, device=device(type="cuda", index=0))
        causal_mask = torch.full(
            (11, 12),
            fill_value=-3.4028234663852886e38,
            dtype=torch.float32,
            device=device(type="cuda", index=0),
        )
        causal_mask_1 = torch.triu(causal_mask, diagonal=1)
        causal_mask = None
        arange_1 = torch.arange(12, device=device(type="cuda", index=0))
        reshape = cache_position.reshape(-1, 1)
        gt = arange_1 > reshape
        arange_1 = reshape = None
        causal_mask_1 *= gt
        causal_mask_2 = causal_mask_1
        causal_mask_1 = gt = None
        getitem = causal_mask_2[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        causal_mask_2 = None
        causal_mask_3 = getitem.expand(1, 1, -1, -1)
        getitem = None
        getitem_1 = l_encoder_attention_mask_[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        l_encoder_attention_mask_ = None
        expand_1 = getitem_1.expand(1, 1, 11, 11)
        getitem_1 = None
        expanded_mask = expand_1.to(torch.float32)
        expand_1 = None
        tensor = torch.tensor(1.0, dtype=torch.float32)
        inverted_mask = tensor - expanded_mask
        tensor = expanded_mask = None
        to_1 = inverted_mask.to(torch.bool)
        encoder_attention_mask = inverted_mask.masked_fill(to_1, -3.4028234663852886e38)
        inverted_mask = to_1 = None
        position_ids = cache_position.unsqueeze(0)
        cache_position = None
        add = position_ids + 2
        position_ids = None
        positions = torch.nn.functional.embedding(
            add,
            l_self_modules_embed_positions_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        add = l_self_modules_embed_positions_parameters_weight_ = None
        positions_1 = positions.to(device(type="cuda", index=0))
        positions = None
        hidden_states = l_inputs_embeds_ + positions_1
        l_inputs_embeds_ = positions_1 = None
        hidden_states_1 = torch.nn.functional.layer_norm(
            hidden_states,
            (16,),
            l_self_modules_layernorm_embedding_parameters_weight_,
            l_self_modules_layernorm_embedding_parameters_bias_,
            1e-05,
        )
        hidden_states = (
            l_self_modules_layernorm_embedding_parameters_weight_
        ) = l_self_modules_layernorm_embedding_parameters_bias_ = None
        hidden_states_2 = torch.nn.functional.dropout(
            hidden_states_1, p=0.1, training=False
        )
        hidden_states_1 = None
        linear = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view = linear.view(1, 11, -1, 4)
        linear = None
        query_states = view.transpose(1, 2)
        view = None
        key_states = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_1 = key_states.view(1, 11, -1, 4)
        key_states = None
        key_states_1 = view_1.transpose(1, 2)
        view_1 = None
        view_2 = value_states.view(1, 11, -1, 4)
        value_states = None
        value_states_1 = view_2.transpose(1, 2)
        view_2 = None
        attention_mask = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 11, None),
            )
        ]
        query = query_states.contiguous()
        query_states = None
        key = key_states_1.contiguous()
        value = value_states_1.contiguous()
        attn_output = torch._C._nn.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            scale=0.5,
            is_causal=False,
        )
        query = key = value = attention_mask = None
        transpose_3 = attn_output.transpose(1, 2)
        attn_output = None
        attn_output_1 = transpose_3.contiguous()
        transpose_3 = None
        reshape_1 = attn_output_1.reshape(1, 11, -1)
        attn_output_1 = None
        attn_output_2 = reshape_1.contiguous()
        reshape_1 = None
        attn_output_3 = torch._C._nn.linear(
            attn_output_2,
            l_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_2 = l_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_3 = torch.nn.functional.dropout(
            attn_output_3, p=0.1, training=False
        )
        attn_output_3 = None
        hidden_states_4 = hidden_states_2 + hidden_states_3
        hidden_states_2 = hidden_states_3 = None
        hidden_states_5 = torch.nn.functional.layer_norm(
            hidden_states_4,
            (16,),
            l_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_4 = l_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_4 = torch._C._nn.linear(
            hidden_states_5,
            l_self_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        view_3 = linear_4.view(1, 11, -1, 4)
        linear_4 = None
        query_states_1 = view_3.transpose(1, 2)
        view_3 = None
        key_states_2 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_2 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_4 = key_states_2.view(1, 11, -1, 4)
        key_states_2 = None
        key_states_3 = view_4.transpose(1, 2)
        view_4 = None
        view_5 = value_states_2.view(1, 11, -1, 4)
        value_states_2 = None
        value_states_3 = view_5.transpose(1, 2)
        view_5 = None
        attention_mask_1 = encoder_attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 11, None),
            )
        ]
        query_1 = query_states_1.contiguous()
        query_states_1 = None
        key_1 = key_states_3.contiguous()
        value_1 = value_states_3.contiguous()
        attn_output_4 = torch._C._nn.scaled_dot_product_attention(
            query_1,
            key_1,
            value_1,
            attn_mask=attention_mask_1,
            dropout_p=0.0,
            scale=0.5,
            is_causal=False,
        )
        query_1 = key_1 = value_1 = attention_mask_1 = None
        transpose_7 = attn_output_4.transpose(1, 2)
        attn_output_4 = None
        attn_output_5 = transpose_7.contiguous()
        transpose_7 = None
        reshape_2 = attn_output_5.reshape(1, 11, -1)
        attn_output_5 = None
        attn_output_6 = reshape_2.contiguous()
        reshape_2 = None
        attn_output_7 = torch._C._nn.linear(
            attn_output_6,
            l_self_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_6 = l_self_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_6 = torch.nn.functional.dropout(
            attn_output_7, p=0.1, training=False
        )
        attn_output_7 = None
        hidden_states_7 = hidden_states_5 + hidden_states_6
        hidden_states_5 = hidden_states_6 = None
        hidden_states_8 = torch.nn.functional.layer_norm(
            hidden_states_7,
            (16,),
            l_self_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_7 = l_self_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_8 = torch._C._nn.linear(
            hidden_states_8,
            l_self_modules_layers_modules_0_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_0_modules_fc1_parameters_bias_,
        )
        l_self_modules_layers_modules_0_modules_fc1_parameters_weight_ = (
            l_self_modules_layers_modules_0_modules_fc1_parameters_bias_
        ) = None
        hidden_states_9 = torch._C._nn.gelu(linear_8)
        linear_8 = None
        hidden_states_10 = torch.nn.functional.dropout(
            hidden_states_9, p=0.0, training=False
        )
        hidden_states_9 = None
        hidden_states_11 = torch._C._nn.linear(
            hidden_states_10,
            l_self_modules_layers_modules_0_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_0_modules_fc2_parameters_bias_,
        )
        hidden_states_10 = (
            l_self_modules_layers_modules_0_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_0_modules_fc2_parameters_bias_ = None
        hidden_states_12 = torch.nn.functional.dropout(
            hidden_states_11, p=0.1, training=False
        )
        hidden_states_11 = None
        hidden_states_13 = hidden_states_8 + hidden_states_12
        hidden_states_8 = hidden_states_12 = None
        hidden_states_14 = torch.nn.functional.layer_norm(
            hidden_states_13,
            (16,),
            l_self_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_13 = (
            l_self_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_
        ) = (
            l_self_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_
        ) = None
        linear_10 = torch._C._nn.linear(
            hidden_states_14,
            l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_6 = linear_10.view(1, 11, -1, 4)
        linear_10 = None
        query_states_2 = view_6.transpose(1, 2)
        view_6 = None
        key_states_4 = torch._C._nn.linear(
            hidden_states_14,
            l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_4 = torch._C._nn.linear(
            hidden_states_14,
            l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_7 = key_states_4.view(1, 11, -1, 4)
        key_states_4 = None
        key_states_5 = view_7.transpose(1, 2)
        view_7 = None
        view_8 = value_states_4.view(1, 11, -1, 4)
        value_states_4 = None
        value_states_5 = view_8.transpose(1, 2)
        view_8 = None
        attention_mask_2 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 11, None),
            )
        ]
        causal_mask_3 = None
        query_2 = query_states_2.contiguous()
        query_states_2 = None
        key_2 = key_states_5.contiguous()
        value_2 = value_states_5.contiguous()
        attn_output_8 = torch._C._nn.scaled_dot_product_attention(
            query_2,
            key_2,
            value_2,
            attn_mask=attention_mask_2,
            dropout_p=0.0,
            scale=0.5,
            is_causal=False,
        )
        query_2 = key_2 = value_2 = attention_mask_2 = None
        transpose_11 = attn_output_8.transpose(1, 2)
        attn_output_8 = None
        attn_output_9 = transpose_11.contiguous()
        transpose_11 = None
        reshape_3 = attn_output_9.reshape(1, 11, -1)
        attn_output_9 = None
        attn_output_10 = reshape_3.contiguous()
        reshape_3 = None
        attn_output_11 = torch._C._nn.linear(
            attn_output_10,
            l_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_10 = l_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_15 = torch.nn.functional.dropout(
            attn_output_11, p=0.1, training=False
        )
        attn_output_11 = None
        hidden_states_16 = hidden_states_14 + hidden_states_15
        hidden_states_14 = hidden_states_15 = None
        hidden_states_17 = torch.nn.functional.layer_norm(
            hidden_states_16,
            (16,),
            l_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_16 = l_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_14 = torch._C._nn.linear(
            hidden_states_17,
            l_self_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        view_9 = linear_14.view(1, 11, -1, 4)
        linear_14 = None
        query_states_3 = view_9.transpose(1, 2)
        view_9 = None
        key_states_6 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_6 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_encoder_hidden_states_ = l_self_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_10 = key_states_6.view(1, 11, -1, 4)
        key_states_6 = None
        key_states_7 = view_10.transpose(1, 2)
        view_10 = None
        view_11 = value_states_6.view(1, 11, -1, 4)
        value_states_6 = None
        value_states_7 = view_11.transpose(1, 2)
        view_11 = None
        attention_mask_3 = encoder_attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 11, None),
            )
        ]
        encoder_attention_mask = None
        query_3 = query_states_3.contiguous()
        query_states_3 = None
        key_3 = key_states_7.contiguous()
        value_3 = value_states_7.contiguous()
        attn_output_12 = torch._C._nn.scaled_dot_product_attention(
            query_3,
            key_3,
            value_3,
            attn_mask=attention_mask_3,
            dropout_p=0.0,
            scale=0.5,
            is_causal=False,
        )
        query_3 = key_3 = value_3 = attention_mask_3 = None
        transpose_15 = attn_output_12.transpose(1, 2)
        attn_output_12 = None
        attn_output_13 = transpose_15.contiguous()
        transpose_15 = None
        reshape_4 = attn_output_13.reshape(1, 11, -1)
        attn_output_13 = None
        attn_output_14 = reshape_4.contiguous()
        reshape_4 = None
        attn_output_15 = torch._C._nn.linear(
            attn_output_14,
            l_self_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_14 = l_self_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_18 = torch.nn.functional.dropout(
            attn_output_15, p=0.1, training=False
        )
        attn_output_15 = None
        hidden_states_19 = hidden_states_17 + hidden_states_18
        hidden_states_17 = hidden_states_18 = None
        hidden_states_20 = torch.nn.functional.layer_norm(
            hidden_states_19,
            (16,),
            l_self_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_19 = l_self_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_18 = torch._C._nn.linear(
            hidden_states_20,
            l_self_modules_layers_modules_1_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_1_modules_fc1_parameters_bias_,
        )
        l_self_modules_layers_modules_1_modules_fc1_parameters_weight_ = (
            l_self_modules_layers_modules_1_modules_fc1_parameters_bias_
        ) = None
        hidden_states_21 = torch._C._nn.gelu(linear_18)
        linear_18 = None
        hidden_states_22 = torch.nn.functional.dropout(
            hidden_states_21, p=0.0, training=False
        )
        hidden_states_21 = None
        hidden_states_23 = torch._C._nn.linear(
            hidden_states_22,
            l_self_modules_layers_modules_1_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_1_modules_fc2_parameters_bias_,
        )
        hidden_states_22 = (
            l_self_modules_layers_modules_1_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_1_modules_fc2_parameters_bias_ = None
        hidden_states_24 = torch.nn.functional.dropout(
            hidden_states_23, p=0.1, training=False
        )
        hidden_states_23 = None
        hidden_states_25 = hidden_states_20 + hidden_states_24
        hidden_states_20 = hidden_states_24 = None
        hidden_states_26 = torch.nn.functional.layer_norm(
            hidden_states_25,
            (16,),
            l_self_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_25 = (
            l_self_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_
        ) = (
            l_self_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_
        ) = None
        return (
            value_states_1,
            key_states_1,
            value_states_3,
            key_states_3,
            value_states_5,
            key_states_5,
            value_states_7,
            key_states_7,
            hidden_states_26,
        )
