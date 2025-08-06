import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_kwargs_input_ids_: torch.Tensor,
        L_self_modules_model_modules_decoder_modules_embed_tokens_parameters_weight_: torch.nn.parameter.Parameter,
        L_kwargs_attention_mask_: torch.Tensor,
        L_self_modules_model_modules_decoder_modules_embed_positions_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_kwargs_input_ids_ = L_kwargs_input_ids_
        l_self_modules_model_modules_decoder_modules_embed_tokens_parameters_weight_ = (
            L_self_modules_model_modules_decoder_modules_embed_tokens_parameters_weight_
        )
        l_kwargs_attention_mask_ = L_kwargs_attention_mask_
        l_self_modules_model_modules_decoder_modules_embed_positions_parameters_weight_ = L_self_modules_model_modules_decoder_modules_embed_positions_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_bias_
        l_self_modules_model_modules_decoder_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_final_layer_norm_parameters_bias_
        input_ids = l_kwargs_input_ids_.view(-1, 20)
        l_kwargs_input_ids_ = None
        inputs_embeds = torch.nn.functional.embedding(
            input_ids,
            l_self_modules_model_modules_decoder_modules_embed_tokens_parameters_weight_,
            1,
            None,
            2.0,
            False,
            False,
        )
        input_ids = None
        cache_position = torch.arange(0, 20, device=device(type="cpu"))
        causal_mask = torch.full(
            (20, 20),
            fill_value=-65504.0,
            dtype=torch.float16,
            device=device(type="cpu"),
        )
        causal_mask_1 = torch.triu(causal_mask, diagonal=1)
        causal_mask = None
        arange_1 = torch.arange(20, device=device(type="cpu"))
        reshape = cache_position.reshape(-1, 1)
        cache_position = None
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
        causal_mask_4 = causal_mask_3.clone()
        causal_mask_3 = None
        getitem_1 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
            )
        ]
        getitem_2 = l_kwargs_attention_mask_[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        to = getitem_2.to(device(type="cpu"))
        getitem_2 = None
        padding_mask = getitem_1 + to
        getitem_1 = to = None
        padding_mask_1 = padding_mask == 0
        padding_mask = None
        getitem_3 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
            )
        ]
        masked_fill = getitem_3.masked_fill(padding_mask_1, -65504.0)
        getitem_3 = padding_mask_1 = None
        causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
            )
        ] = masked_fill
        setitem = causal_mask_4
        masked_fill = setitem = None
        position_ids = torch.cumsum(l_kwargs_attention_mask_, dim=1)
        mul = position_ids * l_kwargs_attention_mask_
        position_ids = l_kwargs_attention_mask_ = None
        sub = mul - 1
        mul = None
        position_ids_1 = sub.long()
        sub = None
        position_ids_2 = position_ids_1[(slice(None, None, None), slice(0, None, None))]
        position_ids_1 = None
        add_1 = position_ids_2 + 2
        position_ids_2 = None
        pos_embeds = torch.nn.functional.embedding(
            add_1,
            l_self_modules_model_modules_decoder_modules_embed_positions_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        add_1 = l_self_modules_model_modules_decoder_modules_embed_positions_parameters_weight_ = (None)
        to_1 = pos_embeds.to(device(type="cpu"))
        pos_embeds = None
        hidden_states = inputs_embeds + to_1
        inputs_embeds = to_1 = None
        hidden_states_1 = torch.nn.functional.layer_norm(
            hidden_states,
            (8,),
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear = torch._C._nn.linear(
            hidden_states_1,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states = linear * 0.5
        linear = None
        view_1 = query_states.view(1, -1, 2, 4)
        query_states = None
        query_states_1 = view_1.transpose(1, 2)
        view_1 = None
        key_states = torch._C._nn.linear(
            hidden_states_1,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states = torch._C._nn.linear(
            hidden_states_1,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_1 = l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_2 = key_states.view(1, -1, 2, 4)
        key_states = None
        key_states_1 = view_2.transpose(1, 2)
        view_2 = None
        view_3 = value_states.view(1, -1, 2, 4)
        value_states = None
        value_states_1 = view_3.transpose(1, 2)
        view_3 = None
        attention_mask = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
            )
        ]
        query = query_states_1.contiguous()
        query_states_1 = None
        key = key_states_1.contiguous()
        key_states_1 = None
        value = value_states_1.contiguous()
        value_states_1 = None
        attn_output = torch._C._nn.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            scale=1.0,
            is_causal=False,
        )
        query = key = value = attention_mask = None
        transpose_3 = attn_output.transpose(1, 2)
        attn_output = None
        attn_output_1 = transpose_3.contiguous()
        transpose_3 = None
        reshape_1 = attn_output_1.reshape(1, 20, -1)
        attn_output_1 = None
        attn_output_2 = reshape_1.contiguous()
        reshape_1 = None
        attn_output_3 = torch._C._nn.linear(
            attn_output_2,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_2 = l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_2 = torch.nn.functional.dropout(
            attn_output_3, p=0.1, training=False
        )
        attn_output_3 = None
        hidden_states_3 = hidden_states + hidden_states_2
        hidden_states = hidden_states_2 = None
        hidden_states_4 = hidden_states_3.reshape(-1, 8)
        hidden_states_3 = None
        hidden_states_5 = torch.nn.functional.layer_norm(
            hidden_states_4,
            (8,),
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_ = (None)
        hidden_states_6 = torch._C._nn.linear(
            hidden_states_5,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_bias_,
        )
        hidden_states_5 = l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_bias_ = (None)
        hidden_states_7 = torch.nn.functional.relu(hidden_states_6, inplace=False)
        hidden_states_6 = None
        hidden_states_8 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_bias_,
        )
        hidden_states_7 = l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_bias_ = (None)
        hidden_states_9 = torch.nn.functional.dropout(
            hidden_states_8, p=0.1, training=False
        )
        hidden_states_8 = None
        add_4 = hidden_states_4 + hidden_states_9
        hidden_states_4 = hidden_states_9 = None
        hidden_states_10 = add_4.view((1, 20, 8))
        add_4 = None
        hidden_states_11 = torch.nn.functional.layer_norm(
            hidden_states_10,
            (8,),
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_6 = torch._C._nn.linear(
            hidden_states_11,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_2 = linear_6 * 0.5
        linear_6 = None
        view_5 = query_states_2.view(1, -1, 2, 4)
        query_states_2 = None
        query_states_3 = view_5.transpose(1, 2)
        view_5 = None
        key_states_2 = torch._C._nn.linear(
            hidden_states_11,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_2 = torch._C._nn.linear(
            hidden_states_11,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_11 = l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_6 = key_states_2.view(1, -1, 2, 4)
        key_states_2 = None
        key_states_3 = view_6.transpose(1, 2)
        view_6 = None
        view_7 = value_states_2.view(1, -1, 2, 4)
        value_states_2 = None
        value_states_3 = view_7.transpose(1, 2)
        view_7 = None
        attention_mask_1 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
            )
        ]
        causal_mask_4 = None
        query_1 = query_states_3.contiguous()
        query_states_3 = None
        key_1 = key_states_3.contiguous()
        key_states_3 = None
        value_1 = value_states_3.contiguous()
        value_states_3 = None
        attn_output_4 = torch._C._nn.scaled_dot_product_attention(
            query_1,
            key_1,
            value_1,
            attn_mask=attention_mask_1,
            dropout_p=0.0,
            scale=1.0,
            is_causal=False,
        )
        query_1 = key_1 = value_1 = attention_mask_1 = None
        transpose_7 = attn_output_4.transpose(1, 2)
        attn_output_4 = None
        attn_output_5 = transpose_7.contiguous()
        transpose_7 = None
        reshape_3 = attn_output_5.reshape(1, 20, -1)
        attn_output_5 = None
        attn_output_6 = reshape_3.contiguous()
        reshape_3 = None
        attn_output_7 = torch._C._nn.linear(
            attn_output_6,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_6 = l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_12 = torch.nn.functional.dropout(
            attn_output_7, p=0.1, training=False
        )
        attn_output_7 = None
        hidden_states_13 = hidden_states_10 + hidden_states_12
        hidden_states_10 = hidden_states_12 = None
        hidden_states_14 = hidden_states_13.reshape(-1, 8)
        hidden_states_13 = None
        hidden_states_15 = torch.nn.functional.layer_norm(
            hidden_states_14,
            (8,),
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_ = (None)
        hidden_states_16 = torch._C._nn.linear(
            hidden_states_15,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_bias_,
        )
        hidden_states_15 = l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_bias_ = (None)
        hidden_states_17 = torch.nn.functional.relu(hidden_states_16, inplace=False)
        hidden_states_16 = None
        hidden_states_18 = torch._C._nn.linear(
            hidden_states_17,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_bias_,
        )
        hidden_states_17 = l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_bias_ = (None)
        hidden_states_19 = torch.nn.functional.dropout(
            hidden_states_18, p=0.1, training=False
        )
        hidden_states_18 = None
        add_6 = hidden_states_14 + hidden_states_19
        hidden_states_14 = hidden_states_19 = None
        hidden_states_20 = add_6.view((1, 20, 8))
        add_6 = None
        hidden_states_21 = torch.nn.functional.layer_norm(
            hidden_states_20,
            (8,),
            l_self_modules_model_modules_decoder_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_20 = l_self_modules_model_modules_decoder_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_final_layer_norm_parameters_bias_ = (None)
        linear_12 = torch._C._nn.linear(
            hidden_states_21,
            l_self_modules_model_modules_decoder_modules_embed_tokens_parameters_weight_,
            None,
        )
        hidden_states_21 = (
            l_self_modules_model_modules_decoder_modules_embed_tokens_parameters_weight_
        ) = None
        logits = linear_12.contiguous()
        linear_12 = None
        return (logits,)
