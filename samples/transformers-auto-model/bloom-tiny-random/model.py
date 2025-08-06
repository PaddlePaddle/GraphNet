import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_input_ids_: torch.Tensor,
        L_self_modules_transformer_modules_word_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_word_embeddings_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_word_embeddings_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_attention_mask_: torch.Tensor,
        L_self_modules_transformer_modules_h_modules_0_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_self_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_self_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_self_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_self_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_self_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_self_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_self_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_self_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_ln_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_ln_f_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_input_ids_ = L_input_ids_
        l_self_modules_transformer_modules_word_embeddings_parameters_weight_ = (
            L_self_modules_transformer_modules_word_embeddings_parameters_weight_
        )
        l_self_modules_transformer_modules_word_embeddings_layernorm_parameters_weight_ = L_self_modules_transformer_modules_word_embeddings_layernorm_parameters_weight_
        l_self_modules_transformer_modules_word_embeddings_layernorm_parameters_bias_ = L_self_modules_transformer_modules_word_embeddings_layernorm_parameters_bias_
        l_attention_mask_ = L_attention_mask_
        l_self_modules_transformer_modules_h_modules_0_modules_input_layernorm_parameters_weight_ = L_self_modules_transformer_modules_h_modules_0_modules_input_layernorm_parameters_weight_
        l_self_modules_transformer_modules_h_modules_0_modules_input_layernorm_parameters_bias_ = L_self_modules_transformer_modules_h_modules_0_modules_input_layernorm_parameters_bias_
        l_self_modules_transformer_modules_h_modules_0_modules_self_attention_modules_query_key_value_parameters_weight_ = L_self_modules_transformer_modules_h_modules_0_modules_self_attention_modules_query_key_value_parameters_weight_
        l_self_modules_transformer_modules_h_modules_0_modules_self_attention_modules_query_key_value_parameters_bias_ = L_self_modules_transformer_modules_h_modules_0_modules_self_attention_modules_query_key_value_parameters_bias_
        l_self_modules_transformer_modules_h_modules_0_modules_self_attention_modules_dense_parameters_weight_ = L_self_modules_transformer_modules_h_modules_0_modules_self_attention_modules_dense_parameters_weight_
        l_self_modules_transformer_modules_h_modules_0_modules_self_attention_modules_dense_parameters_bias_ = L_self_modules_transformer_modules_h_modules_0_modules_self_attention_modules_dense_parameters_bias_
        l_self_modules_transformer_modules_h_modules_0_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_transformer_modules_h_modules_0_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_transformer_modules_h_modules_0_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_transformer_modules_h_modules_0_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_transformer_modules_h_modules_1_modules_input_layernorm_parameters_weight_ = L_self_modules_transformer_modules_h_modules_1_modules_input_layernorm_parameters_weight_
        l_self_modules_transformer_modules_h_modules_1_modules_input_layernorm_parameters_bias_ = L_self_modules_transformer_modules_h_modules_1_modules_input_layernorm_parameters_bias_
        l_self_modules_transformer_modules_h_modules_1_modules_self_attention_modules_query_key_value_parameters_weight_ = L_self_modules_transformer_modules_h_modules_1_modules_self_attention_modules_query_key_value_parameters_weight_
        l_self_modules_transformer_modules_h_modules_1_modules_self_attention_modules_query_key_value_parameters_bias_ = L_self_modules_transformer_modules_h_modules_1_modules_self_attention_modules_query_key_value_parameters_bias_
        l_self_modules_transformer_modules_h_modules_1_modules_self_attention_modules_dense_parameters_weight_ = L_self_modules_transformer_modules_h_modules_1_modules_self_attention_modules_dense_parameters_weight_
        l_self_modules_transformer_modules_h_modules_1_modules_self_attention_modules_dense_parameters_bias_ = L_self_modules_transformer_modules_h_modules_1_modules_self_attention_modules_dense_parameters_bias_
        l_self_modules_transformer_modules_h_modules_1_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_transformer_modules_h_modules_1_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_transformer_modules_h_modules_1_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_transformer_modules_h_modules_1_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_transformer_modules_ln_f_parameters_weight_ = (
            L_self_modules_transformer_modules_ln_f_parameters_weight_
        )
        l_self_modules_transformer_modules_ln_f_parameters_bias_ = (
            L_self_modules_transformer_modules_ln_f_parameters_bias_
        )
        inputs_embeds = torch.nn.functional.embedding(
            l_input_ids_,
            l_self_modules_transformer_modules_word_embeddings_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        l_input_ids_ = None
        cache_position = torch.arange(0, 18, device=device(type="cpu"))
        hidden_states = torch.nn.functional.layer_norm(
            inputs_embeds,
            (8,),
            l_self_modules_transformer_modules_word_embeddings_layernorm_parameters_weight_,
            l_self_modules_transformer_modules_word_embeddings_layernorm_parameters_bias_,
            1e-05,
        )
        inputs_embeds = l_self_modules_transformer_modules_word_embeddings_layernorm_parameters_weight_ = l_self_modules_transformer_modules_word_embeddings_layernorm_parameters_bias_ = (None)
        attention_mask = l_attention_mask_.to(device(type="cpu"))
        l_attention_mask_ = None
        base = torch.tensor(0.0625, device=device(type="cpu"), dtype=torch.float32)
        powers = torch.arange(1, 3, device=device(type="cpu"), dtype=torch.int32)
        slopes = torch.pow(base, powers)
        base = powers = None
        cumsum = attention_mask.cumsum(dim=-1)
        sub = cumsum - 1
        cumsum = None
        mul = sub * attention_mask
        sub = None
        arange_tensor = mul[(slice(None, None, None), None, slice(None, None, None))]
        mul = None
        getitem_1 = slopes[(Ellipsis, None)]
        slopes = None
        alibi = getitem_1 * arange_tensor
        getitem_1 = arange_tensor = None
        reshape = alibi.reshape(2, 1, 18)
        alibi = None
        alibi_1 = reshape.to(torch.float16)
        reshape = None
        causal_mask = torch.full(
            (18, 18),
            fill_value=-65504.0,
            dtype=torch.float16,
            device=device(type="cpu"),
        )
        causal_mask_1 = torch.triu(causal_mask, diagonal=1)
        causal_mask = None
        arange_2 = torch.arange(18, device=device(type="cpu"))
        reshape_1 = cache_position.reshape(-1, 1)
        cache_position = None
        gt = arange_2 > reshape_1
        arange_2 = reshape_1 = None
        causal_mask_1 *= gt
        causal_mask_2 = causal_mask_1
        causal_mask_1 = gt = None
        getitem_2 = causal_mask_2[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        causal_mask_2 = None
        causal_mask_3 = getitem_2.expand(1, 1, -1, -1)
        getitem_2 = None
        causal_mask_4 = causal_mask_3.clone()
        causal_mask_3 = None
        getitem_3 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 18, None),
            )
        ]
        getitem_4 = attention_mask[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        attention_mask = None
        to_2 = getitem_4.to(device(type="cpu"))
        getitem_4 = None
        padding_mask = getitem_3 + to_2
        getitem_3 = to_2 = None
        padding_mask_1 = padding_mask == 0
        padding_mask = None
        getitem_5 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 18, None),
            )
        ]
        masked_fill = getitem_5.masked_fill(padding_mask_1, -65504.0)
        getitem_5 = padding_mask_1 = None
        causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 18, None),
            )
        ] = masked_fill
        setitem = causal_mask_4
        masked_fill = setitem = None
        layernorm_output = torch.nn.functional.layer_norm(
            hidden_states,
            (8,),
            l_self_modules_transformer_modules_h_modules_0_modules_input_layernorm_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_0_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_0_modules_input_layernorm_parameters_weight_ = l_self_modules_transformer_modules_h_modules_0_modules_input_layernorm_parameters_bias_ = (None)
        fused_qkv = torch._C._nn.linear(
            layernorm_output,
            l_self_modules_transformer_modules_h_modules_0_modules_self_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_0_modules_self_attention_modules_query_key_value_parameters_bias_,
        )
        layernorm_output = l_self_modules_transformer_modules_h_modules_0_modules_self_attention_modules_query_key_value_parameters_weight_ = l_self_modules_transformer_modules_h_modules_0_modules_self_attention_modules_query_key_value_parameters_bias_ = (None)
        fused_qkv_1 = fused_qkv.view(1, 18, 2, 3, 4)
        fused_qkv = None
        getitem_6 = fused_qkv_1[(Ellipsis, 0, slice(None, None, None))]
        query_layer = getitem_6.transpose(1, 2)
        getitem_6 = None
        getitem_7 = fused_qkv_1[(Ellipsis, 1, slice(None, None, None))]
        key_layer = getitem_7.transpose(1, 2)
        getitem_7 = None
        getitem_8 = fused_qkv_1[(Ellipsis, 2, slice(None, None, None))]
        fused_qkv_1 = None
        value_layer = getitem_8.transpose(1, 2)
        getitem_8 = None
        query_layer_1 = query_layer.reshape(2, -1, 4)
        query_layer = None
        reshape_3 = key_layer.reshape(2, -1, 4)
        key_layer = None
        key_layer_1 = reshape_3.transpose(-1, -2)
        reshape_3 = None
        value_layer_1 = value_layer.reshape(2, -1, 4)
        value_layer = None
        attention_scores = alibi_1.baddbmm(
            batch1=query_layer_1, batch2=key_layer_1, beta=1.0, alpha=0.5
        )
        query_layer_1 = key_layer_1 = None
        attn_weights = attention_scores.view(1, 2, 18, -1)
        attention_scores = None
        causal_mask_5 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 18, None),
            )
        ]
        attn_weights_1 = attn_weights + causal_mask_5
        attn_weights = causal_mask_5 = None
        softmax = torch.nn.functional.softmax(
            attn_weights_1, dim=-1, dtype=torch.float32
        )
        attn_weights_1 = None
        attention_probs = softmax.to(torch.float16)
        softmax = None
        attention_probs_1 = torch.nn.functional.dropout(
            attention_probs, 0.0, False, False
        )
        attention_probs = None
        attention_probs_reshaped = attention_probs_1.view(2, 18, -1)
        attention_probs_1 = None
        context_layer = torch.bmm(attention_probs_reshaped, value_layer_1)
        attention_probs_reshaped = value_layer_1 = None
        x = context_layer.view(1, 2, 18, 4)
        context_layer = None
        x_1 = x.permute(0, 2, 1, 3)
        x = None
        context_layer_1 = x_1.reshape(1, 18, 8)
        x_1 = None
        output_tensor = torch._C._nn.linear(
            context_layer_1,
            l_self_modules_transformer_modules_h_modules_0_modules_self_attention_modules_dense_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_0_modules_self_attention_modules_dense_parameters_bias_,
        )
        context_layer_1 = l_self_modules_transformer_modules_h_modules_0_modules_self_attention_modules_dense_parameters_weight_ = l_self_modules_transformer_modules_h_modules_0_modules_self_attention_modules_dense_parameters_bias_ = (None)
        out = torch.nn.functional.dropout(output_tensor, p=0.0, training=False)
        output_tensor = None
        out_1 = hidden_states + out
        hidden_states = out = None
        layernorm_output_1 = torch.nn.functional.layer_norm(
            out_1,
            (8,),
            l_self_modules_transformer_modules_h_modules_0_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_0_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_0_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_transformer_modules_h_modules_0_modules_post_attention_layernorm_parameters_bias_ = (None)
        linear_2 = torch._C._nn.linear(
            layernorm_output_1,
            l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layernorm_output_1 = l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        mul_2 = linear_2 * 0.5
        mul_3 = 0.79788456 * linear_2
        mul_4 = 0.044715 * linear_2
        mul_5 = mul_4 * linear_2
        mul_4 = linear_2 = None
        add_3 = 1 + mul_5
        mul_5 = None
        mul_6 = mul_3 * add_3
        mul_3 = add_3 = None
        tanh = torch.tanh(mul_6)
        mul_6 = None
        add_4 = 1.0 + tanh
        tanh = None
        hidden_states_1 = mul_2 * add_4
        mul_2 = add_4 = None
        intermediate_output = torch._C._nn.linear(
            hidden_states_1,
            l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_1 = l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        out_2 = torch.nn.functional.dropout(intermediate_output, p=0.0, training=False)
        intermediate_output = None
        out_3 = out_1 + out_2
        out_1 = out_2 = None
        layernorm_output_2 = torch.nn.functional.layer_norm(
            out_3,
            (8,),
            l_self_modules_transformer_modules_h_modules_1_modules_input_layernorm_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_1_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_1_modules_input_layernorm_parameters_weight_ = l_self_modules_transformer_modules_h_modules_1_modules_input_layernorm_parameters_bias_ = (None)
        fused_qkv_2 = torch._C._nn.linear(
            layernorm_output_2,
            l_self_modules_transformer_modules_h_modules_1_modules_self_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_1_modules_self_attention_modules_query_key_value_parameters_bias_,
        )
        layernorm_output_2 = l_self_modules_transformer_modules_h_modules_1_modules_self_attention_modules_query_key_value_parameters_weight_ = l_self_modules_transformer_modules_h_modules_1_modules_self_attention_modules_query_key_value_parameters_bias_ = (None)
        fused_qkv_3 = fused_qkv_2.view(1, 18, 2, 3, 4)
        fused_qkv_2 = None
        getitem_10 = fused_qkv_3[(Ellipsis, 0, slice(None, None, None))]
        query_layer_2 = getitem_10.transpose(1, 2)
        getitem_10 = None
        getitem_11 = fused_qkv_3[(Ellipsis, 1, slice(None, None, None))]
        key_layer_2 = getitem_11.transpose(1, 2)
        getitem_11 = None
        getitem_12 = fused_qkv_3[(Ellipsis, 2, slice(None, None, None))]
        fused_qkv_3 = None
        value_layer_2 = getitem_12.transpose(1, 2)
        getitem_12 = None
        query_layer_3 = query_layer_2.reshape(2, -1, 4)
        query_layer_2 = None
        reshape_7 = key_layer_2.reshape(2, -1, 4)
        key_layer_2 = None
        key_layer_3 = reshape_7.transpose(-1, -2)
        reshape_7 = None
        value_layer_3 = value_layer_2.reshape(2, -1, 4)
        value_layer_2 = None
        attention_scores_1 = alibi_1.baddbmm(
            batch1=query_layer_3, batch2=key_layer_3, beta=1.0, alpha=0.5
        )
        alibi_1 = query_layer_3 = key_layer_3 = None
        attn_weights_2 = attention_scores_1.view(1, 2, 18, -1)
        attention_scores_1 = None
        causal_mask_6 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 18, None),
            )
        ]
        causal_mask_4 = None
        attn_weights_3 = attn_weights_2 + causal_mask_6
        attn_weights_2 = causal_mask_6 = None
        softmax_1 = torch.nn.functional.softmax(
            attn_weights_3, dim=-1, dtype=torch.float32
        )
        attn_weights_3 = None
        attention_probs_2 = softmax_1.to(torch.float16)
        softmax_1 = None
        attention_probs_3 = torch.nn.functional.dropout(
            attention_probs_2, 0.0, False, False
        )
        attention_probs_2 = None
        attention_probs_reshaped_1 = attention_probs_3.view(2, 18, -1)
        attention_probs_3 = None
        context_layer_2 = torch.bmm(attention_probs_reshaped_1, value_layer_3)
        attention_probs_reshaped_1 = value_layer_3 = None
        x_2 = context_layer_2.view(1, 2, 18, 4)
        context_layer_2 = None
        x_3 = x_2.permute(0, 2, 1, 3)
        x_2 = None
        context_layer_3 = x_3.reshape(1, 18, 8)
        x_3 = None
        output_tensor_1 = torch._C._nn.linear(
            context_layer_3,
            l_self_modules_transformer_modules_h_modules_1_modules_self_attention_modules_dense_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_1_modules_self_attention_modules_dense_parameters_bias_,
        )
        context_layer_3 = l_self_modules_transformer_modules_h_modules_1_modules_self_attention_modules_dense_parameters_weight_ = l_self_modules_transformer_modules_h_modules_1_modules_self_attention_modules_dense_parameters_bias_ = (None)
        out_4 = torch.nn.functional.dropout(output_tensor_1, p=0.0, training=False)
        output_tensor_1 = None
        out_5 = out_3 + out_4
        out_3 = out_4 = None
        layernorm_output_3 = torch.nn.functional.layer_norm(
            out_5,
            (8,),
            l_self_modules_transformer_modules_h_modules_1_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_1_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_1_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_transformer_modules_h_modules_1_modules_post_attention_layernorm_parameters_bias_ = (None)
        linear_6 = torch._C._nn.linear(
            layernorm_output_3,
            l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layernorm_output_3 = l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        mul_8 = linear_6 * 0.5
        mul_9 = 0.79788456 * linear_6
        mul_10 = 0.044715 * linear_6
        mul_11 = mul_10 * linear_6
        mul_10 = linear_6 = None
        add_8 = 1 + mul_11
        mul_11 = None
        mul_12 = mul_9 * add_8
        mul_9 = add_8 = None
        tanh_1 = torch.tanh(mul_12)
        mul_12 = None
        add_9 = 1.0 + tanh_1
        tanh_1 = None
        hidden_states_2 = mul_8 * add_9
        mul_8 = add_9 = None
        intermediate_output_1 = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_2 = l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        out_6 = torch.nn.functional.dropout(
            intermediate_output_1, p=0.0, training=False
        )
        intermediate_output_1 = None
        out_7 = out_5 + out_6
        out_5 = out_6 = None
        hidden_states_3 = torch.nn.functional.layer_norm(
            out_7,
            (8,),
            l_self_modules_transformer_modules_ln_f_parameters_weight_,
            l_self_modules_transformer_modules_ln_f_parameters_bias_,
            1e-05,
        )
        out_7 = (
            l_self_modules_transformer_modules_ln_f_parameters_weight_
        ) = l_self_modules_transformer_modules_ln_f_parameters_bias_ = None
        lm_logits = torch._C._nn.linear(
            hidden_states_3,
            l_self_modules_transformer_modules_word_embeddings_parameters_weight_,
            None,
        )
        hidden_states_3 = (
            l_self_modules_transformer_modules_word_embeddings_parameters_weight_
        ) = None
        return (lm_logits,)
