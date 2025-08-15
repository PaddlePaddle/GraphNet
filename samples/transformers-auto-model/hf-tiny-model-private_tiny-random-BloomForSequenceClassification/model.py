import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_inputs_embeds_: torch.Tensor,
        L_self_modules_word_embeddings_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_word_embeddings_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_attention_mask_: torch.Tensor,
        L_self_modules_h_modules_0_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_self_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_self_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_self_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_self_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_self_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_self_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_self_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_self_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_self_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_self_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_self_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_self_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_self_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_self_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_self_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_self_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_self_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_self_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_self_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_self_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_ln_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_ln_f_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_inputs_embeds_ = L_inputs_embeds_
        l_self_modules_word_embeddings_layernorm_parameters_weight_ = (
            L_self_modules_word_embeddings_layernorm_parameters_weight_
        )
        l_self_modules_word_embeddings_layernorm_parameters_bias_ = (
            L_self_modules_word_embeddings_layernorm_parameters_bias_
        )
        l_attention_mask_ = L_attention_mask_
        l_self_modules_h_modules_0_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_h_modules_0_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_h_modules_0_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_h_modules_0_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_h_modules_0_modules_self_attention_modules_query_key_value_parameters_weight_ = L_self_modules_h_modules_0_modules_self_attention_modules_query_key_value_parameters_weight_
        l_self_modules_h_modules_0_modules_self_attention_modules_query_key_value_parameters_bias_ = L_self_modules_h_modules_0_modules_self_attention_modules_query_key_value_parameters_bias_
        l_self_modules_h_modules_0_modules_self_attention_modules_dense_parameters_weight_ = L_self_modules_h_modules_0_modules_self_attention_modules_dense_parameters_weight_
        l_self_modules_h_modules_0_modules_self_attention_modules_dense_parameters_bias_ = L_self_modules_h_modules_0_modules_self_attention_modules_dense_parameters_bias_
        l_self_modules_h_modules_0_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_h_modules_0_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_h_modules_0_modules_post_attention_layernorm_parameters_bias_ = (
            L_self_modules_h_modules_0_modules_post_attention_layernorm_parameters_bias_
        )
        l_self_modules_h_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_h_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_h_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_h_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_h_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_h_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_h_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_h_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_h_modules_1_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_h_modules_1_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_h_modules_1_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_h_modules_1_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_h_modules_1_modules_self_attention_modules_query_key_value_parameters_weight_ = L_self_modules_h_modules_1_modules_self_attention_modules_query_key_value_parameters_weight_
        l_self_modules_h_modules_1_modules_self_attention_modules_query_key_value_parameters_bias_ = L_self_modules_h_modules_1_modules_self_attention_modules_query_key_value_parameters_bias_
        l_self_modules_h_modules_1_modules_self_attention_modules_dense_parameters_weight_ = L_self_modules_h_modules_1_modules_self_attention_modules_dense_parameters_weight_
        l_self_modules_h_modules_1_modules_self_attention_modules_dense_parameters_bias_ = L_self_modules_h_modules_1_modules_self_attention_modules_dense_parameters_bias_
        l_self_modules_h_modules_1_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_h_modules_1_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_h_modules_1_modules_post_attention_layernorm_parameters_bias_ = (
            L_self_modules_h_modules_1_modules_post_attention_layernorm_parameters_bias_
        )
        l_self_modules_h_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_h_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_h_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_h_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_h_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_h_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_h_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_h_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_h_modules_2_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_h_modules_2_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_h_modules_2_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_h_modules_2_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_h_modules_2_modules_self_attention_modules_query_key_value_parameters_weight_ = L_self_modules_h_modules_2_modules_self_attention_modules_query_key_value_parameters_weight_
        l_self_modules_h_modules_2_modules_self_attention_modules_query_key_value_parameters_bias_ = L_self_modules_h_modules_2_modules_self_attention_modules_query_key_value_parameters_bias_
        l_self_modules_h_modules_2_modules_self_attention_modules_dense_parameters_weight_ = L_self_modules_h_modules_2_modules_self_attention_modules_dense_parameters_weight_
        l_self_modules_h_modules_2_modules_self_attention_modules_dense_parameters_bias_ = L_self_modules_h_modules_2_modules_self_attention_modules_dense_parameters_bias_
        l_self_modules_h_modules_2_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_h_modules_2_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_h_modules_2_modules_post_attention_layernorm_parameters_bias_ = (
            L_self_modules_h_modules_2_modules_post_attention_layernorm_parameters_bias_
        )
        l_self_modules_h_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_h_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_h_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_h_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_h_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_h_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_h_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_h_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_h_modules_3_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_h_modules_3_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_h_modules_3_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_h_modules_3_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_h_modules_3_modules_self_attention_modules_query_key_value_parameters_weight_ = L_self_modules_h_modules_3_modules_self_attention_modules_query_key_value_parameters_weight_
        l_self_modules_h_modules_3_modules_self_attention_modules_query_key_value_parameters_bias_ = L_self_modules_h_modules_3_modules_self_attention_modules_query_key_value_parameters_bias_
        l_self_modules_h_modules_3_modules_self_attention_modules_dense_parameters_weight_ = L_self_modules_h_modules_3_modules_self_attention_modules_dense_parameters_weight_
        l_self_modules_h_modules_3_modules_self_attention_modules_dense_parameters_bias_ = L_self_modules_h_modules_3_modules_self_attention_modules_dense_parameters_bias_
        l_self_modules_h_modules_3_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_h_modules_3_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_h_modules_3_modules_post_attention_layernorm_parameters_bias_ = (
            L_self_modules_h_modules_3_modules_post_attention_layernorm_parameters_bias_
        )
        l_self_modules_h_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_h_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_h_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_h_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_h_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_h_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_h_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_h_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_h_modules_4_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_h_modules_4_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_h_modules_4_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_h_modules_4_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_h_modules_4_modules_self_attention_modules_query_key_value_parameters_weight_ = L_self_modules_h_modules_4_modules_self_attention_modules_query_key_value_parameters_weight_
        l_self_modules_h_modules_4_modules_self_attention_modules_query_key_value_parameters_bias_ = L_self_modules_h_modules_4_modules_self_attention_modules_query_key_value_parameters_bias_
        l_self_modules_h_modules_4_modules_self_attention_modules_dense_parameters_weight_ = L_self_modules_h_modules_4_modules_self_attention_modules_dense_parameters_weight_
        l_self_modules_h_modules_4_modules_self_attention_modules_dense_parameters_bias_ = L_self_modules_h_modules_4_modules_self_attention_modules_dense_parameters_bias_
        l_self_modules_h_modules_4_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_h_modules_4_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_h_modules_4_modules_post_attention_layernorm_parameters_bias_ = (
            L_self_modules_h_modules_4_modules_post_attention_layernorm_parameters_bias_
        )
        l_self_modules_h_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_h_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_h_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_h_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_h_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_h_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_h_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_h_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_ln_f_parameters_weight_ = L_self_modules_ln_f_parameters_weight_
        l_self_modules_ln_f_parameters_bias_ = L_self_modules_ln_f_parameters_bias_
        cache_position = torch.arange(0, 19, device=device(type="cuda", index=0))
        hidden_states = torch.nn.functional.layer_norm(
            l_inputs_embeds_,
            (32,),
            l_self_modules_word_embeddings_layernorm_parameters_weight_,
            l_self_modules_word_embeddings_layernorm_parameters_bias_,
            1e-05,
        )
        l_inputs_embeds_ = (
            l_self_modules_word_embeddings_layernorm_parameters_weight_
        ) = l_self_modules_word_embeddings_layernorm_parameters_bias_ = None
        attention_mask = l_attention_mask_.to(device(type="cuda", index=0))
        l_attention_mask_ = None
        base = torch.tensor(
            0.25, device=device(type="cuda", index=0), dtype=torch.float32
        )
        powers = torch.arange(
            1, 5, device=device(type="cuda", index=0), dtype=torch.int32
        )
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
        reshape = alibi.reshape(4, 1, 19)
        alibi = None
        alibi_1 = reshape.to(torch.float32)
        reshape = None
        causal_mask = torch.full(
            (19, 19),
            fill_value=-3.4028234663852886e38,
            dtype=torch.float32,
            device=device(type="cuda", index=0),
        )
        causal_mask_1 = torch.triu(causal_mask, diagonal=1)
        causal_mask = None
        arange_2 = torch.arange(19, device=device(type="cuda", index=0))
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
                slice(None, 19, None),
            )
        ]
        getitem_4 = attention_mask[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        attention_mask = None
        to_2 = getitem_4.to(device(type="cuda", index=0))
        getitem_4 = None
        padding_mask = getitem_3 + to_2
        getitem_3 = to_2 = None
        padding_mask_1 = padding_mask.__eq__(0)
        padding_mask = None
        getitem_5 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        masked_fill = getitem_5.masked_fill(padding_mask_1, -3.4028234663852886e38)
        getitem_5 = padding_mask_1 = None
        causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ] = masked_fill
        setitem = causal_mask_4
        masked_fill = setitem = None
        layernorm_output = torch.nn.functional.layer_norm(
            hidden_states,
            (32,),
            l_self_modules_h_modules_0_modules_input_layernorm_parameters_weight_,
            l_self_modules_h_modules_0_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_0_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_h_modules_0_modules_input_layernorm_parameters_bias_
        ) = None
        fused_qkv = torch._C._nn.linear(
            layernorm_output,
            l_self_modules_h_modules_0_modules_self_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_h_modules_0_modules_self_attention_modules_query_key_value_parameters_bias_,
        )
        layernorm_output = l_self_modules_h_modules_0_modules_self_attention_modules_query_key_value_parameters_weight_ = l_self_modules_h_modules_0_modules_self_attention_modules_query_key_value_parameters_bias_ = (None)
        fused_qkv_1 = fused_qkv.view(1, 19, 4, 3, 8)
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
        query_layer_1 = query_layer.reshape(4, -1, 8)
        query_layer = None
        reshape_3 = key_layer.reshape(4, -1, 8)
        key_layer_1 = reshape_3.transpose(-1, -2)
        reshape_3 = None
        value_layer_1 = value_layer.reshape(4, -1, 8)
        attention_scores = alibi_1.baddbmm(
            batch1=query_layer_1,
            batch2=key_layer_1,
            beta=1.0,
            alpha=0.35355339059327373,
        )
        query_layer_1 = key_layer_1 = None
        attn_weights = attention_scores.view(1, 4, 19, -1)
        attention_scores = None
        causal_mask_5 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        attn_weights_1 = attn_weights + causal_mask_5
        attn_weights = causal_mask_5 = None
        softmax = torch.nn.functional.softmax(
            attn_weights_1, dim=-1, dtype=torch.float32
        )
        attn_weights_1 = None
        attention_probs = softmax.to(torch.float32)
        softmax = None
        attention_probs_1 = torch.nn.functional.dropout(
            attention_probs, 0.1, False, False
        )
        attention_probs = None
        attention_probs_reshaped = attention_probs_1.view(4, 19, -1)
        attention_probs_1 = None
        context_layer = torch.bmm(attention_probs_reshaped, value_layer_1)
        attention_probs_reshaped = value_layer_1 = None
        x = context_layer.view(1, 4, 19, 8)
        context_layer = None
        x_1 = x.permute(0, 2, 1, 3)
        x = None
        context_layer_1 = x_1.reshape(1, 19, 32)
        x_1 = None
        output_tensor = torch._C._nn.linear(
            context_layer_1,
            l_self_modules_h_modules_0_modules_self_attention_modules_dense_parameters_weight_,
            l_self_modules_h_modules_0_modules_self_attention_modules_dense_parameters_bias_,
        )
        context_layer_1 = l_self_modules_h_modules_0_modules_self_attention_modules_dense_parameters_weight_ = l_self_modules_h_modules_0_modules_self_attention_modules_dense_parameters_bias_ = (None)
        out = torch.nn.functional.dropout(output_tensor, p=0.1, training=False)
        output_tensor = None
        out_1 = hidden_states + out
        hidden_states = out = None
        layernorm_output_1 = torch.nn.functional.layer_norm(
            out_1,
            (32,),
            l_self_modules_h_modules_0_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_h_modules_0_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_0_modules_post_attention_layernorm_parameters_weight_ = (
            l_self_modules_h_modules_0_modules_post_attention_layernorm_parameters_bias_
        ) = None
        linear_2 = torch._C._nn.linear(
            layernorm_output_1,
            l_self_modules_h_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_h_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layernorm_output_1 = l_self_modules_h_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_h_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
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
            l_self_modules_h_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_h_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_1 = l_self_modules_h_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_h_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        out_2 = torch.nn.functional.dropout(intermediate_output, p=0.1, training=False)
        intermediate_output = None
        out_3 = out_1 + out_2
        out_1 = out_2 = None
        layernorm_output_2 = torch.nn.functional.layer_norm(
            out_3,
            (32,),
            l_self_modules_h_modules_1_modules_input_layernorm_parameters_weight_,
            l_self_modules_h_modules_1_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_1_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_h_modules_1_modules_input_layernorm_parameters_bias_
        ) = None
        fused_qkv_2 = torch._C._nn.linear(
            layernorm_output_2,
            l_self_modules_h_modules_1_modules_self_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_h_modules_1_modules_self_attention_modules_query_key_value_parameters_bias_,
        )
        layernorm_output_2 = l_self_modules_h_modules_1_modules_self_attention_modules_query_key_value_parameters_weight_ = l_self_modules_h_modules_1_modules_self_attention_modules_query_key_value_parameters_bias_ = (None)
        fused_qkv_3 = fused_qkv_2.view(1, 19, 4, 3, 8)
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
        query_layer_3 = query_layer_2.reshape(4, -1, 8)
        query_layer_2 = None
        reshape_7 = key_layer_2.reshape(4, -1, 8)
        key_layer_3 = reshape_7.transpose(-1, -2)
        reshape_7 = None
        value_layer_3 = value_layer_2.reshape(4, -1, 8)
        attention_scores_1 = alibi_1.baddbmm(
            batch1=query_layer_3,
            batch2=key_layer_3,
            beta=1.0,
            alpha=0.35355339059327373,
        )
        query_layer_3 = key_layer_3 = None
        attn_weights_2 = attention_scores_1.view(1, 4, 19, -1)
        attention_scores_1 = None
        causal_mask_6 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        attn_weights_3 = attn_weights_2 + causal_mask_6
        attn_weights_2 = causal_mask_6 = None
        softmax_1 = torch.nn.functional.softmax(
            attn_weights_3, dim=-1, dtype=torch.float32
        )
        attn_weights_3 = None
        attention_probs_2 = softmax_1.to(torch.float32)
        softmax_1 = None
        attention_probs_3 = torch.nn.functional.dropout(
            attention_probs_2, 0.1, False, False
        )
        attention_probs_2 = None
        attention_probs_reshaped_1 = attention_probs_3.view(4, 19, -1)
        attention_probs_3 = None
        context_layer_2 = torch.bmm(attention_probs_reshaped_1, value_layer_3)
        attention_probs_reshaped_1 = value_layer_3 = None
        x_2 = context_layer_2.view(1, 4, 19, 8)
        context_layer_2 = None
        x_3 = x_2.permute(0, 2, 1, 3)
        x_2 = None
        context_layer_3 = x_3.reshape(1, 19, 32)
        x_3 = None
        output_tensor_1 = torch._C._nn.linear(
            context_layer_3,
            l_self_modules_h_modules_1_modules_self_attention_modules_dense_parameters_weight_,
            l_self_modules_h_modules_1_modules_self_attention_modules_dense_parameters_bias_,
        )
        context_layer_3 = l_self_modules_h_modules_1_modules_self_attention_modules_dense_parameters_weight_ = l_self_modules_h_modules_1_modules_self_attention_modules_dense_parameters_bias_ = (None)
        out_4 = torch.nn.functional.dropout(output_tensor_1, p=0.1, training=False)
        output_tensor_1 = None
        out_5 = out_3 + out_4
        out_3 = out_4 = None
        layernorm_output_3 = torch.nn.functional.layer_norm(
            out_5,
            (32,),
            l_self_modules_h_modules_1_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_h_modules_1_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_1_modules_post_attention_layernorm_parameters_weight_ = (
            l_self_modules_h_modules_1_modules_post_attention_layernorm_parameters_bias_
        ) = None
        linear_6 = torch._C._nn.linear(
            layernorm_output_3,
            l_self_modules_h_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_h_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layernorm_output_3 = l_self_modules_h_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_h_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
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
            l_self_modules_h_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_h_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_2 = l_self_modules_h_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_h_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        out_6 = torch.nn.functional.dropout(
            intermediate_output_1, p=0.1, training=False
        )
        intermediate_output_1 = None
        out_7 = out_5 + out_6
        out_5 = out_6 = None
        layernorm_output_4 = torch.nn.functional.layer_norm(
            out_7,
            (32,),
            l_self_modules_h_modules_2_modules_input_layernorm_parameters_weight_,
            l_self_modules_h_modules_2_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_2_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_h_modules_2_modules_input_layernorm_parameters_bias_
        ) = None
        fused_qkv_4 = torch._C._nn.linear(
            layernorm_output_4,
            l_self_modules_h_modules_2_modules_self_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_h_modules_2_modules_self_attention_modules_query_key_value_parameters_bias_,
        )
        layernorm_output_4 = l_self_modules_h_modules_2_modules_self_attention_modules_query_key_value_parameters_weight_ = l_self_modules_h_modules_2_modules_self_attention_modules_query_key_value_parameters_bias_ = (None)
        fused_qkv_5 = fused_qkv_4.view(1, 19, 4, 3, 8)
        fused_qkv_4 = None
        getitem_14 = fused_qkv_5[(Ellipsis, 0, slice(None, None, None))]
        query_layer_4 = getitem_14.transpose(1, 2)
        getitem_14 = None
        getitem_15 = fused_qkv_5[(Ellipsis, 1, slice(None, None, None))]
        key_layer_4 = getitem_15.transpose(1, 2)
        getitem_15 = None
        getitem_16 = fused_qkv_5[(Ellipsis, 2, slice(None, None, None))]
        fused_qkv_5 = None
        value_layer_4 = getitem_16.transpose(1, 2)
        getitem_16 = None
        query_layer_5 = query_layer_4.reshape(4, -1, 8)
        query_layer_4 = None
        reshape_11 = key_layer_4.reshape(4, -1, 8)
        key_layer_5 = reshape_11.transpose(-1, -2)
        reshape_11 = None
        value_layer_5 = value_layer_4.reshape(4, -1, 8)
        attention_scores_2 = alibi_1.baddbmm(
            batch1=query_layer_5,
            batch2=key_layer_5,
            beta=1.0,
            alpha=0.35355339059327373,
        )
        query_layer_5 = key_layer_5 = None
        attn_weights_4 = attention_scores_2.view(1, 4, 19, -1)
        attention_scores_2 = None
        causal_mask_7 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        attn_weights_5 = attn_weights_4 + causal_mask_7
        attn_weights_4 = causal_mask_7 = None
        softmax_2 = torch.nn.functional.softmax(
            attn_weights_5, dim=-1, dtype=torch.float32
        )
        attn_weights_5 = None
        attention_probs_4 = softmax_2.to(torch.float32)
        softmax_2 = None
        attention_probs_5 = torch.nn.functional.dropout(
            attention_probs_4, 0.1, False, False
        )
        attention_probs_4 = None
        attention_probs_reshaped_2 = attention_probs_5.view(4, 19, -1)
        attention_probs_5 = None
        context_layer_4 = torch.bmm(attention_probs_reshaped_2, value_layer_5)
        attention_probs_reshaped_2 = value_layer_5 = None
        x_4 = context_layer_4.view(1, 4, 19, 8)
        context_layer_4 = None
        x_5 = x_4.permute(0, 2, 1, 3)
        x_4 = None
        context_layer_5 = x_5.reshape(1, 19, 32)
        x_5 = None
        output_tensor_2 = torch._C._nn.linear(
            context_layer_5,
            l_self_modules_h_modules_2_modules_self_attention_modules_dense_parameters_weight_,
            l_self_modules_h_modules_2_modules_self_attention_modules_dense_parameters_bias_,
        )
        context_layer_5 = l_self_modules_h_modules_2_modules_self_attention_modules_dense_parameters_weight_ = l_self_modules_h_modules_2_modules_self_attention_modules_dense_parameters_bias_ = (None)
        out_8 = torch.nn.functional.dropout(output_tensor_2, p=0.1, training=False)
        output_tensor_2 = None
        out_9 = out_7 + out_8
        out_7 = out_8 = None
        layernorm_output_5 = torch.nn.functional.layer_norm(
            out_9,
            (32,),
            l_self_modules_h_modules_2_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_h_modules_2_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_2_modules_post_attention_layernorm_parameters_weight_ = (
            l_self_modules_h_modules_2_modules_post_attention_layernorm_parameters_bias_
        ) = None
        linear_10 = torch._C._nn.linear(
            layernorm_output_5,
            l_self_modules_h_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_h_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layernorm_output_5 = l_self_modules_h_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_h_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        mul_14 = linear_10 * 0.5
        mul_15 = 0.79788456 * linear_10
        mul_16 = 0.044715 * linear_10
        mul_17 = mul_16 * linear_10
        mul_16 = linear_10 = None
        add_13 = 1 + mul_17
        mul_17 = None
        mul_18 = mul_15 * add_13
        mul_15 = add_13 = None
        tanh_2 = torch.tanh(mul_18)
        mul_18 = None
        add_14 = 1.0 + tanh_2
        tanh_2 = None
        hidden_states_3 = mul_14 * add_14
        mul_14 = add_14 = None
        intermediate_output_2 = torch._C._nn.linear(
            hidden_states_3,
            l_self_modules_h_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_h_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_3 = l_self_modules_h_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_h_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        out_10 = torch.nn.functional.dropout(
            intermediate_output_2, p=0.1, training=False
        )
        intermediate_output_2 = None
        out_11 = out_9 + out_10
        out_9 = out_10 = None
        layernorm_output_6 = torch.nn.functional.layer_norm(
            out_11,
            (32,),
            l_self_modules_h_modules_3_modules_input_layernorm_parameters_weight_,
            l_self_modules_h_modules_3_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_3_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_h_modules_3_modules_input_layernorm_parameters_bias_
        ) = None
        fused_qkv_6 = torch._C._nn.linear(
            layernorm_output_6,
            l_self_modules_h_modules_3_modules_self_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_h_modules_3_modules_self_attention_modules_query_key_value_parameters_bias_,
        )
        layernorm_output_6 = l_self_modules_h_modules_3_modules_self_attention_modules_query_key_value_parameters_weight_ = l_self_modules_h_modules_3_modules_self_attention_modules_query_key_value_parameters_bias_ = (None)
        fused_qkv_7 = fused_qkv_6.view(1, 19, 4, 3, 8)
        fused_qkv_6 = None
        getitem_18 = fused_qkv_7[(Ellipsis, 0, slice(None, None, None))]
        query_layer_6 = getitem_18.transpose(1, 2)
        getitem_18 = None
        getitem_19 = fused_qkv_7[(Ellipsis, 1, slice(None, None, None))]
        key_layer_6 = getitem_19.transpose(1, 2)
        getitem_19 = None
        getitem_20 = fused_qkv_7[(Ellipsis, 2, slice(None, None, None))]
        fused_qkv_7 = None
        value_layer_6 = getitem_20.transpose(1, 2)
        getitem_20 = None
        query_layer_7 = query_layer_6.reshape(4, -1, 8)
        query_layer_6 = None
        reshape_15 = key_layer_6.reshape(4, -1, 8)
        key_layer_7 = reshape_15.transpose(-1, -2)
        reshape_15 = None
        value_layer_7 = value_layer_6.reshape(4, -1, 8)
        attention_scores_3 = alibi_1.baddbmm(
            batch1=query_layer_7,
            batch2=key_layer_7,
            beta=1.0,
            alpha=0.35355339059327373,
        )
        query_layer_7 = key_layer_7 = None
        attn_weights_6 = attention_scores_3.view(1, 4, 19, -1)
        attention_scores_3 = None
        causal_mask_8 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        attn_weights_7 = attn_weights_6 + causal_mask_8
        attn_weights_6 = causal_mask_8 = None
        softmax_3 = torch.nn.functional.softmax(
            attn_weights_7, dim=-1, dtype=torch.float32
        )
        attn_weights_7 = None
        attention_probs_6 = softmax_3.to(torch.float32)
        softmax_3 = None
        attention_probs_7 = torch.nn.functional.dropout(
            attention_probs_6, 0.1, False, False
        )
        attention_probs_6 = None
        attention_probs_reshaped_3 = attention_probs_7.view(4, 19, -1)
        attention_probs_7 = None
        context_layer_6 = torch.bmm(attention_probs_reshaped_3, value_layer_7)
        attention_probs_reshaped_3 = value_layer_7 = None
        x_6 = context_layer_6.view(1, 4, 19, 8)
        context_layer_6 = None
        x_7 = x_6.permute(0, 2, 1, 3)
        x_6 = None
        context_layer_7 = x_7.reshape(1, 19, 32)
        x_7 = None
        output_tensor_3 = torch._C._nn.linear(
            context_layer_7,
            l_self_modules_h_modules_3_modules_self_attention_modules_dense_parameters_weight_,
            l_self_modules_h_modules_3_modules_self_attention_modules_dense_parameters_bias_,
        )
        context_layer_7 = l_self_modules_h_modules_3_modules_self_attention_modules_dense_parameters_weight_ = l_self_modules_h_modules_3_modules_self_attention_modules_dense_parameters_bias_ = (None)
        out_12 = torch.nn.functional.dropout(output_tensor_3, p=0.1, training=False)
        output_tensor_3 = None
        out_13 = out_11 + out_12
        out_11 = out_12 = None
        layernorm_output_7 = torch.nn.functional.layer_norm(
            out_13,
            (32,),
            l_self_modules_h_modules_3_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_h_modules_3_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_3_modules_post_attention_layernorm_parameters_weight_ = (
            l_self_modules_h_modules_3_modules_post_attention_layernorm_parameters_bias_
        ) = None
        linear_14 = torch._C._nn.linear(
            layernorm_output_7,
            l_self_modules_h_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_h_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layernorm_output_7 = l_self_modules_h_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_h_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        mul_20 = linear_14 * 0.5
        mul_21 = 0.79788456 * linear_14
        mul_22 = 0.044715 * linear_14
        mul_23 = mul_22 * linear_14
        mul_22 = linear_14 = None
        add_18 = 1 + mul_23
        mul_23 = None
        mul_24 = mul_21 * add_18
        mul_21 = add_18 = None
        tanh_3 = torch.tanh(mul_24)
        mul_24 = None
        add_19 = 1.0 + tanh_3
        tanh_3 = None
        hidden_states_4 = mul_20 * add_19
        mul_20 = add_19 = None
        intermediate_output_3 = torch._C._nn.linear(
            hidden_states_4,
            l_self_modules_h_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_h_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_4 = l_self_modules_h_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_h_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        out_14 = torch.nn.functional.dropout(
            intermediate_output_3, p=0.1, training=False
        )
        intermediate_output_3 = None
        out_15 = out_13 + out_14
        out_13 = out_14 = None
        layernorm_output_8 = torch.nn.functional.layer_norm(
            out_15,
            (32,),
            l_self_modules_h_modules_4_modules_input_layernorm_parameters_weight_,
            l_self_modules_h_modules_4_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_4_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_h_modules_4_modules_input_layernorm_parameters_bias_
        ) = None
        fused_qkv_8 = torch._C._nn.linear(
            layernorm_output_8,
            l_self_modules_h_modules_4_modules_self_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_h_modules_4_modules_self_attention_modules_query_key_value_parameters_bias_,
        )
        layernorm_output_8 = l_self_modules_h_modules_4_modules_self_attention_modules_query_key_value_parameters_weight_ = l_self_modules_h_modules_4_modules_self_attention_modules_query_key_value_parameters_bias_ = (None)
        fused_qkv_9 = fused_qkv_8.view(1, 19, 4, 3, 8)
        fused_qkv_8 = None
        getitem_22 = fused_qkv_9[(Ellipsis, 0, slice(None, None, None))]
        query_layer_8 = getitem_22.transpose(1, 2)
        getitem_22 = None
        getitem_23 = fused_qkv_9[(Ellipsis, 1, slice(None, None, None))]
        key_layer_8 = getitem_23.transpose(1, 2)
        getitem_23 = None
        getitem_24 = fused_qkv_9[(Ellipsis, 2, slice(None, None, None))]
        fused_qkv_9 = None
        value_layer_8 = getitem_24.transpose(1, 2)
        getitem_24 = None
        query_layer_9 = query_layer_8.reshape(4, -1, 8)
        query_layer_8 = None
        reshape_19 = key_layer_8.reshape(4, -1, 8)
        key_layer_9 = reshape_19.transpose(-1, -2)
        reshape_19 = None
        value_layer_9 = value_layer_8.reshape(4, -1, 8)
        attention_scores_4 = alibi_1.baddbmm(
            batch1=query_layer_9,
            batch2=key_layer_9,
            beta=1.0,
            alpha=0.35355339059327373,
        )
        alibi_1 = query_layer_9 = key_layer_9 = None
        attn_weights_8 = attention_scores_4.view(1, 4, 19, -1)
        attention_scores_4 = None
        causal_mask_9 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        causal_mask_4 = None
        attn_weights_9 = attn_weights_8 + causal_mask_9
        attn_weights_8 = causal_mask_9 = None
        softmax_4 = torch.nn.functional.softmax(
            attn_weights_9, dim=-1, dtype=torch.float32
        )
        attn_weights_9 = None
        attention_probs_8 = softmax_4.to(torch.float32)
        softmax_4 = None
        attention_probs_9 = torch.nn.functional.dropout(
            attention_probs_8, 0.1, False, False
        )
        attention_probs_8 = None
        attention_probs_reshaped_4 = attention_probs_9.view(4, 19, -1)
        attention_probs_9 = None
        context_layer_8 = torch.bmm(attention_probs_reshaped_4, value_layer_9)
        attention_probs_reshaped_4 = value_layer_9 = None
        x_8 = context_layer_8.view(1, 4, 19, 8)
        context_layer_8 = None
        x_9 = x_8.permute(0, 2, 1, 3)
        x_8 = None
        context_layer_9 = x_9.reshape(1, 19, 32)
        x_9 = None
        output_tensor_4 = torch._C._nn.linear(
            context_layer_9,
            l_self_modules_h_modules_4_modules_self_attention_modules_dense_parameters_weight_,
            l_self_modules_h_modules_4_modules_self_attention_modules_dense_parameters_bias_,
        )
        context_layer_9 = l_self_modules_h_modules_4_modules_self_attention_modules_dense_parameters_weight_ = l_self_modules_h_modules_4_modules_self_attention_modules_dense_parameters_bias_ = (None)
        out_16 = torch.nn.functional.dropout(output_tensor_4, p=0.1, training=False)
        output_tensor_4 = None
        out_17 = out_15 + out_16
        out_15 = out_16 = None
        layernorm_output_9 = torch.nn.functional.layer_norm(
            out_17,
            (32,),
            l_self_modules_h_modules_4_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_h_modules_4_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_4_modules_post_attention_layernorm_parameters_weight_ = (
            l_self_modules_h_modules_4_modules_post_attention_layernorm_parameters_bias_
        ) = None
        linear_18 = torch._C._nn.linear(
            layernorm_output_9,
            l_self_modules_h_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_h_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layernorm_output_9 = l_self_modules_h_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_h_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        mul_26 = linear_18 * 0.5
        mul_27 = 0.79788456 * linear_18
        mul_28 = 0.044715 * linear_18
        mul_29 = mul_28 * linear_18
        mul_28 = linear_18 = None
        add_23 = 1 + mul_29
        mul_29 = None
        mul_30 = mul_27 * add_23
        mul_27 = add_23 = None
        tanh_4 = torch.tanh(mul_30)
        mul_30 = None
        add_24 = 1.0 + tanh_4
        tanh_4 = None
        hidden_states_5 = mul_26 * add_24
        mul_26 = add_24 = None
        intermediate_output_4 = torch._C._nn.linear(
            hidden_states_5,
            l_self_modules_h_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_h_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_5 = l_self_modules_h_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_h_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        out_18 = torch.nn.functional.dropout(
            intermediate_output_4, p=0.1, training=False
        )
        intermediate_output_4 = None
        out_19 = out_17 + out_18
        out_17 = out_18 = None
        hidden_states_6 = torch.nn.functional.layer_norm(
            out_19,
            (32,),
            l_self_modules_ln_f_parameters_weight_,
            l_self_modules_ln_f_parameters_bias_,
            1e-05,
        )
        out_19 = (
            l_self_modules_ln_f_parameters_weight_
        ) = l_self_modules_ln_f_parameters_bias_ = None
        return (
            value_layer,
            key_layer,
            value_layer_2,
            key_layer_2,
            value_layer_4,
            key_layer_4,
            value_layer_6,
            key_layer_6,
            value_layer_8,
            key_layer_8,
            hidden_states_6,
        )
