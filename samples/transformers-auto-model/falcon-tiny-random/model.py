import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_input_ids_: torch.Tensor,
        L_self_modules_transformer_modules_word_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_attention_mask_: torch.Tensor,
        L_self_modules_transformer_modules_h_modules_0_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_self_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_self_attention_modules_maybe_rotary_buffers_inv_freq_: torch.Tensor,
        L_self_modules_transformer_modules_h_modules_0_modules_self_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_self_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_self_attention_modules_maybe_rotary_buffers_inv_freq_: torch.Tensor,
        L_self_modules_transformer_modules_h_modules_1_modules_self_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_ln_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_ln_f_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_input_ids_ = L_input_ids_
        l_self_modules_transformer_modules_word_embeddings_parameters_weight_ = (
            L_self_modules_transformer_modules_word_embeddings_parameters_weight_
        )
        l_attention_mask_ = L_attention_mask_
        l_self_modules_transformer_modules_h_modules_0_modules_input_layernorm_parameters_weight_ = L_self_modules_transformer_modules_h_modules_0_modules_input_layernorm_parameters_weight_
        l_self_modules_transformer_modules_h_modules_0_modules_input_layernorm_parameters_bias_ = L_self_modules_transformer_modules_h_modules_0_modules_input_layernorm_parameters_bias_
        l_self_modules_transformer_modules_h_modules_0_modules_self_attention_modules_query_key_value_parameters_weight_ = L_self_modules_transformer_modules_h_modules_0_modules_self_attention_modules_query_key_value_parameters_weight_
        l_self_modules_transformer_modules_h_modules_0_modules_self_attention_modules_maybe_rotary_buffers_inv_freq_ = L_self_modules_transformer_modules_h_modules_0_modules_self_attention_modules_maybe_rotary_buffers_inv_freq_
        l_self_modules_transformer_modules_h_modules_0_modules_self_attention_modules_dense_parameters_weight_ = L_self_modules_transformer_modules_h_modules_0_modules_self_attention_modules_dense_parameters_weight_
        l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_transformer_modules_h_modules_1_modules_input_layernorm_parameters_weight_ = L_self_modules_transformer_modules_h_modules_1_modules_input_layernorm_parameters_weight_
        l_self_modules_transformer_modules_h_modules_1_modules_input_layernorm_parameters_bias_ = L_self_modules_transformer_modules_h_modules_1_modules_input_layernorm_parameters_bias_
        l_self_modules_transformer_modules_h_modules_1_modules_self_attention_modules_query_key_value_parameters_weight_ = L_self_modules_transformer_modules_h_modules_1_modules_self_attention_modules_query_key_value_parameters_weight_
        l_self_modules_transformer_modules_h_modules_1_modules_self_attention_modules_maybe_rotary_buffers_inv_freq_ = L_self_modules_transformer_modules_h_modules_1_modules_self_attention_modules_maybe_rotary_buffers_inv_freq_
        l_self_modules_transformer_modules_h_modules_1_modules_self_attention_modules_dense_parameters_weight_ = L_self_modules_transformer_modules_h_modules_1_modules_self_attention_modules_dense_parameters_weight_
        l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_weight_
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
        attention_mask = l_attention_mask_.to(device(type="cpu"))
        l_attention_mask_ = None
        ones = torch.ones((19, 19), dtype=torch.bool, device=device(type="cpu"))
        mask = torch.triu(ones, diagonal=1)
        ones = None
        past_mask = torch.zeros((19, 0), dtype=torch.bool, device=device(type="cpu"))
        mask_1 = torch.cat([past_mask, mask], dim=-1)
        past_mask = mask = None
        getitem = mask_1[(None, None, slice(None, None, None), slice(None, None, None))]
        mask_1 = None
        expanded_mask = getitem.expand(1, 1, 19, 19)
        getitem = None
        getitem_1 = attention_mask[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        attention_mask = None
        to_1 = getitem_1.to(torch.bool)
        getitem_1 = None
        expanded_mask_1 = ~to_1
        to_1 = None
        expanded_attn_mask = expanded_mask_1.expand(1, 1, 19, 19)
        expanded_mask_1 = None
        combined_attention_mask = expanded_attn_mask | expanded_mask
        expanded_attn_mask = expanded_mask = None
        attention_layernorm_out = torch.nn.functional.layer_norm(
            inputs_embeds,
            (8,),
            l_self_modules_transformer_modules_h_modules_0_modules_input_layernorm_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_0_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_0_modules_input_layernorm_parameters_weight_ = l_self_modules_transformer_modules_h_modules_0_modules_input_layernorm_parameters_bias_ = (None)
        getattr_1 = (
            l_self_modules_transformer_modules_h_modules_0_modules_self_attention_modules_query_key_value_parameters_weight_.T
        )
        l_self_modules_transformer_modules_h_modules_0_modules_self_attention_modules_query_key_value_parameters_weight_ = (
            None
        )
        hidden_states = attention_layernorm_out @ getattr_1
        getattr_1 = None
        fused_qkv = hidden_states.view(1, 19, 4, 4)
        hidden_states = None
        query_layer = fused_qkv[
            (Ellipsis, slice(None, -2, None), slice(None, None, None))
        ]
        key_layer = fused_qkv[(Ellipsis, [-2], slice(None, None, None))]
        value_layer = fused_qkv[(Ellipsis, [-1], slice(None, None, None))]
        fused_qkv = None
        transpose = query_layer.transpose(1, 2)
        query_layer = None
        query_layer_1 = transpose.reshape(2, 19, 4)
        transpose = None
        transpose_1 = key_layer.transpose(1, 2)
        key_layer = None
        key_layer_1 = transpose_1.reshape(1, 19, 4)
        transpose_1 = None
        transpose_2 = value_layer.transpose(1, 2)
        value_layer = None
        value_layer_1 = transpose_2.reshape(1, 19, 4)
        transpose_2 = None
        t = torch.arange(19, device=device(type="cpu"), dtype=torch.float32)
        freqs = torch.functional.einsum(
            "i,j->ij",
            t,
            l_self_modules_transformer_modules_h_modules_0_modules_self_attention_modules_maybe_rotary_buffers_inv_freq_,
        )
        t = l_self_modules_transformer_modules_h_modules_0_modules_self_attention_modules_maybe_rotary_buffers_inv_freq_ = (None)
        cat_1 = torch.cat((freqs, freqs), dim=-1)
        freqs = None
        emb = cat_1.to(device(type="cpu"))
        cat_1 = None
        emb_1 = emb.float()
        emb = None
        cos = emb_1.cos()
        getitem_5 = cos[(None, slice(None, None, None), slice(None, None, None))]
        cos = None
        sin = emb_1.sin()
        emb_1 = None
        getitem_6 = sin[(None, slice(None, None, None), slice(None, None, None))]
        sin = None
        type_1 = getitem_5.type(torch.float16)
        getitem_5 = None
        type_2 = getitem_6.type(torch.float16)
        getitem_6 = None
        cos_1 = type_1[(slice(None, None, None), slice(0, 19, None))]
        sin_1 = type_2[(slice(None, None, None), slice(0, 19, None))]
        mul = query_layer_1 * cos_1
        x1 = query_layer_1[(Ellipsis, slice(None, 2, None))]
        x2 = query_layer_1[(Ellipsis, slice(2, None, None))]
        query_layer_1 = None
        neg = -x2
        x2 = None
        cat_2 = torch.cat((neg, x1), dim=-1)
        neg = x1 = None
        mul_1 = cat_2 * sin_1
        cat_2 = None
        query_layer_2 = mul + mul_1
        mul = mul_1 = None
        mul_2 = key_layer_1 * cos_1
        cos_1 = None
        x1_1 = key_layer_1[(Ellipsis, slice(None, 2, None))]
        x2_1 = key_layer_1[(Ellipsis, slice(2, None, None))]
        key_layer_1 = None
        neg_1 = -x2_1
        x2_1 = None
        cat_3 = torch.cat((neg_1, x1_1), dim=-1)
        neg_1 = x1_1 = None
        mul_3 = cat_3 * sin_1
        cat_3 = sin_1 = None
        key_layer_2 = mul_2 + mul_3
        mul_2 = mul_3 = None
        mul_4 = combined_attention_mask * 1.0
        masked_fill = mul_4.masked_fill(combined_attention_mask, -1000000000.0)
        mul_4 = None
        attention_mask_float = masked_fill.to(torch.float16)
        masked_fill = None
        query_layer_ = query_layer_2.reshape(1, 2, -1, 4)
        query_layer_2 = None
        key_layer_ = key_layer_2.reshape(1, 1, -1, 4)
        key_layer_2 = None
        value_layer_ = value_layer_1.reshape(1, 1, -1, 4)
        value_layer_1 = None
        attn_output = torch._C._nn.scaled_dot_product_attention(
            query_layer_,
            key_layer_,
            value_layer_,
            attention_mask_float,
            0.0,
            is_causal=False,
        )
        query_layer_ = key_layer_ = value_layer_ = attention_mask_float = None
        attn_output_1 = attn_output.view(1, 2, 19, 4)
        attn_output = None
        attn_output_2 = attn_output_1.permute(0, 2, 1, 3)
        attn_output_1 = None
        attn_output_3 = attn_output_2.reshape(1, 19, 8)
        attn_output_2 = None
        getattr_2 = (
            l_self_modules_transformer_modules_h_modules_0_modules_self_attention_modules_dense_parameters_weight_.T
        )
        l_self_modules_transformer_modules_h_modules_0_modules_self_attention_modules_dense_parameters_weight_ = (
            None
        )
        hidden_states_1 = attn_output_3 @ getattr_2
        attn_output_3 = getattr_2 = None
        getattr_3 = (
            l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_weight_.T
        )
        l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = (
            None
        )
        hidden_states_2 = attention_layernorm_out @ getattr_3
        attention_layernorm_out = getattr_3 = None
        x = torch._C._nn.gelu(hidden_states_2, approximate="none")
        hidden_states_2 = None
        getattr_4 = (
            l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_weight_.T
        )
        l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = (
            None
        )
        hidden_states_3 = x @ getattr_4
        x = getattr_4 = None
        hidden_states_3 += hidden_states_1
        mlp_output = hidden_states_3
        hidden_states_3 = hidden_states_1 = None
        out = torch.nn.functional.dropout(mlp_output, p=0.0, training=False)
        mlp_output = None
        out_1 = inputs_embeds + out
        inputs_embeds = out = None
        attention_layernorm_out_1 = torch.nn.functional.layer_norm(
            out_1,
            (8,),
            l_self_modules_transformer_modules_h_modules_1_modules_input_layernorm_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_1_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_1_modules_input_layernorm_parameters_weight_ = l_self_modules_transformer_modules_h_modules_1_modules_input_layernorm_parameters_bias_ = (None)
        getattr_5 = (
            l_self_modules_transformer_modules_h_modules_1_modules_self_attention_modules_query_key_value_parameters_weight_.T
        )
        l_self_modules_transformer_modules_h_modules_1_modules_self_attention_modules_query_key_value_parameters_weight_ = (
            None
        )
        hidden_states_4 = attention_layernorm_out_1 @ getattr_5
        getattr_5 = None
        fused_qkv_1 = hidden_states_4.view(1, 19, 4, 4)
        hidden_states_4 = None
        query_layer_3 = fused_qkv_1[
            (Ellipsis, slice(None, -2, None), slice(None, None, None))
        ]
        key_layer_3 = fused_qkv_1[(Ellipsis, [-2], slice(None, None, None))]
        value_layer_2 = fused_qkv_1[(Ellipsis, [-1], slice(None, None, None))]
        fused_qkv_1 = None
        transpose_3 = query_layer_3.transpose(1, 2)
        query_layer_3 = None
        query_layer_4 = transpose_3.reshape(2, 19, 4)
        transpose_3 = None
        transpose_4 = key_layer_3.transpose(1, 2)
        key_layer_3 = None
        key_layer_4 = transpose_4.reshape(1, 19, 4)
        transpose_4 = None
        transpose_5 = value_layer_2.transpose(1, 2)
        value_layer_2 = None
        value_layer_3 = transpose_5.reshape(1, 19, 4)
        transpose_5 = None
        t_1 = torch.arange(19, device=device(type="cpu"), dtype=torch.float32)
        freqs_1 = torch.functional.einsum(
            "i,j->ij",
            t_1,
            l_self_modules_transformer_modules_h_modules_1_modules_self_attention_modules_maybe_rotary_buffers_inv_freq_,
        )
        t_1 = l_self_modules_transformer_modules_h_modules_1_modules_self_attention_modules_maybe_rotary_buffers_inv_freq_ = (None)
        cat_4 = torch.cat((freqs_1, freqs_1), dim=-1)
        freqs_1 = None
        emb_2 = cat_4.to(device(type="cpu"))
        cat_4 = None
        emb_3 = emb_2.float()
        emb_2 = None
        cos_2 = emb_3.cos()
        getitem_16 = cos_2[(None, slice(None, None, None), slice(None, None, None))]
        cos_2 = None
        sin_2 = emb_3.sin()
        emb_3 = None
        getitem_17 = sin_2[(None, slice(None, None, None), slice(None, None, None))]
        sin_2 = None
        type_3 = getitem_16.type(torch.float16)
        getitem_16 = None
        type_4 = getitem_17.type(torch.float16)
        getitem_17 = None
        cos_3 = type_3[(slice(None, None, None), slice(0, 19, None))]
        sin_3 = type_4[(slice(None, None, None), slice(0, 19, None))]
        mul_5 = query_layer_4 * cos_3
        x1_2 = query_layer_4[(Ellipsis, slice(None, 2, None))]
        x2_2 = query_layer_4[(Ellipsis, slice(2, None, None))]
        query_layer_4 = None
        neg_2 = -x2_2
        x2_2 = None
        cat_5 = torch.cat((neg_2, x1_2), dim=-1)
        neg_2 = x1_2 = None
        mul_6 = cat_5 * sin_3
        cat_5 = None
        query_layer_5 = mul_5 + mul_6
        mul_5 = mul_6 = None
        mul_7 = key_layer_4 * cos_3
        cos_3 = None
        x1_3 = key_layer_4[(Ellipsis, slice(None, 2, None))]
        x2_3 = key_layer_4[(Ellipsis, slice(2, None, None))]
        key_layer_4 = None
        neg_3 = -x2_3
        x2_3 = None
        cat_6 = torch.cat((neg_3, x1_3), dim=-1)
        neg_3 = x1_3 = None
        mul_8 = cat_6 * sin_3
        cat_6 = sin_3 = None
        key_layer_5 = mul_7 + mul_8
        mul_7 = mul_8 = None
        mul_9 = combined_attention_mask * 1.0
        masked_fill_1 = mul_9.masked_fill(combined_attention_mask, -1000000000.0)
        mul_9 = combined_attention_mask = None
        attention_mask_float_1 = masked_fill_1.to(torch.float16)
        masked_fill_1 = None
        query_layer__1 = query_layer_5.reshape(1, 2, -1, 4)
        query_layer_5 = None
        key_layer__1 = key_layer_5.reshape(1, 1, -1, 4)
        key_layer_5 = None
        value_layer__1 = value_layer_3.reshape(1, 1, -1, 4)
        value_layer_3 = None
        attn_output_4 = torch._C._nn.scaled_dot_product_attention(
            query_layer__1,
            key_layer__1,
            value_layer__1,
            attention_mask_float_1,
            0.0,
            is_causal=False,
        )
        query_layer__1 = key_layer__1 = value_layer__1 = attention_mask_float_1 = None
        attn_output_5 = attn_output_4.view(1, 2, 19, 4)
        attn_output_4 = None
        attn_output_6 = attn_output_5.permute(0, 2, 1, 3)
        attn_output_5 = None
        attn_output_7 = attn_output_6.reshape(1, 19, 8)
        attn_output_6 = None
        getattr_6 = (
            l_self_modules_transformer_modules_h_modules_1_modules_self_attention_modules_dense_parameters_weight_.T
        )
        l_self_modules_transformer_modules_h_modules_1_modules_self_attention_modules_dense_parameters_weight_ = (
            None
        )
        hidden_states_5 = attn_output_7 @ getattr_6
        attn_output_7 = getattr_6 = None
        getattr_7 = (
            l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_weight_.T
        )
        l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = (
            None
        )
        hidden_states_6 = attention_layernorm_out_1 @ getattr_7
        attention_layernorm_out_1 = getattr_7 = None
        x_1 = torch._C._nn.gelu(hidden_states_6, approximate="none")
        hidden_states_6 = None
        getattr_8 = (
            l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_weight_.T
        )
        l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = (
            None
        )
        hidden_states_7 = x_1 @ getattr_8
        x_1 = getattr_8 = None
        hidden_states_7 += hidden_states_5
        mlp_output_1 = hidden_states_7
        hidden_states_7 = hidden_states_5 = None
        out_2 = torch.nn.functional.dropout(mlp_output_1, p=0.0, training=False)
        mlp_output_1 = None
        out_3 = out_1 + out_2
        out_1 = out_2 = None
        hidden_states_8 = torch.nn.functional.layer_norm(
            out_3,
            (8,),
            l_self_modules_transformer_modules_ln_f_parameters_weight_,
            l_self_modules_transformer_modules_ln_f_parameters_bias_,
            1e-05,
        )
        out_3 = (
            l_self_modules_transformer_modules_ln_f_parameters_weight_
        ) = l_self_modules_transformer_modules_ln_f_parameters_bias_ = None
        lm_logits = torch._C._nn.linear(
            hidden_states_8,
            l_self_modules_transformer_modules_word_embeddings_parameters_weight_,
            None,
        )
        hidden_states_8 = (
            l_self_modules_transformer_modules_word_embeddings_parameters_weight_
        ) = None
        return (type_2, type_1, type_4, type_3, lm_logits)
