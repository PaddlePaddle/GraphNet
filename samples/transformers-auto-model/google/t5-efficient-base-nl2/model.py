import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_input_ids_: torch.Tensor,
        L_self_modules_embed_tokens_parameters_weight_: torch.nn.parameter.Parameter,
        L_attention_mask_: torch.Tensor,
        L_self_modules_block_modules_0_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_relative_attention_bias_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_0_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_0_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_0_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
    ):
        l_input_ids_ = L_input_ids_
        l_self_modules_embed_tokens_parameters_weight_ = (
            L_self_modules_embed_tokens_parameters_weight_
        )
        l_attention_mask_ = L_attention_mask_
        l_self_modules_block_modules_0_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_0_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_0_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_0_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_0_modules_layer_modules_0_modules_self_attention_modules_relative_attention_bias_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_relative_attention_bias_parameters_weight_
        l_self_modules_block_modules_0_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_0_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_0_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_0_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_final_layer_norm_parameters_weight_
        )
        input_ids = l_input_ids_.view(-1, 12)
        l_input_ids_ = None
        inputs_embeds = torch.nn.functional.embedding(
            input_ids,
            l_self_modules_embed_tokens_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        input_ids = l_self_modules_embed_tokens_parameters_weight_ = None
        cache_position = torch.arange(0, 12, device=device(type="cuda", index=0))
        causal_mask = l_attention_mask_[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        l_attention_mask_ = None
        causal_mask_1 = causal_mask.to(dtype=torch.float32)
        causal_mask = None
        sub = 1.0 - causal_mask_1
        causal_mask_1 = None
        causal_mask_2 = sub * -3.4028234663852886e38
        sub = None
        hidden_states = torch.nn.functional.dropout(inputs_embeds, 0.1, False, False)
        inputs_embeds = None
        to_1 = hidden_states.to(torch.float32)
        pow_1 = to_1.pow(2)
        to_1 = None
        variance = pow_1.mean(-1, keepdim=True)
        pow_1 = None
        add = variance + 1e-06
        variance = None
        rsqrt = torch.rsqrt(add)
        add = None
        hidden_states_1 = hidden_states * rsqrt
        rsqrt = None
        normed_hidden_states = (
            l_self_modules_block_modules_0_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_1
        )
        l_self_modules_block_modules_0_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_1
        ) = None
        query_states = torch._C._nn.linear(
            normed_hidden_states,
            l_self_modules_block_modules_0_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_0_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_1 = query_states.view(1, -1, 12, 64)
        query_states = None
        query_states_1 = view_1.transpose(1, 2)
        view_1 = None
        key_states = torch._C._nn.linear(
            normed_hidden_states,
            l_self_modules_block_modules_0_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_0_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states = torch._C._nn.linear(
            normed_hidden_states,
            l_self_modules_block_modules_0_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states = l_self_modules_block_modules_0_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_2 = key_states.view(1, -1, 12, 64)
        key_states = None
        key_states_1 = view_2.transpose(1, 2)
        view_2 = None
        view_3 = value_states.view(1, -1, 12, 64)
        value_states = None
        value_states_1 = view_3.transpose(1, 2)
        view_3 = None
        transpose_3 = key_states_1.transpose(3, 2)
        key_states_1 = None
        scores = torch.matmul(query_states_1, transpose_3)
        query_states_1 = transpose_3 = None
        getitem_1 = cache_position[-1]
        real_seq_length = getitem_1 + 1
        getitem_1 = real_seq_length = None
        getitem_2 = cache_position[(slice(None, None, None), None)]
        cache_position = None
        context_position = getitem_2.to(device(type="cuda", index=0))
        getitem_2 = None
        arange_1 = torch.arange(
            12, dtype=torch.int64, device=device(type="cuda", index=0)
        )
        memory_position = arange_1[(None, slice(None, None, None))]
        arange_1 = None
        relative_position = memory_position - context_position
        memory_position = context_position = None
        gt = relative_position > 0
        to_3 = gt.to(torch.int64)
        gt = None
        mul_3 = to_3 * 16
        to_3 = None
        relative_buckets = 0 + mul_3
        mul_3 = None
        relative_position_1 = torch.abs(relative_position)
        relative_position = None
        is_small = relative_position_1 < 8
        float_1 = relative_position_1.float()
        truediv = float_1 / 8
        float_1 = None
        log = torch.log(truediv)
        truediv = None
        truediv_1 = log / 2.772588722239781
        log = None
        mul_4 = truediv_1 * 8
        truediv_1 = None
        to_4 = mul_4.to(torch.int64)
        mul_4 = None
        relative_position_if_large = 8 + to_4
        to_4 = None
        full_like = torch.full_like(relative_position_if_large, 15)
        relative_position_if_large_1 = torch.min(relative_position_if_large, full_like)
        relative_position_if_large = full_like = None
        where = torch.where(is_small, relative_position_1, relative_position_if_large_1)
        is_small = relative_position_1 = relative_position_if_large_1 = None
        relative_buckets += where
        relative_buckets_1 = relative_buckets
        relative_buckets = where = None
        values = torch.nn.functional.embedding(
            relative_buckets_1,
            l_self_modules_block_modules_0_modules_layer_modules_0_modules_self_attention_modules_relative_attention_bias_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        relative_buckets_1 = l_self_modules_block_modules_0_modules_layer_modules_0_modules_self_attention_modules_relative_attention_bias_parameters_weight_ = (None)
        permute = values.permute([2, 0, 1])
        values = None
        values_1 = permute.unsqueeze(0)
        permute = None
        position_bias = values_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(-12, None, None),
                slice(None, None, None),
            )
        ]
        values_1 = None
        causal_mask_3 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 12, None),
            )
        ]
        causal_mask_2 = None
        position_bias_1 = position_bias + causal_mask_3
        position_bias = causal_mask_3 = None
        scores += position_bias_1
        scores_1 = scores
        scores = None
        float_2 = scores_1.float()
        softmax = torch.nn.functional.softmax(float_2, dim=-1)
        float_2 = None
        attn_weights = softmax.type_as(scores_1)
        softmax = scores_1 = None
        attn_weights_1 = torch.nn.functional.dropout(
            attn_weights, p=0.1, training=False
        )
        attn_weights = None
        attn_output = torch.matmul(attn_weights_1, value_states_1)
        attn_weights_1 = value_states_1 = None
        transpose_4 = attn_output.transpose(1, 2)
        attn_output = None
        attn_output_1 = transpose_4.contiguous()
        transpose_4 = None
        attn_output_2 = attn_output_1.view(1, -1, 768)
        attn_output_1 = None
        attn_output_3 = torch._C._nn.linear(
            attn_output_2,
            l_self_modules_block_modules_0_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_2 = l_self_modules_block_modules_0_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_2 = torch.nn.functional.dropout(attn_output_3, 0.1, False, False)
        attn_output_3 = None
        hidden_states_2 = hidden_states + dropout_2
        hidden_states = dropout_2 = None
        to_5 = hidden_states_2.to(torch.float32)
        pow_2 = to_5.pow(2)
        to_5 = None
        variance_1 = pow_2.mean(-1, keepdim=True)
        pow_2 = None
        add_6 = variance_1 + 1e-06
        variance_1 = None
        rsqrt_1 = torch.rsqrt(add_6)
        add_6 = None
        hidden_states_3 = hidden_states_2 * rsqrt_1
        rsqrt_1 = None
        forwarded_states = (
            l_self_modules_block_modules_0_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_3
        )
        l_self_modules_block_modules_0_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_3
        ) = None
        hidden_states_4 = torch._C._nn.linear(
            forwarded_states,
            l_self_modules_block_modules_0_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states = l_self_modules_block_modules_0_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_5 = torch.nn.functional.relu(hidden_states_4, inplace=False)
        hidden_states_4 = None
        hidden_states_6 = torch.nn.functional.dropout(
            hidden_states_5, 0.1, False, False
        )
        hidden_states_5 = None
        hidden_states_7 = torch._C._nn.linear(
            hidden_states_6,
            l_self_modules_block_modules_0_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_6 = l_self_modules_block_modules_0_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_4 = torch.nn.functional.dropout(hidden_states_7, 0.1, False, False)
        hidden_states_7 = None
        hidden_states_8 = hidden_states_2 + dropout_4
        hidden_states_2 = dropout_4 = None
        to_6 = hidden_states_8.to(torch.float32)
        pow_3 = to_6.pow(2)
        to_6 = None
        variance_2 = pow_3.mean(-1, keepdim=True)
        pow_3 = None
        add_8 = variance_2 + 1e-06
        variance_2 = None
        rsqrt_2 = torch.rsqrt(add_8)
        add_8 = None
        hidden_states_9 = hidden_states_8 * rsqrt_2
        rsqrt_2 = None
        normed_hidden_states_1 = (
            l_self_modules_block_modules_1_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_9
        )
        l_self_modules_block_modules_1_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_9
        ) = None
        query_states_2 = torch._C._nn.linear(
            normed_hidden_states_1,
            l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_5 = query_states_2.view(1, -1, 12, 64)
        query_states_2 = None
        query_states_3 = view_5.transpose(1, 2)
        view_5 = None
        key_states_2 = torch._C._nn.linear(
            normed_hidden_states_1,
            l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_2 = torch._C._nn.linear(
            normed_hidden_states_1,
            l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_1 = l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_6 = key_states_2.view(1, -1, 12, 64)
        key_states_2 = None
        key_states_3 = view_6.transpose(1, 2)
        view_6 = None
        view_7 = value_states_2.view(1, -1, 12, 64)
        value_states_2 = None
        value_states_3 = view_7.transpose(1, 2)
        view_7 = None
        transpose_8 = key_states_3.transpose(3, 2)
        key_states_3 = None
        scores_2 = torch.matmul(query_states_3, transpose_8)
        query_states_3 = transpose_8 = None
        scores_2 += position_bias_1
        scores_3 = scores_2
        scores_2 = position_bias_1 = None
        float_3 = scores_3.float()
        softmax_1 = torch.nn.functional.softmax(float_3, dim=-1)
        float_3 = None
        attn_weights_2 = softmax_1.type_as(scores_3)
        softmax_1 = scores_3 = None
        attn_weights_3 = torch.nn.functional.dropout(
            attn_weights_2, p=0.1, training=False
        )
        attn_weights_2 = None
        attn_output_4 = torch.matmul(attn_weights_3, value_states_3)
        attn_weights_3 = value_states_3 = None
        transpose_9 = attn_output_4.transpose(1, 2)
        attn_output_4 = None
        attn_output_5 = transpose_9.contiguous()
        transpose_9 = None
        attn_output_6 = attn_output_5.view(1, -1, 768)
        attn_output_5 = None
        attn_output_7 = torch._C._nn.linear(
            attn_output_6,
            l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_6 = l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_6 = torch.nn.functional.dropout(attn_output_7, 0.1, False, False)
        attn_output_7 = None
        hidden_states_10 = hidden_states_8 + dropout_6
        hidden_states_8 = dropout_6 = None
        to_7 = hidden_states_10.to(torch.float32)
        pow_4 = to_7.pow(2)
        to_7 = None
        variance_3 = pow_4.mean(-1, keepdim=True)
        pow_4 = None
        add_10 = variance_3 + 1e-06
        variance_3 = None
        rsqrt_3 = torch.rsqrt(add_10)
        add_10 = None
        hidden_states_11 = hidden_states_10 * rsqrt_3
        rsqrt_3 = None
        forwarded_states_1 = (
            l_self_modules_block_modules_1_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_11
        )
        l_self_modules_block_modules_1_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_11
        ) = None
        hidden_states_12 = torch._C._nn.linear(
            forwarded_states_1,
            l_self_modules_block_modules_1_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_1 = l_self_modules_block_modules_1_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_13 = torch.nn.functional.relu(hidden_states_12, inplace=False)
        hidden_states_12 = None
        hidden_states_14 = torch.nn.functional.dropout(
            hidden_states_13, 0.1, False, False
        )
        hidden_states_13 = None
        hidden_states_15 = torch._C._nn.linear(
            hidden_states_14,
            l_self_modules_block_modules_1_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_14 = l_self_modules_block_modules_1_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_8 = torch.nn.functional.dropout(hidden_states_15, 0.1, False, False)
        hidden_states_15 = None
        hidden_states_16 = hidden_states_10 + dropout_8
        hidden_states_10 = dropout_8 = None
        to_8 = hidden_states_16.to(torch.float32)
        pow_5 = to_8.pow(2)
        to_8 = None
        variance_4 = pow_5.mean(-1, keepdim=True)
        pow_5 = None
        add_12 = variance_4 + 1e-06
        variance_4 = None
        rsqrt_4 = torch.rsqrt(add_12)
        add_12 = None
        hidden_states_17 = hidden_states_16 * rsqrt_4
        hidden_states_16 = rsqrt_4 = None
        hidden_states_18 = (
            l_self_modules_final_layer_norm_parameters_weight_ * hidden_states_17
        )
        l_self_modules_final_layer_norm_parameters_weight_ = hidden_states_17 = None
        hidden_states_19 = torch.nn.functional.dropout(
            hidden_states_18, 0.1, False, False
        )
        hidden_states_18 = None
        return (hidden_states_19,)
