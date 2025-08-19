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
        L_self_modules_block_modules_2_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_4_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_4_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_4_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_4_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_4_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_4_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_4_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_4_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_5_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_5_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_5_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_5_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_5_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_5_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_5_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_5_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
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
        l_self_modules_block_modules_2_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_4_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_4_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_4_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_4_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_4_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_4_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_4_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_4_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_4_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_4_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_4_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_4_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_4_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_4_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_4_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_4_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_5_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_5_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_5_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_5_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_5_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_5_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_5_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_5_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_5_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_5_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_5_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_5_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_5_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = L_self_modules_block_modules_5_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_
        l_self_modules_block_modules_5_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_5_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_
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
        scores_2 = None
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
        rsqrt_4 = None
        normed_hidden_states_2 = (
            l_self_modules_block_modules_2_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_17
        )
        l_self_modules_block_modules_2_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_17
        ) = None
        query_states_4 = torch._C._nn.linear(
            normed_hidden_states_2,
            l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_9 = query_states_4.view(1, -1, 12, 64)
        query_states_4 = None
        query_states_5 = view_9.transpose(1, 2)
        view_9 = None
        key_states_4 = torch._C._nn.linear(
            normed_hidden_states_2,
            l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_4 = torch._C._nn.linear(
            normed_hidden_states_2,
            l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_2 = l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_10 = key_states_4.view(1, -1, 12, 64)
        key_states_4 = None
        key_states_5 = view_10.transpose(1, 2)
        view_10 = None
        view_11 = value_states_4.view(1, -1, 12, 64)
        value_states_4 = None
        value_states_5 = view_11.transpose(1, 2)
        view_11 = None
        transpose_13 = key_states_5.transpose(3, 2)
        key_states_5 = None
        scores_4 = torch.matmul(query_states_5, transpose_13)
        query_states_5 = transpose_13 = None
        scores_4 += position_bias_1
        scores_5 = scores_4
        scores_4 = None
        float_4 = scores_5.float()
        softmax_2 = torch.nn.functional.softmax(float_4, dim=-1)
        float_4 = None
        attn_weights_4 = softmax_2.type_as(scores_5)
        softmax_2 = scores_5 = None
        attn_weights_5 = torch.nn.functional.dropout(
            attn_weights_4, p=0.1, training=False
        )
        attn_weights_4 = None
        attn_output_8 = torch.matmul(attn_weights_5, value_states_5)
        attn_weights_5 = value_states_5 = None
        transpose_14 = attn_output_8.transpose(1, 2)
        attn_output_8 = None
        attn_output_9 = transpose_14.contiguous()
        transpose_14 = None
        attn_output_10 = attn_output_9.view(1, -1, 768)
        attn_output_9 = None
        attn_output_11 = torch._C._nn.linear(
            attn_output_10,
            l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_10 = l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_10 = torch.nn.functional.dropout(attn_output_11, 0.1, False, False)
        attn_output_11 = None
        hidden_states_18 = hidden_states_16 + dropout_10
        hidden_states_16 = dropout_10 = None
        to_9 = hidden_states_18.to(torch.float32)
        pow_6 = to_9.pow(2)
        to_9 = None
        variance_5 = pow_6.mean(-1, keepdim=True)
        pow_6 = None
        add_14 = variance_5 + 1e-06
        variance_5 = None
        rsqrt_5 = torch.rsqrt(add_14)
        add_14 = None
        hidden_states_19 = hidden_states_18 * rsqrt_5
        rsqrt_5 = None
        forwarded_states_2 = (
            l_self_modules_block_modules_2_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_19
        )
        l_self_modules_block_modules_2_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_19
        ) = None
        hidden_states_20 = torch._C._nn.linear(
            forwarded_states_2,
            l_self_modules_block_modules_2_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_2 = l_self_modules_block_modules_2_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_21 = torch.nn.functional.relu(hidden_states_20, inplace=False)
        hidden_states_20 = None
        hidden_states_22 = torch.nn.functional.dropout(
            hidden_states_21, 0.1, False, False
        )
        hidden_states_21 = None
        hidden_states_23 = torch._C._nn.linear(
            hidden_states_22,
            l_self_modules_block_modules_2_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_22 = l_self_modules_block_modules_2_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_12 = torch.nn.functional.dropout(hidden_states_23, 0.1, False, False)
        hidden_states_23 = None
        hidden_states_24 = hidden_states_18 + dropout_12
        hidden_states_18 = dropout_12 = None
        to_10 = hidden_states_24.to(torch.float32)
        pow_7 = to_10.pow(2)
        to_10 = None
        variance_6 = pow_7.mean(-1, keepdim=True)
        pow_7 = None
        add_16 = variance_6 + 1e-06
        variance_6 = None
        rsqrt_6 = torch.rsqrt(add_16)
        add_16 = None
        hidden_states_25 = hidden_states_24 * rsqrt_6
        rsqrt_6 = None
        normed_hidden_states_3 = (
            l_self_modules_block_modules_3_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_25
        )
        l_self_modules_block_modules_3_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_25
        ) = None
        query_states_6 = torch._C._nn.linear(
            normed_hidden_states_3,
            l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_13 = query_states_6.view(1, -1, 12, 64)
        query_states_6 = None
        query_states_7 = view_13.transpose(1, 2)
        view_13 = None
        key_states_6 = torch._C._nn.linear(
            normed_hidden_states_3,
            l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_6 = torch._C._nn.linear(
            normed_hidden_states_3,
            l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_3 = l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_14 = key_states_6.view(1, -1, 12, 64)
        key_states_6 = None
        key_states_7 = view_14.transpose(1, 2)
        view_14 = None
        view_15 = value_states_6.view(1, -1, 12, 64)
        value_states_6 = None
        value_states_7 = view_15.transpose(1, 2)
        view_15 = None
        transpose_18 = key_states_7.transpose(3, 2)
        key_states_7 = None
        scores_6 = torch.matmul(query_states_7, transpose_18)
        query_states_7 = transpose_18 = None
        scores_6 += position_bias_1
        scores_7 = scores_6
        scores_6 = None
        float_5 = scores_7.float()
        softmax_3 = torch.nn.functional.softmax(float_5, dim=-1)
        float_5 = None
        attn_weights_6 = softmax_3.type_as(scores_7)
        softmax_3 = scores_7 = None
        attn_weights_7 = torch.nn.functional.dropout(
            attn_weights_6, p=0.1, training=False
        )
        attn_weights_6 = None
        attn_output_12 = torch.matmul(attn_weights_7, value_states_7)
        attn_weights_7 = value_states_7 = None
        transpose_19 = attn_output_12.transpose(1, 2)
        attn_output_12 = None
        attn_output_13 = transpose_19.contiguous()
        transpose_19 = None
        attn_output_14 = attn_output_13.view(1, -1, 768)
        attn_output_13 = None
        attn_output_15 = torch._C._nn.linear(
            attn_output_14,
            l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_14 = l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_14 = torch.nn.functional.dropout(attn_output_15, 0.1, False, False)
        attn_output_15 = None
        hidden_states_26 = hidden_states_24 + dropout_14
        hidden_states_24 = dropout_14 = None
        to_11 = hidden_states_26.to(torch.float32)
        pow_8 = to_11.pow(2)
        to_11 = None
        variance_7 = pow_8.mean(-1, keepdim=True)
        pow_8 = None
        add_18 = variance_7 + 1e-06
        variance_7 = None
        rsqrt_7 = torch.rsqrt(add_18)
        add_18 = None
        hidden_states_27 = hidden_states_26 * rsqrt_7
        rsqrt_7 = None
        forwarded_states_3 = (
            l_self_modules_block_modules_3_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_27
        )
        l_self_modules_block_modules_3_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_27
        ) = None
        hidden_states_28 = torch._C._nn.linear(
            forwarded_states_3,
            l_self_modules_block_modules_3_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_3 = l_self_modules_block_modules_3_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_29 = torch.nn.functional.relu(hidden_states_28, inplace=False)
        hidden_states_28 = None
        hidden_states_30 = torch.nn.functional.dropout(
            hidden_states_29, 0.1, False, False
        )
        hidden_states_29 = None
        hidden_states_31 = torch._C._nn.linear(
            hidden_states_30,
            l_self_modules_block_modules_3_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_30 = l_self_modules_block_modules_3_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_16 = torch.nn.functional.dropout(hidden_states_31, 0.1, False, False)
        hidden_states_31 = None
        hidden_states_32 = hidden_states_26 + dropout_16
        hidden_states_26 = dropout_16 = None
        to_12 = hidden_states_32.to(torch.float32)
        pow_9 = to_12.pow(2)
        to_12 = None
        variance_8 = pow_9.mean(-1, keepdim=True)
        pow_9 = None
        add_20 = variance_8 + 1e-06
        variance_8 = None
        rsqrt_8 = torch.rsqrt(add_20)
        add_20 = None
        hidden_states_33 = hidden_states_32 * rsqrt_8
        rsqrt_8 = None
        normed_hidden_states_4 = (
            l_self_modules_block_modules_4_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_33
        )
        l_self_modules_block_modules_4_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_33
        ) = None
        query_states_8 = torch._C._nn.linear(
            normed_hidden_states_4,
            l_self_modules_block_modules_4_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_4_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_17 = query_states_8.view(1, -1, 12, 64)
        query_states_8 = None
        query_states_9 = view_17.transpose(1, 2)
        view_17 = None
        key_states_8 = torch._C._nn.linear(
            normed_hidden_states_4,
            l_self_modules_block_modules_4_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_4_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_8 = torch._C._nn.linear(
            normed_hidden_states_4,
            l_self_modules_block_modules_4_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_4 = l_self_modules_block_modules_4_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_18 = key_states_8.view(1, -1, 12, 64)
        key_states_8 = None
        key_states_9 = view_18.transpose(1, 2)
        view_18 = None
        view_19 = value_states_8.view(1, -1, 12, 64)
        value_states_8 = None
        value_states_9 = view_19.transpose(1, 2)
        view_19 = None
        transpose_23 = key_states_9.transpose(3, 2)
        key_states_9 = None
        scores_8 = torch.matmul(query_states_9, transpose_23)
        query_states_9 = transpose_23 = None
        scores_8 += position_bias_1
        scores_9 = scores_8
        scores_8 = None
        float_6 = scores_9.float()
        softmax_4 = torch.nn.functional.softmax(float_6, dim=-1)
        float_6 = None
        attn_weights_8 = softmax_4.type_as(scores_9)
        softmax_4 = scores_9 = None
        attn_weights_9 = torch.nn.functional.dropout(
            attn_weights_8, p=0.1, training=False
        )
        attn_weights_8 = None
        attn_output_16 = torch.matmul(attn_weights_9, value_states_9)
        attn_weights_9 = value_states_9 = None
        transpose_24 = attn_output_16.transpose(1, 2)
        attn_output_16 = None
        attn_output_17 = transpose_24.contiguous()
        transpose_24 = None
        attn_output_18 = attn_output_17.view(1, -1, 768)
        attn_output_17 = None
        attn_output_19 = torch._C._nn.linear(
            attn_output_18,
            l_self_modules_block_modules_4_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_18 = l_self_modules_block_modules_4_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_18 = torch.nn.functional.dropout(attn_output_19, 0.1, False, False)
        attn_output_19 = None
        hidden_states_34 = hidden_states_32 + dropout_18
        hidden_states_32 = dropout_18 = None
        to_13 = hidden_states_34.to(torch.float32)
        pow_10 = to_13.pow(2)
        to_13 = None
        variance_9 = pow_10.mean(-1, keepdim=True)
        pow_10 = None
        add_22 = variance_9 + 1e-06
        variance_9 = None
        rsqrt_9 = torch.rsqrt(add_22)
        add_22 = None
        hidden_states_35 = hidden_states_34 * rsqrt_9
        rsqrt_9 = None
        forwarded_states_4 = (
            l_self_modules_block_modules_4_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_35
        )
        l_self_modules_block_modules_4_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_35
        ) = None
        hidden_states_36 = torch._C._nn.linear(
            forwarded_states_4,
            l_self_modules_block_modules_4_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_4 = l_self_modules_block_modules_4_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_37 = torch.nn.functional.relu(hidden_states_36, inplace=False)
        hidden_states_36 = None
        hidden_states_38 = torch.nn.functional.dropout(
            hidden_states_37, 0.1, False, False
        )
        hidden_states_37 = None
        hidden_states_39 = torch._C._nn.linear(
            hidden_states_38,
            l_self_modules_block_modules_4_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_38 = l_self_modules_block_modules_4_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_20 = torch.nn.functional.dropout(hidden_states_39, 0.1, False, False)
        hidden_states_39 = None
        hidden_states_40 = hidden_states_34 + dropout_20
        hidden_states_34 = dropout_20 = None
        to_14 = hidden_states_40.to(torch.float32)
        pow_11 = to_14.pow(2)
        to_14 = None
        variance_10 = pow_11.mean(-1, keepdim=True)
        pow_11 = None
        add_24 = variance_10 + 1e-06
        variance_10 = None
        rsqrt_10 = torch.rsqrt(add_24)
        add_24 = None
        hidden_states_41 = hidden_states_40 * rsqrt_10
        rsqrt_10 = None
        normed_hidden_states_5 = (
            l_self_modules_block_modules_5_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_41
        )
        l_self_modules_block_modules_5_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_41
        ) = None
        query_states_10 = torch._C._nn.linear(
            normed_hidden_states_5,
            l_self_modules_block_modules_5_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_5_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_21 = query_states_10.view(1, -1, 12, 64)
        query_states_10 = None
        query_states_11 = view_21.transpose(1, 2)
        view_21 = None
        key_states_10 = torch._C._nn.linear(
            normed_hidden_states_5,
            l_self_modules_block_modules_5_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_5_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_10 = torch._C._nn.linear(
            normed_hidden_states_5,
            l_self_modules_block_modules_5_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_5 = l_self_modules_block_modules_5_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_22 = key_states_10.view(1, -1, 12, 64)
        key_states_10 = None
        key_states_11 = view_22.transpose(1, 2)
        view_22 = None
        view_23 = value_states_10.view(1, -1, 12, 64)
        value_states_10 = None
        value_states_11 = view_23.transpose(1, 2)
        view_23 = None
        transpose_28 = key_states_11.transpose(3, 2)
        key_states_11 = None
        scores_10 = torch.matmul(query_states_11, transpose_28)
        query_states_11 = transpose_28 = None
        scores_10 += position_bias_1
        scores_11 = scores_10
        scores_10 = position_bias_1 = None
        float_7 = scores_11.float()
        softmax_5 = torch.nn.functional.softmax(float_7, dim=-1)
        float_7 = None
        attn_weights_10 = softmax_5.type_as(scores_11)
        softmax_5 = scores_11 = None
        attn_weights_11 = torch.nn.functional.dropout(
            attn_weights_10, p=0.1, training=False
        )
        attn_weights_10 = None
        attn_output_20 = torch.matmul(attn_weights_11, value_states_11)
        attn_weights_11 = value_states_11 = None
        transpose_29 = attn_output_20.transpose(1, 2)
        attn_output_20 = None
        attn_output_21 = transpose_29.contiguous()
        transpose_29 = None
        attn_output_22 = attn_output_21.view(1, -1, 768)
        attn_output_21 = None
        attn_output_23 = torch._C._nn.linear(
            attn_output_22,
            l_self_modules_block_modules_5_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_22 = l_self_modules_block_modules_5_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_22 = torch.nn.functional.dropout(attn_output_23, 0.1, False, False)
        attn_output_23 = None
        hidden_states_42 = hidden_states_40 + dropout_22
        hidden_states_40 = dropout_22 = None
        to_15 = hidden_states_42.to(torch.float32)
        pow_12 = to_15.pow(2)
        to_15 = None
        variance_11 = pow_12.mean(-1, keepdim=True)
        pow_12 = None
        add_26 = variance_11 + 1e-06
        variance_11 = None
        rsqrt_11 = torch.rsqrt(add_26)
        add_26 = None
        hidden_states_43 = hidden_states_42 * rsqrt_11
        rsqrt_11 = None
        forwarded_states_5 = (
            l_self_modules_block_modules_5_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_43
        )
        l_self_modules_block_modules_5_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_43
        ) = None
        hidden_states_44 = torch._C._nn.linear(
            forwarded_states_5,
            l_self_modules_block_modules_5_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_,
            None,
        )
        forwarded_states_5 = l_self_modules_block_modules_5_modules_layer_modules_1_modules_dense_relu_dense_modules_wi_parameters_weight_ = (None)
        hidden_states_45 = torch.nn.functional.relu(hidden_states_44, inplace=False)
        hidden_states_44 = None
        hidden_states_46 = torch.nn.functional.dropout(
            hidden_states_45, 0.1, False, False
        )
        hidden_states_45 = None
        hidden_states_47 = torch._C._nn.linear(
            hidden_states_46,
            l_self_modules_block_modules_5_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_46 = l_self_modules_block_modules_5_modules_layer_modules_1_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_24 = torch.nn.functional.dropout(hidden_states_47, 0.1, False, False)
        hidden_states_47 = None
        hidden_states_48 = hidden_states_42 + dropout_24
        hidden_states_42 = dropout_24 = None
        to_16 = hidden_states_48.to(torch.float32)
        pow_13 = to_16.pow(2)
        to_16 = None
        variance_12 = pow_13.mean(-1, keepdim=True)
        pow_13 = None
        add_28 = variance_12 + 1e-06
        variance_12 = None
        rsqrt_12 = torch.rsqrt(add_28)
        add_28 = None
        hidden_states_49 = hidden_states_48 * rsqrt_12
        hidden_states_48 = rsqrt_12 = None
        hidden_states_50 = (
            l_self_modules_final_layer_norm_parameters_weight_ * hidden_states_49
        )
        l_self_modules_final_layer_norm_parameters_weight_ = hidden_states_49 = None
        hidden_states_51 = torch.nn.functional.dropout(
            hidden_states_50, 0.1, False, False
        )
        hidden_states_50 = None
        return (hidden_states_51,)
