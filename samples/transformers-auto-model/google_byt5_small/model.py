import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_inputs_embeds_: torch.Tensor,
        L_encoder_hidden_states_: torch.Tensor,
        L_encoder_attention_mask_: torch.Tensor,
        L_self_modules_block_modules_0_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_relative_attention_bias_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_0_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_0_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_0_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_0_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_0_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_0_modules_layer_modules_2_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_0_modules_layer_modules_2_modules_DenseReluDense_modules_wi_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_0_modules_layer_modules_2_modules_DenseReluDense_modules_wi_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_0_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_2_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_2_modules_DenseReluDense_modules_wi_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_2_modules_DenseReluDense_modules_wi_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_1_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_2_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_2_modules_DenseReluDense_modules_wi_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_2_modules_DenseReluDense_modules_wi_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_2_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_2_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_2_modules_DenseReluDense_modules_wi_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_2_modules_DenseReluDense_modules_wi_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block_modules_3_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
    ):
        l_inputs_embeds_ = L_inputs_embeds_
        l_encoder_hidden_states_ = L_encoder_hidden_states_
        l_encoder_attention_mask_ = L_encoder_attention_mask_
        l_self_modules_block_modules_0_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_0_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_0_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_0_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_0_modules_layer_modules_0_modules_self_attention_modules_relative_attention_bias_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_relative_attention_bias_parameters_weight_
        l_self_modules_block_modules_0_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_0_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_0_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_0_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_0_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_0_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_0_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_2_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_0_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_0_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_2_modules_DenseReluDense_modules_wi_0_parameters_weight_
        l_self_modules_block_modules_0_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_1_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_2_modules_DenseReluDense_modules_wi_1_parameters_weight_
        l_self_modules_block_modules_0_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_0_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_2_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_0_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_2_modules_DenseReluDense_modules_wi_0_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_1_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_2_modules_DenseReluDense_modules_wi_1_parameters_weight_
        l_self_modules_block_modules_1_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_1_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_2_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_0_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_2_modules_DenseReluDense_modules_wi_0_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_1_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_2_modules_DenseReluDense_modules_wi_1_parameters_weight_
        l_self_modules_block_modules_2_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_2_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_2_modules_layer_norm_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_0_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_2_modules_DenseReluDense_modules_wi_0_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_1_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_2_modules_DenseReluDense_modules_wi_1_parameters_weight_
        l_self_modules_block_modules_3_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_ = L_self_modules_block_modules_3_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_
        l_self_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_final_layer_norm_parameters_weight_
        )
        cache_position = torch.arange(0, 1, device=device(type="cuda", index=0))
        causal_mask = torch.full(
            (1, 2),
            fill_value=-3.4028234663852886e38,
            dtype=torch.float32,
            device=device(type="cuda", index=0),
        )
        arange_1 = torch.arange(2, device=device(type="cuda", index=0))
        reshape = cache_position.reshape(-1, 1)
        gt = arange_1 > reshape
        arange_1 = reshape = None
        causal_mask *= gt
        causal_mask_1 = causal_mask
        causal_mask = gt = None
        getitem = causal_mask_1[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        causal_mask_1 = None
        causal_mask_2 = getitem.expand(1, 1, -1, -1)
        getitem = None
        encoder_extended_attention_mask = l_encoder_attention_mask_[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        l_encoder_attention_mask_ = None
        encoder_extended_attention_mask_1 = encoder_extended_attention_mask.to(
            dtype=torch.float32
        )
        encoder_extended_attention_mask = None
        sub = 1.0 - encoder_extended_attention_mask_1
        encoder_extended_attention_mask_1 = None
        encoder_extended_attention_mask_2 = sub * -3.4028234663852886e38
        sub = None
        hidden_states = torch.nn.functional.dropout(l_inputs_embeds_, 0.1, False, False)
        l_inputs_embeds_ = None
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
        view = query_states.view(1, -1, 6, 64)
        query_states = None
        query_states_1 = view.transpose(1, 2)
        view = None
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
        view_1 = key_states.view(1, -1, 6, 64)
        key_states = None
        key_states_1 = view_1.transpose(1, 2)
        view_1 = None
        view_2 = value_states.view(1, -1, 6, 64)
        value_states = None
        value_states_1 = view_2.transpose(1, 2)
        view_2 = None
        transpose_3 = key_states_1.transpose(3, 2)
        scores = torch.matmul(query_states_1, transpose_3)
        query_states_1 = transpose_3 = None
        getitem_2 = cache_position[-1]
        real_seq_length = getitem_2 + 1
        getitem_2 = real_seq_length = None
        getitem_3 = cache_position[(slice(None, None, None), None)]
        context_position = getitem_3.to(device(type="cuda", index=0))
        getitem_3 = None
        arange_2 = torch.arange(
            1, dtype=torch.int64, device=device(type="cuda", index=0)
        )
        memory_position = arange_2[(None, slice(None, None, None))]
        arange_2 = None
        relative_position = memory_position - context_position
        memory_position = context_position = None
        zeros_like = torch.zeros_like(relative_position)
        min_1 = torch.min(relative_position, zeros_like)
        relative_position = zeros_like = None
        relative_position_1 = -min_1
        min_1 = None
        is_small = relative_position_1 < 16
        float_1 = relative_position_1.float()
        truediv = float_1 / 16
        float_1 = None
        log = torch.log(truediv)
        truediv = None
        truediv_1 = log / 2.0794415416798357
        log = None
        mul_3 = truediv_1 * 16
        truediv_1 = None
        to_3 = mul_3.to(torch.int64)
        mul_3 = None
        relative_position_if_large = 16 + to_3
        to_3 = None
        full_like = torch.full_like(relative_position_if_large, 31)
        relative_position_if_large_1 = torch.min(relative_position_if_large, full_like)
        relative_position_if_large = full_like = None
        where = torch.where(is_small, relative_position_1, relative_position_if_large_1)
        is_small = relative_position_1 = relative_position_if_large_1 = None
        relative_buckets = 0 + where
        where = None
        values = torch.nn.functional.embedding(
            relative_buckets,
            l_self_modules_block_modules_0_modules_layer_modules_0_modules_self_attention_modules_relative_attention_bias_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        relative_buckets = l_self_modules_block_modules_0_modules_layer_modules_0_modules_self_attention_modules_relative_attention_bias_parameters_weight_ = (None)
        permute = values.permute([2, 0, 1])
        values = None
        values_1 = permute.unsqueeze(0)
        permute = None
        position_bias = values_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(-1, None, None),
                slice(None, None, None),
            )
        ]
        values_1 = None
        causal_mask_3 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
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
        attn_weights_1 = None
        transpose_4 = attn_output.transpose(1, 2)
        attn_output = None
        attn_output_1 = transpose_4.contiguous()
        transpose_4 = None
        attn_output_2 = attn_output_1.view(1, -1, 384)
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
        getitem_7 = cache_position[-1]
        real_seq_length_1 = getitem_7 + 1
        getitem_7 = real_seq_length_1 = None
        to_4 = hidden_states_2.to(torch.float32)
        pow_2 = to_4.pow(2)
        to_4 = None
        variance_1 = pow_2.mean(-1, keepdim=True)
        pow_2 = None
        add_7 = variance_1 + 1e-06
        variance_1 = None
        rsqrt_1 = torch.rsqrt(add_7)
        add_7 = None
        hidden_states_3 = hidden_states_2 * rsqrt_1
        rsqrt_1 = None
        normed_hidden_states_1 = (
            l_self_modules_block_modules_0_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_3
        )
        l_self_modules_block_modules_0_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_3
        ) = None
        query_states_2 = torch._C._nn.linear(
            normed_hidden_states_1,
            l_self_modules_block_modules_0_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_,
            None,
        )
        normed_hidden_states_1 = l_self_modules_block_modules_0_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_ = (None)
        view_4 = query_states_2.view(1, -1, 6, 64)
        query_states_2 = None
        query_states_3 = view_4.transpose(1, 2)
        view_4 = None
        key_states_2 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_block_modules_0_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_0_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_2 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_block_modules_0_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_0_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_ = (
            None
        )
        view_5 = key_states_2.view(1, -1, 6, 64)
        key_states_2 = None
        key_states_3 = view_5.transpose(1, 2)
        view_5 = None
        view_6 = value_states_2.view(1, -1, 6, 64)
        value_states_2 = None
        value_states_3 = view_6.transpose(1, 2)
        view_6 = None
        transpose_8 = key_states_3.transpose(3, 2)
        scores_2 = torch.matmul(query_states_3, transpose_8)
        query_states_3 = transpose_8 = None
        position_bias_2 = torch.zeros(
            (1, 6, 1, 16), device=device(type="cuda", index=0), dtype=torch.float32
        )
        causal_mask_4 = encoder_extended_attention_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 16, None),
            )
        ]
        encoder_extended_attention_mask_2 = None
        position_bias_3 = position_bias_2 + causal_mask_4
        position_bias_2 = causal_mask_4 = None
        scores_2 += position_bias_3
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
        attn_weights_3 = None
        transpose_9 = attn_output_4.transpose(1, 2)
        attn_output_4 = None
        attn_output_5 = transpose_9.contiguous()
        transpose_9 = None
        attn_output_6 = attn_output_5.view(1, -1, 384)
        attn_output_5 = None
        attn_output_7 = torch._C._nn.linear(
            attn_output_6,
            l_self_modules_block_modules_0_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_6 = l_self_modules_block_modules_0_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_ = (None)
        dropout_4 = torch.nn.functional.dropout(attn_output_7, 0.1, False, False)
        attn_output_7 = None
        layer_output = hidden_states_2 + dropout_4
        hidden_states_2 = dropout_4 = None
        to_5 = layer_output.to(torch.float32)
        pow_3 = to_5.pow(2)
        to_5 = None
        variance_2 = pow_3.mean(-1, keepdim=True)
        pow_3 = None
        add_10 = variance_2 + 1e-06
        variance_2 = None
        rsqrt_2 = torch.rsqrt(add_10)
        add_10 = None
        hidden_states_4 = layer_output * rsqrt_2
        rsqrt_2 = None
        forwarded_states = (
            l_self_modules_block_modules_0_modules_layer_modules_2_modules_layer_norm_parameters_weight_
            * hidden_states_4
        )
        l_self_modules_block_modules_0_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = (
            hidden_states_4
        ) = None
        linear_8 = torch._C._nn.linear(
            forwarded_states,
            l_self_modules_block_modules_0_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_0_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_0_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_0_parameters_weight_ = (
            None
        )
        mul_8 = 0.5 * linear_8
        pow_4 = torch.pow(linear_8, 3.0)
        mul_9 = 0.044715 * pow_4
        pow_4 = None
        add_11 = linear_8 + mul_9
        linear_8 = mul_9 = None
        mul_10 = 0.7978845608028654 * add_11
        add_11 = None
        tanh = torch.tanh(mul_10)
        mul_10 = None
        add_12 = 1.0 + tanh
        tanh = None
        hidden_gelu = mul_8 * add_12
        mul_8 = add_12 = None
        hidden_linear = torch._C._nn.linear(
            forwarded_states,
            l_self_modules_block_modules_0_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_1_parameters_weight_,
            None,
        )
        forwarded_states = l_self_modules_block_modules_0_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_1_parameters_weight_ = (None)
        hidden_states_5 = hidden_gelu * hidden_linear
        hidden_gelu = hidden_linear = None
        hidden_states_6 = torch.nn.functional.dropout(
            hidden_states_5, 0.1, False, False
        )
        hidden_states_5 = None
        hidden_states_7 = torch._C._nn.linear(
            hidden_states_6,
            l_self_modules_block_modules_0_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_6 = l_self_modules_block_modules_0_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_6 = torch.nn.functional.dropout(hidden_states_7, 0.1, False, False)
        hidden_states_7 = None
        hidden_states_8 = layer_output + dropout_6
        layer_output = dropout_6 = None
        to_6 = hidden_states_8.to(torch.float32)
        pow_5 = to_6.pow(2)
        to_6 = None
        variance_3 = pow_5.mean(-1, keepdim=True)
        pow_5 = None
        add_14 = variance_3 + 1e-06
        variance_3 = None
        rsqrt_3 = torch.rsqrt(add_14)
        add_14 = None
        hidden_states_9 = hidden_states_8 * rsqrt_3
        rsqrt_3 = None
        normed_hidden_states_2 = (
            l_self_modules_block_modules_1_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_9
        )
        l_self_modules_block_modules_1_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_9
        ) = None
        query_states_4 = torch._C._nn.linear(
            normed_hidden_states_2,
            l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_8 = query_states_4.view(1, -1, 6, 64)
        query_states_4 = None
        query_states_5 = view_8.transpose(1, 2)
        view_8 = None
        key_states_4 = torch._C._nn.linear(
            normed_hidden_states_2,
            l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_4 = torch._C._nn.linear(
            normed_hidden_states_2,
            l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_2 = l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_9 = key_states_4.view(1, -1, 6, 64)
        key_states_4 = None
        key_states_5 = view_9.transpose(1, 2)
        view_9 = None
        view_10 = value_states_4.view(1, -1, 6, 64)
        value_states_4 = None
        value_states_5 = view_10.transpose(1, 2)
        view_10 = None
        transpose_13 = key_states_5.transpose(3, 2)
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
        attn_weights_5 = None
        transpose_14 = attn_output_8.transpose(1, 2)
        attn_output_8 = None
        attn_output_9 = transpose_14.contiguous()
        transpose_14 = None
        attn_output_10 = attn_output_9.view(1, -1, 384)
        attn_output_9 = None
        attn_output_11 = torch._C._nn.linear(
            attn_output_10,
            l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_10 = l_self_modules_block_modules_1_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_8 = torch.nn.functional.dropout(attn_output_11, 0.1, False, False)
        attn_output_11 = None
        hidden_states_10 = hidden_states_8 + dropout_8
        hidden_states_8 = dropout_8 = None
        getitem_9 = cache_position[-1]
        add_16 = getitem_9 + 1
        getitem_9 = add_16 = None
        to_7 = hidden_states_10.to(torch.float32)
        pow_6 = to_7.pow(2)
        to_7 = None
        variance_4 = pow_6.mean(-1, keepdim=True)
        pow_6 = None
        add_17 = variance_4 + 1e-06
        variance_4 = None
        rsqrt_4 = torch.rsqrt(add_17)
        add_17 = None
        hidden_states_11 = hidden_states_10 * rsqrt_4
        rsqrt_4 = None
        normed_hidden_states_3 = (
            l_self_modules_block_modules_1_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_11
        )
        l_self_modules_block_modules_1_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_11
        ) = None
        query_states_6 = torch._C._nn.linear(
            normed_hidden_states_3,
            l_self_modules_block_modules_1_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_,
            None,
        )
        normed_hidden_states_3 = l_self_modules_block_modules_1_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_ = (None)
        view_12 = query_states_6.view(1, -1, 6, 64)
        query_states_6 = None
        query_states_7 = view_12.transpose(1, 2)
        view_12 = None
        key_states_6 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_block_modules_1_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_1_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_6 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_block_modules_1_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_1_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_ = (
            None
        )
        view_13 = key_states_6.view(1, -1, 6, 64)
        key_states_6 = None
        key_states_7 = view_13.transpose(1, 2)
        view_13 = None
        view_14 = value_states_6.view(1, -1, 6, 64)
        value_states_6 = None
        value_states_7 = view_14.transpose(1, 2)
        view_14 = None
        transpose_18 = key_states_7.transpose(3, 2)
        scores_6 = torch.matmul(query_states_7, transpose_18)
        query_states_7 = transpose_18 = None
        scores_6 += position_bias_3
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
        attn_weights_7 = None
        transpose_19 = attn_output_12.transpose(1, 2)
        attn_output_12 = None
        attn_output_13 = transpose_19.contiguous()
        transpose_19 = None
        attn_output_14 = attn_output_13.view(1, -1, 384)
        attn_output_13 = None
        attn_output_15 = torch._C._nn.linear(
            attn_output_14,
            l_self_modules_block_modules_1_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_14 = l_self_modules_block_modules_1_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_ = (None)
        dropout_10 = torch.nn.functional.dropout(attn_output_15, 0.1, False, False)
        attn_output_15 = None
        layer_output_1 = hidden_states_10 + dropout_10
        hidden_states_10 = dropout_10 = None
        to_8 = layer_output_1.to(torch.float32)
        pow_7 = to_8.pow(2)
        to_8 = None
        variance_5 = pow_7.mean(-1, keepdim=True)
        pow_7 = None
        add_19 = variance_5 + 1e-06
        variance_5 = None
        rsqrt_5 = torch.rsqrt(add_19)
        add_19 = None
        hidden_states_12 = layer_output_1 * rsqrt_5
        rsqrt_5 = None
        forwarded_states_1 = (
            l_self_modules_block_modules_1_modules_layer_modules_2_modules_layer_norm_parameters_weight_
            * hidden_states_12
        )
        l_self_modules_block_modules_1_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = (
            hidden_states_12
        ) = None
        linear_19 = torch._C._nn.linear(
            forwarded_states_1,
            l_self_modules_block_modules_1_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_0_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_1_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_0_parameters_weight_ = (
            None
        )
        mul_19 = 0.5 * linear_19
        pow_8 = torch.pow(linear_19, 3.0)
        mul_20 = 0.044715 * pow_8
        pow_8 = None
        add_20 = linear_19 + mul_20
        linear_19 = mul_20 = None
        mul_21 = 0.7978845608028654 * add_20
        add_20 = None
        tanh_1 = torch.tanh(mul_21)
        mul_21 = None
        add_21 = 1.0 + tanh_1
        tanh_1 = None
        hidden_gelu_1 = mul_19 * add_21
        mul_19 = add_21 = None
        hidden_linear_1 = torch._C._nn.linear(
            forwarded_states_1,
            l_self_modules_block_modules_1_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_1_parameters_weight_,
            None,
        )
        forwarded_states_1 = l_self_modules_block_modules_1_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_1_parameters_weight_ = (None)
        hidden_states_13 = hidden_gelu_1 * hidden_linear_1
        hidden_gelu_1 = hidden_linear_1 = None
        hidden_states_14 = torch.nn.functional.dropout(
            hidden_states_13, 0.1, False, False
        )
        hidden_states_13 = None
        hidden_states_15 = torch._C._nn.linear(
            hidden_states_14,
            l_self_modules_block_modules_1_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_14 = l_self_modules_block_modules_1_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_12 = torch.nn.functional.dropout(hidden_states_15, 0.1, False, False)
        hidden_states_15 = None
        hidden_states_16 = layer_output_1 + dropout_12
        layer_output_1 = dropout_12 = None
        to_9 = hidden_states_16.to(torch.float32)
        pow_9 = to_9.pow(2)
        to_9 = None
        variance_6 = pow_9.mean(-1, keepdim=True)
        pow_9 = None
        add_23 = variance_6 + 1e-06
        variance_6 = None
        rsqrt_6 = torch.rsqrt(add_23)
        add_23 = None
        hidden_states_17 = hidden_states_16 * rsqrt_6
        rsqrt_6 = None
        normed_hidden_states_4 = (
            l_self_modules_block_modules_2_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_17
        )
        l_self_modules_block_modules_2_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_17
        ) = None
        query_states_8 = torch._C._nn.linear(
            normed_hidden_states_4,
            l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_16 = query_states_8.view(1, -1, 6, 64)
        query_states_8 = None
        query_states_9 = view_16.transpose(1, 2)
        view_16 = None
        key_states_8 = torch._C._nn.linear(
            normed_hidden_states_4,
            l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_8 = torch._C._nn.linear(
            normed_hidden_states_4,
            l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_4 = l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_17 = key_states_8.view(1, -1, 6, 64)
        key_states_8 = None
        key_states_9 = view_17.transpose(1, 2)
        view_17 = None
        view_18 = value_states_8.view(1, -1, 6, 64)
        value_states_8 = None
        value_states_9 = view_18.transpose(1, 2)
        view_18 = None
        transpose_23 = key_states_9.transpose(3, 2)
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
        attn_weights_9 = None
        transpose_24 = attn_output_16.transpose(1, 2)
        attn_output_16 = None
        attn_output_17 = transpose_24.contiguous()
        transpose_24 = None
        attn_output_18 = attn_output_17.view(1, -1, 384)
        attn_output_17 = None
        attn_output_19 = torch._C._nn.linear(
            attn_output_18,
            l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_18 = l_self_modules_block_modules_2_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_14 = torch.nn.functional.dropout(attn_output_19, 0.1, False, False)
        attn_output_19 = None
        hidden_states_18 = hidden_states_16 + dropout_14
        hidden_states_16 = dropout_14 = None
        getitem_10 = cache_position[-1]
        add_25 = getitem_10 + 1
        getitem_10 = add_25 = None
        to_10 = hidden_states_18.to(torch.float32)
        pow_10 = to_10.pow(2)
        to_10 = None
        variance_7 = pow_10.mean(-1, keepdim=True)
        pow_10 = None
        add_26 = variance_7 + 1e-06
        variance_7 = None
        rsqrt_7 = torch.rsqrt(add_26)
        add_26 = None
        hidden_states_19 = hidden_states_18 * rsqrt_7
        rsqrt_7 = None
        normed_hidden_states_5 = (
            l_self_modules_block_modules_2_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_19
        )
        l_self_modules_block_modules_2_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_19
        ) = None
        query_states_10 = torch._C._nn.linear(
            normed_hidden_states_5,
            l_self_modules_block_modules_2_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_,
            None,
        )
        normed_hidden_states_5 = l_self_modules_block_modules_2_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_ = (None)
        view_20 = query_states_10.view(1, -1, 6, 64)
        query_states_10 = None
        query_states_11 = view_20.transpose(1, 2)
        view_20 = None
        key_states_10 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_block_modules_2_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_2_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_10 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_block_modules_2_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_2_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_ = (
            None
        )
        view_21 = key_states_10.view(1, -1, 6, 64)
        key_states_10 = None
        key_states_11 = view_21.transpose(1, 2)
        view_21 = None
        view_22 = value_states_10.view(1, -1, 6, 64)
        value_states_10 = None
        value_states_11 = view_22.transpose(1, 2)
        view_22 = None
        transpose_28 = key_states_11.transpose(3, 2)
        scores_10 = torch.matmul(query_states_11, transpose_28)
        query_states_11 = transpose_28 = None
        scores_10 += position_bias_3
        scores_11 = scores_10
        scores_10 = None
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
        attn_weights_11 = None
        transpose_29 = attn_output_20.transpose(1, 2)
        attn_output_20 = None
        attn_output_21 = transpose_29.contiguous()
        transpose_29 = None
        attn_output_22 = attn_output_21.view(1, -1, 384)
        attn_output_21 = None
        attn_output_23 = torch._C._nn.linear(
            attn_output_22,
            l_self_modules_block_modules_2_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_22 = l_self_modules_block_modules_2_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_ = (None)
        dropout_16 = torch.nn.functional.dropout(attn_output_23, 0.1, False, False)
        attn_output_23 = None
        layer_output_2 = hidden_states_18 + dropout_16
        hidden_states_18 = dropout_16 = None
        to_11 = layer_output_2.to(torch.float32)
        pow_11 = to_11.pow(2)
        to_11 = None
        variance_8 = pow_11.mean(-1, keepdim=True)
        pow_11 = None
        add_28 = variance_8 + 1e-06
        variance_8 = None
        rsqrt_8 = torch.rsqrt(add_28)
        add_28 = None
        hidden_states_20 = layer_output_2 * rsqrt_8
        rsqrt_8 = None
        forwarded_states_2 = (
            l_self_modules_block_modules_2_modules_layer_modules_2_modules_layer_norm_parameters_weight_
            * hidden_states_20
        )
        l_self_modules_block_modules_2_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = (
            hidden_states_20
        ) = None
        linear_30 = torch._C._nn.linear(
            forwarded_states_2,
            l_self_modules_block_modules_2_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_0_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_2_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_0_parameters_weight_ = (
            None
        )
        mul_30 = 0.5 * linear_30
        pow_12 = torch.pow(linear_30, 3.0)
        mul_31 = 0.044715 * pow_12
        pow_12 = None
        add_29 = linear_30 + mul_31
        linear_30 = mul_31 = None
        mul_32 = 0.7978845608028654 * add_29
        add_29 = None
        tanh_2 = torch.tanh(mul_32)
        mul_32 = None
        add_30 = 1.0 + tanh_2
        tanh_2 = None
        hidden_gelu_2 = mul_30 * add_30
        mul_30 = add_30 = None
        hidden_linear_2 = torch._C._nn.linear(
            forwarded_states_2,
            l_self_modules_block_modules_2_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_1_parameters_weight_,
            None,
        )
        forwarded_states_2 = l_self_modules_block_modules_2_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_1_parameters_weight_ = (None)
        hidden_states_21 = hidden_gelu_2 * hidden_linear_2
        hidden_gelu_2 = hidden_linear_2 = None
        hidden_states_22 = torch.nn.functional.dropout(
            hidden_states_21, 0.1, False, False
        )
        hidden_states_21 = None
        hidden_states_23 = torch._C._nn.linear(
            hidden_states_22,
            l_self_modules_block_modules_2_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_22 = l_self_modules_block_modules_2_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_18 = torch.nn.functional.dropout(hidden_states_23, 0.1, False, False)
        hidden_states_23 = None
        hidden_states_24 = layer_output_2 + dropout_18
        layer_output_2 = dropout_18 = None
        to_12 = hidden_states_24.to(torch.float32)
        pow_13 = to_12.pow(2)
        to_12 = None
        variance_9 = pow_13.mean(-1, keepdim=True)
        pow_13 = None
        add_32 = variance_9 + 1e-06
        variance_9 = None
        rsqrt_9 = torch.rsqrt(add_32)
        add_32 = None
        hidden_states_25 = hidden_states_24 * rsqrt_9
        rsqrt_9 = None
        normed_hidden_states_6 = (
            l_self_modules_block_modules_3_modules_layer_modules_0_modules_layer_norm_parameters_weight_
            * hidden_states_25
        )
        l_self_modules_block_modules_3_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = (
            hidden_states_25
        ) = None
        query_states_12 = torch._C._nn.linear(
            normed_hidden_states_6,
            l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_q_parameters_weight_ = (
            None
        )
        view_24 = query_states_12.view(1, -1, 6, 64)
        query_states_12 = None
        query_states_13 = view_24.transpose(1, 2)
        view_24 = None
        key_states_12 = torch._C._nn.linear(
            normed_hidden_states_6,
            l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_12 = torch._C._nn.linear(
            normed_hidden_states_6,
            l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_,
            None,
        )
        normed_hidden_states_6 = l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_v_parameters_weight_ = (None)
        view_25 = key_states_12.view(1, -1, 6, 64)
        key_states_12 = None
        key_states_13 = view_25.transpose(1, 2)
        view_25 = None
        view_26 = value_states_12.view(1, -1, 6, 64)
        value_states_12 = None
        value_states_13 = view_26.transpose(1, 2)
        view_26 = None
        transpose_33 = key_states_13.transpose(3, 2)
        scores_12 = torch.matmul(query_states_13, transpose_33)
        query_states_13 = transpose_33 = None
        scores_12 += position_bias_1
        scores_13 = scores_12
        scores_12 = position_bias_1 = None
        float_8 = scores_13.float()
        softmax_6 = torch.nn.functional.softmax(float_8, dim=-1)
        float_8 = None
        attn_weights_12 = softmax_6.type_as(scores_13)
        softmax_6 = scores_13 = None
        attn_weights_13 = torch.nn.functional.dropout(
            attn_weights_12, p=0.1, training=False
        )
        attn_weights_12 = None
        attn_output_24 = torch.matmul(attn_weights_13, value_states_13)
        attn_weights_13 = None
        transpose_34 = attn_output_24.transpose(1, 2)
        attn_output_24 = None
        attn_output_25 = transpose_34.contiguous()
        transpose_34 = None
        attn_output_26 = attn_output_25.view(1, -1, 384)
        attn_output_25 = None
        attn_output_27 = torch._C._nn.linear(
            attn_output_26,
            l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_26 = l_self_modules_block_modules_3_modules_layer_modules_0_modules_self_attention_modules_o_parameters_weight_ = (None)
        dropout_20 = torch.nn.functional.dropout(attn_output_27, 0.1, False, False)
        attn_output_27 = None
        hidden_states_26 = hidden_states_24 + dropout_20
        hidden_states_24 = dropout_20 = None
        getitem_11 = cache_position[-1]
        cache_position = None
        add_34 = getitem_11 + 1
        getitem_11 = add_34 = None
        to_13 = hidden_states_26.to(torch.float32)
        pow_14 = to_13.pow(2)
        to_13 = None
        variance_10 = pow_14.mean(-1, keepdim=True)
        pow_14 = None
        add_35 = variance_10 + 1e-06
        variance_10 = None
        rsqrt_10 = torch.rsqrt(add_35)
        add_35 = None
        hidden_states_27 = hidden_states_26 * rsqrt_10
        rsqrt_10 = None
        normed_hidden_states_7 = (
            l_self_modules_block_modules_3_modules_layer_modules_1_modules_layer_norm_parameters_weight_
            * hidden_states_27
        )
        l_self_modules_block_modules_3_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = (
            hidden_states_27
        ) = None
        query_states_14 = torch._C._nn.linear(
            normed_hidden_states_7,
            l_self_modules_block_modules_3_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_,
            None,
        )
        normed_hidden_states_7 = l_self_modules_block_modules_3_modules_layer_modules_1_modules_enc_dec_attention_modules_q_parameters_weight_ = (None)
        view_28 = query_states_14.view(1, -1, 6, 64)
        query_states_14 = None
        query_states_15 = view_28.transpose(1, 2)
        view_28 = None
        key_states_14 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_block_modules_3_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_3_modules_layer_modules_1_modules_enc_dec_attention_modules_k_parameters_weight_ = (
            None
        )
        value_states_14 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_block_modules_3_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_,
            None,
        )
        l_encoder_hidden_states_ = l_self_modules_block_modules_3_modules_layer_modules_1_modules_enc_dec_attention_modules_v_parameters_weight_ = (None)
        view_29 = key_states_14.view(1, -1, 6, 64)
        key_states_14 = None
        key_states_15 = view_29.transpose(1, 2)
        view_29 = None
        view_30 = value_states_14.view(1, -1, 6, 64)
        value_states_14 = None
        value_states_15 = view_30.transpose(1, 2)
        view_30 = None
        transpose_38 = key_states_15.transpose(3, 2)
        scores_14 = torch.matmul(query_states_15, transpose_38)
        query_states_15 = transpose_38 = None
        scores_14 += position_bias_3
        scores_15 = scores_14
        scores_14 = position_bias_3 = None
        float_9 = scores_15.float()
        softmax_7 = torch.nn.functional.softmax(float_9, dim=-1)
        float_9 = None
        attn_weights_14 = softmax_7.type_as(scores_15)
        softmax_7 = scores_15 = None
        attn_weights_15 = torch.nn.functional.dropout(
            attn_weights_14, p=0.1, training=False
        )
        attn_weights_14 = None
        attn_output_28 = torch.matmul(attn_weights_15, value_states_15)
        attn_weights_15 = None
        transpose_39 = attn_output_28.transpose(1, 2)
        attn_output_28 = None
        attn_output_29 = transpose_39.contiguous()
        transpose_39 = None
        attn_output_30 = attn_output_29.view(1, -1, 384)
        attn_output_29 = None
        attn_output_31 = torch._C._nn.linear(
            attn_output_30,
            l_self_modules_block_modules_3_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_,
            None,
        )
        attn_output_30 = l_self_modules_block_modules_3_modules_layer_modules_1_modules_enc_dec_attention_modules_o_parameters_weight_ = (None)
        dropout_22 = torch.nn.functional.dropout(attn_output_31, 0.1, False, False)
        attn_output_31 = None
        layer_output_3 = hidden_states_26 + dropout_22
        hidden_states_26 = dropout_22 = None
        to_14 = layer_output_3.to(torch.float32)
        pow_15 = to_14.pow(2)
        to_14 = None
        variance_11 = pow_15.mean(-1, keepdim=True)
        pow_15 = None
        add_37 = variance_11 + 1e-06
        variance_11 = None
        rsqrt_11 = torch.rsqrt(add_37)
        add_37 = None
        hidden_states_28 = layer_output_3 * rsqrt_11
        rsqrt_11 = None
        forwarded_states_3 = (
            l_self_modules_block_modules_3_modules_layer_modules_2_modules_layer_norm_parameters_weight_
            * hidden_states_28
        )
        l_self_modules_block_modules_3_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = (
            hidden_states_28
        ) = None
        linear_41 = torch._C._nn.linear(
            forwarded_states_3,
            l_self_modules_block_modules_3_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_0_parameters_weight_,
            None,
        )
        l_self_modules_block_modules_3_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_0_parameters_weight_ = (
            None
        )
        mul_41 = 0.5 * linear_41
        pow_16 = torch.pow(linear_41, 3.0)
        mul_42 = 0.044715 * pow_16
        pow_16 = None
        add_38 = linear_41 + mul_42
        linear_41 = mul_42 = None
        mul_43 = 0.7978845608028654 * add_38
        add_38 = None
        tanh_3 = torch.tanh(mul_43)
        mul_43 = None
        add_39 = 1.0 + tanh_3
        tanh_3 = None
        hidden_gelu_3 = mul_41 * add_39
        mul_41 = add_39 = None
        hidden_linear_3 = torch._C._nn.linear(
            forwarded_states_3,
            l_self_modules_block_modules_3_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_1_parameters_weight_,
            None,
        )
        forwarded_states_3 = l_self_modules_block_modules_3_modules_layer_modules_2_modules_dense_relu_dense_modules_wi_1_parameters_weight_ = (None)
        hidden_states_29 = hidden_gelu_3 * hidden_linear_3
        hidden_gelu_3 = hidden_linear_3 = None
        hidden_states_30 = torch.nn.functional.dropout(
            hidden_states_29, 0.1, False, False
        )
        hidden_states_29 = None
        hidden_states_31 = torch._C._nn.linear(
            hidden_states_30,
            l_self_modules_block_modules_3_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_,
            None,
        )
        hidden_states_30 = l_self_modules_block_modules_3_modules_layer_modules_2_modules_dense_relu_dense_modules_wo_parameters_weight_ = (None)
        dropout_24 = torch.nn.functional.dropout(hidden_states_31, 0.1, False, False)
        hidden_states_31 = None
        hidden_states_32 = layer_output_3 + dropout_24
        layer_output_3 = dropout_24 = None
        to_15 = hidden_states_32.to(torch.float32)
        pow_17 = to_15.pow(2)
        to_15 = None
        variance_12 = pow_17.mean(-1, keepdim=True)
        pow_17 = None
        add_41 = variance_12 + 1e-06
        variance_12 = None
        rsqrt_12 = torch.rsqrt(add_41)
        add_41 = None
        hidden_states_33 = hidden_states_32 * rsqrt_12
        hidden_states_32 = rsqrt_12 = None
        hidden_states_34 = (
            l_self_modules_final_layer_norm_parameters_weight_ * hidden_states_33
        )
        l_self_modules_final_layer_norm_parameters_weight_ = hidden_states_33 = None
        hidden_states_35 = torch.nn.functional.dropout(
            hidden_states_34, 0.1, False, False
        )
        hidden_states_34 = None
        return (
            value_states_1,
            key_states_1,
            value_states_3,
            key_states_3,
            value_states_5,
            key_states_5,
            value_states_7,
            key_states_7,
            value_states_9,
            key_states_9,
            value_states_11,
            key_states_11,
            value_states_13,
            key_states_13,
            value_states_15,
            key_states_15,
            hidden_states_35,
        )
