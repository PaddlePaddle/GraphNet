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
        mask = torch.full(
            (22, 22), -3.4028234663852886e38, device=device(type="cuda", index=0)
        )
        mask_cond = torch.arange(22, device=device(type="cuda", index=0))
        add = mask_cond + 1
        view = add.view(22, 1)
        add = None
        lt = mask_cond < view
        mask_cond = view = None
        masked_fill_ = mask.masked_fill_(lt, 0)
        lt = masked_fill_ = None
        mask_1 = mask.to(torch.float32)
        mask = None
        getitem = mask_1[(None, None, slice(None, None, None), slice(None, None, None))]
        mask_1 = None
        causal_4d_mask = getitem.expand(1, 1, 22, 22)
        getitem = None
        getitem_1 = l_encoder_attention_mask_[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        l_encoder_attention_mask_ = None
        expand_1 = getitem_1.expand(1, 1, 22, 22)
        getitem_1 = None
        expanded_mask = expand_1.to(torch.float32)
        expand_1 = None
        tensor = torch.tensor(1.0, dtype=torch.float32)
        inverted_mask = tensor - expanded_mask
        tensor = expanded_mask = None
        to_2 = inverted_mask.to(torch.bool)
        encoder_attention_mask = inverted_mask.masked_fill(to_2, -3.4028234663852886e38)
        inverted_mask = to_2 = None
        arange_1 = torch.arange(
            0, 22, dtype=torch.int64, device=device(type="cuda", index=0)
        )
        position_ids = arange_1.expand(1, -1)
        arange_1 = None
        add_1 = position_ids + 2
        position_ids = None
        positions = torch.nn.functional.embedding(
            add_1,
            l_self_modules_embed_positions_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        add_1 = l_self_modules_embed_positions_parameters_weight_ = None
        hidden_states = l_inputs_embeds_ + positions
        l_inputs_embeds_ = positions = None
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
        query_states = linear * 0.5
        linear = None
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
        view_1 = key_states.view(1, -1, 4, 4)
        key_states = None
        key_states_1 = view_1.transpose(1, 2)
        view_1 = None
        view_2 = value_states.view(1, -1, 4, 4)
        value_states = None
        value_states_1 = view_2.transpose(1, 2)
        view_2 = None
        view_3 = query_states.view(1, 22, 4, 4)
        query_states = None
        query_states_1 = view_3.transpose(1, 2)
        view_3 = None
        query_states_2 = query_states_1.reshape(4, -1, 4)
        query_states_1 = None
        key_states_2 = key_states_1.reshape(4, -1, 4)
        value_states_2 = value_states_1.reshape(4, -1, 4)
        transpose_3 = key_states_2.transpose(1, 2)
        key_states_2 = None
        attn_weights = torch.bmm(query_states_2, transpose_3)
        query_states_2 = transpose_3 = None
        view_4 = attn_weights.view(1, 4, 22, 22)
        attn_weights = None
        attn_weights_1 = view_4 + causal_4d_mask
        view_4 = None
        attn_weights_2 = attn_weights_1.view(4, 22, 22)
        attn_weights_1 = None
        attn_weights_3 = torch.nn.functional.softmax(attn_weights_2, dim=-1)
        attn_weights_2 = None
        attn_probs = torch.nn.functional.dropout(attn_weights_3, p=0.1, training=False)
        attn_weights_3 = None
        attn_output = torch.bmm(attn_probs, value_states_2)
        attn_probs = value_states_2 = None
        attn_output_1 = attn_output.view(1, 4, 22, 4)
        attn_output = None
        attn_output_2 = attn_output_1.transpose(1, 2)
        attn_output_1 = None
        attn_output_3 = attn_output_2.reshape(1, 22, 16)
        attn_output_2 = None
        attn_output_4 = torch._C._nn.linear(
            attn_output_3,
            l_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_3 = l_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_3 = torch.nn.functional.dropout(
            attn_output_4, p=0.1, training=False
        )
        attn_output_4 = None
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
        query_states_3 = linear_4 * 0.5
        linear_4 = None
        key_states_3 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_3 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_7 = key_states_3.view(1, -1, 4, 4)
        key_states_3 = None
        key_states_4 = view_7.transpose(1, 2)
        view_7 = None
        view_8 = value_states_3.view(1, -1, 4, 4)
        value_states_3 = None
        value_states_4 = view_8.transpose(1, 2)
        view_8 = None
        view_9 = query_states_3.view(1, 22, 4, 4)
        query_states_3 = None
        query_states_4 = view_9.transpose(1, 2)
        view_9 = None
        query_states_5 = query_states_4.reshape(4, -1, 4)
        query_states_4 = None
        key_states_5 = key_states_4.reshape(4, -1, 4)
        value_states_5 = value_states_4.reshape(4, -1, 4)
        transpose_8 = key_states_5.transpose(1, 2)
        key_states_5 = None
        attn_weights_4 = torch.bmm(query_states_5, transpose_8)
        query_states_5 = transpose_8 = None
        view_10 = attn_weights_4.view(1, 4, 22, 22)
        attn_weights_4 = None
        attn_weights_5 = view_10 + encoder_attention_mask
        view_10 = None
        attn_weights_6 = attn_weights_5.view(4, 22, 22)
        attn_weights_5 = None
        attn_weights_7 = torch.nn.functional.softmax(attn_weights_6, dim=-1)
        attn_weights_6 = None
        attn_probs_1 = torch.nn.functional.dropout(
            attn_weights_7, p=0.1, training=False
        )
        attn_weights_7 = None
        attn_output_5 = torch.bmm(attn_probs_1, value_states_5)
        attn_probs_1 = value_states_5 = None
        attn_output_6 = attn_output_5.view(1, 4, 22, 4)
        attn_output_5 = None
        attn_output_7 = attn_output_6.transpose(1, 2)
        attn_output_6 = None
        attn_output_8 = attn_output_7.reshape(1, 22, 16)
        attn_output_7 = None
        attn_output_9 = torch._C._nn.linear(
            attn_output_8,
            l_self_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_8 = l_self_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_6 = torch.nn.functional.dropout(
            attn_output_9, p=0.1, training=False
        )
        attn_output_9 = None
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
        query_states_6 = linear_10 * 0.5
        linear_10 = None
        key_states_6 = torch._C._nn.linear(
            hidden_states_14,
            l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_6 = torch._C._nn.linear(
            hidden_states_14,
            l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_13 = key_states_6.view(1, -1, 4, 4)
        key_states_6 = None
        key_states_7 = view_13.transpose(1, 2)
        view_13 = None
        view_14 = value_states_6.view(1, -1, 4, 4)
        value_states_6 = None
        value_states_7 = view_14.transpose(1, 2)
        view_14 = None
        view_15 = query_states_6.view(1, 22, 4, 4)
        query_states_6 = None
        query_states_7 = view_15.transpose(1, 2)
        view_15 = None
        query_states_8 = query_states_7.reshape(4, -1, 4)
        query_states_7 = None
        key_states_8 = key_states_7.reshape(4, -1, 4)
        value_states_8 = value_states_7.reshape(4, -1, 4)
        transpose_13 = key_states_8.transpose(1, 2)
        key_states_8 = None
        attn_weights_8 = torch.bmm(query_states_8, transpose_13)
        query_states_8 = transpose_13 = None
        view_16 = attn_weights_8.view(1, 4, 22, 22)
        attn_weights_8 = None
        attn_weights_9 = view_16 + causal_4d_mask
        view_16 = causal_4d_mask = None
        attn_weights_10 = attn_weights_9.view(4, 22, 22)
        attn_weights_9 = None
        attn_weights_11 = torch.nn.functional.softmax(attn_weights_10, dim=-1)
        attn_weights_10 = None
        attn_probs_2 = torch.nn.functional.dropout(
            attn_weights_11, p=0.1, training=False
        )
        attn_weights_11 = None
        attn_output_10 = torch.bmm(attn_probs_2, value_states_8)
        attn_probs_2 = value_states_8 = None
        attn_output_11 = attn_output_10.view(1, 4, 22, 4)
        attn_output_10 = None
        attn_output_12 = attn_output_11.transpose(1, 2)
        attn_output_11 = None
        attn_output_13 = attn_output_12.reshape(1, 22, 16)
        attn_output_12 = None
        attn_output_14 = torch._C._nn.linear(
            attn_output_13,
            l_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_13 = l_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_15 = torch.nn.functional.dropout(
            attn_output_14, p=0.1, training=False
        )
        attn_output_14 = None
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
        query_states_9 = linear_14 * 0.5
        linear_14 = None
        key_states_9 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_9 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_encoder_hidden_states_ = l_self_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_19 = key_states_9.view(1, -1, 4, 4)
        key_states_9 = None
        key_states_10 = view_19.transpose(1, 2)
        view_19 = None
        view_20 = value_states_9.view(1, -1, 4, 4)
        value_states_9 = None
        value_states_10 = view_20.transpose(1, 2)
        view_20 = None
        view_21 = query_states_9.view(1, 22, 4, 4)
        query_states_9 = None
        query_states_10 = view_21.transpose(1, 2)
        view_21 = None
        query_states_11 = query_states_10.reshape(4, -1, 4)
        query_states_10 = None
        key_states_11 = key_states_10.reshape(4, -1, 4)
        value_states_11 = value_states_10.reshape(4, -1, 4)
        transpose_18 = key_states_11.transpose(1, 2)
        key_states_11 = None
        attn_weights_12 = torch.bmm(query_states_11, transpose_18)
        query_states_11 = transpose_18 = None
        view_22 = attn_weights_12.view(1, 4, 22, 22)
        attn_weights_12 = None
        attn_weights_13 = view_22 + encoder_attention_mask
        view_22 = encoder_attention_mask = None
        attn_weights_14 = attn_weights_13.view(4, 22, 22)
        attn_weights_13 = None
        attn_weights_15 = torch.nn.functional.softmax(attn_weights_14, dim=-1)
        attn_weights_14 = None
        attn_probs_3 = torch.nn.functional.dropout(
            attn_weights_15, p=0.1, training=False
        )
        attn_weights_15 = None
        attn_output_15 = torch.bmm(attn_probs_3, value_states_11)
        attn_probs_3 = value_states_11 = None
        attn_output_16 = attn_output_15.view(1, 4, 22, 4)
        attn_output_15 = None
        attn_output_17 = attn_output_16.transpose(1, 2)
        attn_output_16 = None
        attn_output_18 = attn_output_17.reshape(1, 22, 16)
        attn_output_17 = None
        attn_output_19 = torch._C._nn.linear(
            attn_output_18,
            l_self_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_18 = l_self_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_18 = torch.nn.functional.dropout(
            attn_output_19, p=0.1, training=False
        )
        attn_output_19 = None
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
            value_states_4,
            key_states_4,
            value_states_7,
            key_states_7,
            value_states_10,
            key_states_10,
            hidden_states_26,
        )
