import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_input_ids_: torch.Tensor,
        L_self_buffers_position_ids_: torch.Tensor,
        L_attention_mask_: torch.Tensor,
        L_self_modules_tokens_embed_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_positions_embed_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_attn_buffers_bias_: torch.Tensor,
        L_self_modules_h_modules_0_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_attn_buffers_bias_: torch.Tensor,
        L_self_modules_h_modules_1_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_attn_buffers_bias_: torch.Tensor,
        L_self_modules_h_modules_2_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_attn_buffers_bias_: torch.Tensor,
        L_self_modules_h_modules_3_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_attn_buffers_bias_: torch.Tensor,
        L_self_modules_h_modules_4_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_input_ids_ = L_input_ids_
        l_self_buffers_position_ids_ = L_self_buffers_position_ids_
        l_attention_mask_ = L_attention_mask_
        l_self_modules_tokens_embed_parameters_weight_ = (
            L_self_modules_tokens_embed_parameters_weight_
        )
        l_self_modules_positions_embed_parameters_weight_ = (
            L_self_modules_positions_embed_parameters_weight_
        )
        l_self_modules_h_modules_0_modules_attn_modules_c_attn_parameters_bias_ = (
            L_self_modules_h_modules_0_modules_attn_modules_c_attn_parameters_bias_
        )
        l_self_modules_h_modules_0_modules_attn_modules_c_attn_parameters_weight_ = (
            L_self_modules_h_modules_0_modules_attn_modules_c_attn_parameters_weight_
        )
        l_self_modules_h_modules_0_modules_attn_buffers_bias_ = (
            L_self_modules_h_modules_0_modules_attn_buffers_bias_
        )
        l_self_modules_h_modules_0_modules_attn_modules_c_proj_parameters_bias_ = (
            L_self_modules_h_modules_0_modules_attn_modules_c_proj_parameters_bias_
        )
        l_self_modules_h_modules_0_modules_attn_modules_c_proj_parameters_weight_ = (
            L_self_modules_h_modules_0_modules_attn_modules_c_proj_parameters_weight_
        )
        l_self_modules_h_modules_0_modules_ln_1_parameters_weight_ = (
            L_self_modules_h_modules_0_modules_ln_1_parameters_weight_
        )
        l_self_modules_h_modules_0_modules_ln_1_parameters_bias_ = (
            L_self_modules_h_modules_0_modules_ln_1_parameters_bias_
        )
        l_self_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_bias_ = (
            L_self_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_bias_
        )
        l_self_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_weight_ = (
            L_self_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_weight_
        )
        l_self_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_bias_ = (
            L_self_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_bias_
        )
        l_self_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_weight_ = (
            L_self_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_weight_
        )
        l_self_modules_h_modules_0_modules_ln_2_parameters_weight_ = (
            L_self_modules_h_modules_0_modules_ln_2_parameters_weight_
        )
        l_self_modules_h_modules_0_modules_ln_2_parameters_bias_ = (
            L_self_modules_h_modules_0_modules_ln_2_parameters_bias_
        )
        l_self_modules_h_modules_1_modules_attn_modules_c_attn_parameters_bias_ = (
            L_self_modules_h_modules_1_modules_attn_modules_c_attn_parameters_bias_
        )
        l_self_modules_h_modules_1_modules_attn_modules_c_attn_parameters_weight_ = (
            L_self_modules_h_modules_1_modules_attn_modules_c_attn_parameters_weight_
        )
        l_self_modules_h_modules_1_modules_attn_buffers_bias_ = (
            L_self_modules_h_modules_1_modules_attn_buffers_bias_
        )
        l_self_modules_h_modules_1_modules_attn_modules_c_proj_parameters_bias_ = (
            L_self_modules_h_modules_1_modules_attn_modules_c_proj_parameters_bias_
        )
        l_self_modules_h_modules_1_modules_attn_modules_c_proj_parameters_weight_ = (
            L_self_modules_h_modules_1_modules_attn_modules_c_proj_parameters_weight_
        )
        l_self_modules_h_modules_1_modules_ln_1_parameters_weight_ = (
            L_self_modules_h_modules_1_modules_ln_1_parameters_weight_
        )
        l_self_modules_h_modules_1_modules_ln_1_parameters_bias_ = (
            L_self_modules_h_modules_1_modules_ln_1_parameters_bias_
        )
        l_self_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_bias_ = (
            L_self_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_bias_
        )
        l_self_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_weight_ = (
            L_self_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_weight_
        )
        l_self_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_bias_ = (
            L_self_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_bias_
        )
        l_self_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_weight_ = (
            L_self_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_weight_
        )
        l_self_modules_h_modules_1_modules_ln_2_parameters_weight_ = (
            L_self_modules_h_modules_1_modules_ln_2_parameters_weight_
        )
        l_self_modules_h_modules_1_modules_ln_2_parameters_bias_ = (
            L_self_modules_h_modules_1_modules_ln_2_parameters_bias_
        )
        l_self_modules_h_modules_2_modules_attn_modules_c_attn_parameters_bias_ = (
            L_self_modules_h_modules_2_modules_attn_modules_c_attn_parameters_bias_
        )
        l_self_modules_h_modules_2_modules_attn_modules_c_attn_parameters_weight_ = (
            L_self_modules_h_modules_2_modules_attn_modules_c_attn_parameters_weight_
        )
        l_self_modules_h_modules_2_modules_attn_buffers_bias_ = (
            L_self_modules_h_modules_2_modules_attn_buffers_bias_
        )
        l_self_modules_h_modules_2_modules_attn_modules_c_proj_parameters_bias_ = (
            L_self_modules_h_modules_2_modules_attn_modules_c_proj_parameters_bias_
        )
        l_self_modules_h_modules_2_modules_attn_modules_c_proj_parameters_weight_ = (
            L_self_modules_h_modules_2_modules_attn_modules_c_proj_parameters_weight_
        )
        l_self_modules_h_modules_2_modules_ln_1_parameters_weight_ = (
            L_self_modules_h_modules_2_modules_ln_1_parameters_weight_
        )
        l_self_modules_h_modules_2_modules_ln_1_parameters_bias_ = (
            L_self_modules_h_modules_2_modules_ln_1_parameters_bias_
        )
        l_self_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_bias_ = (
            L_self_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_bias_
        )
        l_self_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_weight_ = (
            L_self_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_weight_
        )
        l_self_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_bias_ = (
            L_self_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_bias_
        )
        l_self_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_weight_ = (
            L_self_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_weight_
        )
        l_self_modules_h_modules_2_modules_ln_2_parameters_weight_ = (
            L_self_modules_h_modules_2_modules_ln_2_parameters_weight_
        )
        l_self_modules_h_modules_2_modules_ln_2_parameters_bias_ = (
            L_self_modules_h_modules_2_modules_ln_2_parameters_bias_
        )
        l_self_modules_h_modules_3_modules_attn_modules_c_attn_parameters_bias_ = (
            L_self_modules_h_modules_3_modules_attn_modules_c_attn_parameters_bias_
        )
        l_self_modules_h_modules_3_modules_attn_modules_c_attn_parameters_weight_ = (
            L_self_modules_h_modules_3_modules_attn_modules_c_attn_parameters_weight_
        )
        l_self_modules_h_modules_3_modules_attn_buffers_bias_ = (
            L_self_modules_h_modules_3_modules_attn_buffers_bias_
        )
        l_self_modules_h_modules_3_modules_attn_modules_c_proj_parameters_bias_ = (
            L_self_modules_h_modules_3_modules_attn_modules_c_proj_parameters_bias_
        )
        l_self_modules_h_modules_3_modules_attn_modules_c_proj_parameters_weight_ = (
            L_self_modules_h_modules_3_modules_attn_modules_c_proj_parameters_weight_
        )
        l_self_modules_h_modules_3_modules_ln_1_parameters_weight_ = (
            L_self_modules_h_modules_3_modules_ln_1_parameters_weight_
        )
        l_self_modules_h_modules_3_modules_ln_1_parameters_bias_ = (
            L_self_modules_h_modules_3_modules_ln_1_parameters_bias_
        )
        l_self_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_bias_ = (
            L_self_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_bias_
        )
        l_self_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_weight_ = (
            L_self_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_weight_
        )
        l_self_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_bias_ = (
            L_self_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_bias_
        )
        l_self_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_weight_ = (
            L_self_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_weight_
        )
        l_self_modules_h_modules_3_modules_ln_2_parameters_weight_ = (
            L_self_modules_h_modules_3_modules_ln_2_parameters_weight_
        )
        l_self_modules_h_modules_3_modules_ln_2_parameters_bias_ = (
            L_self_modules_h_modules_3_modules_ln_2_parameters_bias_
        )
        l_self_modules_h_modules_4_modules_attn_modules_c_attn_parameters_bias_ = (
            L_self_modules_h_modules_4_modules_attn_modules_c_attn_parameters_bias_
        )
        l_self_modules_h_modules_4_modules_attn_modules_c_attn_parameters_weight_ = (
            L_self_modules_h_modules_4_modules_attn_modules_c_attn_parameters_weight_
        )
        l_self_modules_h_modules_4_modules_attn_buffers_bias_ = (
            L_self_modules_h_modules_4_modules_attn_buffers_bias_
        )
        l_self_modules_h_modules_4_modules_attn_modules_c_proj_parameters_bias_ = (
            L_self_modules_h_modules_4_modules_attn_modules_c_proj_parameters_bias_
        )
        l_self_modules_h_modules_4_modules_attn_modules_c_proj_parameters_weight_ = (
            L_self_modules_h_modules_4_modules_attn_modules_c_proj_parameters_weight_
        )
        l_self_modules_h_modules_4_modules_ln_1_parameters_weight_ = (
            L_self_modules_h_modules_4_modules_ln_1_parameters_weight_
        )
        l_self_modules_h_modules_4_modules_ln_1_parameters_bias_ = (
            L_self_modules_h_modules_4_modules_ln_1_parameters_bias_
        )
        l_self_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_bias_ = (
            L_self_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_bias_
        )
        l_self_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_weight_ = (
            L_self_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_weight_
        )
        l_self_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_bias_ = (
            L_self_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_bias_
        )
        l_self_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_weight_ = (
            L_self_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_weight_
        )
        l_self_modules_h_modules_4_modules_ln_2_parameters_weight_ = (
            L_self_modules_h_modules_4_modules_ln_2_parameters_weight_
        )
        l_self_modules_h_modules_4_modules_ln_2_parameters_bias_ = (
            L_self_modules_h_modules_4_modules_ln_2_parameters_bias_
        )
        input_ids = l_input_ids_.view(-1, 43)
        l_input_ids_ = None
        position_ids = l_self_buffers_position_ids_[(None, slice(None, 43, None))]
        l_self_buffers_position_ids_ = None
        unsqueeze = l_attention_mask_.unsqueeze(1)
        l_attention_mask_ = None
        attention_mask = unsqueeze.unsqueeze(2)
        unsqueeze = None
        attention_mask_1 = attention_mask.to(dtype=torch.float32)
        attention_mask = None
        sub = 1.0 - attention_mask_1
        attention_mask_1 = None
        attention_mask_2 = sub * -3.4028234663852886e38
        sub = None
        inputs_embeds = torch.nn.functional.embedding(
            input_ids,
            l_self_modules_tokens_embed_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        input_ids = l_self_modules_tokens_embed_parameters_weight_ = None
        position_embeds = torch.nn.functional.embedding(
            position_ids,
            l_self_modules_positions_embed_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        position_ids = l_self_modules_positions_embed_parameters_weight_ = None
        add = inputs_embeds + position_embeds
        inputs_embeds = position_embeds = None
        hidden_states = add + 0
        add = None
        hidden_states_1 = torch.nn.functional.dropout(hidden_states, 0.1, False, False)
        hidden_states = None
        view_1 = hidden_states_1.view(-1, 32)
        x = torch.addmm(
            l_self_modules_h_modules_0_modules_attn_modules_c_attn_parameters_bias_,
            view_1,
            l_self_modules_h_modules_0_modules_attn_modules_c_attn_parameters_weight_,
        )
        l_self_modules_h_modules_0_modules_attn_modules_c_attn_parameters_bias_ = (
            view_1
        ) = (
            l_self_modules_h_modules_0_modules_attn_modules_c_attn_parameters_weight_
        ) = None
        x_1 = x.view((1, 43, 96))
        x = None
        split = x_1.split(32, dim=2)
        x_1 = None
        query = split[0]
        key = split[1]
        value = split[2]
        split = None
        x_2 = query.view(1, 43, 4, 8)
        query = None
        query_1 = x_2.permute(0, 2, 1, 3)
        x_2 = None
        x_3 = key.view(1, 43, 4, 8)
        key = None
        key_1 = x_3.permute(0, 2, 3, 1)
        x_3 = None
        x_4 = value.view(1, 43, 4, 8)
        value = None
        value_1 = x_4.permute(0, 2, 1, 3)
        x_4 = None
        w = torch.matmul(query_1, key_1)
        query_1 = key_1 = None
        w_1 = w / 2.8284271247461903
        w = None
        b = l_self_modules_h_modules_0_modules_attn_buffers_bias_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 43, None),
                slice(None, 43, None),
            )
        ]
        l_self_modules_h_modules_0_modules_attn_buffers_bias_ = None
        mul_1 = w_1 * b
        w_1 = None
        sub_1 = 1 - b
        b = None
        mul_2 = -10000.0 * sub_1
        sub_1 = None
        w_2 = mul_1 + mul_2
        mul_1 = mul_2 = None
        w_3 = w_2 + attention_mask_2
        w_2 = None
        w_4 = torch.nn.functional.softmax(w_3, dim=-1)
        w_3 = None
        w_5 = torch.nn.functional.dropout(w_4, 0.1, False, False)
        w_4 = None
        a = torch.matmul(w_5, value_1)
        w_5 = value_1 = None
        permute_3 = a.permute(0, 2, 1, 3)
        a = None
        x_5 = permute_3.contiguous()
        permute_3 = None
        a_1 = x_5.view(1, 43, 32)
        x_5 = None
        view_7 = a_1.view(-1, 32)
        a_1 = None
        x_6 = torch.addmm(
            l_self_modules_h_modules_0_modules_attn_modules_c_proj_parameters_bias_,
            view_7,
            l_self_modules_h_modules_0_modules_attn_modules_c_proj_parameters_weight_,
        )
        l_self_modules_h_modules_0_modules_attn_modules_c_proj_parameters_bias_ = (
            view_7
        ) = (
            l_self_modules_h_modules_0_modules_attn_modules_c_proj_parameters_weight_
        ) = None
        x_7 = x_6.view((1, 43, 32))
        x_6 = None
        a_2 = torch.nn.functional.dropout(x_7, 0.1, False, False)
        x_7 = None
        add_4 = hidden_states_1 + a_2
        hidden_states_1 = a_2 = None
        n = torch.nn.functional.layer_norm(
            add_4,
            (32,),
            l_self_modules_h_modules_0_modules_ln_1_parameters_weight_,
            l_self_modules_h_modules_0_modules_ln_1_parameters_bias_,
            1e-05,
        )
        add_4 = (
            l_self_modules_h_modules_0_modules_ln_1_parameters_weight_
        ) = l_self_modules_h_modules_0_modules_ln_1_parameters_bias_ = None
        view_9 = n.view(-1, 32)
        x_8 = torch.addmm(
            l_self_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_bias_,
            view_9,
            l_self_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_weight_,
        )
        l_self_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_bias_ = (
            view_9
        ) = (
            l_self_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_weight_
        ) = None
        x_9 = x_8.view((1, 43, 128))
        x_8 = None
        mul_3 = 0.5 * x_9
        pow_1 = torch.pow(x_9, 3.0)
        mul_4 = 0.044715 * pow_1
        pow_1 = None
        add_5 = x_9 + mul_4
        x_9 = mul_4 = None
        mul_5 = 0.7978845608028654 * add_5
        add_5 = None
        tanh = torch.tanh(mul_5)
        mul_5 = None
        add_6 = 1.0 + tanh
        tanh = None
        h = mul_3 * add_6
        mul_3 = add_6 = None
        view_11 = h.view(-1, 128)
        h = None
        x_10 = torch.addmm(
            l_self_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_bias_,
            view_11,
            l_self_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_weight_,
        )
        l_self_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_bias_ = (
            view_11
        ) = (
            l_self_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_weight_
        ) = None
        x_11 = x_10.view((1, 43, 32))
        x_10 = None
        m = torch.nn.functional.dropout(x_11, 0.1, False, False)
        x_11 = None
        add_7 = n + m
        n = m = None
        h_1 = torch.nn.functional.layer_norm(
            add_7,
            (32,),
            l_self_modules_h_modules_0_modules_ln_2_parameters_weight_,
            l_self_modules_h_modules_0_modules_ln_2_parameters_bias_,
            1e-05,
        )
        add_7 = (
            l_self_modules_h_modules_0_modules_ln_2_parameters_weight_
        ) = l_self_modules_h_modules_0_modules_ln_2_parameters_bias_ = None
        view_13 = h_1.view(-1, 32)
        x_12 = torch.addmm(
            l_self_modules_h_modules_1_modules_attn_modules_c_attn_parameters_bias_,
            view_13,
            l_self_modules_h_modules_1_modules_attn_modules_c_attn_parameters_weight_,
        )
        l_self_modules_h_modules_1_modules_attn_modules_c_attn_parameters_bias_ = (
            view_13
        ) = (
            l_self_modules_h_modules_1_modules_attn_modules_c_attn_parameters_weight_
        ) = None
        x_13 = x_12.view((1, 43, 96))
        x_12 = None
        split_1 = x_13.split(32, dim=2)
        x_13 = None
        query_2 = split_1[0]
        key_2 = split_1[1]
        value_2 = split_1[2]
        split_1 = None
        x_14 = query_2.view(1, 43, 4, 8)
        query_2 = None
        query_3 = x_14.permute(0, 2, 1, 3)
        x_14 = None
        x_15 = key_2.view(1, 43, 4, 8)
        key_2 = None
        key_3 = x_15.permute(0, 2, 3, 1)
        x_15 = None
        x_16 = value_2.view(1, 43, 4, 8)
        value_2 = None
        value_3 = x_16.permute(0, 2, 1, 3)
        x_16 = None
        w_6 = torch.matmul(query_3, key_3)
        query_3 = key_3 = None
        w_7 = w_6 / 2.8284271247461903
        w_6 = None
        b_1 = l_self_modules_h_modules_1_modules_attn_buffers_bias_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 43, None),
                slice(None, 43, None),
            )
        ]
        l_self_modules_h_modules_1_modules_attn_buffers_bias_ = None
        mul_7 = w_7 * b_1
        w_7 = None
        sub_2 = 1 - b_1
        b_1 = None
        mul_8 = -10000.0 * sub_2
        sub_2 = None
        w_8 = mul_7 + mul_8
        mul_7 = mul_8 = None
        w_9 = w_8 + attention_mask_2
        w_8 = None
        w_10 = torch.nn.functional.softmax(w_9, dim=-1)
        w_9 = None
        w_11 = torch.nn.functional.dropout(w_10, 0.1, False, False)
        w_10 = None
        a_3 = torch.matmul(w_11, value_3)
        w_11 = value_3 = None
        permute_7 = a_3.permute(0, 2, 1, 3)
        a_3 = None
        x_17 = permute_7.contiguous()
        permute_7 = None
        a_4 = x_17.view(1, 43, 32)
        x_17 = None
        view_19 = a_4.view(-1, 32)
        a_4 = None
        x_18 = torch.addmm(
            l_self_modules_h_modules_1_modules_attn_modules_c_proj_parameters_bias_,
            view_19,
            l_self_modules_h_modules_1_modules_attn_modules_c_proj_parameters_weight_,
        )
        l_self_modules_h_modules_1_modules_attn_modules_c_proj_parameters_bias_ = (
            view_19
        ) = (
            l_self_modules_h_modules_1_modules_attn_modules_c_proj_parameters_weight_
        ) = None
        x_19 = x_18.view((1, 43, 32))
        x_18 = None
        a_5 = torch.nn.functional.dropout(x_19, 0.1, False, False)
        x_19 = None
        add_10 = h_1 + a_5
        h_1 = a_5 = None
        n_1 = torch.nn.functional.layer_norm(
            add_10,
            (32,),
            l_self_modules_h_modules_1_modules_ln_1_parameters_weight_,
            l_self_modules_h_modules_1_modules_ln_1_parameters_bias_,
            1e-05,
        )
        add_10 = (
            l_self_modules_h_modules_1_modules_ln_1_parameters_weight_
        ) = l_self_modules_h_modules_1_modules_ln_1_parameters_bias_ = None
        view_21 = n_1.view(-1, 32)
        x_20 = torch.addmm(
            l_self_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_bias_,
            view_21,
            l_self_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_weight_,
        )
        l_self_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_bias_ = (
            view_21
        ) = (
            l_self_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_weight_
        ) = None
        x_21 = x_20.view((1, 43, 128))
        x_20 = None
        mul_9 = 0.5 * x_21
        pow_2 = torch.pow(x_21, 3.0)
        mul_10 = 0.044715 * pow_2
        pow_2 = None
        add_11 = x_21 + mul_10
        x_21 = mul_10 = None
        mul_11 = 0.7978845608028654 * add_11
        add_11 = None
        tanh_1 = torch.tanh(mul_11)
        mul_11 = None
        add_12 = 1.0 + tanh_1
        tanh_1 = None
        h_2 = mul_9 * add_12
        mul_9 = add_12 = None
        view_23 = h_2.view(-1, 128)
        h_2 = None
        x_22 = torch.addmm(
            l_self_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_bias_,
            view_23,
            l_self_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_weight_,
        )
        l_self_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_bias_ = (
            view_23
        ) = (
            l_self_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_weight_
        ) = None
        x_23 = x_22.view((1, 43, 32))
        x_22 = None
        m_1 = torch.nn.functional.dropout(x_23, 0.1, False, False)
        x_23 = None
        add_13 = n_1 + m_1
        n_1 = m_1 = None
        h_3 = torch.nn.functional.layer_norm(
            add_13,
            (32,),
            l_self_modules_h_modules_1_modules_ln_2_parameters_weight_,
            l_self_modules_h_modules_1_modules_ln_2_parameters_bias_,
            1e-05,
        )
        add_13 = (
            l_self_modules_h_modules_1_modules_ln_2_parameters_weight_
        ) = l_self_modules_h_modules_1_modules_ln_2_parameters_bias_ = None
        view_25 = h_3.view(-1, 32)
        x_24 = torch.addmm(
            l_self_modules_h_modules_2_modules_attn_modules_c_attn_parameters_bias_,
            view_25,
            l_self_modules_h_modules_2_modules_attn_modules_c_attn_parameters_weight_,
        )
        l_self_modules_h_modules_2_modules_attn_modules_c_attn_parameters_bias_ = (
            view_25
        ) = (
            l_self_modules_h_modules_2_modules_attn_modules_c_attn_parameters_weight_
        ) = None
        x_25 = x_24.view((1, 43, 96))
        x_24 = None
        split_2 = x_25.split(32, dim=2)
        x_25 = None
        query_4 = split_2[0]
        key_4 = split_2[1]
        value_4 = split_2[2]
        split_2 = None
        x_26 = query_4.view(1, 43, 4, 8)
        query_4 = None
        query_5 = x_26.permute(0, 2, 1, 3)
        x_26 = None
        x_27 = key_4.view(1, 43, 4, 8)
        key_4 = None
        key_5 = x_27.permute(0, 2, 3, 1)
        x_27 = None
        x_28 = value_4.view(1, 43, 4, 8)
        value_4 = None
        value_5 = x_28.permute(0, 2, 1, 3)
        x_28 = None
        w_12 = torch.matmul(query_5, key_5)
        query_5 = key_5 = None
        w_13 = w_12 / 2.8284271247461903
        w_12 = None
        b_2 = l_self_modules_h_modules_2_modules_attn_buffers_bias_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 43, None),
                slice(None, 43, None),
            )
        ]
        l_self_modules_h_modules_2_modules_attn_buffers_bias_ = None
        mul_13 = w_13 * b_2
        w_13 = None
        sub_3 = 1 - b_2
        b_2 = None
        mul_14 = -10000.0 * sub_3
        sub_3 = None
        w_14 = mul_13 + mul_14
        mul_13 = mul_14 = None
        w_15 = w_14 + attention_mask_2
        w_14 = None
        w_16 = torch.nn.functional.softmax(w_15, dim=-1)
        w_15 = None
        w_17 = torch.nn.functional.dropout(w_16, 0.1, False, False)
        w_16 = None
        a_6 = torch.matmul(w_17, value_5)
        w_17 = value_5 = None
        permute_11 = a_6.permute(0, 2, 1, 3)
        a_6 = None
        x_29 = permute_11.contiguous()
        permute_11 = None
        a_7 = x_29.view(1, 43, 32)
        x_29 = None
        view_31 = a_7.view(-1, 32)
        a_7 = None
        x_30 = torch.addmm(
            l_self_modules_h_modules_2_modules_attn_modules_c_proj_parameters_bias_,
            view_31,
            l_self_modules_h_modules_2_modules_attn_modules_c_proj_parameters_weight_,
        )
        l_self_modules_h_modules_2_modules_attn_modules_c_proj_parameters_bias_ = (
            view_31
        ) = (
            l_self_modules_h_modules_2_modules_attn_modules_c_proj_parameters_weight_
        ) = None
        x_31 = x_30.view((1, 43, 32))
        x_30 = None
        a_8 = torch.nn.functional.dropout(x_31, 0.1, False, False)
        x_31 = None
        add_16 = h_3 + a_8
        h_3 = a_8 = None
        n_2 = torch.nn.functional.layer_norm(
            add_16,
            (32,),
            l_self_modules_h_modules_2_modules_ln_1_parameters_weight_,
            l_self_modules_h_modules_2_modules_ln_1_parameters_bias_,
            1e-05,
        )
        add_16 = (
            l_self_modules_h_modules_2_modules_ln_1_parameters_weight_
        ) = l_self_modules_h_modules_2_modules_ln_1_parameters_bias_ = None
        view_33 = n_2.view(-1, 32)
        x_32 = torch.addmm(
            l_self_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_bias_,
            view_33,
            l_self_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_weight_,
        )
        l_self_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_bias_ = (
            view_33
        ) = (
            l_self_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_weight_
        ) = None
        x_33 = x_32.view((1, 43, 128))
        x_32 = None
        mul_15 = 0.5 * x_33
        pow_3 = torch.pow(x_33, 3.0)
        mul_16 = 0.044715 * pow_3
        pow_3 = None
        add_17 = x_33 + mul_16
        x_33 = mul_16 = None
        mul_17 = 0.7978845608028654 * add_17
        add_17 = None
        tanh_2 = torch.tanh(mul_17)
        mul_17 = None
        add_18 = 1.0 + tanh_2
        tanh_2 = None
        h_4 = mul_15 * add_18
        mul_15 = add_18 = None
        view_35 = h_4.view(-1, 128)
        h_4 = None
        x_34 = torch.addmm(
            l_self_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_bias_,
            view_35,
            l_self_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_weight_,
        )
        l_self_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_bias_ = (
            view_35
        ) = (
            l_self_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_weight_
        ) = None
        x_35 = x_34.view((1, 43, 32))
        x_34 = None
        m_2 = torch.nn.functional.dropout(x_35, 0.1, False, False)
        x_35 = None
        add_19 = n_2 + m_2
        n_2 = m_2 = None
        h_5 = torch.nn.functional.layer_norm(
            add_19,
            (32,),
            l_self_modules_h_modules_2_modules_ln_2_parameters_weight_,
            l_self_modules_h_modules_2_modules_ln_2_parameters_bias_,
            1e-05,
        )
        add_19 = (
            l_self_modules_h_modules_2_modules_ln_2_parameters_weight_
        ) = l_self_modules_h_modules_2_modules_ln_2_parameters_bias_ = None
        view_37 = h_5.view(-1, 32)
        x_36 = torch.addmm(
            l_self_modules_h_modules_3_modules_attn_modules_c_attn_parameters_bias_,
            view_37,
            l_self_modules_h_modules_3_modules_attn_modules_c_attn_parameters_weight_,
        )
        l_self_modules_h_modules_3_modules_attn_modules_c_attn_parameters_bias_ = (
            view_37
        ) = (
            l_self_modules_h_modules_3_modules_attn_modules_c_attn_parameters_weight_
        ) = None
        x_37 = x_36.view((1, 43, 96))
        x_36 = None
        split_3 = x_37.split(32, dim=2)
        x_37 = None
        query_6 = split_3[0]
        key_6 = split_3[1]
        value_6 = split_3[2]
        split_3 = None
        x_38 = query_6.view(1, 43, 4, 8)
        query_6 = None
        query_7 = x_38.permute(0, 2, 1, 3)
        x_38 = None
        x_39 = key_6.view(1, 43, 4, 8)
        key_6 = None
        key_7 = x_39.permute(0, 2, 3, 1)
        x_39 = None
        x_40 = value_6.view(1, 43, 4, 8)
        value_6 = None
        value_7 = x_40.permute(0, 2, 1, 3)
        x_40 = None
        w_18 = torch.matmul(query_7, key_7)
        query_7 = key_7 = None
        w_19 = w_18 / 2.8284271247461903
        w_18 = None
        b_3 = l_self_modules_h_modules_3_modules_attn_buffers_bias_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 43, None),
                slice(None, 43, None),
            )
        ]
        l_self_modules_h_modules_3_modules_attn_buffers_bias_ = None
        mul_19 = w_19 * b_3
        w_19 = None
        sub_4 = 1 - b_3
        b_3 = None
        mul_20 = -10000.0 * sub_4
        sub_4 = None
        w_20 = mul_19 + mul_20
        mul_19 = mul_20 = None
        w_21 = w_20 + attention_mask_2
        w_20 = None
        w_22 = torch.nn.functional.softmax(w_21, dim=-1)
        w_21 = None
        w_23 = torch.nn.functional.dropout(w_22, 0.1, False, False)
        w_22 = None
        a_9 = torch.matmul(w_23, value_7)
        w_23 = value_7 = None
        permute_15 = a_9.permute(0, 2, 1, 3)
        a_9 = None
        x_41 = permute_15.contiguous()
        permute_15 = None
        a_10 = x_41.view(1, 43, 32)
        x_41 = None
        view_43 = a_10.view(-1, 32)
        a_10 = None
        x_42 = torch.addmm(
            l_self_modules_h_modules_3_modules_attn_modules_c_proj_parameters_bias_,
            view_43,
            l_self_modules_h_modules_3_modules_attn_modules_c_proj_parameters_weight_,
        )
        l_self_modules_h_modules_3_modules_attn_modules_c_proj_parameters_bias_ = (
            view_43
        ) = (
            l_self_modules_h_modules_3_modules_attn_modules_c_proj_parameters_weight_
        ) = None
        x_43 = x_42.view((1, 43, 32))
        x_42 = None
        a_11 = torch.nn.functional.dropout(x_43, 0.1, False, False)
        x_43 = None
        add_22 = h_5 + a_11
        h_5 = a_11 = None
        n_3 = torch.nn.functional.layer_norm(
            add_22,
            (32,),
            l_self_modules_h_modules_3_modules_ln_1_parameters_weight_,
            l_self_modules_h_modules_3_modules_ln_1_parameters_bias_,
            1e-05,
        )
        add_22 = (
            l_self_modules_h_modules_3_modules_ln_1_parameters_weight_
        ) = l_self_modules_h_modules_3_modules_ln_1_parameters_bias_ = None
        view_45 = n_3.view(-1, 32)
        x_44 = torch.addmm(
            l_self_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_bias_,
            view_45,
            l_self_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_weight_,
        )
        l_self_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_bias_ = (
            view_45
        ) = (
            l_self_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_weight_
        ) = None
        x_45 = x_44.view((1, 43, 128))
        x_44 = None
        mul_21 = 0.5 * x_45
        pow_4 = torch.pow(x_45, 3.0)
        mul_22 = 0.044715 * pow_4
        pow_4 = None
        add_23 = x_45 + mul_22
        x_45 = mul_22 = None
        mul_23 = 0.7978845608028654 * add_23
        add_23 = None
        tanh_3 = torch.tanh(mul_23)
        mul_23 = None
        add_24 = 1.0 + tanh_3
        tanh_3 = None
        h_6 = mul_21 * add_24
        mul_21 = add_24 = None
        view_47 = h_6.view(-1, 128)
        h_6 = None
        x_46 = torch.addmm(
            l_self_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_bias_,
            view_47,
            l_self_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_weight_,
        )
        l_self_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_bias_ = (
            view_47
        ) = (
            l_self_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_weight_
        ) = None
        x_47 = x_46.view((1, 43, 32))
        x_46 = None
        m_3 = torch.nn.functional.dropout(x_47, 0.1, False, False)
        x_47 = None
        add_25 = n_3 + m_3
        n_3 = m_3 = None
        h_7 = torch.nn.functional.layer_norm(
            add_25,
            (32,),
            l_self_modules_h_modules_3_modules_ln_2_parameters_weight_,
            l_self_modules_h_modules_3_modules_ln_2_parameters_bias_,
            1e-05,
        )
        add_25 = (
            l_self_modules_h_modules_3_modules_ln_2_parameters_weight_
        ) = l_self_modules_h_modules_3_modules_ln_2_parameters_bias_ = None
        view_49 = h_7.view(-1, 32)
        x_48 = torch.addmm(
            l_self_modules_h_modules_4_modules_attn_modules_c_attn_parameters_bias_,
            view_49,
            l_self_modules_h_modules_4_modules_attn_modules_c_attn_parameters_weight_,
        )
        l_self_modules_h_modules_4_modules_attn_modules_c_attn_parameters_bias_ = (
            view_49
        ) = (
            l_self_modules_h_modules_4_modules_attn_modules_c_attn_parameters_weight_
        ) = None
        x_49 = x_48.view((1, 43, 96))
        x_48 = None
        split_4 = x_49.split(32, dim=2)
        x_49 = None
        query_8 = split_4[0]
        key_8 = split_4[1]
        value_8 = split_4[2]
        split_4 = None
        x_50 = query_8.view(1, 43, 4, 8)
        query_8 = None
        query_9 = x_50.permute(0, 2, 1, 3)
        x_50 = None
        x_51 = key_8.view(1, 43, 4, 8)
        key_8 = None
        key_9 = x_51.permute(0, 2, 3, 1)
        x_51 = None
        x_52 = value_8.view(1, 43, 4, 8)
        value_8 = None
        value_9 = x_52.permute(0, 2, 1, 3)
        x_52 = None
        w_24 = torch.matmul(query_9, key_9)
        query_9 = key_9 = None
        w_25 = w_24 / 2.8284271247461903
        w_24 = None
        b_4 = l_self_modules_h_modules_4_modules_attn_buffers_bias_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 43, None),
                slice(None, 43, None),
            )
        ]
        l_self_modules_h_modules_4_modules_attn_buffers_bias_ = None
        mul_25 = w_25 * b_4
        w_25 = None
        sub_5 = 1 - b_4
        b_4 = None
        mul_26 = -10000.0 * sub_5
        sub_5 = None
        w_26 = mul_25 + mul_26
        mul_25 = mul_26 = None
        w_27 = w_26 + attention_mask_2
        w_26 = attention_mask_2 = None
        w_28 = torch.nn.functional.softmax(w_27, dim=-1)
        w_27 = None
        w_29 = torch.nn.functional.dropout(w_28, 0.1, False, False)
        w_28 = None
        a_12 = torch.matmul(w_29, value_9)
        w_29 = value_9 = None
        permute_19 = a_12.permute(0, 2, 1, 3)
        a_12 = None
        x_53 = permute_19.contiguous()
        permute_19 = None
        a_13 = x_53.view(1, 43, 32)
        x_53 = None
        view_55 = a_13.view(-1, 32)
        a_13 = None
        x_54 = torch.addmm(
            l_self_modules_h_modules_4_modules_attn_modules_c_proj_parameters_bias_,
            view_55,
            l_self_modules_h_modules_4_modules_attn_modules_c_proj_parameters_weight_,
        )
        l_self_modules_h_modules_4_modules_attn_modules_c_proj_parameters_bias_ = (
            view_55
        ) = (
            l_self_modules_h_modules_4_modules_attn_modules_c_proj_parameters_weight_
        ) = None
        x_55 = x_54.view((1, 43, 32))
        x_54 = None
        a_14 = torch.nn.functional.dropout(x_55, 0.1, False, False)
        x_55 = None
        add_28 = h_7 + a_14
        h_7 = a_14 = None
        n_4 = torch.nn.functional.layer_norm(
            add_28,
            (32,),
            l_self_modules_h_modules_4_modules_ln_1_parameters_weight_,
            l_self_modules_h_modules_4_modules_ln_1_parameters_bias_,
            1e-05,
        )
        add_28 = (
            l_self_modules_h_modules_4_modules_ln_1_parameters_weight_
        ) = l_self_modules_h_modules_4_modules_ln_1_parameters_bias_ = None
        view_57 = n_4.view(-1, 32)
        x_56 = torch.addmm(
            l_self_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_bias_,
            view_57,
            l_self_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_weight_,
        )
        l_self_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_bias_ = (
            view_57
        ) = (
            l_self_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_weight_
        ) = None
        x_57 = x_56.view((1, 43, 128))
        x_56 = None
        mul_27 = 0.5 * x_57
        pow_5 = torch.pow(x_57, 3.0)
        mul_28 = 0.044715 * pow_5
        pow_5 = None
        add_29 = x_57 + mul_28
        x_57 = mul_28 = None
        mul_29 = 0.7978845608028654 * add_29
        add_29 = None
        tanh_4 = torch.tanh(mul_29)
        mul_29 = None
        add_30 = 1.0 + tanh_4
        tanh_4 = None
        h_8 = mul_27 * add_30
        mul_27 = add_30 = None
        view_59 = h_8.view(-1, 128)
        h_8 = None
        x_58 = torch.addmm(
            l_self_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_bias_,
            view_59,
            l_self_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_weight_,
        )
        l_self_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_bias_ = (
            view_59
        ) = (
            l_self_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_weight_
        ) = None
        x_59 = x_58.view((1, 43, 32))
        x_58 = None
        m_4 = torch.nn.functional.dropout(x_59, 0.1, False, False)
        x_59 = None
        add_31 = n_4 + m_4
        n_4 = m_4 = None
        h_9 = torch.nn.functional.layer_norm(
            add_31,
            (32,),
            l_self_modules_h_modules_4_modules_ln_2_parameters_weight_,
            l_self_modules_h_modules_4_modules_ln_2_parameters_bias_,
            1e-05,
        )
        add_31 = (
            l_self_modules_h_modules_4_modules_ln_2_parameters_weight_
        ) = l_self_modules_h_modules_4_modules_ln_2_parameters_bias_ = None
        hidden_states_2 = h_9.view(1, 43, 32)
        h_9 = None
        return (hidden_states_2,)
