import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_inputs_embeds_: torch.Tensor,
        L_attention_mask_: torch.Tensor,
        L_self_modules_h_modules_0_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_attn_embed_positions: torch.Tensor,
        L_self_modules_h_modules_0_modules_attn_scale_attn: torch.Tensor,
        L_self_modules_h_modules_0_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_mlp_modules_fc_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_mlp_modules_fc_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_mlp_modules_fc_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_mlp_modules_fc_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_attn_embed_positions: torch.Tensor,
        L_self_modules_h_modules_1_modules_attn_scale_attn: torch.Tensor,
        L_self_modules_h_modules_1_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_mlp_modules_fc_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_mlp_modules_fc_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_mlp_modules_fc_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_mlp_modules_fc_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_attn_embed_positions: torch.Tensor,
        L_self_modules_h_modules_2_modules_attn_scale_attn: torch.Tensor,
        L_self_modules_h_modules_2_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_mlp_modules_fc_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_mlp_modules_fc_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_mlp_modules_fc_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_mlp_modules_fc_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_attn_embed_positions: torch.Tensor,
        L_self_modules_h_modules_3_modules_attn_scale_attn: torch.Tensor,
        L_self_modules_h_modules_3_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_mlp_modules_fc_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_mlp_modules_fc_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_mlp_modules_fc_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_mlp_modules_fc_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_attn_embed_positions: torch.Tensor,
        L_self_modules_h_modules_4_modules_attn_scale_attn: torch.Tensor,
        L_self_modules_h_modules_4_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_mlp_modules_fc_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_mlp_modules_fc_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_mlp_modules_fc_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_mlp_modules_fc_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_ln_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_ln_f_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_inputs_embeds_ = L_inputs_embeds_
        l_attention_mask_ = L_attention_mask_
        l_self_modules_h_modules_0_modules_ln_1_parameters_weight_ = (
            L_self_modules_h_modules_0_modules_ln_1_parameters_weight_
        )
        l_self_modules_h_modules_0_modules_ln_1_parameters_bias_ = (
            L_self_modules_h_modules_0_modules_ln_1_parameters_bias_
        )
        l_self_modules_h_modules_0_modules_attn_modules_q_proj_parameters_weight_ = (
            L_self_modules_h_modules_0_modules_attn_modules_q_proj_parameters_weight_
        )
        l_self_modules_h_modules_0_modules_attn_modules_k_proj_parameters_weight_ = (
            L_self_modules_h_modules_0_modules_attn_modules_k_proj_parameters_weight_
        )
        l_self_modules_h_modules_0_modules_attn_modules_v_proj_parameters_weight_ = (
            L_self_modules_h_modules_0_modules_attn_modules_v_proj_parameters_weight_
        )
        l_self_modules_h_modules_0_modules_attn_embed_positions = (
            L_self_modules_h_modules_0_modules_attn_embed_positions
        )
        l_self_modules_h_modules_0_modules_attn_scale_attn = (
            L_self_modules_h_modules_0_modules_attn_scale_attn
        )
        l_self_modules_h_modules_0_modules_attn_modules_out_proj_parameters_weight_ = (
            L_self_modules_h_modules_0_modules_attn_modules_out_proj_parameters_weight_
        )
        l_self_modules_h_modules_0_modules_mlp_modules_fc_in_parameters_weight_ = (
            L_self_modules_h_modules_0_modules_mlp_modules_fc_in_parameters_weight_
        )
        l_self_modules_h_modules_0_modules_mlp_modules_fc_in_parameters_bias_ = (
            L_self_modules_h_modules_0_modules_mlp_modules_fc_in_parameters_bias_
        )
        l_self_modules_h_modules_0_modules_mlp_modules_fc_out_parameters_weight_ = (
            L_self_modules_h_modules_0_modules_mlp_modules_fc_out_parameters_weight_
        )
        l_self_modules_h_modules_0_modules_mlp_modules_fc_out_parameters_bias_ = (
            L_self_modules_h_modules_0_modules_mlp_modules_fc_out_parameters_bias_
        )
        l_self_modules_h_modules_1_modules_ln_1_parameters_weight_ = (
            L_self_modules_h_modules_1_modules_ln_1_parameters_weight_
        )
        l_self_modules_h_modules_1_modules_ln_1_parameters_bias_ = (
            L_self_modules_h_modules_1_modules_ln_1_parameters_bias_
        )
        l_self_modules_h_modules_1_modules_attn_modules_q_proj_parameters_weight_ = (
            L_self_modules_h_modules_1_modules_attn_modules_q_proj_parameters_weight_
        )
        l_self_modules_h_modules_1_modules_attn_modules_k_proj_parameters_weight_ = (
            L_self_modules_h_modules_1_modules_attn_modules_k_proj_parameters_weight_
        )
        l_self_modules_h_modules_1_modules_attn_modules_v_proj_parameters_weight_ = (
            L_self_modules_h_modules_1_modules_attn_modules_v_proj_parameters_weight_
        )
        l_self_modules_h_modules_1_modules_attn_embed_positions = (
            L_self_modules_h_modules_1_modules_attn_embed_positions
        )
        l_self_modules_h_modules_1_modules_attn_scale_attn = (
            L_self_modules_h_modules_1_modules_attn_scale_attn
        )
        l_self_modules_h_modules_1_modules_attn_modules_out_proj_parameters_weight_ = (
            L_self_modules_h_modules_1_modules_attn_modules_out_proj_parameters_weight_
        )
        l_self_modules_h_modules_1_modules_mlp_modules_fc_in_parameters_weight_ = (
            L_self_modules_h_modules_1_modules_mlp_modules_fc_in_parameters_weight_
        )
        l_self_modules_h_modules_1_modules_mlp_modules_fc_in_parameters_bias_ = (
            L_self_modules_h_modules_1_modules_mlp_modules_fc_in_parameters_bias_
        )
        l_self_modules_h_modules_1_modules_mlp_modules_fc_out_parameters_weight_ = (
            L_self_modules_h_modules_1_modules_mlp_modules_fc_out_parameters_weight_
        )
        l_self_modules_h_modules_1_modules_mlp_modules_fc_out_parameters_bias_ = (
            L_self_modules_h_modules_1_modules_mlp_modules_fc_out_parameters_bias_
        )
        l_self_modules_h_modules_2_modules_ln_1_parameters_weight_ = (
            L_self_modules_h_modules_2_modules_ln_1_parameters_weight_
        )
        l_self_modules_h_modules_2_modules_ln_1_parameters_bias_ = (
            L_self_modules_h_modules_2_modules_ln_1_parameters_bias_
        )
        l_self_modules_h_modules_2_modules_attn_modules_q_proj_parameters_weight_ = (
            L_self_modules_h_modules_2_modules_attn_modules_q_proj_parameters_weight_
        )
        l_self_modules_h_modules_2_modules_attn_modules_k_proj_parameters_weight_ = (
            L_self_modules_h_modules_2_modules_attn_modules_k_proj_parameters_weight_
        )
        l_self_modules_h_modules_2_modules_attn_modules_v_proj_parameters_weight_ = (
            L_self_modules_h_modules_2_modules_attn_modules_v_proj_parameters_weight_
        )
        l_self_modules_h_modules_2_modules_attn_embed_positions = (
            L_self_modules_h_modules_2_modules_attn_embed_positions
        )
        l_self_modules_h_modules_2_modules_attn_scale_attn = (
            L_self_modules_h_modules_2_modules_attn_scale_attn
        )
        l_self_modules_h_modules_2_modules_attn_modules_out_proj_parameters_weight_ = (
            L_self_modules_h_modules_2_modules_attn_modules_out_proj_parameters_weight_
        )
        l_self_modules_h_modules_2_modules_mlp_modules_fc_in_parameters_weight_ = (
            L_self_modules_h_modules_2_modules_mlp_modules_fc_in_parameters_weight_
        )
        l_self_modules_h_modules_2_modules_mlp_modules_fc_in_parameters_bias_ = (
            L_self_modules_h_modules_2_modules_mlp_modules_fc_in_parameters_bias_
        )
        l_self_modules_h_modules_2_modules_mlp_modules_fc_out_parameters_weight_ = (
            L_self_modules_h_modules_2_modules_mlp_modules_fc_out_parameters_weight_
        )
        l_self_modules_h_modules_2_modules_mlp_modules_fc_out_parameters_bias_ = (
            L_self_modules_h_modules_2_modules_mlp_modules_fc_out_parameters_bias_
        )
        l_self_modules_h_modules_3_modules_ln_1_parameters_weight_ = (
            L_self_modules_h_modules_3_modules_ln_1_parameters_weight_
        )
        l_self_modules_h_modules_3_modules_ln_1_parameters_bias_ = (
            L_self_modules_h_modules_3_modules_ln_1_parameters_bias_
        )
        l_self_modules_h_modules_3_modules_attn_modules_q_proj_parameters_weight_ = (
            L_self_modules_h_modules_3_modules_attn_modules_q_proj_parameters_weight_
        )
        l_self_modules_h_modules_3_modules_attn_modules_k_proj_parameters_weight_ = (
            L_self_modules_h_modules_3_modules_attn_modules_k_proj_parameters_weight_
        )
        l_self_modules_h_modules_3_modules_attn_modules_v_proj_parameters_weight_ = (
            L_self_modules_h_modules_3_modules_attn_modules_v_proj_parameters_weight_
        )
        l_self_modules_h_modules_3_modules_attn_embed_positions = (
            L_self_modules_h_modules_3_modules_attn_embed_positions
        )
        l_self_modules_h_modules_3_modules_attn_scale_attn = (
            L_self_modules_h_modules_3_modules_attn_scale_attn
        )
        l_self_modules_h_modules_3_modules_attn_modules_out_proj_parameters_weight_ = (
            L_self_modules_h_modules_3_modules_attn_modules_out_proj_parameters_weight_
        )
        l_self_modules_h_modules_3_modules_mlp_modules_fc_in_parameters_weight_ = (
            L_self_modules_h_modules_3_modules_mlp_modules_fc_in_parameters_weight_
        )
        l_self_modules_h_modules_3_modules_mlp_modules_fc_in_parameters_bias_ = (
            L_self_modules_h_modules_3_modules_mlp_modules_fc_in_parameters_bias_
        )
        l_self_modules_h_modules_3_modules_mlp_modules_fc_out_parameters_weight_ = (
            L_self_modules_h_modules_3_modules_mlp_modules_fc_out_parameters_weight_
        )
        l_self_modules_h_modules_3_modules_mlp_modules_fc_out_parameters_bias_ = (
            L_self_modules_h_modules_3_modules_mlp_modules_fc_out_parameters_bias_
        )
        l_self_modules_h_modules_4_modules_ln_1_parameters_weight_ = (
            L_self_modules_h_modules_4_modules_ln_1_parameters_weight_
        )
        l_self_modules_h_modules_4_modules_ln_1_parameters_bias_ = (
            L_self_modules_h_modules_4_modules_ln_1_parameters_bias_
        )
        l_self_modules_h_modules_4_modules_attn_modules_q_proj_parameters_weight_ = (
            L_self_modules_h_modules_4_modules_attn_modules_q_proj_parameters_weight_
        )
        l_self_modules_h_modules_4_modules_attn_modules_k_proj_parameters_weight_ = (
            L_self_modules_h_modules_4_modules_attn_modules_k_proj_parameters_weight_
        )
        l_self_modules_h_modules_4_modules_attn_modules_v_proj_parameters_weight_ = (
            L_self_modules_h_modules_4_modules_attn_modules_v_proj_parameters_weight_
        )
        l_self_modules_h_modules_4_modules_attn_embed_positions = (
            L_self_modules_h_modules_4_modules_attn_embed_positions
        )
        l_self_modules_h_modules_4_modules_attn_scale_attn = (
            L_self_modules_h_modules_4_modules_attn_scale_attn
        )
        l_self_modules_h_modules_4_modules_attn_modules_out_proj_parameters_weight_ = (
            L_self_modules_h_modules_4_modules_attn_modules_out_proj_parameters_weight_
        )
        l_self_modules_h_modules_4_modules_mlp_modules_fc_in_parameters_weight_ = (
            L_self_modules_h_modules_4_modules_mlp_modules_fc_in_parameters_weight_
        )
        l_self_modules_h_modules_4_modules_mlp_modules_fc_in_parameters_bias_ = (
            L_self_modules_h_modules_4_modules_mlp_modules_fc_in_parameters_bias_
        )
        l_self_modules_h_modules_4_modules_mlp_modules_fc_out_parameters_weight_ = (
            L_self_modules_h_modules_4_modules_mlp_modules_fc_out_parameters_weight_
        )
        l_self_modules_h_modules_4_modules_mlp_modules_fc_out_parameters_bias_ = (
            L_self_modules_h_modules_4_modules_mlp_modules_fc_out_parameters_bias_
        )
        l_self_modules_ln_f_parameters_weight_ = L_self_modules_ln_f_parameters_weight_
        l_self_modules_ln_f_parameters_bias_ = L_self_modules_ln_f_parameters_bias_
        cache_position = torch.arange(0, 20, device=device(type="cuda", index=0))
        position_ids = cache_position.unsqueeze(0)
        causal_mask = torch.full(
            (20, 20),
            fill_value=-3.4028234663852886e38,
            dtype=torch.float32,
            device=device(type="cuda", index=0),
        )
        causal_mask_1 = torch.triu(causal_mask, diagonal=1)
        causal_mask = None
        arange_1 = torch.arange(20, device=device(type="cuda", index=0))
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
        getitem_2 = l_attention_mask_[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        l_attention_mask_ = None
        to = getitem_2.to(device(type="cuda", index=0))
        getitem_2 = None
        padding_mask = getitem_1 + to
        getitem_1 = to = None
        padding_mask_1 = padding_mask.__eq__(0)
        padding_mask = None
        getitem_3 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
            )
        ]
        masked_fill = getitem_3.masked_fill(padding_mask_1, -3.4028234663852886e38)
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
        hidden_states = torch.nn.functional.dropout(l_inputs_embeds_, 0.0, False, False)
        l_inputs_embeds_ = None
        hidden_states_1 = torch.nn.functional.layer_norm(
            hidden_states,
            (32,),
            l_self_modules_h_modules_0_modules_ln_1_parameters_weight_,
            l_self_modules_h_modules_0_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_0_modules_ln_1_parameters_weight_ = (
            l_self_modules_h_modules_0_modules_ln_1_parameters_bias_
        ) = None
        query = torch._C._nn.linear(
            hidden_states_1,
            l_self_modules_h_modules_0_modules_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_h_modules_0_modules_attn_modules_q_proj_parameters_weight_ = None
        key = torch._C._nn.linear(
            hidden_states_1,
            l_self_modules_h_modules_0_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_h_modules_0_modules_attn_modules_k_proj_parameters_weight_ = None
        value = torch._C._nn.linear(
            hidden_states_1,
            l_self_modules_h_modules_0_modules_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_h_modules_0_modules_attn_modules_v_proj_parameters_weight_ = None
        tensor = query.view((1, 20, 4, 8))
        query = None
        tensor_1 = key.view((1, 20, 4, 8))
        key = None
        tensor_2 = value.view((1, 20, 4, 8))
        value = None
        value_1 = tensor_2.permute(0, 2, 1, 3)
        tensor_2 = None
        embed_positions = l_self_modules_h_modules_0_modules_attn_embed_positions.to(
            device(type="cuda", index=0)
        )
        l_self_modules_h_modules_0_modules_attn_embed_positions = None
        embed_positions_1 = embed_positions.repeat(1, 1, 1)
        unsqueeze_1 = position_ids.unsqueeze(-1)
        repeated_position_ids = unsqueeze_1.repeat(1, 1, 4)
        unsqueeze_1 = None
        sincos = torch.gather(embed_positions_1, 1, repeated_position_ids)
        embed_positions_1 = repeated_position_ids = None
        split = torch.functional.split(sincos, 2, dim=-1)
        sincos = None
        sin = split[0]
        cos = split[1]
        split = None
        k_rot = tensor_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 4, None),
            )
        ]
        k_pass = tensor_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(4, None, None),
            )
        ]
        tensor_1 = None
        q_rot = tensor[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 4, None),
            )
        ]
        q_pass = tensor[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(4, None, None),
            )
        ]
        tensor = None
        getitem_10 = sin[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        sin_1 = torch.repeat_interleave(getitem_10, 2, 3)
        getitem_10 = None
        getitem_11 = cos[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        cos_1 = torch.repeat_interleave(getitem_11, 2, 3)
        getitem_11 = None
        mul = k_rot * cos_1
        cos_1 = None
        x1 = k_rot[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, 2),
            )
        ]
        x2 = k_rot[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, 2),
            )
        ]
        k_rot = None
        neg = -x2
        x2 = None
        x = torch.stack((neg, x1), dim=-1)
        neg = x1 = None
        flatten = x.flatten(-2)
        x = None
        mul_1 = flatten * sin_1
        flatten = sin_1 = None
        k_rot_1 = mul + mul_1
        mul = mul_1 = None
        getitem_14 = sin[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        sin = None
        sin_2 = torch.repeat_interleave(getitem_14, 2, 3)
        getitem_14 = None
        getitem_15 = cos[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        cos = None
        cos_2 = torch.repeat_interleave(getitem_15, 2, 3)
        getitem_15 = None
        mul_2 = q_rot * cos_2
        cos_2 = None
        x1_1 = q_rot[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, 2),
            )
        ]
        x2_1 = q_rot[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, 2),
            )
        ]
        q_rot = None
        neg_1 = -x2_1
        x2_1 = None
        x_1 = torch.stack((neg_1, x1_1), dim=-1)
        neg_1 = x1_1 = None
        flatten_1 = x_1.flatten(-2)
        x_1 = None
        mul_3 = flatten_1 * sin_2
        flatten_1 = sin_2 = None
        q_rot_1 = mul_2 + mul_3
        mul_2 = mul_3 = None
        key_1 = torch.cat([k_rot_1, k_pass], dim=-1)
        k_rot_1 = k_pass = None
        query_1 = torch.cat([q_rot_1, q_pass], dim=-1)
        q_rot_1 = q_pass = None
        key_2 = key_1.permute(0, 2, 1, 3)
        key_1 = None
        query_2 = query_1.permute(0, 2, 1, 3)
        query_1 = None
        query_3 = query_2.to(torch.float32)
        query_2 = None
        key_3 = key_2.to(torch.float32)
        transpose = key_3.transpose(-1, -2)
        key_3 = None
        attn_weights = torch.matmul(query_3, transpose)
        query_3 = transpose = None
        attn_weights_1 = (
            attn_weights / l_self_modules_h_modules_0_modules_attn_scale_attn
        )
        attn_weights = l_self_modules_h_modules_0_modules_attn_scale_attn = None
        causal_mask_5 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
            )
        ]
        attn_weights_2 = attn_weights_1 + causal_mask_5
        attn_weights_1 = causal_mask_5 = None
        attn_weights_3 = torch.nn.functional.softmax(attn_weights_2, dim=-1)
        attn_weights_2 = None
        attn_weights_4 = attn_weights_3.to(torch.float32)
        attn_weights_3 = None
        attn_weights_5 = torch.nn.functional.dropout(attn_weights_4, 0.0, False, False)
        attn_weights_4 = None
        attn_output = torch.matmul(attn_weights_5, value_1)
        attn_weights_5 = None
        permute_3 = attn_output.permute(0, 2, 1, 3)
        attn_output = None
        tensor_3 = permute_3.contiguous()
        permute_3 = None
        attn_output_1 = tensor_3.view((1, 20, 32))
        tensor_3 = None
        attn_output_2 = torch._C._nn.linear(
            attn_output_1,
            l_self_modules_h_modules_0_modules_attn_modules_out_proj_parameters_weight_,
            None,
        )
        attn_output_1 = (
            l_self_modules_h_modules_0_modules_attn_modules_out_proj_parameters_weight_
        ) = None
        attn_output_3 = torch.nn.functional.dropout(attn_output_2, 0.0, False, False)
        attn_output_2 = None
        hidden_states_2 = torch._C._nn.linear(
            hidden_states_1,
            l_self_modules_h_modules_0_modules_mlp_modules_fc_in_parameters_weight_,
            l_self_modules_h_modules_0_modules_mlp_modules_fc_in_parameters_bias_,
        )
        hidden_states_1 = (
            l_self_modules_h_modules_0_modules_mlp_modules_fc_in_parameters_weight_
        ) = l_self_modules_h_modules_0_modules_mlp_modules_fc_in_parameters_bias_ = None
        mul_4 = 0.5 * hidden_states_2
        pow_1 = torch.pow(hidden_states_2, 3.0)
        mul_5 = 0.044715 * pow_1
        pow_1 = None
        add_4 = hidden_states_2 + mul_5
        hidden_states_2 = mul_5 = None
        mul_6 = 0.7978845608028654 * add_4
        add_4 = None
        tanh = torch.tanh(mul_6)
        mul_6 = None
        add_5 = 1.0 + tanh
        tanh = None
        hidden_states_3 = mul_4 * add_5
        mul_4 = add_5 = None
        hidden_states_4 = torch._C._nn.linear(
            hidden_states_3,
            l_self_modules_h_modules_0_modules_mlp_modules_fc_out_parameters_weight_,
            l_self_modules_h_modules_0_modules_mlp_modules_fc_out_parameters_bias_,
        )
        hidden_states_3 = (
            l_self_modules_h_modules_0_modules_mlp_modules_fc_out_parameters_weight_
        ) = (
            l_self_modules_h_modules_0_modules_mlp_modules_fc_out_parameters_bias_
        ) = None
        hidden_states_5 = torch.nn.functional.dropout(
            hidden_states_4, 0.0, False, False
        )
        hidden_states_4 = None
        add_6 = attn_output_3 + hidden_states_5
        attn_output_3 = hidden_states_5 = None
        hidden_states_6 = add_6 + hidden_states
        add_6 = hidden_states = None
        hidden_states_7 = torch.nn.functional.layer_norm(
            hidden_states_6,
            (32,),
            l_self_modules_h_modules_1_modules_ln_1_parameters_weight_,
            l_self_modules_h_modules_1_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_1_modules_ln_1_parameters_weight_ = (
            l_self_modules_h_modules_1_modules_ln_1_parameters_bias_
        ) = None
        query_4 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_h_modules_1_modules_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_h_modules_1_modules_attn_modules_q_proj_parameters_weight_ = None
        key_4 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_h_modules_1_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_h_modules_1_modules_attn_modules_k_proj_parameters_weight_ = None
        value_2 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_h_modules_1_modules_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_h_modules_1_modules_attn_modules_v_proj_parameters_weight_ = None
        tensor_4 = query_4.view((1, 20, 4, 8))
        query_4 = None
        tensor_5 = key_4.view((1, 20, 4, 8))
        key_4 = None
        tensor_6 = value_2.view((1, 20, 4, 8))
        value_2 = None
        value_3 = tensor_6.permute(0, 2, 1, 3)
        tensor_6 = None
        embed_positions_2 = l_self_modules_h_modules_1_modules_attn_embed_positions.to(
            device(type="cuda", index=0)
        )
        l_self_modules_h_modules_1_modules_attn_embed_positions = None
        embed_positions_3 = embed_positions_2.repeat(1, 1, 1)
        unsqueeze_2 = position_ids.unsqueeze(-1)
        repeated_position_ids_1 = unsqueeze_2.repeat(1, 1, 4)
        unsqueeze_2 = None
        sincos_1 = torch.gather(embed_positions_3, 1, repeated_position_ids_1)
        embed_positions_3 = repeated_position_ids_1 = None
        split_1 = torch.functional.split(sincos_1, 2, dim=-1)
        sincos_1 = None
        sin_3 = split_1[0]
        cos_3 = split_1[1]
        split_1 = None
        k_rot_2 = tensor_5[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 4, None),
            )
        ]
        k_pass_1 = tensor_5[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(4, None, None),
            )
        ]
        tensor_5 = None
        q_rot_2 = tensor_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 4, None),
            )
        ]
        q_pass_1 = tensor_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(4, None, None),
            )
        ]
        tensor_4 = None
        getitem_25 = sin_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        sin_4 = torch.repeat_interleave(getitem_25, 2, 3)
        getitem_25 = None
        getitem_26 = cos_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        cos_4 = torch.repeat_interleave(getitem_26, 2, 3)
        getitem_26 = None
        mul_8 = k_rot_2 * cos_4
        cos_4 = None
        x1_2 = k_rot_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, 2),
            )
        ]
        x2_2 = k_rot_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, 2),
            )
        ]
        k_rot_2 = None
        neg_2 = -x2_2
        x2_2 = None
        x_2 = torch.stack((neg_2, x1_2), dim=-1)
        neg_2 = x1_2 = None
        flatten_2 = x_2.flatten(-2)
        x_2 = None
        mul_9 = flatten_2 * sin_4
        flatten_2 = sin_4 = None
        k_rot_3 = mul_8 + mul_9
        mul_8 = mul_9 = None
        getitem_29 = sin_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        sin_3 = None
        sin_5 = torch.repeat_interleave(getitem_29, 2, 3)
        getitem_29 = None
        getitem_30 = cos_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        cos_3 = None
        cos_5 = torch.repeat_interleave(getitem_30, 2, 3)
        getitem_30 = None
        mul_10 = q_rot_2 * cos_5
        cos_5 = None
        x1_3 = q_rot_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, 2),
            )
        ]
        x2_3 = q_rot_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, 2),
            )
        ]
        q_rot_2 = None
        neg_3 = -x2_3
        x2_3 = None
        x_3 = torch.stack((neg_3, x1_3), dim=-1)
        neg_3 = x1_3 = None
        flatten_3 = x_3.flatten(-2)
        x_3 = None
        mul_11 = flatten_3 * sin_5
        flatten_3 = sin_5 = None
        q_rot_3 = mul_10 + mul_11
        mul_10 = mul_11 = None
        key_5 = torch.cat([k_rot_3, k_pass_1], dim=-1)
        k_rot_3 = k_pass_1 = None
        query_5 = torch.cat([q_rot_3, q_pass_1], dim=-1)
        q_rot_3 = q_pass_1 = None
        key_6 = key_5.permute(0, 2, 1, 3)
        key_5 = None
        query_6 = query_5.permute(0, 2, 1, 3)
        query_5 = None
        query_7 = query_6.to(torch.float32)
        query_6 = None
        key_7 = key_6.to(torch.float32)
        transpose_1 = key_7.transpose(-1, -2)
        key_7 = None
        attn_weights_6 = torch.matmul(query_7, transpose_1)
        query_7 = transpose_1 = None
        attn_weights_7 = (
            attn_weights_6 / l_self_modules_h_modules_1_modules_attn_scale_attn
        )
        attn_weights_6 = l_self_modules_h_modules_1_modules_attn_scale_attn = None
        causal_mask_6 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
            )
        ]
        attn_weights_8 = attn_weights_7 + causal_mask_6
        attn_weights_7 = causal_mask_6 = None
        attn_weights_9 = torch.nn.functional.softmax(attn_weights_8, dim=-1)
        attn_weights_8 = None
        attn_weights_10 = attn_weights_9.to(torch.float32)
        attn_weights_9 = None
        attn_weights_11 = torch.nn.functional.dropout(
            attn_weights_10, 0.0, False, False
        )
        attn_weights_10 = None
        attn_output_4 = torch.matmul(attn_weights_11, value_3)
        attn_weights_11 = None
        permute_7 = attn_output_4.permute(0, 2, 1, 3)
        attn_output_4 = None
        tensor_7 = permute_7.contiguous()
        permute_7 = None
        attn_output_5 = tensor_7.view((1, 20, 32))
        tensor_7 = None
        attn_output_6 = torch._C._nn.linear(
            attn_output_5,
            l_self_modules_h_modules_1_modules_attn_modules_out_proj_parameters_weight_,
            None,
        )
        attn_output_5 = (
            l_self_modules_h_modules_1_modules_attn_modules_out_proj_parameters_weight_
        ) = None
        attn_output_7 = torch.nn.functional.dropout(attn_output_6, 0.0, False, False)
        attn_output_6 = None
        hidden_states_8 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_h_modules_1_modules_mlp_modules_fc_in_parameters_weight_,
            l_self_modules_h_modules_1_modules_mlp_modules_fc_in_parameters_bias_,
        )
        hidden_states_7 = (
            l_self_modules_h_modules_1_modules_mlp_modules_fc_in_parameters_weight_
        ) = l_self_modules_h_modules_1_modules_mlp_modules_fc_in_parameters_bias_ = None
        mul_12 = 0.5 * hidden_states_8
        pow_2 = torch.pow(hidden_states_8, 3.0)
        mul_13 = 0.044715 * pow_2
        pow_2 = None
        add_11 = hidden_states_8 + mul_13
        hidden_states_8 = mul_13 = None
        mul_14 = 0.7978845608028654 * add_11
        add_11 = None
        tanh_1 = torch.tanh(mul_14)
        mul_14 = None
        add_12 = 1.0 + tanh_1
        tanh_1 = None
        hidden_states_9 = mul_12 * add_12
        mul_12 = add_12 = None
        hidden_states_10 = torch._C._nn.linear(
            hidden_states_9,
            l_self_modules_h_modules_1_modules_mlp_modules_fc_out_parameters_weight_,
            l_self_modules_h_modules_1_modules_mlp_modules_fc_out_parameters_bias_,
        )
        hidden_states_9 = (
            l_self_modules_h_modules_1_modules_mlp_modules_fc_out_parameters_weight_
        ) = (
            l_self_modules_h_modules_1_modules_mlp_modules_fc_out_parameters_bias_
        ) = None
        hidden_states_11 = torch.nn.functional.dropout(
            hidden_states_10, 0.0, False, False
        )
        hidden_states_10 = None
        add_13 = attn_output_7 + hidden_states_11
        attn_output_7 = hidden_states_11 = None
        hidden_states_12 = add_13 + hidden_states_6
        add_13 = hidden_states_6 = None
        hidden_states_13 = torch.nn.functional.layer_norm(
            hidden_states_12,
            (32,),
            l_self_modules_h_modules_2_modules_ln_1_parameters_weight_,
            l_self_modules_h_modules_2_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_2_modules_ln_1_parameters_weight_ = (
            l_self_modules_h_modules_2_modules_ln_1_parameters_bias_
        ) = None
        query_8 = torch._C._nn.linear(
            hidden_states_13,
            l_self_modules_h_modules_2_modules_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_h_modules_2_modules_attn_modules_q_proj_parameters_weight_ = None
        key_8 = torch._C._nn.linear(
            hidden_states_13,
            l_self_modules_h_modules_2_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_h_modules_2_modules_attn_modules_k_proj_parameters_weight_ = None
        value_4 = torch._C._nn.linear(
            hidden_states_13,
            l_self_modules_h_modules_2_modules_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_h_modules_2_modules_attn_modules_v_proj_parameters_weight_ = None
        tensor_8 = query_8.view((1, 20, 4, 8))
        query_8 = None
        tensor_9 = key_8.view((1, 20, 4, 8))
        key_8 = None
        tensor_10 = value_4.view((1, 20, 4, 8))
        value_4 = None
        value_5 = tensor_10.permute(0, 2, 1, 3)
        tensor_10 = None
        embed_positions_4 = l_self_modules_h_modules_2_modules_attn_embed_positions.to(
            device(type="cuda", index=0)
        )
        l_self_modules_h_modules_2_modules_attn_embed_positions = None
        embed_positions_5 = embed_positions_4.repeat(1, 1, 1)
        unsqueeze_3 = position_ids.unsqueeze(-1)
        repeated_position_ids_2 = unsqueeze_3.repeat(1, 1, 4)
        unsqueeze_3 = None
        sincos_2 = torch.gather(embed_positions_5, 1, repeated_position_ids_2)
        embed_positions_5 = repeated_position_ids_2 = None
        split_2 = torch.functional.split(sincos_2, 2, dim=-1)
        sincos_2 = None
        sin_6 = split_2[0]
        cos_6 = split_2[1]
        split_2 = None
        k_rot_4 = tensor_9[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 4, None),
            )
        ]
        k_pass_2 = tensor_9[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(4, None, None),
            )
        ]
        tensor_9 = None
        q_rot_4 = tensor_8[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 4, None),
            )
        ]
        q_pass_2 = tensor_8[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(4, None, None),
            )
        ]
        tensor_8 = None
        getitem_40 = sin_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        sin_7 = torch.repeat_interleave(getitem_40, 2, 3)
        getitem_40 = None
        getitem_41 = cos_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        cos_7 = torch.repeat_interleave(getitem_41, 2, 3)
        getitem_41 = None
        mul_16 = k_rot_4 * cos_7
        cos_7 = None
        x1_4 = k_rot_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, 2),
            )
        ]
        x2_4 = k_rot_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, 2),
            )
        ]
        k_rot_4 = None
        neg_4 = -x2_4
        x2_4 = None
        x_4 = torch.stack((neg_4, x1_4), dim=-1)
        neg_4 = x1_4 = None
        flatten_4 = x_4.flatten(-2)
        x_4 = None
        mul_17 = flatten_4 * sin_7
        flatten_4 = sin_7 = None
        k_rot_5 = mul_16 + mul_17
        mul_16 = mul_17 = None
        getitem_44 = sin_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        sin_6 = None
        sin_8 = torch.repeat_interleave(getitem_44, 2, 3)
        getitem_44 = None
        getitem_45 = cos_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        cos_6 = None
        cos_8 = torch.repeat_interleave(getitem_45, 2, 3)
        getitem_45 = None
        mul_18 = q_rot_4 * cos_8
        cos_8 = None
        x1_5 = q_rot_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, 2),
            )
        ]
        x2_5 = q_rot_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, 2),
            )
        ]
        q_rot_4 = None
        neg_5 = -x2_5
        x2_5 = None
        x_5 = torch.stack((neg_5, x1_5), dim=-1)
        neg_5 = x1_5 = None
        flatten_5 = x_5.flatten(-2)
        x_5 = None
        mul_19 = flatten_5 * sin_8
        flatten_5 = sin_8 = None
        q_rot_5 = mul_18 + mul_19
        mul_18 = mul_19 = None
        key_9 = torch.cat([k_rot_5, k_pass_2], dim=-1)
        k_rot_5 = k_pass_2 = None
        query_9 = torch.cat([q_rot_5, q_pass_2], dim=-1)
        q_rot_5 = q_pass_2 = None
        key_10 = key_9.permute(0, 2, 1, 3)
        key_9 = None
        query_10 = query_9.permute(0, 2, 1, 3)
        query_9 = None
        query_11 = query_10.to(torch.float32)
        query_10 = None
        key_11 = key_10.to(torch.float32)
        transpose_2 = key_11.transpose(-1, -2)
        key_11 = None
        attn_weights_12 = torch.matmul(query_11, transpose_2)
        query_11 = transpose_2 = None
        attn_weights_13 = (
            attn_weights_12 / l_self_modules_h_modules_2_modules_attn_scale_attn
        )
        attn_weights_12 = l_self_modules_h_modules_2_modules_attn_scale_attn = None
        causal_mask_7 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
            )
        ]
        attn_weights_14 = attn_weights_13 + causal_mask_7
        attn_weights_13 = causal_mask_7 = None
        attn_weights_15 = torch.nn.functional.softmax(attn_weights_14, dim=-1)
        attn_weights_14 = None
        attn_weights_16 = attn_weights_15.to(torch.float32)
        attn_weights_15 = None
        attn_weights_17 = torch.nn.functional.dropout(
            attn_weights_16, 0.0, False, False
        )
        attn_weights_16 = None
        attn_output_8 = torch.matmul(attn_weights_17, value_5)
        attn_weights_17 = None
        permute_11 = attn_output_8.permute(0, 2, 1, 3)
        attn_output_8 = None
        tensor_11 = permute_11.contiguous()
        permute_11 = None
        attn_output_9 = tensor_11.view((1, 20, 32))
        tensor_11 = None
        attn_output_10 = torch._C._nn.linear(
            attn_output_9,
            l_self_modules_h_modules_2_modules_attn_modules_out_proj_parameters_weight_,
            None,
        )
        attn_output_9 = (
            l_self_modules_h_modules_2_modules_attn_modules_out_proj_parameters_weight_
        ) = None
        attn_output_11 = torch.nn.functional.dropout(attn_output_10, 0.0, False, False)
        attn_output_10 = None
        hidden_states_14 = torch._C._nn.linear(
            hidden_states_13,
            l_self_modules_h_modules_2_modules_mlp_modules_fc_in_parameters_weight_,
            l_self_modules_h_modules_2_modules_mlp_modules_fc_in_parameters_bias_,
        )
        hidden_states_13 = (
            l_self_modules_h_modules_2_modules_mlp_modules_fc_in_parameters_weight_
        ) = l_self_modules_h_modules_2_modules_mlp_modules_fc_in_parameters_bias_ = None
        mul_20 = 0.5 * hidden_states_14
        pow_3 = torch.pow(hidden_states_14, 3.0)
        mul_21 = 0.044715 * pow_3
        pow_3 = None
        add_18 = hidden_states_14 + mul_21
        hidden_states_14 = mul_21 = None
        mul_22 = 0.7978845608028654 * add_18
        add_18 = None
        tanh_2 = torch.tanh(mul_22)
        mul_22 = None
        add_19 = 1.0 + tanh_2
        tanh_2 = None
        hidden_states_15 = mul_20 * add_19
        mul_20 = add_19 = None
        hidden_states_16 = torch._C._nn.linear(
            hidden_states_15,
            l_self_modules_h_modules_2_modules_mlp_modules_fc_out_parameters_weight_,
            l_self_modules_h_modules_2_modules_mlp_modules_fc_out_parameters_bias_,
        )
        hidden_states_15 = (
            l_self_modules_h_modules_2_modules_mlp_modules_fc_out_parameters_weight_
        ) = (
            l_self_modules_h_modules_2_modules_mlp_modules_fc_out_parameters_bias_
        ) = None
        hidden_states_17 = torch.nn.functional.dropout(
            hidden_states_16, 0.0, False, False
        )
        hidden_states_16 = None
        add_20 = attn_output_11 + hidden_states_17
        attn_output_11 = hidden_states_17 = None
        hidden_states_18 = add_20 + hidden_states_12
        add_20 = hidden_states_12 = None
        hidden_states_19 = torch.nn.functional.layer_norm(
            hidden_states_18,
            (32,),
            l_self_modules_h_modules_3_modules_ln_1_parameters_weight_,
            l_self_modules_h_modules_3_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_3_modules_ln_1_parameters_weight_ = (
            l_self_modules_h_modules_3_modules_ln_1_parameters_bias_
        ) = None
        query_12 = torch._C._nn.linear(
            hidden_states_19,
            l_self_modules_h_modules_3_modules_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_h_modules_3_modules_attn_modules_q_proj_parameters_weight_ = None
        key_12 = torch._C._nn.linear(
            hidden_states_19,
            l_self_modules_h_modules_3_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_h_modules_3_modules_attn_modules_k_proj_parameters_weight_ = None
        value_6 = torch._C._nn.linear(
            hidden_states_19,
            l_self_modules_h_modules_3_modules_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_h_modules_3_modules_attn_modules_v_proj_parameters_weight_ = None
        tensor_12 = query_12.view((1, 20, 4, 8))
        query_12 = None
        tensor_13 = key_12.view((1, 20, 4, 8))
        key_12 = None
        tensor_14 = value_6.view((1, 20, 4, 8))
        value_6 = None
        value_7 = tensor_14.permute(0, 2, 1, 3)
        tensor_14 = None
        embed_positions_6 = l_self_modules_h_modules_3_modules_attn_embed_positions.to(
            device(type="cuda", index=0)
        )
        l_self_modules_h_modules_3_modules_attn_embed_positions = None
        embed_positions_7 = embed_positions_6.repeat(1, 1, 1)
        unsqueeze_4 = position_ids.unsqueeze(-1)
        repeated_position_ids_3 = unsqueeze_4.repeat(1, 1, 4)
        unsqueeze_4 = None
        sincos_3 = torch.gather(embed_positions_7, 1, repeated_position_ids_3)
        embed_positions_7 = repeated_position_ids_3 = None
        split_3 = torch.functional.split(sincos_3, 2, dim=-1)
        sincos_3 = None
        sin_9 = split_3[0]
        cos_9 = split_3[1]
        split_3 = None
        k_rot_6 = tensor_13[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 4, None),
            )
        ]
        k_pass_3 = tensor_13[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(4, None, None),
            )
        ]
        tensor_13 = None
        q_rot_6 = tensor_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 4, None),
            )
        ]
        q_pass_3 = tensor_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(4, None, None),
            )
        ]
        tensor_12 = None
        getitem_55 = sin_9[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        sin_10 = torch.repeat_interleave(getitem_55, 2, 3)
        getitem_55 = None
        getitem_56 = cos_9[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        cos_10 = torch.repeat_interleave(getitem_56, 2, 3)
        getitem_56 = None
        mul_24 = k_rot_6 * cos_10
        cos_10 = None
        x1_6 = k_rot_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, 2),
            )
        ]
        x2_6 = k_rot_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, 2),
            )
        ]
        k_rot_6 = None
        neg_6 = -x2_6
        x2_6 = None
        x_6 = torch.stack((neg_6, x1_6), dim=-1)
        neg_6 = x1_6 = None
        flatten_6 = x_6.flatten(-2)
        x_6 = None
        mul_25 = flatten_6 * sin_10
        flatten_6 = sin_10 = None
        k_rot_7 = mul_24 + mul_25
        mul_24 = mul_25 = None
        getitem_59 = sin_9[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        sin_9 = None
        sin_11 = torch.repeat_interleave(getitem_59, 2, 3)
        getitem_59 = None
        getitem_60 = cos_9[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        cos_9 = None
        cos_11 = torch.repeat_interleave(getitem_60, 2, 3)
        getitem_60 = None
        mul_26 = q_rot_6 * cos_11
        cos_11 = None
        x1_7 = q_rot_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, 2),
            )
        ]
        x2_7 = q_rot_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, 2),
            )
        ]
        q_rot_6 = None
        neg_7 = -x2_7
        x2_7 = None
        x_7 = torch.stack((neg_7, x1_7), dim=-1)
        neg_7 = x1_7 = None
        flatten_7 = x_7.flatten(-2)
        x_7 = None
        mul_27 = flatten_7 * sin_11
        flatten_7 = sin_11 = None
        q_rot_7 = mul_26 + mul_27
        mul_26 = mul_27 = None
        key_13 = torch.cat([k_rot_7, k_pass_3], dim=-1)
        k_rot_7 = k_pass_3 = None
        query_13 = torch.cat([q_rot_7, q_pass_3], dim=-1)
        q_rot_7 = q_pass_3 = None
        key_14 = key_13.permute(0, 2, 1, 3)
        key_13 = None
        query_14 = query_13.permute(0, 2, 1, 3)
        query_13 = None
        query_15 = query_14.to(torch.float32)
        query_14 = None
        key_15 = key_14.to(torch.float32)
        transpose_3 = key_15.transpose(-1, -2)
        key_15 = None
        attn_weights_18 = torch.matmul(query_15, transpose_3)
        query_15 = transpose_3 = None
        attn_weights_19 = (
            attn_weights_18 / l_self_modules_h_modules_3_modules_attn_scale_attn
        )
        attn_weights_18 = l_self_modules_h_modules_3_modules_attn_scale_attn = None
        causal_mask_8 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
            )
        ]
        attn_weights_20 = attn_weights_19 + causal_mask_8
        attn_weights_19 = causal_mask_8 = None
        attn_weights_21 = torch.nn.functional.softmax(attn_weights_20, dim=-1)
        attn_weights_20 = None
        attn_weights_22 = attn_weights_21.to(torch.float32)
        attn_weights_21 = None
        attn_weights_23 = torch.nn.functional.dropout(
            attn_weights_22, 0.0, False, False
        )
        attn_weights_22 = None
        attn_output_12 = torch.matmul(attn_weights_23, value_7)
        attn_weights_23 = None
        permute_15 = attn_output_12.permute(0, 2, 1, 3)
        attn_output_12 = None
        tensor_15 = permute_15.contiguous()
        permute_15 = None
        attn_output_13 = tensor_15.view((1, 20, 32))
        tensor_15 = None
        attn_output_14 = torch._C._nn.linear(
            attn_output_13,
            l_self_modules_h_modules_3_modules_attn_modules_out_proj_parameters_weight_,
            None,
        )
        attn_output_13 = (
            l_self_modules_h_modules_3_modules_attn_modules_out_proj_parameters_weight_
        ) = None
        attn_output_15 = torch.nn.functional.dropout(attn_output_14, 0.0, False, False)
        attn_output_14 = None
        hidden_states_20 = torch._C._nn.linear(
            hidden_states_19,
            l_self_modules_h_modules_3_modules_mlp_modules_fc_in_parameters_weight_,
            l_self_modules_h_modules_3_modules_mlp_modules_fc_in_parameters_bias_,
        )
        hidden_states_19 = (
            l_self_modules_h_modules_3_modules_mlp_modules_fc_in_parameters_weight_
        ) = l_self_modules_h_modules_3_modules_mlp_modules_fc_in_parameters_bias_ = None
        mul_28 = 0.5 * hidden_states_20
        pow_4 = torch.pow(hidden_states_20, 3.0)
        mul_29 = 0.044715 * pow_4
        pow_4 = None
        add_25 = hidden_states_20 + mul_29
        hidden_states_20 = mul_29 = None
        mul_30 = 0.7978845608028654 * add_25
        add_25 = None
        tanh_3 = torch.tanh(mul_30)
        mul_30 = None
        add_26 = 1.0 + tanh_3
        tanh_3 = None
        hidden_states_21 = mul_28 * add_26
        mul_28 = add_26 = None
        hidden_states_22 = torch._C._nn.linear(
            hidden_states_21,
            l_self_modules_h_modules_3_modules_mlp_modules_fc_out_parameters_weight_,
            l_self_modules_h_modules_3_modules_mlp_modules_fc_out_parameters_bias_,
        )
        hidden_states_21 = (
            l_self_modules_h_modules_3_modules_mlp_modules_fc_out_parameters_weight_
        ) = (
            l_self_modules_h_modules_3_modules_mlp_modules_fc_out_parameters_bias_
        ) = None
        hidden_states_23 = torch.nn.functional.dropout(
            hidden_states_22, 0.0, False, False
        )
        hidden_states_22 = None
        add_27 = attn_output_15 + hidden_states_23
        attn_output_15 = hidden_states_23 = None
        hidden_states_24 = add_27 + hidden_states_18
        add_27 = hidden_states_18 = None
        hidden_states_25 = torch.nn.functional.layer_norm(
            hidden_states_24,
            (32,),
            l_self_modules_h_modules_4_modules_ln_1_parameters_weight_,
            l_self_modules_h_modules_4_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_4_modules_ln_1_parameters_weight_ = (
            l_self_modules_h_modules_4_modules_ln_1_parameters_bias_
        ) = None
        query_16 = torch._C._nn.linear(
            hidden_states_25,
            l_self_modules_h_modules_4_modules_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_h_modules_4_modules_attn_modules_q_proj_parameters_weight_ = None
        key_16 = torch._C._nn.linear(
            hidden_states_25,
            l_self_modules_h_modules_4_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_h_modules_4_modules_attn_modules_k_proj_parameters_weight_ = None
        value_8 = torch._C._nn.linear(
            hidden_states_25,
            l_self_modules_h_modules_4_modules_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_h_modules_4_modules_attn_modules_v_proj_parameters_weight_ = None
        tensor_16 = query_16.view((1, 20, 4, 8))
        query_16 = None
        tensor_17 = key_16.view((1, 20, 4, 8))
        key_16 = None
        tensor_18 = value_8.view((1, 20, 4, 8))
        value_8 = None
        value_9 = tensor_18.permute(0, 2, 1, 3)
        tensor_18 = None
        embed_positions_8 = l_self_modules_h_modules_4_modules_attn_embed_positions.to(
            device(type="cuda", index=0)
        )
        l_self_modules_h_modules_4_modules_attn_embed_positions = None
        embed_positions_9 = embed_positions_8.repeat(1, 1, 1)
        unsqueeze_5 = position_ids.unsqueeze(-1)
        position_ids = None
        repeated_position_ids_4 = unsqueeze_5.repeat(1, 1, 4)
        unsqueeze_5 = None
        sincos_4 = torch.gather(embed_positions_9, 1, repeated_position_ids_4)
        embed_positions_9 = repeated_position_ids_4 = None
        split_4 = torch.functional.split(sincos_4, 2, dim=-1)
        sincos_4 = None
        sin_12 = split_4[0]
        cos_12 = split_4[1]
        split_4 = None
        k_rot_8 = tensor_17[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 4, None),
            )
        ]
        k_pass_4 = tensor_17[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(4, None, None),
            )
        ]
        tensor_17 = None
        q_rot_8 = tensor_16[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 4, None),
            )
        ]
        q_pass_4 = tensor_16[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(4, None, None),
            )
        ]
        tensor_16 = None
        getitem_70 = sin_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        sin_13 = torch.repeat_interleave(getitem_70, 2, 3)
        getitem_70 = None
        getitem_71 = cos_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        cos_13 = torch.repeat_interleave(getitem_71, 2, 3)
        getitem_71 = None
        mul_32 = k_rot_8 * cos_13
        cos_13 = None
        x1_8 = k_rot_8[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, 2),
            )
        ]
        x2_8 = k_rot_8[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, 2),
            )
        ]
        k_rot_8 = None
        neg_8 = -x2_8
        x2_8 = None
        x_8 = torch.stack((neg_8, x1_8), dim=-1)
        neg_8 = x1_8 = None
        flatten_8 = x_8.flatten(-2)
        x_8 = None
        mul_33 = flatten_8 * sin_13
        flatten_8 = sin_13 = None
        k_rot_9 = mul_32 + mul_33
        mul_32 = mul_33 = None
        getitem_74 = sin_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        sin_12 = None
        sin_14 = torch.repeat_interleave(getitem_74, 2, 3)
        getitem_74 = None
        getitem_75 = cos_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        cos_12 = None
        cos_14 = torch.repeat_interleave(getitem_75, 2, 3)
        getitem_75 = None
        mul_34 = q_rot_8 * cos_14
        cos_14 = None
        x1_9 = q_rot_8[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, 2),
            )
        ]
        x2_9 = q_rot_8[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, 2),
            )
        ]
        q_rot_8 = None
        neg_9 = -x2_9
        x2_9 = None
        x_9 = torch.stack((neg_9, x1_9), dim=-1)
        neg_9 = x1_9 = None
        flatten_9 = x_9.flatten(-2)
        x_9 = None
        mul_35 = flatten_9 * sin_14
        flatten_9 = sin_14 = None
        q_rot_9 = mul_34 + mul_35
        mul_34 = mul_35 = None
        key_17 = torch.cat([k_rot_9, k_pass_4], dim=-1)
        k_rot_9 = k_pass_4 = None
        query_17 = torch.cat([q_rot_9, q_pass_4], dim=-1)
        q_rot_9 = q_pass_4 = None
        key_18 = key_17.permute(0, 2, 1, 3)
        key_17 = None
        query_18 = query_17.permute(0, 2, 1, 3)
        query_17 = None
        query_19 = query_18.to(torch.float32)
        query_18 = None
        key_19 = key_18.to(torch.float32)
        transpose_4 = key_19.transpose(-1, -2)
        key_19 = None
        attn_weights_24 = torch.matmul(query_19, transpose_4)
        query_19 = transpose_4 = None
        attn_weights_25 = (
            attn_weights_24 / l_self_modules_h_modules_4_modules_attn_scale_attn
        )
        attn_weights_24 = l_self_modules_h_modules_4_modules_attn_scale_attn = None
        causal_mask_9 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
            )
        ]
        causal_mask_4 = None
        attn_weights_26 = attn_weights_25 + causal_mask_9
        attn_weights_25 = causal_mask_9 = None
        attn_weights_27 = torch.nn.functional.softmax(attn_weights_26, dim=-1)
        attn_weights_26 = None
        attn_weights_28 = attn_weights_27.to(torch.float32)
        attn_weights_27 = None
        attn_weights_29 = torch.nn.functional.dropout(
            attn_weights_28, 0.0, False, False
        )
        attn_weights_28 = None
        attn_output_16 = torch.matmul(attn_weights_29, value_9)
        attn_weights_29 = None
        permute_19 = attn_output_16.permute(0, 2, 1, 3)
        attn_output_16 = None
        tensor_19 = permute_19.contiguous()
        permute_19 = None
        attn_output_17 = tensor_19.view((1, 20, 32))
        tensor_19 = None
        attn_output_18 = torch._C._nn.linear(
            attn_output_17,
            l_self_modules_h_modules_4_modules_attn_modules_out_proj_parameters_weight_,
            None,
        )
        attn_output_17 = (
            l_self_modules_h_modules_4_modules_attn_modules_out_proj_parameters_weight_
        ) = None
        attn_output_19 = torch.nn.functional.dropout(attn_output_18, 0.0, False, False)
        attn_output_18 = None
        hidden_states_26 = torch._C._nn.linear(
            hidden_states_25,
            l_self_modules_h_modules_4_modules_mlp_modules_fc_in_parameters_weight_,
            l_self_modules_h_modules_4_modules_mlp_modules_fc_in_parameters_bias_,
        )
        hidden_states_25 = (
            l_self_modules_h_modules_4_modules_mlp_modules_fc_in_parameters_weight_
        ) = l_self_modules_h_modules_4_modules_mlp_modules_fc_in_parameters_bias_ = None
        mul_36 = 0.5 * hidden_states_26
        pow_5 = torch.pow(hidden_states_26, 3.0)
        mul_37 = 0.044715 * pow_5
        pow_5 = None
        add_32 = hidden_states_26 + mul_37
        hidden_states_26 = mul_37 = None
        mul_38 = 0.7978845608028654 * add_32
        add_32 = None
        tanh_4 = torch.tanh(mul_38)
        mul_38 = None
        add_33 = 1.0 + tanh_4
        tanh_4 = None
        hidden_states_27 = mul_36 * add_33
        mul_36 = add_33 = None
        hidden_states_28 = torch._C._nn.linear(
            hidden_states_27,
            l_self_modules_h_modules_4_modules_mlp_modules_fc_out_parameters_weight_,
            l_self_modules_h_modules_4_modules_mlp_modules_fc_out_parameters_bias_,
        )
        hidden_states_27 = (
            l_self_modules_h_modules_4_modules_mlp_modules_fc_out_parameters_weight_
        ) = (
            l_self_modules_h_modules_4_modules_mlp_modules_fc_out_parameters_bias_
        ) = None
        hidden_states_29 = torch.nn.functional.dropout(
            hidden_states_28, 0.0, False, False
        )
        hidden_states_28 = None
        add_34 = attn_output_19 + hidden_states_29
        attn_output_19 = hidden_states_29 = None
        hidden_states_30 = add_34 + hidden_states_24
        add_34 = hidden_states_24 = None
        hidden_states_31 = torch.nn.functional.layer_norm(
            hidden_states_30,
            (32,),
            l_self_modules_ln_f_parameters_weight_,
            l_self_modules_ln_f_parameters_bias_,
            1e-05,
        )
        hidden_states_30 = (
            l_self_modules_ln_f_parameters_weight_
        ) = l_self_modules_ln_f_parameters_bias_ = None
        hidden_states_32 = hidden_states_31.view((-1, 20, 32))
        hidden_states_31 = None
        return (
            value_1,
            key_2,
            embed_positions,
            embed_positions_2,
            value_3,
            key_6,
            embed_positions_4,
            value_5,
            key_10,
            embed_positions_6,
            value_7,
            key_14,
            embed_positions_8,
            value_9,
            key_18,
            hidden_states_32,
        )
