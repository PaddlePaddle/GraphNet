import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_inputs_embeds_: torch.Tensor,
        L_attention_mask_: torch.Tensor,
        L_self_modules_embed_positions_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_inputs_embeds_ = L_inputs_embeds_
        l_attention_mask_ = L_attention_mask_
        l_self_modules_embed_positions_parameters_weight_ = (
            L_self_modules_embed_positions_parameters_weight_
        )
        l_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_
        )
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
        l_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_
        )
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
        l_self_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_
        )
        l_self_modules_layers_modules_2_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_2_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_2_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_2_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_2_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_2_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_2_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_2_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_
        )
        l_self_modules_layers_modules_3_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_3_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_3_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_3_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_3_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_3_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_3_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_3_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_
        )
        l_self_modules_layers_modules_4_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_4_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_4_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_4_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_4_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_4_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_4_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_4_modules_fc2_parameters_bias_
        )
        l_self_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_final_layer_norm_parameters_bias_
        )
        cache_position = torch.arange(0, 21, device=device(type="cuda", index=0))
        causal_mask = torch.full(
            (21, 21),
            fill_value=-3.4028234663852886e38,
            dtype=torch.float32,
            device=device(type="cuda", index=0),
        )
        causal_mask_1 = torch.triu(causal_mask, diagonal=1)
        causal_mask = None
        arange_1 = torch.arange(21, device=device(type="cuda", index=0))
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
                slice(None, 21, None),
            )
        ]
        getitem_2 = l_attention_mask_[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
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
                slice(None, 21, None),
            )
        ]
        masked_fill = getitem_3.masked_fill(padding_mask_1, -3.4028234663852886e38)
        getitem_3 = padding_mask_1 = None
        causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 21, None),
            )
        ] = masked_fill
        setitem = causal_mask_4
        masked_fill = setitem = None
        eq_1 = causal_mask_4.__eq__(-3.4028234663852886e38)
        all_1 = torch.all(eq_1, dim=-1, keepdim=True)
        eq_1 = None
        invert = ~all_1
        all_1 = None
        causal_mask_5 = causal_mask_4.mul(invert)
        causal_mask_4 = invert = None
        position_ids = torch.cumsum(l_attention_mask_, dim=1)
        mul_1 = position_ids * l_attention_mask_
        position_ids = l_attention_mask_ = None
        sub = mul_1 - 1
        mul_1 = None
        position_ids_1 = sub.long()
        sub = None
        position_ids_2 = position_ids_1[(slice(None, None, None), slice(0, None, None))]
        position_ids_1 = None
        add_1 = position_ids_2 + 2
        position_ids_2 = None
        pos_embeds = torch.nn.functional.embedding(
            add_1,
            l_self_modules_embed_positions_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        add_1 = l_self_modules_embed_positions_parameters_weight_ = None
        to_1 = pos_embeds.to(device(type="cuda", index=0))
        pos_embeds = None
        hidden_states = l_inputs_embeds_ + to_1
        l_inputs_embeds_ = to_1 = None
        hidden_states_1 = torch.nn.functional.layer_norm(
            hidden_states,
            (16,),
            l_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear = torch._C._nn.linear(
            hidden_states_1,
            l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states = linear * 0.5
        linear = None
        view = query_states.view(1, -1, 4, 4)
        query_states = None
        query_states_1 = view.transpose(1, 2)
        view = None
        key_states = torch._C._nn.linear(
            hidden_states_1,
            l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states = torch._C._nn.linear(
            hidden_states_1,
            l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_1 = l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_1 = key_states.view(1, -1, 4, 4)
        key_states = None
        key_states_1 = view_1.transpose(1, 2)
        view_1 = None
        view_2 = value_states.view(1, -1, 4, 4)
        value_states = None
        value_states_1 = view_2.transpose(1, 2)
        view_2 = None
        attention_mask = causal_mask_5[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 21, None),
            )
        ]
        query = query_states_1.contiguous()
        query_states_1 = None
        key = key_states_1.contiguous()
        value = value_states_1.contiguous()
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
        reshape_1 = attn_output_1.reshape(1, 21, -1)
        attn_output_1 = None
        attn_output_2 = reshape_1.contiguous()
        reshape_1 = None
        attn_output_3 = torch._C._nn.linear(
            attn_output_2,
            l_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_2 = l_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_2 = torch.nn.functional.dropout(
            attn_output_3, p=0.1, training=False
        )
        attn_output_3 = None
        hidden_states_3 = hidden_states + hidden_states_2
        hidden_states = hidden_states_2 = None
        hidden_states_4 = hidden_states_3.reshape(-1, 16)
        hidden_states_3 = None
        hidden_states_5 = torch.nn.functional.layer_norm(
            hidden_states_4,
            (16,),
            l_self_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_ = (
            l_self_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_
        ) = None
        hidden_states_6 = torch._C._nn.linear(
            hidden_states_5,
            l_self_modules_layers_modules_0_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_0_modules_fc1_parameters_bias_,
        )
        hidden_states_5 = (
            l_self_modules_layers_modules_0_modules_fc1_parameters_weight_
        ) = l_self_modules_layers_modules_0_modules_fc1_parameters_bias_ = None
        hidden_states_7 = torch.nn.functional.relu(hidden_states_6, inplace=False)
        hidden_states_6 = None
        hidden_states_8 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_layers_modules_0_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_0_modules_fc2_parameters_bias_,
        )
        hidden_states_7 = (
            l_self_modules_layers_modules_0_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_0_modules_fc2_parameters_bias_ = None
        hidden_states_9 = torch.nn.functional.dropout(
            hidden_states_8, p=0.1, training=False
        )
        hidden_states_8 = None
        add_4 = hidden_states_4 + hidden_states_9
        hidden_states_4 = hidden_states_9 = None
        hidden_states_10 = add_4.view((1, 21, 16))
        add_4 = None
        hidden_states_11 = torch.nn.functional.layer_norm(
            hidden_states_10,
            (16,),
            l_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_6 = torch._C._nn.linear(
            hidden_states_11,
            l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_2 = linear_6 * 0.5
        linear_6 = None
        view_4 = query_states_2.view(1, -1, 4, 4)
        query_states_2 = None
        query_states_3 = view_4.transpose(1, 2)
        view_4 = None
        key_states_2 = torch._C._nn.linear(
            hidden_states_11,
            l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_2 = torch._C._nn.linear(
            hidden_states_11,
            l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_11 = l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_5 = key_states_2.view(1, -1, 4, 4)
        key_states_2 = None
        key_states_3 = view_5.transpose(1, 2)
        view_5 = None
        view_6 = value_states_2.view(1, -1, 4, 4)
        value_states_2 = None
        value_states_3 = view_6.transpose(1, 2)
        view_6 = None
        attention_mask_1 = causal_mask_5[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 21, None),
            )
        ]
        query_1 = query_states_3.contiguous()
        query_states_3 = None
        key_1 = key_states_3.contiguous()
        value_1 = value_states_3.contiguous()
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
        reshape_3 = attn_output_5.reshape(1, 21, -1)
        attn_output_5 = None
        attn_output_6 = reshape_3.contiguous()
        reshape_3 = None
        attn_output_7 = torch._C._nn.linear(
            attn_output_6,
            l_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_6 = l_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_12 = torch.nn.functional.dropout(
            attn_output_7, p=0.1, training=False
        )
        attn_output_7 = None
        hidden_states_13 = hidden_states_10 + hidden_states_12
        hidden_states_10 = hidden_states_12 = None
        hidden_states_14 = hidden_states_13.reshape(-1, 16)
        hidden_states_13 = None
        hidden_states_15 = torch.nn.functional.layer_norm(
            hidden_states_14,
            (16,),
            l_self_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_ = (
            l_self_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_
        ) = None
        hidden_states_16 = torch._C._nn.linear(
            hidden_states_15,
            l_self_modules_layers_modules_1_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_1_modules_fc1_parameters_bias_,
        )
        hidden_states_15 = (
            l_self_modules_layers_modules_1_modules_fc1_parameters_weight_
        ) = l_self_modules_layers_modules_1_modules_fc1_parameters_bias_ = None
        hidden_states_17 = torch.nn.functional.relu(hidden_states_16, inplace=False)
        hidden_states_16 = None
        hidden_states_18 = torch._C._nn.linear(
            hidden_states_17,
            l_self_modules_layers_modules_1_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_1_modules_fc2_parameters_bias_,
        )
        hidden_states_17 = (
            l_self_modules_layers_modules_1_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_1_modules_fc2_parameters_bias_ = None
        hidden_states_19 = torch.nn.functional.dropout(
            hidden_states_18, p=0.1, training=False
        )
        hidden_states_18 = None
        add_6 = hidden_states_14 + hidden_states_19
        hidden_states_14 = hidden_states_19 = None
        hidden_states_20 = add_6.view((1, 21, 16))
        add_6 = None
        hidden_states_21 = torch.nn.functional.layer_norm(
            hidden_states_20,
            (16,),
            l_self_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_12 = torch._C._nn.linear(
            hidden_states_21,
            l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_4 = linear_12 * 0.5
        linear_12 = None
        view_8 = query_states_4.view(1, -1, 4, 4)
        query_states_4 = None
        query_states_5 = view_8.transpose(1, 2)
        view_8 = None
        key_states_4 = torch._C._nn.linear(
            hidden_states_21,
            l_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_4 = torch._C._nn.linear(
            hidden_states_21,
            l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_21 = l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_9 = key_states_4.view(1, -1, 4, 4)
        key_states_4 = None
        key_states_5 = view_9.transpose(1, 2)
        view_9 = None
        view_10 = value_states_4.view(1, -1, 4, 4)
        value_states_4 = None
        value_states_5 = view_10.transpose(1, 2)
        view_10 = None
        attention_mask_2 = causal_mask_5[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 21, None),
            )
        ]
        query_2 = query_states_5.contiguous()
        query_states_5 = None
        key_2 = key_states_5.contiguous()
        value_2 = value_states_5.contiguous()
        attn_output_8 = torch._C._nn.scaled_dot_product_attention(
            query_2,
            key_2,
            value_2,
            attn_mask=attention_mask_2,
            dropout_p=0.0,
            scale=1.0,
            is_causal=False,
        )
        query_2 = key_2 = value_2 = attention_mask_2 = None
        transpose_11 = attn_output_8.transpose(1, 2)
        attn_output_8 = None
        attn_output_9 = transpose_11.contiguous()
        transpose_11 = None
        reshape_5 = attn_output_9.reshape(1, 21, -1)
        attn_output_9 = None
        attn_output_10 = reshape_5.contiguous()
        reshape_5 = None
        attn_output_11 = torch._C._nn.linear(
            attn_output_10,
            l_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_10 = l_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_22 = torch.nn.functional.dropout(
            attn_output_11, p=0.1, training=False
        )
        attn_output_11 = None
        hidden_states_23 = hidden_states_20 + hidden_states_22
        hidden_states_20 = hidden_states_22 = None
        hidden_states_24 = hidden_states_23.reshape(-1, 16)
        hidden_states_23 = None
        hidden_states_25 = torch.nn.functional.layer_norm(
            hidden_states_24,
            (16,),
            l_self_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_ = (
            l_self_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_
        ) = None
        hidden_states_26 = torch._C._nn.linear(
            hidden_states_25,
            l_self_modules_layers_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_fc1_parameters_bias_,
        )
        hidden_states_25 = (
            l_self_modules_layers_modules_2_modules_fc1_parameters_weight_
        ) = l_self_modules_layers_modules_2_modules_fc1_parameters_bias_ = None
        hidden_states_27 = torch.nn.functional.relu(hidden_states_26, inplace=False)
        hidden_states_26 = None
        hidden_states_28 = torch._C._nn.linear(
            hidden_states_27,
            l_self_modules_layers_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_fc2_parameters_bias_,
        )
        hidden_states_27 = (
            l_self_modules_layers_modules_2_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_2_modules_fc2_parameters_bias_ = None
        hidden_states_29 = torch.nn.functional.dropout(
            hidden_states_28, p=0.1, training=False
        )
        hidden_states_28 = None
        add_8 = hidden_states_24 + hidden_states_29
        hidden_states_24 = hidden_states_29 = None
        hidden_states_30 = add_8.view((1, 21, 16))
        add_8 = None
        hidden_states_31 = torch.nn.functional.layer_norm(
            hidden_states_30,
            (16,),
            l_self_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_18 = torch._C._nn.linear(
            hidden_states_31,
            l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_6 = linear_18 * 0.5
        linear_18 = None
        view_12 = query_states_6.view(1, -1, 4, 4)
        query_states_6 = None
        query_states_7 = view_12.transpose(1, 2)
        view_12 = None
        key_states_6 = torch._C._nn.linear(
            hidden_states_31,
            l_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_6 = torch._C._nn.linear(
            hidden_states_31,
            l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_31 = l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_13 = key_states_6.view(1, -1, 4, 4)
        key_states_6 = None
        key_states_7 = view_13.transpose(1, 2)
        view_13 = None
        view_14 = value_states_6.view(1, -1, 4, 4)
        value_states_6 = None
        value_states_7 = view_14.transpose(1, 2)
        view_14 = None
        attention_mask_3 = causal_mask_5[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 21, None),
            )
        ]
        query_3 = query_states_7.contiguous()
        query_states_7 = None
        key_3 = key_states_7.contiguous()
        value_3 = value_states_7.contiguous()
        attn_output_12 = torch._C._nn.scaled_dot_product_attention(
            query_3,
            key_3,
            value_3,
            attn_mask=attention_mask_3,
            dropout_p=0.0,
            scale=1.0,
            is_causal=False,
        )
        query_3 = key_3 = value_3 = attention_mask_3 = None
        transpose_15 = attn_output_12.transpose(1, 2)
        attn_output_12 = None
        attn_output_13 = transpose_15.contiguous()
        transpose_15 = None
        reshape_7 = attn_output_13.reshape(1, 21, -1)
        attn_output_13 = None
        attn_output_14 = reshape_7.contiguous()
        reshape_7 = None
        attn_output_15 = torch._C._nn.linear(
            attn_output_14,
            l_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_14 = l_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_32 = torch.nn.functional.dropout(
            attn_output_15, p=0.1, training=False
        )
        attn_output_15 = None
        hidden_states_33 = hidden_states_30 + hidden_states_32
        hidden_states_30 = hidden_states_32 = None
        hidden_states_34 = hidden_states_33.reshape(-1, 16)
        hidden_states_33 = None
        hidden_states_35 = torch.nn.functional.layer_norm(
            hidden_states_34,
            (16,),
            l_self_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_ = (
            l_self_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_
        ) = None
        hidden_states_36 = torch._C._nn.linear(
            hidden_states_35,
            l_self_modules_layers_modules_3_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_3_modules_fc1_parameters_bias_,
        )
        hidden_states_35 = (
            l_self_modules_layers_modules_3_modules_fc1_parameters_weight_
        ) = l_self_modules_layers_modules_3_modules_fc1_parameters_bias_ = None
        hidden_states_37 = torch.nn.functional.relu(hidden_states_36, inplace=False)
        hidden_states_36 = None
        hidden_states_38 = torch._C._nn.linear(
            hidden_states_37,
            l_self_modules_layers_modules_3_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_3_modules_fc2_parameters_bias_,
        )
        hidden_states_37 = (
            l_self_modules_layers_modules_3_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_3_modules_fc2_parameters_bias_ = None
        hidden_states_39 = torch.nn.functional.dropout(
            hidden_states_38, p=0.1, training=False
        )
        hidden_states_38 = None
        add_10 = hidden_states_34 + hidden_states_39
        hidden_states_34 = hidden_states_39 = None
        hidden_states_40 = add_10.view((1, 21, 16))
        add_10 = None
        hidden_states_41 = torch.nn.functional.layer_norm(
            hidden_states_40,
            (16,),
            l_self_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_24 = torch._C._nn.linear(
            hidden_states_41,
            l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_8 = linear_24 * 0.5
        linear_24 = None
        view_16 = query_states_8.view(1, -1, 4, 4)
        query_states_8 = None
        query_states_9 = view_16.transpose(1, 2)
        view_16 = None
        key_states_8 = torch._C._nn.linear(
            hidden_states_41,
            l_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_8 = torch._C._nn.linear(
            hidden_states_41,
            l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_41 = l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_17 = key_states_8.view(1, -1, 4, 4)
        key_states_8 = None
        key_states_9 = view_17.transpose(1, 2)
        view_17 = None
        view_18 = value_states_8.view(1, -1, 4, 4)
        value_states_8 = None
        value_states_9 = view_18.transpose(1, 2)
        view_18 = None
        attention_mask_4 = causal_mask_5[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 21, None),
            )
        ]
        causal_mask_5 = None
        query_4 = query_states_9.contiguous()
        query_states_9 = None
        key_4 = key_states_9.contiguous()
        value_4 = value_states_9.contiguous()
        attn_output_16 = torch._C._nn.scaled_dot_product_attention(
            query_4,
            key_4,
            value_4,
            attn_mask=attention_mask_4,
            dropout_p=0.0,
            scale=1.0,
            is_causal=False,
        )
        query_4 = key_4 = value_4 = attention_mask_4 = None
        transpose_19 = attn_output_16.transpose(1, 2)
        attn_output_16 = None
        attn_output_17 = transpose_19.contiguous()
        transpose_19 = None
        reshape_9 = attn_output_17.reshape(1, 21, -1)
        attn_output_17 = None
        attn_output_18 = reshape_9.contiguous()
        reshape_9 = None
        attn_output_19 = torch._C._nn.linear(
            attn_output_18,
            l_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_18 = l_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_42 = torch.nn.functional.dropout(
            attn_output_19, p=0.1, training=False
        )
        attn_output_19 = None
        hidden_states_43 = hidden_states_40 + hidden_states_42
        hidden_states_40 = hidden_states_42 = None
        hidden_states_44 = hidden_states_43.reshape(-1, 16)
        hidden_states_43 = None
        hidden_states_45 = torch.nn.functional.layer_norm(
            hidden_states_44,
            (16,),
            l_self_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_ = (
            l_self_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_
        ) = None
        hidden_states_46 = torch._C._nn.linear(
            hidden_states_45,
            l_self_modules_layers_modules_4_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_4_modules_fc1_parameters_bias_,
        )
        hidden_states_45 = (
            l_self_modules_layers_modules_4_modules_fc1_parameters_weight_
        ) = l_self_modules_layers_modules_4_modules_fc1_parameters_bias_ = None
        hidden_states_47 = torch.nn.functional.relu(hidden_states_46, inplace=False)
        hidden_states_46 = None
        hidden_states_48 = torch._C._nn.linear(
            hidden_states_47,
            l_self_modules_layers_modules_4_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_4_modules_fc2_parameters_bias_,
        )
        hidden_states_47 = (
            l_self_modules_layers_modules_4_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_4_modules_fc2_parameters_bias_ = None
        hidden_states_49 = torch.nn.functional.dropout(
            hidden_states_48, p=0.1, training=False
        )
        hidden_states_48 = None
        add_12 = hidden_states_44 + hidden_states_49
        hidden_states_44 = hidden_states_49 = None
        hidden_states_50 = add_12.view((1, 21, 16))
        add_12 = None
        hidden_states_51 = torch.nn.functional.layer_norm(
            hidden_states_50,
            (16,),
            l_self_modules_final_layer_norm_parameters_weight_,
            l_self_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_50 = (
            l_self_modules_final_layer_norm_parameters_weight_
        ) = l_self_modules_final_layer_norm_parameters_bias_ = None
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
            hidden_states_51,
        )
