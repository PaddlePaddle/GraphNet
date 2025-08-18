import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_input_ids_: torch.Tensor,
        L_attention_mask_: torch.Tensor,
        L_self_buffers_position_ids_: torch.Tensor,
        L_self_modules_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_position_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_token_type_ids_: torch.Tensor,
        L_self_modules_layer_norm_emb_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm_emb_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_0_modules_q_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_0_modules_q_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_0_modules_k_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_0_modules_k_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_0_modules_v_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_0_modules_v_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_0_modules_out_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_0_modules_out_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm1_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_0_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_0_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_0_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_0_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm2_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_1_modules_q_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_1_modules_q_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_1_modules_k_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_1_modules_k_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_1_modules_v_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_1_modules_v_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_1_modules_out_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_1_modules_out_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_1_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_1_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_1_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_1_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_2_modules_q_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_2_modules_q_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_2_modules_k_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_2_modules_k_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_2_modules_v_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_2_modules_v_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_2_modules_out_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_2_modules_out_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm1_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm1_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_2_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_2_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_2_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_2_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_3_modules_q_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_3_modules_q_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_3_modules_k_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_3_modules_k_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_3_modules_v_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_3_modules_v_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_3_modules_out_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_3_modules_out_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm1_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm1_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_3_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_3_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_3_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_3_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm2_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm2_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_4_modules_q_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_4_modules_q_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_4_modules_k_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_4_modules_k_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_4_modules_v_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_4_modules_v_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_4_modules_out_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_attentions_modules_4_modules_out_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm1_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm1_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_4_modules_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_4_modules_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_4_modules_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_ffns_modules_4_modules_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm2_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm2_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_input_ids_ = L_input_ids_
        l_attention_mask_ = L_attention_mask_
        l_self_buffers_position_ids_ = L_self_buffers_position_ids_
        l_self_modules_embeddings_parameters_weight_ = (
            L_self_modules_embeddings_parameters_weight_
        )
        l_self_modules_position_embeddings_parameters_weight_ = (
            L_self_modules_position_embeddings_parameters_weight_
        )
        l_token_type_ids_ = L_token_type_ids_
        l_self_modules_layer_norm_emb_parameters_weight_ = (
            L_self_modules_layer_norm_emb_parameters_weight_
        )
        l_self_modules_layer_norm_emb_parameters_bias_ = (
            L_self_modules_layer_norm_emb_parameters_bias_
        )
        l_self_modules_attentions_modules_0_modules_q_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_0_modules_q_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_0_modules_q_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_0_modules_q_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_0_modules_k_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_0_modules_k_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_0_modules_k_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_0_modules_k_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_0_modules_v_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_0_modules_v_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_0_modules_v_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_0_modules_v_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_0_modules_out_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_0_modules_out_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_0_modules_out_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_0_modules_out_lin_parameters_bias_
        )
        l_self_modules_layer_norm1_modules_0_parameters_weight_ = (
            L_self_modules_layer_norm1_modules_0_parameters_weight_
        )
        l_self_modules_layer_norm1_modules_0_parameters_bias_ = (
            L_self_modules_layer_norm1_modules_0_parameters_bias_
        )
        l_self_modules_ffns_modules_0_modules_lin1_parameters_weight_ = (
            L_self_modules_ffns_modules_0_modules_lin1_parameters_weight_
        )
        l_self_modules_ffns_modules_0_modules_lin1_parameters_bias_ = (
            L_self_modules_ffns_modules_0_modules_lin1_parameters_bias_
        )
        l_self_modules_ffns_modules_0_modules_lin2_parameters_weight_ = (
            L_self_modules_ffns_modules_0_modules_lin2_parameters_weight_
        )
        l_self_modules_ffns_modules_0_modules_lin2_parameters_bias_ = (
            L_self_modules_ffns_modules_0_modules_lin2_parameters_bias_
        )
        l_self_modules_layer_norm2_modules_0_parameters_weight_ = (
            L_self_modules_layer_norm2_modules_0_parameters_weight_
        )
        l_self_modules_layer_norm2_modules_0_parameters_bias_ = (
            L_self_modules_layer_norm2_modules_0_parameters_bias_
        )
        l_self_modules_attentions_modules_1_modules_q_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_1_modules_q_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_1_modules_q_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_1_modules_q_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_1_modules_k_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_1_modules_k_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_1_modules_k_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_1_modules_k_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_1_modules_v_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_1_modules_v_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_1_modules_v_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_1_modules_v_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_1_modules_out_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_1_modules_out_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_1_modules_out_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_1_modules_out_lin_parameters_bias_
        )
        l_self_modules_layer_norm1_modules_1_parameters_weight_ = (
            L_self_modules_layer_norm1_modules_1_parameters_weight_
        )
        l_self_modules_layer_norm1_modules_1_parameters_bias_ = (
            L_self_modules_layer_norm1_modules_1_parameters_bias_
        )
        l_self_modules_ffns_modules_1_modules_lin1_parameters_weight_ = (
            L_self_modules_ffns_modules_1_modules_lin1_parameters_weight_
        )
        l_self_modules_ffns_modules_1_modules_lin1_parameters_bias_ = (
            L_self_modules_ffns_modules_1_modules_lin1_parameters_bias_
        )
        l_self_modules_ffns_modules_1_modules_lin2_parameters_weight_ = (
            L_self_modules_ffns_modules_1_modules_lin2_parameters_weight_
        )
        l_self_modules_ffns_modules_1_modules_lin2_parameters_bias_ = (
            L_self_modules_ffns_modules_1_modules_lin2_parameters_bias_
        )
        l_self_modules_layer_norm2_modules_1_parameters_weight_ = (
            L_self_modules_layer_norm2_modules_1_parameters_weight_
        )
        l_self_modules_layer_norm2_modules_1_parameters_bias_ = (
            L_self_modules_layer_norm2_modules_1_parameters_bias_
        )
        l_self_modules_attentions_modules_2_modules_q_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_2_modules_q_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_2_modules_q_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_2_modules_q_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_2_modules_k_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_2_modules_k_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_2_modules_k_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_2_modules_k_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_2_modules_v_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_2_modules_v_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_2_modules_v_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_2_modules_v_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_2_modules_out_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_2_modules_out_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_2_modules_out_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_2_modules_out_lin_parameters_bias_
        )
        l_self_modules_layer_norm1_modules_2_parameters_weight_ = (
            L_self_modules_layer_norm1_modules_2_parameters_weight_
        )
        l_self_modules_layer_norm1_modules_2_parameters_bias_ = (
            L_self_modules_layer_norm1_modules_2_parameters_bias_
        )
        l_self_modules_ffns_modules_2_modules_lin1_parameters_weight_ = (
            L_self_modules_ffns_modules_2_modules_lin1_parameters_weight_
        )
        l_self_modules_ffns_modules_2_modules_lin1_parameters_bias_ = (
            L_self_modules_ffns_modules_2_modules_lin1_parameters_bias_
        )
        l_self_modules_ffns_modules_2_modules_lin2_parameters_weight_ = (
            L_self_modules_ffns_modules_2_modules_lin2_parameters_weight_
        )
        l_self_modules_ffns_modules_2_modules_lin2_parameters_bias_ = (
            L_self_modules_ffns_modules_2_modules_lin2_parameters_bias_
        )
        l_self_modules_layer_norm2_modules_2_parameters_weight_ = (
            L_self_modules_layer_norm2_modules_2_parameters_weight_
        )
        l_self_modules_layer_norm2_modules_2_parameters_bias_ = (
            L_self_modules_layer_norm2_modules_2_parameters_bias_
        )
        l_self_modules_attentions_modules_3_modules_q_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_3_modules_q_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_3_modules_q_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_3_modules_q_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_3_modules_k_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_3_modules_k_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_3_modules_k_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_3_modules_k_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_3_modules_v_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_3_modules_v_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_3_modules_v_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_3_modules_v_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_3_modules_out_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_3_modules_out_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_3_modules_out_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_3_modules_out_lin_parameters_bias_
        )
        l_self_modules_layer_norm1_modules_3_parameters_weight_ = (
            L_self_modules_layer_norm1_modules_3_parameters_weight_
        )
        l_self_modules_layer_norm1_modules_3_parameters_bias_ = (
            L_self_modules_layer_norm1_modules_3_parameters_bias_
        )
        l_self_modules_ffns_modules_3_modules_lin1_parameters_weight_ = (
            L_self_modules_ffns_modules_3_modules_lin1_parameters_weight_
        )
        l_self_modules_ffns_modules_3_modules_lin1_parameters_bias_ = (
            L_self_modules_ffns_modules_3_modules_lin1_parameters_bias_
        )
        l_self_modules_ffns_modules_3_modules_lin2_parameters_weight_ = (
            L_self_modules_ffns_modules_3_modules_lin2_parameters_weight_
        )
        l_self_modules_ffns_modules_3_modules_lin2_parameters_bias_ = (
            L_self_modules_ffns_modules_3_modules_lin2_parameters_bias_
        )
        l_self_modules_layer_norm2_modules_3_parameters_weight_ = (
            L_self_modules_layer_norm2_modules_3_parameters_weight_
        )
        l_self_modules_layer_norm2_modules_3_parameters_bias_ = (
            L_self_modules_layer_norm2_modules_3_parameters_bias_
        )
        l_self_modules_attentions_modules_4_modules_q_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_4_modules_q_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_4_modules_q_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_4_modules_q_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_4_modules_k_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_4_modules_k_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_4_modules_k_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_4_modules_k_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_4_modules_v_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_4_modules_v_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_4_modules_v_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_4_modules_v_lin_parameters_bias_
        )
        l_self_modules_attentions_modules_4_modules_out_lin_parameters_weight_ = (
            L_self_modules_attentions_modules_4_modules_out_lin_parameters_weight_
        )
        l_self_modules_attentions_modules_4_modules_out_lin_parameters_bias_ = (
            L_self_modules_attentions_modules_4_modules_out_lin_parameters_bias_
        )
        l_self_modules_layer_norm1_modules_4_parameters_weight_ = (
            L_self_modules_layer_norm1_modules_4_parameters_weight_
        )
        l_self_modules_layer_norm1_modules_4_parameters_bias_ = (
            L_self_modules_layer_norm1_modules_4_parameters_bias_
        )
        l_self_modules_ffns_modules_4_modules_lin1_parameters_weight_ = (
            L_self_modules_ffns_modules_4_modules_lin1_parameters_weight_
        )
        l_self_modules_ffns_modules_4_modules_lin1_parameters_bias_ = (
            L_self_modules_ffns_modules_4_modules_lin1_parameters_bias_
        )
        l_self_modules_ffns_modules_4_modules_lin2_parameters_weight_ = (
            L_self_modules_ffns_modules_4_modules_lin2_parameters_weight_
        )
        l_self_modules_ffns_modules_4_modules_lin2_parameters_bias_ = (
            L_self_modules_ffns_modules_4_modules_lin2_parameters_bias_
        )
        l_self_modules_layer_norm2_modules_4_parameters_weight_ = (
            L_self_modules_layer_norm2_modules_4_parameters_weight_
        )
        l_self_modules_layer_norm2_modules_4_parameters_bias_ = (
            L_self_modules_layer_norm2_modules_4_parameters_bias_
        )
        ne = l_input_ids_ != 2
        sum_1 = ne.sum(dim=1)
        ne = None
        lengths = sum_1.long()
        sum_1 = None
        max_1 = lengths.max()
        lengths = None
        item = max_1.item()
        max_1 = None
        le_1 = item <= 17
        item = None
        _assert_scalar_default = torch.ops.aten._assert_scalar.default(
            le_1, "Runtime assertion failed for expression u0 <= 17 on node 'le_1'"
        )
        le_1 = _assert_scalar_default = None
        alen = torch.arange(17, dtype=torch.int64, device=device(type="cuda", index=0))
        alen = None
        position_ids = l_self_buffers_position_ids_[
            (slice(None, None, None), slice(None, 17, None))
        ]
        l_self_buffers_position_ids_ = None
        position_ids_1 = position_ids.expand((1, 17))
        position_ids = None
        input_ids = l_input_ids_[(slice(None, None, None), slice(-17, None, None))]
        l_input_ids_ = None
        position_ids_2 = position_ids_1[
            (slice(None, None, None), slice(-17, None, None))
        ]
        position_ids_1 = None
        mask = l_attention_mask_[(slice(None, None, None), slice(-17, None, None))]
        attn_mask = l_attention_mask_[(slice(None, None, None), slice(-17, None, None))]
        l_attention_mask_ = None
        inputs_embeds = torch.nn.functional.embedding(
            input_ids,
            l_self_modules_embeddings_parameters_weight_,
            2,
            None,
            2.0,
            False,
            False,
        )
        input_ids = None
        embedding_1 = torch.nn.functional.embedding(
            position_ids_2,
            l_self_modules_position_embeddings_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        position_ids_2 = l_self_modules_position_embeddings_parameters_weight_ = None
        expand_as = embedding_1.expand_as(inputs_embeds)
        embedding_1 = None
        tensor = inputs_embeds + expand_as
        inputs_embeds = expand_as = None
        embedding_2 = torch.nn.functional.embedding(
            l_token_type_ids_,
            l_self_modules_embeddings_parameters_weight_,
            2,
            None,
            2.0,
            False,
            False,
        )
        l_token_type_ids_ = l_self_modules_embeddings_parameters_weight_ = None
        tensor_1 = tensor + embedding_2
        tensor = embedding_2 = None
        tensor_2 = torch.nn.functional.layer_norm(
            tensor_1,
            (32,),
            l_self_modules_layer_norm_emb_parameters_weight_,
            l_self_modules_layer_norm_emb_parameters_bias_,
            1e-12,
        )
        tensor_1 = (
            l_self_modules_layer_norm_emb_parameters_weight_
        ) = l_self_modules_layer_norm_emb_parameters_bias_ = None
        tensor_3 = torch.nn.functional.dropout(tensor_2, p=0.1, training=False)
        tensor_2 = None
        unsqueeze = mask.unsqueeze(-1)
        to = unsqueeze.to(torch.float32)
        unsqueeze = None
        tensor_3 *= to
        tensor_4 = tensor_3
        tensor_3 = to = None
        linear = torch._C._nn.linear(
            tensor_4,
            l_self_modules_attentions_modules_0_modules_q_lin_parameters_weight_,
            l_self_modules_attentions_modules_0_modules_q_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_0_modules_q_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_0_modules_q_lin_parameters_bias_
        ) = None
        view = linear.view(1, -1, 4, 8)
        linear = None
        q = view.transpose(1, 2)
        view = None
        k = torch._C._nn.linear(
            tensor_4,
            l_self_modules_attentions_modules_0_modules_k_lin_parameters_weight_,
            l_self_modules_attentions_modules_0_modules_k_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_0_modules_k_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_0_modules_k_lin_parameters_bias_
        ) = None
        v = torch._C._nn.linear(
            tensor_4,
            l_self_modules_attentions_modules_0_modules_v_lin_parameters_weight_,
            l_self_modules_attentions_modules_0_modules_v_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_0_modules_v_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_0_modules_v_lin_parameters_bias_
        ) = None
        view_1 = k.view(1, -1, 4, 8)
        k = None
        k_1 = view_1.transpose(1, 2)
        view_1 = None
        view_2 = v.view(1, -1, 4, 8)
        v = None
        v_1 = view_2.transpose(1, 2)
        view_2 = None
        q_1 = q / 2.8284271247461903
        q = None
        transpose_3 = k_1.transpose(2, 3)
        scores = torch.matmul(q_1, transpose_3)
        q_1 = transpose_3 = None
        eq = attn_mask.__eq__(0)
        view_3 = eq.view((1, 1, 1, -1))
        eq = None
        mask_1 = view_3.expand_as(scores)
        view_3 = None
        masked_fill_ = scores.masked_fill_(mask_1, -3.4028234663852886e38)
        mask_1 = masked_fill_ = None
        float_1 = scores.float()
        softmax = torch.nn.functional.softmax(float_1, dim=-1)
        float_1 = None
        weights = softmax.type_as(scores)
        softmax = scores = None
        weights_1 = torch.nn.functional.dropout(weights, p=0.1, training=False)
        weights = None
        context = torch.matmul(weights_1, v_1)
        weights_1 = None
        transpose_4 = context.transpose(1, 2)
        context = None
        contiguous = transpose_4.contiguous()
        transpose_4 = None
        context_1 = contiguous.view(1, -1, 32)
        contiguous = None
        attn = torch._C._nn.linear(
            context_1,
            l_self_modules_attentions_modules_0_modules_out_lin_parameters_weight_,
            l_self_modules_attentions_modules_0_modules_out_lin_parameters_bias_,
        )
        context_1 = (
            l_self_modules_attentions_modules_0_modules_out_lin_parameters_weight_
        ) = l_self_modules_attentions_modules_0_modules_out_lin_parameters_bias_ = None
        attn_1 = torch.nn.functional.dropout(attn, p=0.1, training=False)
        attn = None
        tensor_5 = tensor_4 + attn_1
        tensor_4 = attn_1 = None
        tensor_6 = torch.nn.functional.layer_norm(
            tensor_5,
            (32,),
            l_self_modules_layer_norm1_modules_0_parameters_weight_,
            l_self_modules_layer_norm1_modules_0_parameters_bias_,
            1e-12,
        )
        tensor_5 = (
            l_self_modules_layer_norm1_modules_0_parameters_weight_
        ) = l_self_modules_layer_norm1_modules_0_parameters_bias_ = None
        x = torch._C._nn.linear(
            tensor_6,
            l_self_modules_ffns_modules_0_modules_lin1_parameters_weight_,
            l_self_modules_ffns_modules_0_modules_lin1_parameters_bias_,
        )
        l_self_modules_ffns_modules_0_modules_lin1_parameters_weight_ = (
            l_self_modules_ffns_modules_0_modules_lin1_parameters_bias_
        ) = None
        x_1 = torch._C._nn.gelu(x)
        x = None
        x_2 = torch._C._nn.linear(
            x_1,
            l_self_modules_ffns_modules_0_modules_lin2_parameters_weight_,
            l_self_modules_ffns_modules_0_modules_lin2_parameters_bias_,
        )
        x_1 = (
            l_self_modules_ffns_modules_0_modules_lin2_parameters_weight_
        ) = l_self_modules_ffns_modules_0_modules_lin2_parameters_bias_ = None
        x_3 = torch.nn.functional.dropout(x_2, p=0.1, training=False)
        x_2 = None
        tensor_7 = tensor_6 + x_3
        tensor_6 = x_3 = None
        tensor_8 = torch.nn.functional.layer_norm(
            tensor_7,
            (32,),
            l_self_modules_layer_norm2_modules_0_parameters_weight_,
            l_self_modules_layer_norm2_modules_0_parameters_bias_,
            1e-12,
        )
        tensor_7 = (
            l_self_modules_layer_norm2_modules_0_parameters_weight_
        ) = l_self_modules_layer_norm2_modules_0_parameters_bias_ = None
        unsqueeze_1 = mask.unsqueeze(-1)
        to_1 = unsqueeze_1.to(torch.float32)
        unsqueeze_1 = None
        tensor_8 *= to_1
        tensor_9 = tensor_8
        tensor_8 = to_1 = None
        linear_6 = torch._C._nn.linear(
            tensor_9,
            l_self_modules_attentions_modules_1_modules_q_lin_parameters_weight_,
            l_self_modules_attentions_modules_1_modules_q_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_1_modules_q_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_1_modules_q_lin_parameters_bias_
        ) = None
        view_5 = linear_6.view(1, -1, 4, 8)
        linear_6 = None
        q_2 = view_5.transpose(1, 2)
        view_5 = None
        k_2 = torch._C._nn.linear(
            tensor_9,
            l_self_modules_attentions_modules_1_modules_k_lin_parameters_weight_,
            l_self_modules_attentions_modules_1_modules_k_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_1_modules_k_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_1_modules_k_lin_parameters_bias_
        ) = None
        v_2 = torch._C._nn.linear(
            tensor_9,
            l_self_modules_attentions_modules_1_modules_v_lin_parameters_weight_,
            l_self_modules_attentions_modules_1_modules_v_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_1_modules_v_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_1_modules_v_lin_parameters_bias_
        ) = None
        view_6 = k_2.view(1, -1, 4, 8)
        k_2 = None
        k_3 = view_6.transpose(1, 2)
        view_6 = None
        view_7 = v_2.view(1, -1, 4, 8)
        v_2 = None
        v_3 = view_7.transpose(1, 2)
        view_7 = None
        q_3 = q_2 / 2.8284271247461903
        q_2 = None
        transpose_8 = k_3.transpose(2, 3)
        scores_1 = torch.matmul(q_3, transpose_8)
        q_3 = transpose_8 = None
        eq_1 = attn_mask.__eq__(0)
        view_8 = eq_1.view((1, 1, 1, -1))
        eq_1 = None
        mask_2 = view_8.expand_as(scores_1)
        view_8 = None
        masked_fill__1 = scores_1.masked_fill_(mask_2, -3.4028234663852886e38)
        mask_2 = masked_fill__1 = None
        float_2 = scores_1.float()
        softmax_1 = torch.nn.functional.softmax(float_2, dim=-1)
        float_2 = None
        weights_2 = softmax_1.type_as(scores_1)
        softmax_1 = scores_1 = None
        weights_3 = torch.nn.functional.dropout(weights_2, p=0.1, training=False)
        weights_2 = None
        context_2 = torch.matmul(weights_3, v_3)
        weights_3 = None
        transpose_9 = context_2.transpose(1, 2)
        context_2 = None
        contiguous_1 = transpose_9.contiguous()
        transpose_9 = None
        context_3 = contiguous_1.view(1, -1, 32)
        contiguous_1 = None
        attn_2 = torch._C._nn.linear(
            context_3,
            l_self_modules_attentions_modules_1_modules_out_lin_parameters_weight_,
            l_self_modules_attentions_modules_1_modules_out_lin_parameters_bias_,
        )
        context_3 = (
            l_self_modules_attentions_modules_1_modules_out_lin_parameters_weight_
        ) = l_self_modules_attentions_modules_1_modules_out_lin_parameters_bias_ = None
        attn_3 = torch.nn.functional.dropout(attn_2, p=0.1, training=False)
        attn_2 = None
        tensor_10 = tensor_9 + attn_3
        tensor_9 = attn_3 = None
        tensor_11 = torch.nn.functional.layer_norm(
            tensor_10,
            (32,),
            l_self_modules_layer_norm1_modules_1_parameters_weight_,
            l_self_modules_layer_norm1_modules_1_parameters_bias_,
            1e-12,
        )
        tensor_10 = (
            l_self_modules_layer_norm1_modules_1_parameters_weight_
        ) = l_self_modules_layer_norm1_modules_1_parameters_bias_ = None
        x_4 = torch._C._nn.linear(
            tensor_11,
            l_self_modules_ffns_modules_1_modules_lin1_parameters_weight_,
            l_self_modules_ffns_modules_1_modules_lin1_parameters_bias_,
        )
        l_self_modules_ffns_modules_1_modules_lin1_parameters_weight_ = (
            l_self_modules_ffns_modules_1_modules_lin1_parameters_bias_
        ) = None
        x_5 = torch._C._nn.gelu(x_4)
        x_4 = None
        x_6 = torch._C._nn.linear(
            x_5,
            l_self_modules_ffns_modules_1_modules_lin2_parameters_weight_,
            l_self_modules_ffns_modules_1_modules_lin2_parameters_bias_,
        )
        x_5 = (
            l_self_modules_ffns_modules_1_modules_lin2_parameters_weight_
        ) = l_self_modules_ffns_modules_1_modules_lin2_parameters_bias_ = None
        x_7 = torch.nn.functional.dropout(x_6, p=0.1, training=False)
        x_6 = None
        tensor_12 = tensor_11 + x_7
        tensor_11 = x_7 = None
        tensor_13 = torch.nn.functional.layer_norm(
            tensor_12,
            (32,),
            l_self_modules_layer_norm2_modules_1_parameters_weight_,
            l_self_modules_layer_norm2_modules_1_parameters_bias_,
            1e-12,
        )
        tensor_12 = (
            l_self_modules_layer_norm2_modules_1_parameters_weight_
        ) = l_self_modules_layer_norm2_modules_1_parameters_bias_ = None
        unsqueeze_2 = mask.unsqueeze(-1)
        to_2 = unsqueeze_2.to(torch.float32)
        unsqueeze_2 = None
        tensor_13 *= to_2
        tensor_14 = tensor_13
        tensor_13 = to_2 = None
        linear_12 = torch._C._nn.linear(
            tensor_14,
            l_self_modules_attentions_modules_2_modules_q_lin_parameters_weight_,
            l_self_modules_attentions_modules_2_modules_q_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_2_modules_q_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_2_modules_q_lin_parameters_bias_
        ) = None
        view_10 = linear_12.view(1, -1, 4, 8)
        linear_12 = None
        q_4 = view_10.transpose(1, 2)
        view_10 = None
        k_4 = torch._C._nn.linear(
            tensor_14,
            l_self_modules_attentions_modules_2_modules_k_lin_parameters_weight_,
            l_self_modules_attentions_modules_2_modules_k_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_2_modules_k_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_2_modules_k_lin_parameters_bias_
        ) = None
        v_4 = torch._C._nn.linear(
            tensor_14,
            l_self_modules_attentions_modules_2_modules_v_lin_parameters_weight_,
            l_self_modules_attentions_modules_2_modules_v_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_2_modules_v_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_2_modules_v_lin_parameters_bias_
        ) = None
        view_11 = k_4.view(1, -1, 4, 8)
        k_4 = None
        k_5 = view_11.transpose(1, 2)
        view_11 = None
        view_12 = v_4.view(1, -1, 4, 8)
        v_4 = None
        v_5 = view_12.transpose(1, 2)
        view_12 = None
        q_5 = q_4 / 2.8284271247461903
        q_4 = None
        transpose_13 = k_5.transpose(2, 3)
        scores_2 = torch.matmul(q_5, transpose_13)
        q_5 = transpose_13 = None
        eq_2 = attn_mask.__eq__(0)
        view_13 = eq_2.view((1, 1, 1, -1))
        eq_2 = None
        mask_3 = view_13.expand_as(scores_2)
        view_13 = None
        masked_fill__2 = scores_2.masked_fill_(mask_3, -3.4028234663852886e38)
        mask_3 = masked_fill__2 = None
        float_3 = scores_2.float()
        softmax_2 = torch.nn.functional.softmax(float_3, dim=-1)
        float_3 = None
        weights_4 = softmax_2.type_as(scores_2)
        softmax_2 = scores_2 = None
        weights_5 = torch.nn.functional.dropout(weights_4, p=0.1, training=False)
        weights_4 = None
        context_4 = torch.matmul(weights_5, v_5)
        weights_5 = None
        transpose_14 = context_4.transpose(1, 2)
        context_4 = None
        contiguous_2 = transpose_14.contiguous()
        transpose_14 = None
        context_5 = contiguous_2.view(1, -1, 32)
        contiguous_2 = None
        attn_4 = torch._C._nn.linear(
            context_5,
            l_self_modules_attentions_modules_2_modules_out_lin_parameters_weight_,
            l_self_modules_attentions_modules_2_modules_out_lin_parameters_bias_,
        )
        context_5 = (
            l_self_modules_attentions_modules_2_modules_out_lin_parameters_weight_
        ) = l_self_modules_attentions_modules_2_modules_out_lin_parameters_bias_ = None
        attn_5 = torch.nn.functional.dropout(attn_4, p=0.1, training=False)
        attn_4 = None
        tensor_15 = tensor_14 + attn_5
        tensor_14 = attn_5 = None
        tensor_16 = torch.nn.functional.layer_norm(
            tensor_15,
            (32,),
            l_self_modules_layer_norm1_modules_2_parameters_weight_,
            l_self_modules_layer_norm1_modules_2_parameters_bias_,
            1e-12,
        )
        tensor_15 = (
            l_self_modules_layer_norm1_modules_2_parameters_weight_
        ) = l_self_modules_layer_norm1_modules_2_parameters_bias_ = None
        x_8 = torch._C._nn.linear(
            tensor_16,
            l_self_modules_ffns_modules_2_modules_lin1_parameters_weight_,
            l_self_modules_ffns_modules_2_modules_lin1_parameters_bias_,
        )
        l_self_modules_ffns_modules_2_modules_lin1_parameters_weight_ = (
            l_self_modules_ffns_modules_2_modules_lin1_parameters_bias_
        ) = None
        x_9 = torch._C._nn.gelu(x_8)
        x_8 = None
        x_10 = torch._C._nn.linear(
            x_9,
            l_self_modules_ffns_modules_2_modules_lin2_parameters_weight_,
            l_self_modules_ffns_modules_2_modules_lin2_parameters_bias_,
        )
        x_9 = (
            l_self_modules_ffns_modules_2_modules_lin2_parameters_weight_
        ) = l_self_modules_ffns_modules_2_modules_lin2_parameters_bias_ = None
        x_11 = torch.nn.functional.dropout(x_10, p=0.1, training=False)
        x_10 = None
        tensor_17 = tensor_16 + x_11
        tensor_16 = x_11 = None
        tensor_18 = torch.nn.functional.layer_norm(
            tensor_17,
            (32,),
            l_self_modules_layer_norm2_modules_2_parameters_weight_,
            l_self_modules_layer_norm2_modules_2_parameters_bias_,
            1e-12,
        )
        tensor_17 = (
            l_self_modules_layer_norm2_modules_2_parameters_weight_
        ) = l_self_modules_layer_norm2_modules_2_parameters_bias_ = None
        unsqueeze_3 = mask.unsqueeze(-1)
        to_3 = unsqueeze_3.to(torch.float32)
        unsqueeze_3 = None
        tensor_18 *= to_3
        tensor_19 = tensor_18
        tensor_18 = to_3 = None
        linear_18 = torch._C._nn.linear(
            tensor_19,
            l_self_modules_attentions_modules_3_modules_q_lin_parameters_weight_,
            l_self_modules_attentions_modules_3_modules_q_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_3_modules_q_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_3_modules_q_lin_parameters_bias_
        ) = None
        view_15 = linear_18.view(1, -1, 4, 8)
        linear_18 = None
        q_6 = view_15.transpose(1, 2)
        view_15 = None
        k_6 = torch._C._nn.linear(
            tensor_19,
            l_self_modules_attentions_modules_3_modules_k_lin_parameters_weight_,
            l_self_modules_attentions_modules_3_modules_k_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_3_modules_k_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_3_modules_k_lin_parameters_bias_
        ) = None
        v_6 = torch._C._nn.linear(
            tensor_19,
            l_self_modules_attentions_modules_3_modules_v_lin_parameters_weight_,
            l_self_modules_attentions_modules_3_modules_v_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_3_modules_v_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_3_modules_v_lin_parameters_bias_
        ) = None
        view_16 = k_6.view(1, -1, 4, 8)
        k_6 = None
        k_7 = view_16.transpose(1, 2)
        view_16 = None
        view_17 = v_6.view(1, -1, 4, 8)
        v_6 = None
        v_7 = view_17.transpose(1, 2)
        view_17 = None
        q_7 = q_6 / 2.8284271247461903
        q_6 = None
        transpose_18 = k_7.transpose(2, 3)
        scores_3 = torch.matmul(q_7, transpose_18)
        q_7 = transpose_18 = None
        eq_3 = attn_mask.__eq__(0)
        view_18 = eq_3.view((1, 1, 1, -1))
        eq_3 = None
        mask_4 = view_18.expand_as(scores_3)
        view_18 = None
        masked_fill__3 = scores_3.masked_fill_(mask_4, -3.4028234663852886e38)
        mask_4 = masked_fill__3 = None
        float_4 = scores_3.float()
        softmax_3 = torch.nn.functional.softmax(float_4, dim=-1)
        float_4 = None
        weights_6 = softmax_3.type_as(scores_3)
        softmax_3 = scores_3 = None
        weights_7 = torch.nn.functional.dropout(weights_6, p=0.1, training=False)
        weights_6 = None
        context_6 = torch.matmul(weights_7, v_7)
        weights_7 = None
        transpose_19 = context_6.transpose(1, 2)
        context_6 = None
        contiguous_3 = transpose_19.contiguous()
        transpose_19 = None
        context_7 = contiguous_3.view(1, -1, 32)
        contiguous_3 = None
        attn_6 = torch._C._nn.linear(
            context_7,
            l_self_modules_attentions_modules_3_modules_out_lin_parameters_weight_,
            l_self_modules_attentions_modules_3_modules_out_lin_parameters_bias_,
        )
        context_7 = (
            l_self_modules_attentions_modules_3_modules_out_lin_parameters_weight_
        ) = l_self_modules_attentions_modules_3_modules_out_lin_parameters_bias_ = None
        attn_7 = torch.nn.functional.dropout(attn_6, p=0.1, training=False)
        attn_6 = None
        tensor_20 = tensor_19 + attn_7
        tensor_19 = attn_7 = None
        tensor_21 = torch.nn.functional.layer_norm(
            tensor_20,
            (32,),
            l_self_modules_layer_norm1_modules_3_parameters_weight_,
            l_self_modules_layer_norm1_modules_3_parameters_bias_,
            1e-12,
        )
        tensor_20 = (
            l_self_modules_layer_norm1_modules_3_parameters_weight_
        ) = l_self_modules_layer_norm1_modules_3_parameters_bias_ = None
        x_12 = torch._C._nn.linear(
            tensor_21,
            l_self_modules_ffns_modules_3_modules_lin1_parameters_weight_,
            l_self_modules_ffns_modules_3_modules_lin1_parameters_bias_,
        )
        l_self_modules_ffns_modules_3_modules_lin1_parameters_weight_ = (
            l_self_modules_ffns_modules_3_modules_lin1_parameters_bias_
        ) = None
        x_13 = torch._C._nn.gelu(x_12)
        x_12 = None
        x_14 = torch._C._nn.linear(
            x_13,
            l_self_modules_ffns_modules_3_modules_lin2_parameters_weight_,
            l_self_modules_ffns_modules_3_modules_lin2_parameters_bias_,
        )
        x_13 = (
            l_self_modules_ffns_modules_3_modules_lin2_parameters_weight_
        ) = l_self_modules_ffns_modules_3_modules_lin2_parameters_bias_ = None
        x_15 = torch.nn.functional.dropout(x_14, p=0.1, training=False)
        x_14 = None
        tensor_22 = tensor_21 + x_15
        tensor_21 = x_15 = None
        tensor_23 = torch.nn.functional.layer_norm(
            tensor_22,
            (32,),
            l_self_modules_layer_norm2_modules_3_parameters_weight_,
            l_self_modules_layer_norm2_modules_3_parameters_bias_,
            1e-12,
        )
        tensor_22 = (
            l_self_modules_layer_norm2_modules_3_parameters_weight_
        ) = l_self_modules_layer_norm2_modules_3_parameters_bias_ = None
        unsqueeze_4 = mask.unsqueeze(-1)
        to_4 = unsqueeze_4.to(torch.float32)
        unsqueeze_4 = None
        tensor_23 *= to_4
        tensor_24 = tensor_23
        tensor_23 = to_4 = None
        linear_24 = torch._C._nn.linear(
            tensor_24,
            l_self_modules_attentions_modules_4_modules_q_lin_parameters_weight_,
            l_self_modules_attentions_modules_4_modules_q_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_4_modules_q_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_4_modules_q_lin_parameters_bias_
        ) = None
        view_20 = linear_24.view(1, -1, 4, 8)
        linear_24 = None
        q_8 = view_20.transpose(1, 2)
        view_20 = None
        k_8 = torch._C._nn.linear(
            tensor_24,
            l_self_modules_attentions_modules_4_modules_k_lin_parameters_weight_,
            l_self_modules_attentions_modules_4_modules_k_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_4_modules_k_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_4_modules_k_lin_parameters_bias_
        ) = None
        v_8 = torch._C._nn.linear(
            tensor_24,
            l_self_modules_attentions_modules_4_modules_v_lin_parameters_weight_,
            l_self_modules_attentions_modules_4_modules_v_lin_parameters_bias_,
        )
        l_self_modules_attentions_modules_4_modules_v_lin_parameters_weight_ = (
            l_self_modules_attentions_modules_4_modules_v_lin_parameters_bias_
        ) = None
        view_21 = k_8.view(1, -1, 4, 8)
        k_8 = None
        k_9 = view_21.transpose(1, 2)
        view_21 = None
        view_22 = v_8.view(1, -1, 4, 8)
        v_8 = None
        v_9 = view_22.transpose(1, 2)
        view_22 = None
        q_9 = q_8 / 2.8284271247461903
        q_8 = None
        transpose_23 = k_9.transpose(2, 3)
        scores_4 = torch.matmul(q_9, transpose_23)
        q_9 = transpose_23 = None
        eq_4 = attn_mask.__eq__(0)
        attn_mask = None
        view_23 = eq_4.view((1, 1, 1, -1))
        eq_4 = None
        mask_5 = view_23.expand_as(scores_4)
        view_23 = None
        masked_fill__4 = scores_4.masked_fill_(mask_5, -3.4028234663852886e38)
        mask_5 = masked_fill__4 = None
        float_5 = scores_4.float()
        softmax_4 = torch.nn.functional.softmax(float_5, dim=-1)
        float_5 = None
        weights_8 = softmax_4.type_as(scores_4)
        softmax_4 = scores_4 = None
        weights_9 = torch.nn.functional.dropout(weights_8, p=0.1, training=False)
        weights_8 = None
        context_8 = torch.matmul(weights_9, v_9)
        weights_9 = None
        transpose_24 = context_8.transpose(1, 2)
        context_8 = None
        contiguous_4 = transpose_24.contiguous()
        transpose_24 = None
        context_9 = contiguous_4.view(1, -1, 32)
        contiguous_4 = None
        attn_8 = torch._C._nn.linear(
            context_9,
            l_self_modules_attentions_modules_4_modules_out_lin_parameters_weight_,
            l_self_modules_attentions_modules_4_modules_out_lin_parameters_bias_,
        )
        context_9 = (
            l_self_modules_attentions_modules_4_modules_out_lin_parameters_weight_
        ) = l_self_modules_attentions_modules_4_modules_out_lin_parameters_bias_ = None
        attn_9 = torch.nn.functional.dropout(attn_8, p=0.1, training=False)
        attn_8 = None
        tensor_25 = tensor_24 + attn_9
        tensor_24 = attn_9 = None
        tensor_26 = torch.nn.functional.layer_norm(
            tensor_25,
            (32,),
            l_self_modules_layer_norm1_modules_4_parameters_weight_,
            l_self_modules_layer_norm1_modules_4_parameters_bias_,
            1e-12,
        )
        tensor_25 = (
            l_self_modules_layer_norm1_modules_4_parameters_weight_
        ) = l_self_modules_layer_norm1_modules_4_parameters_bias_ = None
        x_16 = torch._C._nn.linear(
            tensor_26,
            l_self_modules_ffns_modules_4_modules_lin1_parameters_weight_,
            l_self_modules_ffns_modules_4_modules_lin1_parameters_bias_,
        )
        l_self_modules_ffns_modules_4_modules_lin1_parameters_weight_ = (
            l_self_modules_ffns_modules_4_modules_lin1_parameters_bias_
        ) = None
        x_17 = torch._C._nn.gelu(x_16)
        x_16 = None
        x_18 = torch._C._nn.linear(
            x_17,
            l_self_modules_ffns_modules_4_modules_lin2_parameters_weight_,
            l_self_modules_ffns_modules_4_modules_lin2_parameters_bias_,
        )
        x_17 = (
            l_self_modules_ffns_modules_4_modules_lin2_parameters_weight_
        ) = l_self_modules_ffns_modules_4_modules_lin2_parameters_bias_ = None
        x_19 = torch.nn.functional.dropout(x_18, p=0.1, training=False)
        x_18 = None
        tensor_27 = tensor_26 + x_19
        tensor_26 = x_19 = None
        tensor_28 = torch.nn.functional.layer_norm(
            tensor_27,
            (32,),
            l_self_modules_layer_norm2_modules_4_parameters_weight_,
            l_self_modules_layer_norm2_modules_4_parameters_bias_,
            1e-12,
        )
        tensor_27 = (
            l_self_modules_layer_norm2_modules_4_parameters_weight_
        ) = l_self_modules_layer_norm2_modules_4_parameters_bias_ = None
        unsqueeze_5 = mask.unsqueeze(-1)
        mask = None
        to_5 = unsqueeze_5.to(torch.float32)
        unsqueeze_5 = None
        tensor_28 *= to_5
        tensor_29 = tensor_28
        tensor_28 = to_5 = None
        return (v_1, k_1, v_3, k_3, v_5, k_5, v_7, k_7, v_9, k_9, tensor_29)
