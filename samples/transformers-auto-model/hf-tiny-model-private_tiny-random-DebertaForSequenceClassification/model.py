import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_input_ids_: torch.Tensor,
        L_attention_mask_: torch.Tensor,
        L_token_type_ids_: torch.Tensor,
        L_self_modules_embeddings_buffers_position_ids_: torch.Tensor,
        L_self_modules_embeddings_modules_word_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_position_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_token_type_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_in_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_in_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_in_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_in_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_in_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_input_ids_ = L_input_ids_
        l_attention_mask_ = L_attention_mask_
        l_token_type_ids_ = L_token_type_ids_
        l_self_modules_embeddings_buffers_position_ids_ = (
            L_self_modules_embeddings_buffers_position_ids_
        )
        l_self_modules_embeddings_modules_word_embeddings_parameters_weight_ = (
            L_self_modules_embeddings_modules_word_embeddings_parameters_weight_
        )
        l_self_modules_embeddings_modules_position_embeddings_parameters_weight_ = (
            L_self_modules_embeddings_modules_position_embeddings_parameters_weight_
        )
        l_self_modules_embeddings_modules_token_type_embeddings_parameters_weight_ = (
            L_self_modules_embeddings_modules_token_type_embeddings_parameters_weight_
        )
        l_self_modules_embeddings_modules_layer_norm_parameters_weight_ = (
            L_self_modules_embeddings_modules_LayerNorm_parameters_weight_
        )
        l_self_modules_embeddings_modules_layer_norm_parameters_bias_ = (
            L_self_modules_embeddings_modules_LayerNorm_parameters_bias_
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_in_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_in_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_parameters_q_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_parameters_q_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_parameters_v_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_parameters_v_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_in_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_in_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_parameters_q_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_parameters_q_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_parameters_v_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_parameters_v_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_in_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_in_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_parameters_q_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_parameters_q_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_parameters_v_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_parameters_v_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_in_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_in_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_parameters_q_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_parameters_q_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_parameters_v_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_parameters_v_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_in_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_in_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_parameters_q_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_parameters_q_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_parameters_v_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_parameters_v_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_parameters_bias_
        position_ids = l_self_modules_embeddings_buffers_position_ids_[
            (slice(None, None, None), slice(None, 22, None))
        ]
        l_self_modules_embeddings_buffers_position_ids_ = None
        inputs_embeds = torch.nn.functional.embedding(
            l_input_ids_,
            l_self_modules_embeddings_modules_word_embeddings_parameters_weight_,
            0,
            None,
            2.0,
            False,
            False,
        )
        l_input_ids_ = (
            l_self_modules_embeddings_modules_word_embeddings_parameters_weight_
        ) = None
        long = position_ids.long()
        position_ids = None
        position_embeddings = torch.nn.functional.embedding(
            long,
            l_self_modules_embeddings_modules_position_embeddings_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        long = (
            l_self_modules_embeddings_modules_position_embeddings_parameters_weight_
        ) = None
        embeddings = inputs_embeds + position_embeddings
        inputs_embeds = position_embeddings = None
        token_type_embeddings = torch.nn.functional.embedding(
            l_token_type_ids_,
            l_self_modules_embeddings_modules_token_type_embeddings_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        l_token_type_ids_ = (
            l_self_modules_embeddings_modules_token_type_embeddings_parameters_weight_
        ) = None
        embeddings_1 = embeddings + token_type_embeddings
        embeddings = token_type_embeddings = None
        hidden_states = embeddings_1.float()
        embeddings_1 = None
        mean = hidden_states.mean(-1, keepdim=True)
        sub = hidden_states - mean
        pow_1 = sub.pow(2)
        sub = None
        variance = pow_1.mean(-1, keepdim=True)
        pow_1 = None
        sub_1 = hidden_states - mean
        hidden_states = mean = None
        add_2 = variance + 1e-07
        variance = None
        sqrt = torch.sqrt(add_2)
        add_2 = None
        hidden_states_1 = sub_1 / sqrt
        sub_1 = sqrt = None
        hidden_states_2 = hidden_states_1.to(torch.float32)
        hidden_states_1 = None
        mul = (
            l_self_modules_embeddings_modules_layer_norm_parameters_weight_
            * hidden_states_2
        )
        l_self_modules_embeddings_modules_layer_norm_parameters_weight_ = (
            hidden_states_2
        ) = None
        y = mul + l_self_modules_embeddings_modules_layer_norm_parameters_bias_
        mul = l_self_modules_embeddings_modules_layer_norm_parameters_bias_ = None
        mask = l_attention_mask_.unsqueeze(2)
        mask_1 = mask.to(torch.float32)
        mask = None
        embeddings_2 = y * mask_1
        y = mask_1 = None
        embeddings_3 = torch.nn.functional.dropout(embeddings_2, 0.1, False, False)
        embeddings_2 = None
        unsqueeze_1 = l_attention_mask_.unsqueeze(1)
        l_attention_mask_ = None
        extended_attention_mask = unsqueeze_1.unsqueeze(2)
        unsqueeze_1 = None
        squeeze = extended_attention_mask.squeeze(-2)
        unsqueeze_3 = squeeze.unsqueeze(-1)
        squeeze = None
        attention_mask = extended_attention_mask * unsqueeze_3
        extended_attention_mask = unsqueeze_3 = None
        qp = torch._C._nn.linear(
            embeddings_3,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_in_proj_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_in_proj_parameters_weight_ = (
            None
        )
        x = qp.view((1, 22, 4, -1))
        qp = None
        permute = x.permute(0, 2, 1, 3)
        x = None
        chunk = permute.chunk(3, dim=-1)
        permute = None
        query_layer = chunk[0]
        key_layer = chunk[1]
        value_layer = chunk[2]
        chunk = None
        getitem_4 = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_parameters_q_bias_[
            (None, None, slice(None, None, None))
        ]
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_parameters_q_bias_ = (
            None
        )
        x_1 = getitem_4.view((1, 1, 4, -1))
        getitem_4 = None
        permute_1 = x_1.permute(0, 2, 1, 3)
        x_1 = None
        query_layer_1 = query_layer + permute_1
        query_layer = permute_1 = None
        getitem_5 = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_parameters_v_bias_[
            (None, None, slice(None, None, None))
        ]
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_parameters_v_bias_ = (
            None
        )
        x_2 = getitem_5.view((1, 1, 4, -1))
        getitem_5 = None
        permute_2 = x_2.permute(0, 2, 1, 3)
        x_2 = None
        value_layer_1 = value_layer + permute_2
        value_layer = permute_2 = None
        tensor = torch.tensor(8, dtype=torch.float32)
        mul_3 = tensor * 2
        tensor = None
        scale = torch.sqrt(mul_3)
        mul_3 = None
        to_2 = scale.to(dtype=torch.float32)
        scale = None
        query_layer_2 = query_layer_1 / to_2
        query_layer_1 = to_2 = None
        transpose = key_layer.transpose(-1, -2)
        key_layer = None
        attention_scores = torch.matmul(query_layer_2, transpose)
        query_layer_2 = transpose = None
        attention_scores_1 = attention_scores + 0
        attention_scores = None
        attention_mask_1 = attention_mask.bool()
        invert = ~attention_mask_1
        attention_mask_1 = None
        attention_scores_2 = attention_scores_1.masked_fill(
            invert, -3.4028234663852886e38
        )
        attention_scores_1 = invert = None
        attention_probs = torch.nn.functional.softmax(attention_scores_2, dim=-1)
        attention_scores_2 = None
        attention_probs_1 = torch.nn.functional.dropout(
            attention_probs, 0.1, False, False
        )
        attention_probs = None
        context_layer = torch.matmul(attention_probs_1, value_layer_1)
        attention_probs_1 = value_layer_1 = None
        permute_3 = context_layer.permute(0, 2, 1, 3)
        context_layer = None
        context_layer_1 = permute_3.contiguous()
        permute_3 = None
        context_layer_2 = context_layer_1.view((1, 22, -1))
        context_layer_1 = None
        hidden_states_3 = torch._C._nn.linear(
            context_layer_2,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_2 = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_4 = torch.nn.functional.dropout(
            hidden_states_3, 0.1, False, False
        )
        hidden_states_3 = None
        add_7 = hidden_states_4 + embeddings_3
        hidden_states_4 = embeddings_3 = None
        hidden_states_5 = add_7.float()
        add_7 = None
        mean_3 = hidden_states_5.mean(-1, keepdim=True)
        sub_2 = hidden_states_5 - mean_3
        pow_2 = sub_2.pow(2)
        sub_2 = None
        variance_1 = pow_2.mean(-1, keepdim=True)
        pow_2 = None
        sub_3 = hidden_states_5 - mean_3
        hidden_states_5 = mean_3 = None
        add_8 = variance_1 + 1e-07
        variance_1 = None
        sqrt_2 = torch.sqrt(add_8)
        add_8 = None
        hidden_states_6 = sub_3 / sqrt_2
        sub_3 = sqrt_2 = None
        hidden_states_7 = hidden_states_6.to(torch.float32)
        hidden_states_6 = None
        mul_4 = (
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_layer_norm_parameters_weight_
            * hidden_states_7
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = (
            hidden_states_7
        ) = None
        y_1 = (
            mul_4
            + l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_layer_norm_parameters_bias_
        )
        mul_4 = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_8 = torch._C._nn.linear(
            y_1,
            l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_9 = torch._C._nn.gelu(hidden_states_8)
        hidden_states_8 = None
        hidden_states_10 = torch._C._nn.linear(
            hidden_states_9,
            l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_9 = l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_11 = torch.nn.functional.dropout(
            hidden_states_10, 0.1, False, False
        )
        hidden_states_10 = None
        add_10 = hidden_states_11 + y_1
        hidden_states_11 = y_1 = None
        hidden_states_12 = add_10.float()
        add_10 = None
        mean_6 = hidden_states_12.mean(-1, keepdim=True)
        sub_4 = hidden_states_12 - mean_6
        pow_3 = sub_4.pow(2)
        sub_4 = None
        variance_2 = pow_3.mean(-1, keepdim=True)
        pow_3 = None
        sub_5 = hidden_states_12 - mean_6
        hidden_states_12 = mean_6 = None
        add_11 = variance_2 + 1e-07
        variance_2 = None
        sqrt_3 = torch.sqrt(add_11)
        add_11 = None
        hidden_states_13 = sub_5 / sqrt_3
        sub_5 = sqrt_3 = None
        hidden_states_14 = hidden_states_13.to(torch.float32)
        hidden_states_13 = None
        mul_5 = (
            l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_weight_
            * hidden_states_14
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_weight_ = (
            hidden_states_14
        ) = None
        y_2 = (
            mul_5
            + l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_bias_
        )
        mul_5 = l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_bias_ = (None)
        qp_1 = torch._C._nn.linear(
            y_2,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_in_proj_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_in_proj_parameters_weight_ = (
            None
        )
        x_3 = qp_1.view((1, 22, 4, -1))
        qp_1 = None
        permute_4 = x_3.permute(0, 2, 1, 3)
        x_3 = None
        chunk_1 = permute_4.chunk(3, dim=-1)
        permute_4 = None
        query_layer_3 = chunk_1[0]
        key_layer_1 = chunk_1[1]
        value_layer_2 = chunk_1[2]
        chunk_1 = None
        getitem_9 = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_parameters_q_bias_[
            (None, None, slice(None, None, None))
        ]
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_parameters_q_bias_ = (
            None
        )
        x_4 = getitem_9.view((1, 1, 4, -1))
        getitem_9 = None
        permute_5 = x_4.permute(0, 2, 1, 3)
        x_4 = None
        query_layer_4 = query_layer_3 + permute_5
        query_layer_3 = permute_5 = None
        getitem_10 = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_parameters_v_bias_[
            (None, None, slice(None, None, None))
        ]
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_parameters_v_bias_ = (
            None
        )
        x_5 = getitem_10.view((1, 1, 4, -1))
        getitem_10 = None
        permute_6 = x_5.permute(0, 2, 1, 3)
        x_5 = None
        value_layer_3 = value_layer_2 + permute_6
        value_layer_2 = permute_6 = None
        tensor_1 = torch.tensor(8, dtype=torch.float32)
        mul_6 = tensor_1 * 2
        tensor_1 = None
        scale_1 = torch.sqrt(mul_6)
        mul_6 = None
        to_5 = scale_1.to(dtype=torch.float32)
        scale_1 = None
        query_layer_5 = query_layer_4 / to_5
        query_layer_4 = to_5 = None
        transpose_1 = key_layer_1.transpose(-1, -2)
        key_layer_1 = None
        attention_scores_3 = torch.matmul(query_layer_5, transpose_1)
        query_layer_5 = transpose_1 = None
        attention_scores_4 = attention_scores_3 + 0
        attention_scores_3 = None
        attention_mask_2 = attention_mask.bool()
        invert_1 = ~attention_mask_2
        attention_mask_2 = None
        attention_scores_5 = attention_scores_4.masked_fill(
            invert_1, -3.4028234663852886e38
        )
        attention_scores_4 = invert_1 = None
        attention_probs_2 = torch.nn.functional.softmax(attention_scores_5, dim=-1)
        attention_scores_5 = None
        attention_probs_3 = torch.nn.functional.dropout(
            attention_probs_2, 0.1, False, False
        )
        attention_probs_2 = None
        context_layer_3 = torch.matmul(attention_probs_3, value_layer_3)
        attention_probs_3 = value_layer_3 = None
        permute_7 = context_layer_3.permute(0, 2, 1, 3)
        context_layer_3 = None
        context_layer_4 = permute_7.contiguous()
        permute_7 = None
        context_layer_5 = context_layer_4.view((1, 22, -1))
        context_layer_4 = None
        hidden_states_15 = torch._C._nn.linear(
            context_layer_5,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_5 = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_16 = torch.nn.functional.dropout(
            hidden_states_15, 0.1, False, False
        )
        hidden_states_15 = None
        add_16 = hidden_states_16 + y_2
        hidden_states_16 = y_2 = None
        hidden_states_17 = add_16.float()
        add_16 = None
        mean_9 = hidden_states_17.mean(-1, keepdim=True)
        sub_6 = hidden_states_17 - mean_9
        pow_4 = sub_6.pow(2)
        sub_6 = None
        variance_3 = pow_4.mean(-1, keepdim=True)
        pow_4 = None
        sub_7 = hidden_states_17 - mean_9
        hidden_states_17 = mean_9 = None
        add_17 = variance_3 + 1e-07
        variance_3 = None
        sqrt_5 = torch.sqrt(add_17)
        add_17 = None
        hidden_states_18 = sub_7 / sqrt_5
        sub_7 = sqrt_5 = None
        hidden_states_19 = hidden_states_18.to(torch.float32)
        hidden_states_18 = None
        mul_7 = (
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_layer_norm_parameters_weight_
            * hidden_states_19
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = (
            hidden_states_19
        ) = None
        y_3 = (
            mul_7
            + l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_layer_norm_parameters_bias_
        )
        mul_7 = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_20 = torch._C._nn.linear(
            y_3,
            l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_21 = torch._C._nn.gelu(hidden_states_20)
        hidden_states_20 = None
        hidden_states_22 = torch._C._nn.linear(
            hidden_states_21,
            l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_21 = l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_23 = torch.nn.functional.dropout(
            hidden_states_22, 0.1, False, False
        )
        hidden_states_22 = None
        add_19 = hidden_states_23 + y_3
        hidden_states_23 = y_3 = None
        hidden_states_24 = add_19.float()
        add_19 = None
        mean_12 = hidden_states_24.mean(-1, keepdim=True)
        sub_8 = hidden_states_24 - mean_12
        pow_5 = sub_8.pow(2)
        sub_8 = None
        variance_4 = pow_5.mean(-1, keepdim=True)
        pow_5 = None
        sub_9 = hidden_states_24 - mean_12
        hidden_states_24 = mean_12 = None
        add_20 = variance_4 + 1e-07
        variance_4 = None
        sqrt_6 = torch.sqrt(add_20)
        add_20 = None
        hidden_states_25 = sub_9 / sqrt_6
        sub_9 = sqrt_6 = None
        hidden_states_26 = hidden_states_25.to(torch.float32)
        hidden_states_25 = None
        mul_8 = (
            l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_weight_
            * hidden_states_26
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_weight_ = (
            hidden_states_26
        ) = None
        y_4 = (
            mul_8
            + l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_bias_
        )
        mul_8 = l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_bias_ = (None)
        qp_2 = torch._C._nn.linear(
            y_4,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_in_proj_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_in_proj_parameters_weight_ = (
            None
        )
        x_6 = qp_2.view((1, 22, 4, -1))
        qp_2 = None
        permute_8 = x_6.permute(0, 2, 1, 3)
        x_6 = None
        chunk_2 = permute_8.chunk(3, dim=-1)
        permute_8 = None
        query_layer_6 = chunk_2[0]
        key_layer_2 = chunk_2[1]
        value_layer_4 = chunk_2[2]
        chunk_2 = None
        getitem_14 = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_parameters_q_bias_[
            (None, None, slice(None, None, None))
        ]
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_parameters_q_bias_ = (
            None
        )
        x_7 = getitem_14.view((1, 1, 4, -1))
        getitem_14 = None
        permute_9 = x_7.permute(0, 2, 1, 3)
        x_7 = None
        query_layer_7 = query_layer_6 + permute_9
        query_layer_6 = permute_9 = None
        getitem_15 = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_parameters_v_bias_[
            (None, None, slice(None, None, None))
        ]
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_parameters_v_bias_ = (
            None
        )
        x_8 = getitem_15.view((1, 1, 4, -1))
        getitem_15 = None
        permute_10 = x_8.permute(0, 2, 1, 3)
        x_8 = None
        value_layer_5 = value_layer_4 + permute_10
        value_layer_4 = permute_10 = None
        tensor_2 = torch.tensor(8, dtype=torch.float32)
        mul_9 = tensor_2 * 2
        tensor_2 = None
        scale_2 = torch.sqrt(mul_9)
        mul_9 = None
        to_8 = scale_2.to(dtype=torch.float32)
        scale_2 = None
        query_layer_8 = query_layer_7 / to_8
        query_layer_7 = to_8 = None
        transpose_2 = key_layer_2.transpose(-1, -2)
        key_layer_2 = None
        attention_scores_6 = torch.matmul(query_layer_8, transpose_2)
        query_layer_8 = transpose_2 = None
        attention_scores_7 = attention_scores_6 + 0
        attention_scores_6 = None
        attention_mask_3 = attention_mask.bool()
        invert_2 = ~attention_mask_3
        attention_mask_3 = None
        attention_scores_8 = attention_scores_7.masked_fill(
            invert_2, -3.4028234663852886e38
        )
        attention_scores_7 = invert_2 = None
        attention_probs_4 = torch.nn.functional.softmax(attention_scores_8, dim=-1)
        attention_scores_8 = None
        attention_probs_5 = torch.nn.functional.dropout(
            attention_probs_4, 0.1, False, False
        )
        attention_probs_4 = None
        context_layer_6 = torch.matmul(attention_probs_5, value_layer_5)
        attention_probs_5 = value_layer_5 = None
        permute_11 = context_layer_6.permute(0, 2, 1, 3)
        context_layer_6 = None
        context_layer_7 = permute_11.contiguous()
        permute_11 = None
        context_layer_8 = context_layer_7.view((1, 22, -1))
        context_layer_7 = None
        hidden_states_27 = torch._C._nn.linear(
            context_layer_8,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_8 = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_28 = torch.nn.functional.dropout(
            hidden_states_27, 0.1, False, False
        )
        hidden_states_27 = None
        add_25 = hidden_states_28 + y_4
        hidden_states_28 = y_4 = None
        hidden_states_29 = add_25.float()
        add_25 = None
        mean_15 = hidden_states_29.mean(-1, keepdim=True)
        sub_10 = hidden_states_29 - mean_15
        pow_6 = sub_10.pow(2)
        sub_10 = None
        variance_5 = pow_6.mean(-1, keepdim=True)
        pow_6 = None
        sub_11 = hidden_states_29 - mean_15
        hidden_states_29 = mean_15 = None
        add_26 = variance_5 + 1e-07
        variance_5 = None
        sqrt_8 = torch.sqrt(add_26)
        add_26 = None
        hidden_states_30 = sub_11 / sqrt_8
        sub_11 = sqrt_8 = None
        hidden_states_31 = hidden_states_30.to(torch.float32)
        hidden_states_30 = None
        mul_10 = (
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_layer_norm_parameters_weight_
            * hidden_states_31
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = (
            hidden_states_31
        ) = None
        y_5 = (
            mul_10
            + l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_layer_norm_parameters_bias_
        )
        mul_10 = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_32 = torch._C._nn.linear(
            y_5,
            l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_33 = torch._C._nn.gelu(hidden_states_32)
        hidden_states_32 = None
        hidden_states_34 = torch._C._nn.linear(
            hidden_states_33,
            l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_33 = l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_35 = torch.nn.functional.dropout(
            hidden_states_34, 0.1, False, False
        )
        hidden_states_34 = None
        add_28 = hidden_states_35 + y_5
        hidden_states_35 = y_5 = None
        hidden_states_36 = add_28.float()
        add_28 = None
        mean_18 = hidden_states_36.mean(-1, keepdim=True)
        sub_12 = hidden_states_36 - mean_18
        pow_7 = sub_12.pow(2)
        sub_12 = None
        variance_6 = pow_7.mean(-1, keepdim=True)
        pow_7 = None
        sub_13 = hidden_states_36 - mean_18
        hidden_states_36 = mean_18 = None
        add_29 = variance_6 + 1e-07
        variance_6 = None
        sqrt_9 = torch.sqrt(add_29)
        add_29 = None
        hidden_states_37 = sub_13 / sqrt_9
        sub_13 = sqrt_9 = None
        hidden_states_38 = hidden_states_37.to(torch.float32)
        hidden_states_37 = None
        mul_11 = (
            l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_weight_
            * hidden_states_38
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_weight_ = (
            hidden_states_38
        ) = None
        y_6 = (
            mul_11
            + l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_bias_
        )
        mul_11 = l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_bias_ = (None)
        qp_3 = torch._C._nn.linear(
            y_6,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_in_proj_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_in_proj_parameters_weight_ = (
            None
        )
        x_9 = qp_3.view((1, 22, 4, -1))
        qp_3 = None
        permute_12 = x_9.permute(0, 2, 1, 3)
        x_9 = None
        chunk_3 = permute_12.chunk(3, dim=-1)
        permute_12 = None
        query_layer_9 = chunk_3[0]
        key_layer_3 = chunk_3[1]
        value_layer_6 = chunk_3[2]
        chunk_3 = None
        getitem_19 = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_parameters_q_bias_[
            (None, None, slice(None, None, None))
        ]
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_parameters_q_bias_ = (
            None
        )
        x_10 = getitem_19.view((1, 1, 4, -1))
        getitem_19 = None
        permute_13 = x_10.permute(0, 2, 1, 3)
        x_10 = None
        query_layer_10 = query_layer_9 + permute_13
        query_layer_9 = permute_13 = None
        getitem_20 = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_parameters_v_bias_[
            (None, None, slice(None, None, None))
        ]
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_parameters_v_bias_ = (
            None
        )
        x_11 = getitem_20.view((1, 1, 4, -1))
        getitem_20 = None
        permute_14 = x_11.permute(0, 2, 1, 3)
        x_11 = None
        value_layer_7 = value_layer_6 + permute_14
        value_layer_6 = permute_14 = None
        tensor_3 = torch.tensor(8, dtype=torch.float32)
        mul_12 = tensor_3 * 2
        tensor_3 = None
        scale_3 = torch.sqrt(mul_12)
        mul_12 = None
        to_11 = scale_3.to(dtype=torch.float32)
        scale_3 = None
        query_layer_11 = query_layer_10 / to_11
        query_layer_10 = to_11 = None
        transpose_3 = key_layer_3.transpose(-1, -2)
        key_layer_3 = None
        attention_scores_9 = torch.matmul(query_layer_11, transpose_3)
        query_layer_11 = transpose_3 = None
        attention_scores_10 = attention_scores_9 + 0
        attention_scores_9 = None
        attention_mask_4 = attention_mask.bool()
        invert_3 = ~attention_mask_4
        attention_mask_4 = None
        attention_scores_11 = attention_scores_10.masked_fill(
            invert_3, -3.4028234663852886e38
        )
        attention_scores_10 = invert_3 = None
        attention_probs_6 = torch.nn.functional.softmax(attention_scores_11, dim=-1)
        attention_scores_11 = None
        attention_probs_7 = torch.nn.functional.dropout(
            attention_probs_6, 0.1, False, False
        )
        attention_probs_6 = None
        context_layer_9 = torch.matmul(attention_probs_7, value_layer_7)
        attention_probs_7 = value_layer_7 = None
        permute_15 = context_layer_9.permute(0, 2, 1, 3)
        context_layer_9 = None
        context_layer_10 = permute_15.contiguous()
        permute_15 = None
        context_layer_11 = context_layer_10.view((1, 22, -1))
        context_layer_10 = None
        hidden_states_39 = torch._C._nn.linear(
            context_layer_11,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_11 = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_40 = torch.nn.functional.dropout(
            hidden_states_39, 0.1, False, False
        )
        hidden_states_39 = None
        add_34 = hidden_states_40 + y_6
        hidden_states_40 = y_6 = None
        hidden_states_41 = add_34.float()
        add_34 = None
        mean_21 = hidden_states_41.mean(-1, keepdim=True)
        sub_14 = hidden_states_41 - mean_21
        pow_8 = sub_14.pow(2)
        sub_14 = None
        variance_7 = pow_8.mean(-1, keepdim=True)
        pow_8 = None
        sub_15 = hidden_states_41 - mean_21
        hidden_states_41 = mean_21 = None
        add_35 = variance_7 + 1e-07
        variance_7 = None
        sqrt_11 = torch.sqrt(add_35)
        add_35 = None
        hidden_states_42 = sub_15 / sqrt_11
        sub_15 = sqrt_11 = None
        hidden_states_43 = hidden_states_42.to(torch.float32)
        hidden_states_42 = None
        mul_13 = (
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_layer_norm_parameters_weight_
            * hidden_states_43
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = (
            hidden_states_43
        ) = None
        y_7 = (
            mul_13
            + l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_layer_norm_parameters_bias_
        )
        mul_13 = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_44 = torch._C._nn.linear(
            y_7,
            l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_45 = torch._C._nn.gelu(hidden_states_44)
        hidden_states_44 = None
        hidden_states_46 = torch._C._nn.linear(
            hidden_states_45,
            l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_45 = l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_47 = torch.nn.functional.dropout(
            hidden_states_46, 0.1, False, False
        )
        hidden_states_46 = None
        add_37 = hidden_states_47 + y_7
        hidden_states_47 = y_7 = None
        hidden_states_48 = add_37.float()
        add_37 = None
        mean_24 = hidden_states_48.mean(-1, keepdim=True)
        sub_16 = hidden_states_48 - mean_24
        pow_9 = sub_16.pow(2)
        sub_16 = None
        variance_8 = pow_9.mean(-1, keepdim=True)
        pow_9 = None
        sub_17 = hidden_states_48 - mean_24
        hidden_states_48 = mean_24 = None
        add_38 = variance_8 + 1e-07
        variance_8 = None
        sqrt_12 = torch.sqrt(add_38)
        add_38 = None
        hidden_states_49 = sub_17 / sqrt_12
        sub_17 = sqrt_12 = None
        hidden_states_50 = hidden_states_49.to(torch.float32)
        hidden_states_49 = None
        mul_14 = (
            l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_weight_
            * hidden_states_50
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_weight_ = (
            hidden_states_50
        ) = None
        y_8 = (
            mul_14
            + l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_bias_
        )
        mul_14 = l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_bias_ = (None)
        qp_4 = torch._C._nn.linear(
            y_8,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_in_proj_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_in_proj_parameters_weight_ = (
            None
        )
        x_12 = qp_4.view((1, 22, 4, -1))
        qp_4 = None
        permute_16 = x_12.permute(0, 2, 1, 3)
        x_12 = None
        chunk_4 = permute_16.chunk(3, dim=-1)
        permute_16 = None
        query_layer_12 = chunk_4[0]
        key_layer_4 = chunk_4[1]
        value_layer_8 = chunk_4[2]
        chunk_4 = None
        getitem_24 = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_parameters_q_bias_[
            (None, None, slice(None, None, None))
        ]
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_parameters_q_bias_ = (
            None
        )
        x_13 = getitem_24.view((1, 1, 4, -1))
        getitem_24 = None
        permute_17 = x_13.permute(0, 2, 1, 3)
        x_13 = None
        query_layer_13 = query_layer_12 + permute_17
        query_layer_12 = permute_17 = None
        getitem_25 = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_parameters_v_bias_[
            (None, None, slice(None, None, None))
        ]
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_parameters_v_bias_ = (
            None
        )
        x_14 = getitem_25.view((1, 1, 4, -1))
        getitem_25 = None
        permute_18 = x_14.permute(0, 2, 1, 3)
        x_14 = None
        value_layer_9 = value_layer_8 + permute_18
        value_layer_8 = permute_18 = None
        tensor_4 = torch.tensor(8, dtype=torch.float32)
        mul_15 = tensor_4 * 2
        tensor_4 = None
        scale_4 = torch.sqrt(mul_15)
        mul_15 = None
        to_14 = scale_4.to(dtype=torch.float32)
        scale_4 = None
        query_layer_14 = query_layer_13 / to_14
        query_layer_13 = to_14 = None
        transpose_4 = key_layer_4.transpose(-1, -2)
        key_layer_4 = None
        attention_scores_12 = torch.matmul(query_layer_14, transpose_4)
        query_layer_14 = transpose_4 = None
        attention_scores_13 = attention_scores_12 + 0
        attention_scores_12 = None
        attention_mask_5 = attention_mask.bool()
        attention_mask = None
        invert_4 = ~attention_mask_5
        attention_mask_5 = None
        attention_scores_14 = attention_scores_13.masked_fill(
            invert_4, -3.4028234663852886e38
        )
        attention_scores_13 = invert_4 = None
        attention_probs_8 = torch.nn.functional.softmax(attention_scores_14, dim=-1)
        attention_scores_14 = None
        attention_probs_9 = torch.nn.functional.dropout(
            attention_probs_8, 0.1, False, False
        )
        attention_probs_8 = None
        context_layer_12 = torch.matmul(attention_probs_9, value_layer_9)
        attention_probs_9 = value_layer_9 = None
        permute_19 = context_layer_12.permute(0, 2, 1, 3)
        context_layer_12 = None
        context_layer_13 = permute_19.contiguous()
        permute_19 = None
        context_layer_14 = context_layer_13.view((1, 22, -1))
        context_layer_13 = None
        hidden_states_51 = torch._C._nn.linear(
            context_layer_14,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_14 = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_52 = torch.nn.functional.dropout(
            hidden_states_51, 0.1, False, False
        )
        hidden_states_51 = None
        add_43 = hidden_states_52 + y_8
        hidden_states_52 = y_8 = None
        hidden_states_53 = add_43.float()
        add_43 = None
        mean_27 = hidden_states_53.mean(-1, keepdim=True)
        sub_18 = hidden_states_53 - mean_27
        pow_10 = sub_18.pow(2)
        sub_18 = None
        variance_9 = pow_10.mean(-1, keepdim=True)
        pow_10 = None
        sub_19 = hidden_states_53 - mean_27
        hidden_states_53 = mean_27 = None
        add_44 = variance_9 + 1e-07
        variance_9 = None
        sqrt_14 = torch.sqrt(add_44)
        add_44 = None
        hidden_states_54 = sub_19 / sqrt_14
        sub_19 = sqrt_14 = None
        hidden_states_55 = hidden_states_54.to(torch.float32)
        hidden_states_54 = None
        mul_16 = (
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_layer_norm_parameters_weight_
            * hidden_states_55
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = (
            hidden_states_55
        ) = None
        y_9 = (
            mul_16
            + l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_layer_norm_parameters_bias_
        )
        mul_16 = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_56 = torch._C._nn.linear(
            y_9,
            l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_57 = torch._C._nn.gelu(hidden_states_56)
        hidden_states_56 = None
        hidden_states_58 = torch._C._nn.linear(
            hidden_states_57,
            l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_57 = l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_59 = torch.nn.functional.dropout(
            hidden_states_58, 0.1, False, False
        )
        hidden_states_58 = None
        add_46 = hidden_states_59 + y_9
        hidden_states_59 = y_9 = None
        hidden_states_60 = add_46.float()
        add_46 = None
        mean_30 = hidden_states_60.mean(-1, keepdim=True)
        sub_20 = hidden_states_60 - mean_30
        pow_11 = sub_20.pow(2)
        sub_20 = None
        variance_10 = pow_11.mean(-1, keepdim=True)
        pow_11 = None
        sub_21 = hidden_states_60 - mean_30
        hidden_states_60 = mean_30 = None
        add_47 = variance_10 + 1e-07
        variance_10 = None
        sqrt_15 = torch.sqrt(add_47)
        add_47 = None
        hidden_states_61 = sub_21 / sqrt_15
        sub_21 = sqrt_15 = None
        hidden_states_62 = hidden_states_61.to(torch.float32)
        hidden_states_61 = None
        mul_17 = (
            l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_weight_
            * hidden_states_62
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_weight_ = (
            hidden_states_62
        ) = None
        y_10 = (
            mul_17
            + l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_bias_
        )
        mul_17 = l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_bias_ = (None)
        return (y_10,)
