import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_classifier_parameters_bias_,
        L_self_modules_classifier_parameters_weight_,
        L_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_,
        L_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_,
        L_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_buffers_relative_coords_table_,
        L_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_buffers_relative_position_index_,
        L_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_0_parameters_bias_,
        L_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_0_parameters_weight_,
        L_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_2_parameters_weight_,
        L_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_bias_,
        L_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_weight_,
        L_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_parameters_logit_scale_,
        L_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_bias_,
        L_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_weight_,
        L_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_layernorm_after_parameters_bias_,
        L_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_layernorm_after_parameters_weight_,
        L_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_layernorm_before_parameters_bias_,
        L_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_layernorm_before_parameters_weight_,
        L_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_output_modules_dense_parameters_bias_,
        L_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_output_modules_dense_parameters_weight_,
        L_self_modules_swinv2_modules_layernorm_parameters_bias_,
        L_self_modules_swinv2_modules_layernorm_parameters_weight_,
        hidden_states_windows_3,
        input_feature_20,
        key_layer_3,
        query_layer_3,
    ):
        l_self_modules_classifier_parameters_bias_ = (
            L_self_modules_classifier_parameters_bias_
        )
        l_self_modules_classifier_parameters_weight_ = (
            L_self_modules_classifier_parameters_weight_
        )
        l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_buffers_relative_coords_table_ = L_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_buffers_relative_coords_table_
        l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_buffers_relative_position_index_ = L_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_buffers_relative_position_index_
        l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_0_parameters_bias_ = L_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_0_parameters_bias_
        l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_0_parameters_weight_ = L_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_0_parameters_weight_
        l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_2_parameters_weight_ = L_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_2_parameters_weight_
        l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_parameters_logit_scale_ = L_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_parameters_logit_scale_
        l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_layernorm_after_parameters_bias_ = L_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_layernorm_after_parameters_bias_
        l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_layernorm_after_parameters_weight_ = L_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_layernorm_after_parameters_weight_
        l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_layernorm_before_parameters_bias_ = L_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_layernorm_before_parameters_bias_
        l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_layernorm_before_parameters_weight_ = L_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_layernorm_before_parameters_weight_
        l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_output_modules_dense_parameters_bias_ = L_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_output_modules_dense_parameters_bias_
        l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_output_modules_dense_parameters_weight_ = L_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_output_modules_dense_parameters_weight_
        l_self_modules_swinv2_modules_layernorm_parameters_bias_ = (
            L_self_modules_swinv2_modules_layernorm_parameters_bias_
        )
        l_self_modules_swinv2_modules_layernorm_parameters_weight_ = (
            L_self_modules_swinv2_modules_layernorm_parameters_weight_
        )
        linear_28 = torch.nn.functional.linear(
            hidden_states_windows_3,
            l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_windows_3 = l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_60 = linear_28.view(4, -1, 4, 16)
        linear_28 = None
        value_layer_3 = view_60.transpose(1, 2)
        view_60 = None
        normalize_6 = torch.nn.functional.normalize(query_layer_3, dim=-1)
        query_layer_3 = None
        normalize_7 = torch.nn.functional.normalize(key_layer_3, dim=-1)
        key_layer_3 = None
        transpose_16 = normalize_7.transpose(-2, -1)
        normalize_7 = None
        attention_scores_12 = normalize_6 @ transpose_16
        normalize_6 = transpose_16 = None
        clamp_3 = torch.clamp(
            l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_parameters_logit_scale_,
            max=4.605170185988092,
        )
        l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_parameters_logit_scale_ = (
            None
        )
        logit_scale_3 = clamp_3.exp()
        clamp_3 = None
        attention_scores_13 = attention_scores_12 * logit_scale_3
        attention_scores_12 = logit_scale_3 = None
        input_10 = torch.nn.functional.linear(
            l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_buffers_relative_coords_table_,
            l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_0_parameters_weight_,
            l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_buffers_relative_coords_table_ = l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_0_parameters_weight_ = l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_0_parameters_bias_ = (None)
        input_11 = torch.nn.functional.relu(input_10, inplace=True)
        input_10 = None
        input_12 = torch.nn.functional.linear(
            input_11,
            l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_2_parameters_weight_,
            None,
        )
        input_11 = l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_2_parameters_weight_ = (None)
        relative_position_bias_table_3 = input_12.view(-1, 4)
        input_12 = None
        view_62 = l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_buffers_relative_position_index_ = (
            None
        )
        getitem_11 = relative_position_bias_table_3[view_62]
        relative_position_bias_table_3 = view_62 = None
        relative_position_bias_9 = getitem_11.view(4, 4, -1)
        getitem_11 = None
        permute_14 = relative_position_bias_9.permute(2, 0, 1)
        relative_position_bias_9 = None
        relative_position_bias_10 = permute_14.contiguous()
        permute_14 = None
        sigmoid_3 = torch.sigmoid(relative_position_bias_10)
        relative_position_bias_10 = None
        relative_position_bias_11 = 16 * sigmoid_3
        sigmoid_3 = None
        unsqueeze_9 = relative_position_bias_11.unsqueeze(0)
        relative_position_bias_11 = None
        attention_scores_14 = attention_scores_13 + unsqueeze_9
        attention_scores_13 = unsqueeze_9 = None
        attention_probs_6 = torch.nn.functional.softmax(attention_scores_14, dim=-1)
        attention_scores_14 = None
        attention_probs_7 = torch.nn.functional.dropout(
            attention_probs_6, 0.0, False, False
        )
        attention_probs_6 = None
        context_layer_9 = torch.matmul(attention_probs_7, value_layer_3)
        attention_probs_7 = value_layer_3 = None
        permute_15 = context_layer_9.permute(0, 2, 1, 3)
        context_layer_9 = None
        context_layer_10 = permute_15.contiguous()
        permute_15 = None
        context_layer_11 = context_layer_10.view((4, 4, 64))
        context_layer_10 = None
        hidden_states_32 = torch.nn.functional.linear(
            context_layer_11,
            l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_11 = l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_33 = torch.nn.functional.dropout(
            hidden_states_32, 0.0, False, False
        )
        hidden_states_32 = None
        attention_windows_7 = hidden_states_33.view(-1, 2, 2, 64)
        hidden_states_33 = None
        windows_11 = attention_windows_7.view(-1, 2, 2, 2, 2, 64)
        attention_windows_7 = None
        permute_16 = windows_11.permute(0, 1, 3, 2, 4, 5)
        windows_11 = None
        contiguous_16 = permute_16.contiguous()
        permute_16 = None
        windows_12 = contiguous_16.view(-1, 4, 4, 64)
        contiguous_16 = None
        attention_windows_8 = windows_12.view(1, 16, 64)
        windows_12 = None
        hidden_states_34 = torch.nn.functional.layer_norm(
            attention_windows_8,
            (64,),
            l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_layernorm_before_parameters_weight_,
            l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_layernorm_before_parameters_bias_,
            1e-05,
        )
        attention_windows_8 = l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_layernorm_before_parameters_weight_ = l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_layernorm_before_parameters_bias_ = (None)
        hidden_states_35 = input_feature_20 + hidden_states_34
        input_feature_20 = hidden_states_34 = None
        hidden_states_36 = torch.nn.functional.linear(
            hidden_states_35,
            l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_37 = torch.nn.functional.gelu(hidden_states_36)
        hidden_states_36 = None
        hidden_states_38 = torch.nn.functional.linear(
            hidden_states_37,
            l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_output_modules_dense_parameters_weight_,
            l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_37 = l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_output_modules_dense_parameters_weight_ = l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_39 = torch.nn.functional.dropout(
            hidden_states_38, 0.0, False, False
        )
        hidden_states_38 = None
        layer_norm_10 = torch.nn.functional.layer_norm(
            hidden_states_39,
            (64,),
            l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_layernorm_after_parameters_weight_,
            l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_layernorm_after_parameters_bias_,
            1e-05,
        )
        hidden_states_39 = l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_layernorm_after_parameters_weight_ = l_self_modules_swinv2_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_layernorm_after_parameters_bias_ = (None)
        layer_output_3 = hidden_states_35 + layer_norm_10
        hidden_states_35 = layer_norm_10 = None
        sequence_output = torch.nn.functional.layer_norm(
            layer_output_3,
            (64,),
            l_self_modules_swinv2_modules_layernorm_parameters_weight_,
            l_self_modules_swinv2_modules_layernorm_parameters_bias_,
            1e-05,
        )
        layer_output_3 = (
            l_self_modules_swinv2_modules_layernorm_parameters_weight_
        ) = l_self_modules_swinv2_modules_layernorm_parameters_bias_ = None
        transpose_17 = sequence_output.transpose(1, 2)
        sequence_output = None
        pooled_output = torch.adaptive_avg_pool1d(transpose_17, 1)
        transpose_17 = None
        pooled_output_1 = torch.flatten(pooled_output, 1)
        pooled_output = None
        logits = torch.nn.functional.linear(
            pooled_output_1,
            l_self_modules_classifier_parameters_weight_,
            l_self_modules_classifier_parameters_bias_,
        )
        pooled_output_1 = (
            l_self_modules_classifier_parameters_weight_
        ) = l_self_modules_classifier_parameters_bias_ = None
        return (logits,)
