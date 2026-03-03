import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_,
        L_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_,
        L_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_buffers_relative_coords_table_,
        L_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_buffers_relative_position_index_,
        L_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_0_parameters_bias_,
        L_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_0_parameters_weight_,
        L_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_2_parameters_weight_,
        L_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_bias_,
        L_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_weight_,
        L_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_parameters_logit_scale_,
        L_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_bias_,
        L_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_weight_,
        L_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_layernorm_after_parameters_bias_,
        L_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_layernorm_after_parameters_weight_,
        L_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_layernorm_before_parameters_bias_,
        L_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_layernorm_before_parameters_weight_,
        L_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_output_modules_dense_parameters_bias_,
        L_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_output_modules_dense_parameters_weight_,
        L_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_downsample_modules_norm_parameters_bias_,
        L_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_downsample_modules_norm_parameters_weight_,
        L_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_downsample_modules_reduction_parameters_weight_,
        embeddings_3,
        hidden_states_windows,
        key_layer,
        query_layer,
    ):
        l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_buffers_relative_coords_table_ = L_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_buffers_relative_coords_table_
        l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_buffers_relative_position_index_ = L_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_buffers_relative_position_index_
        l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_0_parameters_bias_ = L_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_0_parameters_bias_
        l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_0_parameters_weight_ = L_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_0_parameters_weight_
        l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_2_parameters_weight_ = L_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_2_parameters_weight_
        l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_parameters_logit_scale_ = L_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_parameters_logit_scale_
        l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_layernorm_after_parameters_bias_ = L_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_layernorm_after_parameters_bias_
        l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_layernorm_after_parameters_weight_ = L_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_layernorm_after_parameters_weight_
        l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_layernorm_before_parameters_bias_ = L_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_layernorm_before_parameters_bias_
        l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_layernorm_before_parameters_weight_ = L_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_layernorm_before_parameters_weight_
        l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_output_modules_dense_parameters_bias_ = L_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_output_modules_dense_parameters_bias_
        l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_output_modules_dense_parameters_weight_ = L_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_output_modules_dense_parameters_weight_
        l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_downsample_modules_norm_parameters_bias_ = L_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_downsample_modules_norm_parameters_bias_
        l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_downsample_modules_norm_parameters_weight_ = L_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_downsample_modules_norm_parameters_weight_
        l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_downsample_modules_reduction_parameters_weight_ = L_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_downsample_modules_reduction_parameters_weight_
        linear_2 = torch.nn.functional.linear(
            hidden_states_windows,
            l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_windows = l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_6 = linear_2.view(64, -1, 2, 8)
        linear_2 = None
        value_layer = view_6.transpose(1, 2)
        view_6 = None
        normalize = torch.nn.functional.normalize(query_layer, dim=-1)
        query_layer = None
        normalize_1 = torch.nn.functional.normalize(key_layer, dim=-1)
        key_layer = None
        transpose_4 = normalize_1.transpose(-2, -1)
        normalize_1 = None
        attention_scores = normalize @ transpose_4
        normalize = transpose_4 = None
        clamp = torch.clamp(
            l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_parameters_logit_scale_,
            max=4.605170185988092,
        )
        l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_parameters_logit_scale_ = (
            None
        )
        logit_scale = clamp.exp()
        clamp = None
        attention_scores_1 = attention_scores * logit_scale
        attention_scores = logit_scale = None
        input_1 = torch.nn.functional.linear(
            l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_buffers_relative_coords_table_,
            l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_0_parameters_weight_,
            l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_buffers_relative_coords_table_ = l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_0_parameters_weight_ = l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_0_parameters_bias_ = (None)
        input_2 = torch.nn.functional.relu(input_1, inplace=True)
        input_1 = None
        input_3 = torch.nn.functional.linear(
            input_2,
            l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_2_parameters_weight_,
            None,
        )
        input_2 = l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_2_parameters_weight_ = (None)
        relative_position_bias_table = input_3.view(-1, 2)
        input_3 = None
        view_8 = l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_buffers_relative_position_index_.view(
            -1
        )
        l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_buffers_relative_position_index_ = (
            None
        )
        getitem = relative_position_bias_table[view_8]
        relative_position_bias_table = view_8 = None
        relative_position_bias = getitem.view(4, 4, -1)
        getitem = None
        permute_1 = relative_position_bias.permute(2, 0, 1)
        relative_position_bias = None
        relative_position_bias_1 = permute_1.contiguous()
        permute_1 = None
        sigmoid = torch.sigmoid(relative_position_bias_1)
        relative_position_bias_1 = None
        relative_position_bias_2 = 16 * sigmoid
        sigmoid = None
        unsqueeze = relative_position_bias_2.unsqueeze(0)
        relative_position_bias_2 = None
        attention_scores_2 = attention_scores_1 + unsqueeze
        attention_scores_1 = unsqueeze = None
        attention_probs = torch.nn.functional.softmax(attention_scores_2, dim=-1)
        attention_scores_2 = None
        attention_probs_1 = torch.nn.functional.dropout(
            attention_probs, 0.0, False, False
        )
        attention_probs = None
        context_layer = torch.matmul(attention_probs_1, value_layer)
        attention_probs_1 = value_layer = None
        permute_2 = context_layer.permute(0, 2, 1, 3)
        context_layer = None
        context_layer_1 = permute_2.contiguous()
        permute_2 = None
        context_layer_2 = context_layer_1.view((64, 4, 16))
        context_layer_1 = None
        hidden_states_2 = torch.nn.functional.linear(
            context_layer_2,
            l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_2 = l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_3 = torch.nn.functional.dropout(
            hidden_states_2, 0.0, False, False
        )
        hidden_states_2 = None
        attention_windows = hidden_states_3.view(-1, 2, 2, 16)
        hidden_states_3 = None
        windows_1 = attention_windows.view(-1, 8, 8, 2, 2, 16)
        attention_windows = None
        permute_3 = windows_1.permute(0, 1, 3, 2, 4, 5)
        windows_1 = None
        contiguous_3 = permute_3.contiguous()
        permute_3 = None
        windows_2 = contiguous_3.view(-1, 16, 16, 16)
        contiguous_3 = None
        attention_windows_1 = windows_2.view(1, 256, 16)
        windows_2 = None
        hidden_states_4 = torch.nn.functional.layer_norm(
            attention_windows_1,
            (16,),
            l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_layernorm_before_parameters_weight_,
            l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_layernorm_before_parameters_bias_,
            1e-05,
        )
        attention_windows_1 = l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_layernorm_before_parameters_weight_ = l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_layernorm_before_parameters_bias_ = (None)
        hidden_states_5 = embeddings_3 + hidden_states_4
        embeddings_3 = hidden_states_4 = None
        hidden_states_6 = torch.nn.functional.linear(
            hidden_states_5,
            l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_7 = torch.nn.functional.gelu(hidden_states_6)
        hidden_states_6 = None
        hidden_states_8 = torch.nn.functional.linear(
            hidden_states_7,
            l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_output_modules_dense_parameters_weight_,
            l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_7 = l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_output_modules_dense_parameters_weight_ = l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_9 = torch.nn.functional.dropout(
            hidden_states_8, 0.0, False, False
        )
        hidden_states_8 = None
        layer_norm_2 = torch.nn.functional.layer_norm(
            hidden_states_9,
            (16,),
            l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_layernorm_after_parameters_weight_,
            l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_layernorm_after_parameters_bias_,
            1e-05,
        )
        hidden_states_9 = l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_layernorm_after_parameters_weight_ = l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_layernorm_after_parameters_bias_ = (None)
        layer_output = hidden_states_5 + layer_norm_2
        hidden_states_5 = layer_norm_2 = None
        input_feature_1 = layer_output.view(1, 16, 16, 16)
        layer_output = None
        input_feature_0 = input_feature_1[
            (
                slice(None, None, None),
                slice(0, None, 2),
                slice(0, None, 2),
                slice(None, None, None),
            )
        ]
        input_feature_2 = input_feature_1[
            (
                slice(None, None, None),
                slice(1, None, 2),
                slice(0, None, 2),
                slice(None, None, None),
            )
        ]
        input_feature_3 = input_feature_1[
            (
                slice(None, None, None),
                slice(0, None, 2),
                slice(1, None, 2),
                slice(None, None, None),
            )
        ]
        input_feature_4 = input_feature_1[
            (
                slice(None, None, None),
                slice(1, None, 2),
                slice(1, None, 2),
                slice(None, None, None),
            )
        ]
        input_feature_1 = None
        input_feature_5 = torch.cat(
            [input_feature_0, input_feature_2, input_feature_3, input_feature_4], -1
        )
        input_feature_0 = input_feature_2 = input_feature_3 = input_feature_4 = None
        input_feature_6 = input_feature_5.view(1, -1, 64)
        input_feature_5 = None
        input_feature_7 = torch.nn.functional.linear(
            input_feature_6,
            l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_downsample_modules_reduction_parameters_weight_,
            None,
        )
        input_feature_6 = l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_downsample_modules_reduction_parameters_weight_ = (None)
        input_feature_8 = torch.nn.functional.layer_norm(
            input_feature_7,
            (32,),
            l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_downsample_modules_norm_parameters_weight_,
            l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_downsample_modules_norm_parameters_bias_,
            1e-05,
        )
        input_feature_7 = l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_downsample_modules_norm_parameters_weight_ = l_self_modules_swinv2_modules_encoder_modules_layers_modules_0_modules_downsample_modules_norm_parameters_bias_ = (None)
        hidden_states_10 = input_feature_8.view(1, 8, 8, 32)
        hidden_states_11 = torch.nn.functional.pad(
            hidden_states_10, (0, 0, 0, 0, 0, 0), "constant", None
        )
        hidden_states_10 = None
        input_feature_9 = hidden_states_11.view(1, 4, 2, 4, 2, 32)
        hidden_states_11 = None
        permute_4 = input_feature_9.permute(0, 1, 3, 2, 4, 5)
        input_feature_9 = None
        contiguous_4 = permute_4.contiguous()
        permute_4 = None
        windows_3 = contiguous_4.view(-1, 2, 2, 32)
        contiguous_4 = None
        hidden_states_windows_1 = windows_3.view(-1, 4, 32)
        windows_3 = None
        return (hidden_states_windows_1, input_feature_8)
