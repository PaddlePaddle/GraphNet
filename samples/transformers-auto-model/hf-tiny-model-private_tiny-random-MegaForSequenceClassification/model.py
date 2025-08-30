import torch

from torch import inf


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_input_ids_: torch.Tensor,
        L_self_modules_embedding_layer_modules_word_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embedding_layer_buffers_token_type_ids_: torch.Tensor,
        L_self_modules_embedding_layer_modules_token_type_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_attention_mask_: torch.Tensor,
        L_self_modules_layers_modules_0_modules_mega_layer_modules_norm_modules_norm_parameters_scalar_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_mega_layer_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_mega_layer_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_mega_layer_modules_ema_gate_parameters_residual_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_mega_layer_modules_ema_gate_parameters_damping_factor_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_mega_layer_modules_ema_gate_parameters_decay_factor_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_mega_layer_modules_ema_gate_parameters_ema_expansion_matrix_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_mega_layer_modules_ema_gate_parameters_kernel_projection_matrix_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_mega_layer_modules_mx_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_mega_layer_modules_mx_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_mega_layer_parameters_qk_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_mega_layer_parameters_qk_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_mega_layer_modules_rel_pos_bias_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_mega_layer_modules_rel_pos_bias_sine: torch.Tensor,
        L_self_modules_layers_modules_0_modules_mega_layer_modules_rel_pos_bias_buffers_float_tensor_: torch.Tensor,
        L_self_modules_layers_modules_0_modules_mega_layer_modules_rel_pos_bias_cosine: torch.Tensor,
        L_self_modules_layers_modules_0_modules_mega_layer_modules_rel_pos_bias_parameters_b_param_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_mega_layer_modules_h_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_mega_layer_modules_h_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_nffn_modules_norm_modules_norm_parameters_scalar_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_nffn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_nffn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_nffn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_nffn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_mega_layer_modules_norm_modules_norm_parameters_scalar_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_mega_layer_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_mega_layer_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_mega_layer_modules_ema_gate_parameters_residual_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_mega_layer_modules_ema_gate_parameters_damping_factor_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_mega_layer_modules_ema_gate_parameters_decay_factor_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_mega_layer_modules_ema_gate_parameters_ema_expansion_matrix_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_mega_layer_modules_ema_gate_parameters_kernel_projection_matrix_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_mega_layer_modules_mx_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_mega_layer_modules_mx_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_mega_layer_parameters_qk_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_mega_layer_parameters_qk_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_mega_layer_modules_rel_pos_bias_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_mega_layer_modules_rel_pos_bias_sine: torch.Tensor,
        L_self_modules_layers_modules_1_modules_mega_layer_modules_rel_pos_bias_buffers_float_tensor_: torch.Tensor,
        L_self_modules_layers_modules_1_modules_mega_layer_modules_rel_pos_bias_cosine: torch.Tensor,
        L_self_modules_layers_modules_1_modules_mega_layer_modules_rel_pos_bias_parameters_b_param_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_mega_layer_modules_h_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_mega_layer_modules_h_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_nffn_modules_norm_modules_norm_parameters_scalar_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_nffn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_nffn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_nffn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_nffn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_mega_layer_modules_norm_modules_norm_parameters_scalar_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_mega_layer_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_mega_layer_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_mega_layer_modules_ema_gate_parameters_residual_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_mega_layer_modules_ema_gate_parameters_damping_factor_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_mega_layer_modules_ema_gate_parameters_decay_factor_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_mega_layer_modules_ema_gate_parameters_ema_expansion_matrix_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_mega_layer_modules_ema_gate_parameters_kernel_projection_matrix_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_mega_layer_modules_mx_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_mega_layer_modules_mx_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_mega_layer_parameters_qk_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_mega_layer_parameters_qk_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_mega_layer_modules_rel_pos_bias_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_mega_layer_modules_rel_pos_bias_sine: torch.Tensor,
        L_self_modules_layers_modules_2_modules_mega_layer_modules_rel_pos_bias_buffers_float_tensor_: torch.Tensor,
        L_self_modules_layers_modules_2_modules_mega_layer_modules_rel_pos_bias_cosine: torch.Tensor,
        L_self_modules_layers_modules_2_modules_mega_layer_modules_rel_pos_bias_parameters_b_param_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_mega_layer_modules_h_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_mega_layer_modules_h_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_nffn_modules_norm_modules_norm_parameters_scalar_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_nffn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_nffn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_nffn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_nffn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_mega_layer_modules_norm_modules_norm_parameters_scalar_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_mega_layer_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_mega_layer_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_mega_layer_modules_ema_gate_parameters_residual_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_mega_layer_modules_ema_gate_parameters_damping_factor_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_mega_layer_modules_ema_gate_parameters_decay_factor_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_mega_layer_modules_ema_gate_parameters_ema_expansion_matrix_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_mega_layer_modules_ema_gate_parameters_kernel_projection_matrix_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_mega_layer_modules_mx_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_mega_layer_modules_mx_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_mega_layer_parameters_qk_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_mega_layer_parameters_qk_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_mega_layer_modules_rel_pos_bias_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_mega_layer_modules_rel_pos_bias_sine: torch.Tensor,
        L_self_modules_layers_modules_3_modules_mega_layer_modules_rel_pos_bias_buffers_float_tensor_: torch.Tensor,
        L_self_modules_layers_modules_3_modules_mega_layer_modules_rel_pos_bias_cosine: torch.Tensor,
        L_self_modules_layers_modules_3_modules_mega_layer_modules_rel_pos_bias_parameters_b_param_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_mega_layer_modules_h_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_mega_layer_modules_h_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_nffn_modules_norm_modules_norm_parameters_scalar_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_nffn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_nffn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_nffn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_nffn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_mega_layer_modules_norm_modules_norm_parameters_scalar_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_mega_layer_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_mega_layer_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_mega_layer_modules_ema_gate_parameters_residual_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_mega_layer_modules_ema_gate_parameters_damping_factor_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_mega_layer_modules_ema_gate_parameters_decay_factor_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_mega_layer_modules_ema_gate_parameters_ema_expansion_matrix_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_mega_layer_modules_ema_gate_parameters_kernel_projection_matrix_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_mega_layer_modules_mx_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_mega_layer_modules_mx_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_mega_layer_parameters_qk_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_mega_layer_parameters_qk_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_mega_layer_modules_rel_pos_bias_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_mega_layer_modules_rel_pos_bias_sine: torch.Tensor,
        L_self_modules_layers_modules_4_modules_mega_layer_modules_rel_pos_bias_buffers_float_tensor_: torch.Tensor,
        L_self_modules_layers_modules_4_modules_mega_layer_modules_rel_pos_bias_cosine: torch.Tensor,
        L_self_modules_layers_modules_4_modules_mega_layer_modules_rel_pos_bias_parameters_b_param_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_mega_layer_modules_h_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_mega_layer_modules_h_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_nffn_modules_norm_modules_norm_parameters_scalar_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_nffn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_nffn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_nffn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_nffn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_pooler_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_pooler_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_input_ids_ = L_input_ids_
        l_self_modules_embedding_layer_modules_word_embeddings_parameters_weight_ = (
            L_self_modules_embedding_layer_modules_word_embeddings_parameters_weight_
        )
        l_self_modules_embedding_layer_buffers_token_type_ids_ = (
            L_self_modules_embedding_layer_buffers_token_type_ids_
        )
        l_self_modules_embedding_layer_modules_token_type_embeddings_parameters_weight_ = L_self_modules_embedding_layer_modules_token_type_embeddings_parameters_weight_
        l_attention_mask_ = L_attention_mask_
        l_self_modules_layers_modules_0_modules_mega_layer_modules_norm_modules_norm_parameters_scalar_ = L_self_modules_layers_modules_0_modules_mega_layer_modules_norm_modules_norm_parameters_scalar_
        l_self_modules_layers_modules_0_modules_mega_layer_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_mega_layer_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_mega_layer_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_0_modules_mega_layer_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_0_modules_mega_layer_modules_ema_gate_parameters_residual_weight_ = L_self_modules_layers_modules_0_modules_mega_layer_modules_ema_gate_parameters_residual_weight_
        l_self_modules_layers_modules_0_modules_mega_layer_modules_ema_gate_parameters_damping_factor_ = L_self_modules_layers_modules_0_modules_mega_layer_modules_ema_gate_parameters_damping_factor_
        l_self_modules_layers_modules_0_modules_mega_layer_modules_ema_gate_parameters_decay_factor_ = L_self_modules_layers_modules_0_modules_mega_layer_modules_ema_gate_parameters_decay_factor_
        l_self_modules_layers_modules_0_modules_mega_layer_modules_ema_gate_parameters_ema_expansion_matrix_ = L_self_modules_layers_modules_0_modules_mega_layer_modules_ema_gate_parameters_ema_expansion_matrix_
        l_self_modules_layers_modules_0_modules_mega_layer_modules_ema_gate_parameters_kernel_projection_matrix_ = L_self_modules_layers_modules_0_modules_mega_layer_modules_ema_gate_parameters_kernel_projection_matrix_
        l_self_modules_layers_modules_0_modules_mega_layer_modules_mx_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_mega_layer_modules_mx_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_mega_layer_modules_mx_proj_parameters_bias_ = L_self_modules_layers_modules_0_modules_mega_layer_modules_mx_proj_parameters_bias_
        l_self_modules_layers_modules_0_modules_mega_layer_parameters_qk_weight_ = (
            L_self_modules_layers_modules_0_modules_mega_layer_parameters_qk_weight_
        )
        l_self_modules_layers_modules_0_modules_mega_layer_parameters_qk_bias_ = (
            L_self_modules_layers_modules_0_modules_mega_layer_parameters_qk_bias_
        )
        l_self_modules_layers_modules_0_modules_mega_layer_modules_rel_pos_bias_parameters_alpha_ = L_self_modules_layers_modules_0_modules_mega_layer_modules_rel_pos_bias_parameters_alpha_
        l_self_modules_layers_modules_0_modules_mega_layer_modules_rel_pos_bias_sine = (
            L_self_modules_layers_modules_0_modules_mega_layer_modules_rel_pos_bias_sine
        )
        l_self_modules_layers_modules_0_modules_mega_layer_modules_rel_pos_bias_buffers_float_tensor_ = L_self_modules_layers_modules_0_modules_mega_layer_modules_rel_pos_bias_buffers_float_tensor_
        l_self_modules_layers_modules_0_modules_mega_layer_modules_rel_pos_bias_cosine = L_self_modules_layers_modules_0_modules_mega_layer_modules_rel_pos_bias_cosine
        l_self_modules_layers_modules_0_modules_mega_layer_modules_rel_pos_bias_parameters_b_param_ = L_self_modules_layers_modules_0_modules_mega_layer_modules_rel_pos_bias_parameters_b_param_
        l_self_modules_layers_modules_0_modules_mega_layer_modules_h_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_mega_layer_modules_h_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_mega_layer_modules_h_proj_parameters_bias_ = L_self_modules_layers_modules_0_modules_mega_layer_modules_h_proj_parameters_bias_
        l_self_modules_layers_modules_0_modules_nffn_modules_norm_modules_norm_parameters_scalar_ = L_self_modules_layers_modules_0_modules_nffn_modules_norm_modules_norm_parameters_scalar_
        l_self_modules_layers_modules_0_modules_nffn_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_0_modules_nffn_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_0_modules_nffn_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_0_modules_nffn_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_0_modules_nffn_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_0_modules_nffn_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_0_modules_nffn_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_0_modules_nffn_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_1_modules_mega_layer_modules_norm_modules_norm_parameters_scalar_ = L_self_modules_layers_modules_1_modules_mega_layer_modules_norm_modules_norm_parameters_scalar_
        l_self_modules_layers_modules_1_modules_mega_layer_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_mega_layer_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_mega_layer_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_1_modules_mega_layer_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_1_modules_mega_layer_modules_ema_gate_parameters_residual_weight_ = L_self_modules_layers_modules_1_modules_mega_layer_modules_ema_gate_parameters_residual_weight_
        l_self_modules_layers_modules_1_modules_mega_layer_modules_ema_gate_parameters_damping_factor_ = L_self_modules_layers_modules_1_modules_mega_layer_modules_ema_gate_parameters_damping_factor_
        l_self_modules_layers_modules_1_modules_mega_layer_modules_ema_gate_parameters_decay_factor_ = L_self_modules_layers_modules_1_modules_mega_layer_modules_ema_gate_parameters_decay_factor_
        l_self_modules_layers_modules_1_modules_mega_layer_modules_ema_gate_parameters_ema_expansion_matrix_ = L_self_modules_layers_modules_1_modules_mega_layer_modules_ema_gate_parameters_ema_expansion_matrix_
        l_self_modules_layers_modules_1_modules_mega_layer_modules_ema_gate_parameters_kernel_projection_matrix_ = L_self_modules_layers_modules_1_modules_mega_layer_modules_ema_gate_parameters_kernel_projection_matrix_
        l_self_modules_layers_modules_1_modules_mega_layer_modules_mx_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_mega_layer_modules_mx_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_mega_layer_modules_mx_proj_parameters_bias_ = L_self_modules_layers_modules_1_modules_mega_layer_modules_mx_proj_parameters_bias_
        l_self_modules_layers_modules_1_modules_mega_layer_parameters_qk_weight_ = (
            L_self_modules_layers_modules_1_modules_mega_layer_parameters_qk_weight_
        )
        l_self_modules_layers_modules_1_modules_mega_layer_parameters_qk_bias_ = (
            L_self_modules_layers_modules_1_modules_mega_layer_parameters_qk_bias_
        )
        l_self_modules_layers_modules_1_modules_mega_layer_modules_rel_pos_bias_parameters_alpha_ = L_self_modules_layers_modules_1_modules_mega_layer_modules_rel_pos_bias_parameters_alpha_
        l_self_modules_layers_modules_1_modules_mega_layer_modules_rel_pos_bias_sine = (
            L_self_modules_layers_modules_1_modules_mega_layer_modules_rel_pos_bias_sine
        )
        l_self_modules_layers_modules_1_modules_mega_layer_modules_rel_pos_bias_buffers_float_tensor_ = L_self_modules_layers_modules_1_modules_mega_layer_modules_rel_pos_bias_buffers_float_tensor_
        l_self_modules_layers_modules_1_modules_mega_layer_modules_rel_pos_bias_cosine = L_self_modules_layers_modules_1_modules_mega_layer_modules_rel_pos_bias_cosine
        l_self_modules_layers_modules_1_modules_mega_layer_modules_rel_pos_bias_parameters_b_param_ = L_self_modules_layers_modules_1_modules_mega_layer_modules_rel_pos_bias_parameters_b_param_
        l_self_modules_layers_modules_1_modules_mega_layer_modules_h_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_mega_layer_modules_h_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_mega_layer_modules_h_proj_parameters_bias_ = L_self_modules_layers_modules_1_modules_mega_layer_modules_h_proj_parameters_bias_
        l_self_modules_layers_modules_1_modules_nffn_modules_norm_modules_norm_parameters_scalar_ = L_self_modules_layers_modules_1_modules_nffn_modules_norm_modules_norm_parameters_scalar_
        l_self_modules_layers_modules_1_modules_nffn_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_1_modules_nffn_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_1_modules_nffn_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_1_modules_nffn_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_1_modules_nffn_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_1_modules_nffn_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_1_modules_nffn_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_1_modules_nffn_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_2_modules_mega_layer_modules_norm_modules_norm_parameters_scalar_ = L_self_modules_layers_modules_2_modules_mega_layer_modules_norm_modules_norm_parameters_scalar_
        l_self_modules_layers_modules_2_modules_mega_layer_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_mega_layer_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_mega_layer_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_mega_layer_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_mega_layer_modules_ema_gate_parameters_residual_weight_ = L_self_modules_layers_modules_2_modules_mega_layer_modules_ema_gate_parameters_residual_weight_
        l_self_modules_layers_modules_2_modules_mega_layer_modules_ema_gate_parameters_damping_factor_ = L_self_modules_layers_modules_2_modules_mega_layer_modules_ema_gate_parameters_damping_factor_
        l_self_modules_layers_modules_2_modules_mega_layer_modules_ema_gate_parameters_decay_factor_ = L_self_modules_layers_modules_2_modules_mega_layer_modules_ema_gate_parameters_decay_factor_
        l_self_modules_layers_modules_2_modules_mega_layer_modules_ema_gate_parameters_ema_expansion_matrix_ = L_self_modules_layers_modules_2_modules_mega_layer_modules_ema_gate_parameters_ema_expansion_matrix_
        l_self_modules_layers_modules_2_modules_mega_layer_modules_ema_gate_parameters_kernel_projection_matrix_ = L_self_modules_layers_modules_2_modules_mega_layer_modules_ema_gate_parameters_kernel_projection_matrix_
        l_self_modules_layers_modules_2_modules_mega_layer_modules_mx_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_mega_layer_modules_mx_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_mega_layer_modules_mx_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_mega_layer_modules_mx_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_mega_layer_parameters_qk_weight_ = (
            L_self_modules_layers_modules_2_modules_mega_layer_parameters_qk_weight_
        )
        l_self_modules_layers_modules_2_modules_mega_layer_parameters_qk_bias_ = (
            L_self_modules_layers_modules_2_modules_mega_layer_parameters_qk_bias_
        )
        l_self_modules_layers_modules_2_modules_mega_layer_modules_rel_pos_bias_parameters_alpha_ = L_self_modules_layers_modules_2_modules_mega_layer_modules_rel_pos_bias_parameters_alpha_
        l_self_modules_layers_modules_2_modules_mega_layer_modules_rel_pos_bias_sine = (
            L_self_modules_layers_modules_2_modules_mega_layer_modules_rel_pos_bias_sine
        )
        l_self_modules_layers_modules_2_modules_mega_layer_modules_rel_pos_bias_buffers_float_tensor_ = L_self_modules_layers_modules_2_modules_mega_layer_modules_rel_pos_bias_buffers_float_tensor_
        l_self_modules_layers_modules_2_modules_mega_layer_modules_rel_pos_bias_cosine = L_self_modules_layers_modules_2_modules_mega_layer_modules_rel_pos_bias_cosine
        l_self_modules_layers_modules_2_modules_mega_layer_modules_rel_pos_bias_parameters_b_param_ = L_self_modules_layers_modules_2_modules_mega_layer_modules_rel_pos_bias_parameters_b_param_
        l_self_modules_layers_modules_2_modules_mega_layer_modules_h_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_mega_layer_modules_h_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_mega_layer_modules_h_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_mega_layer_modules_h_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_nffn_modules_norm_modules_norm_parameters_scalar_ = L_self_modules_layers_modules_2_modules_nffn_modules_norm_modules_norm_parameters_scalar_
        l_self_modules_layers_modules_2_modules_nffn_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_2_modules_nffn_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_2_modules_nffn_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_2_modules_nffn_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_2_modules_nffn_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_2_modules_nffn_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_2_modules_nffn_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_2_modules_nffn_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_3_modules_mega_layer_modules_norm_modules_norm_parameters_scalar_ = L_self_modules_layers_modules_3_modules_mega_layer_modules_norm_modules_norm_parameters_scalar_
        l_self_modules_layers_modules_3_modules_mega_layer_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_mega_layer_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_mega_layer_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_3_modules_mega_layer_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_3_modules_mega_layer_modules_ema_gate_parameters_residual_weight_ = L_self_modules_layers_modules_3_modules_mega_layer_modules_ema_gate_parameters_residual_weight_
        l_self_modules_layers_modules_3_modules_mega_layer_modules_ema_gate_parameters_damping_factor_ = L_self_modules_layers_modules_3_modules_mega_layer_modules_ema_gate_parameters_damping_factor_
        l_self_modules_layers_modules_3_modules_mega_layer_modules_ema_gate_parameters_decay_factor_ = L_self_modules_layers_modules_3_modules_mega_layer_modules_ema_gate_parameters_decay_factor_
        l_self_modules_layers_modules_3_modules_mega_layer_modules_ema_gate_parameters_ema_expansion_matrix_ = L_self_modules_layers_modules_3_modules_mega_layer_modules_ema_gate_parameters_ema_expansion_matrix_
        l_self_modules_layers_modules_3_modules_mega_layer_modules_ema_gate_parameters_kernel_projection_matrix_ = L_self_modules_layers_modules_3_modules_mega_layer_modules_ema_gate_parameters_kernel_projection_matrix_
        l_self_modules_layers_modules_3_modules_mega_layer_modules_mx_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_mega_layer_modules_mx_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_mega_layer_modules_mx_proj_parameters_bias_ = L_self_modules_layers_modules_3_modules_mega_layer_modules_mx_proj_parameters_bias_
        l_self_modules_layers_modules_3_modules_mega_layer_parameters_qk_weight_ = (
            L_self_modules_layers_modules_3_modules_mega_layer_parameters_qk_weight_
        )
        l_self_modules_layers_modules_3_modules_mega_layer_parameters_qk_bias_ = (
            L_self_modules_layers_modules_3_modules_mega_layer_parameters_qk_bias_
        )
        l_self_modules_layers_modules_3_modules_mega_layer_modules_rel_pos_bias_parameters_alpha_ = L_self_modules_layers_modules_3_modules_mega_layer_modules_rel_pos_bias_parameters_alpha_
        l_self_modules_layers_modules_3_modules_mega_layer_modules_rel_pos_bias_sine = (
            L_self_modules_layers_modules_3_modules_mega_layer_modules_rel_pos_bias_sine
        )
        l_self_modules_layers_modules_3_modules_mega_layer_modules_rel_pos_bias_buffers_float_tensor_ = L_self_modules_layers_modules_3_modules_mega_layer_modules_rel_pos_bias_buffers_float_tensor_
        l_self_modules_layers_modules_3_modules_mega_layer_modules_rel_pos_bias_cosine = L_self_modules_layers_modules_3_modules_mega_layer_modules_rel_pos_bias_cosine
        l_self_modules_layers_modules_3_modules_mega_layer_modules_rel_pos_bias_parameters_b_param_ = L_self_modules_layers_modules_3_modules_mega_layer_modules_rel_pos_bias_parameters_b_param_
        l_self_modules_layers_modules_3_modules_mega_layer_modules_h_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_mega_layer_modules_h_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_mega_layer_modules_h_proj_parameters_bias_ = L_self_modules_layers_modules_3_modules_mega_layer_modules_h_proj_parameters_bias_
        l_self_modules_layers_modules_3_modules_nffn_modules_norm_modules_norm_parameters_scalar_ = L_self_modules_layers_modules_3_modules_nffn_modules_norm_modules_norm_parameters_scalar_
        l_self_modules_layers_modules_3_modules_nffn_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_3_modules_nffn_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_3_modules_nffn_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_3_modules_nffn_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_3_modules_nffn_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_3_modules_nffn_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_3_modules_nffn_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_3_modules_nffn_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_4_modules_mega_layer_modules_norm_modules_norm_parameters_scalar_ = L_self_modules_layers_modules_4_modules_mega_layer_modules_norm_modules_norm_parameters_scalar_
        l_self_modules_layers_modules_4_modules_mega_layer_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_mega_layer_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_mega_layer_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_4_modules_mega_layer_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_4_modules_mega_layer_modules_ema_gate_parameters_residual_weight_ = L_self_modules_layers_modules_4_modules_mega_layer_modules_ema_gate_parameters_residual_weight_
        l_self_modules_layers_modules_4_modules_mega_layer_modules_ema_gate_parameters_damping_factor_ = L_self_modules_layers_modules_4_modules_mega_layer_modules_ema_gate_parameters_damping_factor_
        l_self_modules_layers_modules_4_modules_mega_layer_modules_ema_gate_parameters_decay_factor_ = L_self_modules_layers_modules_4_modules_mega_layer_modules_ema_gate_parameters_decay_factor_
        l_self_modules_layers_modules_4_modules_mega_layer_modules_ema_gate_parameters_ema_expansion_matrix_ = L_self_modules_layers_modules_4_modules_mega_layer_modules_ema_gate_parameters_ema_expansion_matrix_
        l_self_modules_layers_modules_4_modules_mega_layer_modules_ema_gate_parameters_kernel_projection_matrix_ = L_self_modules_layers_modules_4_modules_mega_layer_modules_ema_gate_parameters_kernel_projection_matrix_
        l_self_modules_layers_modules_4_modules_mega_layer_modules_mx_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_mega_layer_modules_mx_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_mega_layer_modules_mx_proj_parameters_bias_ = L_self_modules_layers_modules_4_modules_mega_layer_modules_mx_proj_parameters_bias_
        l_self_modules_layers_modules_4_modules_mega_layer_parameters_qk_weight_ = (
            L_self_modules_layers_modules_4_modules_mega_layer_parameters_qk_weight_
        )
        l_self_modules_layers_modules_4_modules_mega_layer_parameters_qk_bias_ = (
            L_self_modules_layers_modules_4_modules_mega_layer_parameters_qk_bias_
        )
        l_self_modules_layers_modules_4_modules_mega_layer_modules_rel_pos_bias_parameters_alpha_ = L_self_modules_layers_modules_4_modules_mega_layer_modules_rel_pos_bias_parameters_alpha_
        l_self_modules_layers_modules_4_modules_mega_layer_modules_rel_pos_bias_sine = (
            L_self_modules_layers_modules_4_modules_mega_layer_modules_rel_pos_bias_sine
        )
        l_self_modules_layers_modules_4_modules_mega_layer_modules_rel_pos_bias_buffers_float_tensor_ = L_self_modules_layers_modules_4_modules_mega_layer_modules_rel_pos_bias_buffers_float_tensor_
        l_self_modules_layers_modules_4_modules_mega_layer_modules_rel_pos_bias_cosine = L_self_modules_layers_modules_4_modules_mega_layer_modules_rel_pos_bias_cosine
        l_self_modules_layers_modules_4_modules_mega_layer_modules_rel_pos_bias_parameters_b_param_ = L_self_modules_layers_modules_4_modules_mega_layer_modules_rel_pos_bias_parameters_b_param_
        l_self_modules_layers_modules_4_modules_mega_layer_modules_h_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_mega_layer_modules_h_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_mega_layer_modules_h_proj_parameters_bias_ = L_self_modules_layers_modules_4_modules_mega_layer_modules_h_proj_parameters_bias_
        l_self_modules_layers_modules_4_modules_nffn_modules_norm_modules_norm_parameters_scalar_ = L_self_modules_layers_modules_4_modules_nffn_modules_norm_modules_norm_parameters_scalar_
        l_self_modules_layers_modules_4_modules_nffn_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_4_modules_nffn_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_4_modules_nffn_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_4_modules_nffn_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_4_modules_nffn_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_4_modules_nffn_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_4_modules_nffn_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_4_modules_nffn_modules_fc2_parameters_bias_
        )
        l_self_modules_pooler_modules_dense_parameters_weight_ = (
            L_self_modules_pooler_modules_dense_parameters_weight_
        )
        l_self_modules_pooler_modules_dense_parameters_bias_ = (
            L_self_modules_pooler_modules_dense_parameters_bias_
        )
        inputs_embeds = torch.nn.functional.embedding(
            l_input_ids_,
            l_self_modules_embedding_layer_modules_word_embeddings_parameters_weight_,
            1,
            None,
            2.0,
            False,
            False,
        )
        l_input_ids_ = (
            l_self_modules_embedding_layer_modules_word_embeddings_parameters_weight_
        ) = None
        buffered_token_type_ids = (
            l_self_modules_embedding_layer_buffers_token_type_ids_[
                (slice(None, None, None), slice(None, 22, None))
            ]
        )
        l_self_modules_embedding_layer_buffers_token_type_ids_ = None
        buffered_token_type_ids_expanded = buffered_token_type_ids.expand(1, 22)
        buffered_token_type_ids = None
        token_type_embeddings = torch.nn.functional.embedding(
            buffered_token_type_ids_expanded,
            l_self_modules_embedding_layer_modules_token_type_embeddings_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        buffered_token_type_ids_expanded = l_self_modules_embedding_layer_modules_token_type_embeddings_parameters_weight_ = (None)
        embeddings = inputs_embeds + token_type_embeddings
        inputs_embeds = token_type_embeddings = None
        hidden_states = embeddings.transpose(0, 1)
        embeddings = None
        square = torch.square(hidden_states)
        mean_square = torch.mean(square, dim=-1, keepdim=True)
        square = None
        input_1 = (
            l_self_modules_layers_modules_0_modules_mega_layer_modules_norm_modules_norm_parameters_scalar_
            * hidden_states
        )
        l_self_modules_layers_modules_0_modules_mega_layer_modules_norm_modules_norm_parameters_scalar_ = (
            None
        )
        add_1 = mean_square + 1e-05
        mean_square = None
        rsqrt = torch.rsqrt(add_1)
        add_1 = None
        output = input_1 * rsqrt
        input_1 = rsqrt = None
        linear = torch._C._nn.linear(
            output,
            l_self_modules_layers_modules_0_modules_mega_layer_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_0_modules_mega_layer_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_0_modules_mega_layer_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_mega_layer_modules_v_proj_parameters_bias_ = (None)
        value = torch.nn.functional.silu(linear, inplace=False)
        linear = None
        residual = (
            output
            * l_self_modules_layers_modules_0_modules_mega_layer_modules_ema_gate_parameters_residual_weight_
        )
        l_self_modules_layers_modules_0_modules_mega_layer_modules_ema_gate_parameters_residual_weight_ = (
            None
        )
        inputs = output.permute(1, 2, 0)
        output = None
        unsqueeze = l_attention_mask_.unsqueeze(1)
        type_as = unsqueeze.type_as(inputs)
        unsqueeze = None
        inputs_1 = inputs * type_as
        inputs = type_as = None
        damping_factor = torch.sigmoid(
            l_self_modules_layers_modules_0_modules_mega_layer_modules_ema_gate_parameters_damping_factor_
        )
        l_self_modules_layers_modules_0_modules_mega_layer_modules_ema_gate_parameters_damping_factor_ = (
            None
        )
        decay_factor = torch.sigmoid(
            l_self_modules_layers_modules_0_modules_mega_layer_modules_ema_gate_parameters_decay_factor_
        )
        l_self_modules_layers_modules_0_modules_mega_layer_modules_ema_gate_parameters_decay_factor_ = (
            None
        )
        mul_4 = damping_factor * decay_factor
        decay_factor = None
        previous_timestep_weight = 1.0 - mul_4
        mul_4 = None
        arange = torch.arange(22)
        to = arange.to(damping_factor)
        arange = None
        view = to.view(1, 1, 22)
        to = None
        log = torch.log(previous_timestep_weight)
        previous_timestep_weight = None
        vander = view * log
        view = log = None
        mul_6 = (
            damping_factor
            * l_self_modules_layers_modules_0_modules_mega_layer_modules_ema_gate_parameters_ema_expansion_matrix_
        )
        damping_factor = l_self_modules_layers_modules_0_modules_mega_layer_modules_ema_gate_parameters_ema_expansion_matrix_ = (None)
        exp = torch.exp(vander)
        vander = None
        kernel = mul_6 * exp
        mul_6 = exp = None
        mul_8 = (
            l_self_modules_layers_modules_0_modules_mega_layer_modules_ema_gate_parameters_kernel_projection_matrix_
            * 0.25
        )
        l_self_modules_layers_modules_0_modules_mega_layer_modules_ema_gate_parameters_kernel_projection_matrix_ = (
            None
        )
        einsum = torch.functional.einsum("dnl,dn->dl", kernel, mul_8)
        kernel = mul_8 = None
        kernel_1 = einsum[(Ellipsis, slice(None, 22, None))]
        float_1 = inputs_1.float()
        inputs_fft = torch._C._fft.fft_rfft(float_1, n=44)
        float_1 = None
        float_2 = kernel_1.float()
        kernel_1 = None
        kernel_fft = torch._C._fft.fft_rfft(float_2, n=44)
        float_2 = None
        mul_9 = inputs_fft * kernel_fft
        inputs_fft = kernel_fft = None
        convolved_sequence = torch._C._fft.fft_irfft(mul_9, n=44)
        mul_9 = None
        ema_output = convolved_sequence[(Ellipsis, slice(0, 22, None))]
        convolved_sequence = None
        ema_output_1 = ema_output.type_as(inputs_1)
        ema_output = inputs_1 = None
        permute_1 = ema_output_1.permute(2, 0, 1)
        ema_output_1 = None
        add_2 = permute_1 + residual
        permute_1 = residual = None
        gated_ema_output = torch.nn.functional.silu(add_2)
        add_2 = None
        ema_out = torch.nn.functional.dropout(gated_ema_output, p=0.1, training=False)
        gated_ema_output = None
        base = torch._C._nn.linear(
            ema_out,
            l_self_modules_layers_modules_0_modules_mega_layer_modules_mx_proj_parameters_weight_,
            l_self_modules_layers_modules_0_modules_mega_layer_modules_mx_proj_parameters_bias_,
        )
        ema_out = l_self_modules_layers_modules_0_modules_mega_layer_modules_mx_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_mega_layer_modules_mx_proj_parameters_bias_ = (None)
        split = torch.functional.split(base, [32, 101, 32], dim=-1)
        base = None
        residual_weight = split[0]
        query_key_gates = split[1]
        intermediate_state = split[2]
        split = None
        residual_weight_1 = torch.sigmoid(residual_weight)
        residual_weight = None
        query_key_gates_1 = torch.nn.functional.silu(query_key_gates)
        query_key_gates = None
        split_1 = torch.functional.split(query_key_gates_1, [64, 37], dim=-1)
        query_key_gates_1 = None
        query_key = split_1[0]
        attention_gate = split_1[1]
        split_1 = None
        unsqueeze_1 = query_key.unsqueeze(2)
        query_key = None
        mul_10 = (
            unsqueeze_1
            * l_self_modules_layers_modules_0_modules_mega_layer_parameters_qk_weight_
        )
        unsqueeze_1 = (
            l_self_modules_layers_modules_0_modules_mega_layer_parameters_qk_weight_
        ) = None
        query_key_1 = (
            mul_10
            + l_self_modules_layers_modules_0_modules_mega_layer_parameters_qk_bias_
        )
        mul_10 = (
            l_self_modules_layers_modules_0_modules_mega_layer_parameters_qk_bias_
        ) = None
        unbind = torch.unbind(query_key_1, dim=2)
        query_key_1 = None
        query = unbind[0]
        key = unbind[1]
        unbind = None
        query_1 = query.transpose(0, 1)
        query = None
        key_1 = key.transpose(0, 1)
        key = None
        value_1 = value.transpose(0, 1)
        value = None
        query_2 = query_1.unsqueeze(1)
        query_1 = None
        key_2 = key_1.unsqueeze(1)
        key_1 = None
        value_2 = value_1.unsqueeze(1)
        value_1 = None
        padding_mask = l_attention_mask_.unsqueeze(1)
        expand_1 = l_self_modules_layers_modules_0_modules_mega_layer_modules_rel_pos_bias_parameters_alpha_.expand(
            22, 64
        )
        l_self_modules_layers_modules_0_modules_mega_layer_modules_rel_pos_bias_parameters_alpha_ = (
            None
        )
        chunk = torch.chunk(expand_1, 2, dim=-1)
        expand_1 = None
        chunk_1 = chunk[0]
        chunk_2 = chunk[1]
        chunk = None
        to_1 = l_self_modules_layers_modules_0_modules_mega_layer_modules_rel_pos_bias_sine.to(
            l_self_modules_layers_modules_0_modules_mega_layer_modules_rel_pos_bias_buffers_float_tensor_
        )
        l_self_modules_layers_modules_0_modules_mega_layer_modules_rel_pos_bias_sine = (
            None
        )
        to_2 = l_self_modules_layers_modules_0_modules_mega_layer_modules_rel_pos_bias_cosine.to(
            l_self_modules_layers_modules_0_modules_mega_layer_modules_rel_pos_bias_buffers_float_tensor_
        )
        l_self_modules_layers_modules_0_modules_mega_layer_modules_rel_pos_bias_cosine = (
            None
        )
        sin = to_1[slice(None, 22, None)]
        cos = to_2[slice(None, 22, None)]
        mul_11 = chunk_1 * cos
        mul_12 = chunk_2 * sin
        sub_1 = mul_11 - mul_12
        mul_11 = mul_12 = None
        mul_13 = chunk_2 * cos
        chunk_2 = cos = None
        mul_14 = chunk_1 * sin
        chunk_1 = sin = None
        add_4 = mul_13 + mul_14
        mul_13 = mul_14 = None
        rotary_alpha = torch.cat([sub_1, add_4], dim=1)
        sub_1 = add_4 = None
        expand_2 = l_self_modules_layers_modules_0_modules_mega_layer_modules_rel_pos_bias_parameters_b_param_.expand(
            22, 64
        )
        l_self_modules_layers_modules_0_modules_mega_layer_modules_rel_pos_bias_parameters_b_param_ = (
            None
        )
        chunk_3 = torch.chunk(expand_2, 2, dim=-1)
        expand_2 = None
        chunk_4 = chunk_3[0]
        chunk_5 = chunk_3[1]
        chunk_3 = None
        to_3 = to_1.to(
            l_self_modules_layers_modules_0_modules_mega_layer_modules_rel_pos_bias_buffers_float_tensor_
        )
        to_1 = None
        to_4 = to_2.to(
            l_self_modules_layers_modules_0_modules_mega_layer_modules_rel_pos_bias_buffers_float_tensor_
        )
        to_2 = l_self_modules_layers_modules_0_modules_mega_layer_modules_rel_pos_bias_buffers_float_tensor_ = (None)
        sin_1 = to_3[slice(None, 22, None)]
        cos_1 = to_4[slice(None, 22, None)]
        mul_15 = chunk_4 * cos_1
        mul_16 = chunk_5 * sin_1
        sub_2 = mul_15 - mul_16
        mul_15 = mul_16 = None
        mul_17 = chunk_5 * cos_1
        chunk_5 = cos_1 = None
        mul_18 = chunk_4 * sin_1
        chunk_4 = sin_1 = None
        add_5 = mul_17 + mul_18
        mul_17 = mul_18 = None
        rotary_beta = torch.cat([sub_2, add_5], dim=1)
        sub_2 = add_5 = None
        bias = torch.functional.einsum("mk,nk->mn", rotary_alpha, rotary_beta)
        rotary_alpha = rotary_beta = None
        query_3 = query_2 * 0.125
        query_2 = None
        transpose_4 = key_2.transpose(2, 3)
        key_2 = None
        matmul = torch.matmul(query_3, transpose_4)
        query_3 = transpose_4 = None
        qk = matmul + bias
        matmul = bias = None
        padding_mask_1 = 1 - padding_mask
        padding_mask = None
        padding_mask_all = padding_mask_1.all(dim=-1, keepdim=True)
        invert = ~padding_mask_all
        padding_mask_all = None
        padding_mask_2 = torch.logical_and(padding_mask_1, invert)
        padding_mask_1 = invert = None
        unsqueeze_6 = padding_mask_2.unsqueeze(2)
        padding_mask_2 = None
        to_5 = unsqueeze_6.to(torch.bool)
        unsqueeze_6 = None
        qk_1 = qk.masked_fill(to_5, -inf)
        qk = to_5 = None
        softmax = torch.nn.functional.softmax(qk_1, -1, _stacklevel=5)
        attn_weights = softmax.type_as(qk_1)
        softmax = qk_1 = None
        value_3 = torch.nn.functional.dropout(value_2, p=0.1, training=False)
        value_2 = None
        kernel_2 = torch.nn.functional.dropout(attn_weights, p=0.1, training=False)
        attn_weights = None
        matmul_1 = torch.matmul(kernel_2, value_3)
        kernel_2 = value_3 = None
        view_1 = matmul_1.view(1, 22, 37)
        matmul_1 = None
        weighted_self_output = view_1.transpose(0, 1)
        view_1 = None
        mul_20 = weighted_self_output * attention_gate
        weighted_self_output = attention_gate = None
        linear_2 = torch._C._nn.linear(
            mul_20,
            l_self_modules_layers_modules_0_modules_mega_layer_modules_h_proj_parameters_weight_,
            l_self_modules_layers_modules_0_modules_mega_layer_modules_h_proj_parameters_bias_,
        )
        mul_20 = l_self_modules_layers_modules_0_modules_mega_layer_modules_h_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_mega_layer_modules_h_proj_parameters_bias_ = (None)
        add_7 = intermediate_state + linear_2
        intermediate_state = linear_2 = None
        weighted_self_output_1 = torch.nn.functional.silu(add_7, inplace=False)
        add_7 = None
        weighted_self_output_2 = torch.nn.functional.dropout(
            weighted_self_output_1, p=0.1, training=False
        )
        weighted_self_output_1 = None
        sub_4 = weighted_self_output_2 - hidden_states
        weighted_self_output_2 = None
        out = torch.addcmul(hidden_states, residual_weight_1, sub_4)
        hidden_states = residual_weight_1 = sub_4 = None
        square_1 = torch.square(out)
        mean_square_1 = torch.mean(square_1, dim=-1, keepdim=True)
        square_1 = None
        input_2 = (
            l_self_modules_layers_modules_0_modules_nffn_modules_norm_modules_norm_parameters_scalar_
            * out
        )
        l_self_modules_layers_modules_0_modules_nffn_modules_norm_modules_norm_parameters_scalar_ = (
            None
        )
        add_8 = mean_square_1 + 1e-05
        mean_square_1 = None
        rsqrt_1 = torch.rsqrt(add_8)
        add_8 = None
        output_1 = input_2 * rsqrt_1
        input_2 = rsqrt_1 = None
        linear_3 = torch._C._nn.linear(
            output_1,
            l_self_modules_layers_modules_0_modules_nffn_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_0_modules_nffn_modules_fc1_parameters_bias_,
        )
        output_1 = (
            l_self_modules_layers_modules_0_modules_nffn_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layers_modules_0_modules_nffn_modules_fc1_parameters_bias_
        ) = None
        hidden = torch.nn.functional.silu(linear_3, inplace=False)
        linear_3 = None
        hidden_1 = torch.nn.functional.dropout(hidden, p=0.1, training=False)
        hidden = None
        output_2 = torch._C._nn.linear(
            hidden_1,
            l_self_modules_layers_modules_0_modules_nffn_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_0_modules_nffn_modules_fc2_parameters_bias_,
        )
        hidden_1 = (
            l_self_modules_layers_modules_0_modules_nffn_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layers_modules_0_modules_nffn_modules_fc2_parameters_bias_
        ) = None
        output_3 = torch.nn.functional.dropout(output_2, p=0.1, training=False)
        output_2 = None
        output_4 = output_3 + out
        output_3 = out = None
        square_2 = torch.square(output_4)
        mean_square_2 = torch.mean(square_2, dim=-1, keepdim=True)
        square_2 = None
        input_3 = (
            l_self_modules_layers_modules_1_modules_mega_layer_modules_norm_modules_norm_parameters_scalar_
            * output_4
        )
        l_self_modules_layers_modules_1_modules_mega_layer_modules_norm_modules_norm_parameters_scalar_ = (
            None
        )
        add_10 = mean_square_2 + 1e-05
        mean_square_2 = None
        rsqrt_2 = torch.rsqrt(add_10)
        add_10 = None
        output_5 = input_3 * rsqrt_2
        input_3 = rsqrt_2 = None
        linear_5 = torch._C._nn.linear(
            output_5,
            l_self_modules_layers_modules_1_modules_mega_layer_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_mega_layer_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_1_modules_mega_layer_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_mega_layer_modules_v_proj_parameters_bias_ = (None)
        value_4 = torch.nn.functional.silu(linear_5, inplace=False)
        linear_5 = None
        residual_1 = (
            output_5
            * l_self_modules_layers_modules_1_modules_mega_layer_modules_ema_gate_parameters_residual_weight_
        )
        l_self_modules_layers_modules_1_modules_mega_layer_modules_ema_gate_parameters_residual_weight_ = (
            None
        )
        inputs_2 = output_5.permute(1, 2, 0)
        output_5 = None
        unsqueeze_7 = l_attention_mask_.unsqueeze(1)
        type_as_3 = unsqueeze_7.type_as(inputs_2)
        unsqueeze_7 = None
        inputs_3 = inputs_2 * type_as_3
        inputs_2 = type_as_3 = None
        damping_factor_1 = torch.sigmoid(
            l_self_modules_layers_modules_1_modules_mega_layer_modules_ema_gate_parameters_damping_factor_
        )
        l_self_modules_layers_modules_1_modules_mega_layer_modules_ema_gate_parameters_damping_factor_ = (
            None
        )
        decay_factor_1 = torch.sigmoid(
            l_self_modules_layers_modules_1_modules_mega_layer_modules_ema_gate_parameters_decay_factor_
        )
        l_self_modules_layers_modules_1_modules_mega_layer_modules_ema_gate_parameters_decay_factor_ = (
            None
        )
        mul_27 = damping_factor_1 * decay_factor_1
        decay_factor_1 = None
        previous_timestep_weight_1 = 1.0 - mul_27
        mul_27 = None
        arange_1 = torch.arange(22)
        to_6 = arange_1.to(damping_factor_1)
        arange_1 = None
        view_2 = to_6.view(1, 1, 22)
        to_6 = None
        log_1 = torch.log(previous_timestep_weight_1)
        previous_timestep_weight_1 = None
        vander_1 = view_2 * log_1
        view_2 = log_1 = None
        mul_29 = (
            damping_factor_1
            * l_self_modules_layers_modules_1_modules_mega_layer_modules_ema_gate_parameters_ema_expansion_matrix_
        )
        damping_factor_1 = l_self_modules_layers_modules_1_modules_mega_layer_modules_ema_gate_parameters_ema_expansion_matrix_ = (None)
        exp_1 = torch.exp(vander_1)
        vander_1 = None
        kernel_3 = mul_29 * exp_1
        mul_29 = exp_1 = None
        mul_31 = (
            l_self_modules_layers_modules_1_modules_mega_layer_modules_ema_gate_parameters_kernel_projection_matrix_
            * 0.25
        )
        l_self_modules_layers_modules_1_modules_mega_layer_modules_ema_gate_parameters_kernel_projection_matrix_ = (
            None
        )
        einsum_2 = torch.functional.einsum("dnl,dn->dl", kernel_3, mul_31)
        kernel_3 = mul_31 = None
        kernel_4 = einsum_2[(Ellipsis, slice(None, 22, None))]
        float_3 = inputs_3.float()
        inputs_fft_1 = torch._C._fft.fft_rfft(float_3, n=44)
        float_3 = None
        float_4 = kernel_4.float()
        kernel_4 = None
        kernel_fft_1 = torch._C._fft.fft_rfft(float_4, n=44)
        float_4 = None
        mul_32 = inputs_fft_1 * kernel_fft_1
        inputs_fft_1 = kernel_fft_1 = None
        convolved_sequence_1 = torch._C._fft.fft_irfft(mul_32, n=44)
        mul_32 = None
        ema_output_2 = convolved_sequence_1[(Ellipsis, slice(0, 22, None))]
        convolved_sequence_1 = None
        ema_output_3 = ema_output_2.type_as(inputs_3)
        ema_output_2 = inputs_3 = None
        permute_3 = ema_output_3.permute(2, 0, 1)
        ema_output_3 = None
        add_11 = permute_3 + residual_1
        permute_3 = residual_1 = None
        gated_ema_output_1 = torch.nn.functional.silu(add_11)
        add_11 = None
        ema_out_1 = torch.nn.functional.dropout(
            gated_ema_output_1, p=0.1, training=False
        )
        gated_ema_output_1 = None
        base_1 = torch._C._nn.linear(
            ema_out_1,
            l_self_modules_layers_modules_1_modules_mega_layer_modules_mx_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_mega_layer_modules_mx_proj_parameters_bias_,
        )
        ema_out_1 = l_self_modules_layers_modules_1_modules_mega_layer_modules_mx_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_mega_layer_modules_mx_proj_parameters_bias_ = (None)
        split_2 = torch.functional.split(base_1, [32, 101, 32], dim=-1)
        base_1 = None
        residual_weight_2 = split_2[0]
        query_key_gates_2 = split_2[1]
        intermediate_state_1 = split_2[2]
        split_2 = None
        residual_weight_3 = torch.sigmoid(residual_weight_2)
        residual_weight_2 = None
        query_key_gates_3 = torch.nn.functional.silu(query_key_gates_2)
        query_key_gates_2 = None
        split_3 = torch.functional.split(query_key_gates_3, [64, 37], dim=-1)
        query_key_gates_3 = None
        query_key_2 = split_3[0]
        attention_gate_1 = split_3[1]
        split_3 = None
        unsqueeze_8 = query_key_2.unsqueeze(2)
        query_key_2 = None
        mul_33 = (
            unsqueeze_8
            * l_self_modules_layers_modules_1_modules_mega_layer_parameters_qk_weight_
        )
        unsqueeze_8 = (
            l_self_modules_layers_modules_1_modules_mega_layer_parameters_qk_weight_
        ) = None
        query_key_3 = (
            mul_33
            + l_self_modules_layers_modules_1_modules_mega_layer_parameters_qk_bias_
        )
        mul_33 = (
            l_self_modules_layers_modules_1_modules_mega_layer_parameters_qk_bias_
        ) = None
        unbind_1 = torch.unbind(query_key_3, dim=2)
        query_key_3 = None
        query_4 = unbind_1[0]
        key_3 = unbind_1[1]
        unbind_1 = None
        query_5 = query_4.transpose(0, 1)
        query_4 = None
        key_4 = key_3.transpose(0, 1)
        key_3 = None
        value_5 = value_4.transpose(0, 1)
        value_4 = None
        query_6 = query_5.unsqueeze(1)
        query_5 = None
        key_5 = key_4.unsqueeze(1)
        key_4 = None
        value_6 = value_5.unsqueeze(1)
        value_5 = None
        padding_mask_3 = l_attention_mask_.unsqueeze(1)
        expand_3 = l_self_modules_layers_modules_1_modules_mega_layer_modules_rel_pos_bias_parameters_alpha_.expand(
            22, 64
        )
        l_self_modules_layers_modules_1_modules_mega_layer_modules_rel_pos_bias_parameters_alpha_ = (
            None
        )
        chunk_6 = torch.chunk(expand_3, 2, dim=-1)
        expand_3 = None
        chunk_7 = chunk_6[0]
        chunk_8 = chunk_6[1]
        chunk_6 = None
        to_7 = l_self_modules_layers_modules_1_modules_mega_layer_modules_rel_pos_bias_sine.to(
            l_self_modules_layers_modules_1_modules_mega_layer_modules_rel_pos_bias_buffers_float_tensor_
        )
        l_self_modules_layers_modules_1_modules_mega_layer_modules_rel_pos_bias_sine = (
            None
        )
        to_8 = l_self_modules_layers_modules_1_modules_mega_layer_modules_rel_pos_bias_cosine.to(
            l_self_modules_layers_modules_1_modules_mega_layer_modules_rel_pos_bias_buffers_float_tensor_
        )
        l_self_modules_layers_modules_1_modules_mega_layer_modules_rel_pos_bias_cosine = (
            None
        )
        sin_2 = to_7[slice(None, 22, None)]
        cos_2 = to_8[slice(None, 22, None)]
        mul_34 = chunk_7 * cos_2
        mul_35 = chunk_8 * sin_2
        sub_6 = mul_34 - mul_35
        mul_34 = mul_35 = None
        mul_36 = chunk_8 * cos_2
        chunk_8 = cos_2 = None
        mul_37 = chunk_7 * sin_2
        chunk_7 = sin_2 = None
        add_13 = mul_36 + mul_37
        mul_36 = mul_37 = None
        rotary_alpha_1 = torch.cat([sub_6, add_13], dim=1)
        sub_6 = add_13 = None
        expand_4 = l_self_modules_layers_modules_1_modules_mega_layer_modules_rel_pos_bias_parameters_b_param_.expand(
            22, 64
        )
        l_self_modules_layers_modules_1_modules_mega_layer_modules_rel_pos_bias_parameters_b_param_ = (
            None
        )
        chunk_9 = torch.chunk(expand_4, 2, dim=-1)
        expand_4 = None
        chunk_10 = chunk_9[0]
        chunk_11 = chunk_9[1]
        chunk_9 = None
        to_9 = to_7.to(
            l_self_modules_layers_modules_1_modules_mega_layer_modules_rel_pos_bias_buffers_float_tensor_
        )
        to_7 = None
        to_10 = to_8.to(
            l_self_modules_layers_modules_1_modules_mega_layer_modules_rel_pos_bias_buffers_float_tensor_
        )
        to_8 = l_self_modules_layers_modules_1_modules_mega_layer_modules_rel_pos_bias_buffers_float_tensor_ = (None)
        sin_3 = to_9[slice(None, 22, None)]
        cos_3 = to_10[slice(None, 22, None)]
        mul_38 = chunk_10 * cos_3
        mul_39 = chunk_11 * sin_3
        sub_7 = mul_38 - mul_39
        mul_38 = mul_39 = None
        mul_40 = chunk_11 * cos_3
        chunk_11 = cos_3 = None
        mul_41 = chunk_10 * sin_3
        chunk_10 = sin_3 = None
        add_14 = mul_40 + mul_41
        mul_40 = mul_41 = None
        rotary_beta_1 = torch.cat([sub_7, add_14], dim=1)
        sub_7 = add_14 = None
        bias_1 = torch.functional.einsum("mk,nk->mn", rotary_alpha_1, rotary_beta_1)
        rotary_alpha_1 = rotary_beta_1 = None
        query_7 = query_6 * 0.125
        query_6 = None
        transpose_9 = key_5.transpose(2, 3)
        key_5 = None
        matmul_2 = torch.matmul(query_7, transpose_9)
        query_7 = transpose_9 = None
        qk_2 = matmul_2 + bias_1
        matmul_2 = bias_1 = None
        padding_mask_4 = 1 - padding_mask_3
        padding_mask_3 = None
        padding_mask_all_1 = padding_mask_4.all(dim=-1, keepdim=True)
        invert_1 = ~padding_mask_all_1
        padding_mask_all_1 = None
        padding_mask_5 = torch.logical_and(padding_mask_4, invert_1)
        padding_mask_4 = invert_1 = None
        unsqueeze_13 = padding_mask_5.unsqueeze(2)
        padding_mask_5 = None
        to_11 = unsqueeze_13.to(torch.bool)
        unsqueeze_13 = None
        qk_3 = qk_2.masked_fill(to_11, -inf)
        qk_2 = to_11 = None
        softmax_1 = torch.nn.functional.softmax(qk_3, -1, _stacklevel=5)
        attn_weights_1 = softmax_1.type_as(qk_3)
        softmax_1 = qk_3 = None
        value_7 = torch.nn.functional.dropout(value_6, p=0.1, training=False)
        value_6 = None
        kernel_5 = torch.nn.functional.dropout(attn_weights_1, p=0.1, training=False)
        attn_weights_1 = None
        matmul_3 = torch.matmul(kernel_5, value_7)
        kernel_5 = value_7 = None
        view_3 = matmul_3.view(1, 22, 37)
        matmul_3 = None
        weighted_self_output_3 = view_3.transpose(0, 1)
        view_3 = None
        mul_43 = weighted_self_output_3 * attention_gate_1
        weighted_self_output_3 = attention_gate_1 = None
        linear_7 = torch._C._nn.linear(
            mul_43,
            l_self_modules_layers_modules_1_modules_mega_layer_modules_h_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_mega_layer_modules_h_proj_parameters_bias_,
        )
        mul_43 = l_self_modules_layers_modules_1_modules_mega_layer_modules_h_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_mega_layer_modules_h_proj_parameters_bias_ = (None)
        add_16 = intermediate_state_1 + linear_7
        intermediate_state_1 = linear_7 = None
        weighted_self_output_4 = torch.nn.functional.silu(add_16, inplace=False)
        add_16 = None
        weighted_self_output_5 = torch.nn.functional.dropout(
            weighted_self_output_4, p=0.1, training=False
        )
        weighted_self_output_4 = None
        sub_9 = weighted_self_output_5 - output_4
        weighted_self_output_5 = None
        out_1 = torch.addcmul(output_4, residual_weight_3, sub_9)
        output_4 = residual_weight_3 = sub_9 = None
        square_3 = torch.square(out_1)
        mean_square_3 = torch.mean(square_3, dim=-1, keepdim=True)
        square_3 = None
        input_4 = (
            l_self_modules_layers_modules_1_modules_nffn_modules_norm_modules_norm_parameters_scalar_
            * out_1
        )
        l_self_modules_layers_modules_1_modules_nffn_modules_norm_modules_norm_parameters_scalar_ = (
            None
        )
        add_17 = mean_square_3 + 1e-05
        mean_square_3 = None
        rsqrt_3 = torch.rsqrt(add_17)
        add_17 = None
        output_6 = input_4 * rsqrt_3
        input_4 = rsqrt_3 = None
        linear_8 = torch._C._nn.linear(
            output_6,
            l_self_modules_layers_modules_1_modules_nffn_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_1_modules_nffn_modules_fc1_parameters_bias_,
        )
        output_6 = (
            l_self_modules_layers_modules_1_modules_nffn_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layers_modules_1_modules_nffn_modules_fc1_parameters_bias_
        ) = None
        hidden_2 = torch.nn.functional.silu(linear_8, inplace=False)
        linear_8 = None
        hidden_3 = torch.nn.functional.dropout(hidden_2, p=0.1, training=False)
        hidden_2 = None
        output_7 = torch._C._nn.linear(
            hidden_3,
            l_self_modules_layers_modules_1_modules_nffn_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_1_modules_nffn_modules_fc2_parameters_bias_,
        )
        hidden_3 = (
            l_self_modules_layers_modules_1_modules_nffn_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layers_modules_1_modules_nffn_modules_fc2_parameters_bias_
        ) = None
        output_8 = torch.nn.functional.dropout(output_7, p=0.1, training=False)
        output_7 = None
        output_9 = output_8 + out_1
        output_8 = out_1 = None
        square_4 = torch.square(output_9)
        mean_square_4 = torch.mean(square_4, dim=-1, keepdim=True)
        square_4 = None
        input_5 = (
            l_self_modules_layers_modules_2_modules_mega_layer_modules_norm_modules_norm_parameters_scalar_
            * output_9
        )
        l_self_modules_layers_modules_2_modules_mega_layer_modules_norm_modules_norm_parameters_scalar_ = (
            None
        )
        add_19 = mean_square_4 + 1e-05
        mean_square_4 = None
        rsqrt_4 = torch.rsqrt(add_19)
        add_19 = None
        output_10 = input_5 * rsqrt_4
        input_5 = rsqrt_4 = None
        linear_10 = torch._C._nn.linear(
            output_10,
            l_self_modules_layers_modules_2_modules_mega_layer_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_mega_layer_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_2_modules_mega_layer_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_mega_layer_modules_v_proj_parameters_bias_ = (None)
        value_8 = torch.nn.functional.silu(linear_10, inplace=False)
        linear_10 = None
        residual_2 = (
            output_10
            * l_self_modules_layers_modules_2_modules_mega_layer_modules_ema_gate_parameters_residual_weight_
        )
        l_self_modules_layers_modules_2_modules_mega_layer_modules_ema_gate_parameters_residual_weight_ = (
            None
        )
        inputs_4 = output_10.permute(1, 2, 0)
        output_10 = None
        unsqueeze_14 = l_attention_mask_.unsqueeze(1)
        type_as_6 = unsqueeze_14.type_as(inputs_4)
        unsqueeze_14 = None
        inputs_5 = inputs_4 * type_as_6
        inputs_4 = type_as_6 = None
        damping_factor_2 = torch.sigmoid(
            l_self_modules_layers_modules_2_modules_mega_layer_modules_ema_gate_parameters_damping_factor_
        )
        l_self_modules_layers_modules_2_modules_mega_layer_modules_ema_gate_parameters_damping_factor_ = (
            None
        )
        decay_factor_2 = torch.sigmoid(
            l_self_modules_layers_modules_2_modules_mega_layer_modules_ema_gate_parameters_decay_factor_
        )
        l_self_modules_layers_modules_2_modules_mega_layer_modules_ema_gate_parameters_decay_factor_ = (
            None
        )
        mul_50 = damping_factor_2 * decay_factor_2
        decay_factor_2 = None
        previous_timestep_weight_2 = 1.0 - mul_50
        mul_50 = None
        arange_2 = torch.arange(22)
        to_12 = arange_2.to(damping_factor_2)
        arange_2 = None
        view_4 = to_12.view(1, 1, 22)
        to_12 = None
        log_2 = torch.log(previous_timestep_weight_2)
        previous_timestep_weight_2 = None
        vander_2 = view_4 * log_2
        view_4 = log_2 = None
        mul_52 = (
            damping_factor_2
            * l_self_modules_layers_modules_2_modules_mega_layer_modules_ema_gate_parameters_ema_expansion_matrix_
        )
        damping_factor_2 = l_self_modules_layers_modules_2_modules_mega_layer_modules_ema_gate_parameters_ema_expansion_matrix_ = (None)
        exp_2 = torch.exp(vander_2)
        vander_2 = None
        kernel_6 = mul_52 * exp_2
        mul_52 = exp_2 = None
        mul_54 = (
            l_self_modules_layers_modules_2_modules_mega_layer_modules_ema_gate_parameters_kernel_projection_matrix_
            * 0.25
        )
        l_self_modules_layers_modules_2_modules_mega_layer_modules_ema_gate_parameters_kernel_projection_matrix_ = (
            None
        )
        einsum_4 = torch.functional.einsum("dnl,dn->dl", kernel_6, mul_54)
        kernel_6 = mul_54 = None
        kernel_7 = einsum_4[(Ellipsis, slice(None, 22, None))]
        float_5 = inputs_5.float()
        inputs_fft_2 = torch._C._fft.fft_rfft(float_5, n=44)
        float_5 = None
        float_6 = kernel_7.float()
        kernel_7 = None
        kernel_fft_2 = torch._C._fft.fft_rfft(float_6, n=44)
        float_6 = None
        mul_55 = inputs_fft_2 * kernel_fft_2
        inputs_fft_2 = kernel_fft_2 = None
        convolved_sequence_2 = torch._C._fft.fft_irfft(mul_55, n=44)
        mul_55 = None
        ema_output_4 = convolved_sequence_2[(Ellipsis, slice(0, 22, None))]
        convolved_sequence_2 = None
        ema_output_5 = ema_output_4.type_as(inputs_5)
        ema_output_4 = inputs_5 = None
        permute_5 = ema_output_5.permute(2, 0, 1)
        ema_output_5 = None
        add_20 = permute_5 + residual_2
        permute_5 = residual_2 = None
        gated_ema_output_2 = torch.nn.functional.silu(add_20)
        add_20 = None
        ema_out_2 = torch.nn.functional.dropout(
            gated_ema_output_2, p=0.1, training=False
        )
        gated_ema_output_2 = None
        base_2 = torch._C._nn.linear(
            ema_out_2,
            l_self_modules_layers_modules_2_modules_mega_layer_modules_mx_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_mega_layer_modules_mx_proj_parameters_bias_,
        )
        ema_out_2 = l_self_modules_layers_modules_2_modules_mega_layer_modules_mx_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_mega_layer_modules_mx_proj_parameters_bias_ = (None)
        split_4 = torch.functional.split(base_2, [32, 101, 32], dim=-1)
        base_2 = None
        residual_weight_4 = split_4[0]
        query_key_gates_4 = split_4[1]
        intermediate_state_2 = split_4[2]
        split_4 = None
        residual_weight_5 = torch.sigmoid(residual_weight_4)
        residual_weight_4 = None
        query_key_gates_5 = torch.nn.functional.silu(query_key_gates_4)
        query_key_gates_4 = None
        split_5 = torch.functional.split(query_key_gates_5, [64, 37], dim=-1)
        query_key_gates_5 = None
        query_key_4 = split_5[0]
        attention_gate_2 = split_5[1]
        split_5 = None
        unsqueeze_15 = query_key_4.unsqueeze(2)
        query_key_4 = None
        mul_56 = (
            unsqueeze_15
            * l_self_modules_layers_modules_2_modules_mega_layer_parameters_qk_weight_
        )
        unsqueeze_15 = (
            l_self_modules_layers_modules_2_modules_mega_layer_parameters_qk_weight_
        ) = None
        query_key_5 = (
            mul_56
            + l_self_modules_layers_modules_2_modules_mega_layer_parameters_qk_bias_
        )
        mul_56 = (
            l_self_modules_layers_modules_2_modules_mega_layer_parameters_qk_bias_
        ) = None
        unbind_2 = torch.unbind(query_key_5, dim=2)
        query_key_5 = None
        query_8 = unbind_2[0]
        key_6 = unbind_2[1]
        unbind_2 = None
        query_9 = query_8.transpose(0, 1)
        query_8 = None
        key_7 = key_6.transpose(0, 1)
        key_6 = None
        value_9 = value_8.transpose(0, 1)
        value_8 = None
        query_10 = query_9.unsqueeze(1)
        query_9 = None
        key_8 = key_7.unsqueeze(1)
        key_7 = None
        value_10 = value_9.unsqueeze(1)
        value_9 = None
        padding_mask_6 = l_attention_mask_.unsqueeze(1)
        expand_5 = l_self_modules_layers_modules_2_modules_mega_layer_modules_rel_pos_bias_parameters_alpha_.expand(
            22, 64
        )
        l_self_modules_layers_modules_2_modules_mega_layer_modules_rel_pos_bias_parameters_alpha_ = (
            None
        )
        chunk_12 = torch.chunk(expand_5, 2, dim=-1)
        expand_5 = None
        chunk_13 = chunk_12[0]
        chunk_14 = chunk_12[1]
        chunk_12 = None
        to_13 = l_self_modules_layers_modules_2_modules_mega_layer_modules_rel_pos_bias_sine.to(
            l_self_modules_layers_modules_2_modules_mega_layer_modules_rel_pos_bias_buffers_float_tensor_
        )
        l_self_modules_layers_modules_2_modules_mega_layer_modules_rel_pos_bias_sine = (
            None
        )
        to_14 = l_self_modules_layers_modules_2_modules_mega_layer_modules_rel_pos_bias_cosine.to(
            l_self_modules_layers_modules_2_modules_mega_layer_modules_rel_pos_bias_buffers_float_tensor_
        )
        l_self_modules_layers_modules_2_modules_mega_layer_modules_rel_pos_bias_cosine = (
            None
        )
        sin_4 = to_13[slice(None, 22, None)]
        cos_4 = to_14[slice(None, 22, None)]
        mul_57 = chunk_13 * cos_4
        mul_58 = chunk_14 * sin_4
        sub_11 = mul_57 - mul_58
        mul_57 = mul_58 = None
        mul_59 = chunk_14 * cos_4
        chunk_14 = cos_4 = None
        mul_60 = chunk_13 * sin_4
        chunk_13 = sin_4 = None
        add_22 = mul_59 + mul_60
        mul_59 = mul_60 = None
        rotary_alpha_2 = torch.cat([sub_11, add_22], dim=1)
        sub_11 = add_22 = None
        expand_6 = l_self_modules_layers_modules_2_modules_mega_layer_modules_rel_pos_bias_parameters_b_param_.expand(
            22, 64
        )
        l_self_modules_layers_modules_2_modules_mega_layer_modules_rel_pos_bias_parameters_b_param_ = (
            None
        )
        chunk_15 = torch.chunk(expand_6, 2, dim=-1)
        expand_6 = None
        chunk_16 = chunk_15[0]
        chunk_17 = chunk_15[1]
        chunk_15 = None
        to_15 = to_13.to(
            l_self_modules_layers_modules_2_modules_mega_layer_modules_rel_pos_bias_buffers_float_tensor_
        )
        to_13 = None
        to_16 = to_14.to(
            l_self_modules_layers_modules_2_modules_mega_layer_modules_rel_pos_bias_buffers_float_tensor_
        )
        to_14 = l_self_modules_layers_modules_2_modules_mega_layer_modules_rel_pos_bias_buffers_float_tensor_ = (None)
        sin_5 = to_15[slice(None, 22, None)]
        cos_5 = to_16[slice(None, 22, None)]
        mul_61 = chunk_16 * cos_5
        mul_62 = chunk_17 * sin_5
        sub_12 = mul_61 - mul_62
        mul_61 = mul_62 = None
        mul_63 = chunk_17 * cos_5
        chunk_17 = cos_5 = None
        mul_64 = chunk_16 * sin_5
        chunk_16 = sin_5 = None
        add_23 = mul_63 + mul_64
        mul_63 = mul_64 = None
        rotary_beta_2 = torch.cat([sub_12, add_23], dim=1)
        sub_12 = add_23 = None
        bias_2 = torch.functional.einsum("mk,nk->mn", rotary_alpha_2, rotary_beta_2)
        rotary_alpha_2 = rotary_beta_2 = None
        query_11 = query_10 * 0.125
        query_10 = None
        transpose_14 = key_8.transpose(2, 3)
        key_8 = None
        matmul_4 = torch.matmul(query_11, transpose_14)
        query_11 = transpose_14 = None
        qk_4 = matmul_4 + bias_2
        matmul_4 = bias_2 = None
        padding_mask_7 = 1 - padding_mask_6
        padding_mask_6 = None
        padding_mask_all_2 = padding_mask_7.all(dim=-1, keepdim=True)
        invert_2 = ~padding_mask_all_2
        padding_mask_all_2 = None
        padding_mask_8 = torch.logical_and(padding_mask_7, invert_2)
        padding_mask_7 = invert_2 = None
        unsqueeze_20 = padding_mask_8.unsqueeze(2)
        padding_mask_8 = None
        to_17 = unsqueeze_20.to(torch.bool)
        unsqueeze_20 = None
        qk_5 = qk_4.masked_fill(to_17, -inf)
        qk_4 = to_17 = None
        softmax_2 = torch.nn.functional.softmax(qk_5, -1, _stacklevel=5)
        attn_weights_2 = softmax_2.type_as(qk_5)
        softmax_2 = qk_5 = None
        value_11 = torch.nn.functional.dropout(value_10, p=0.1, training=False)
        value_10 = None
        kernel_8 = torch.nn.functional.dropout(attn_weights_2, p=0.1, training=False)
        attn_weights_2 = None
        matmul_5 = torch.matmul(kernel_8, value_11)
        kernel_8 = value_11 = None
        view_5 = matmul_5.view(1, 22, 37)
        matmul_5 = None
        weighted_self_output_6 = view_5.transpose(0, 1)
        view_5 = None
        mul_66 = weighted_self_output_6 * attention_gate_2
        weighted_self_output_6 = attention_gate_2 = None
        linear_12 = torch._C._nn.linear(
            mul_66,
            l_self_modules_layers_modules_2_modules_mega_layer_modules_h_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_mega_layer_modules_h_proj_parameters_bias_,
        )
        mul_66 = l_self_modules_layers_modules_2_modules_mega_layer_modules_h_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_mega_layer_modules_h_proj_parameters_bias_ = (None)
        add_25 = intermediate_state_2 + linear_12
        intermediate_state_2 = linear_12 = None
        weighted_self_output_7 = torch.nn.functional.silu(add_25, inplace=False)
        add_25 = None
        weighted_self_output_8 = torch.nn.functional.dropout(
            weighted_self_output_7, p=0.1, training=False
        )
        weighted_self_output_7 = None
        sub_14 = weighted_self_output_8 - output_9
        weighted_self_output_8 = None
        out_2 = torch.addcmul(output_9, residual_weight_5, sub_14)
        output_9 = residual_weight_5 = sub_14 = None
        square_5 = torch.square(out_2)
        mean_square_5 = torch.mean(square_5, dim=-1, keepdim=True)
        square_5 = None
        input_6 = (
            l_self_modules_layers_modules_2_modules_nffn_modules_norm_modules_norm_parameters_scalar_
            * out_2
        )
        l_self_modules_layers_modules_2_modules_nffn_modules_norm_modules_norm_parameters_scalar_ = (
            None
        )
        add_26 = mean_square_5 + 1e-05
        mean_square_5 = None
        rsqrt_5 = torch.rsqrt(add_26)
        add_26 = None
        output_11 = input_6 * rsqrt_5
        input_6 = rsqrt_5 = None
        linear_13 = torch._C._nn.linear(
            output_11,
            l_self_modules_layers_modules_2_modules_nffn_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_nffn_modules_fc1_parameters_bias_,
        )
        output_11 = (
            l_self_modules_layers_modules_2_modules_nffn_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layers_modules_2_modules_nffn_modules_fc1_parameters_bias_
        ) = None
        hidden_4 = torch.nn.functional.silu(linear_13, inplace=False)
        linear_13 = None
        hidden_5 = torch.nn.functional.dropout(hidden_4, p=0.1, training=False)
        hidden_4 = None
        output_12 = torch._C._nn.linear(
            hidden_5,
            l_self_modules_layers_modules_2_modules_nffn_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_nffn_modules_fc2_parameters_bias_,
        )
        hidden_5 = (
            l_self_modules_layers_modules_2_modules_nffn_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layers_modules_2_modules_nffn_modules_fc2_parameters_bias_
        ) = None
        output_13 = torch.nn.functional.dropout(output_12, p=0.1, training=False)
        output_12 = None
        output_14 = output_13 + out_2
        output_13 = out_2 = None
        square_6 = torch.square(output_14)
        mean_square_6 = torch.mean(square_6, dim=-1, keepdim=True)
        square_6 = None
        input_7 = (
            l_self_modules_layers_modules_3_modules_mega_layer_modules_norm_modules_norm_parameters_scalar_
            * output_14
        )
        l_self_modules_layers_modules_3_modules_mega_layer_modules_norm_modules_norm_parameters_scalar_ = (
            None
        )
        add_28 = mean_square_6 + 1e-05
        mean_square_6 = None
        rsqrt_6 = torch.rsqrt(add_28)
        add_28 = None
        output_15 = input_7 * rsqrt_6
        input_7 = rsqrt_6 = None
        linear_15 = torch._C._nn.linear(
            output_15,
            l_self_modules_layers_modules_3_modules_mega_layer_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_3_modules_mega_layer_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_3_modules_mega_layer_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_3_modules_mega_layer_modules_v_proj_parameters_bias_ = (None)
        value_12 = torch.nn.functional.silu(linear_15, inplace=False)
        linear_15 = None
        residual_3 = (
            output_15
            * l_self_modules_layers_modules_3_modules_mega_layer_modules_ema_gate_parameters_residual_weight_
        )
        l_self_modules_layers_modules_3_modules_mega_layer_modules_ema_gate_parameters_residual_weight_ = (
            None
        )
        inputs_6 = output_15.permute(1, 2, 0)
        output_15 = None
        unsqueeze_21 = l_attention_mask_.unsqueeze(1)
        type_as_9 = unsqueeze_21.type_as(inputs_6)
        unsqueeze_21 = None
        inputs_7 = inputs_6 * type_as_9
        inputs_6 = type_as_9 = None
        damping_factor_3 = torch.sigmoid(
            l_self_modules_layers_modules_3_modules_mega_layer_modules_ema_gate_parameters_damping_factor_
        )
        l_self_modules_layers_modules_3_modules_mega_layer_modules_ema_gate_parameters_damping_factor_ = (
            None
        )
        decay_factor_3 = torch.sigmoid(
            l_self_modules_layers_modules_3_modules_mega_layer_modules_ema_gate_parameters_decay_factor_
        )
        l_self_modules_layers_modules_3_modules_mega_layer_modules_ema_gate_parameters_decay_factor_ = (
            None
        )
        mul_73 = damping_factor_3 * decay_factor_3
        decay_factor_3 = None
        previous_timestep_weight_3 = 1.0 - mul_73
        mul_73 = None
        arange_3 = torch.arange(22)
        to_18 = arange_3.to(damping_factor_3)
        arange_3 = None
        view_6 = to_18.view(1, 1, 22)
        to_18 = None
        log_3 = torch.log(previous_timestep_weight_3)
        previous_timestep_weight_3 = None
        vander_3 = view_6 * log_3
        view_6 = log_3 = None
        mul_75 = (
            damping_factor_3
            * l_self_modules_layers_modules_3_modules_mega_layer_modules_ema_gate_parameters_ema_expansion_matrix_
        )
        damping_factor_3 = l_self_modules_layers_modules_3_modules_mega_layer_modules_ema_gate_parameters_ema_expansion_matrix_ = (None)
        exp_3 = torch.exp(vander_3)
        vander_3 = None
        kernel_9 = mul_75 * exp_3
        mul_75 = exp_3 = None
        mul_77 = (
            l_self_modules_layers_modules_3_modules_mega_layer_modules_ema_gate_parameters_kernel_projection_matrix_
            * 0.25
        )
        l_self_modules_layers_modules_3_modules_mega_layer_modules_ema_gate_parameters_kernel_projection_matrix_ = (
            None
        )
        einsum_6 = torch.functional.einsum("dnl,dn->dl", kernel_9, mul_77)
        kernel_9 = mul_77 = None
        kernel_10 = einsum_6[(Ellipsis, slice(None, 22, None))]
        float_7 = inputs_7.float()
        inputs_fft_3 = torch._C._fft.fft_rfft(float_7, n=44)
        float_7 = None
        float_8 = kernel_10.float()
        kernel_10 = None
        kernel_fft_3 = torch._C._fft.fft_rfft(float_8, n=44)
        float_8 = None
        mul_78 = inputs_fft_3 * kernel_fft_3
        inputs_fft_3 = kernel_fft_3 = None
        convolved_sequence_3 = torch._C._fft.fft_irfft(mul_78, n=44)
        mul_78 = None
        ema_output_6 = convolved_sequence_3[(Ellipsis, slice(0, 22, None))]
        convolved_sequence_3 = None
        ema_output_7 = ema_output_6.type_as(inputs_7)
        ema_output_6 = inputs_7 = None
        permute_7 = ema_output_7.permute(2, 0, 1)
        ema_output_7 = None
        add_29 = permute_7 + residual_3
        permute_7 = residual_3 = None
        gated_ema_output_3 = torch.nn.functional.silu(add_29)
        add_29 = None
        ema_out_3 = torch.nn.functional.dropout(
            gated_ema_output_3, p=0.1, training=False
        )
        gated_ema_output_3 = None
        base_3 = torch._C._nn.linear(
            ema_out_3,
            l_self_modules_layers_modules_3_modules_mega_layer_modules_mx_proj_parameters_weight_,
            l_self_modules_layers_modules_3_modules_mega_layer_modules_mx_proj_parameters_bias_,
        )
        ema_out_3 = l_self_modules_layers_modules_3_modules_mega_layer_modules_mx_proj_parameters_weight_ = l_self_modules_layers_modules_3_modules_mega_layer_modules_mx_proj_parameters_bias_ = (None)
        split_6 = torch.functional.split(base_3, [32, 101, 32], dim=-1)
        base_3 = None
        residual_weight_6 = split_6[0]
        query_key_gates_6 = split_6[1]
        intermediate_state_3 = split_6[2]
        split_6 = None
        residual_weight_7 = torch.sigmoid(residual_weight_6)
        residual_weight_6 = None
        query_key_gates_7 = torch.nn.functional.silu(query_key_gates_6)
        query_key_gates_6 = None
        split_7 = torch.functional.split(query_key_gates_7, [64, 37], dim=-1)
        query_key_gates_7 = None
        query_key_6 = split_7[0]
        attention_gate_3 = split_7[1]
        split_7 = None
        unsqueeze_22 = query_key_6.unsqueeze(2)
        query_key_6 = None
        mul_79 = (
            unsqueeze_22
            * l_self_modules_layers_modules_3_modules_mega_layer_parameters_qk_weight_
        )
        unsqueeze_22 = (
            l_self_modules_layers_modules_3_modules_mega_layer_parameters_qk_weight_
        ) = None
        query_key_7 = (
            mul_79
            + l_self_modules_layers_modules_3_modules_mega_layer_parameters_qk_bias_
        )
        mul_79 = (
            l_self_modules_layers_modules_3_modules_mega_layer_parameters_qk_bias_
        ) = None
        unbind_3 = torch.unbind(query_key_7, dim=2)
        query_key_7 = None
        query_12 = unbind_3[0]
        key_9 = unbind_3[1]
        unbind_3 = None
        query_13 = query_12.transpose(0, 1)
        query_12 = None
        key_10 = key_9.transpose(0, 1)
        key_9 = None
        value_13 = value_12.transpose(0, 1)
        value_12 = None
        query_14 = query_13.unsqueeze(1)
        query_13 = None
        key_11 = key_10.unsqueeze(1)
        key_10 = None
        value_14 = value_13.unsqueeze(1)
        value_13 = None
        padding_mask_9 = l_attention_mask_.unsqueeze(1)
        expand_7 = l_self_modules_layers_modules_3_modules_mega_layer_modules_rel_pos_bias_parameters_alpha_.expand(
            22, 64
        )
        l_self_modules_layers_modules_3_modules_mega_layer_modules_rel_pos_bias_parameters_alpha_ = (
            None
        )
        chunk_18 = torch.chunk(expand_7, 2, dim=-1)
        expand_7 = None
        chunk_19 = chunk_18[0]
        chunk_20 = chunk_18[1]
        chunk_18 = None
        to_19 = l_self_modules_layers_modules_3_modules_mega_layer_modules_rel_pos_bias_sine.to(
            l_self_modules_layers_modules_3_modules_mega_layer_modules_rel_pos_bias_buffers_float_tensor_
        )
        l_self_modules_layers_modules_3_modules_mega_layer_modules_rel_pos_bias_sine = (
            None
        )
        to_20 = l_self_modules_layers_modules_3_modules_mega_layer_modules_rel_pos_bias_cosine.to(
            l_self_modules_layers_modules_3_modules_mega_layer_modules_rel_pos_bias_buffers_float_tensor_
        )
        l_self_modules_layers_modules_3_modules_mega_layer_modules_rel_pos_bias_cosine = (
            None
        )
        sin_6 = to_19[slice(None, 22, None)]
        cos_6 = to_20[slice(None, 22, None)]
        mul_80 = chunk_19 * cos_6
        mul_81 = chunk_20 * sin_6
        sub_16 = mul_80 - mul_81
        mul_80 = mul_81 = None
        mul_82 = chunk_20 * cos_6
        chunk_20 = cos_6 = None
        mul_83 = chunk_19 * sin_6
        chunk_19 = sin_6 = None
        add_31 = mul_82 + mul_83
        mul_82 = mul_83 = None
        rotary_alpha_3 = torch.cat([sub_16, add_31], dim=1)
        sub_16 = add_31 = None
        expand_8 = l_self_modules_layers_modules_3_modules_mega_layer_modules_rel_pos_bias_parameters_b_param_.expand(
            22, 64
        )
        l_self_modules_layers_modules_3_modules_mega_layer_modules_rel_pos_bias_parameters_b_param_ = (
            None
        )
        chunk_21 = torch.chunk(expand_8, 2, dim=-1)
        expand_8 = None
        chunk_22 = chunk_21[0]
        chunk_23 = chunk_21[1]
        chunk_21 = None
        to_21 = to_19.to(
            l_self_modules_layers_modules_3_modules_mega_layer_modules_rel_pos_bias_buffers_float_tensor_
        )
        to_19 = None
        to_22 = to_20.to(
            l_self_modules_layers_modules_3_modules_mega_layer_modules_rel_pos_bias_buffers_float_tensor_
        )
        to_20 = l_self_modules_layers_modules_3_modules_mega_layer_modules_rel_pos_bias_buffers_float_tensor_ = (None)
        sin_7 = to_21[slice(None, 22, None)]
        cos_7 = to_22[slice(None, 22, None)]
        mul_84 = chunk_22 * cos_7
        mul_85 = chunk_23 * sin_7
        sub_17 = mul_84 - mul_85
        mul_84 = mul_85 = None
        mul_86 = chunk_23 * cos_7
        chunk_23 = cos_7 = None
        mul_87 = chunk_22 * sin_7
        chunk_22 = sin_7 = None
        add_32 = mul_86 + mul_87
        mul_86 = mul_87 = None
        rotary_beta_3 = torch.cat([sub_17, add_32], dim=1)
        sub_17 = add_32 = None
        bias_3 = torch.functional.einsum("mk,nk->mn", rotary_alpha_3, rotary_beta_3)
        rotary_alpha_3 = rotary_beta_3 = None
        query_15 = query_14 * 0.125
        query_14 = None
        transpose_19 = key_11.transpose(2, 3)
        key_11 = None
        matmul_6 = torch.matmul(query_15, transpose_19)
        query_15 = transpose_19 = None
        qk_6 = matmul_6 + bias_3
        matmul_6 = bias_3 = None
        padding_mask_10 = 1 - padding_mask_9
        padding_mask_9 = None
        padding_mask_all_3 = padding_mask_10.all(dim=-1, keepdim=True)
        invert_3 = ~padding_mask_all_3
        padding_mask_all_3 = None
        padding_mask_11 = torch.logical_and(padding_mask_10, invert_3)
        padding_mask_10 = invert_3 = None
        unsqueeze_27 = padding_mask_11.unsqueeze(2)
        padding_mask_11 = None
        to_23 = unsqueeze_27.to(torch.bool)
        unsqueeze_27 = None
        qk_7 = qk_6.masked_fill(to_23, -inf)
        qk_6 = to_23 = None
        softmax_3 = torch.nn.functional.softmax(qk_7, -1, _stacklevel=5)
        attn_weights_3 = softmax_3.type_as(qk_7)
        softmax_3 = qk_7 = None
        value_15 = torch.nn.functional.dropout(value_14, p=0.1, training=False)
        value_14 = None
        kernel_11 = torch.nn.functional.dropout(attn_weights_3, p=0.1, training=False)
        attn_weights_3 = None
        matmul_7 = torch.matmul(kernel_11, value_15)
        kernel_11 = value_15 = None
        view_7 = matmul_7.view(1, 22, 37)
        matmul_7 = None
        weighted_self_output_9 = view_7.transpose(0, 1)
        view_7 = None
        mul_89 = weighted_self_output_9 * attention_gate_3
        weighted_self_output_9 = attention_gate_3 = None
        linear_17 = torch._C._nn.linear(
            mul_89,
            l_self_modules_layers_modules_3_modules_mega_layer_modules_h_proj_parameters_weight_,
            l_self_modules_layers_modules_3_modules_mega_layer_modules_h_proj_parameters_bias_,
        )
        mul_89 = l_self_modules_layers_modules_3_modules_mega_layer_modules_h_proj_parameters_weight_ = l_self_modules_layers_modules_3_modules_mega_layer_modules_h_proj_parameters_bias_ = (None)
        add_34 = intermediate_state_3 + linear_17
        intermediate_state_3 = linear_17 = None
        weighted_self_output_10 = torch.nn.functional.silu(add_34, inplace=False)
        add_34 = None
        weighted_self_output_11 = torch.nn.functional.dropout(
            weighted_self_output_10, p=0.1, training=False
        )
        weighted_self_output_10 = None
        sub_19 = weighted_self_output_11 - output_14
        weighted_self_output_11 = None
        out_3 = torch.addcmul(output_14, residual_weight_7, sub_19)
        output_14 = residual_weight_7 = sub_19 = None
        square_7 = torch.square(out_3)
        mean_square_7 = torch.mean(square_7, dim=-1, keepdim=True)
        square_7 = None
        input_8 = (
            l_self_modules_layers_modules_3_modules_nffn_modules_norm_modules_norm_parameters_scalar_
            * out_3
        )
        l_self_modules_layers_modules_3_modules_nffn_modules_norm_modules_norm_parameters_scalar_ = (
            None
        )
        add_35 = mean_square_7 + 1e-05
        mean_square_7 = None
        rsqrt_7 = torch.rsqrt(add_35)
        add_35 = None
        output_16 = input_8 * rsqrt_7
        input_8 = rsqrt_7 = None
        linear_18 = torch._C._nn.linear(
            output_16,
            l_self_modules_layers_modules_3_modules_nffn_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_3_modules_nffn_modules_fc1_parameters_bias_,
        )
        output_16 = (
            l_self_modules_layers_modules_3_modules_nffn_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layers_modules_3_modules_nffn_modules_fc1_parameters_bias_
        ) = None
        hidden_6 = torch.nn.functional.silu(linear_18, inplace=False)
        linear_18 = None
        hidden_7 = torch.nn.functional.dropout(hidden_6, p=0.1, training=False)
        hidden_6 = None
        output_17 = torch._C._nn.linear(
            hidden_7,
            l_self_modules_layers_modules_3_modules_nffn_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_3_modules_nffn_modules_fc2_parameters_bias_,
        )
        hidden_7 = (
            l_self_modules_layers_modules_3_modules_nffn_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layers_modules_3_modules_nffn_modules_fc2_parameters_bias_
        ) = None
        output_18 = torch.nn.functional.dropout(output_17, p=0.1, training=False)
        output_17 = None
        output_19 = output_18 + out_3
        output_18 = out_3 = None
        square_8 = torch.square(output_19)
        mean_square_8 = torch.mean(square_8, dim=-1, keepdim=True)
        square_8 = None
        input_9 = (
            l_self_modules_layers_modules_4_modules_mega_layer_modules_norm_modules_norm_parameters_scalar_
            * output_19
        )
        l_self_modules_layers_modules_4_modules_mega_layer_modules_norm_modules_norm_parameters_scalar_ = (
            None
        )
        add_37 = mean_square_8 + 1e-05
        mean_square_8 = None
        rsqrt_8 = torch.rsqrt(add_37)
        add_37 = None
        output_20 = input_9 * rsqrt_8
        input_9 = rsqrt_8 = None
        linear_20 = torch._C._nn.linear(
            output_20,
            l_self_modules_layers_modules_4_modules_mega_layer_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_4_modules_mega_layer_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_4_modules_mega_layer_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_4_modules_mega_layer_modules_v_proj_parameters_bias_ = (None)
        value_16 = torch.nn.functional.silu(linear_20, inplace=False)
        linear_20 = None
        residual_4 = (
            output_20
            * l_self_modules_layers_modules_4_modules_mega_layer_modules_ema_gate_parameters_residual_weight_
        )
        l_self_modules_layers_modules_4_modules_mega_layer_modules_ema_gate_parameters_residual_weight_ = (
            None
        )
        inputs_8 = output_20.permute(1, 2, 0)
        output_20 = None
        unsqueeze_28 = l_attention_mask_.unsqueeze(1)
        type_as_12 = unsqueeze_28.type_as(inputs_8)
        unsqueeze_28 = None
        inputs_9 = inputs_8 * type_as_12
        inputs_8 = type_as_12 = None
        damping_factor_4 = torch.sigmoid(
            l_self_modules_layers_modules_4_modules_mega_layer_modules_ema_gate_parameters_damping_factor_
        )
        l_self_modules_layers_modules_4_modules_mega_layer_modules_ema_gate_parameters_damping_factor_ = (
            None
        )
        decay_factor_4 = torch.sigmoid(
            l_self_modules_layers_modules_4_modules_mega_layer_modules_ema_gate_parameters_decay_factor_
        )
        l_self_modules_layers_modules_4_modules_mega_layer_modules_ema_gate_parameters_decay_factor_ = (
            None
        )
        mul_96 = damping_factor_4 * decay_factor_4
        decay_factor_4 = None
        previous_timestep_weight_4 = 1.0 - mul_96
        mul_96 = None
        arange_4 = torch.arange(22)
        to_24 = arange_4.to(damping_factor_4)
        arange_4 = None
        view_8 = to_24.view(1, 1, 22)
        to_24 = None
        log_4 = torch.log(previous_timestep_weight_4)
        previous_timestep_weight_4 = None
        vander_4 = view_8 * log_4
        view_8 = log_4 = None
        mul_98 = (
            damping_factor_4
            * l_self_modules_layers_modules_4_modules_mega_layer_modules_ema_gate_parameters_ema_expansion_matrix_
        )
        damping_factor_4 = l_self_modules_layers_modules_4_modules_mega_layer_modules_ema_gate_parameters_ema_expansion_matrix_ = (None)
        exp_4 = torch.exp(vander_4)
        vander_4 = None
        kernel_12 = mul_98 * exp_4
        mul_98 = exp_4 = None
        mul_100 = (
            l_self_modules_layers_modules_4_modules_mega_layer_modules_ema_gate_parameters_kernel_projection_matrix_
            * 0.25
        )
        l_self_modules_layers_modules_4_modules_mega_layer_modules_ema_gate_parameters_kernel_projection_matrix_ = (
            None
        )
        einsum_8 = torch.functional.einsum("dnl,dn->dl", kernel_12, mul_100)
        kernel_12 = mul_100 = None
        kernel_13 = einsum_8[(Ellipsis, slice(None, 22, None))]
        float_9 = inputs_9.float()
        inputs_fft_4 = torch._C._fft.fft_rfft(float_9, n=44)
        float_9 = None
        float_10 = kernel_13.float()
        kernel_13 = None
        kernel_fft_4 = torch._C._fft.fft_rfft(float_10, n=44)
        float_10 = None
        mul_101 = inputs_fft_4 * kernel_fft_4
        inputs_fft_4 = kernel_fft_4 = None
        convolved_sequence_4 = torch._C._fft.fft_irfft(mul_101, n=44)
        mul_101 = None
        ema_output_8 = convolved_sequence_4[(Ellipsis, slice(0, 22, None))]
        convolved_sequence_4 = None
        ema_output_9 = ema_output_8.type_as(inputs_9)
        ema_output_8 = inputs_9 = None
        permute_9 = ema_output_9.permute(2, 0, 1)
        ema_output_9 = None
        add_38 = permute_9 + residual_4
        permute_9 = residual_4 = None
        gated_ema_output_4 = torch.nn.functional.silu(add_38)
        add_38 = None
        ema_out_4 = torch.nn.functional.dropout(
            gated_ema_output_4, p=0.1, training=False
        )
        gated_ema_output_4 = None
        base_4 = torch._C._nn.linear(
            ema_out_4,
            l_self_modules_layers_modules_4_modules_mega_layer_modules_mx_proj_parameters_weight_,
            l_self_modules_layers_modules_4_modules_mega_layer_modules_mx_proj_parameters_bias_,
        )
        ema_out_4 = l_self_modules_layers_modules_4_modules_mega_layer_modules_mx_proj_parameters_weight_ = l_self_modules_layers_modules_4_modules_mega_layer_modules_mx_proj_parameters_bias_ = (None)
        split_8 = torch.functional.split(base_4, [32, 101, 32], dim=-1)
        base_4 = None
        residual_weight_8 = split_8[0]
        query_key_gates_8 = split_8[1]
        intermediate_state_4 = split_8[2]
        split_8 = None
        residual_weight_9 = torch.sigmoid(residual_weight_8)
        residual_weight_8 = None
        query_key_gates_9 = torch.nn.functional.silu(query_key_gates_8)
        query_key_gates_8 = None
        split_9 = torch.functional.split(query_key_gates_9, [64, 37], dim=-1)
        query_key_gates_9 = None
        query_key_8 = split_9[0]
        attention_gate_4 = split_9[1]
        split_9 = None
        unsqueeze_29 = query_key_8.unsqueeze(2)
        query_key_8 = None
        mul_102 = (
            unsqueeze_29
            * l_self_modules_layers_modules_4_modules_mega_layer_parameters_qk_weight_
        )
        unsqueeze_29 = (
            l_self_modules_layers_modules_4_modules_mega_layer_parameters_qk_weight_
        ) = None
        query_key_9 = (
            mul_102
            + l_self_modules_layers_modules_4_modules_mega_layer_parameters_qk_bias_
        )
        mul_102 = (
            l_self_modules_layers_modules_4_modules_mega_layer_parameters_qk_bias_
        ) = None
        unbind_4 = torch.unbind(query_key_9, dim=2)
        query_key_9 = None
        query_16 = unbind_4[0]
        key_12 = unbind_4[1]
        unbind_4 = None
        query_17 = query_16.transpose(0, 1)
        query_16 = None
        key_13 = key_12.transpose(0, 1)
        key_12 = None
        value_17 = value_16.transpose(0, 1)
        value_16 = None
        query_18 = query_17.unsqueeze(1)
        query_17 = None
        key_14 = key_13.unsqueeze(1)
        key_13 = None
        value_18 = value_17.unsqueeze(1)
        value_17 = None
        padding_mask_12 = l_attention_mask_.unsqueeze(1)
        l_attention_mask_ = None
        expand_9 = l_self_modules_layers_modules_4_modules_mega_layer_modules_rel_pos_bias_parameters_alpha_.expand(
            22, 64
        )
        l_self_modules_layers_modules_4_modules_mega_layer_modules_rel_pos_bias_parameters_alpha_ = (
            None
        )
        chunk_24 = torch.chunk(expand_9, 2, dim=-1)
        expand_9 = None
        chunk_25 = chunk_24[0]
        chunk_26 = chunk_24[1]
        chunk_24 = None
        to_25 = l_self_modules_layers_modules_4_modules_mega_layer_modules_rel_pos_bias_sine.to(
            l_self_modules_layers_modules_4_modules_mega_layer_modules_rel_pos_bias_buffers_float_tensor_
        )
        l_self_modules_layers_modules_4_modules_mega_layer_modules_rel_pos_bias_sine = (
            None
        )
        to_26 = l_self_modules_layers_modules_4_modules_mega_layer_modules_rel_pos_bias_cosine.to(
            l_self_modules_layers_modules_4_modules_mega_layer_modules_rel_pos_bias_buffers_float_tensor_
        )
        l_self_modules_layers_modules_4_modules_mega_layer_modules_rel_pos_bias_cosine = (
            None
        )
        sin_8 = to_25[slice(None, 22, None)]
        cos_8 = to_26[slice(None, 22, None)]
        mul_103 = chunk_25 * cos_8
        mul_104 = chunk_26 * sin_8
        sub_21 = mul_103 - mul_104
        mul_103 = mul_104 = None
        mul_105 = chunk_26 * cos_8
        chunk_26 = cos_8 = None
        mul_106 = chunk_25 * sin_8
        chunk_25 = sin_8 = None
        add_40 = mul_105 + mul_106
        mul_105 = mul_106 = None
        rotary_alpha_4 = torch.cat([sub_21, add_40], dim=1)
        sub_21 = add_40 = None
        expand_10 = l_self_modules_layers_modules_4_modules_mega_layer_modules_rel_pos_bias_parameters_b_param_.expand(
            22, 64
        )
        l_self_modules_layers_modules_4_modules_mega_layer_modules_rel_pos_bias_parameters_b_param_ = (
            None
        )
        chunk_27 = torch.chunk(expand_10, 2, dim=-1)
        expand_10 = None
        chunk_28 = chunk_27[0]
        chunk_29 = chunk_27[1]
        chunk_27 = None
        to_27 = to_25.to(
            l_self_modules_layers_modules_4_modules_mega_layer_modules_rel_pos_bias_buffers_float_tensor_
        )
        to_25 = None
        to_28 = to_26.to(
            l_self_modules_layers_modules_4_modules_mega_layer_modules_rel_pos_bias_buffers_float_tensor_
        )
        to_26 = l_self_modules_layers_modules_4_modules_mega_layer_modules_rel_pos_bias_buffers_float_tensor_ = (None)
        sin_9 = to_27[slice(None, 22, None)]
        cos_9 = to_28[slice(None, 22, None)]
        mul_107 = chunk_28 * cos_9
        mul_108 = chunk_29 * sin_9
        sub_22 = mul_107 - mul_108
        mul_107 = mul_108 = None
        mul_109 = chunk_29 * cos_9
        chunk_29 = cos_9 = None
        mul_110 = chunk_28 * sin_9
        chunk_28 = sin_9 = None
        add_41 = mul_109 + mul_110
        mul_109 = mul_110 = None
        rotary_beta_4 = torch.cat([sub_22, add_41], dim=1)
        sub_22 = add_41 = None
        bias_4 = torch.functional.einsum("mk,nk->mn", rotary_alpha_4, rotary_beta_4)
        rotary_alpha_4 = rotary_beta_4 = None
        query_19 = query_18 * 0.125
        query_18 = None
        transpose_24 = key_14.transpose(2, 3)
        key_14 = None
        matmul_8 = torch.matmul(query_19, transpose_24)
        query_19 = transpose_24 = None
        qk_8 = matmul_8 + bias_4
        matmul_8 = bias_4 = None
        padding_mask_13 = 1 - padding_mask_12
        padding_mask_12 = None
        padding_mask_all_4 = padding_mask_13.all(dim=-1, keepdim=True)
        invert_4 = ~padding_mask_all_4
        padding_mask_all_4 = None
        padding_mask_14 = torch.logical_and(padding_mask_13, invert_4)
        padding_mask_13 = invert_4 = None
        unsqueeze_34 = padding_mask_14.unsqueeze(2)
        padding_mask_14 = None
        to_29 = unsqueeze_34.to(torch.bool)
        unsqueeze_34 = None
        qk_9 = qk_8.masked_fill(to_29, -inf)
        qk_8 = to_29 = None
        softmax_4 = torch.nn.functional.softmax(qk_9, -1, _stacklevel=5)
        attn_weights_4 = softmax_4.type_as(qk_9)
        softmax_4 = qk_9 = None
        value_19 = torch.nn.functional.dropout(value_18, p=0.1, training=False)
        value_18 = None
        kernel_14 = torch.nn.functional.dropout(attn_weights_4, p=0.1, training=False)
        attn_weights_4 = None
        matmul_9 = torch.matmul(kernel_14, value_19)
        kernel_14 = value_19 = None
        view_9 = matmul_9.view(1, 22, 37)
        matmul_9 = None
        weighted_self_output_12 = view_9.transpose(0, 1)
        view_9 = None
        mul_112 = weighted_self_output_12 * attention_gate_4
        weighted_self_output_12 = attention_gate_4 = None
        linear_22 = torch._C._nn.linear(
            mul_112,
            l_self_modules_layers_modules_4_modules_mega_layer_modules_h_proj_parameters_weight_,
            l_self_modules_layers_modules_4_modules_mega_layer_modules_h_proj_parameters_bias_,
        )
        mul_112 = l_self_modules_layers_modules_4_modules_mega_layer_modules_h_proj_parameters_weight_ = l_self_modules_layers_modules_4_modules_mega_layer_modules_h_proj_parameters_bias_ = (None)
        add_43 = intermediate_state_4 + linear_22
        intermediate_state_4 = linear_22 = None
        weighted_self_output_13 = torch.nn.functional.silu(add_43, inplace=False)
        add_43 = None
        weighted_self_output_14 = torch.nn.functional.dropout(
            weighted_self_output_13, p=0.1, training=False
        )
        weighted_self_output_13 = None
        sub_24 = weighted_self_output_14 - output_19
        weighted_self_output_14 = None
        out_4 = torch.addcmul(output_19, residual_weight_9, sub_24)
        output_19 = residual_weight_9 = sub_24 = None
        square_9 = torch.square(out_4)
        mean_square_9 = torch.mean(square_9, dim=-1, keepdim=True)
        square_9 = None
        input_10 = (
            l_self_modules_layers_modules_4_modules_nffn_modules_norm_modules_norm_parameters_scalar_
            * out_4
        )
        l_self_modules_layers_modules_4_modules_nffn_modules_norm_modules_norm_parameters_scalar_ = (
            None
        )
        add_44 = mean_square_9 + 1e-05
        mean_square_9 = None
        rsqrt_9 = torch.rsqrt(add_44)
        add_44 = None
        output_21 = input_10 * rsqrt_9
        input_10 = rsqrt_9 = None
        linear_23 = torch._C._nn.linear(
            output_21,
            l_self_modules_layers_modules_4_modules_nffn_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_4_modules_nffn_modules_fc1_parameters_bias_,
        )
        output_21 = (
            l_self_modules_layers_modules_4_modules_nffn_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layers_modules_4_modules_nffn_modules_fc1_parameters_bias_
        ) = None
        hidden_8 = torch.nn.functional.silu(linear_23, inplace=False)
        linear_23 = None
        hidden_9 = torch.nn.functional.dropout(hidden_8, p=0.1, training=False)
        hidden_8 = None
        output_22 = torch._C._nn.linear(
            hidden_9,
            l_self_modules_layers_modules_4_modules_nffn_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_4_modules_nffn_modules_fc2_parameters_bias_,
        )
        hidden_9 = (
            l_self_modules_layers_modules_4_modules_nffn_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layers_modules_4_modules_nffn_modules_fc2_parameters_bias_
        ) = None
        output_23 = torch.nn.functional.dropout(output_22, p=0.1, training=False)
        output_22 = None
        output_24 = output_23 + out_4
        output_23 = out_4 = None
        hidden_states_1 = output_24.transpose(0, 1)
        output_24 = None
        first_token_tensor = hidden_states_1[(slice(None, None, None), 0)]
        pooled_output = torch._C._nn.linear(
            first_token_tensor,
            l_self_modules_pooler_modules_dense_parameters_weight_,
            l_self_modules_pooler_modules_dense_parameters_bias_,
        )
        first_token_tensor = (
            l_self_modules_pooler_modules_dense_parameters_weight_
        ) = l_self_modules_pooler_modules_dense_parameters_bias_ = None
        pooled_output_1 = torch.tanh(pooled_output)
        pooled_output = None
        return (
            einsum,
            to_4,
            to_3,
            einsum_2,
            to_10,
            to_9,
            einsum_4,
            to_16,
            to_15,
            einsum_6,
            to_22,
            to_21,
            einsum_8,
            to_28,
            to_27,
            hidden_states_1,
            pooled_output_1,
        )
