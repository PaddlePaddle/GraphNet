import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_pixel_values_: torch.Tensor,
        L_self_modules_vision_model_modules_embeddings_modules_patch_embedding_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_embeddings_parameters_class_embedding_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_embeddings_buffers_position_ids_: torch.Tensor,
        L_self_modules_vision_model_modules_embeddings_modules_position_embedding_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_embeddings_modules_position_embedding_norm_type: torch.Tensor,
        L_self_modules_vision_model_modules_pre_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_pre_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_pre_layernorm_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_ln_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_ln_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_ln_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_ln_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_ln_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_ln_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_ln_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_ln_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_ln_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_ln_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_ln_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_ln_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_ln_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_ln_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_ln_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_ln_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_ln_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_ln_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_ln_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_ln_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_ln_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_ln_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_ln_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_ln_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_ln_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_ln_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_ln_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_ln_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_ln_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_ln_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_ln_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_ln_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_ln_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_ln_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_ln_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_ln_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_post_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_post_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_post_layernorm_eps: torch.Tensor,
        L_self_modules_visual_projection_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mit_parameters_position_embedding_: torch.nn.parameter.Parameter,
        L_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_scale: torch.Tensor,
        L_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_prompts_visual_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_prompts_visual_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_prompts_visual_layernorm_eps: torch.Tensor,
        L_self_parameters_prompts_visual_projection_: torch.nn.parameter.Parameter,
        L_input_ids_: torch.Tensor,
        L_self_modules_text_model_modules_embeddings_modules_position_embedding_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_embeddings_buffers_position_ids_: torch.Tensor,
        L_self_modules_text_model_modules_embeddings_modules_token_embedding_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_embeddings_modules_token_embedding_norm_type: torch.Tensor,
        L_self_modules_text_model_modules_embeddings_modules_position_embedding_norm_type: torch.Tensor,
        L_attention_mask_: torch.Tensor,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_scale: torch.Tensor,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_scale: torch.Tensor,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_scale: torch.Tensor,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_scale: torch.Tensor,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_scale: torch.Tensor,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_scale: torch.Tensor,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_scale: torch.Tensor,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_scale: torch.Tensor,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_scale: torch.Tensor,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_scale: torch.Tensor,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_scale: torch.Tensor,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_scale: torch.Tensor,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_text_model_modules_final_layer_norm_eps: torch.Tensor,
        L_self_modules_text_projection_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_prompts_generator_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_prompts_generator_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_prompts_generator_modules_layernorm_eps: torch.Tensor,
        L_self_modules_prompts_generator_modules_decoder_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_prompts_generator_modules_decoder_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_prompts_generator_modules_decoder_modules_0_modules_norm1_eps: torch.Tensor,
        L_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_scale: torch.Tensor,
        L_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_attn_drop_p: torch.Tensor,
        L_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_proj_drop_p: torch.Tensor,
        L_self_modules_prompts_generator_modules_decoder_modules_0_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_prompts_generator_modules_decoder_modules_0_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_prompts_generator_modules_decoder_modules_0_modules_norm3_eps: torch.Tensor,
        L_self_modules_prompts_generator_modules_decoder_modules_0_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_prompts_generator_modules_decoder_modules_0_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_prompts_generator_modules_decoder_modules_0_modules_mlp_modules_2_p: torch.Tensor,
        L_self_modules_prompts_generator_modules_decoder_modules_0_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_prompts_generator_modules_decoder_modules_0_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_prompts_generator_modules_decoder_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_prompts_generator_modules_decoder_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_prompts_generator_modules_decoder_modules_1_modules_norm1_eps: torch.Tensor,
        L_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_scale: torch.Tensor,
        L_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_attn_drop_p: torch.Tensor,
        L_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_proj_drop_p: torch.Tensor,
        L_self_modules_prompts_generator_modules_decoder_modules_1_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_prompts_generator_modules_decoder_modules_1_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_prompts_generator_modules_decoder_modules_1_modules_norm3_eps: torch.Tensor,
        L_self_modules_prompts_generator_modules_decoder_modules_1_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_prompts_generator_modules_decoder_modules_1_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_prompts_generator_modules_decoder_modules_1_modules_mlp_modules_2_p: torch.Tensor,
        L_self_modules_prompts_generator_modules_decoder_modules_1_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_prompts_generator_modules_decoder_modules_1_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_prompts_generator_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_parameters_logit_scale_: torch.nn.parameter.Parameter,
    ):
        l_pixel_values_ = L_pixel_values_
        l_self_modules_vision_model_modules_embeddings_modules_patch_embedding_parameters_weight_ = L_self_modules_vision_model_modules_embeddings_modules_patch_embedding_parameters_weight_
        l_self_modules_vision_model_modules_embeddings_parameters_class_embedding_ = (
            L_self_modules_vision_model_modules_embeddings_parameters_class_embedding_
        )
        l_self_modules_vision_model_modules_embeddings_buffers_position_ids_ = (
            L_self_modules_vision_model_modules_embeddings_buffers_position_ids_
        )
        l_self_modules_vision_model_modules_embeddings_modules_position_embedding_parameters_weight_ = L_self_modules_vision_model_modules_embeddings_modules_position_embedding_parameters_weight_
        l_self_modules_vision_model_modules_embeddings_modules_position_embedding_norm_type = L_self_modules_vision_model_modules_embeddings_modules_position_embedding_norm_type
        l_self_modules_vision_model_modules_pre_layernorm_parameters_weight_ = (
            L_self_modules_vision_model_modules_pre_layernorm_parameters_weight_
        )
        l_self_modules_vision_model_modules_pre_layernorm_parameters_bias_ = (
            L_self_modules_vision_model_modules_pre_layernorm_parameters_bias_
        )
        l_self_modules_vision_model_modules_pre_layernorm_eps = (
            L_self_modules_vision_model_modules_pre_layernorm_eps
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_fc_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_fc_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_fc_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_fc_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_ln_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_ln_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_ln_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_ln_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_ln_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_ln_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_fc_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_fc_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_fc_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_fc_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_ln_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_ln_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_ln_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_ln_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_ln_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_ln_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_fc_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_fc_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_fc_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_fc_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_ln_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_ln_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_ln_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_ln_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_ln_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_ln_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_fc_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_fc_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_fc_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_fc_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_ln_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_ln_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_ln_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_ln_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_ln_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_ln_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_fc_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_fc_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_fc_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_fc_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_ln_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_ln_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_ln_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_ln_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_ln_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_ln_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_fc_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_fc_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_fc_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_fc_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_ln_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_ln_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_ln_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_ln_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_ln_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_ln_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_fc_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_fc_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_fc_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_fc_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_ln_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_ln_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_ln_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_ln_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_ln_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_ln_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_fc_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_fc_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_fc_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_fc_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_ln_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_ln_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_ln_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_ln_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_ln_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_ln_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_fc_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_fc_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_fc_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_fc_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_ln_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_ln_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_ln_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_ln_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_ln_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_ln_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_fc_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_fc_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_fc_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_fc_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_ln_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_ln_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_ln_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_ln_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_ln_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_ln_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_fc_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_fc_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_fc_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_fc_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_ln_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_ln_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_ln_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_ln_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_ln_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_ln_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_fc_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_fc_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_fc_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_fc_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_ln_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_ln_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_ln_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_ln_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_ln_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_ln_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_vision_model_modules_post_layernorm_parameters_weight_ = (
            L_self_modules_vision_model_modules_post_layernorm_parameters_weight_
        )
        l_self_modules_vision_model_modules_post_layernorm_parameters_bias_ = (
            L_self_modules_vision_model_modules_post_layernorm_parameters_bias_
        )
        l_self_modules_vision_model_modules_post_layernorm_eps = (
            L_self_modules_vision_model_modules_post_layernorm_eps
        )
        l_self_modules_visual_projection_parameters_weight_ = (
            L_self_modules_visual_projection_parameters_weight_
        )
        l_self_modules_mit_parameters_position_embedding_ = (
            L_self_modules_mit_parameters_position_embedding_
        )
        l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_weight_ = L_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_weight_
        l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_bias_ = L_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_bias_
        l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_layer_norm1_eps = L_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_layer_norm1_eps
        l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_scale = L_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_scale
        l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_weight_ = L_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_weight_
        l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_bias_ = L_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_bias_
        l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_layer_norm2_eps = L_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_layer_norm2_eps
        l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_prompts_visual_layernorm_parameters_weight_ = (
            L_self_modules_prompts_visual_layernorm_parameters_weight_
        )
        l_self_modules_prompts_visual_layernorm_parameters_bias_ = (
            L_self_modules_prompts_visual_layernorm_parameters_bias_
        )
        l_self_modules_prompts_visual_layernorm_eps = (
            L_self_modules_prompts_visual_layernorm_eps
        )
        l_self_parameters_prompts_visual_projection_ = (
            L_self_parameters_prompts_visual_projection_
        )
        l_input_ids_ = L_input_ids_
        l_self_modules_text_model_modules_embeddings_modules_position_embedding_parameters_weight_ = L_self_modules_text_model_modules_embeddings_modules_position_embedding_parameters_weight_
        l_self_modules_text_model_modules_embeddings_buffers_position_ids_ = (
            L_self_modules_text_model_modules_embeddings_buffers_position_ids_
        )
        l_self_modules_text_model_modules_embeddings_modules_token_embedding_parameters_weight_ = L_self_modules_text_model_modules_embeddings_modules_token_embedding_parameters_weight_
        l_self_modules_text_model_modules_embeddings_modules_token_embedding_norm_type = L_self_modules_text_model_modules_embeddings_modules_token_embedding_norm_type
        l_self_modules_text_model_modules_embeddings_modules_position_embedding_norm_type = L_self_modules_text_model_modules_embeddings_modules_position_embedding_norm_type
        l_attention_mask_ = L_attention_mask_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_eps = L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_eps
        l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_scale = L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_scale
        l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_eps = L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_eps
        l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_eps = L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_eps
        l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_scale = L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_scale
        l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_eps = L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_eps
        l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_eps = L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_eps
        l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_scale = L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_scale
        l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_eps = L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_eps
        l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_eps = L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_eps
        l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_scale = L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_scale
        l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_eps = L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_eps
        l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_eps = L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_eps
        l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_scale = L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_scale
        l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_eps = L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_eps
        l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_eps = L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_eps
        l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_scale = L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_scale
        l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_eps = L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_eps
        l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_eps = L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_eps
        l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_scale = L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_scale
        l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_eps = L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_eps
        l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_eps = L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_eps
        l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_scale = L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_scale
        l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_eps = L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_eps
        l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_eps = L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_eps
        l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_scale = L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_scale
        l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_eps = L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_eps
        l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_eps = L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_eps
        l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_scale = L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_scale
        l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_eps = L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_eps
        l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_eps = L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_eps
        l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_scale = L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_scale
        l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_eps = L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_eps
        l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_eps = L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_eps
        l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_scale = L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_scale
        l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_eps = L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_eps
        l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_text_model_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_text_model_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_text_model_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_text_model_modules_final_layer_norm_parameters_bias_
        )
        l_self_modules_text_model_modules_final_layer_norm_eps = (
            L_self_modules_text_model_modules_final_layer_norm_eps
        )
        l_self_modules_text_projection_parameters_weight_ = (
            L_self_modules_text_projection_parameters_weight_
        )
        l_self_modules_prompts_generator_modules_layernorm_parameters_weight_ = (
            L_self_modules_prompts_generator_modules_layernorm_parameters_weight_
        )
        l_self_modules_prompts_generator_modules_layernorm_parameters_bias_ = (
            L_self_modules_prompts_generator_modules_layernorm_parameters_bias_
        )
        l_self_modules_prompts_generator_modules_layernorm_eps = (
            L_self_modules_prompts_generator_modules_layernorm_eps
        )
        l_self_modules_prompts_generator_modules_decoder_modules_0_modules_norm1_parameters_weight_ = L_self_modules_prompts_generator_modules_decoder_modules_0_modules_norm1_parameters_weight_
        l_self_modules_prompts_generator_modules_decoder_modules_0_modules_norm1_parameters_bias_ = L_self_modules_prompts_generator_modules_decoder_modules_0_modules_norm1_parameters_bias_
        l_self_modules_prompts_generator_modules_decoder_modules_0_modules_norm1_eps = (
            L_self_modules_prompts_generator_modules_decoder_modules_0_modules_norm1_eps
        )
        l_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_q_proj_parameters_weight_ = L_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_q_proj_parameters_weight_
        l_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_k_proj_parameters_weight_ = L_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_k_proj_parameters_weight_
        l_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_v_proj_parameters_weight_ = L_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_v_proj_parameters_weight_
        l_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_scale = L_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_scale
        l_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_attn_drop_p = L_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_attn_drop_p
        l_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_proj_parameters_weight_ = L_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_proj_parameters_weight_
        l_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_proj_parameters_bias_ = L_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_proj_parameters_bias_
        l_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_proj_drop_p = L_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_proj_drop_p
        l_self_modules_prompts_generator_modules_decoder_modules_0_modules_norm3_parameters_weight_ = L_self_modules_prompts_generator_modules_decoder_modules_0_modules_norm3_parameters_weight_
        l_self_modules_prompts_generator_modules_decoder_modules_0_modules_norm3_parameters_bias_ = L_self_modules_prompts_generator_modules_decoder_modules_0_modules_norm3_parameters_bias_
        l_self_modules_prompts_generator_modules_decoder_modules_0_modules_norm3_eps = (
            L_self_modules_prompts_generator_modules_decoder_modules_0_modules_norm3_eps
        )
        l_self_modules_prompts_generator_modules_decoder_modules_0_modules_mlp_modules_0_parameters_weight_ = L_self_modules_prompts_generator_modules_decoder_modules_0_modules_mlp_modules_0_parameters_weight_
        l_self_modules_prompts_generator_modules_decoder_modules_0_modules_mlp_modules_0_parameters_bias_ = L_self_modules_prompts_generator_modules_decoder_modules_0_modules_mlp_modules_0_parameters_bias_
        l_self_modules_prompts_generator_modules_decoder_modules_0_modules_mlp_modules_2_p = L_self_modules_prompts_generator_modules_decoder_modules_0_modules_mlp_modules_2_p
        l_self_modules_prompts_generator_modules_decoder_modules_0_modules_mlp_modules_3_parameters_weight_ = L_self_modules_prompts_generator_modules_decoder_modules_0_modules_mlp_modules_3_parameters_weight_
        l_self_modules_prompts_generator_modules_decoder_modules_0_modules_mlp_modules_3_parameters_bias_ = L_self_modules_prompts_generator_modules_decoder_modules_0_modules_mlp_modules_3_parameters_bias_
        l_self_modules_prompts_generator_modules_decoder_modules_1_modules_norm1_parameters_weight_ = L_self_modules_prompts_generator_modules_decoder_modules_1_modules_norm1_parameters_weight_
        l_self_modules_prompts_generator_modules_decoder_modules_1_modules_norm1_parameters_bias_ = L_self_modules_prompts_generator_modules_decoder_modules_1_modules_norm1_parameters_bias_
        l_self_modules_prompts_generator_modules_decoder_modules_1_modules_norm1_eps = (
            L_self_modules_prompts_generator_modules_decoder_modules_1_modules_norm1_eps
        )
        l_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_q_proj_parameters_weight_ = L_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_q_proj_parameters_weight_
        l_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_k_proj_parameters_weight_ = L_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_k_proj_parameters_weight_
        l_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_v_proj_parameters_weight_ = L_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_v_proj_parameters_weight_
        l_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_scale = L_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_scale
        l_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_attn_drop_p = L_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_attn_drop_p
        l_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_proj_parameters_weight_ = L_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_proj_parameters_weight_
        l_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_proj_parameters_bias_ = L_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_proj_parameters_bias_
        l_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_proj_drop_p = L_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_proj_drop_p
        l_self_modules_prompts_generator_modules_decoder_modules_1_modules_norm3_parameters_weight_ = L_self_modules_prompts_generator_modules_decoder_modules_1_modules_norm3_parameters_weight_
        l_self_modules_prompts_generator_modules_decoder_modules_1_modules_norm3_parameters_bias_ = L_self_modules_prompts_generator_modules_decoder_modules_1_modules_norm3_parameters_bias_
        l_self_modules_prompts_generator_modules_decoder_modules_1_modules_norm3_eps = (
            L_self_modules_prompts_generator_modules_decoder_modules_1_modules_norm3_eps
        )
        l_self_modules_prompts_generator_modules_decoder_modules_1_modules_mlp_modules_0_parameters_weight_ = L_self_modules_prompts_generator_modules_decoder_modules_1_modules_mlp_modules_0_parameters_weight_
        l_self_modules_prompts_generator_modules_decoder_modules_1_modules_mlp_modules_0_parameters_bias_ = L_self_modules_prompts_generator_modules_decoder_modules_1_modules_mlp_modules_0_parameters_bias_
        l_self_modules_prompts_generator_modules_decoder_modules_1_modules_mlp_modules_2_p = L_self_modules_prompts_generator_modules_decoder_modules_1_modules_mlp_modules_2_p
        l_self_modules_prompts_generator_modules_decoder_modules_1_modules_mlp_modules_3_parameters_weight_ = L_self_modules_prompts_generator_modules_decoder_modules_1_modules_mlp_modules_3_parameters_weight_
        l_self_modules_prompts_generator_modules_decoder_modules_1_modules_mlp_modules_3_parameters_bias_ = L_self_modules_prompts_generator_modules_decoder_modules_1_modules_mlp_modules_3_parameters_bias_
        l_self_modules_prompts_generator_parameters_alpha_ = (
            L_self_modules_prompts_generator_parameters_alpha_
        )
        l_self_parameters_logit_scale_ = L_self_parameters_logit_scale_
        size = l_pixel_values_.size()
        getitem_1 = size[1]
        getitem_2 = size[2]
        getitem_3 = size[3]
        getitem_4 = size[4]
        size = None
        pixel_values = l_pixel_values_.reshape(-1, getitem_2, getitem_3, getitem_4)
        l_pixel_values_ = getitem_2 = getitem_3 = getitem_4 = None
        size_1 = pixel_values.size()
        getitem_5 = size_1[0]
        size_1 = None
        to = pixel_values.to(dtype=torch.float32)
        pixel_values = None
        patch_embeds = torch.conv2d(
            to,
            l_self_modules_vision_model_modules_embeddings_modules_patch_embedding_parameters_weight_,
            None,
            (16, 16),
            (0, 0),
            (1, 1),
            1,
        )
        to = l_self_modules_vision_model_modules_embeddings_modules_patch_embedding_parameters_weight_ = (None)
        flatten = patch_embeds.flatten(2)
        patch_embeds = None
        patch_embeds_1 = flatten.transpose(1, 2)
        flatten = None
        class_embeds = l_self_modules_vision_model_modules_embeddings_parameters_class_embedding_.expand(
            getitem_5, 1, -1
        )
        l_self_modules_vision_model_modules_embeddings_parameters_class_embedding_ = (
            getitem_5
        ) = None
        embeddings = torch.cat([class_embeds, patch_embeds_1], dim=1)
        class_embeds = patch_embeds_1 = None
        item = (
            l_self_modules_vision_model_modules_embeddings_modules_position_embedding_norm_type.item()
        )
        l_self_modules_vision_model_modules_embeddings_modules_position_embedding_norm_type = (
            None
        )
        embedding = torch.nn.functional.embedding(
            l_self_modules_vision_model_modules_embeddings_buffers_position_ids_,
            l_self_modules_vision_model_modules_embeddings_modules_position_embedding_parameters_weight_,
            None,
            None,
            item,
            False,
            False,
        )
        l_self_modules_vision_model_modules_embeddings_buffers_position_ids_ = l_self_modules_vision_model_modules_embeddings_modules_position_embedding_parameters_weight_ = (item) = (
            None
        )
        embeddings_1 = embeddings + embedding
        embeddings = embedding = None
        item_1 = l_self_modules_vision_model_modules_pre_layernorm_eps.item()
        l_self_modules_vision_model_modules_pre_layernorm_eps = None
        hidden_states = torch.nn.functional.layer_norm(
            embeddings_1,
            (768,),
            l_self_modules_vision_model_modules_pre_layernorm_parameters_weight_,
            l_self_modules_vision_model_modules_pre_layernorm_parameters_bias_,
            item_1,
        )
        embeddings_1 = (
            l_self_modules_vision_model_modules_pre_layernorm_parameters_weight_
        ) = (
            l_self_modules_vision_model_modules_pre_layernorm_parameters_bias_
        ) = item_1 = None
        size_2 = hidden_states.size()
        getitem_9 = size_2[0]
        size_2 = None
        floordiv = getitem_9 // 8
        getitem_9 = None
        getitem_12 = hidden_states[
            (slice(None, None, None), 0, slice(None, None, None))
        ]
        msg_token = torch._C._nn.linear(
            getitem_12,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_fc_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_fc_parameters_bias_,
        )
        getitem_12 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_fc_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_fc_parameters_bias_ = (None)
        msg_token_1 = msg_token.view(floordiv, 8, 768)
        msg_token = floordiv = None
        item_2 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_ln_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_ln_eps = (
            None
        )
        layer_norm_1 = torch.nn.functional.layer_norm(
            msg_token_1,
            (768,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_ln_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_ln_parameters_bias_,
            item_2,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_ln_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_ln_parameters_bias_ = (item_2) = (
            None
        )
        size_3 = layer_norm_1.size()
        getitem_14 = size_3[1]
        size_3 = None
        queries = torch._C._nn.linear(
            layer_norm_1,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_attn_modules_q_proj_parameters_bias_ = (None)
        keys = torch._C._nn.linear(
            layer_norm_1,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_attn_modules_k_proj_parameters_bias_ = (None)
        values = torch._C._nn.linear(
            layer_norm_1,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_attn_modules_v_proj_parameters_bias_,
        )
        layer_norm_1 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_attn_modules_v_proj_parameters_bias_ = (None)
        view_1 = queries.view(1, getitem_14, 12, 64)
        queries = None
        queries_1 = view_1.transpose(1, 2)
        view_1 = None
        view_2 = keys.view(1, getitem_14, 12, 64)
        keys = None
        keys_1 = view_2.transpose(1, 2)
        view_2 = None
        view_3 = values.view(1, getitem_14, 12, 64)
        values = None
        values_1 = view_3.transpose(1, 2)
        view_3 = None
        transpose_4 = keys_1.transpose(-1, -2)
        keys_1 = None
        matmul = torch.matmul(queries_1, transpose_4)
        queries_1 = transpose_4 = None
        item_3 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_attn_scale = (
            None
        )
        attn_weights = matmul * item_3
        matmul = item_3 = None
        softmax = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = None
        attn_weights_1 = softmax.to(torch.float32)
        softmax = None
        attn_weights_2 = torch.nn.functional.dropout(
            attn_weights_1, p=0.0, training=False
        )
        attn_weights_1 = None
        attn_output = torch.matmul(attn_weights_2, values_1)
        attn_weights_2 = values_1 = None
        transpose_5 = attn_output.transpose(1, 2)
        attn_output = None
        attn_output_1 = transpose_5.contiguous()
        transpose_5 = None
        reshape_1 = attn_output_1.reshape(1, getitem_14, 768)
        attn_output_1 = getitem_14 = None
        attn_output_2 = reshape_1.contiguous()
        reshape_1 = None
        attn_output_3 = torch._C._nn.linear(
            attn_output_2,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_2 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_attn_modules_out_proj_parameters_bias_ = (None)
        msg_token_2 = msg_token_1 + attn_output_3
        msg_token_1 = attn_output_3 = None
        msg_token_3 = msg_token_2.view(-1, 1, 768)
        msg_token_2 = None
        hidden_states_1 = torch.cat([hidden_states, msg_token_3], dim=1)
        hidden_states = msg_token_3 = None
        item_4 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_eps = (
            None
        )
        hidden_states_2 = torch.nn.functional.layer_norm(
            hidden_states_1,
            (768,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_bias_,
            item_4,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_bias_ = (item_4) = (
            None
        )
        size_4 = hidden_states_2.size()
        getitem_16 = size_4[0]
        size_4 = None
        queries_2 = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_2 = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_2 = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_2 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_5 = queries_2.view(getitem_16, 198, 12, 64)
        queries_2 = None
        queries_3 = view_5.transpose(1, 2)
        view_5 = None
        view_6 = keys_2.view(getitem_16, 198, 12, 64)
        keys_2 = None
        keys_3 = view_6.transpose(1, 2)
        view_6 = None
        view_7 = values_2.view(getitem_16, 198, 12, 64)
        values_2 = None
        values_3 = view_7.transpose(1, 2)
        view_7 = None
        transpose_9 = keys_3.transpose(-1, -2)
        keys_3 = None
        matmul_2 = torch.matmul(queries_3, transpose_9)
        queries_3 = transpose_9 = None
        item_5 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_scale = (
            None
        )
        attn_weights_3 = matmul_2 * item_5
        matmul_2 = item_5 = None
        softmax_1 = torch.nn.functional.softmax(
            attn_weights_3, dim=-1, dtype=torch.float32
        )
        attn_weights_3 = None
        attn_weights_4 = softmax_1.to(torch.float32)
        softmax_1 = None
        attn_weights_5 = torch.nn.functional.dropout(
            attn_weights_4, p=0.0, training=False
        )
        attn_weights_4 = None
        attn_output_4 = torch.matmul(attn_weights_5, values_3)
        attn_weights_5 = values_3 = None
        transpose_10 = attn_output_4.transpose(1, 2)
        attn_output_4 = None
        attn_output_5 = transpose_10.contiguous()
        transpose_10 = None
        reshape_2 = attn_output_5.reshape(getitem_16, 198, 768)
        attn_output_5 = getitem_16 = None
        attn_output_6 = reshape_2.contiguous()
        reshape_2 = None
        attn_output_7 = torch._C._nn.linear(
            attn_output_6,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_6 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_3 = hidden_states_1 + attn_output_7
        hidden_states_1 = attn_output_7 = None
        hidden_states_4 = hidden_states_3[
            (slice(None, None, None), slice(None, 197, None), slice(None, None, None))
        ]
        hidden_states_3 = None
        item_6 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_eps = (
            None
        )
        hidden_states_5 = torch.nn.functional.layer_norm(
            hidden_states_4,
            (768,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_bias_,
            item_6,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_bias_ = (item_6) = (
            None
        )
        hidden_states_6 = torch._C._nn.linear(
            hidden_states_5,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_5 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_2 = 1.702 * hidden_states_6
        sigmoid = torch.sigmoid(mul_2)
        mul_2 = None
        hidden_states_7 = hidden_states_6 * sigmoid
        hidden_states_6 = sigmoid = None
        hidden_states_8 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_7 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_9 = hidden_states_4 + hidden_states_8
        hidden_states_4 = hidden_states_8 = None
        size_5 = hidden_states_9.size()
        getitem_20 = size_5[0]
        size_5 = None
        floordiv_1 = getitem_20 // 8
        getitem_20 = None
        getitem_23 = hidden_states_9[
            (slice(None, None, None), 0, slice(None, None, None))
        ]
        msg_token_4 = torch._C._nn.linear(
            getitem_23,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_fc_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_fc_parameters_bias_,
        )
        getitem_23 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_fc_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_fc_parameters_bias_ = (None)
        msg_token_5 = msg_token_4.view(floordiv_1, 8, 768)
        msg_token_4 = floordiv_1 = None
        item_7 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_ln_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_ln_eps = (
            None
        )
        layer_norm_4 = torch.nn.functional.layer_norm(
            msg_token_5,
            (768,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_ln_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_ln_parameters_bias_,
            item_7,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_ln_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_ln_parameters_bias_ = (item_7) = (
            None
        )
        size_6 = layer_norm_4.size()
        getitem_25 = size_6[1]
        size_6 = None
        queries_4 = torch._C._nn.linear(
            layer_norm_4,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_attn_modules_q_proj_parameters_bias_ = (None)
        keys_4 = torch._C._nn.linear(
            layer_norm_4,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_attn_modules_k_proj_parameters_bias_ = (None)
        values_4 = torch._C._nn.linear(
            layer_norm_4,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_attn_modules_v_proj_parameters_bias_,
        )
        layer_norm_4 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_attn_modules_v_proj_parameters_bias_ = (None)
        view_9 = queries_4.view(1, getitem_25, 12, 64)
        queries_4 = None
        queries_5 = view_9.transpose(1, 2)
        view_9 = None
        view_10 = keys_4.view(1, getitem_25, 12, 64)
        keys_4 = None
        keys_5 = view_10.transpose(1, 2)
        view_10 = None
        view_11 = values_4.view(1, getitem_25, 12, 64)
        values_4 = None
        values_5 = view_11.transpose(1, 2)
        view_11 = None
        transpose_14 = keys_5.transpose(-1, -2)
        keys_5 = None
        matmul_4 = torch.matmul(queries_5, transpose_14)
        queries_5 = transpose_14 = None
        item_8 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_attn_scale = (
            None
        )
        attn_weights_6 = matmul_4 * item_8
        matmul_4 = item_8 = None
        softmax_2 = torch.nn.functional.softmax(
            attn_weights_6, dim=-1, dtype=torch.float32
        )
        attn_weights_6 = None
        attn_weights_7 = softmax_2.to(torch.float32)
        softmax_2 = None
        attn_weights_8 = torch.nn.functional.dropout(
            attn_weights_7, p=0.0, training=False
        )
        attn_weights_7 = None
        attn_output_8 = torch.matmul(attn_weights_8, values_5)
        attn_weights_8 = values_5 = None
        transpose_15 = attn_output_8.transpose(1, 2)
        attn_output_8 = None
        attn_output_9 = transpose_15.contiguous()
        transpose_15 = None
        reshape_3 = attn_output_9.reshape(1, getitem_25, 768)
        attn_output_9 = getitem_25 = None
        attn_output_10 = reshape_3.contiguous()
        reshape_3 = None
        attn_output_11 = torch._C._nn.linear(
            attn_output_10,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_10 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_attn_modules_out_proj_parameters_bias_ = (None)
        msg_token_6 = msg_token_5 + attn_output_11
        msg_token_5 = attn_output_11 = None
        msg_token_7 = msg_token_6.view(-1, 1, 768)
        msg_token_6 = None
        hidden_states_10 = torch.cat([hidden_states_9, msg_token_7], dim=1)
        hidden_states_9 = msg_token_7 = None
        item_9 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_eps = (
            None
        )
        hidden_states_11 = torch.nn.functional.layer_norm(
            hidden_states_10,
            (768,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_bias_,
            item_9,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_bias_ = (item_9) = (
            None
        )
        size_7 = hidden_states_11.size()
        getitem_27 = size_7[0]
        size_7 = None
        queries_6 = torch._C._nn.linear(
            hidden_states_11,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_6 = torch._C._nn.linear(
            hidden_states_11,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_6 = torch._C._nn.linear(
            hidden_states_11,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_11 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_13 = queries_6.view(getitem_27, 198, 12, 64)
        queries_6 = None
        queries_7 = view_13.transpose(1, 2)
        view_13 = None
        view_14 = keys_6.view(getitem_27, 198, 12, 64)
        keys_6 = None
        keys_7 = view_14.transpose(1, 2)
        view_14 = None
        view_15 = values_6.view(getitem_27, 198, 12, 64)
        values_6 = None
        values_7 = view_15.transpose(1, 2)
        view_15 = None
        transpose_19 = keys_7.transpose(-1, -2)
        keys_7 = None
        matmul_6 = torch.matmul(queries_7, transpose_19)
        queries_7 = transpose_19 = None
        item_10 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_scale = (
            None
        )
        attn_weights_9 = matmul_6 * item_10
        matmul_6 = item_10 = None
        softmax_3 = torch.nn.functional.softmax(
            attn_weights_9, dim=-1, dtype=torch.float32
        )
        attn_weights_9 = None
        attn_weights_10 = softmax_3.to(torch.float32)
        softmax_3 = None
        attn_weights_11 = torch.nn.functional.dropout(
            attn_weights_10, p=0.0, training=False
        )
        attn_weights_10 = None
        attn_output_12 = torch.matmul(attn_weights_11, values_7)
        attn_weights_11 = values_7 = None
        transpose_20 = attn_output_12.transpose(1, 2)
        attn_output_12 = None
        attn_output_13 = transpose_20.contiguous()
        transpose_20 = None
        reshape_4 = attn_output_13.reshape(getitem_27, 198, 768)
        attn_output_13 = getitem_27 = None
        attn_output_14 = reshape_4.contiguous()
        reshape_4 = None
        attn_output_15 = torch._C._nn.linear(
            attn_output_14,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_14 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_12 = hidden_states_10 + attn_output_15
        hidden_states_10 = attn_output_15 = None
        hidden_states_13 = hidden_states_12[
            (slice(None, None, None), slice(None, 197, None), slice(None, None, None))
        ]
        hidden_states_12 = None
        item_11 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_eps = (
            None
        )
        hidden_states_14 = torch.nn.functional.layer_norm(
            hidden_states_13,
            (768,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_bias_,
            item_11,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_bias_ = (item_11) = (
            None
        )
        hidden_states_15 = torch._C._nn.linear(
            hidden_states_14,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_14 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_6 = 1.702 * hidden_states_15
        sigmoid_1 = torch.sigmoid(mul_6)
        mul_6 = None
        hidden_states_16 = hidden_states_15 * sigmoid_1
        hidden_states_15 = sigmoid_1 = None
        hidden_states_17 = torch._C._nn.linear(
            hidden_states_16,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_16 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_18 = hidden_states_13 + hidden_states_17
        hidden_states_13 = hidden_states_17 = None
        size_8 = hidden_states_18.size()
        getitem_31 = size_8[0]
        size_8 = None
        floordiv_2 = getitem_31 // 8
        getitem_31 = None
        getitem_34 = hidden_states_18[
            (slice(None, None, None), 0, slice(None, None, None))
        ]
        msg_token_8 = torch._C._nn.linear(
            getitem_34,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_fc_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_fc_parameters_bias_,
        )
        getitem_34 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_fc_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_fc_parameters_bias_ = (None)
        msg_token_9 = msg_token_8.view(floordiv_2, 8, 768)
        msg_token_8 = floordiv_2 = None
        item_12 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_ln_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_ln_eps = (
            None
        )
        layer_norm_7 = torch.nn.functional.layer_norm(
            msg_token_9,
            (768,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_ln_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_ln_parameters_bias_,
            item_12,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_ln_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_ln_parameters_bias_ = (item_12) = (
            None
        )
        size_9 = layer_norm_7.size()
        getitem_36 = size_9[1]
        size_9 = None
        queries_8 = torch._C._nn.linear(
            layer_norm_7,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_attn_modules_q_proj_parameters_bias_ = (None)
        keys_8 = torch._C._nn.linear(
            layer_norm_7,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_attn_modules_k_proj_parameters_bias_ = (None)
        values_8 = torch._C._nn.linear(
            layer_norm_7,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_attn_modules_v_proj_parameters_bias_,
        )
        layer_norm_7 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_attn_modules_v_proj_parameters_bias_ = (None)
        view_17 = queries_8.view(1, getitem_36, 12, 64)
        queries_8 = None
        queries_9 = view_17.transpose(1, 2)
        view_17 = None
        view_18 = keys_8.view(1, getitem_36, 12, 64)
        keys_8 = None
        keys_9 = view_18.transpose(1, 2)
        view_18 = None
        view_19 = values_8.view(1, getitem_36, 12, 64)
        values_8 = None
        values_9 = view_19.transpose(1, 2)
        view_19 = None
        transpose_24 = keys_9.transpose(-1, -2)
        keys_9 = None
        matmul_8 = torch.matmul(queries_9, transpose_24)
        queries_9 = transpose_24 = None
        item_13 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_attn_scale = (
            None
        )
        attn_weights_12 = matmul_8 * item_13
        matmul_8 = item_13 = None
        softmax_4 = torch.nn.functional.softmax(
            attn_weights_12, dim=-1, dtype=torch.float32
        )
        attn_weights_12 = None
        attn_weights_13 = softmax_4.to(torch.float32)
        softmax_4 = None
        attn_weights_14 = torch.nn.functional.dropout(
            attn_weights_13, p=0.0, training=False
        )
        attn_weights_13 = None
        attn_output_16 = torch.matmul(attn_weights_14, values_9)
        attn_weights_14 = values_9 = None
        transpose_25 = attn_output_16.transpose(1, 2)
        attn_output_16 = None
        attn_output_17 = transpose_25.contiguous()
        transpose_25 = None
        reshape_5 = attn_output_17.reshape(1, getitem_36, 768)
        attn_output_17 = getitem_36 = None
        attn_output_18 = reshape_5.contiguous()
        reshape_5 = None
        attn_output_19 = torch._C._nn.linear(
            attn_output_18,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_18 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_attn_modules_out_proj_parameters_bias_ = (None)
        msg_token_10 = msg_token_9 + attn_output_19
        msg_token_9 = attn_output_19 = None
        msg_token_11 = msg_token_10.view(-1, 1, 768)
        msg_token_10 = None
        hidden_states_19 = torch.cat([hidden_states_18, msg_token_11], dim=1)
        hidden_states_18 = msg_token_11 = None
        item_14 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_eps = (
            None
        )
        hidden_states_20 = torch.nn.functional.layer_norm(
            hidden_states_19,
            (768,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_bias_,
            item_14,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_bias_ = (item_14) = (
            None
        )
        size_10 = hidden_states_20.size()
        getitem_38 = size_10[0]
        size_10 = None
        queries_10 = torch._C._nn.linear(
            hidden_states_20,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_10 = torch._C._nn.linear(
            hidden_states_20,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_10 = torch._C._nn.linear(
            hidden_states_20,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_20 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_21 = queries_10.view(getitem_38, 198, 12, 64)
        queries_10 = None
        queries_11 = view_21.transpose(1, 2)
        view_21 = None
        view_22 = keys_10.view(getitem_38, 198, 12, 64)
        keys_10 = None
        keys_11 = view_22.transpose(1, 2)
        view_22 = None
        view_23 = values_10.view(getitem_38, 198, 12, 64)
        values_10 = None
        values_11 = view_23.transpose(1, 2)
        view_23 = None
        transpose_29 = keys_11.transpose(-1, -2)
        keys_11 = None
        matmul_10 = torch.matmul(queries_11, transpose_29)
        queries_11 = transpose_29 = None
        item_15 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_scale = (
            None
        )
        attn_weights_15 = matmul_10 * item_15
        matmul_10 = item_15 = None
        softmax_5 = torch.nn.functional.softmax(
            attn_weights_15, dim=-1, dtype=torch.float32
        )
        attn_weights_15 = None
        attn_weights_16 = softmax_5.to(torch.float32)
        softmax_5 = None
        attn_weights_17 = torch.nn.functional.dropout(
            attn_weights_16, p=0.0, training=False
        )
        attn_weights_16 = None
        attn_output_20 = torch.matmul(attn_weights_17, values_11)
        attn_weights_17 = values_11 = None
        transpose_30 = attn_output_20.transpose(1, 2)
        attn_output_20 = None
        attn_output_21 = transpose_30.contiguous()
        transpose_30 = None
        reshape_6 = attn_output_21.reshape(getitem_38, 198, 768)
        attn_output_21 = getitem_38 = None
        attn_output_22 = reshape_6.contiguous()
        reshape_6 = None
        attn_output_23 = torch._C._nn.linear(
            attn_output_22,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_22 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_21 = hidden_states_19 + attn_output_23
        hidden_states_19 = attn_output_23 = None
        hidden_states_22 = hidden_states_21[
            (slice(None, None, None), slice(None, 197, None), slice(None, None, None))
        ]
        hidden_states_21 = None
        item_16 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_eps = (
            None
        )
        hidden_states_23 = torch.nn.functional.layer_norm(
            hidden_states_22,
            (768,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_bias_,
            item_16,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_bias_ = (item_16) = (
            None
        )
        hidden_states_24 = torch._C._nn.linear(
            hidden_states_23,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_23 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_10 = 1.702 * hidden_states_24
        sigmoid_2 = torch.sigmoid(mul_10)
        mul_10 = None
        hidden_states_25 = hidden_states_24 * sigmoid_2
        hidden_states_24 = sigmoid_2 = None
        hidden_states_26 = torch._C._nn.linear(
            hidden_states_25,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_25 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_27 = hidden_states_22 + hidden_states_26
        hidden_states_22 = hidden_states_26 = None
        size_11 = hidden_states_27.size()
        getitem_42 = size_11[0]
        size_11 = None
        floordiv_3 = getitem_42 // 8
        getitem_42 = None
        getitem_45 = hidden_states_27[
            (slice(None, None, None), 0, slice(None, None, None))
        ]
        msg_token_12 = torch._C._nn.linear(
            getitem_45,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_fc_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_fc_parameters_bias_,
        )
        getitem_45 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_fc_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_fc_parameters_bias_ = (None)
        msg_token_13 = msg_token_12.view(floordiv_3, 8, 768)
        msg_token_12 = floordiv_3 = None
        item_17 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_ln_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_ln_eps = (
            None
        )
        layer_norm_10 = torch.nn.functional.layer_norm(
            msg_token_13,
            (768,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_ln_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_ln_parameters_bias_,
            item_17,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_ln_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_ln_parameters_bias_ = (item_17) = (
            None
        )
        size_12 = layer_norm_10.size()
        getitem_47 = size_12[1]
        size_12 = None
        queries_12 = torch._C._nn.linear(
            layer_norm_10,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_attn_modules_q_proj_parameters_bias_ = (None)
        keys_12 = torch._C._nn.linear(
            layer_norm_10,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_attn_modules_k_proj_parameters_bias_ = (None)
        values_12 = torch._C._nn.linear(
            layer_norm_10,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_attn_modules_v_proj_parameters_bias_,
        )
        layer_norm_10 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_attn_modules_v_proj_parameters_bias_ = (None)
        view_25 = queries_12.view(1, getitem_47, 12, 64)
        queries_12 = None
        queries_13 = view_25.transpose(1, 2)
        view_25 = None
        view_26 = keys_12.view(1, getitem_47, 12, 64)
        keys_12 = None
        keys_13 = view_26.transpose(1, 2)
        view_26 = None
        view_27 = values_12.view(1, getitem_47, 12, 64)
        values_12 = None
        values_13 = view_27.transpose(1, 2)
        view_27 = None
        transpose_34 = keys_13.transpose(-1, -2)
        keys_13 = None
        matmul_12 = torch.matmul(queries_13, transpose_34)
        queries_13 = transpose_34 = None
        item_18 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_attn_scale = (
            None
        )
        attn_weights_18 = matmul_12 * item_18
        matmul_12 = item_18 = None
        softmax_6 = torch.nn.functional.softmax(
            attn_weights_18, dim=-1, dtype=torch.float32
        )
        attn_weights_18 = None
        attn_weights_19 = softmax_6.to(torch.float32)
        softmax_6 = None
        attn_weights_20 = torch.nn.functional.dropout(
            attn_weights_19, p=0.0, training=False
        )
        attn_weights_19 = None
        attn_output_24 = torch.matmul(attn_weights_20, values_13)
        attn_weights_20 = values_13 = None
        transpose_35 = attn_output_24.transpose(1, 2)
        attn_output_24 = None
        attn_output_25 = transpose_35.contiguous()
        transpose_35 = None
        reshape_7 = attn_output_25.reshape(1, getitem_47, 768)
        attn_output_25 = getitem_47 = None
        attn_output_26 = reshape_7.contiguous()
        reshape_7 = None
        attn_output_27 = torch._C._nn.linear(
            attn_output_26,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_26 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_attn_modules_out_proj_parameters_bias_ = (None)
        msg_token_14 = msg_token_13 + attn_output_27
        msg_token_13 = attn_output_27 = None
        msg_token_15 = msg_token_14.view(-1, 1, 768)
        msg_token_14 = None
        hidden_states_28 = torch.cat([hidden_states_27, msg_token_15], dim=1)
        hidden_states_27 = msg_token_15 = None
        item_19 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_eps = (
            None
        )
        hidden_states_29 = torch.nn.functional.layer_norm(
            hidden_states_28,
            (768,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_bias_,
            item_19,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_bias_ = (item_19) = (
            None
        )
        size_13 = hidden_states_29.size()
        getitem_49 = size_13[0]
        size_13 = None
        queries_14 = torch._C._nn.linear(
            hidden_states_29,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_14 = torch._C._nn.linear(
            hidden_states_29,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_14 = torch._C._nn.linear(
            hidden_states_29,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_29 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_29 = queries_14.view(getitem_49, 198, 12, 64)
        queries_14 = None
        queries_15 = view_29.transpose(1, 2)
        view_29 = None
        view_30 = keys_14.view(getitem_49, 198, 12, 64)
        keys_14 = None
        keys_15 = view_30.transpose(1, 2)
        view_30 = None
        view_31 = values_14.view(getitem_49, 198, 12, 64)
        values_14 = None
        values_15 = view_31.transpose(1, 2)
        view_31 = None
        transpose_39 = keys_15.transpose(-1, -2)
        keys_15 = None
        matmul_14 = torch.matmul(queries_15, transpose_39)
        queries_15 = transpose_39 = None
        item_20 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_scale = (
            None
        )
        attn_weights_21 = matmul_14 * item_20
        matmul_14 = item_20 = None
        softmax_7 = torch.nn.functional.softmax(
            attn_weights_21, dim=-1, dtype=torch.float32
        )
        attn_weights_21 = None
        attn_weights_22 = softmax_7.to(torch.float32)
        softmax_7 = None
        attn_weights_23 = torch.nn.functional.dropout(
            attn_weights_22, p=0.0, training=False
        )
        attn_weights_22 = None
        attn_output_28 = torch.matmul(attn_weights_23, values_15)
        attn_weights_23 = values_15 = None
        transpose_40 = attn_output_28.transpose(1, 2)
        attn_output_28 = None
        attn_output_29 = transpose_40.contiguous()
        transpose_40 = None
        reshape_8 = attn_output_29.reshape(getitem_49, 198, 768)
        attn_output_29 = getitem_49 = None
        attn_output_30 = reshape_8.contiguous()
        reshape_8 = None
        attn_output_31 = torch._C._nn.linear(
            attn_output_30,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_30 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_30 = hidden_states_28 + attn_output_31
        hidden_states_28 = attn_output_31 = None
        hidden_states_31 = hidden_states_30[
            (slice(None, None, None), slice(None, 197, None), slice(None, None, None))
        ]
        hidden_states_30 = None
        item_21 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_eps = (
            None
        )
        hidden_states_32 = torch.nn.functional.layer_norm(
            hidden_states_31,
            (768,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_bias_,
            item_21,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_bias_ = (item_21) = (
            None
        )
        hidden_states_33 = torch._C._nn.linear(
            hidden_states_32,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_32 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_14 = 1.702 * hidden_states_33
        sigmoid_3 = torch.sigmoid(mul_14)
        mul_14 = None
        hidden_states_34 = hidden_states_33 * sigmoid_3
        hidden_states_33 = sigmoid_3 = None
        hidden_states_35 = torch._C._nn.linear(
            hidden_states_34,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_34 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_36 = hidden_states_31 + hidden_states_35
        hidden_states_31 = hidden_states_35 = None
        size_14 = hidden_states_36.size()
        getitem_53 = size_14[0]
        size_14 = None
        floordiv_4 = getitem_53 // 8
        getitem_53 = None
        getitem_56 = hidden_states_36[
            (slice(None, None, None), 0, slice(None, None, None))
        ]
        msg_token_16 = torch._C._nn.linear(
            getitem_56,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_fc_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_fc_parameters_bias_,
        )
        getitem_56 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_fc_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_fc_parameters_bias_ = (None)
        msg_token_17 = msg_token_16.view(floordiv_4, 8, 768)
        msg_token_16 = floordiv_4 = None
        item_22 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_ln_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_ln_eps = (
            None
        )
        layer_norm_13 = torch.nn.functional.layer_norm(
            msg_token_17,
            (768,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_ln_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_ln_parameters_bias_,
            item_22,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_ln_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_ln_parameters_bias_ = (item_22) = (
            None
        )
        size_15 = layer_norm_13.size()
        getitem_58 = size_15[1]
        size_15 = None
        queries_16 = torch._C._nn.linear(
            layer_norm_13,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_attn_modules_q_proj_parameters_bias_ = (None)
        keys_16 = torch._C._nn.linear(
            layer_norm_13,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_attn_modules_k_proj_parameters_bias_ = (None)
        values_16 = torch._C._nn.linear(
            layer_norm_13,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_attn_modules_v_proj_parameters_bias_,
        )
        layer_norm_13 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_attn_modules_v_proj_parameters_bias_ = (None)
        view_33 = queries_16.view(1, getitem_58, 12, 64)
        queries_16 = None
        queries_17 = view_33.transpose(1, 2)
        view_33 = None
        view_34 = keys_16.view(1, getitem_58, 12, 64)
        keys_16 = None
        keys_17 = view_34.transpose(1, 2)
        view_34 = None
        view_35 = values_16.view(1, getitem_58, 12, 64)
        values_16 = None
        values_17 = view_35.transpose(1, 2)
        view_35 = None
        transpose_44 = keys_17.transpose(-1, -2)
        keys_17 = None
        matmul_16 = torch.matmul(queries_17, transpose_44)
        queries_17 = transpose_44 = None
        item_23 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_attn_scale = (
            None
        )
        attn_weights_24 = matmul_16 * item_23
        matmul_16 = item_23 = None
        softmax_8 = torch.nn.functional.softmax(
            attn_weights_24, dim=-1, dtype=torch.float32
        )
        attn_weights_24 = None
        attn_weights_25 = softmax_8.to(torch.float32)
        softmax_8 = None
        attn_weights_26 = torch.nn.functional.dropout(
            attn_weights_25, p=0.0, training=False
        )
        attn_weights_25 = None
        attn_output_32 = torch.matmul(attn_weights_26, values_17)
        attn_weights_26 = values_17 = None
        transpose_45 = attn_output_32.transpose(1, 2)
        attn_output_32 = None
        attn_output_33 = transpose_45.contiguous()
        transpose_45 = None
        reshape_9 = attn_output_33.reshape(1, getitem_58, 768)
        attn_output_33 = getitem_58 = None
        attn_output_34 = reshape_9.contiguous()
        reshape_9 = None
        attn_output_35 = torch._C._nn.linear(
            attn_output_34,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_34 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_attn_modules_out_proj_parameters_bias_ = (None)
        msg_token_18 = msg_token_17 + attn_output_35
        msg_token_17 = attn_output_35 = None
        msg_token_19 = msg_token_18.view(-1, 1, 768)
        msg_token_18 = None
        hidden_states_37 = torch.cat([hidden_states_36, msg_token_19], dim=1)
        hidden_states_36 = msg_token_19 = None
        item_24 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_eps = (
            None
        )
        hidden_states_38 = torch.nn.functional.layer_norm(
            hidden_states_37,
            (768,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_bias_,
            item_24,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_bias_ = (item_24) = (
            None
        )
        size_16 = hidden_states_38.size()
        getitem_60 = size_16[0]
        size_16 = None
        queries_18 = torch._C._nn.linear(
            hidden_states_38,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_18 = torch._C._nn.linear(
            hidden_states_38,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_18 = torch._C._nn.linear(
            hidden_states_38,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_38 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_37 = queries_18.view(getitem_60, 198, 12, 64)
        queries_18 = None
        queries_19 = view_37.transpose(1, 2)
        view_37 = None
        view_38 = keys_18.view(getitem_60, 198, 12, 64)
        keys_18 = None
        keys_19 = view_38.transpose(1, 2)
        view_38 = None
        view_39 = values_18.view(getitem_60, 198, 12, 64)
        values_18 = None
        values_19 = view_39.transpose(1, 2)
        view_39 = None
        transpose_49 = keys_19.transpose(-1, -2)
        keys_19 = None
        matmul_18 = torch.matmul(queries_19, transpose_49)
        queries_19 = transpose_49 = None
        item_25 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_scale = (
            None
        )
        attn_weights_27 = matmul_18 * item_25
        matmul_18 = item_25 = None
        softmax_9 = torch.nn.functional.softmax(
            attn_weights_27, dim=-1, dtype=torch.float32
        )
        attn_weights_27 = None
        attn_weights_28 = softmax_9.to(torch.float32)
        softmax_9 = None
        attn_weights_29 = torch.nn.functional.dropout(
            attn_weights_28, p=0.0, training=False
        )
        attn_weights_28 = None
        attn_output_36 = torch.matmul(attn_weights_29, values_19)
        attn_weights_29 = values_19 = None
        transpose_50 = attn_output_36.transpose(1, 2)
        attn_output_36 = None
        attn_output_37 = transpose_50.contiguous()
        transpose_50 = None
        reshape_10 = attn_output_37.reshape(getitem_60, 198, 768)
        attn_output_37 = getitem_60 = None
        attn_output_38 = reshape_10.contiguous()
        reshape_10 = None
        attn_output_39 = torch._C._nn.linear(
            attn_output_38,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_38 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_39 = hidden_states_37 + attn_output_39
        hidden_states_37 = attn_output_39 = None
        hidden_states_40 = hidden_states_39[
            (slice(None, None, None), slice(None, 197, None), slice(None, None, None))
        ]
        hidden_states_39 = None
        item_26 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_eps = (
            None
        )
        hidden_states_41 = torch.nn.functional.layer_norm(
            hidden_states_40,
            (768,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_bias_,
            item_26,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_bias_ = (item_26) = (
            None
        )
        hidden_states_42 = torch._C._nn.linear(
            hidden_states_41,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_41 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_18 = 1.702 * hidden_states_42
        sigmoid_4 = torch.sigmoid(mul_18)
        mul_18 = None
        hidden_states_43 = hidden_states_42 * sigmoid_4
        hidden_states_42 = sigmoid_4 = None
        hidden_states_44 = torch._C._nn.linear(
            hidden_states_43,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_43 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_45 = hidden_states_40 + hidden_states_44
        hidden_states_40 = hidden_states_44 = None
        size_17 = hidden_states_45.size()
        getitem_64 = size_17[0]
        size_17 = None
        floordiv_5 = getitem_64 // 8
        getitem_64 = None
        getitem_67 = hidden_states_45[
            (slice(None, None, None), 0, slice(None, None, None))
        ]
        msg_token_20 = torch._C._nn.linear(
            getitem_67,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_fc_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_fc_parameters_bias_,
        )
        getitem_67 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_fc_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_fc_parameters_bias_ = (None)
        msg_token_21 = msg_token_20.view(floordiv_5, 8, 768)
        msg_token_20 = floordiv_5 = None
        item_27 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_ln_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_ln_eps = (
            None
        )
        layer_norm_16 = torch.nn.functional.layer_norm(
            msg_token_21,
            (768,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_ln_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_ln_parameters_bias_,
            item_27,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_ln_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_ln_parameters_bias_ = (item_27) = (
            None
        )
        size_18 = layer_norm_16.size()
        getitem_69 = size_18[1]
        size_18 = None
        queries_20 = torch._C._nn.linear(
            layer_norm_16,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_attn_modules_q_proj_parameters_bias_ = (None)
        keys_20 = torch._C._nn.linear(
            layer_norm_16,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_attn_modules_k_proj_parameters_bias_ = (None)
        values_20 = torch._C._nn.linear(
            layer_norm_16,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_attn_modules_v_proj_parameters_bias_,
        )
        layer_norm_16 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_attn_modules_v_proj_parameters_bias_ = (None)
        view_41 = queries_20.view(1, getitem_69, 12, 64)
        queries_20 = None
        queries_21 = view_41.transpose(1, 2)
        view_41 = None
        view_42 = keys_20.view(1, getitem_69, 12, 64)
        keys_20 = None
        keys_21 = view_42.transpose(1, 2)
        view_42 = None
        view_43 = values_20.view(1, getitem_69, 12, 64)
        values_20 = None
        values_21 = view_43.transpose(1, 2)
        view_43 = None
        transpose_54 = keys_21.transpose(-1, -2)
        keys_21 = None
        matmul_20 = torch.matmul(queries_21, transpose_54)
        queries_21 = transpose_54 = None
        item_28 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_attn_scale = (
            None
        )
        attn_weights_30 = matmul_20 * item_28
        matmul_20 = item_28 = None
        softmax_10 = torch.nn.functional.softmax(
            attn_weights_30, dim=-1, dtype=torch.float32
        )
        attn_weights_30 = None
        attn_weights_31 = softmax_10.to(torch.float32)
        softmax_10 = None
        attn_weights_32 = torch.nn.functional.dropout(
            attn_weights_31, p=0.0, training=False
        )
        attn_weights_31 = None
        attn_output_40 = torch.matmul(attn_weights_32, values_21)
        attn_weights_32 = values_21 = None
        transpose_55 = attn_output_40.transpose(1, 2)
        attn_output_40 = None
        attn_output_41 = transpose_55.contiguous()
        transpose_55 = None
        reshape_11 = attn_output_41.reshape(1, getitem_69, 768)
        attn_output_41 = getitem_69 = None
        attn_output_42 = reshape_11.contiguous()
        reshape_11 = None
        attn_output_43 = torch._C._nn.linear(
            attn_output_42,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_42 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_attn_modules_out_proj_parameters_bias_ = (None)
        msg_token_22 = msg_token_21 + attn_output_43
        msg_token_21 = attn_output_43 = None
        msg_token_23 = msg_token_22.view(-1, 1, 768)
        msg_token_22 = None
        hidden_states_46 = torch.cat([hidden_states_45, msg_token_23], dim=1)
        hidden_states_45 = msg_token_23 = None
        item_29 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_eps = (
            None
        )
        hidden_states_47 = torch.nn.functional.layer_norm(
            hidden_states_46,
            (768,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_bias_,
            item_29,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_bias_ = (item_29) = (
            None
        )
        size_19 = hidden_states_47.size()
        getitem_71 = size_19[0]
        size_19 = None
        queries_22 = torch._C._nn.linear(
            hidden_states_47,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_22 = torch._C._nn.linear(
            hidden_states_47,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_22 = torch._C._nn.linear(
            hidden_states_47,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_47 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_45 = queries_22.view(getitem_71, 198, 12, 64)
        queries_22 = None
        queries_23 = view_45.transpose(1, 2)
        view_45 = None
        view_46 = keys_22.view(getitem_71, 198, 12, 64)
        keys_22 = None
        keys_23 = view_46.transpose(1, 2)
        view_46 = None
        view_47 = values_22.view(getitem_71, 198, 12, 64)
        values_22 = None
        values_23 = view_47.transpose(1, 2)
        view_47 = None
        transpose_59 = keys_23.transpose(-1, -2)
        keys_23 = None
        matmul_22 = torch.matmul(queries_23, transpose_59)
        queries_23 = transpose_59 = None
        item_30 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_scale = (
            None
        )
        attn_weights_33 = matmul_22 * item_30
        matmul_22 = item_30 = None
        softmax_11 = torch.nn.functional.softmax(
            attn_weights_33, dim=-1, dtype=torch.float32
        )
        attn_weights_33 = None
        attn_weights_34 = softmax_11.to(torch.float32)
        softmax_11 = None
        attn_weights_35 = torch.nn.functional.dropout(
            attn_weights_34, p=0.0, training=False
        )
        attn_weights_34 = None
        attn_output_44 = torch.matmul(attn_weights_35, values_23)
        attn_weights_35 = values_23 = None
        transpose_60 = attn_output_44.transpose(1, 2)
        attn_output_44 = None
        attn_output_45 = transpose_60.contiguous()
        transpose_60 = None
        reshape_12 = attn_output_45.reshape(getitem_71, 198, 768)
        attn_output_45 = getitem_71 = None
        attn_output_46 = reshape_12.contiguous()
        reshape_12 = None
        attn_output_47 = torch._C._nn.linear(
            attn_output_46,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_46 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_48 = hidden_states_46 + attn_output_47
        hidden_states_46 = attn_output_47 = None
        hidden_states_49 = hidden_states_48[
            (slice(None, None, None), slice(None, 197, None), slice(None, None, None))
        ]
        hidden_states_48 = None
        item_31 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_eps = (
            None
        )
        hidden_states_50 = torch.nn.functional.layer_norm(
            hidden_states_49,
            (768,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_bias_,
            item_31,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_bias_ = (item_31) = (
            None
        )
        hidden_states_51 = torch._C._nn.linear(
            hidden_states_50,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_50 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_22 = 1.702 * hidden_states_51
        sigmoid_5 = torch.sigmoid(mul_22)
        mul_22 = None
        hidden_states_52 = hidden_states_51 * sigmoid_5
        hidden_states_51 = sigmoid_5 = None
        hidden_states_53 = torch._C._nn.linear(
            hidden_states_52,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_52 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_54 = hidden_states_49 + hidden_states_53
        hidden_states_49 = hidden_states_53 = None
        size_20 = hidden_states_54.size()
        getitem_75 = size_20[0]
        size_20 = None
        floordiv_6 = getitem_75 // 8
        getitem_75 = None
        getitem_78 = hidden_states_54[
            (slice(None, None, None), 0, slice(None, None, None))
        ]
        msg_token_24 = torch._C._nn.linear(
            getitem_78,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_fc_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_fc_parameters_bias_,
        )
        getitem_78 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_fc_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_fc_parameters_bias_ = (None)
        msg_token_25 = msg_token_24.view(floordiv_6, 8, 768)
        msg_token_24 = floordiv_6 = None
        item_32 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_ln_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_ln_eps = (
            None
        )
        layer_norm_19 = torch.nn.functional.layer_norm(
            msg_token_25,
            (768,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_ln_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_ln_parameters_bias_,
            item_32,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_ln_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_ln_parameters_bias_ = (item_32) = (
            None
        )
        size_21 = layer_norm_19.size()
        getitem_80 = size_21[1]
        size_21 = None
        queries_24 = torch._C._nn.linear(
            layer_norm_19,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_attn_modules_q_proj_parameters_bias_ = (None)
        keys_24 = torch._C._nn.linear(
            layer_norm_19,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_attn_modules_k_proj_parameters_bias_ = (None)
        values_24 = torch._C._nn.linear(
            layer_norm_19,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_attn_modules_v_proj_parameters_bias_,
        )
        layer_norm_19 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_attn_modules_v_proj_parameters_bias_ = (None)
        view_49 = queries_24.view(1, getitem_80, 12, 64)
        queries_24 = None
        queries_25 = view_49.transpose(1, 2)
        view_49 = None
        view_50 = keys_24.view(1, getitem_80, 12, 64)
        keys_24 = None
        keys_25 = view_50.transpose(1, 2)
        view_50 = None
        view_51 = values_24.view(1, getitem_80, 12, 64)
        values_24 = None
        values_25 = view_51.transpose(1, 2)
        view_51 = None
        transpose_64 = keys_25.transpose(-1, -2)
        keys_25 = None
        matmul_24 = torch.matmul(queries_25, transpose_64)
        queries_25 = transpose_64 = None
        item_33 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_attn_scale = (
            None
        )
        attn_weights_36 = matmul_24 * item_33
        matmul_24 = item_33 = None
        softmax_12 = torch.nn.functional.softmax(
            attn_weights_36, dim=-1, dtype=torch.float32
        )
        attn_weights_36 = None
        attn_weights_37 = softmax_12.to(torch.float32)
        softmax_12 = None
        attn_weights_38 = torch.nn.functional.dropout(
            attn_weights_37, p=0.0, training=False
        )
        attn_weights_37 = None
        attn_output_48 = torch.matmul(attn_weights_38, values_25)
        attn_weights_38 = values_25 = None
        transpose_65 = attn_output_48.transpose(1, 2)
        attn_output_48 = None
        attn_output_49 = transpose_65.contiguous()
        transpose_65 = None
        reshape_13 = attn_output_49.reshape(1, getitem_80, 768)
        attn_output_49 = getitem_80 = None
        attn_output_50 = reshape_13.contiguous()
        reshape_13 = None
        attn_output_51 = torch._C._nn.linear(
            attn_output_50,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_50 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_attn_modules_out_proj_parameters_bias_ = (None)
        msg_token_26 = msg_token_25 + attn_output_51
        msg_token_25 = attn_output_51 = None
        msg_token_27 = msg_token_26.view(-1, 1, 768)
        msg_token_26 = None
        hidden_states_55 = torch.cat([hidden_states_54, msg_token_27], dim=1)
        hidden_states_54 = msg_token_27 = None
        item_34 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_eps = (
            None
        )
        hidden_states_56 = torch.nn.functional.layer_norm(
            hidden_states_55,
            (768,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_bias_,
            item_34,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_bias_ = (item_34) = (
            None
        )
        size_22 = hidden_states_56.size()
        getitem_82 = size_22[0]
        size_22 = None
        queries_26 = torch._C._nn.linear(
            hidden_states_56,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_26 = torch._C._nn.linear(
            hidden_states_56,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_26 = torch._C._nn.linear(
            hidden_states_56,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_56 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_53 = queries_26.view(getitem_82, 198, 12, 64)
        queries_26 = None
        queries_27 = view_53.transpose(1, 2)
        view_53 = None
        view_54 = keys_26.view(getitem_82, 198, 12, 64)
        keys_26 = None
        keys_27 = view_54.transpose(1, 2)
        view_54 = None
        view_55 = values_26.view(getitem_82, 198, 12, 64)
        values_26 = None
        values_27 = view_55.transpose(1, 2)
        view_55 = None
        transpose_69 = keys_27.transpose(-1, -2)
        keys_27 = None
        matmul_26 = torch.matmul(queries_27, transpose_69)
        queries_27 = transpose_69 = None
        item_35 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_scale = (
            None
        )
        attn_weights_39 = matmul_26 * item_35
        matmul_26 = item_35 = None
        softmax_13 = torch.nn.functional.softmax(
            attn_weights_39, dim=-1, dtype=torch.float32
        )
        attn_weights_39 = None
        attn_weights_40 = softmax_13.to(torch.float32)
        softmax_13 = None
        attn_weights_41 = torch.nn.functional.dropout(
            attn_weights_40, p=0.0, training=False
        )
        attn_weights_40 = None
        attn_output_52 = torch.matmul(attn_weights_41, values_27)
        attn_weights_41 = values_27 = None
        transpose_70 = attn_output_52.transpose(1, 2)
        attn_output_52 = None
        attn_output_53 = transpose_70.contiguous()
        transpose_70 = None
        reshape_14 = attn_output_53.reshape(getitem_82, 198, 768)
        attn_output_53 = getitem_82 = None
        attn_output_54 = reshape_14.contiguous()
        reshape_14 = None
        attn_output_55 = torch._C._nn.linear(
            attn_output_54,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_54 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_57 = hidden_states_55 + attn_output_55
        hidden_states_55 = attn_output_55 = None
        hidden_states_58 = hidden_states_57[
            (slice(None, None, None), slice(None, 197, None), slice(None, None, None))
        ]
        hidden_states_57 = None
        item_36 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_eps = (
            None
        )
        hidden_states_59 = torch.nn.functional.layer_norm(
            hidden_states_58,
            (768,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_bias_,
            item_36,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_bias_ = (item_36) = (
            None
        )
        hidden_states_60 = torch._C._nn.linear(
            hidden_states_59,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_59 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_26 = 1.702 * hidden_states_60
        sigmoid_6 = torch.sigmoid(mul_26)
        mul_26 = None
        hidden_states_61 = hidden_states_60 * sigmoid_6
        hidden_states_60 = sigmoid_6 = None
        hidden_states_62 = torch._C._nn.linear(
            hidden_states_61,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_61 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_63 = hidden_states_58 + hidden_states_62
        hidden_states_58 = hidden_states_62 = None
        size_23 = hidden_states_63.size()
        getitem_86 = size_23[0]
        size_23 = None
        floordiv_7 = getitem_86 // 8
        getitem_86 = None
        getitem_89 = hidden_states_63[
            (slice(None, None, None), 0, slice(None, None, None))
        ]
        msg_token_28 = torch._C._nn.linear(
            getitem_89,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_fc_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_fc_parameters_bias_,
        )
        getitem_89 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_fc_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_fc_parameters_bias_ = (None)
        msg_token_29 = msg_token_28.view(floordiv_7, 8, 768)
        msg_token_28 = floordiv_7 = None
        item_37 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_ln_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_ln_eps = (
            None
        )
        layer_norm_22 = torch.nn.functional.layer_norm(
            msg_token_29,
            (768,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_ln_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_ln_parameters_bias_,
            item_37,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_ln_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_ln_parameters_bias_ = (item_37) = (
            None
        )
        size_24 = layer_norm_22.size()
        getitem_91 = size_24[1]
        size_24 = None
        queries_28 = torch._C._nn.linear(
            layer_norm_22,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_attn_modules_q_proj_parameters_bias_ = (None)
        keys_28 = torch._C._nn.linear(
            layer_norm_22,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_attn_modules_k_proj_parameters_bias_ = (None)
        values_28 = torch._C._nn.linear(
            layer_norm_22,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_attn_modules_v_proj_parameters_bias_,
        )
        layer_norm_22 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_attn_modules_v_proj_parameters_bias_ = (None)
        view_57 = queries_28.view(1, getitem_91, 12, 64)
        queries_28 = None
        queries_29 = view_57.transpose(1, 2)
        view_57 = None
        view_58 = keys_28.view(1, getitem_91, 12, 64)
        keys_28 = None
        keys_29 = view_58.transpose(1, 2)
        view_58 = None
        view_59 = values_28.view(1, getitem_91, 12, 64)
        values_28 = None
        values_29 = view_59.transpose(1, 2)
        view_59 = None
        transpose_74 = keys_29.transpose(-1, -2)
        keys_29 = None
        matmul_28 = torch.matmul(queries_29, transpose_74)
        queries_29 = transpose_74 = None
        item_38 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_attn_scale = (
            None
        )
        attn_weights_42 = matmul_28 * item_38
        matmul_28 = item_38 = None
        softmax_14 = torch.nn.functional.softmax(
            attn_weights_42, dim=-1, dtype=torch.float32
        )
        attn_weights_42 = None
        attn_weights_43 = softmax_14.to(torch.float32)
        softmax_14 = None
        attn_weights_44 = torch.nn.functional.dropout(
            attn_weights_43, p=0.0, training=False
        )
        attn_weights_43 = None
        attn_output_56 = torch.matmul(attn_weights_44, values_29)
        attn_weights_44 = values_29 = None
        transpose_75 = attn_output_56.transpose(1, 2)
        attn_output_56 = None
        attn_output_57 = transpose_75.contiguous()
        transpose_75 = None
        reshape_15 = attn_output_57.reshape(1, getitem_91, 768)
        attn_output_57 = getitem_91 = None
        attn_output_58 = reshape_15.contiguous()
        reshape_15 = None
        attn_output_59 = torch._C._nn.linear(
            attn_output_58,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_58 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_attn_modules_out_proj_parameters_bias_ = (None)
        msg_token_30 = msg_token_29 + attn_output_59
        msg_token_29 = attn_output_59 = None
        msg_token_31 = msg_token_30.view(-1, 1, 768)
        msg_token_30 = None
        hidden_states_64 = torch.cat([hidden_states_63, msg_token_31], dim=1)
        hidden_states_63 = msg_token_31 = None
        item_39 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_eps = (
            None
        )
        hidden_states_65 = torch.nn.functional.layer_norm(
            hidden_states_64,
            (768,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_bias_,
            item_39,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_bias_ = (item_39) = (
            None
        )
        size_25 = hidden_states_65.size()
        getitem_93 = size_25[0]
        size_25 = None
        queries_30 = torch._C._nn.linear(
            hidden_states_65,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_30 = torch._C._nn.linear(
            hidden_states_65,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_30 = torch._C._nn.linear(
            hidden_states_65,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_65 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_61 = queries_30.view(getitem_93, 198, 12, 64)
        queries_30 = None
        queries_31 = view_61.transpose(1, 2)
        view_61 = None
        view_62 = keys_30.view(getitem_93, 198, 12, 64)
        keys_30 = None
        keys_31 = view_62.transpose(1, 2)
        view_62 = None
        view_63 = values_30.view(getitem_93, 198, 12, 64)
        values_30 = None
        values_31 = view_63.transpose(1, 2)
        view_63 = None
        transpose_79 = keys_31.transpose(-1, -2)
        keys_31 = None
        matmul_30 = torch.matmul(queries_31, transpose_79)
        queries_31 = transpose_79 = None
        item_40 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_scale = (
            None
        )
        attn_weights_45 = matmul_30 * item_40
        matmul_30 = item_40 = None
        softmax_15 = torch.nn.functional.softmax(
            attn_weights_45, dim=-1, dtype=torch.float32
        )
        attn_weights_45 = None
        attn_weights_46 = softmax_15.to(torch.float32)
        softmax_15 = None
        attn_weights_47 = torch.nn.functional.dropout(
            attn_weights_46, p=0.0, training=False
        )
        attn_weights_46 = None
        attn_output_60 = torch.matmul(attn_weights_47, values_31)
        attn_weights_47 = values_31 = None
        transpose_80 = attn_output_60.transpose(1, 2)
        attn_output_60 = None
        attn_output_61 = transpose_80.contiguous()
        transpose_80 = None
        reshape_16 = attn_output_61.reshape(getitem_93, 198, 768)
        attn_output_61 = getitem_93 = None
        attn_output_62 = reshape_16.contiguous()
        reshape_16 = None
        attn_output_63 = torch._C._nn.linear(
            attn_output_62,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_62 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_66 = hidden_states_64 + attn_output_63
        hidden_states_64 = attn_output_63 = None
        hidden_states_67 = hidden_states_66[
            (slice(None, None, None), slice(None, 197, None), slice(None, None, None))
        ]
        hidden_states_66 = None
        item_41 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_eps = (
            None
        )
        hidden_states_68 = torch.nn.functional.layer_norm(
            hidden_states_67,
            (768,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_bias_,
            item_41,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_bias_ = (item_41) = (
            None
        )
        hidden_states_69 = torch._C._nn.linear(
            hidden_states_68,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_68 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_30 = 1.702 * hidden_states_69
        sigmoid_7 = torch.sigmoid(mul_30)
        mul_30 = None
        hidden_states_70 = hidden_states_69 * sigmoid_7
        hidden_states_69 = sigmoid_7 = None
        hidden_states_71 = torch._C._nn.linear(
            hidden_states_70,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_70 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_72 = hidden_states_67 + hidden_states_71
        hidden_states_67 = hidden_states_71 = None
        size_26 = hidden_states_72.size()
        getitem_97 = size_26[0]
        size_26 = None
        floordiv_8 = getitem_97 // 8
        getitem_97 = None
        getitem_100 = hidden_states_72[
            (slice(None, None, None), 0, slice(None, None, None))
        ]
        msg_token_32 = torch._C._nn.linear(
            getitem_100,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_fc_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_fc_parameters_bias_,
        )
        getitem_100 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_fc_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_fc_parameters_bias_ = (None)
        msg_token_33 = msg_token_32.view(floordiv_8, 8, 768)
        msg_token_32 = floordiv_8 = None
        item_42 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_ln_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_ln_eps = (
            None
        )
        layer_norm_25 = torch.nn.functional.layer_norm(
            msg_token_33,
            (768,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_ln_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_ln_parameters_bias_,
            item_42,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_ln_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_ln_parameters_bias_ = (item_42) = (
            None
        )
        size_27 = layer_norm_25.size()
        getitem_102 = size_27[1]
        size_27 = None
        queries_32 = torch._C._nn.linear(
            layer_norm_25,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_attn_modules_q_proj_parameters_bias_ = (None)
        keys_32 = torch._C._nn.linear(
            layer_norm_25,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_attn_modules_k_proj_parameters_bias_ = (None)
        values_32 = torch._C._nn.linear(
            layer_norm_25,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_attn_modules_v_proj_parameters_bias_,
        )
        layer_norm_25 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_attn_modules_v_proj_parameters_bias_ = (None)
        view_65 = queries_32.view(1, getitem_102, 12, 64)
        queries_32 = None
        queries_33 = view_65.transpose(1, 2)
        view_65 = None
        view_66 = keys_32.view(1, getitem_102, 12, 64)
        keys_32 = None
        keys_33 = view_66.transpose(1, 2)
        view_66 = None
        view_67 = values_32.view(1, getitem_102, 12, 64)
        values_32 = None
        values_33 = view_67.transpose(1, 2)
        view_67 = None
        transpose_84 = keys_33.transpose(-1, -2)
        keys_33 = None
        matmul_32 = torch.matmul(queries_33, transpose_84)
        queries_33 = transpose_84 = None
        item_43 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_attn_scale = (
            None
        )
        attn_weights_48 = matmul_32 * item_43
        matmul_32 = item_43 = None
        softmax_16 = torch.nn.functional.softmax(
            attn_weights_48, dim=-1, dtype=torch.float32
        )
        attn_weights_48 = None
        attn_weights_49 = softmax_16.to(torch.float32)
        softmax_16 = None
        attn_weights_50 = torch.nn.functional.dropout(
            attn_weights_49, p=0.0, training=False
        )
        attn_weights_49 = None
        attn_output_64 = torch.matmul(attn_weights_50, values_33)
        attn_weights_50 = values_33 = None
        transpose_85 = attn_output_64.transpose(1, 2)
        attn_output_64 = None
        attn_output_65 = transpose_85.contiguous()
        transpose_85 = None
        reshape_17 = attn_output_65.reshape(1, getitem_102, 768)
        attn_output_65 = getitem_102 = None
        attn_output_66 = reshape_17.contiguous()
        reshape_17 = None
        attn_output_67 = torch._C._nn.linear(
            attn_output_66,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_66 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_attn_modules_out_proj_parameters_bias_ = (None)
        msg_token_34 = msg_token_33 + attn_output_67
        msg_token_33 = attn_output_67 = None
        msg_token_35 = msg_token_34.view(-1, 1, 768)
        msg_token_34 = None
        hidden_states_73 = torch.cat([hidden_states_72, msg_token_35], dim=1)
        hidden_states_72 = msg_token_35 = None
        item_44 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_eps = (
            None
        )
        hidden_states_74 = torch.nn.functional.layer_norm(
            hidden_states_73,
            (768,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_bias_,
            item_44,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_bias_ = (item_44) = (
            None
        )
        size_28 = hidden_states_74.size()
        getitem_104 = size_28[0]
        size_28 = None
        queries_34 = torch._C._nn.linear(
            hidden_states_74,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_34 = torch._C._nn.linear(
            hidden_states_74,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_34 = torch._C._nn.linear(
            hidden_states_74,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_74 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_69 = queries_34.view(getitem_104, 198, 12, 64)
        queries_34 = None
        queries_35 = view_69.transpose(1, 2)
        view_69 = None
        view_70 = keys_34.view(getitem_104, 198, 12, 64)
        keys_34 = None
        keys_35 = view_70.transpose(1, 2)
        view_70 = None
        view_71 = values_34.view(getitem_104, 198, 12, 64)
        values_34 = None
        values_35 = view_71.transpose(1, 2)
        view_71 = None
        transpose_89 = keys_35.transpose(-1, -2)
        keys_35 = None
        matmul_34 = torch.matmul(queries_35, transpose_89)
        queries_35 = transpose_89 = None
        item_45 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_scale = (
            None
        )
        attn_weights_51 = matmul_34 * item_45
        matmul_34 = item_45 = None
        softmax_17 = torch.nn.functional.softmax(
            attn_weights_51, dim=-1, dtype=torch.float32
        )
        attn_weights_51 = None
        attn_weights_52 = softmax_17.to(torch.float32)
        softmax_17 = None
        attn_weights_53 = torch.nn.functional.dropout(
            attn_weights_52, p=0.0, training=False
        )
        attn_weights_52 = None
        attn_output_68 = torch.matmul(attn_weights_53, values_35)
        attn_weights_53 = values_35 = None
        transpose_90 = attn_output_68.transpose(1, 2)
        attn_output_68 = None
        attn_output_69 = transpose_90.contiguous()
        transpose_90 = None
        reshape_18 = attn_output_69.reshape(getitem_104, 198, 768)
        attn_output_69 = getitem_104 = None
        attn_output_70 = reshape_18.contiguous()
        reshape_18 = None
        attn_output_71 = torch._C._nn.linear(
            attn_output_70,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_70 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_75 = hidden_states_73 + attn_output_71
        hidden_states_73 = attn_output_71 = None
        hidden_states_76 = hidden_states_75[
            (slice(None, None, None), slice(None, 197, None), slice(None, None, None))
        ]
        hidden_states_75 = None
        item_46 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_eps = (
            None
        )
        hidden_states_77 = torch.nn.functional.layer_norm(
            hidden_states_76,
            (768,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_bias_,
            item_46,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_bias_ = (item_46) = (
            None
        )
        hidden_states_78 = torch._C._nn.linear(
            hidden_states_77,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_77 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_34 = 1.702 * hidden_states_78
        sigmoid_8 = torch.sigmoid(mul_34)
        mul_34 = None
        hidden_states_79 = hidden_states_78 * sigmoid_8
        hidden_states_78 = sigmoid_8 = None
        hidden_states_80 = torch._C._nn.linear(
            hidden_states_79,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_79 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_81 = hidden_states_76 + hidden_states_80
        hidden_states_76 = hidden_states_80 = None
        size_29 = hidden_states_81.size()
        getitem_108 = size_29[0]
        size_29 = None
        floordiv_9 = getitem_108 // 8
        getitem_108 = None
        getitem_111 = hidden_states_81[
            (slice(None, None, None), 0, slice(None, None, None))
        ]
        msg_token_36 = torch._C._nn.linear(
            getitem_111,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_fc_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_fc_parameters_bias_,
        )
        getitem_111 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_fc_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_fc_parameters_bias_ = (None)
        msg_token_37 = msg_token_36.view(floordiv_9, 8, 768)
        msg_token_36 = floordiv_9 = None
        item_47 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_ln_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_ln_eps = (
            None
        )
        layer_norm_28 = torch.nn.functional.layer_norm(
            msg_token_37,
            (768,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_ln_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_ln_parameters_bias_,
            item_47,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_ln_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_ln_parameters_bias_ = (item_47) = (
            None
        )
        size_30 = layer_norm_28.size()
        getitem_113 = size_30[1]
        size_30 = None
        queries_36 = torch._C._nn.linear(
            layer_norm_28,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_attn_modules_q_proj_parameters_bias_ = (None)
        keys_36 = torch._C._nn.linear(
            layer_norm_28,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_attn_modules_k_proj_parameters_bias_ = (None)
        values_36 = torch._C._nn.linear(
            layer_norm_28,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_attn_modules_v_proj_parameters_bias_,
        )
        layer_norm_28 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_attn_modules_v_proj_parameters_bias_ = (None)
        view_73 = queries_36.view(1, getitem_113, 12, 64)
        queries_36 = None
        queries_37 = view_73.transpose(1, 2)
        view_73 = None
        view_74 = keys_36.view(1, getitem_113, 12, 64)
        keys_36 = None
        keys_37 = view_74.transpose(1, 2)
        view_74 = None
        view_75 = values_36.view(1, getitem_113, 12, 64)
        values_36 = None
        values_37 = view_75.transpose(1, 2)
        view_75 = None
        transpose_94 = keys_37.transpose(-1, -2)
        keys_37 = None
        matmul_36 = torch.matmul(queries_37, transpose_94)
        queries_37 = transpose_94 = None
        item_48 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_attn_scale = (
            None
        )
        attn_weights_54 = matmul_36 * item_48
        matmul_36 = item_48 = None
        softmax_18 = torch.nn.functional.softmax(
            attn_weights_54, dim=-1, dtype=torch.float32
        )
        attn_weights_54 = None
        attn_weights_55 = softmax_18.to(torch.float32)
        softmax_18 = None
        attn_weights_56 = torch.nn.functional.dropout(
            attn_weights_55, p=0.0, training=False
        )
        attn_weights_55 = None
        attn_output_72 = torch.matmul(attn_weights_56, values_37)
        attn_weights_56 = values_37 = None
        transpose_95 = attn_output_72.transpose(1, 2)
        attn_output_72 = None
        attn_output_73 = transpose_95.contiguous()
        transpose_95 = None
        reshape_19 = attn_output_73.reshape(1, getitem_113, 768)
        attn_output_73 = getitem_113 = None
        attn_output_74 = reshape_19.contiguous()
        reshape_19 = None
        attn_output_75 = torch._C._nn.linear(
            attn_output_74,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_74 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_attn_modules_out_proj_parameters_bias_ = (None)
        msg_token_38 = msg_token_37 + attn_output_75
        msg_token_37 = attn_output_75 = None
        msg_token_39 = msg_token_38.view(-1, 1, 768)
        msg_token_38 = None
        hidden_states_82 = torch.cat([hidden_states_81, msg_token_39], dim=1)
        hidden_states_81 = msg_token_39 = None
        item_49 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_eps = (
            None
        )
        hidden_states_83 = torch.nn.functional.layer_norm(
            hidden_states_82,
            (768,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_bias_,
            item_49,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_bias_ = (item_49) = (
            None
        )
        size_31 = hidden_states_83.size()
        getitem_115 = size_31[0]
        size_31 = None
        queries_38 = torch._C._nn.linear(
            hidden_states_83,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_38 = torch._C._nn.linear(
            hidden_states_83,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_38 = torch._C._nn.linear(
            hidden_states_83,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_83 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_77 = queries_38.view(getitem_115, 198, 12, 64)
        queries_38 = None
        queries_39 = view_77.transpose(1, 2)
        view_77 = None
        view_78 = keys_38.view(getitem_115, 198, 12, 64)
        keys_38 = None
        keys_39 = view_78.transpose(1, 2)
        view_78 = None
        view_79 = values_38.view(getitem_115, 198, 12, 64)
        values_38 = None
        values_39 = view_79.transpose(1, 2)
        view_79 = None
        transpose_99 = keys_39.transpose(-1, -2)
        keys_39 = None
        matmul_38 = torch.matmul(queries_39, transpose_99)
        queries_39 = transpose_99 = None
        item_50 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_scale = (
            None
        )
        attn_weights_57 = matmul_38 * item_50
        matmul_38 = item_50 = None
        softmax_19 = torch.nn.functional.softmax(
            attn_weights_57, dim=-1, dtype=torch.float32
        )
        attn_weights_57 = None
        attn_weights_58 = softmax_19.to(torch.float32)
        softmax_19 = None
        attn_weights_59 = torch.nn.functional.dropout(
            attn_weights_58, p=0.0, training=False
        )
        attn_weights_58 = None
        attn_output_76 = torch.matmul(attn_weights_59, values_39)
        attn_weights_59 = values_39 = None
        transpose_100 = attn_output_76.transpose(1, 2)
        attn_output_76 = None
        attn_output_77 = transpose_100.contiguous()
        transpose_100 = None
        reshape_20 = attn_output_77.reshape(getitem_115, 198, 768)
        attn_output_77 = getitem_115 = None
        attn_output_78 = reshape_20.contiguous()
        reshape_20 = None
        attn_output_79 = torch._C._nn.linear(
            attn_output_78,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_78 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_84 = hidden_states_82 + attn_output_79
        hidden_states_82 = attn_output_79 = None
        hidden_states_85 = hidden_states_84[
            (slice(None, None, None), slice(None, 197, None), slice(None, None, None))
        ]
        hidden_states_84 = None
        item_51 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_eps = (
            None
        )
        hidden_states_86 = torch.nn.functional.layer_norm(
            hidden_states_85,
            (768,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_bias_,
            item_51,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_bias_ = (item_51) = (
            None
        )
        hidden_states_87 = torch._C._nn.linear(
            hidden_states_86,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_86 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_38 = 1.702 * hidden_states_87
        sigmoid_9 = torch.sigmoid(mul_38)
        mul_38 = None
        hidden_states_88 = hidden_states_87 * sigmoid_9
        hidden_states_87 = sigmoid_9 = None
        hidden_states_89 = torch._C._nn.linear(
            hidden_states_88,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_88 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_90 = hidden_states_85 + hidden_states_89
        hidden_states_85 = hidden_states_89 = None
        size_32 = hidden_states_90.size()
        getitem_119 = size_32[0]
        size_32 = None
        floordiv_10 = getitem_119 // 8
        getitem_119 = None
        getitem_122 = hidden_states_90[
            (slice(None, None, None), 0, slice(None, None, None))
        ]
        msg_token_40 = torch._C._nn.linear(
            getitem_122,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_fc_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_fc_parameters_bias_,
        )
        getitem_122 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_fc_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_fc_parameters_bias_ = (None)
        msg_token_41 = msg_token_40.view(floordiv_10, 8, 768)
        msg_token_40 = floordiv_10 = None
        item_52 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_ln_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_ln_eps = (
            None
        )
        layer_norm_31 = torch.nn.functional.layer_norm(
            msg_token_41,
            (768,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_ln_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_ln_parameters_bias_,
            item_52,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_ln_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_ln_parameters_bias_ = (item_52) = (
            None
        )
        size_33 = layer_norm_31.size()
        getitem_124 = size_33[1]
        size_33 = None
        queries_40 = torch._C._nn.linear(
            layer_norm_31,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_attn_modules_q_proj_parameters_bias_ = (None)
        keys_40 = torch._C._nn.linear(
            layer_norm_31,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_attn_modules_k_proj_parameters_bias_ = (None)
        values_40 = torch._C._nn.linear(
            layer_norm_31,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_attn_modules_v_proj_parameters_bias_,
        )
        layer_norm_31 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_attn_modules_v_proj_parameters_bias_ = (None)
        view_81 = queries_40.view(1, getitem_124, 12, 64)
        queries_40 = None
        queries_41 = view_81.transpose(1, 2)
        view_81 = None
        view_82 = keys_40.view(1, getitem_124, 12, 64)
        keys_40 = None
        keys_41 = view_82.transpose(1, 2)
        view_82 = None
        view_83 = values_40.view(1, getitem_124, 12, 64)
        values_40 = None
        values_41 = view_83.transpose(1, 2)
        view_83 = None
        transpose_104 = keys_41.transpose(-1, -2)
        keys_41 = None
        matmul_40 = torch.matmul(queries_41, transpose_104)
        queries_41 = transpose_104 = None
        item_53 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_attn_scale = (
            None
        )
        attn_weights_60 = matmul_40 * item_53
        matmul_40 = item_53 = None
        softmax_20 = torch.nn.functional.softmax(
            attn_weights_60, dim=-1, dtype=torch.float32
        )
        attn_weights_60 = None
        attn_weights_61 = softmax_20.to(torch.float32)
        softmax_20 = None
        attn_weights_62 = torch.nn.functional.dropout(
            attn_weights_61, p=0.0, training=False
        )
        attn_weights_61 = None
        attn_output_80 = torch.matmul(attn_weights_62, values_41)
        attn_weights_62 = values_41 = None
        transpose_105 = attn_output_80.transpose(1, 2)
        attn_output_80 = None
        attn_output_81 = transpose_105.contiguous()
        transpose_105 = None
        reshape_21 = attn_output_81.reshape(1, getitem_124, 768)
        attn_output_81 = getitem_124 = None
        attn_output_82 = reshape_21.contiguous()
        reshape_21 = None
        attn_output_83 = torch._C._nn.linear(
            attn_output_82,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_82 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_attn_modules_out_proj_parameters_bias_ = (None)
        msg_token_42 = msg_token_41 + attn_output_83
        msg_token_41 = attn_output_83 = None
        msg_token_43 = msg_token_42.view(-1, 1, 768)
        msg_token_42 = None
        hidden_states_91 = torch.cat([hidden_states_90, msg_token_43], dim=1)
        hidden_states_90 = msg_token_43 = None
        item_54 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_eps = (
            None
        )
        hidden_states_92 = torch.nn.functional.layer_norm(
            hidden_states_91,
            (768,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_bias_,
            item_54,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_bias_ = (item_54) = (
            None
        )
        size_34 = hidden_states_92.size()
        getitem_126 = size_34[0]
        size_34 = None
        queries_42 = torch._C._nn.linear(
            hidden_states_92,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_42 = torch._C._nn.linear(
            hidden_states_92,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_42 = torch._C._nn.linear(
            hidden_states_92,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_92 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_85 = queries_42.view(getitem_126, 198, 12, 64)
        queries_42 = None
        queries_43 = view_85.transpose(1, 2)
        view_85 = None
        view_86 = keys_42.view(getitem_126, 198, 12, 64)
        keys_42 = None
        keys_43 = view_86.transpose(1, 2)
        view_86 = None
        view_87 = values_42.view(getitem_126, 198, 12, 64)
        values_42 = None
        values_43 = view_87.transpose(1, 2)
        view_87 = None
        transpose_109 = keys_43.transpose(-1, -2)
        keys_43 = None
        matmul_42 = torch.matmul(queries_43, transpose_109)
        queries_43 = transpose_109 = None
        item_55 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_scale = (
            None
        )
        attn_weights_63 = matmul_42 * item_55
        matmul_42 = item_55 = None
        softmax_21 = torch.nn.functional.softmax(
            attn_weights_63, dim=-1, dtype=torch.float32
        )
        attn_weights_63 = None
        attn_weights_64 = softmax_21.to(torch.float32)
        softmax_21 = None
        attn_weights_65 = torch.nn.functional.dropout(
            attn_weights_64, p=0.0, training=False
        )
        attn_weights_64 = None
        attn_output_84 = torch.matmul(attn_weights_65, values_43)
        attn_weights_65 = values_43 = None
        transpose_110 = attn_output_84.transpose(1, 2)
        attn_output_84 = None
        attn_output_85 = transpose_110.contiguous()
        transpose_110 = None
        reshape_22 = attn_output_85.reshape(getitem_126, 198, 768)
        attn_output_85 = getitem_126 = None
        attn_output_86 = reshape_22.contiguous()
        reshape_22 = None
        attn_output_87 = torch._C._nn.linear(
            attn_output_86,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_86 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_93 = hidden_states_91 + attn_output_87
        hidden_states_91 = attn_output_87 = None
        hidden_states_94 = hidden_states_93[
            (slice(None, None, None), slice(None, 197, None), slice(None, None, None))
        ]
        hidden_states_93 = None
        item_56 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_eps = (
            None
        )
        hidden_states_95 = torch.nn.functional.layer_norm(
            hidden_states_94,
            (768,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_bias_,
            item_56,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_bias_ = (item_56) = (
            None
        )
        hidden_states_96 = torch._C._nn.linear(
            hidden_states_95,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_95 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_42 = 1.702 * hidden_states_96
        sigmoid_10 = torch.sigmoid(mul_42)
        mul_42 = None
        hidden_states_97 = hidden_states_96 * sigmoid_10
        hidden_states_96 = sigmoid_10 = None
        hidden_states_98 = torch._C._nn.linear(
            hidden_states_97,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_97 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_99 = hidden_states_94 + hidden_states_98
        hidden_states_94 = hidden_states_98 = None
        size_35 = hidden_states_99.size()
        getitem_130 = size_35[0]
        size_35 = None
        floordiv_11 = getitem_130 // 8
        getitem_130 = None
        getitem_133 = hidden_states_99[
            (slice(None, None, None), 0, slice(None, None, None))
        ]
        msg_token_44 = torch._C._nn.linear(
            getitem_133,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_fc_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_fc_parameters_bias_,
        )
        getitem_133 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_fc_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_fc_parameters_bias_ = (None)
        msg_token_45 = msg_token_44.view(floordiv_11, 8, 768)
        msg_token_44 = floordiv_11 = None
        item_57 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_ln_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_ln_eps = (
            None
        )
        layer_norm_34 = torch.nn.functional.layer_norm(
            msg_token_45,
            (768,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_ln_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_ln_parameters_bias_,
            item_57,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_ln_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_ln_parameters_bias_ = (item_57) = (
            None
        )
        size_36 = layer_norm_34.size()
        getitem_135 = size_36[1]
        size_36 = None
        queries_44 = torch._C._nn.linear(
            layer_norm_34,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_attn_modules_q_proj_parameters_bias_ = (None)
        keys_44 = torch._C._nn.linear(
            layer_norm_34,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_attn_modules_k_proj_parameters_bias_ = (None)
        values_44 = torch._C._nn.linear(
            layer_norm_34,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_attn_modules_v_proj_parameters_bias_,
        )
        layer_norm_34 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_attn_modules_v_proj_parameters_bias_ = (None)
        view_89 = queries_44.view(1, getitem_135, 12, 64)
        queries_44 = None
        queries_45 = view_89.transpose(1, 2)
        view_89 = None
        view_90 = keys_44.view(1, getitem_135, 12, 64)
        keys_44 = None
        keys_45 = view_90.transpose(1, 2)
        view_90 = None
        view_91 = values_44.view(1, getitem_135, 12, 64)
        values_44 = None
        values_45 = view_91.transpose(1, 2)
        view_91 = None
        transpose_114 = keys_45.transpose(-1, -2)
        keys_45 = None
        matmul_44 = torch.matmul(queries_45, transpose_114)
        queries_45 = transpose_114 = None
        item_58 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_attn_scale = (
            None
        )
        attn_weights_66 = matmul_44 * item_58
        matmul_44 = item_58 = None
        softmax_22 = torch.nn.functional.softmax(
            attn_weights_66, dim=-1, dtype=torch.float32
        )
        attn_weights_66 = None
        attn_weights_67 = softmax_22.to(torch.float32)
        softmax_22 = None
        attn_weights_68 = torch.nn.functional.dropout(
            attn_weights_67, p=0.0, training=False
        )
        attn_weights_67 = None
        attn_output_88 = torch.matmul(attn_weights_68, values_45)
        attn_weights_68 = values_45 = None
        transpose_115 = attn_output_88.transpose(1, 2)
        attn_output_88 = None
        attn_output_89 = transpose_115.contiguous()
        transpose_115 = None
        reshape_23 = attn_output_89.reshape(1, getitem_135, 768)
        attn_output_89 = getitem_135 = None
        attn_output_90 = reshape_23.contiguous()
        reshape_23 = None
        attn_output_91 = torch._C._nn.linear(
            attn_output_90,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_90 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_attn_modules_out_proj_parameters_bias_ = (None)
        msg_token_46 = msg_token_45 + attn_output_91
        msg_token_45 = attn_output_91 = None
        msg_token_47 = msg_token_46.view(-1, 1, 768)
        msg_token_46 = None
        hidden_states_100 = torch.cat([hidden_states_99, msg_token_47], dim=1)
        hidden_states_99 = msg_token_47 = None
        item_59 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_eps = (
            None
        )
        hidden_states_101 = torch.nn.functional.layer_norm(
            hidden_states_100,
            (768,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_bias_,
            item_59,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_bias_ = (item_59) = (
            None
        )
        size_37 = hidden_states_101.size()
        getitem_137 = size_37[0]
        size_37 = None
        queries_46 = torch._C._nn.linear(
            hidden_states_101,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_46 = torch._C._nn.linear(
            hidden_states_101,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_46 = torch._C._nn.linear(
            hidden_states_101,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_101 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_93 = queries_46.view(getitem_137, 198, 12, 64)
        queries_46 = None
        queries_47 = view_93.transpose(1, 2)
        view_93 = None
        view_94 = keys_46.view(getitem_137, 198, 12, 64)
        keys_46 = None
        keys_47 = view_94.transpose(1, 2)
        view_94 = None
        view_95 = values_46.view(getitem_137, 198, 12, 64)
        values_46 = None
        values_47 = view_95.transpose(1, 2)
        view_95 = None
        transpose_119 = keys_47.transpose(-1, -2)
        keys_47 = None
        matmul_46 = torch.matmul(queries_47, transpose_119)
        queries_47 = transpose_119 = None
        item_60 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_scale = (
            None
        )
        attn_weights_69 = matmul_46 * item_60
        matmul_46 = item_60 = None
        softmax_23 = torch.nn.functional.softmax(
            attn_weights_69, dim=-1, dtype=torch.float32
        )
        attn_weights_69 = None
        attn_weights_70 = softmax_23.to(torch.float32)
        softmax_23 = None
        attn_weights_71 = torch.nn.functional.dropout(
            attn_weights_70, p=0.0, training=False
        )
        attn_weights_70 = None
        attn_output_92 = torch.matmul(attn_weights_71, values_47)
        attn_weights_71 = values_47 = None
        transpose_120 = attn_output_92.transpose(1, 2)
        attn_output_92 = None
        attn_output_93 = transpose_120.contiguous()
        transpose_120 = None
        reshape_24 = attn_output_93.reshape(getitem_137, 198, 768)
        attn_output_93 = getitem_137 = None
        attn_output_94 = reshape_24.contiguous()
        reshape_24 = None
        attn_output_95 = torch._C._nn.linear(
            attn_output_94,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_94 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_102 = hidden_states_100 + attn_output_95
        hidden_states_100 = attn_output_95 = None
        hidden_states_103 = hidden_states_102[
            (slice(None, None, None), slice(None, 197, None), slice(None, None, None))
        ]
        hidden_states_102 = None
        item_61 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_eps = (
            None
        )
        hidden_states_104 = torch.nn.functional.layer_norm(
            hidden_states_103,
            (768,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_bias_,
            item_61,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_bias_ = (item_61) = (
            None
        )
        hidden_states_105 = torch._C._nn.linear(
            hidden_states_104,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_104 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_46 = 1.702 * hidden_states_105
        sigmoid_11 = torch.sigmoid(mul_46)
        mul_46 = None
        hidden_states_106 = hidden_states_105 * sigmoid_11
        hidden_states_105 = sigmoid_11 = None
        hidden_states_107 = torch._C._nn.linear(
            hidden_states_106,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_106 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_108 = hidden_states_103 + hidden_states_107
        hidden_states_103 = hidden_states_107 = None
        pooled_output = hidden_states_108[
            (slice(None, None, None), 0, slice(None, None, None))
        ]
        item_62 = l_self_modules_vision_model_modules_post_layernorm_eps.item()
        l_self_modules_vision_model_modules_post_layernorm_eps = None
        pooled_output_1 = torch.nn.functional.layer_norm(
            pooled_output,
            (768,),
            l_self_modules_vision_model_modules_post_layernorm_parameters_weight_,
            l_self_modules_vision_model_modules_post_layernorm_parameters_bias_,
            item_62,
        )
        pooled_output = (
            l_self_modules_vision_model_modules_post_layernorm_parameters_weight_
        ) = (
            l_self_modules_vision_model_modules_post_layernorm_parameters_bias_
        ) = item_62 = None
        video_embeds = torch._C._nn.linear(
            pooled_output_1, l_self_modules_visual_projection_parameters_weight_, None
        )
        l_self_modules_visual_projection_parameters_weight_ = None
        cls_features = video_embeds.view(1, getitem_1, -1)
        video_embeds = None
        hidden_states_109 = (
            cls_features + l_self_modules_mit_parameters_position_embedding_
        )
        l_self_modules_mit_parameters_position_embedding_ = None
        item_63 = (
            l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_layer_norm1_eps.item()
        )
        l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_layer_norm1_eps = (
            None
        )
        hidden_states_110 = torch.nn.functional.layer_norm(
            hidden_states_109,
            (512,),
            l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_weight_,
            l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_bias_,
            item_63,
        )
        l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_weight_ = l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_bias_ = (item_63) = (
            None
        )
        queries_48 = torch._C._nn.linear(
            hidden_states_110,
            l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_48 = torch._C._nn.linear(
            hidden_states_110,
            l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_48 = torch._C._nn.linear(
            hidden_states_110,
            l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_110 = l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_97 = queries_48.view(1, 8, 8, 64)
        queries_48 = None
        queries_49 = view_97.transpose(1, 2)
        view_97 = None
        view_98 = keys_48.view(1, 8, 8, 64)
        keys_48 = None
        keys_49 = view_98.transpose(1, 2)
        view_98 = None
        view_99 = values_48.view(1, 8, 8, 64)
        values_48 = None
        values_49 = view_99.transpose(1, 2)
        view_99 = None
        transpose_124 = keys_49.transpose(-1, -2)
        keys_49 = None
        matmul_48 = torch.matmul(queries_49, transpose_124)
        queries_49 = transpose_124 = None
        item_64 = (
            l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_scale.item()
        )
        l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_scale = (
            None
        )
        attn_weights_72 = matmul_48 * item_64
        matmul_48 = item_64 = None
        softmax_24 = torch.nn.functional.softmax(
            attn_weights_72, dim=-1, dtype=torch.float32
        )
        attn_weights_72 = None
        attn_weights_73 = softmax_24.to(torch.float32)
        softmax_24 = None
        attn_weights_74 = torch.nn.functional.dropout(
            attn_weights_73, p=0.0, training=False
        )
        attn_weights_73 = None
        attn_output_96 = torch.matmul(attn_weights_74, values_49)
        attn_weights_74 = values_49 = None
        transpose_125 = attn_output_96.transpose(1, 2)
        attn_output_96 = None
        attn_output_97 = transpose_125.contiguous()
        transpose_125 = None
        reshape_25 = attn_output_97.reshape(1, 8, 512)
        attn_output_97 = None
        attn_output_98 = reshape_25.contiguous()
        reshape_25 = None
        attn_output_99 = torch._C._nn.linear(
            attn_output_98,
            l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_98 = l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_111 = hidden_states_109 + attn_output_99
        hidden_states_109 = attn_output_99 = None
        item_65 = (
            l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_layer_norm2_eps.item()
        )
        l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_layer_norm2_eps = (
            None
        )
        hidden_states_112 = torch.nn.functional.layer_norm(
            hidden_states_111,
            (512,),
            l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_weight_,
            l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_bias_,
            item_65,
        )
        l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_weight_ = l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_bias_ = (item_65) = (
            None
        )
        hidden_states_113 = torch._C._nn.linear(
            hidden_states_112,
            l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_112 = l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_49 = 1.702 * hidden_states_113
        sigmoid_12 = torch.sigmoid(mul_49)
        mul_49 = None
        hidden_states_114 = hidden_states_113 * sigmoid_12
        hidden_states_113 = sigmoid_12 = None
        hidden_states_115 = torch._C._nn.linear(
            hidden_states_114,
            l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_114 = l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_116 = hidden_states_111 + hidden_states_115
        hidden_states_111 = hidden_states_115 = None
        type_1 = hidden_states_116.type(torch.float32)
        hidden_states_116 = None
        last_hidden_state = type_1 + cls_features
        type_1 = cls_features = None
        pooled_output_2 = last_hidden_state.mean(dim=1, keepdim=False)
        img_features = hidden_states_108[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        item_66 = l_self_modules_prompts_visual_layernorm_eps.item()
        l_self_modules_prompts_visual_layernorm_eps = None
        img_features_1 = torch.nn.functional.layer_norm(
            img_features,
            (768,),
            l_self_modules_prompts_visual_layernorm_parameters_weight_,
            l_self_modules_prompts_visual_layernorm_parameters_bias_,
            item_66,
        )
        img_features = (
            l_self_modules_prompts_visual_layernorm_parameters_weight_
        ) = l_self_modules_prompts_visual_layernorm_parameters_bias_ = item_66 = None
        img_features_2 = img_features_1 @ l_self_parameters_prompts_visual_projection_
        img_features_1 = l_self_parameters_prompts_visual_projection_ = None
        img_features_3 = img_features_2.view(1, getitem_1, -1, 512)
        img_features_2 = getitem_1 = None
        img_features_4 = img_features_3.mean(dim=1, keepdim=False)
        img_features_3 = None
        input_ids = l_input_ids_.view(-1, 8)
        l_input_ids_ = None
        position_ids = (
            l_self_modules_text_model_modules_embeddings_buffers_position_ids_[
                (slice(None, None, None), slice(None, 8, None))
            ]
        )
        l_self_modules_text_model_modules_embeddings_buffers_position_ids_ = None
        item_67 = (
            l_self_modules_text_model_modules_embeddings_modules_token_embedding_norm_type.item()
        )
        l_self_modules_text_model_modules_embeddings_modules_token_embedding_norm_type = (
            None
        )
        inputs_embeds = torch.nn.functional.embedding(
            input_ids,
            l_self_modules_text_model_modules_embeddings_modules_token_embedding_parameters_weight_,
            None,
            None,
            item_67,
            False,
            False,
        )
        l_self_modules_text_model_modules_embeddings_modules_token_embedding_parameters_weight_ = (
            item_67
        ) = None
        item_68 = (
            l_self_modules_text_model_modules_embeddings_modules_position_embedding_norm_type.item()
        )
        l_self_modules_text_model_modules_embeddings_modules_position_embedding_norm_type = (
            None
        )
        position_embeddings = torch.nn.functional.embedding(
            position_ids,
            l_self_modules_text_model_modules_embeddings_modules_position_embedding_parameters_weight_,
            None,
            None,
            item_68,
            False,
            False,
        )
        position_ids = l_self_modules_text_model_modules_embeddings_modules_position_embedding_parameters_weight_ = (item_68) = (
            None
        )
        embeddings_2 = inputs_embeds + position_embeddings
        inputs_embeds = position_embeddings = None
        mask = torch.full(
            (8, 8), -3.4028234663852886e38, device=device(type="cuda", index=0)
        )
        mask_cond = torch.arange(8, device=device(type="cuda", index=0))
        add_42 = mask_cond + 1
        view_102 = add_42.view(8, 1)
        add_42 = None
        lt = mask_cond < view_102
        mask_cond = view_102 = None
        masked_fill_ = mask.masked_fill_(lt, 0)
        lt = masked_fill_ = None
        mask_1 = mask.to(torch.float32)
        mask = None
        getitem_144 = mask_1[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        mask_1 = None
        causal_4d_mask = getitem_144.expand(1, 1, 8, 8)
        getitem_144 = None
        getitem_145 = l_attention_mask_[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        l_attention_mask_ = None
        expand_2 = getitem_145.expand(1, 1, 8, 8)
        getitem_145 = None
        expanded_mask = expand_2.to(torch.float32)
        expand_2 = None
        tensor = torch.tensor(1.0, dtype=torch.float32)
        inverted_mask = tensor - expanded_mask
        tensor = expanded_mask = None
        to_28 = inverted_mask.to(torch.bool)
        attention_mask = inverted_mask.masked_fill(to_28, -3.4028234663852886e38)
        inverted_mask = to_28 = None
        item_69 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_eps.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_eps = (
            None
        )
        hidden_states_117 = torch.nn.functional.layer_norm(
            embeddings_2,
            (512,),
            l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_bias_,
            item_69,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_bias_ = (item_69) = (
            None
        )
        queries_50 = torch._C._nn.linear(
            hidden_states_117,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_50 = torch._C._nn.linear(
            hidden_states_117,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_50 = torch._C._nn.linear(
            hidden_states_117,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_117 = l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_103 = queries_50.view(1, 8, 8, 64)
        queries_50 = None
        queries_51 = view_103.transpose(1, 2)
        view_103 = None
        view_104 = keys_50.view(1, 8, 8, 64)
        keys_50 = None
        keys_51 = view_104.transpose(1, 2)
        view_104 = None
        view_105 = values_50.view(1, 8, 8, 64)
        values_50 = None
        values_51 = view_105.transpose(1, 2)
        view_105 = None
        attention_mask_1 = attention_mask + causal_4d_mask
        transpose_129 = keys_51.transpose(-1, -2)
        keys_51 = None
        matmul_51 = torch.matmul(queries_51, transpose_129)
        queries_51 = transpose_129 = None
        item_70 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_scale.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_scale = (
            None
        )
        attn_weights_75 = matmul_51 * item_70
        matmul_51 = item_70 = None
        attn_weights_76 = attn_weights_75 + attention_mask_1
        attn_weights_75 = attention_mask_1 = None
        softmax_25 = torch.nn.functional.softmax(
            attn_weights_76, dim=-1, dtype=torch.float32
        )
        attn_weights_76 = None
        attn_weights_77 = softmax_25.to(torch.float32)
        softmax_25 = None
        attn_weights_78 = torch.nn.functional.dropout(
            attn_weights_77, p=0.0, training=False
        )
        attn_weights_77 = None
        attn_output_100 = torch.matmul(attn_weights_78, values_51)
        attn_weights_78 = values_51 = None
        transpose_130 = attn_output_100.transpose(1, 2)
        attn_output_100 = None
        attn_output_101 = transpose_130.contiguous()
        transpose_130 = None
        reshape_26 = attn_output_101.reshape(1, 8, 512)
        attn_output_101 = None
        attn_output_102 = reshape_26.contiguous()
        reshape_26 = None
        attn_output_103 = torch._C._nn.linear(
            attn_output_102,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_102 = l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_118 = embeddings_2 + attn_output_103
        embeddings_2 = attn_output_103 = None
        item_71 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_eps.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_eps = (
            None
        )
        hidden_states_119 = torch.nn.functional.layer_norm(
            hidden_states_118,
            (512,),
            l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_bias_,
            item_71,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_bias_ = (item_71) = (
            None
        )
        hidden_states_120 = torch._C._nn.linear(
            hidden_states_119,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_119 = l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_52 = 1.702 * hidden_states_120
        sigmoid_13 = torch.sigmoid(mul_52)
        mul_52 = None
        hidden_states_121 = hidden_states_120 * sigmoid_13
        hidden_states_120 = sigmoid_13 = None
        hidden_states_122 = torch._C._nn.linear(
            hidden_states_121,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_121 = l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_123 = hidden_states_118 + hidden_states_122
        hidden_states_118 = hidden_states_122 = None
        item_72 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_eps.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_eps = (
            None
        )
        hidden_states_124 = torch.nn.functional.layer_norm(
            hidden_states_123,
            (512,),
            l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_bias_,
            item_72,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_bias_ = (item_72) = (
            None
        )
        queries_52 = torch._C._nn.linear(
            hidden_states_124,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_52 = torch._C._nn.linear(
            hidden_states_124,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_52 = torch._C._nn.linear(
            hidden_states_124,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_124 = l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_106 = queries_52.view(1, 8, 8, 64)
        queries_52 = None
        queries_53 = view_106.transpose(1, 2)
        view_106 = None
        view_107 = keys_52.view(1, 8, 8, 64)
        keys_52 = None
        keys_53 = view_107.transpose(1, 2)
        view_107 = None
        view_108 = values_52.view(1, 8, 8, 64)
        values_52 = None
        values_53 = view_108.transpose(1, 2)
        view_108 = None
        attention_mask_2 = attention_mask + causal_4d_mask
        transpose_134 = keys_53.transpose(-1, -2)
        keys_53 = None
        matmul_53 = torch.matmul(queries_53, transpose_134)
        queries_53 = transpose_134 = None
        item_73 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_scale.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_scale = (
            None
        )
        attn_weights_79 = matmul_53 * item_73
        matmul_53 = item_73 = None
        attn_weights_80 = attn_weights_79 + attention_mask_2
        attn_weights_79 = attention_mask_2 = None
        softmax_26 = torch.nn.functional.softmax(
            attn_weights_80, dim=-1, dtype=torch.float32
        )
        attn_weights_80 = None
        attn_weights_81 = softmax_26.to(torch.float32)
        softmax_26 = None
        attn_weights_82 = torch.nn.functional.dropout(
            attn_weights_81, p=0.0, training=False
        )
        attn_weights_81 = None
        attn_output_104 = torch.matmul(attn_weights_82, values_53)
        attn_weights_82 = values_53 = None
        transpose_135 = attn_output_104.transpose(1, 2)
        attn_output_104 = None
        attn_output_105 = transpose_135.contiguous()
        transpose_135 = None
        reshape_27 = attn_output_105.reshape(1, 8, 512)
        attn_output_105 = None
        attn_output_106 = reshape_27.contiguous()
        reshape_27 = None
        attn_output_107 = torch._C._nn.linear(
            attn_output_106,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_106 = l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_125 = hidden_states_123 + attn_output_107
        hidden_states_123 = attn_output_107 = None
        item_74 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_eps.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_eps = (
            None
        )
        hidden_states_126 = torch.nn.functional.layer_norm(
            hidden_states_125,
            (512,),
            l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_bias_,
            item_74,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_bias_ = (item_74) = (
            None
        )
        hidden_states_127 = torch._C._nn.linear(
            hidden_states_126,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_126 = l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_55 = 1.702 * hidden_states_127
        sigmoid_14 = torch.sigmoid(mul_55)
        mul_55 = None
        hidden_states_128 = hidden_states_127 * sigmoid_14
        hidden_states_127 = sigmoid_14 = None
        hidden_states_129 = torch._C._nn.linear(
            hidden_states_128,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_128 = l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_130 = hidden_states_125 + hidden_states_129
        hidden_states_125 = hidden_states_129 = None
        item_75 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_eps.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_eps = (
            None
        )
        hidden_states_131 = torch.nn.functional.layer_norm(
            hidden_states_130,
            (512,),
            l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_bias_,
            item_75,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_bias_ = (item_75) = (
            None
        )
        queries_54 = torch._C._nn.linear(
            hidden_states_131,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_54 = torch._C._nn.linear(
            hidden_states_131,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_54 = torch._C._nn.linear(
            hidden_states_131,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_131 = l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_109 = queries_54.view(1, 8, 8, 64)
        queries_54 = None
        queries_55 = view_109.transpose(1, 2)
        view_109 = None
        view_110 = keys_54.view(1, 8, 8, 64)
        keys_54 = None
        keys_55 = view_110.transpose(1, 2)
        view_110 = None
        view_111 = values_54.view(1, 8, 8, 64)
        values_54 = None
        values_55 = view_111.transpose(1, 2)
        view_111 = None
        attention_mask_3 = attention_mask + causal_4d_mask
        transpose_139 = keys_55.transpose(-1, -2)
        keys_55 = None
        matmul_55 = torch.matmul(queries_55, transpose_139)
        queries_55 = transpose_139 = None
        item_76 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_scale.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_scale = (
            None
        )
        attn_weights_83 = matmul_55 * item_76
        matmul_55 = item_76 = None
        attn_weights_84 = attn_weights_83 + attention_mask_3
        attn_weights_83 = attention_mask_3 = None
        softmax_27 = torch.nn.functional.softmax(
            attn_weights_84, dim=-1, dtype=torch.float32
        )
        attn_weights_84 = None
        attn_weights_85 = softmax_27.to(torch.float32)
        softmax_27 = None
        attn_weights_86 = torch.nn.functional.dropout(
            attn_weights_85, p=0.0, training=False
        )
        attn_weights_85 = None
        attn_output_108 = torch.matmul(attn_weights_86, values_55)
        attn_weights_86 = values_55 = None
        transpose_140 = attn_output_108.transpose(1, 2)
        attn_output_108 = None
        attn_output_109 = transpose_140.contiguous()
        transpose_140 = None
        reshape_28 = attn_output_109.reshape(1, 8, 512)
        attn_output_109 = None
        attn_output_110 = reshape_28.contiguous()
        reshape_28 = None
        attn_output_111 = torch._C._nn.linear(
            attn_output_110,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_110 = l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_132 = hidden_states_130 + attn_output_111
        hidden_states_130 = attn_output_111 = None
        item_77 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_eps.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_eps = (
            None
        )
        hidden_states_133 = torch.nn.functional.layer_norm(
            hidden_states_132,
            (512,),
            l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_bias_,
            item_77,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_bias_ = (item_77) = (
            None
        )
        hidden_states_134 = torch._C._nn.linear(
            hidden_states_133,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_133 = l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_58 = 1.702 * hidden_states_134
        sigmoid_15 = torch.sigmoid(mul_58)
        mul_58 = None
        hidden_states_135 = hidden_states_134 * sigmoid_15
        hidden_states_134 = sigmoid_15 = None
        hidden_states_136 = torch._C._nn.linear(
            hidden_states_135,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_135 = l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_137 = hidden_states_132 + hidden_states_136
        hidden_states_132 = hidden_states_136 = None
        item_78 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_eps.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_eps = (
            None
        )
        hidden_states_138 = torch.nn.functional.layer_norm(
            hidden_states_137,
            (512,),
            l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_bias_,
            item_78,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_bias_ = (item_78) = (
            None
        )
        queries_56 = torch._C._nn.linear(
            hidden_states_138,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_56 = torch._C._nn.linear(
            hidden_states_138,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_56 = torch._C._nn.linear(
            hidden_states_138,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_138 = l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_112 = queries_56.view(1, 8, 8, 64)
        queries_56 = None
        queries_57 = view_112.transpose(1, 2)
        view_112 = None
        view_113 = keys_56.view(1, 8, 8, 64)
        keys_56 = None
        keys_57 = view_113.transpose(1, 2)
        view_113 = None
        view_114 = values_56.view(1, 8, 8, 64)
        values_56 = None
        values_57 = view_114.transpose(1, 2)
        view_114 = None
        attention_mask_4 = attention_mask + causal_4d_mask
        transpose_144 = keys_57.transpose(-1, -2)
        keys_57 = None
        matmul_57 = torch.matmul(queries_57, transpose_144)
        queries_57 = transpose_144 = None
        item_79 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_scale.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_scale = (
            None
        )
        attn_weights_87 = matmul_57 * item_79
        matmul_57 = item_79 = None
        attn_weights_88 = attn_weights_87 + attention_mask_4
        attn_weights_87 = attention_mask_4 = None
        softmax_28 = torch.nn.functional.softmax(
            attn_weights_88, dim=-1, dtype=torch.float32
        )
        attn_weights_88 = None
        attn_weights_89 = softmax_28.to(torch.float32)
        softmax_28 = None
        attn_weights_90 = torch.nn.functional.dropout(
            attn_weights_89, p=0.0, training=False
        )
        attn_weights_89 = None
        attn_output_112 = torch.matmul(attn_weights_90, values_57)
        attn_weights_90 = values_57 = None
        transpose_145 = attn_output_112.transpose(1, 2)
        attn_output_112 = None
        attn_output_113 = transpose_145.contiguous()
        transpose_145 = None
        reshape_29 = attn_output_113.reshape(1, 8, 512)
        attn_output_113 = None
        attn_output_114 = reshape_29.contiguous()
        reshape_29 = None
        attn_output_115 = torch._C._nn.linear(
            attn_output_114,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_114 = l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_139 = hidden_states_137 + attn_output_115
        hidden_states_137 = attn_output_115 = None
        item_80 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_eps.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_eps = (
            None
        )
        hidden_states_140 = torch.nn.functional.layer_norm(
            hidden_states_139,
            (512,),
            l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_bias_,
            item_80,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_bias_ = (item_80) = (
            None
        )
        hidden_states_141 = torch._C._nn.linear(
            hidden_states_140,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_140 = l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_61 = 1.702 * hidden_states_141
        sigmoid_16 = torch.sigmoid(mul_61)
        mul_61 = None
        hidden_states_142 = hidden_states_141 * sigmoid_16
        hidden_states_141 = sigmoid_16 = None
        hidden_states_143 = torch._C._nn.linear(
            hidden_states_142,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_142 = l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_144 = hidden_states_139 + hidden_states_143
        hidden_states_139 = hidden_states_143 = None
        item_81 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_eps.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_eps = (
            None
        )
        hidden_states_145 = torch.nn.functional.layer_norm(
            hidden_states_144,
            (512,),
            l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_bias_,
            item_81,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_bias_ = (item_81) = (
            None
        )
        queries_58 = torch._C._nn.linear(
            hidden_states_145,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_58 = torch._C._nn.linear(
            hidden_states_145,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_58 = torch._C._nn.linear(
            hidden_states_145,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_145 = l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_115 = queries_58.view(1, 8, 8, 64)
        queries_58 = None
        queries_59 = view_115.transpose(1, 2)
        view_115 = None
        view_116 = keys_58.view(1, 8, 8, 64)
        keys_58 = None
        keys_59 = view_116.transpose(1, 2)
        view_116 = None
        view_117 = values_58.view(1, 8, 8, 64)
        values_58 = None
        values_59 = view_117.transpose(1, 2)
        view_117 = None
        attention_mask_5 = attention_mask + causal_4d_mask
        transpose_149 = keys_59.transpose(-1, -2)
        keys_59 = None
        matmul_59 = torch.matmul(queries_59, transpose_149)
        queries_59 = transpose_149 = None
        item_82 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_scale.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_scale = (
            None
        )
        attn_weights_91 = matmul_59 * item_82
        matmul_59 = item_82 = None
        attn_weights_92 = attn_weights_91 + attention_mask_5
        attn_weights_91 = attention_mask_5 = None
        softmax_29 = torch.nn.functional.softmax(
            attn_weights_92, dim=-1, dtype=torch.float32
        )
        attn_weights_92 = None
        attn_weights_93 = softmax_29.to(torch.float32)
        softmax_29 = None
        attn_weights_94 = torch.nn.functional.dropout(
            attn_weights_93, p=0.0, training=False
        )
        attn_weights_93 = None
        attn_output_116 = torch.matmul(attn_weights_94, values_59)
        attn_weights_94 = values_59 = None
        transpose_150 = attn_output_116.transpose(1, 2)
        attn_output_116 = None
        attn_output_117 = transpose_150.contiguous()
        transpose_150 = None
        reshape_30 = attn_output_117.reshape(1, 8, 512)
        attn_output_117 = None
        attn_output_118 = reshape_30.contiguous()
        reshape_30 = None
        attn_output_119 = torch._C._nn.linear(
            attn_output_118,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_118 = l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_146 = hidden_states_144 + attn_output_119
        hidden_states_144 = attn_output_119 = None
        item_83 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_eps.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_eps = (
            None
        )
        hidden_states_147 = torch.nn.functional.layer_norm(
            hidden_states_146,
            (512,),
            l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_bias_,
            item_83,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_bias_ = (item_83) = (
            None
        )
        hidden_states_148 = torch._C._nn.linear(
            hidden_states_147,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_147 = l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_64 = 1.702 * hidden_states_148
        sigmoid_17 = torch.sigmoid(mul_64)
        mul_64 = None
        hidden_states_149 = hidden_states_148 * sigmoid_17
        hidden_states_148 = sigmoid_17 = None
        hidden_states_150 = torch._C._nn.linear(
            hidden_states_149,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_149 = l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_151 = hidden_states_146 + hidden_states_150
        hidden_states_146 = hidden_states_150 = None
        item_84 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_eps.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_eps = (
            None
        )
        hidden_states_152 = torch.nn.functional.layer_norm(
            hidden_states_151,
            (512,),
            l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_bias_,
            item_84,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_bias_ = (item_84) = (
            None
        )
        queries_60 = torch._C._nn.linear(
            hidden_states_152,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_60 = torch._C._nn.linear(
            hidden_states_152,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_60 = torch._C._nn.linear(
            hidden_states_152,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_152 = l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_118 = queries_60.view(1, 8, 8, 64)
        queries_60 = None
        queries_61 = view_118.transpose(1, 2)
        view_118 = None
        view_119 = keys_60.view(1, 8, 8, 64)
        keys_60 = None
        keys_61 = view_119.transpose(1, 2)
        view_119 = None
        view_120 = values_60.view(1, 8, 8, 64)
        values_60 = None
        values_61 = view_120.transpose(1, 2)
        view_120 = None
        attention_mask_6 = attention_mask + causal_4d_mask
        transpose_154 = keys_61.transpose(-1, -2)
        keys_61 = None
        matmul_61 = torch.matmul(queries_61, transpose_154)
        queries_61 = transpose_154 = None
        item_85 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_scale.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_scale = (
            None
        )
        attn_weights_95 = matmul_61 * item_85
        matmul_61 = item_85 = None
        attn_weights_96 = attn_weights_95 + attention_mask_6
        attn_weights_95 = attention_mask_6 = None
        softmax_30 = torch.nn.functional.softmax(
            attn_weights_96, dim=-1, dtype=torch.float32
        )
        attn_weights_96 = None
        attn_weights_97 = softmax_30.to(torch.float32)
        softmax_30 = None
        attn_weights_98 = torch.nn.functional.dropout(
            attn_weights_97, p=0.0, training=False
        )
        attn_weights_97 = None
        attn_output_120 = torch.matmul(attn_weights_98, values_61)
        attn_weights_98 = values_61 = None
        transpose_155 = attn_output_120.transpose(1, 2)
        attn_output_120 = None
        attn_output_121 = transpose_155.contiguous()
        transpose_155 = None
        reshape_31 = attn_output_121.reshape(1, 8, 512)
        attn_output_121 = None
        attn_output_122 = reshape_31.contiguous()
        reshape_31 = None
        attn_output_123 = torch._C._nn.linear(
            attn_output_122,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_122 = l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_153 = hidden_states_151 + attn_output_123
        hidden_states_151 = attn_output_123 = None
        item_86 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_eps.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_eps = (
            None
        )
        hidden_states_154 = torch.nn.functional.layer_norm(
            hidden_states_153,
            (512,),
            l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_bias_,
            item_86,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_bias_ = (item_86) = (
            None
        )
        hidden_states_155 = torch._C._nn.linear(
            hidden_states_154,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_154 = l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_67 = 1.702 * hidden_states_155
        sigmoid_18 = torch.sigmoid(mul_67)
        mul_67 = None
        hidden_states_156 = hidden_states_155 * sigmoid_18
        hidden_states_155 = sigmoid_18 = None
        hidden_states_157 = torch._C._nn.linear(
            hidden_states_156,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_156 = l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_158 = hidden_states_153 + hidden_states_157
        hidden_states_153 = hidden_states_157 = None
        item_87 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_eps.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_eps = (
            None
        )
        hidden_states_159 = torch.nn.functional.layer_norm(
            hidden_states_158,
            (512,),
            l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_bias_,
            item_87,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_bias_ = (item_87) = (
            None
        )
        queries_62 = torch._C._nn.linear(
            hidden_states_159,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_62 = torch._C._nn.linear(
            hidden_states_159,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_62 = torch._C._nn.linear(
            hidden_states_159,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_159 = l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_121 = queries_62.view(1, 8, 8, 64)
        queries_62 = None
        queries_63 = view_121.transpose(1, 2)
        view_121 = None
        view_122 = keys_62.view(1, 8, 8, 64)
        keys_62 = None
        keys_63 = view_122.transpose(1, 2)
        view_122 = None
        view_123 = values_62.view(1, 8, 8, 64)
        values_62 = None
        values_63 = view_123.transpose(1, 2)
        view_123 = None
        attention_mask_7 = attention_mask + causal_4d_mask
        transpose_159 = keys_63.transpose(-1, -2)
        keys_63 = None
        matmul_63 = torch.matmul(queries_63, transpose_159)
        queries_63 = transpose_159 = None
        item_88 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_scale.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_scale = (
            None
        )
        attn_weights_99 = matmul_63 * item_88
        matmul_63 = item_88 = None
        attn_weights_100 = attn_weights_99 + attention_mask_7
        attn_weights_99 = attention_mask_7 = None
        softmax_31 = torch.nn.functional.softmax(
            attn_weights_100, dim=-1, dtype=torch.float32
        )
        attn_weights_100 = None
        attn_weights_101 = softmax_31.to(torch.float32)
        softmax_31 = None
        attn_weights_102 = torch.nn.functional.dropout(
            attn_weights_101, p=0.0, training=False
        )
        attn_weights_101 = None
        attn_output_124 = torch.matmul(attn_weights_102, values_63)
        attn_weights_102 = values_63 = None
        transpose_160 = attn_output_124.transpose(1, 2)
        attn_output_124 = None
        attn_output_125 = transpose_160.contiguous()
        transpose_160 = None
        reshape_32 = attn_output_125.reshape(1, 8, 512)
        attn_output_125 = None
        attn_output_126 = reshape_32.contiguous()
        reshape_32 = None
        attn_output_127 = torch._C._nn.linear(
            attn_output_126,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_126 = l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_160 = hidden_states_158 + attn_output_127
        hidden_states_158 = attn_output_127 = None
        item_89 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_eps.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_eps = (
            None
        )
        hidden_states_161 = torch.nn.functional.layer_norm(
            hidden_states_160,
            (512,),
            l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_bias_,
            item_89,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_bias_ = (item_89) = (
            None
        )
        hidden_states_162 = torch._C._nn.linear(
            hidden_states_161,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_161 = l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_70 = 1.702 * hidden_states_162
        sigmoid_19 = torch.sigmoid(mul_70)
        mul_70 = None
        hidden_states_163 = hidden_states_162 * sigmoid_19
        hidden_states_162 = sigmoid_19 = None
        hidden_states_164 = torch._C._nn.linear(
            hidden_states_163,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_163 = l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_165 = hidden_states_160 + hidden_states_164
        hidden_states_160 = hidden_states_164 = None
        item_90 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_eps.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_eps = (
            None
        )
        hidden_states_166 = torch.nn.functional.layer_norm(
            hidden_states_165,
            (512,),
            l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_bias_,
            item_90,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_bias_ = (item_90) = (
            None
        )
        queries_64 = torch._C._nn.linear(
            hidden_states_166,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_64 = torch._C._nn.linear(
            hidden_states_166,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_64 = torch._C._nn.linear(
            hidden_states_166,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_166 = l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_124 = queries_64.view(1, 8, 8, 64)
        queries_64 = None
        queries_65 = view_124.transpose(1, 2)
        view_124 = None
        view_125 = keys_64.view(1, 8, 8, 64)
        keys_64 = None
        keys_65 = view_125.transpose(1, 2)
        view_125 = None
        view_126 = values_64.view(1, 8, 8, 64)
        values_64 = None
        values_65 = view_126.transpose(1, 2)
        view_126 = None
        attention_mask_8 = attention_mask + causal_4d_mask
        transpose_164 = keys_65.transpose(-1, -2)
        keys_65 = None
        matmul_65 = torch.matmul(queries_65, transpose_164)
        queries_65 = transpose_164 = None
        item_91 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_scale.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_scale = (
            None
        )
        attn_weights_103 = matmul_65 * item_91
        matmul_65 = item_91 = None
        attn_weights_104 = attn_weights_103 + attention_mask_8
        attn_weights_103 = attention_mask_8 = None
        softmax_32 = torch.nn.functional.softmax(
            attn_weights_104, dim=-1, dtype=torch.float32
        )
        attn_weights_104 = None
        attn_weights_105 = softmax_32.to(torch.float32)
        softmax_32 = None
        attn_weights_106 = torch.nn.functional.dropout(
            attn_weights_105, p=0.0, training=False
        )
        attn_weights_105 = None
        attn_output_128 = torch.matmul(attn_weights_106, values_65)
        attn_weights_106 = values_65 = None
        transpose_165 = attn_output_128.transpose(1, 2)
        attn_output_128 = None
        attn_output_129 = transpose_165.contiguous()
        transpose_165 = None
        reshape_33 = attn_output_129.reshape(1, 8, 512)
        attn_output_129 = None
        attn_output_130 = reshape_33.contiguous()
        reshape_33 = None
        attn_output_131 = torch._C._nn.linear(
            attn_output_130,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_130 = l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_167 = hidden_states_165 + attn_output_131
        hidden_states_165 = attn_output_131 = None
        item_92 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_eps.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_eps = (
            None
        )
        hidden_states_168 = torch.nn.functional.layer_norm(
            hidden_states_167,
            (512,),
            l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_bias_,
            item_92,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_bias_ = (item_92) = (
            None
        )
        hidden_states_169 = torch._C._nn.linear(
            hidden_states_168,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_168 = l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_73 = 1.702 * hidden_states_169
        sigmoid_20 = torch.sigmoid(mul_73)
        mul_73 = None
        hidden_states_170 = hidden_states_169 * sigmoid_20
        hidden_states_169 = sigmoid_20 = None
        hidden_states_171 = torch._C._nn.linear(
            hidden_states_170,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_170 = l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_172 = hidden_states_167 + hidden_states_171
        hidden_states_167 = hidden_states_171 = None
        item_93 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_eps.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_eps = (
            None
        )
        hidden_states_173 = torch.nn.functional.layer_norm(
            hidden_states_172,
            (512,),
            l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_bias_,
            item_93,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_bias_ = (item_93) = (
            None
        )
        queries_66 = torch._C._nn.linear(
            hidden_states_173,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_66 = torch._C._nn.linear(
            hidden_states_173,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_66 = torch._C._nn.linear(
            hidden_states_173,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_173 = l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_127 = queries_66.view(1, 8, 8, 64)
        queries_66 = None
        queries_67 = view_127.transpose(1, 2)
        view_127 = None
        view_128 = keys_66.view(1, 8, 8, 64)
        keys_66 = None
        keys_67 = view_128.transpose(1, 2)
        view_128 = None
        view_129 = values_66.view(1, 8, 8, 64)
        values_66 = None
        values_67 = view_129.transpose(1, 2)
        view_129 = None
        attention_mask_9 = attention_mask + causal_4d_mask
        transpose_169 = keys_67.transpose(-1, -2)
        keys_67 = None
        matmul_67 = torch.matmul(queries_67, transpose_169)
        queries_67 = transpose_169 = None
        item_94 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_scale.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_scale = (
            None
        )
        attn_weights_107 = matmul_67 * item_94
        matmul_67 = item_94 = None
        attn_weights_108 = attn_weights_107 + attention_mask_9
        attn_weights_107 = attention_mask_9 = None
        softmax_33 = torch.nn.functional.softmax(
            attn_weights_108, dim=-1, dtype=torch.float32
        )
        attn_weights_108 = None
        attn_weights_109 = softmax_33.to(torch.float32)
        softmax_33 = None
        attn_weights_110 = torch.nn.functional.dropout(
            attn_weights_109, p=0.0, training=False
        )
        attn_weights_109 = None
        attn_output_132 = torch.matmul(attn_weights_110, values_67)
        attn_weights_110 = values_67 = None
        transpose_170 = attn_output_132.transpose(1, 2)
        attn_output_132 = None
        attn_output_133 = transpose_170.contiguous()
        transpose_170 = None
        reshape_34 = attn_output_133.reshape(1, 8, 512)
        attn_output_133 = None
        attn_output_134 = reshape_34.contiguous()
        reshape_34 = None
        attn_output_135 = torch._C._nn.linear(
            attn_output_134,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_134 = l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_174 = hidden_states_172 + attn_output_135
        hidden_states_172 = attn_output_135 = None
        item_95 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_eps.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_eps = (
            None
        )
        hidden_states_175 = torch.nn.functional.layer_norm(
            hidden_states_174,
            (512,),
            l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_bias_,
            item_95,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_bias_ = (item_95) = (
            None
        )
        hidden_states_176 = torch._C._nn.linear(
            hidden_states_175,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_175 = l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_76 = 1.702 * hidden_states_176
        sigmoid_21 = torch.sigmoid(mul_76)
        mul_76 = None
        hidden_states_177 = hidden_states_176 * sigmoid_21
        hidden_states_176 = sigmoid_21 = None
        hidden_states_178 = torch._C._nn.linear(
            hidden_states_177,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_177 = l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_179 = hidden_states_174 + hidden_states_178
        hidden_states_174 = hidden_states_178 = None
        item_96 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_eps.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_eps = (
            None
        )
        hidden_states_180 = torch.nn.functional.layer_norm(
            hidden_states_179,
            (512,),
            l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_bias_,
            item_96,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_bias_ = (item_96) = (
            None
        )
        queries_68 = torch._C._nn.linear(
            hidden_states_180,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_68 = torch._C._nn.linear(
            hidden_states_180,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_68 = torch._C._nn.linear(
            hidden_states_180,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_180 = l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_130 = queries_68.view(1, 8, 8, 64)
        queries_68 = None
        queries_69 = view_130.transpose(1, 2)
        view_130 = None
        view_131 = keys_68.view(1, 8, 8, 64)
        keys_68 = None
        keys_69 = view_131.transpose(1, 2)
        view_131 = None
        view_132 = values_68.view(1, 8, 8, 64)
        values_68 = None
        values_69 = view_132.transpose(1, 2)
        view_132 = None
        attention_mask_10 = attention_mask + causal_4d_mask
        transpose_174 = keys_69.transpose(-1, -2)
        keys_69 = None
        matmul_69 = torch.matmul(queries_69, transpose_174)
        queries_69 = transpose_174 = None
        item_97 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_scale.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_scale = (
            None
        )
        attn_weights_111 = matmul_69 * item_97
        matmul_69 = item_97 = None
        attn_weights_112 = attn_weights_111 + attention_mask_10
        attn_weights_111 = attention_mask_10 = None
        softmax_34 = torch.nn.functional.softmax(
            attn_weights_112, dim=-1, dtype=torch.float32
        )
        attn_weights_112 = None
        attn_weights_113 = softmax_34.to(torch.float32)
        softmax_34 = None
        attn_weights_114 = torch.nn.functional.dropout(
            attn_weights_113, p=0.0, training=False
        )
        attn_weights_113 = None
        attn_output_136 = torch.matmul(attn_weights_114, values_69)
        attn_weights_114 = values_69 = None
        transpose_175 = attn_output_136.transpose(1, 2)
        attn_output_136 = None
        attn_output_137 = transpose_175.contiguous()
        transpose_175 = None
        reshape_35 = attn_output_137.reshape(1, 8, 512)
        attn_output_137 = None
        attn_output_138 = reshape_35.contiguous()
        reshape_35 = None
        attn_output_139 = torch._C._nn.linear(
            attn_output_138,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_138 = l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_181 = hidden_states_179 + attn_output_139
        hidden_states_179 = attn_output_139 = None
        item_98 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_eps.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_eps = (
            None
        )
        hidden_states_182 = torch.nn.functional.layer_norm(
            hidden_states_181,
            (512,),
            l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_bias_,
            item_98,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_bias_ = (item_98) = (
            None
        )
        hidden_states_183 = torch._C._nn.linear(
            hidden_states_182,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_182 = l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_79 = 1.702 * hidden_states_183
        sigmoid_22 = torch.sigmoid(mul_79)
        mul_79 = None
        hidden_states_184 = hidden_states_183 * sigmoid_22
        hidden_states_183 = sigmoid_22 = None
        hidden_states_185 = torch._C._nn.linear(
            hidden_states_184,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_184 = l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_186 = hidden_states_181 + hidden_states_185
        hidden_states_181 = hidden_states_185 = None
        item_99 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_eps.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_eps = (
            None
        )
        hidden_states_187 = torch.nn.functional.layer_norm(
            hidden_states_186,
            (512,),
            l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_bias_,
            item_99,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_bias_ = (item_99) = (
            None
        )
        queries_70 = torch._C._nn.linear(
            hidden_states_187,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_70 = torch._C._nn.linear(
            hidden_states_187,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_70 = torch._C._nn.linear(
            hidden_states_187,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_187 = l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_133 = queries_70.view(1, 8, 8, 64)
        queries_70 = None
        queries_71 = view_133.transpose(1, 2)
        view_133 = None
        view_134 = keys_70.view(1, 8, 8, 64)
        keys_70 = None
        keys_71 = view_134.transpose(1, 2)
        view_134 = None
        view_135 = values_70.view(1, 8, 8, 64)
        values_70 = None
        values_71 = view_135.transpose(1, 2)
        view_135 = None
        attention_mask_11 = attention_mask + causal_4d_mask
        transpose_179 = keys_71.transpose(-1, -2)
        keys_71 = None
        matmul_71 = torch.matmul(queries_71, transpose_179)
        queries_71 = transpose_179 = None
        item_100 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_scale.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_scale = (
            None
        )
        attn_weights_115 = matmul_71 * item_100
        matmul_71 = item_100 = None
        attn_weights_116 = attn_weights_115 + attention_mask_11
        attn_weights_115 = attention_mask_11 = None
        softmax_35 = torch.nn.functional.softmax(
            attn_weights_116, dim=-1, dtype=torch.float32
        )
        attn_weights_116 = None
        attn_weights_117 = softmax_35.to(torch.float32)
        softmax_35 = None
        attn_weights_118 = torch.nn.functional.dropout(
            attn_weights_117, p=0.0, training=False
        )
        attn_weights_117 = None
        attn_output_140 = torch.matmul(attn_weights_118, values_71)
        attn_weights_118 = values_71 = None
        transpose_180 = attn_output_140.transpose(1, 2)
        attn_output_140 = None
        attn_output_141 = transpose_180.contiguous()
        transpose_180 = None
        reshape_36 = attn_output_141.reshape(1, 8, 512)
        attn_output_141 = None
        attn_output_142 = reshape_36.contiguous()
        reshape_36 = None
        attn_output_143 = torch._C._nn.linear(
            attn_output_142,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_142 = l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_188 = hidden_states_186 + attn_output_143
        hidden_states_186 = attn_output_143 = None
        item_101 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_eps.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_eps = (
            None
        )
        hidden_states_189 = torch.nn.functional.layer_norm(
            hidden_states_188,
            (512,),
            l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_bias_,
            item_101,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_bias_ = (item_101) = (
            None
        )
        hidden_states_190 = torch._C._nn.linear(
            hidden_states_189,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_189 = l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_82 = 1.702 * hidden_states_190
        sigmoid_23 = torch.sigmoid(mul_82)
        mul_82 = None
        hidden_states_191 = hidden_states_190 * sigmoid_23
        hidden_states_190 = sigmoid_23 = None
        hidden_states_192 = torch._C._nn.linear(
            hidden_states_191,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_191 = l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_193 = hidden_states_188 + hidden_states_192
        hidden_states_188 = hidden_states_192 = None
        item_102 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_eps.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_eps = (
            None
        )
        hidden_states_194 = torch.nn.functional.layer_norm(
            hidden_states_193,
            (512,),
            l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_bias_,
            item_102,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_bias_ = (item_102) = (
            None
        )
        queries_72 = torch._C._nn.linear(
            hidden_states_194,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_72 = torch._C._nn.linear(
            hidden_states_194,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_72 = torch._C._nn.linear(
            hidden_states_194,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_194 = l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_136 = queries_72.view(1, 8, 8, 64)
        queries_72 = None
        queries_73 = view_136.transpose(1, 2)
        view_136 = None
        view_137 = keys_72.view(1, 8, 8, 64)
        keys_72 = None
        keys_73 = view_137.transpose(1, 2)
        view_137 = None
        view_138 = values_72.view(1, 8, 8, 64)
        values_72 = None
        values_73 = view_138.transpose(1, 2)
        view_138 = None
        attention_mask_12 = attention_mask + causal_4d_mask
        attention_mask = causal_4d_mask = None
        transpose_184 = keys_73.transpose(-1, -2)
        keys_73 = None
        matmul_73 = torch.matmul(queries_73, transpose_184)
        queries_73 = transpose_184 = None
        item_103 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_scale.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_scale = (
            None
        )
        attn_weights_119 = matmul_73 * item_103
        matmul_73 = item_103 = None
        attn_weights_120 = attn_weights_119 + attention_mask_12
        attn_weights_119 = attention_mask_12 = None
        softmax_36 = torch.nn.functional.softmax(
            attn_weights_120, dim=-1, dtype=torch.float32
        )
        attn_weights_120 = None
        attn_weights_121 = softmax_36.to(torch.float32)
        softmax_36 = None
        attn_weights_122 = torch.nn.functional.dropout(
            attn_weights_121, p=0.0, training=False
        )
        attn_weights_121 = None
        attn_output_144 = torch.matmul(attn_weights_122, values_73)
        attn_weights_122 = values_73 = None
        transpose_185 = attn_output_144.transpose(1, 2)
        attn_output_144 = None
        attn_output_145 = transpose_185.contiguous()
        transpose_185 = None
        reshape_37 = attn_output_145.reshape(1, 8, 512)
        attn_output_145 = None
        attn_output_146 = reshape_37.contiguous()
        reshape_37 = None
        attn_output_147 = torch._C._nn.linear(
            attn_output_146,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_146 = l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_195 = hidden_states_193 + attn_output_147
        hidden_states_193 = attn_output_147 = None
        item_104 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_eps.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_eps = (
            None
        )
        hidden_states_196 = torch.nn.functional.layer_norm(
            hidden_states_195,
            (512,),
            l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_bias_,
            item_104,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_bias_ = (item_104) = (
            None
        )
        hidden_states_197 = torch._C._nn.linear(
            hidden_states_196,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_196 = l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_85 = 1.702 * hidden_states_197
        sigmoid_24 = torch.sigmoid(mul_85)
        mul_85 = None
        hidden_states_198 = hidden_states_197 * sigmoid_24
        hidden_states_197 = sigmoid_24 = None
        hidden_states_199 = torch._C._nn.linear(
            hidden_states_198,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_198 = l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_200 = hidden_states_195 + hidden_states_199
        hidden_states_195 = hidden_states_199 = None
        item_105 = l_self_modules_text_model_modules_final_layer_norm_eps.item()
        l_self_modules_text_model_modules_final_layer_norm_eps = None
        last_hidden_state_1 = torch.nn.functional.layer_norm(
            hidden_states_200,
            (512,),
            l_self_modules_text_model_modules_final_layer_norm_parameters_weight_,
            l_self_modules_text_model_modules_final_layer_norm_parameters_bias_,
            item_105,
        )
        hidden_states_200 = (
            l_self_modules_text_model_modules_final_layer_norm_parameters_weight_
        ) = (
            l_self_modules_text_model_modules_final_layer_norm_parameters_bias_
        ) = item_105 = None
        arange_1 = torch.arange(1)
        argmax = input_ids.argmax(dim=-1)
        input_ids = None
        pooled_output_3 = last_hidden_state_1[(arange_1, argmax)]
        arange_1 = argmax = None
        text_embeds = torch._C._nn.linear(
            pooled_output_3, l_self_modules_text_projection_parameters_weight_, None
        )
        l_self_modules_text_projection_parameters_weight_ = None
        unsqueeze = text_embeds.unsqueeze(0)
        text_embeds = None
        text_embeds_1 = unsqueeze.expand(1, -1, -1)
        unsqueeze = None
        item_106 = l_self_modules_prompts_generator_modules_layernorm_eps.item()
        l_self_modules_prompts_generator_modules_layernorm_eps = None
        visual = torch.nn.functional.layer_norm(
            img_features_4,
            (512,),
            l_self_modules_prompts_generator_modules_layernorm_parameters_weight_,
            l_self_modules_prompts_generator_modules_layernorm_parameters_bias_,
            item_106,
        )
        img_features_4 = (
            l_self_modules_prompts_generator_modules_layernorm_parameters_weight_
        ) = (
            l_self_modules_prompts_generator_modules_layernorm_parameters_bias_
        ) = item_106 = None
        item_107 = (
            l_self_modules_prompts_generator_modules_decoder_modules_0_modules_norm1_eps.item()
        )
        l_self_modules_prompts_generator_modules_decoder_modules_0_modules_norm1_eps = (
            None
        )
        layer_norm_67 = torch.nn.functional.layer_norm(
            text_embeds_1,
            (512,),
            l_self_modules_prompts_generator_modules_decoder_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_prompts_generator_modules_decoder_modules_0_modules_norm1_parameters_bias_,
            item_107,
        )
        l_self_modules_prompts_generator_modules_decoder_modules_0_modules_norm1_parameters_weight_ = l_self_modules_prompts_generator_modules_decoder_modules_0_modules_norm1_parameters_bias_ = (item_107) = (
            None
        )
        linear_212 = torch._C._nn.linear(
            layer_norm_67,
            l_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_q_proj_parameters_weight_,
            None,
        )
        layer_norm_67 = l_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_q_proj_parameters_weight_ = (None)
        reshape_38 = linear_212.reshape(1, 1, 8, 64)
        linear_212 = None
        queries_74 = reshape_38.permute(0, 2, 1, 3)
        reshape_38 = None
        linear_213 = torch._C._nn.linear(
            visual,
            l_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        reshape_39 = linear_213.reshape(1, 196, 8, 64)
        linear_213 = None
        keys_74 = reshape_39.permute(0, 2, 1, 3)
        reshape_39 = None
        linear_214 = torch._C._nn.linear(
            visual,
            l_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_v_proj_parameters_weight_ = (
            None
        )
        reshape_40 = linear_214.reshape(1, 196, 8, 64)
        linear_214 = None
        values_74 = reshape_40.permute(0, 2, 1, 3)
        reshape_40 = None
        transpose_186 = keys_74.transpose(-2, -1)
        keys_74 = None
        matmul_75 = queries_74 @ transpose_186
        queries_74 = transpose_186 = None
        item_108 = (
            l_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_scale.item()
        )
        l_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_scale = (
            None
        )
        attn = matmul_75 * item_108
        matmul_75 = item_108 = None
        attn_1 = attn.softmax(dim=-1)
        attn = None
        item_109 = (
            l_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_attn_drop_p.item()
        )
        l_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_attn_drop_p = (
            None
        )
        attn_2 = torch.nn.functional.dropout(attn_1, item_109, False, False)
        attn_1 = item_109 = None
        matmul_76 = attn_2 @ values_74
        attn_2 = values_74 = None
        transpose_187 = matmul_76.transpose(1, 2)
        matmul_76 = None
        x = transpose_187.reshape(1, 1, 512)
        transpose_187 = None
        x_1 = torch._C._nn.linear(
            x,
            l_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_proj_parameters_weight_,
            l_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_proj_parameters_bias_,
        )
        x = l_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_proj_parameters_weight_ = l_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_proj_parameters_bias_ = (None)
        item_110 = (
            l_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_proj_drop_p.item()
        )
        l_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_proj_drop_p = (
            None
        )
        x_2 = torch.nn.functional.dropout(x_1, item_110, False, False)
        x_1 = item_110 = None
        x_3 = text_embeds_1 + x_2
        x_2 = None
        item_111 = (
            l_self_modules_prompts_generator_modules_decoder_modules_0_modules_norm3_eps.item()
        )
        l_self_modules_prompts_generator_modules_decoder_modules_0_modules_norm3_eps = (
            None
        )
        layer_norm_68 = torch.nn.functional.layer_norm(
            x_3,
            (512,),
            l_self_modules_prompts_generator_modules_decoder_modules_0_modules_norm3_parameters_weight_,
            l_self_modules_prompts_generator_modules_decoder_modules_0_modules_norm3_parameters_bias_,
            item_111,
        )
        l_self_modules_prompts_generator_modules_decoder_modules_0_modules_norm3_parameters_weight_ = l_self_modules_prompts_generator_modules_decoder_modules_0_modules_norm3_parameters_bias_ = (item_111) = (
            None
        )
        input_1 = torch._C._nn.linear(
            layer_norm_68,
            l_self_modules_prompts_generator_modules_decoder_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_prompts_generator_modules_decoder_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        layer_norm_68 = l_self_modules_prompts_generator_modules_decoder_modules_0_modules_mlp_modules_0_parameters_weight_ = l_self_modules_prompts_generator_modules_decoder_modules_0_modules_mlp_modules_0_parameters_bias_ = (None)
        mul_88 = 1.702 * input_1
        sigmoid_25 = torch.sigmoid(mul_88)
        mul_88 = None
        input_2 = input_1 * sigmoid_25
        input_1 = sigmoid_25 = None
        item_112 = (
            l_self_modules_prompts_generator_modules_decoder_modules_0_modules_mlp_modules_2_p.item()
        )
        l_self_modules_prompts_generator_modules_decoder_modules_0_modules_mlp_modules_2_p = (
            None
        )
        input_3 = torch.nn.functional.dropout(input_2, item_112, False, False)
        input_2 = item_112 = None
        input_4 = torch._C._nn.linear(
            input_3,
            l_self_modules_prompts_generator_modules_decoder_modules_0_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_prompts_generator_modules_decoder_modules_0_modules_mlp_modules_3_parameters_bias_,
        )
        input_3 = l_self_modules_prompts_generator_modules_decoder_modules_0_modules_mlp_modules_3_parameters_weight_ = l_self_modules_prompts_generator_modules_decoder_modules_0_modules_mlp_modules_3_parameters_bias_ = (None)
        x_4 = x_3 + input_4
        x_3 = input_4 = None
        item_113 = (
            l_self_modules_prompts_generator_modules_decoder_modules_1_modules_norm1_eps.item()
        )
        l_self_modules_prompts_generator_modules_decoder_modules_1_modules_norm1_eps = (
            None
        )
        layer_norm_69 = torch.nn.functional.layer_norm(
            x_4,
            (512,),
            l_self_modules_prompts_generator_modules_decoder_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_prompts_generator_modules_decoder_modules_1_modules_norm1_parameters_bias_,
            item_113,
        )
        l_self_modules_prompts_generator_modules_decoder_modules_1_modules_norm1_parameters_weight_ = l_self_modules_prompts_generator_modules_decoder_modules_1_modules_norm1_parameters_bias_ = (item_113) = (
            None
        )
        linear_218 = torch._C._nn.linear(
            layer_norm_69,
            l_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_q_proj_parameters_weight_,
            None,
        )
        layer_norm_69 = l_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_q_proj_parameters_weight_ = (None)
        reshape_42 = linear_218.reshape(1, 1, 8, 64)
        linear_218 = None
        queries_75 = reshape_42.permute(0, 2, 1, 3)
        reshape_42 = None
        linear_219 = torch._C._nn.linear(
            visual,
            l_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        reshape_43 = linear_219.reshape(1, 196, 8, 64)
        linear_219 = None
        keys_75 = reshape_43.permute(0, 2, 1, 3)
        reshape_43 = None
        linear_220 = torch._C._nn.linear(
            visual,
            l_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_v_proj_parameters_weight_,
            None,
        )
        visual = l_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_v_proj_parameters_weight_ = (None)
        reshape_44 = linear_220.reshape(1, 196, 8, 64)
        linear_220 = None
        values_75 = reshape_44.permute(0, 2, 1, 3)
        reshape_44 = None
        transpose_188 = keys_75.transpose(-2, -1)
        keys_75 = None
        matmul_77 = queries_75 @ transpose_188
        queries_75 = transpose_188 = None
        item_114 = (
            l_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_scale.item()
        )
        l_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_scale = (
            None
        )
        attn_3 = matmul_77 * item_114
        matmul_77 = item_114 = None
        attn_4 = attn_3.softmax(dim=-1)
        attn_3 = None
        item_115 = (
            l_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_attn_drop_p.item()
        )
        l_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_attn_drop_p = (
            None
        )
        attn_5 = torch.nn.functional.dropout(attn_4, item_115, False, False)
        attn_4 = item_115 = None
        matmul_78 = attn_5 @ values_75
        attn_5 = values_75 = None
        transpose_189 = matmul_78.transpose(1, 2)
        matmul_78 = None
        x_5 = transpose_189.reshape(1, 1, 512)
        transpose_189 = None
        x_6 = torch._C._nn.linear(
            x_5,
            l_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_proj_parameters_weight_,
            l_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_proj_parameters_bias_,
        )
        x_5 = l_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_proj_parameters_weight_ = l_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_proj_parameters_bias_ = (None)
        item_116 = (
            l_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_proj_drop_p.item()
        )
        l_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_proj_drop_p = (
            None
        )
        x_7 = torch.nn.functional.dropout(x_6, item_116, False, False)
        x_6 = item_116 = None
        x_8 = x_4 + x_7
        x_4 = x_7 = None
        item_117 = (
            l_self_modules_prompts_generator_modules_decoder_modules_1_modules_norm3_eps.item()
        )
        l_self_modules_prompts_generator_modules_decoder_modules_1_modules_norm3_eps = (
            None
        )
        layer_norm_70 = torch.nn.functional.layer_norm(
            x_8,
            (512,),
            l_self_modules_prompts_generator_modules_decoder_modules_1_modules_norm3_parameters_weight_,
            l_self_modules_prompts_generator_modules_decoder_modules_1_modules_norm3_parameters_bias_,
            item_117,
        )
        l_self_modules_prompts_generator_modules_decoder_modules_1_modules_norm3_parameters_weight_ = l_self_modules_prompts_generator_modules_decoder_modules_1_modules_norm3_parameters_bias_ = (item_117) = (
            None
        )
        input_5 = torch._C._nn.linear(
            layer_norm_70,
            l_self_modules_prompts_generator_modules_decoder_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_prompts_generator_modules_decoder_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        layer_norm_70 = l_self_modules_prompts_generator_modules_decoder_modules_1_modules_mlp_modules_0_parameters_weight_ = l_self_modules_prompts_generator_modules_decoder_modules_1_modules_mlp_modules_0_parameters_bias_ = (None)
        mul_91 = 1.702 * input_5
        sigmoid_26 = torch.sigmoid(mul_91)
        mul_91 = None
        input_6 = input_5 * sigmoid_26
        input_5 = sigmoid_26 = None
        item_118 = (
            l_self_modules_prompts_generator_modules_decoder_modules_1_modules_mlp_modules_2_p.item()
        )
        l_self_modules_prompts_generator_modules_decoder_modules_1_modules_mlp_modules_2_p = (
            None
        )
        input_7 = torch.nn.functional.dropout(input_6, item_118, False, False)
        input_6 = item_118 = None
        input_8 = torch._C._nn.linear(
            input_7,
            l_self_modules_prompts_generator_modules_decoder_modules_1_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_prompts_generator_modules_decoder_modules_1_modules_mlp_modules_3_parameters_bias_,
        )
        input_7 = l_self_modules_prompts_generator_modules_decoder_modules_1_modules_mlp_modules_3_parameters_weight_ = l_self_modules_prompts_generator_modules_decoder_modules_1_modules_mlp_modules_3_parameters_bias_ = (None)
        x_9 = x_8 + input_8
        x_8 = input_8 = None
        mul_93 = l_self_modules_prompts_generator_parameters_alpha_ * x_9
        l_self_modules_prompts_generator_parameters_alpha_ = x_9 = None
        text_embeds_2 = text_embeds_1 + mul_93
        text_embeds_1 = mul_93 = None
        norm = pooled_output_2.norm(p=2, dim=-1, keepdim=True)
        video_embeds_1 = pooled_output_2 / norm
        norm = None
        norm_1 = text_embeds_2.norm(p=2, dim=-1, keepdim=True)
        text_embeds_3 = text_embeds_2 / norm_1
        text_embeds_2 = norm_1 = None
        logit_scale = l_self_parameters_logit_scale_.exp()
        l_self_parameters_logit_scale_ = None
        mul_94 = logit_scale * text_embeds_3
        logit_scale = None
        logits_per_video = torch.functional.einsum("bd,bkd->bk", video_embeds_1, mul_94)
        mul_94 = None
        logits_per_text = logits_per_video.T
        return (
            hidden_states_108,
            pooled_output_1,
            last_hidden_state,
            pooled_output_2,
            last_hidden_state_1,
            pooled_output_3,
            logits_per_video,
            logits_per_text,
            text_embeds_3,
            video_embeds_1,
        )
