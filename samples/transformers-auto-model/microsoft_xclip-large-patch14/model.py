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
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_ln_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_ln_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_ln_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_ln_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_ln_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_ln_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_ln_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_ln_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_ln_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_ln_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_ln_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_ln_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_ln_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_ln_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_ln_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_ln_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_ln_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_ln_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_ln_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_ln_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_ln_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_ln_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_ln_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_ln_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_ln_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_ln_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_ln_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_ln_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_ln_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_ln_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_ln_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_ln_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_ln_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_ln_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_ln_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_ln_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm1_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_scale: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm2_eps: torch.Tensor,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_fc_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_fc_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_fc_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_fc_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_ln_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_ln_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_ln_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_ln_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_ln_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_ln_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm1_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm1_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm2_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm2_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_fc_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_fc_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_fc_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_fc_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_ln_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_ln_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_ln_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_ln_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_ln_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_ln_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm1_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm1_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm2_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm2_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_fc_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_fc_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_fc_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_fc_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_ln_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_ln_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_ln_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_ln_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_ln_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_ln_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm1_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm1_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm2_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm2_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_fc_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_fc_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_fc_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_fc_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_ln_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_ln_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_ln_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_ln_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_ln_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_ln_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm1_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm1_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm2_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm2_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_fc_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_fc_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_fc_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_fc_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_ln_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_ln_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_ln_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_ln_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_ln_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_ln_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm1_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm1_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm2_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm2_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_fc_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_fc_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_fc_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_fc_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_ln_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_ln_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_ln_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_ln_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_ln_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_ln_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm1_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm1_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm2_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm2_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_fc_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_fc_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_fc_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_fc_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_ln_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_ln_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_ln_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_ln_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_ln_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_ln_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm1_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm1_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm2_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm2_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_fc_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_fc_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_fc_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_fc_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_ln_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_ln_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_ln_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_ln_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_ln_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_ln_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm1_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm1_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm2_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm2_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_fc_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_fc_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_fc_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_fc_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_ln_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_ln_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_ln_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_ln_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_ln_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_ln_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm1_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm1_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm2_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm2_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_fc_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_fc_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_fc_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_fc_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_ln_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_ln_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_ln_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_ln_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_ln_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_ln_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm1_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm1_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm2_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm2_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_fc_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_fc_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_fc_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_fc_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_ln_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_ln_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_ln_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_ln_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_ln_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_ln_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm1_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm1_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm2_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm2_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_fc_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_fc_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_fc_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_fc_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_ln_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_ln_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_ln_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_ln_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_ln_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_ln_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm1_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm1_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_scale = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_scale
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm2_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm2_eps = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm2_eps
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc2_parameters_bias_
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
            (14, 14),
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
            (1024,),
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
        msg_token_1 = msg_token.view(floordiv, 8, 1024)
        msg_token = floordiv = None
        item_2 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_ln_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_message_ln_eps = (
            None
        )
        layer_norm_1 = torch.nn.functional.layer_norm(
            msg_token_1,
            (1024,),
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
        view_1 = queries.view(1, getitem_14, 16, 64)
        queries = None
        queries_1 = view_1.transpose(1, 2)
        view_1 = None
        view_2 = keys.view(1, getitem_14, 16, 64)
        keys = None
        keys_1 = view_2.transpose(1, 2)
        view_2 = None
        view_3 = values.view(1, getitem_14, 16, 64)
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
        reshape_1 = attn_output_1.reshape(1, getitem_14, 1024)
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
        msg_token_3 = msg_token_2.view(-1, 1, 1024)
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
            (1024,),
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
        view_5 = queries_2.view(getitem_16, 258, 16, 64)
        queries_2 = None
        queries_3 = view_5.transpose(1, 2)
        view_5 = None
        view_6 = keys_2.view(getitem_16, 258, 16, 64)
        keys_2 = None
        keys_3 = view_6.transpose(1, 2)
        view_6 = None
        view_7 = values_2.view(getitem_16, 258, 16, 64)
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
        reshape_2 = attn_output_5.reshape(getitem_16, 258, 1024)
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
            (slice(None, None, None), slice(None, 257, None), slice(None, None, None))
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
            (1024,),
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
        msg_token_5 = msg_token_4.view(floordiv_1, 8, 1024)
        msg_token_4 = floordiv_1 = None
        item_7 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_ln_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_message_ln_eps = (
            None
        )
        layer_norm_4 = torch.nn.functional.layer_norm(
            msg_token_5,
            (1024,),
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
        view_9 = queries_4.view(1, getitem_25, 16, 64)
        queries_4 = None
        queries_5 = view_9.transpose(1, 2)
        view_9 = None
        view_10 = keys_4.view(1, getitem_25, 16, 64)
        keys_4 = None
        keys_5 = view_10.transpose(1, 2)
        view_10 = None
        view_11 = values_4.view(1, getitem_25, 16, 64)
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
        reshape_3 = attn_output_9.reshape(1, getitem_25, 1024)
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
        msg_token_7 = msg_token_6.view(-1, 1, 1024)
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
            (1024,),
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
        view_13 = queries_6.view(getitem_27, 258, 16, 64)
        queries_6 = None
        queries_7 = view_13.transpose(1, 2)
        view_13 = None
        view_14 = keys_6.view(getitem_27, 258, 16, 64)
        keys_6 = None
        keys_7 = view_14.transpose(1, 2)
        view_14 = None
        view_15 = values_6.view(getitem_27, 258, 16, 64)
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
        reshape_4 = attn_output_13.reshape(getitem_27, 258, 1024)
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
            (slice(None, None, None), slice(None, 257, None), slice(None, None, None))
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
            (1024,),
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
        msg_token_9 = msg_token_8.view(floordiv_2, 8, 1024)
        msg_token_8 = floordiv_2 = None
        item_12 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_ln_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_message_ln_eps = (
            None
        )
        layer_norm_7 = torch.nn.functional.layer_norm(
            msg_token_9,
            (1024,),
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
        view_17 = queries_8.view(1, getitem_36, 16, 64)
        queries_8 = None
        queries_9 = view_17.transpose(1, 2)
        view_17 = None
        view_18 = keys_8.view(1, getitem_36, 16, 64)
        keys_8 = None
        keys_9 = view_18.transpose(1, 2)
        view_18 = None
        view_19 = values_8.view(1, getitem_36, 16, 64)
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
        reshape_5 = attn_output_17.reshape(1, getitem_36, 1024)
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
        msg_token_11 = msg_token_10.view(-1, 1, 1024)
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
            (1024,),
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
        view_21 = queries_10.view(getitem_38, 258, 16, 64)
        queries_10 = None
        queries_11 = view_21.transpose(1, 2)
        view_21 = None
        view_22 = keys_10.view(getitem_38, 258, 16, 64)
        keys_10 = None
        keys_11 = view_22.transpose(1, 2)
        view_22 = None
        view_23 = values_10.view(getitem_38, 258, 16, 64)
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
        reshape_6 = attn_output_21.reshape(getitem_38, 258, 1024)
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
            (slice(None, None, None), slice(None, 257, None), slice(None, None, None))
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
            (1024,),
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
        msg_token_13 = msg_token_12.view(floordiv_3, 8, 1024)
        msg_token_12 = floordiv_3 = None
        item_17 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_ln_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_message_ln_eps = (
            None
        )
        layer_norm_10 = torch.nn.functional.layer_norm(
            msg_token_13,
            (1024,),
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
        view_25 = queries_12.view(1, getitem_47, 16, 64)
        queries_12 = None
        queries_13 = view_25.transpose(1, 2)
        view_25 = None
        view_26 = keys_12.view(1, getitem_47, 16, 64)
        keys_12 = None
        keys_13 = view_26.transpose(1, 2)
        view_26 = None
        view_27 = values_12.view(1, getitem_47, 16, 64)
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
        reshape_7 = attn_output_25.reshape(1, getitem_47, 1024)
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
        msg_token_15 = msg_token_14.view(-1, 1, 1024)
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
            (1024,),
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
        view_29 = queries_14.view(getitem_49, 258, 16, 64)
        queries_14 = None
        queries_15 = view_29.transpose(1, 2)
        view_29 = None
        view_30 = keys_14.view(getitem_49, 258, 16, 64)
        keys_14 = None
        keys_15 = view_30.transpose(1, 2)
        view_30 = None
        view_31 = values_14.view(getitem_49, 258, 16, 64)
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
        reshape_8 = attn_output_29.reshape(getitem_49, 258, 1024)
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
            (slice(None, None, None), slice(None, 257, None), slice(None, None, None))
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
            (1024,),
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
        msg_token_17 = msg_token_16.view(floordiv_4, 8, 1024)
        msg_token_16 = floordiv_4 = None
        item_22 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_ln_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_message_ln_eps = (
            None
        )
        layer_norm_13 = torch.nn.functional.layer_norm(
            msg_token_17,
            (1024,),
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
        view_33 = queries_16.view(1, getitem_58, 16, 64)
        queries_16 = None
        queries_17 = view_33.transpose(1, 2)
        view_33 = None
        view_34 = keys_16.view(1, getitem_58, 16, 64)
        keys_16 = None
        keys_17 = view_34.transpose(1, 2)
        view_34 = None
        view_35 = values_16.view(1, getitem_58, 16, 64)
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
        reshape_9 = attn_output_33.reshape(1, getitem_58, 1024)
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
        msg_token_19 = msg_token_18.view(-1, 1, 1024)
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
            (1024,),
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
        view_37 = queries_18.view(getitem_60, 258, 16, 64)
        queries_18 = None
        queries_19 = view_37.transpose(1, 2)
        view_37 = None
        view_38 = keys_18.view(getitem_60, 258, 16, 64)
        keys_18 = None
        keys_19 = view_38.transpose(1, 2)
        view_38 = None
        view_39 = values_18.view(getitem_60, 258, 16, 64)
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
        reshape_10 = attn_output_37.reshape(getitem_60, 258, 1024)
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
            (slice(None, None, None), slice(None, 257, None), slice(None, None, None))
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
            (1024,),
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
        msg_token_21 = msg_token_20.view(floordiv_5, 8, 1024)
        msg_token_20 = floordiv_5 = None
        item_27 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_ln_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_message_ln_eps = (
            None
        )
        layer_norm_16 = torch.nn.functional.layer_norm(
            msg_token_21,
            (1024,),
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
        view_41 = queries_20.view(1, getitem_69, 16, 64)
        queries_20 = None
        queries_21 = view_41.transpose(1, 2)
        view_41 = None
        view_42 = keys_20.view(1, getitem_69, 16, 64)
        keys_20 = None
        keys_21 = view_42.transpose(1, 2)
        view_42 = None
        view_43 = values_20.view(1, getitem_69, 16, 64)
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
        reshape_11 = attn_output_41.reshape(1, getitem_69, 1024)
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
        msg_token_23 = msg_token_22.view(-1, 1, 1024)
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
            (1024,),
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
        view_45 = queries_22.view(getitem_71, 258, 16, 64)
        queries_22 = None
        queries_23 = view_45.transpose(1, 2)
        view_45 = None
        view_46 = keys_22.view(getitem_71, 258, 16, 64)
        keys_22 = None
        keys_23 = view_46.transpose(1, 2)
        view_46 = None
        view_47 = values_22.view(getitem_71, 258, 16, 64)
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
        reshape_12 = attn_output_45.reshape(getitem_71, 258, 1024)
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
            (slice(None, None, None), slice(None, 257, None), slice(None, None, None))
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
            (1024,),
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
        msg_token_25 = msg_token_24.view(floordiv_6, 8, 1024)
        msg_token_24 = floordiv_6 = None
        item_32 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_ln_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_message_ln_eps = (
            None
        )
        layer_norm_19 = torch.nn.functional.layer_norm(
            msg_token_25,
            (1024,),
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
        view_49 = queries_24.view(1, getitem_80, 16, 64)
        queries_24 = None
        queries_25 = view_49.transpose(1, 2)
        view_49 = None
        view_50 = keys_24.view(1, getitem_80, 16, 64)
        keys_24 = None
        keys_25 = view_50.transpose(1, 2)
        view_50 = None
        view_51 = values_24.view(1, getitem_80, 16, 64)
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
        reshape_13 = attn_output_49.reshape(1, getitem_80, 1024)
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
        msg_token_27 = msg_token_26.view(-1, 1, 1024)
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
            (1024,),
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
        view_53 = queries_26.view(getitem_82, 258, 16, 64)
        queries_26 = None
        queries_27 = view_53.transpose(1, 2)
        view_53 = None
        view_54 = keys_26.view(getitem_82, 258, 16, 64)
        keys_26 = None
        keys_27 = view_54.transpose(1, 2)
        view_54 = None
        view_55 = values_26.view(getitem_82, 258, 16, 64)
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
        reshape_14 = attn_output_53.reshape(getitem_82, 258, 1024)
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
            (slice(None, None, None), slice(None, 257, None), slice(None, None, None))
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
            (1024,),
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
        msg_token_29 = msg_token_28.view(floordiv_7, 8, 1024)
        msg_token_28 = floordiv_7 = None
        item_37 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_ln_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_message_ln_eps = (
            None
        )
        layer_norm_22 = torch.nn.functional.layer_norm(
            msg_token_29,
            (1024,),
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
        view_57 = queries_28.view(1, getitem_91, 16, 64)
        queries_28 = None
        queries_29 = view_57.transpose(1, 2)
        view_57 = None
        view_58 = keys_28.view(1, getitem_91, 16, 64)
        keys_28 = None
        keys_29 = view_58.transpose(1, 2)
        view_58 = None
        view_59 = values_28.view(1, getitem_91, 16, 64)
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
        reshape_15 = attn_output_57.reshape(1, getitem_91, 1024)
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
        msg_token_31 = msg_token_30.view(-1, 1, 1024)
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
            (1024,),
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
        view_61 = queries_30.view(getitem_93, 258, 16, 64)
        queries_30 = None
        queries_31 = view_61.transpose(1, 2)
        view_61 = None
        view_62 = keys_30.view(getitem_93, 258, 16, 64)
        keys_30 = None
        keys_31 = view_62.transpose(1, 2)
        view_62 = None
        view_63 = values_30.view(getitem_93, 258, 16, 64)
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
        reshape_16 = attn_output_61.reshape(getitem_93, 258, 1024)
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
            (slice(None, None, None), slice(None, 257, None), slice(None, None, None))
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
            (1024,),
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
        msg_token_33 = msg_token_32.view(floordiv_8, 8, 1024)
        msg_token_32 = floordiv_8 = None
        item_42 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_ln_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_message_ln_eps = (
            None
        )
        layer_norm_25 = torch.nn.functional.layer_norm(
            msg_token_33,
            (1024,),
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
        view_65 = queries_32.view(1, getitem_102, 16, 64)
        queries_32 = None
        queries_33 = view_65.transpose(1, 2)
        view_65 = None
        view_66 = keys_32.view(1, getitem_102, 16, 64)
        keys_32 = None
        keys_33 = view_66.transpose(1, 2)
        view_66 = None
        view_67 = values_32.view(1, getitem_102, 16, 64)
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
        reshape_17 = attn_output_65.reshape(1, getitem_102, 1024)
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
        msg_token_35 = msg_token_34.view(-1, 1, 1024)
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
            (1024,),
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
        view_69 = queries_34.view(getitem_104, 258, 16, 64)
        queries_34 = None
        queries_35 = view_69.transpose(1, 2)
        view_69 = None
        view_70 = keys_34.view(getitem_104, 258, 16, 64)
        keys_34 = None
        keys_35 = view_70.transpose(1, 2)
        view_70 = None
        view_71 = values_34.view(getitem_104, 258, 16, 64)
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
        reshape_18 = attn_output_69.reshape(getitem_104, 258, 1024)
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
            (slice(None, None, None), slice(None, 257, None), slice(None, None, None))
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
            (1024,),
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
        msg_token_37 = msg_token_36.view(floordiv_9, 8, 1024)
        msg_token_36 = floordiv_9 = None
        item_47 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_ln_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_message_ln_eps = (
            None
        )
        layer_norm_28 = torch.nn.functional.layer_norm(
            msg_token_37,
            (1024,),
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
        view_73 = queries_36.view(1, getitem_113, 16, 64)
        queries_36 = None
        queries_37 = view_73.transpose(1, 2)
        view_73 = None
        view_74 = keys_36.view(1, getitem_113, 16, 64)
        keys_36 = None
        keys_37 = view_74.transpose(1, 2)
        view_74 = None
        view_75 = values_36.view(1, getitem_113, 16, 64)
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
        reshape_19 = attn_output_73.reshape(1, getitem_113, 1024)
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
        msg_token_39 = msg_token_38.view(-1, 1, 1024)
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
            (1024,),
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
        view_77 = queries_38.view(getitem_115, 258, 16, 64)
        queries_38 = None
        queries_39 = view_77.transpose(1, 2)
        view_77 = None
        view_78 = keys_38.view(getitem_115, 258, 16, 64)
        keys_38 = None
        keys_39 = view_78.transpose(1, 2)
        view_78 = None
        view_79 = values_38.view(getitem_115, 258, 16, 64)
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
        reshape_20 = attn_output_77.reshape(getitem_115, 258, 1024)
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
            (slice(None, None, None), slice(None, 257, None), slice(None, None, None))
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
            (1024,),
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
        msg_token_41 = msg_token_40.view(floordiv_10, 8, 1024)
        msg_token_40 = floordiv_10 = None
        item_52 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_ln_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_message_ln_eps = (
            None
        )
        layer_norm_31 = torch.nn.functional.layer_norm(
            msg_token_41,
            (1024,),
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
        view_81 = queries_40.view(1, getitem_124, 16, 64)
        queries_40 = None
        queries_41 = view_81.transpose(1, 2)
        view_81 = None
        view_82 = keys_40.view(1, getitem_124, 16, 64)
        keys_40 = None
        keys_41 = view_82.transpose(1, 2)
        view_82 = None
        view_83 = values_40.view(1, getitem_124, 16, 64)
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
        reshape_21 = attn_output_81.reshape(1, getitem_124, 1024)
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
        msg_token_43 = msg_token_42.view(-1, 1, 1024)
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
            (1024,),
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
        view_85 = queries_42.view(getitem_126, 258, 16, 64)
        queries_42 = None
        queries_43 = view_85.transpose(1, 2)
        view_85 = None
        view_86 = keys_42.view(getitem_126, 258, 16, 64)
        keys_42 = None
        keys_43 = view_86.transpose(1, 2)
        view_86 = None
        view_87 = values_42.view(getitem_126, 258, 16, 64)
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
        reshape_22 = attn_output_85.reshape(getitem_126, 258, 1024)
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
            (slice(None, None, None), slice(None, 257, None), slice(None, None, None))
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
            (1024,),
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
        msg_token_45 = msg_token_44.view(floordiv_11, 8, 1024)
        msg_token_44 = floordiv_11 = None
        item_57 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_ln_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_message_ln_eps = (
            None
        )
        layer_norm_34 = torch.nn.functional.layer_norm(
            msg_token_45,
            (1024,),
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
        view_89 = queries_44.view(1, getitem_135, 16, 64)
        queries_44 = None
        queries_45 = view_89.transpose(1, 2)
        view_89 = None
        view_90 = keys_44.view(1, getitem_135, 16, 64)
        keys_44 = None
        keys_45 = view_90.transpose(1, 2)
        view_90 = None
        view_91 = values_44.view(1, getitem_135, 16, 64)
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
        reshape_23 = attn_output_89.reshape(1, getitem_135, 1024)
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
        msg_token_47 = msg_token_46.view(-1, 1, 1024)
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
            (1024,),
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
        view_93 = queries_46.view(getitem_137, 258, 16, 64)
        queries_46 = None
        queries_47 = view_93.transpose(1, 2)
        view_93 = None
        view_94 = keys_46.view(getitem_137, 258, 16, 64)
        keys_46 = None
        keys_47 = view_94.transpose(1, 2)
        view_94 = None
        view_95 = values_46.view(getitem_137, 258, 16, 64)
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
        reshape_24 = attn_output_93.reshape(getitem_137, 258, 1024)
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
            (slice(None, None, None), slice(None, 257, None), slice(None, None, None))
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
            (1024,),
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
        size_38 = hidden_states_108.size()
        getitem_141 = size_38[0]
        size_38 = None
        floordiv_12 = getitem_141 // 8
        getitem_141 = None
        getitem_144 = hidden_states_108[
            (slice(None, None, None), 0, slice(None, None, None))
        ]
        msg_token_48 = torch._C._nn.linear(
            getitem_144,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_fc_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_fc_parameters_bias_,
        )
        getitem_144 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_fc_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_fc_parameters_bias_ = (None)
        msg_token_49 = msg_token_48.view(floordiv_12, 8, 1024)
        msg_token_48 = floordiv_12 = None
        item_62 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_ln_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_ln_eps = (
            None
        )
        layer_norm_37 = torch.nn.functional.layer_norm(
            msg_token_49,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_ln_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_ln_parameters_bias_,
            item_62,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_ln_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_ln_parameters_bias_ = (item_62) = (
            None
        )
        size_39 = layer_norm_37.size()
        getitem_146 = size_39[1]
        size_39 = None
        queries_48 = torch._C._nn.linear(
            layer_norm_37,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_attn_modules_q_proj_parameters_bias_ = (None)
        keys_48 = torch._C._nn.linear(
            layer_norm_37,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_attn_modules_k_proj_parameters_bias_ = (None)
        values_48 = torch._C._nn.linear(
            layer_norm_37,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_attn_modules_v_proj_parameters_bias_,
        )
        layer_norm_37 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_attn_modules_v_proj_parameters_bias_ = (None)
        view_97 = queries_48.view(1, getitem_146, 16, 64)
        queries_48 = None
        queries_49 = view_97.transpose(1, 2)
        view_97 = None
        view_98 = keys_48.view(1, getitem_146, 16, 64)
        keys_48 = None
        keys_49 = view_98.transpose(1, 2)
        view_98 = None
        view_99 = values_48.view(1, getitem_146, 16, 64)
        values_48 = None
        values_49 = view_99.transpose(1, 2)
        view_99 = None
        transpose_124 = keys_49.transpose(-1, -2)
        keys_49 = None
        matmul_48 = torch.matmul(queries_49, transpose_124)
        queries_49 = transpose_124 = None
        item_63 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_attn_scale = (
            None
        )
        attn_weights_72 = matmul_48 * item_63
        matmul_48 = item_63 = None
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
        reshape_25 = attn_output_97.reshape(1, getitem_146, 1024)
        attn_output_97 = getitem_146 = None
        attn_output_98 = reshape_25.contiguous()
        reshape_25 = None
        attn_output_99 = torch._C._nn.linear(
            attn_output_98,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_98 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_message_attn_modules_out_proj_parameters_bias_ = (None)
        msg_token_50 = msg_token_49 + attn_output_99
        msg_token_49 = attn_output_99 = None
        msg_token_51 = msg_token_50.view(-1, 1, 1024)
        msg_token_50 = None
        hidden_states_109 = torch.cat([hidden_states_108, msg_token_51], dim=1)
        hidden_states_108 = msg_token_51 = None
        item_64 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm1_eps = (
            None
        )
        hidden_states_110 = torch.nn.functional.layer_norm(
            hidden_states_109,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm1_parameters_bias_,
            item_64,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm1_parameters_bias_ = (item_64) = (
            None
        )
        size_40 = hidden_states_110.size()
        getitem_148 = size_40[0]
        size_40 = None
        queries_50 = torch._C._nn.linear(
            hidden_states_110,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_50 = torch._C._nn.linear(
            hidden_states_110,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_50 = torch._C._nn.linear(
            hidden_states_110,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_110 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_101 = queries_50.view(getitem_148, 258, 16, 64)
        queries_50 = None
        queries_51 = view_101.transpose(1, 2)
        view_101 = None
        view_102 = keys_50.view(getitem_148, 258, 16, 64)
        keys_50 = None
        keys_51 = view_102.transpose(1, 2)
        view_102 = None
        view_103 = values_50.view(getitem_148, 258, 16, 64)
        values_50 = None
        values_51 = view_103.transpose(1, 2)
        view_103 = None
        transpose_129 = keys_51.transpose(-1, -2)
        keys_51 = None
        matmul_50 = torch.matmul(queries_51, transpose_129)
        queries_51 = transpose_129 = None
        item_65 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_scale = (
            None
        )
        attn_weights_75 = matmul_50 * item_65
        matmul_50 = item_65 = None
        softmax_25 = torch.nn.functional.softmax(
            attn_weights_75, dim=-1, dtype=torch.float32
        )
        attn_weights_75 = None
        attn_weights_76 = softmax_25.to(torch.float32)
        softmax_25 = None
        attn_weights_77 = torch.nn.functional.dropout(
            attn_weights_76, p=0.0, training=False
        )
        attn_weights_76 = None
        attn_output_100 = torch.matmul(attn_weights_77, values_51)
        attn_weights_77 = values_51 = None
        transpose_130 = attn_output_100.transpose(1, 2)
        attn_output_100 = None
        attn_output_101 = transpose_130.contiguous()
        transpose_130 = None
        reshape_26 = attn_output_101.reshape(getitem_148, 258, 1024)
        attn_output_101 = getitem_148 = None
        attn_output_102 = reshape_26.contiguous()
        reshape_26 = None
        attn_output_103 = torch._C._nn.linear(
            attn_output_102,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_102 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_111 = hidden_states_109 + attn_output_103
        hidden_states_109 = attn_output_103 = None
        hidden_states_112 = hidden_states_111[
            (slice(None, None, None), slice(None, 257, None), slice(None, None, None))
        ]
        hidden_states_111 = None
        item_66 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm2_eps = (
            None
        )
        hidden_states_113 = torch.nn.functional.layer_norm(
            hidden_states_112,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm2_parameters_bias_,
            item_66,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm2_parameters_bias_ = (item_66) = (
            None
        )
        hidden_states_114 = torch._C._nn.linear(
            hidden_states_113,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_113 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_50 = 1.702 * hidden_states_114
        sigmoid_12 = torch.sigmoid(mul_50)
        mul_50 = None
        hidden_states_115 = hidden_states_114 * sigmoid_12
        hidden_states_114 = sigmoid_12 = None
        hidden_states_116 = torch._C._nn.linear(
            hidden_states_115,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_115 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_117 = hidden_states_112 + hidden_states_116
        hidden_states_112 = hidden_states_116 = None
        size_41 = hidden_states_117.size()
        getitem_152 = size_41[0]
        size_41 = None
        floordiv_13 = getitem_152 // 8
        getitem_152 = None
        getitem_155 = hidden_states_117[
            (slice(None, None, None), 0, slice(None, None, None))
        ]
        msg_token_52 = torch._C._nn.linear(
            getitem_155,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_fc_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_fc_parameters_bias_,
        )
        getitem_155 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_fc_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_fc_parameters_bias_ = (None)
        msg_token_53 = msg_token_52.view(floordiv_13, 8, 1024)
        msg_token_52 = floordiv_13 = None
        item_67 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_ln_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_ln_eps = (
            None
        )
        layer_norm_40 = torch.nn.functional.layer_norm(
            msg_token_53,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_ln_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_ln_parameters_bias_,
            item_67,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_ln_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_ln_parameters_bias_ = (item_67) = (
            None
        )
        size_42 = layer_norm_40.size()
        getitem_157 = size_42[1]
        size_42 = None
        queries_52 = torch._C._nn.linear(
            layer_norm_40,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_attn_modules_q_proj_parameters_bias_ = (None)
        keys_52 = torch._C._nn.linear(
            layer_norm_40,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_attn_modules_k_proj_parameters_bias_ = (None)
        values_52 = torch._C._nn.linear(
            layer_norm_40,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_attn_modules_v_proj_parameters_bias_,
        )
        layer_norm_40 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_attn_modules_v_proj_parameters_bias_ = (None)
        view_105 = queries_52.view(1, getitem_157, 16, 64)
        queries_52 = None
        queries_53 = view_105.transpose(1, 2)
        view_105 = None
        view_106 = keys_52.view(1, getitem_157, 16, 64)
        keys_52 = None
        keys_53 = view_106.transpose(1, 2)
        view_106 = None
        view_107 = values_52.view(1, getitem_157, 16, 64)
        values_52 = None
        values_53 = view_107.transpose(1, 2)
        view_107 = None
        transpose_134 = keys_53.transpose(-1, -2)
        keys_53 = None
        matmul_52 = torch.matmul(queries_53, transpose_134)
        queries_53 = transpose_134 = None
        item_68 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_attn_scale = (
            None
        )
        attn_weights_78 = matmul_52 * item_68
        matmul_52 = item_68 = None
        softmax_26 = torch.nn.functional.softmax(
            attn_weights_78, dim=-1, dtype=torch.float32
        )
        attn_weights_78 = None
        attn_weights_79 = softmax_26.to(torch.float32)
        softmax_26 = None
        attn_weights_80 = torch.nn.functional.dropout(
            attn_weights_79, p=0.0, training=False
        )
        attn_weights_79 = None
        attn_output_104 = torch.matmul(attn_weights_80, values_53)
        attn_weights_80 = values_53 = None
        transpose_135 = attn_output_104.transpose(1, 2)
        attn_output_104 = None
        attn_output_105 = transpose_135.contiguous()
        transpose_135 = None
        reshape_27 = attn_output_105.reshape(1, getitem_157, 1024)
        attn_output_105 = getitem_157 = None
        attn_output_106 = reshape_27.contiguous()
        reshape_27 = None
        attn_output_107 = torch._C._nn.linear(
            attn_output_106,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_106 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_message_attn_modules_out_proj_parameters_bias_ = (None)
        msg_token_54 = msg_token_53 + attn_output_107
        msg_token_53 = attn_output_107 = None
        msg_token_55 = msg_token_54.view(-1, 1, 1024)
        msg_token_54 = None
        hidden_states_118 = torch.cat([hidden_states_117, msg_token_55], dim=1)
        hidden_states_117 = msg_token_55 = None
        item_69 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm1_eps = (
            None
        )
        hidden_states_119 = torch.nn.functional.layer_norm(
            hidden_states_118,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm1_parameters_bias_,
            item_69,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm1_parameters_bias_ = (item_69) = (
            None
        )
        size_43 = hidden_states_119.size()
        getitem_159 = size_43[0]
        size_43 = None
        queries_54 = torch._C._nn.linear(
            hidden_states_119,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_54 = torch._C._nn.linear(
            hidden_states_119,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_54 = torch._C._nn.linear(
            hidden_states_119,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_119 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_109 = queries_54.view(getitem_159, 258, 16, 64)
        queries_54 = None
        queries_55 = view_109.transpose(1, 2)
        view_109 = None
        view_110 = keys_54.view(getitem_159, 258, 16, 64)
        keys_54 = None
        keys_55 = view_110.transpose(1, 2)
        view_110 = None
        view_111 = values_54.view(getitem_159, 258, 16, 64)
        values_54 = None
        values_55 = view_111.transpose(1, 2)
        view_111 = None
        transpose_139 = keys_55.transpose(-1, -2)
        keys_55 = None
        matmul_54 = torch.matmul(queries_55, transpose_139)
        queries_55 = transpose_139 = None
        item_70 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_scale = (
            None
        )
        attn_weights_81 = matmul_54 * item_70
        matmul_54 = item_70 = None
        softmax_27 = torch.nn.functional.softmax(
            attn_weights_81, dim=-1, dtype=torch.float32
        )
        attn_weights_81 = None
        attn_weights_82 = softmax_27.to(torch.float32)
        softmax_27 = None
        attn_weights_83 = torch.nn.functional.dropout(
            attn_weights_82, p=0.0, training=False
        )
        attn_weights_82 = None
        attn_output_108 = torch.matmul(attn_weights_83, values_55)
        attn_weights_83 = values_55 = None
        transpose_140 = attn_output_108.transpose(1, 2)
        attn_output_108 = None
        attn_output_109 = transpose_140.contiguous()
        transpose_140 = None
        reshape_28 = attn_output_109.reshape(getitem_159, 258, 1024)
        attn_output_109 = getitem_159 = None
        attn_output_110 = reshape_28.contiguous()
        reshape_28 = None
        attn_output_111 = torch._C._nn.linear(
            attn_output_110,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_110 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_120 = hidden_states_118 + attn_output_111
        hidden_states_118 = attn_output_111 = None
        hidden_states_121 = hidden_states_120[
            (slice(None, None, None), slice(None, 257, None), slice(None, None, None))
        ]
        hidden_states_120 = None
        item_71 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm2_eps = (
            None
        )
        hidden_states_122 = torch.nn.functional.layer_norm(
            hidden_states_121,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm2_parameters_bias_,
            item_71,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm2_parameters_bias_ = (item_71) = (
            None
        )
        hidden_states_123 = torch._C._nn.linear(
            hidden_states_122,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_122 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_54 = 1.702 * hidden_states_123
        sigmoid_13 = torch.sigmoid(mul_54)
        mul_54 = None
        hidden_states_124 = hidden_states_123 * sigmoid_13
        hidden_states_123 = sigmoid_13 = None
        hidden_states_125 = torch._C._nn.linear(
            hidden_states_124,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_124 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_126 = hidden_states_121 + hidden_states_125
        hidden_states_121 = hidden_states_125 = None
        size_44 = hidden_states_126.size()
        getitem_163 = size_44[0]
        size_44 = None
        floordiv_14 = getitem_163 // 8
        getitem_163 = None
        getitem_166 = hidden_states_126[
            (slice(None, None, None), 0, slice(None, None, None))
        ]
        msg_token_56 = torch._C._nn.linear(
            getitem_166,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_fc_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_fc_parameters_bias_,
        )
        getitem_166 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_fc_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_fc_parameters_bias_ = (None)
        msg_token_57 = msg_token_56.view(floordiv_14, 8, 1024)
        msg_token_56 = floordiv_14 = None
        item_72 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_ln_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_ln_eps = (
            None
        )
        layer_norm_43 = torch.nn.functional.layer_norm(
            msg_token_57,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_ln_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_ln_parameters_bias_,
            item_72,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_ln_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_ln_parameters_bias_ = (item_72) = (
            None
        )
        size_45 = layer_norm_43.size()
        getitem_168 = size_45[1]
        size_45 = None
        queries_56 = torch._C._nn.linear(
            layer_norm_43,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_attn_modules_q_proj_parameters_bias_ = (None)
        keys_56 = torch._C._nn.linear(
            layer_norm_43,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_attn_modules_k_proj_parameters_bias_ = (None)
        values_56 = torch._C._nn.linear(
            layer_norm_43,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_attn_modules_v_proj_parameters_bias_,
        )
        layer_norm_43 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_attn_modules_v_proj_parameters_bias_ = (None)
        view_113 = queries_56.view(1, getitem_168, 16, 64)
        queries_56 = None
        queries_57 = view_113.transpose(1, 2)
        view_113 = None
        view_114 = keys_56.view(1, getitem_168, 16, 64)
        keys_56 = None
        keys_57 = view_114.transpose(1, 2)
        view_114 = None
        view_115 = values_56.view(1, getitem_168, 16, 64)
        values_56 = None
        values_57 = view_115.transpose(1, 2)
        view_115 = None
        transpose_144 = keys_57.transpose(-1, -2)
        keys_57 = None
        matmul_56 = torch.matmul(queries_57, transpose_144)
        queries_57 = transpose_144 = None
        item_73 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_attn_scale = (
            None
        )
        attn_weights_84 = matmul_56 * item_73
        matmul_56 = item_73 = None
        softmax_28 = torch.nn.functional.softmax(
            attn_weights_84, dim=-1, dtype=torch.float32
        )
        attn_weights_84 = None
        attn_weights_85 = softmax_28.to(torch.float32)
        softmax_28 = None
        attn_weights_86 = torch.nn.functional.dropout(
            attn_weights_85, p=0.0, training=False
        )
        attn_weights_85 = None
        attn_output_112 = torch.matmul(attn_weights_86, values_57)
        attn_weights_86 = values_57 = None
        transpose_145 = attn_output_112.transpose(1, 2)
        attn_output_112 = None
        attn_output_113 = transpose_145.contiguous()
        transpose_145 = None
        reshape_29 = attn_output_113.reshape(1, getitem_168, 1024)
        attn_output_113 = getitem_168 = None
        attn_output_114 = reshape_29.contiguous()
        reshape_29 = None
        attn_output_115 = torch._C._nn.linear(
            attn_output_114,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_114 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_message_attn_modules_out_proj_parameters_bias_ = (None)
        msg_token_58 = msg_token_57 + attn_output_115
        msg_token_57 = attn_output_115 = None
        msg_token_59 = msg_token_58.view(-1, 1, 1024)
        msg_token_58 = None
        hidden_states_127 = torch.cat([hidden_states_126, msg_token_59], dim=1)
        hidden_states_126 = msg_token_59 = None
        item_74 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm1_eps = (
            None
        )
        hidden_states_128 = torch.nn.functional.layer_norm(
            hidden_states_127,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm1_parameters_bias_,
            item_74,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm1_parameters_bias_ = (item_74) = (
            None
        )
        size_46 = hidden_states_128.size()
        getitem_170 = size_46[0]
        size_46 = None
        queries_58 = torch._C._nn.linear(
            hidden_states_128,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_58 = torch._C._nn.linear(
            hidden_states_128,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_58 = torch._C._nn.linear(
            hidden_states_128,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_128 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_117 = queries_58.view(getitem_170, 258, 16, 64)
        queries_58 = None
        queries_59 = view_117.transpose(1, 2)
        view_117 = None
        view_118 = keys_58.view(getitem_170, 258, 16, 64)
        keys_58 = None
        keys_59 = view_118.transpose(1, 2)
        view_118 = None
        view_119 = values_58.view(getitem_170, 258, 16, 64)
        values_58 = None
        values_59 = view_119.transpose(1, 2)
        view_119 = None
        transpose_149 = keys_59.transpose(-1, -2)
        keys_59 = None
        matmul_58 = torch.matmul(queries_59, transpose_149)
        queries_59 = transpose_149 = None
        item_75 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_scale = (
            None
        )
        attn_weights_87 = matmul_58 * item_75
        matmul_58 = item_75 = None
        softmax_29 = torch.nn.functional.softmax(
            attn_weights_87, dim=-1, dtype=torch.float32
        )
        attn_weights_87 = None
        attn_weights_88 = softmax_29.to(torch.float32)
        softmax_29 = None
        attn_weights_89 = torch.nn.functional.dropout(
            attn_weights_88, p=0.0, training=False
        )
        attn_weights_88 = None
        attn_output_116 = torch.matmul(attn_weights_89, values_59)
        attn_weights_89 = values_59 = None
        transpose_150 = attn_output_116.transpose(1, 2)
        attn_output_116 = None
        attn_output_117 = transpose_150.contiguous()
        transpose_150 = None
        reshape_30 = attn_output_117.reshape(getitem_170, 258, 1024)
        attn_output_117 = getitem_170 = None
        attn_output_118 = reshape_30.contiguous()
        reshape_30 = None
        attn_output_119 = torch._C._nn.linear(
            attn_output_118,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_118 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_129 = hidden_states_127 + attn_output_119
        hidden_states_127 = attn_output_119 = None
        hidden_states_130 = hidden_states_129[
            (slice(None, None, None), slice(None, 257, None), slice(None, None, None))
        ]
        hidden_states_129 = None
        item_76 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm2_eps = (
            None
        )
        hidden_states_131 = torch.nn.functional.layer_norm(
            hidden_states_130,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm2_parameters_bias_,
            item_76,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm2_parameters_bias_ = (item_76) = (
            None
        )
        hidden_states_132 = torch._C._nn.linear(
            hidden_states_131,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_131 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_58 = 1.702 * hidden_states_132
        sigmoid_14 = torch.sigmoid(mul_58)
        mul_58 = None
        hidden_states_133 = hidden_states_132 * sigmoid_14
        hidden_states_132 = sigmoid_14 = None
        hidden_states_134 = torch._C._nn.linear(
            hidden_states_133,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_133 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_135 = hidden_states_130 + hidden_states_134
        hidden_states_130 = hidden_states_134 = None
        size_47 = hidden_states_135.size()
        getitem_174 = size_47[0]
        size_47 = None
        floordiv_15 = getitem_174 // 8
        getitem_174 = None
        getitem_177 = hidden_states_135[
            (slice(None, None, None), 0, slice(None, None, None))
        ]
        msg_token_60 = torch._C._nn.linear(
            getitem_177,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_fc_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_fc_parameters_bias_,
        )
        getitem_177 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_fc_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_fc_parameters_bias_ = (None)
        msg_token_61 = msg_token_60.view(floordiv_15, 8, 1024)
        msg_token_60 = floordiv_15 = None
        item_77 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_ln_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_ln_eps = (
            None
        )
        layer_norm_46 = torch.nn.functional.layer_norm(
            msg_token_61,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_ln_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_ln_parameters_bias_,
            item_77,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_ln_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_ln_parameters_bias_ = (item_77) = (
            None
        )
        size_48 = layer_norm_46.size()
        getitem_179 = size_48[1]
        size_48 = None
        queries_60 = torch._C._nn.linear(
            layer_norm_46,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_attn_modules_q_proj_parameters_bias_ = (None)
        keys_60 = torch._C._nn.linear(
            layer_norm_46,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_attn_modules_k_proj_parameters_bias_ = (None)
        values_60 = torch._C._nn.linear(
            layer_norm_46,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_attn_modules_v_proj_parameters_bias_,
        )
        layer_norm_46 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_attn_modules_v_proj_parameters_bias_ = (None)
        view_121 = queries_60.view(1, getitem_179, 16, 64)
        queries_60 = None
        queries_61 = view_121.transpose(1, 2)
        view_121 = None
        view_122 = keys_60.view(1, getitem_179, 16, 64)
        keys_60 = None
        keys_61 = view_122.transpose(1, 2)
        view_122 = None
        view_123 = values_60.view(1, getitem_179, 16, 64)
        values_60 = None
        values_61 = view_123.transpose(1, 2)
        view_123 = None
        transpose_154 = keys_61.transpose(-1, -2)
        keys_61 = None
        matmul_60 = torch.matmul(queries_61, transpose_154)
        queries_61 = transpose_154 = None
        item_78 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_attn_scale = (
            None
        )
        attn_weights_90 = matmul_60 * item_78
        matmul_60 = item_78 = None
        softmax_30 = torch.nn.functional.softmax(
            attn_weights_90, dim=-1, dtype=torch.float32
        )
        attn_weights_90 = None
        attn_weights_91 = softmax_30.to(torch.float32)
        softmax_30 = None
        attn_weights_92 = torch.nn.functional.dropout(
            attn_weights_91, p=0.0, training=False
        )
        attn_weights_91 = None
        attn_output_120 = torch.matmul(attn_weights_92, values_61)
        attn_weights_92 = values_61 = None
        transpose_155 = attn_output_120.transpose(1, 2)
        attn_output_120 = None
        attn_output_121 = transpose_155.contiguous()
        transpose_155 = None
        reshape_31 = attn_output_121.reshape(1, getitem_179, 1024)
        attn_output_121 = getitem_179 = None
        attn_output_122 = reshape_31.contiguous()
        reshape_31 = None
        attn_output_123 = torch._C._nn.linear(
            attn_output_122,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_122 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_message_attn_modules_out_proj_parameters_bias_ = (None)
        msg_token_62 = msg_token_61 + attn_output_123
        msg_token_61 = attn_output_123 = None
        msg_token_63 = msg_token_62.view(-1, 1, 1024)
        msg_token_62 = None
        hidden_states_136 = torch.cat([hidden_states_135, msg_token_63], dim=1)
        hidden_states_135 = msg_token_63 = None
        item_79 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm1_eps = (
            None
        )
        hidden_states_137 = torch.nn.functional.layer_norm(
            hidden_states_136,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm1_parameters_bias_,
            item_79,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm1_parameters_bias_ = (item_79) = (
            None
        )
        size_49 = hidden_states_137.size()
        getitem_181 = size_49[0]
        size_49 = None
        queries_62 = torch._C._nn.linear(
            hidden_states_137,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_62 = torch._C._nn.linear(
            hidden_states_137,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_62 = torch._C._nn.linear(
            hidden_states_137,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_137 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_125 = queries_62.view(getitem_181, 258, 16, 64)
        queries_62 = None
        queries_63 = view_125.transpose(1, 2)
        view_125 = None
        view_126 = keys_62.view(getitem_181, 258, 16, 64)
        keys_62 = None
        keys_63 = view_126.transpose(1, 2)
        view_126 = None
        view_127 = values_62.view(getitem_181, 258, 16, 64)
        values_62 = None
        values_63 = view_127.transpose(1, 2)
        view_127 = None
        transpose_159 = keys_63.transpose(-1, -2)
        keys_63 = None
        matmul_62 = torch.matmul(queries_63, transpose_159)
        queries_63 = transpose_159 = None
        item_80 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_scale = (
            None
        )
        attn_weights_93 = matmul_62 * item_80
        matmul_62 = item_80 = None
        softmax_31 = torch.nn.functional.softmax(
            attn_weights_93, dim=-1, dtype=torch.float32
        )
        attn_weights_93 = None
        attn_weights_94 = softmax_31.to(torch.float32)
        softmax_31 = None
        attn_weights_95 = torch.nn.functional.dropout(
            attn_weights_94, p=0.0, training=False
        )
        attn_weights_94 = None
        attn_output_124 = torch.matmul(attn_weights_95, values_63)
        attn_weights_95 = values_63 = None
        transpose_160 = attn_output_124.transpose(1, 2)
        attn_output_124 = None
        attn_output_125 = transpose_160.contiguous()
        transpose_160 = None
        reshape_32 = attn_output_125.reshape(getitem_181, 258, 1024)
        attn_output_125 = getitem_181 = None
        attn_output_126 = reshape_32.contiguous()
        reshape_32 = None
        attn_output_127 = torch._C._nn.linear(
            attn_output_126,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_126 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_138 = hidden_states_136 + attn_output_127
        hidden_states_136 = attn_output_127 = None
        hidden_states_139 = hidden_states_138[
            (slice(None, None, None), slice(None, 257, None), slice(None, None, None))
        ]
        hidden_states_138 = None
        item_81 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm2_eps = (
            None
        )
        hidden_states_140 = torch.nn.functional.layer_norm(
            hidden_states_139,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm2_parameters_bias_,
            item_81,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm2_parameters_bias_ = (item_81) = (
            None
        )
        hidden_states_141 = torch._C._nn.linear(
            hidden_states_140,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_140 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_62 = 1.702 * hidden_states_141
        sigmoid_15 = torch.sigmoid(mul_62)
        mul_62 = None
        hidden_states_142 = hidden_states_141 * sigmoid_15
        hidden_states_141 = sigmoid_15 = None
        hidden_states_143 = torch._C._nn.linear(
            hidden_states_142,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_142 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_144 = hidden_states_139 + hidden_states_143
        hidden_states_139 = hidden_states_143 = None
        size_50 = hidden_states_144.size()
        getitem_185 = size_50[0]
        size_50 = None
        floordiv_16 = getitem_185 // 8
        getitem_185 = None
        getitem_188 = hidden_states_144[
            (slice(None, None, None), 0, slice(None, None, None))
        ]
        msg_token_64 = torch._C._nn.linear(
            getitem_188,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_fc_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_fc_parameters_bias_,
        )
        getitem_188 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_fc_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_fc_parameters_bias_ = (None)
        msg_token_65 = msg_token_64.view(floordiv_16, 8, 1024)
        msg_token_64 = floordiv_16 = None
        item_82 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_ln_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_ln_eps = (
            None
        )
        layer_norm_49 = torch.nn.functional.layer_norm(
            msg_token_65,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_ln_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_ln_parameters_bias_,
            item_82,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_ln_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_ln_parameters_bias_ = (item_82) = (
            None
        )
        size_51 = layer_norm_49.size()
        getitem_190 = size_51[1]
        size_51 = None
        queries_64 = torch._C._nn.linear(
            layer_norm_49,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_attn_modules_q_proj_parameters_bias_ = (None)
        keys_64 = torch._C._nn.linear(
            layer_norm_49,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_attn_modules_k_proj_parameters_bias_ = (None)
        values_64 = torch._C._nn.linear(
            layer_norm_49,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_attn_modules_v_proj_parameters_bias_,
        )
        layer_norm_49 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_attn_modules_v_proj_parameters_bias_ = (None)
        view_129 = queries_64.view(1, getitem_190, 16, 64)
        queries_64 = None
        queries_65 = view_129.transpose(1, 2)
        view_129 = None
        view_130 = keys_64.view(1, getitem_190, 16, 64)
        keys_64 = None
        keys_65 = view_130.transpose(1, 2)
        view_130 = None
        view_131 = values_64.view(1, getitem_190, 16, 64)
        values_64 = None
        values_65 = view_131.transpose(1, 2)
        view_131 = None
        transpose_164 = keys_65.transpose(-1, -2)
        keys_65 = None
        matmul_64 = torch.matmul(queries_65, transpose_164)
        queries_65 = transpose_164 = None
        item_83 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_attn_scale = (
            None
        )
        attn_weights_96 = matmul_64 * item_83
        matmul_64 = item_83 = None
        softmax_32 = torch.nn.functional.softmax(
            attn_weights_96, dim=-1, dtype=torch.float32
        )
        attn_weights_96 = None
        attn_weights_97 = softmax_32.to(torch.float32)
        softmax_32 = None
        attn_weights_98 = torch.nn.functional.dropout(
            attn_weights_97, p=0.0, training=False
        )
        attn_weights_97 = None
        attn_output_128 = torch.matmul(attn_weights_98, values_65)
        attn_weights_98 = values_65 = None
        transpose_165 = attn_output_128.transpose(1, 2)
        attn_output_128 = None
        attn_output_129 = transpose_165.contiguous()
        transpose_165 = None
        reshape_33 = attn_output_129.reshape(1, getitem_190, 1024)
        attn_output_129 = getitem_190 = None
        attn_output_130 = reshape_33.contiguous()
        reshape_33 = None
        attn_output_131 = torch._C._nn.linear(
            attn_output_130,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_130 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_message_attn_modules_out_proj_parameters_bias_ = (None)
        msg_token_66 = msg_token_65 + attn_output_131
        msg_token_65 = attn_output_131 = None
        msg_token_67 = msg_token_66.view(-1, 1, 1024)
        msg_token_66 = None
        hidden_states_145 = torch.cat([hidden_states_144, msg_token_67], dim=1)
        hidden_states_144 = msg_token_67 = None
        item_84 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm1_eps = (
            None
        )
        hidden_states_146 = torch.nn.functional.layer_norm(
            hidden_states_145,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm1_parameters_bias_,
            item_84,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm1_parameters_bias_ = (item_84) = (
            None
        )
        size_52 = hidden_states_146.size()
        getitem_192 = size_52[0]
        size_52 = None
        queries_66 = torch._C._nn.linear(
            hidden_states_146,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_66 = torch._C._nn.linear(
            hidden_states_146,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_66 = torch._C._nn.linear(
            hidden_states_146,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_146 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_133 = queries_66.view(getitem_192, 258, 16, 64)
        queries_66 = None
        queries_67 = view_133.transpose(1, 2)
        view_133 = None
        view_134 = keys_66.view(getitem_192, 258, 16, 64)
        keys_66 = None
        keys_67 = view_134.transpose(1, 2)
        view_134 = None
        view_135 = values_66.view(getitem_192, 258, 16, 64)
        values_66 = None
        values_67 = view_135.transpose(1, 2)
        view_135 = None
        transpose_169 = keys_67.transpose(-1, -2)
        keys_67 = None
        matmul_66 = torch.matmul(queries_67, transpose_169)
        queries_67 = transpose_169 = None
        item_85 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_scale = (
            None
        )
        attn_weights_99 = matmul_66 * item_85
        matmul_66 = item_85 = None
        softmax_33 = torch.nn.functional.softmax(
            attn_weights_99, dim=-1, dtype=torch.float32
        )
        attn_weights_99 = None
        attn_weights_100 = softmax_33.to(torch.float32)
        softmax_33 = None
        attn_weights_101 = torch.nn.functional.dropout(
            attn_weights_100, p=0.0, training=False
        )
        attn_weights_100 = None
        attn_output_132 = torch.matmul(attn_weights_101, values_67)
        attn_weights_101 = values_67 = None
        transpose_170 = attn_output_132.transpose(1, 2)
        attn_output_132 = None
        attn_output_133 = transpose_170.contiguous()
        transpose_170 = None
        reshape_34 = attn_output_133.reshape(getitem_192, 258, 1024)
        attn_output_133 = getitem_192 = None
        attn_output_134 = reshape_34.contiguous()
        reshape_34 = None
        attn_output_135 = torch._C._nn.linear(
            attn_output_134,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_134 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_147 = hidden_states_145 + attn_output_135
        hidden_states_145 = attn_output_135 = None
        hidden_states_148 = hidden_states_147[
            (slice(None, None, None), slice(None, 257, None), slice(None, None, None))
        ]
        hidden_states_147 = None
        item_86 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm2_eps = (
            None
        )
        hidden_states_149 = torch.nn.functional.layer_norm(
            hidden_states_148,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm2_parameters_bias_,
            item_86,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm2_parameters_bias_ = (item_86) = (
            None
        )
        hidden_states_150 = torch._C._nn.linear(
            hidden_states_149,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_149 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_66 = 1.702 * hidden_states_150
        sigmoid_16 = torch.sigmoid(mul_66)
        mul_66 = None
        hidden_states_151 = hidden_states_150 * sigmoid_16
        hidden_states_150 = sigmoid_16 = None
        hidden_states_152 = torch._C._nn.linear(
            hidden_states_151,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_151 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_153 = hidden_states_148 + hidden_states_152
        hidden_states_148 = hidden_states_152 = None
        size_53 = hidden_states_153.size()
        getitem_196 = size_53[0]
        size_53 = None
        floordiv_17 = getitem_196 // 8
        getitem_196 = None
        getitem_199 = hidden_states_153[
            (slice(None, None, None), 0, slice(None, None, None))
        ]
        msg_token_68 = torch._C._nn.linear(
            getitem_199,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_fc_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_fc_parameters_bias_,
        )
        getitem_199 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_fc_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_fc_parameters_bias_ = (None)
        msg_token_69 = msg_token_68.view(floordiv_17, 8, 1024)
        msg_token_68 = floordiv_17 = None
        item_87 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_ln_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_ln_eps = (
            None
        )
        layer_norm_52 = torch.nn.functional.layer_norm(
            msg_token_69,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_ln_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_ln_parameters_bias_,
            item_87,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_ln_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_ln_parameters_bias_ = (item_87) = (
            None
        )
        size_54 = layer_norm_52.size()
        getitem_201 = size_54[1]
        size_54 = None
        queries_68 = torch._C._nn.linear(
            layer_norm_52,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_attn_modules_q_proj_parameters_bias_ = (None)
        keys_68 = torch._C._nn.linear(
            layer_norm_52,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_attn_modules_k_proj_parameters_bias_ = (None)
        values_68 = torch._C._nn.linear(
            layer_norm_52,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_attn_modules_v_proj_parameters_bias_,
        )
        layer_norm_52 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_attn_modules_v_proj_parameters_bias_ = (None)
        view_137 = queries_68.view(1, getitem_201, 16, 64)
        queries_68 = None
        queries_69 = view_137.transpose(1, 2)
        view_137 = None
        view_138 = keys_68.view(1, getitem_201, 16, 64)
        keys_68 = None
        keys_69 = view_138.transpose(1, 2)
        view_138 = None
        view_139 = values_68.view(1, getitem_201, 16, 64)
        values_68 = None
        values_69 = view_139.transpose(1, 2)
        view_139 = None
        transpose_174 = keys_69.transpose(-1, -2)
        keys_69 = None
        matmul_68 = torch.matmul(queries_69, transpose_174)
        queries_69 = transpose_174 = None
        item_88 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_attn_scale = (
            None
        )
        attn_weights_102 = matmul_68 * item_88
        matmul_68 = item_88 = None
        softmax_34 = torch.nn.functional.softmax(
            attn_weights_102, dim=-1, dtype=torch.float32
        )
        attn_weights_102 = None
        attn_weights_103 = softmax_34.to(torch.float32)
        softmax_34 = None
        attn_weights_104 = torch.nn.functional.dropout(
            attn_weights_103, p=0.0, training=False
        )
        attn_weights_103 = None
        attn_output_136 = torch.matmul(attn_weights_104, values_69)
        attn_weights_104 = values_69 = None
        transpose_175 = attn_output_136.transpose(1, 2)
        attn_output_136 = None
        attn_output_137 = transpose_175.contiguous()
        transpose_175 = None
        reshape_35 = attn_output_137.reshape(1, getitem_201, 1024)
        attn_output_137 = getitem_201 = None
        attn_output_138 = reshape_35.contiguous()
        reshape_35 = None
        attn_output_139 = torch._C._nn.linear(
            attn_output_138,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_138 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_message_attn_modules_out_proj_parameters_bias_ = (None)
        msg_token_70 = msg_token_69 + attn_output_139
        msg_token_69 = attn_output_139 = None
        msg_token_71 = msg_token_70.view(-1, 1, 1024)
        msg_token_70 = None
        hidden_states_154 = torch.cat([hidden_states_153, msg_token_71], dim=1)
        hidden_states_153 = msg_token_71 = None
        item_89 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm1_eps = (
            None
        )
        hidden_states_155 = torch.nn.functional.layer_norm(
            hidden_states_154,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm1_parameters_bias_,
            item_89,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm1_parameters_bias_ = (item_89) = (
            None
        )
        size_55 = hidden_states_155.size()
        getitem_203 = size_55[0]
        size_55 = None
        queries_70 = torch._C._nn.linear(
            hidden_states_155,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_70 = torch._C._nn.linear(
            hidden_states_155,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_70 = torch._C._nn.linear(
            hidden_states_155,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_155 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_141 = queries_70.view(getitem_203, 258, 16, 64)
        queries_70 = None
        queries_71 = view_141.transpose(1, 2)
        view_141 = None
        view_142 = keys_70.view(getitem_203, 258, 16, 64)
        keys_70 = None
        keys_71 = view_142.transpose(1, 2)
        view_142 = None
        view_143 = values_70.view(getitem_203, 258, 16, 64)
        values_70 = None
        values_71 = view_143.transpose(1, 2)
        view_143 = None
        transpose_179 = keys_71.transpose(-1, -2)
        keys_71 = None
        matmul_70 = torch.matmul(queries_71, transpose_179)
        queries_71 = transpose_179 = None
        item_90 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_scale = (
            None
        )
        attn_weights_105 = matmul_70 * item_90
        matmul_70 = item_90 = None
        softmax_35 = torch.nn.functional.softmax(
            attn_weights_105, dim=-1, dtype=torch.float32
        )
        attn_weights_105 = None
        attn_weights_106 = softmax_35.to(torch.float32)
        softmax_35 = None
        attn_weights_107 = torch.nn.functional.dropout(
            attn_weights_106, p=0.0, training=False
        )
        attn_weights_106 = None
        attn_output_140 = torch.matmul(attn_weights_107, values_71)
        attn_weights_107 = values_71 = None
        transpose_180 = attn_output_140.transpose(1, 2)
        attn_output_140 = None
        attn_output_141 = transpose_180.contiguous()
        transpose_180 = None
        reshape_36 = attn_output_141.reshape(getitem_203, 258, 1024)
        attn_output_141 = getitem_203 = None
        attn_output_142 = reshape_36.contiguous()
        reshape_36 = None
        attn_output_143 = torch._C._nn.linear(
            attn_output_142,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_142 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_156 = hidden_states_154 + attn_output_143
        hidden_states_154 = attn_output_143 = None
        hidden_states_157 = hidden_states_156[
            (slice(None, None, None), slice(None, 257, None), slice(None, None, None))
        ]
        hidden_states_156 = None
        item_91 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm2_eps = (
            None
        )
        hidden_states_158 = torch.nn.functional.layer_norm(
            hidden_states_157,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm2_parameters_bias_,
            item_91,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm2_parameters_bias_ = (item_91) = (
            None
        )
        hidden_states_159 = torch._C._nn.linear(
            hidden_states_158,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_158 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_70 = 1.702 * hidden_states_159
        sigmoid_17 = torch.sigmoid(mul_70)
        mul_70 = None
        hidden_states_160 = hidden_states_159 * sigmoid_17
        hidden_states_159 = sigmoid_17 = None
        hidden_states_161 = torch._C._nn.linear(
            hidden_states_160,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_160 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_162 = hidden_states_157 + hidden_states_161
        hidden_states_157 = hidden_states_161 = None
        size_56 = hidden_states_162.size()
        getitem_207 = size_56[0]
        size_56 = None
        floordiv_18 = getitem_207 // 8
        getitem_207 = None
        getitem_210 = hidden_states_162[
            (slice(None, None, None), 0, slice(None, None, None))
        ]
        msg_token_72 = torch._C._nn.linear(
            getitem_210,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_fc_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_fc_parameters_bias_,
        )
        getitem_210 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_fc_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_fc_parameters_bias_ = (None)
        msg_token_73 = msg_token_72.view(floordiv_18, 8, 1024)
        msg_token_72 = floordiv_18 = None
        item_92 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_ln_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_ln_eps = (
            None
        )
        layer_norm_55 = torch.nn.functional.layer_norm(
            msg_token_73,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_ln_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_ln_parameters_bias_,
            item_92,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_ln_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_ln_parameters_bias_ = (item_92) = (
            None
        )
        size_57 = layer_norm_55.size()
        getitem_212 = size_57[1]
        size_57 = None
        queries_72 = torch._C._nn.linear(
            layer_norm_55,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_attn_modules_q_proj_parameters_bias_ = (None)
        keys_72 = torch._C._nn.linear(
            layer_norm_55,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_attn_modules_k_proj_parameters_bias_ = (None)
        values_72 = torch._C._nn.linear(
            layer_norm_55,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_attn_modules_v_proj_parameters_bias_,
        )
        layer_norm_55 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_attn_modules_v_proj_parameters_bias_ = (None)
        view_145 = queries_72.view(1, getitem_212, 16, 64)
        queries_72 = None
        queries_73 = view_145.transpose(1, 2)
        view_145 = None
        view_146 = keys_72.view(1, getitem_212, 16, 64)
        keys_72 = None
        keys_73 = view_146.transpose(1, 2)
        view_146 = None
        view_147 = values_72.view(1, getitem_212, 16, 64)
        values_72 = None
        values_73 = view_147.transpose(1, 2)
        view_147 = None
        transpose_184 = keys_73.transpose(-1, -2)
        keys_73 = None
        matmul_72 = torch.matmul(queries_73, transpose_184)
        queries_73 = transpose_184 = None
        item_93 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_attn_scale = (
            None
        )
        attn_weights_108 = matmul_72 * item_93
        matmul_72 = item_93 = None
        softmax_36 = torch.nn.functional.softmax(
            attn_weights_108, dim=-1, dtype=torch.float32
        )
        attn_weights_108 = None
        attn_weights_109 = softmax_36.to(torch.float32)
        softmax_36 = None
        attn_weights_110 = torch.nn.functional.dropout(
            attn_weights_109, p=0.0, training=False
        )
        attn_weights_109 = None
        attn_output_144 = torch.matmul(attn_weights_110, values_73)
        attn_weights_110 = values_73 = None
        transpose_185 = attn_output_144.transpose(1, 2)
        attn_output_144 = None
        attn_output_145 = transpose_185.contiguous()
        transpose_185 = None
        reshape_37 = attn_output_145.reshape(1, getitem_212, 1024)
        attn_output_145 = getitem_212 = None
        attn_output_146 = reshape_37.contiguous()
        reshape_37 = None
        attn_output_147 = torch._C._nn.linear(
            attn_output_146,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_146 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_message_attn_modules_out_proj_parameters_bias_ = (None)
        msg_token_74 = msg_token_73 + attn_output_147
        msg_token_73 = attn_output_147 = None
        msg_token_75 = msg_token_74.view(-1, 1, 1024)
        msg_token_74 = None
        hidden_states_163 = torch.cat([hidden_states_162, msg_token_75], dim=1)
        hidden_states_162 = msg_token_75 = None
        item_94 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm1_eps = (
            None
        )
        hidden_states_164 = torch.nn.functional.layer_norm(
            hidden_states_163,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm1_parameters_bias_,
            item_94,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm1_parameters_bias_ = (item_94) = (
            None
        )
        size_58 = hidden_states_164.size()
        getitem_214 = size_58[0]
        size_58 = None
        queries_74 = torch._C._nn.linear(
            hidden_states_164,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_74 = torch._C._nn.linear(
            hidden_states_164,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_74 = torch._C._nn.linear(
            hidden_states_164,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_164 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_149 = queries_74.view(getitem_214, 258, 16, 64)
        queries_74 = None
        queries_75 = view_149.transpose(1, 2)
        view_149 = None
        view_150 = keys_74.view(getitem_214, 258, 16, 64)
        keys_74 = None
        keys_75 = view_150.transpose(1, 2)
        view_150 = None
        view_151 = values_74.view(getitem_214, 258, 16, 64)
        values_74 = None
        values_75 = view_151.transpose(1, 2)
        view_151 = None
        transpose_189 = keys_75.transpose(-1, -2)
        keys_75 = None
        matmul_74 = torch.matmul(queries_75, transpose_189)
        queries_75 = transpose_189 = None
        item_95 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_scale = (
            None
        )
        attn_weights_111 = matmul_74 * item_95
        matmul_74 = item_95 = None
        softmax_37 = torch.nn.functional.softmax(
            attn_weights_111, dim=-1, dtype=torch.float32
        )
        attn_weights_111 = None
        attn_weights_112 = softmax_37.to(torch.float32)
        softmax_37 = None
        attn_weights_113 = torch.nn.functional.dropout(
            attn_weights_112, p=0.0, training=False
        )
        attn_weights_112 = None
        attn_output_148 = torch.matmul(attn_weights_113, values_75)
        attn_weights_113 = values_75 = None
        transpose_190 = attn_output_148.transpose(1, 2)
        attn_output_148 = None
        attn_output_149 = transpose_190.contiguous()
        transpose_190 = None
        reshape_38 = attn_output_149.reshape(getitem_214, 258, 1024)
        attn_output_149 = getitem_214 = None
        attn_output_150 = reshape_38.contiguous()
        reshape_38 = None
        attn_output_151 = torch._C._nn.linear(
            attn_output_150,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_150 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_165 = hidden_states_163 + attn_output_151
        hidden_states_163 = attn_output_151 = None
        hidden_states_166 = hidden_states_165[
            (slice(None, None, None), slice(None, 257, None), slice(None, None, None))
        ]
        hidden_states_165 = None
        item_96 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm2_eps = (
            None
        )
        hidden_states_167 = torch.nn.functional.layer_norm(
            hidden_states_166,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm2_parameters_bias_,
            item_96,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm2_parameters_bias_ = (item_96) = (
            None
        )
        hidden_states_168 = torch._C._nn.linear(
            hidden_states_167,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_167 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_74 = 1.702 * hidden_states_168
        sigmoid_18 = torch.sigmoid(mul_74)
        mul_74 = None
        hidden_states_169 = hidden_states_168 * sigmoid_18
        hidden_states_168 = sigmoid_18 = None
        hidden_states_170 = torch._C._nn.linear(
            hidden_states_169,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_169 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_171 = hidden_states_166 + hidden_states_170
        hidden_states_166 = hidden_states_170 = None
        size_59 = hidden_states_171.size()
        getitem_218 = size_59[0]
        size_59 = None
        floordiv_19 = getitem_218 // 8
        getitem_218 = None
        getitem_221 = hidden_states_171[
            (slice(None, None, None), 0, slice(None, None, None))
        ]
        msg_token_76 = torch._C._nn.linear(
            getitem_221,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_fc_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_fc_parameters_bias_,
        )
        getitem_221 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_fc_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_fc_parameters_bias_ = (None)
        msg_token_77 = msg_token_76.view(floordiv_19, 8, 1024)
        msg_token_76 = floordiv_19 = None
        item_97 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_ln_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_ln_eps = (
            None
        )
        layer_norm_58 = torch.nn.functional.layer_norm(
            msg_token_77,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_ln_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_ln_parameters_bias_,
            item_97,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_ln_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_ln_parameters_bias_ = (item_97) = (
            None
        )
        size_60 = layer_norm_58.size()
        getitem_223 = size_60[1]
        size_60 = None
        queries_76 = torch._C._nn.linear(
            layer_norm_58,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_attn_modules_q_proj_parameters_bias_ = (None)
        keys_76 = torch._C._nn.linear(
            layer_norm_58,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_attn_modules_k_proj_parameters_bias_ = (None)
        values_76 = torch._C._nn.linear(
            layer_norm_58,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_attn_modules_v_proj_parameters_bias_,
        )
        layer_norm_58 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_attn_modules_v_proj_parameters_bias_ = (None)
        view_153 = queries_76.view(1, getitem_223, 16, 64)
        queries_76 = None
        queries_77 = view_153.transpose(1, 2)
        view_153 = None
        view_154 = keys_76.view(1, getitem_223, 16, 64)
        keys_76 = None
        keys_77 = view_154.transpose(1, 2)
        view_154 = None
        view_155 = values_76.view(1, getitem_223, 16, 64)
        values_76 = None
        values_77 = view_155.transpose(1, 2)
        view_155 = None
        transpose_194 = keys_77.transpose(-1, -2)
        keys_77 = None
        matmul_76 = torch.matmul(queries_77, transpose_194)
        queries_77 = transpose_194 = None
        item_98 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_attn_scale = (
            None
        )
        attn_weights_114 = matmul_76 * item_98
        matmul_76 = item_98 = None
        softmax_38 = torch.nn.functional.softmax(
            attn_weights_114, dim=-1, dtype=torch.float32
        )
        attn_weights_114 = None
        attn_weights_115 = softmax_38.to(torch.float32)
        softmax_38 = None
        attn_weights_116 = torch.nn.functional.dropout(
            attn_weights_115, p=0.0, training=False
        )
        attn_weights_115 = None
        attn_output_152 = torch.matmul(attn_weights_116, values_77)
        attn_weights_116 = values_77 = None
        transpose_195 = attn_output_152.transpose(1, 2)
        attn_output_152 = None
        attn_output_153 = transpose_195.contiguous()
        transpose_195 = None
        reshape_39 = attn_output_153.reshape(1, getitem_223, 1024)
        attn_output_153 = getitem_223 = None
        attn_output_154 = reshape_39.contiguous()
        reshape_39 = None
        attn_output_155 = torch._C._nn.linear(
            attn_output_154,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_154 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_message_attn_modules_out_proj_parameters_bias_ = (None)
        msg_token_78 = msg_token_77 + attn_output_155
        msg_token_77 = attn_output_155 = None
        msg_token_79 = msg_token_78.view(-1, 1, 1024)
        msg_token_78 = None
        hidden_states_172 = torch.cat([hidden_states_171, msg_token_79], dim=1)
        hidden_states_171 = msg_token_79 = None
        item_99 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm1_eps = (
            None
        )
        hidden_states_173 = torch.nn.functional.layer_norm(
            hidden_states_172,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm1_parameters_bias_,
            item_99,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm1_parameters_bias_ = (item_99) = (
            None
        )
        size_61 = hidden_states_173.size()
        getitem_225 = size_61[0]
        size_61 = None
        queries_78 = torch._C._nn.linear(
            hidden_states_173,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_78 = torch._C._nn.linear(
            hidden_states_173,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_78 = torch._C._nn.linear(
            hidden_states_173,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_173 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_157 = queries_78.view(getitem_225, 258, 16, 64)
        queries_78 = None
        queries_79 = view_157.transpose(1, 2)
        view_157 = None
        view_158 = keys_78.view(getitem_225, 258, 16, 64)
        keys_78 = None
        keys_79 = view_158.transpose(1, 2)
        view_158 = None
        view_159 = values_78.view(getitem_225, 258, 16, 64)
        values_78 = None
        values_79 = view_159.transpose(1, 2)
        view_159 = None
        transpose_199 = keys_79.transpose(-1, -2)
        keys_79 = None
        matmul_78 = torch.matmul(queries_79, transpose_199)
        queries_79 = transpose_199 = None
        item_100 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_scale = (
            None
        )
        attn_weights_117 = matmul_78 * item_100
        matmul_78 = item_100 = None
        softmax_39 = torch.nn.functional.softmax(
            attn_weights_117, dim=-1, dtype=torch.float32
        )
        attn_weights_117 = None
        attn_weights_118 = softmax_39.to(torch.float32)
        softmax_39 = None
        attn_weights_119 = torch.nn.functional.dropout(
            attn_weights_118, p=0.0, training=False
        )
        attn_weights_118 = None
        attn_output_156 = torch.matmul(attn_weights_119, values_79)
        attn_weights_119 = values_79 = None
        transpose_200 = attn_output_156.transpose(1, 2)
        attn_output_156 = None
        attn_output_157 = transpose_200.contiguous()
        transpose_200 = None
        reshape_40 = attn_output_157.reshape(getitem_225, 258, 1024)
        attn_output_157 = getitem_225 = None
        attn_output_158 = reshape_40.contiguous()
        reshape_40 = None
        attn_output_159 = torch._C._nn.linear(
            attn_output_158,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_158 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_174 = hidden_states_172 + attn_output_159
        hidden_states_172 = attn_output_159 = None
        hidden_states_175 = hidden_states_174[
            (slice(None, None, None), slice(None, 257, None), slice(None, None, None))
        ]
        hidden_states_174 = None
        item_101 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm2_eps = (
            None
        )
        hidden_states_176 = torch.nn.functional.layer_norm(
            hidden_states_175,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm2_parameters_bias_,
            item_101,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm2_parameters_bias_ = (item_101) = (
            None
        )
        hidden_states_177 = torch._C._nn.linear(
            hidden_states_176,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_176 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_78 = 1.702 * hidden_states_177
        sigmoid_19 = torch.sigmoid(mul_78)
        mul_78 = None
        hidden_states_178 = hidden_states_177 * sigmoid_19
        hidden_states_177 = sigmoid_19 = None
        hidden_states_179 = torch._C._nn.linear(
            hidden_states_178,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_178 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_180 = hidden_states_175 + hidden_states_179
        hidden_states_175 = hidden_states_179 = None
        size_62 = hidden_states_180.size()
        getitem_229 = size_62[0]
        size_62 = None
        floordiv_20 = getitem_229 // 8
        getitem_229 = None
        getitem_232 = hidden_states_180[
            (slice(None, None, None), 0, slice(None, None, None))
        ]
        msg_token_80 = torch._C._nn.linear(
            getitem_232,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_fc_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_fc_parameters_bias_,
        )
        getitem_232 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_fc_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_fc_parameters_bias_ = (None)
        msg_token_81 = msg_token_80.view(floordiv_20, 8, 1024)
        msg_token_80 = floordiv_20 = None
        item_102 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_ln_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_ln_eps = (
            None
        )
        layer_norm_61 = torch.nn.functional.layer_norm(
            msg_token_81,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_ln_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_ln_parameters_bias_,
            item_102,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_ln_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_ln_parameters_bias_ = (item_102) = (
            None
        )
        size_63 = layer_norm_61.size()
        getitem_234 = size_63[1]
        size_63 = None
        queries_80 = torch._C._nn.linear(
            layer_norm_61,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_attn_modules_q_proj_parameters_bias_ = (None)
        keys_80 = torch._C._nn.linear(
            layer_norm_61,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_attn_modules_k_proj_parameters_bias_ = (None)
        values_80 = torch._C._nn.linear(
            layer_norm_61,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_attn_modules_v_proj_parameters_bias_,
        )
        layer_norm_61 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_attn_modules_v_proj_parameters_bias_ = (None)
        view_161 = queries_80.view(1, getitem_234, 16, 64)
        queries_80 = None
        queries_81 = view_161.transpose(1, 2)
        view_161 = None
        view_162 = keys_80.view(1, getitem_234, 16, 64)
        keys_80 = None
        keys_81 = view_162.transpose(1, 2)
        view_162 = None
        view_163 = values_80.view(1, getitem_234, 16, 64)
        values_80 = None
        values_81 = view_163.transpose(1, 2)
        view_163 = None
        transpose_204 = keys_81.transpose(-1, -2)
        keys_81 = None
        matmul_80 = torch.matmul(queries_81, transpose_204)
        queries_81 = transpose_204 = None
        item_103 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_attn_scale = (
            None
        )
        attn_weights_120 = matmul_80 * item_103
        matmul_80 = item_103 = None
        softmax_40 = torch.nn.functional.softmax(
            attn_weights_120, dim=-1, dtype=torch.float32
        )
        attn_weights_120 = None
        attn_weights_121 = softmax_40.to(torch.float32)
        softmax_40 = None
        attn_weights_122 = torch.nn.functional.dropout(
            attn_weights_121, p=0.0, training=False
        )
        attn_weights_121 = None
        attn_output_160 = torch.matmul(attn_weights_122, values_81)
        attn_weights_122 = values_81 = None
        transpose_205 = attn_output_160.transpose(1, 2)
        attn_output_160 = None
        attn_output_161 = transpose_205.contiguous()
        transpose_205 = None
        reshape_41 = attn_output_161.reshape(1, getitem_234, 1024)
        attn_output_161 = getitem_234 = None
        attn_output_162 = reshape_41.contiguous()
        reshape_41 = None
        attn_output_163 = torch._C._nn.linear(
            attn_output_162,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_162 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_message_attn_modules_out_proj_parameters_bias_ = (None)
        msg_token_82 = msg_token_81 + attn_output_163
        msg_token_81 = attn_output_163 = None
        msg_token_83 = msg_token_82.view(-1, 1, 1024)
        msg_token_82 = None
        hidden_states_181 = torch.cat([hidden_states_180, msg_token_83], dim=1)
        hidden_states_180 = msg_token_83 = None
        item_104 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm1_eps = (
            None
        )
        hidden_states_182 = torch.nn.functional.layer_norm(
            hidden_states_181,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm1_parameters_bias_,
            item_104,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm1_parameters_bias_ = (item_104) = (
            None
        )
        size_64 = hidden_states_182.size()
        getitem_236 = size_64[0]
        size_64 = None
        queries_82 = torch._C._nn.linear(
            hidden_states_182,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_82 = torch._C._nn.linear(
            hidden_states_182,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_82 = torch._C._nn.linear(
            hidden_states_182,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_182 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_165 = queries_82.view(getitem_236, 258, 16, 64)
        queries_82 = None
        queries_83 = view_165.transpose(1, 2)
        view_165 = None
        view_166 = keys_82.view(getitem_236, 258, 16, 64)
        keys_82 = None
        keys_83 = view_166.transpose(1, 2)
        view_166 = None
        view_167 = values_82.view(getitem_236, 258, 16, 64)
        values_82 = None
        values_83 = view_167.transpose(1, 2)
        view_167 = None
        transpose_209 = keys_83.transpose(-1, -2)
        keys_83 = None
        matmul_82 = torch.matmul(queries_83, transpose_209)
        queries_83 = transpose_209 = None
        item_105 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_scale = (
            None
        )
        attn_weights_123 = matmul_82 * item_105
        matmul_82 = item_105 = None
        softmax_41 = torch.nn.functional.softmax(
            attn_weights_123, dim=-1, dtype=torch.float32
        )
        attn_weights_123 = None
        attn_weights_124 = softmax_41.to(torch.float32)
        softmax_41 = None
        attn_weights_125 = torch.nn.functional.dropout(
            attn_weights_124, p=0.0, training=False
        )
        attn_weights_124 = None
        attn_output_164 = torch.matmul(attn_weights_125, values_83)
        attn_weights_125 = values_83 = None
        transpose_210 = attn_output_164.transpose(1, 2)
        attn_output_164 = None
        attn_output_165 = transpose_210.contiguous()
        transpose_210 = None
        reshape_42 = attn_output_165.reshape(getitem_236, 258, 1024)
        attn_output_165 = getitem_236 = None
        attn_output_166 = reshape_42.contiguous()
        reshape_42 = None
        attn_output_167 = torch._C._nn.linear(
            attn_output_166,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_166 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_183 = hidden_states_181 + attn_output_167
        hidden_states_181 = attn_output_167 = None
        hidden_states_184 = hidden_states_183[
            (slice(None, None, None), slice(None, 257, None), slice(None, None, None))
        ]
        hidden_states_183 = None
        item_106 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm2_eps = (
            None
        )
        hidden_states_185 = torch.nn.functional.layer_norm(
            hidden_states_184,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm2_parameters_bias_,
            item_106,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm2_parameters_bias_ = (item_106) = (
            None
        )
        hidden_states_186 = torch._C._nn.linear(
            hidden_states_185,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_185 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_82 = 1.702 * hidden_states_186
        sigmoid_20 = torch.sigmoid(mul_82)
        mul_82 = None
        hidden_states_187 = hidden_states_186 * sigmoid_20
        hidden_states_186 = sigmoid_20 = None
        hidden_states_188 = torch._C._nn.linear(
            hidden_states_187,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_187 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_189 = hidden_states_184 + hidden_states_188
        hidden_states_184 = hidden_states_188 = None
        size_65 = hidden_states_189.size()
        getitem_240 = size_65[0]
        size_65 = None
        floordiv_21 = getitem_240 // 8
        getitem_240 = None
        getitem_243 = hidden_states_189[
            (slice(None, None, None), 0, slice(None, None, None))
        ]
        msg_token_84 = torch._C._nn.linear(
            getitem_243,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_fc_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_fc_parameters_bias_,
        )
        getitem_243 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_fc_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_fc_parameters_bias_ = (None)
        msg_token_85 = msg_token_84.view(floordiv_21, 8, 1024)
        msg_token_84 = floordiv_21 = None
        item_107 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_ln_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_ln_eps = (
            None
        )
        layer_norm_64 = torch.nn.functional.layer_norm(
            msg_token_85,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_ln_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_ln_parameters_bias_,
            item_107,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_ln_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_ln_parameters_bias_ = (item_107) = (
            None
        )
        size_66 = layer_norm_64.size()
        getitem_245 = size_66[1]
        size_66 = None
        queries_84 = torch._C._nn.linear(
            layer_norm_64,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_attn_modules_q_proj_parameters_bias_ = (None)
        keys_84 = torch._C._nn.linear(
            layer_norm_64,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_attn_modules_k_proj_parameters_bias_ = (None)
        values_84 = torch._C._nn.linear(
            layer_norm_64,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_attn_modules_v_proj_parameters_bias_,
        )
        layer_norm_64 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_attn_modules_v_proj_parameters_bias_ = (None)
        view_169 = queries_84.view(1, getitem_245, 16, 64)
        queries_84 = None
        queries_85 = view_169.transpose(1, 2)
        view_169 = None
        view_170 = keys_84.view(1, getitem_245, 16, 64)
        keys_84 = None
        keys_85 = view_170.transpose(1, 2)
        view_170 = None
        view_171 = values_84.view(1, getitem_245, 16, 64)
        values_84 = None
        values_85 = view_171.transpose(1, 2)
        view_171 = None
        transpose_214 = keys_85.transpose(-1, -2)
        keys_85 = None
        matmul_84 = torch.matmul(queries_85, transpose_214)
        queries_85 = transpose_214 = None
        item_108 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_attn_scale = (
            None
        )
        attn_weights_126 = matmul_84 * item_108
        matmul_84 = item_108 = None
        softmax_42 = torch.nn.functional.softmax(
            attn_weights_126, dim=-1, dtype=torch.float32
        )
        attn_weights_126 = None
        attn_weights_127 = softmax_42.to(torch.float32)
        softmax_42 = None
        attn_weights_128 = torch.nn.functional.dropout(
            attn_weights_127, p=0.0, training=False
        )
        attn_weights_127 = None
        attn_output_168 = torch.matmul(attn_weights_128, values_85)
        attn_weights_128 = values_85 = None
        transpose_215 = attn_output_168.transpose(1, 2)
        attn_output_168 = None
        attn_output_169 = transpose_215.contiguous()
        transpose_215 = None
        reshape_43 = attn_output_169.reshape(1, getitem_245, 1024)
        attn_output_169 = getitem_245 = None
        attn_output_170 = reshape_43.contiguous()
        reshape_43 = None
        attn_output_171 = torch._C._nn.linear(
            attn_output_170,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_170 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_message_attn_modules_out_proj_parameters_bias_ = (None)
        msg_token_86 = msg_token_85 + attn_output_171
        msg_token_85 = attn_output_171 = None
        msg_token_87 = msg_token_86.view(-1, 1, 1024)
        msg_token_86 = None
        hidden_states_190 = torch.cat([hidden_states_189, msg_token_87], dim=1)
        hidden_states_189 = msg_token_87 = None
        item_109 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm1_eps = (
            None
        )
        hidden_states_191 = torch.nn.functional.layer_norm(
            hidden_states_190,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm1_parameters_bias_,
            item_109,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm1_parameters_bias_ = (item_109) = (
            None
        )
        size_67 = hidden_states_191.size()
        getitem_247 = size_67[0]
        size_67 = None
        queries_86 = torch._C._nn.linear(
            hidden_states_191,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_86 = torch._C._nn.linear(
            hidden_states_191,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_86 = torch._C._nn.linear(
            hidden_states_191,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_191 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_173 = queries_86.view(getitem_247, 258, 16, 64)
        queries_86 = None
        queries_87 = view_173.transpose(1, 2)
        view_173 = None
        view_174 = keys_86.view(getitem_247, 258, 16, 64)
        keys_86 = None
        keys_87 = view_174.transpose(1, 2)
        view_174 = None
        view_175 = values_86.view(getitem_247, 258, 16, 64)
        values_86 = None
        values_87 = view_175.transpose(1, 2)
        view_175 = None
        transpose_219 = keys_87.transpose(-1, -2)
        keys_87 = None
        matmul_86 = torch.matmul(queries_87, transpose_219)
        queries_87 = transpose_219 = None
        item_110 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_scale = (
            None
        )
        attn_weights_129 = matmul_86 * item_110
        matmul_86 = item_110 = None
        softmax_43 = torch.nn.functional.softmax(
            attn_weights_129, dim=-1, dtype=torch.float32
        )
        attn_weights_129 = None
        attn_weights_130 = softmax_43.to(torch.float32)
        softmax_43 = None
        attn_weights_131 = torch.nn.functional.dropout(
            attn_weights_130, p=0.0, training=False
        )
        attn_weights_130 = None
        attn_output_172 = torch.matmul(attn_weights_131, values_87)
        attn_weights_131 = values_87 = None
        transpose_220 = attn_output_172.transpose(1, 2)
        attn_output_172 = None
        attn_output_173 = transpose_220.contiguous()
        transpose_220 = None
        reshape_44 = attn_output_173.reshape(getitem_247, 258, 1024)
        attn_output_173 = getitem_247 = None
        attn_output_174 = reshape_44.contiguous()
        reshape_44 = None
        attn_output_175 = torch._C._nn.linear(
            attn_output_174,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_174 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_192 = hidden_states_190 + attn_output_175
        hidden_states_190 = attn_output_175 = None
        hidden_states_193 = hidden_states_192[
            (slice(None, None, None), slice(None, 257, None), slice(None, None, None))
        ]
        hidden_states_192 = None
        item_111 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm2_eps = (
            None
        )
        hidden_states_194 = torch.nn.functional.layer_norm(
            hidden_states_193,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm2_parameters_bias_,
            item_111,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm2_parameters_bias_ = (item_111) = (
            None
        )
        hidden_states_195 = torch._C._nn.linear(
            hidden_states_194,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_194 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_86 = 1.702 * hidden_states_195
        sigmoid_21 = torch.sigmoid(mul_86)
        mul_86 = None
        hidden_states_196 = hidden_states_195 * sigmoid_21
        hidden_states_195 = sigmoid_21 = None
        hidden_states_197 = torch._C._nn.linear(
            hidden_states_196,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_196 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_198 = hidden_states_193 + hidden_states_197
        hidden_states_193 = hidden_states_197 = None
        size_68 = hidden_states_198.size()
        getitem_251 = size_68[0]
        size_68 = None
        floordiv_22 = getitem_251 // 8
        getitem_251 = None
        getitem_254 = hidden_states_198[
            (slice(None, None, None), 0, slice(None, None, None))
        ]
        msg_token_88 = torch._C._nn.linear(
            getitem_254,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_fc_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_fc_parameters_bias_,
        )
        getitem_254 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_fc_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_fc_parameters_bias_ = (None)
        msg_token_89 = msg_token_88.view(floordiv_22, 8, 1024)
        msg_token_88 = floordiv_22 = None
        item_112 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_ln_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_ln_eps = (
            None
        )
        layer_norm_67 = torch.nn.functional.layer_norm(
            msg_token_89,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_ln_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_ln_parameters_bias_,
            item_112,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_ln_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_ln_parameters_bias_ = (item_112) = (
            None
        )
        size_69 = layer_norm_67.size()
        getitem_256 = size_69[1]
        size_69 = None
        queries_88 = torch._C._nn.linear(
            layer_norm_67,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_attn_modules_q_proj_parameters_bias_ = (None)
        keys_88 = torch._C._nn.linear(
            layer_norm_67,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_attn_modules_k_proj_parameters_bias_ = (None)
        values_88 = torch._C._nn.linear(
            layer_norm_67,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_attn_modules_v_proj_parameters_bias_,
        )
        layer_norm_67 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_attn_modules_v_proj_parameters_bias_ = (None)
        view_177 = queries_88.view(1, getitem_256, 16, 64)
        queries_88 = None
        queries_89 = view_177.transpose(1, 2)
        view_177 = None
        view_178 = keys_88.view(1, getitem_256, 16, 64)
        keys_88 = None
        keys_89 = view_178.transpose(1, 2)
        view_178 = None
        view_179 = values_88.view(1, getitem_256, 16, 64)
        values_88 = None
        values_89 = view_179.transpose(1, 2)
        view_179 = None
        transpose_224 = keys_89.transpose(-1, -2)
        keys_89 = None
        matmul_88 = torch.matmul(queries_89, transpose_224)
        queries_89 = transpose_224 = None
        item_113 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_attn_scale = (
            None
        )
        attn_weights_132 = matmul_88 * item_113
        matmul_88 = item_113 = None
        softmax_44 = torch.nn.functional.softmax(
            attn_weights_132, dim=-1, dtype=torch.float32
        )
        attn_weights_132 = None
        attn_weights_133 = softmax_44.to(torch.float32)
        softmax_44 = None
        attn_weights_134 = torch.nn.functional.dropout(
            attn_weights_133, p=0.0, training=False
        )
        attn_weights_133 = None
        attn_output_176 = torch.matmul(attn_weights_134, values_89)
        attn_weights_134 = values_89 = None
        transpose_225 = attn_output_176.transpose(1, 2)
        attn_output_176 = None
        attn_output_177 = transpose_225.contiguous()
        transpose_225 = None
        reshape_45 = attn_output_177.reshape(1, getitem_256, 1024)
        attn_output_177 = getitem_256 = None
        attn_output_178 = reshape_45.contiguous()
        reshape_45 = None
        attn_output_179 = torch._C._nn.linear(
            attn_output_178,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_178 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_message_attn_modules_out_proj_parameters_bias_ = (None)
        msg_token_90 = msg_token_89 + attn_output_179
        msg_token_89 = attn_output_179 = None
        msg_token_91 = msg_token_90.view(-1, 1, 1024)
        msg_token_90 = None
        hidden_states_199 = torch.cat([hidden_states_198, msg_token_91], dim=1)
        hidden_states_198 = msg_token_91 = None
        item_114 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm1_eps = (
            None
        )
        hidden_states_200 = torch.nn.functional.layer_norm(
            hidden_states_199,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm1_parameters_bias_,
            item_114,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm1_parameters_bias_ = (item_114) = (
            None
        )
        size_70 = hidden_states_200.size()
        getitem_258 = size_70[0]
        size_70 = None
        queries_90 = torch._C._nn.linear(
            hidden_states_200,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_90 = torch._C._nn.linear(
            hidden_states_200,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_90 = torch._C._nn.linear(
            hidden_states_200,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_200 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_181 = queries_90.view(getitem_258, 258, 16, 64)
        queries_90 = None
        queries_91 = view_181.transpose(1, 2)
        view_181 = None
        view_182 = keys_90.view(getitem_258, 258, 16, 64)
        keys_90 = None
        keys_91 = view_182.transpose(1, 2)
        view_182 = None
        view_183 = values_90.view(getitem_258, 258, 16, 64)
        values_90 = None
        values_91 = view_183.transpose(1, 2)
        view_183 = None
        transpose_229 = keys_91.transpose(-1, -2)
        keys_91 = None
        matmul_90 = torch.matmul(queries_91, transpose_229)
        queries_91 = transpose_229 = None
        item_115 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_scale = (
            None
        )
        attn_weights_135 = matmul_90 * item_115
        matmul_90 = item_115 = None
        softmax_45 = torch.nn.functional.softmax(
            attn_weights_135, dim=-1, dtype=torch.float32
        )
        attn_weights_135 = None
        attn_weights_136 = softmax_45.to(torch.float32)
        softmax_45 = None
        attn_weights_137 = torch.nn.functional.dropout(
            attn_weights_136, p=0.0, training=False
        )
        attn_weights_136 = None
        attn_output_180 = torch.matmul(attn_weights_137, values_91)
        attn_weights_137 = values_91 = None
        transpose_230 = attn_output_180.transpose(1, 2)
        attn_output_180 = None
        attn_output_181 = transpose_230.contiguous()
        transpose_230 = None
        reshape_46 = attn_output_181.reshape(getitem_258, 258, 1024)
        attn_output_181 = getitem_258 = None
        attn_output_182 = reshape_46.contiguous()
        reshape_46 = None
        attn_output_183 = torch._C._nn.linear(
            attn_output_182,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_182 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_201 = hidden_states_199 + attn_output_183
        hidden_states_199 = attn_output_183 = None
        hidden_states_202 = hidden_states_201[
            (slice(None, None, None), slice(None, 257, None), slice(None, None, None))
        ]
        hidden_states_201 = None
        item_116 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm2_eps = (
            None
        )
        hidden_states_203 = torch.nn.functional.layer_norm(
            hidden_states_202,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm2_parameters_bias_,
            item_116,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm2_parameters_bias_ = (item_116) = (
            None
        )
        hidden_states_204 = torch._C._nn.linear(
            hidden_states_203,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_203 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_90 = 1.702 * hidden_states_204
        sigmoid_22 = torch.sigmoid(mul_90)
        mul_90 = None
        hidden_states_205 = hidden_states_204 * sigmoid_22
        hidden_states_204 = sigmoid_22 = None
        hidden_states_206 = torch._C._nn.linear(
            hidden_states_205,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_205 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_207 = hidden_states_202 + hidden_states_206
        hidden_states_202 = hidden_states_206 = None
        size_71 = hidden_states_207.size()
        getitem_262 = size_71[0]
        size_71 = None
        floordiv_23 = getitem_262 // 8
        getitem_262 = None
        getitem_265 = hidden_states_207[
            (slice(None, None, None), 0, slice(None, None, None))
        ]
        msg_token_92 = torch._C._nn.linear(
            getitem_265,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_fc_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_fc_parameters_bias_,
        )
        getitem_265 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_fc_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_fc_parameters_bias_ = (None)
        msg_token_93 = msg_token_92.view(floordiv_23, 8, 1024)
        msg_token_92 = floordiv_23 = None
        item_117 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_ln_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_ln_eps = (
            None
        )
        layer_norm_70 = torch.nn.functional.layer_norm(
            msg_token_93,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_ln_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_ln_parameters_bias_,
            item_117,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_ln_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_ln_parameters_bias_ = (item_117) = (
            None
        )
        size_72 = layer_norm_70.size()
        getitem_267 = size_72[1]
        size_72 = None
        queries_92 = torch._C._nn.linear(
            layer_norm_70,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_attn_modules_q_proj_parameters_bias_ = (None)
        keys_92 = torch._C._nn.linear(
            layer_norm_70,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_attn_modules_k_proj_parameters_bias_ = (None)
        values_92 = torch._C._nn.linear(
            layer_norm_70,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_attn_modules_v_proj_parameters_bias_,
        )
        layer_norm_70 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_attn_modules_v_proj_parameters_bias_ = (None)
        view_185 = queries_92.view(1, getitem_267, 16, 64)
        queries_92 = None
        queries_93 = view_185.transpose(1, 2)
        view_185 = None
        view_186 = keys_92.view(1, getitem_267, 16, 64)
        keys_92 = None
        keys_93 = view_186.transpose(1, 2)
        view_186 = None
        view_187 = values_92.view(1, getitem_267, 16, 64)
        values_92 = None
        values_93 = view_187.transpose(1, 2)
        view_187 = None
        transpose_234 = keys_93.transpose(-1, -2)
        keys_93 = None
        matmul_92 = torch.matmul(queries_93, transpose_234)
        queries_93 = transpose_234 = None
        item_118 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_attn_scale = (
            None
        )
        attn_weights_138 = matmul_92 * item_118
        matmul_92 = item_118 = None
        softmax_46 = torch.nn.functional.softmax(
            attn_weights_138, dim=-1, dtype=torch.float32
        )
        attn_weights_138 = None
        attn_weights_139 = softmax_46.to(torch.float32)
        softmax_46 = None
        attn_weights_140 = torch.nn.functional.dropout(
            attn_weights_139, p=0.0, training=False
        )
        attn_weights_139 = None
        attn_output_184 = torch.matmul(attn_weights_140, values_93)
        attn_weights_140 = values_93 = None
        transpose_235 = attn_output_184.transpose(1, 2)
        attn_output_184 = None
        attn_output_185 = transpose_235.contiguous()
        transpose_235 = None
        reshape_47 = attn_output_185.reshape(1, getitem_267, 1024)
        attn_output_185 = getitem_267 = None
        attn_output_186 = reshape_47.contiguous()
        reshape_47 = None
        attn_output_187 = torch._C._nn.linear(
            attn_output_186,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_186 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_message_attn_modules_out_proj_parameters_bias_ = (None)
        msg_token_94 = msg_token_93 + attn_output_187
        msg_token_93 = attn_output_187 = None
        msg_token_95 = msg_token_94.view(-1, 1, 1024)
        msg_token_94 = None
        hidden_states_208 = torch.cat([hidden_states_207, msg_token_95], dim=1)
        hidden_states_207 = msg_token_95 = None
        item_119 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm1_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm1_eps = (
            None
        )
        hidden_states_209 = torch.nn.functional.layer_norm(
            hidden_states_208,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm1_parameters_bias_,
            item_119,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm1_parameters_bias_ = (item_119) = (
            None
        )
        size_73 = hidden_states_209.size()
        getitem_269 = size_73[0]
        size_73 = None
        queries_94 = torch._C._nn.linear(
            hidden_states_209,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_94 = torch._C._nn.linear(
            hidden_states_209,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_94 = torch._C._nn.linear(
            hidden_states_209,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_209 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_189 = queries_94.view(getitem_269, 258, 16, 64)
        queries_94 = None
        queries_95 = view_189.transpose(1, 2)
        view_189 = None
        view_190 = keys_94.view(getitem_269, 258, 16, 64)
        keys_94 = None
        keys_95 = view_190.transpose(1, 2)
        view_190 = None
        view_191 = values_94.view(getitem_269, 258, 16, 64)
        values_94 = None
        values_95 = view_191.transpose(1, 2)
        view_191 = None
        transpose_239 = keys_95.transpose(-1, -2)
        keys_95 = None
        matmul_94 = torch.matmul(queries_95, transpose_239)
        queries_95 = transpose_239 = None
        item_120 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_scale.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_scale = (
            None
        )
        attn_weights_141 = matmul_94 * item_120
        matmul_94 = item_120 = None
        softmax_47 = torch.nn.functional.softmax(
            attn_weights_141, dim=-1, dtype=torch.float32
        )
        attn_weights_141 = None
        attn_weights_142 = softmax_47.to(torch.float32)
        softmax_47 = None
        attn_weights_143 = torch.nn.functional.dropout(
            attn_weights_142, p=0.0, training=False
        )
        attn_weights_142 = None
        attn_output_188 = torch.matmul(attn_weights_143, values_95)
        attn_weights_143 = values_95 = None
        transpose_240 = attn_output_188.transpose(1, 2)
        attn_output_188 = None
        attn_output_189 = transpose_240.contiguous()
        transpose_240 = None
        reshape_48 = attn_output_189.reshape(getitem_269, 258, 1024)
        attn_output_189 = getitem_269 = None
        attn_output_190 = reshape_48.contiguous()
        reshape_48 = None
        attn_output_191 = torch._C._nn.linear(
            attn_output_190,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_190 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_210 = hidden_states_208 + attn_output_191
        hidden_states_208 = attn_output_191 = None
        hidden_states_211 = hidden_states_210[
            (slice(None, None, None), slice(None, 257, None), slice(None, None, None))
        ]
        hidden_states_210 = None
        item_121 = (
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm2_eps.item()
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm2_eps = (
            None
        )
        hidden_states_212 = torch.nn.functional.layer_norm(
            hidden_states_211,
            (1024,),
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm2_parameters_bias_,
            item_121,
        )
        l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm2_parameters_bias_ = (item_121) = (
            None
        )
        hidden_states_213 = torch._C._nn.linear(
            hidden_states_212,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_212 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_94 = 1.702 * hidden_states_213
        sigmoid_23 = torch.sigmoid(mul_94)
        mul_94 = None
        hidden_states_214 = hidden_states_213 * sigmoid_23
        hidden_states_213 = sigmoid_23 = None
        hidden_states_215 = torch._C._nn.linear(
            hidden_states_214,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_214 = l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_216 = hidden_states_211 + hidden_states_215
        hidden_states_211 = hidden_states_215 = None
        pooled_output = hidden_states_216[
            (slice(None, None, None), 0, slice(None, None, None))
        ]
        item_122 = l_self_modules_vision_model_modules_post_layernorm_eps.item()
        l_self_modules_vision_model_modules_post_layernorm_eps = None
        pooled_output_1 = torch.nn.functional.layer_norm(
            pooled_output,
            (1024,),
            l_self_modules_vision_model_modules_post_layernorm_parameters_weight_,
            l_self_modules_vision_model_modules_post_layernorm_parameters_bias_,
            item_122,
        )
        pooled_output = (
            l_self_modules_vision_model_modules_post_layernorm_parameters_weight_
        ) = (
            l_self_modules_vision_model_modules_post_layernorm_parameters_bias_
        ) = item_122 = None
        video_embeds = torch._C._nn.linear(
            pooled_output_1, l_self_modules_visual_projection_parameters_weight_, None
        )
        l_self_modules_visual_projection_parameters_weight_ = None
        cls_features = video_embeds.view(1, getitem_1, -1)
        video_embeds = None
        hidden_states_217 = (
            cls_features + l_self_modules_mit_parameters_position_embedding_
        )
        l_self_modules_mit_parameters_position_embedding_ = None
        item_123 = (
            l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_layer_norm1_eps.item()
        )
        l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_layer_norm1_eps = (
            None
        )
        hidden_states_218 = torch.nn.functional.layer_norm(
            hidden_states_217,
            (768,),
            l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_weight_,
            l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_bias_,
            item_123,
        )
        l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_weight_ = l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_bias_ = (item_123) = (
            None
        )
        queries_96 = torch._C._nn.linear(
            hidden_states_218,
            l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_96 = torch._C._nn.linear(
            hidden_states_218,
            l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_96 = torch._C._nn.linear(
            hidden_states_218,
            l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_218 = l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_193 = queries_96.view(1, 8, 8, 96)
        queries_96 = None
        queries_97 = view_193.transpose(1, 2)
        view_193 = None
        view_194 = keys_96.view(1, 8, 8, 96)
        keys_96 = None
        keys_97 = view_194.transpose(1, 2)
        view_194 = None
        view_195 = values_96.view(1, 8, 8, 96)
        values_96 = None
        values_97 = view_195.transpose(1, 2)
        view_195 = None
        transpose_244 = keys_97.transpose(-1, -2)
        keys_97 = None
        matmul_96 = torch.matmul(queries_97, transpose_244)
        queries_97 = transpose_244 = None
        item_124 = (
            l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_scale.item()
        )
        l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_scale = (
            None
        )
        attn_weights_144 = matmul_96 * item_124
        matmul_96 = item_124 = None
        softmax_48 = torch.nn.functional.softmax(
            attn_weights_144, dim=-1, dtype=torch.float32
        )
        attn_weights_144 = None
        attn_weights_145 = softmax_48.to(torch.float32)
        softmax_48 = None
        attn_weights_146 = torch.nn.functional.dropout(
            attn_weights_145, p=0.0, training=False
        )
        attn_weights_145 = None
        attn_output_192 = torch.matmul(attn_weights_146, values_97)
        attn_weights_146 = values_97 = None
        transpose_245 = attn_output_192.transpose(1, 2)
        attn_output_192 = None
        attn_output_193 = transpose_245.contiguous()
        transpose_245 = None
        reshape_49 = attn_output_193.reshape(1, 8, 768)
        attn_output_193 = None
        attn_output_194 = reshape_49.contiguous()
        reshape_49 = None
        attn_output_195 = torch._C._nn.linear(
            attn_output_194,
            l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_194 = l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_219 = hidden_states_217 + attn_output_195
        hidden_states_217 = attn_output_195 = None
        item_125 = (
            l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_layer_norm2_eps.item()
        )
        l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_layer_norm2_eps = (
            None
        )
        hidden_states_220 = torch.nn.functional.layer_norm(
            hidden_states_219,
            (768,),
            l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_weight_,
            l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_bias_,
            item_125,
        )
        l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_weight_ = l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_bias_ = (item_125) = (
            None
        )
        hidden_states_221 = torch._C._nn.linear(
            hidden_states_220,
            l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_220 = l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_97 = 1.702 * hidden_states_221
        sigmoid_24 = torch.sigmoid(mul_97)
        mul_97 = None
        hidden_states_222 = hidden_states_221 * sigmoid_24
        hidden_states_221 = sigmoid_24 = None
        hidden_states_223 = torch._C._nn.linear(
            hidden_states_222,
            l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_222 = l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_mit_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_224 = hidden_states_219 + hidden_states_223
        hidden_states_219 = hidden_states_223 = None
        type_1 = hidden_states_224.type(torch.float32)
        hidden_states_224 = None
        last_hidden_state = type_1 + cls_features
        type_1 = cls_features = None
        pooled_output_2 = last_hidden_state.mean(dim=1, keepdim=False)
        img_features = hidden_states_216[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        item_126 = l_self_modules_prompts_visual_layernorm_eps.item()
        l_self_modules_prompts_visual_layernorm_eps = None
        img_features_1 = torch.nn.functional.layer_norm(
            img_features,
            (1024,),
            l_self_modules_prompts_visual_layernorm_parameters_weight_,
            l_self_modules_prompts_visual_layernorm_parameters_bias_,
            item_126,
        )
        img_features = (
            l_self_modules_prompts_visual_layernorm_parameters_weight_
        ) = l_self_modules_prompts_visual_layernorm_parameters_bias_ = item_126 = None
        img_features_2 = img_features_1 @ l_self_parameters_prompts_visual_projection_
        img_features_1 = l_self_parameters_prompts_visual_projection_ = None
        img_features_3 = img_features_2.view(1, getitem_1, -1, 768)
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
        item_127 = (
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
            item_127,
            False,
            False,
        )
        l_self_modules_text_model_modules_embeddings_modules_token_embedding_parameters_weight_ = (
            item_127
        ) = None
        item_128 = (
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
            item_128,
            False,
            False,
        )
        position_ids = l_self_modules_text_model_modules_embeddings_modules_position_embedding_parameters_weight_ = (item_128) = (
            None
        )
        embeddings_2 = inputs_embeds + position_embeddings
        inputs_embeds = position_embeddings = None
        mask = torch.full(
            (8, 8), -3.4028234663852886e38, device=device(type="cuda", index=0)
        )
        mask_cond = torch.arange(8, device=device(type="cuda", index=0))
        add_78 = mask_cond + 1
        view_198 = add_78.view(8, 1)
        add_78 = None
        lt = mask_cond < view_198
        mask_cond = view_198 = None
        masked_fill_ = mask.masked_fill_(lt, 0)
        lt = masked_fill_ = None
        mask_1 = mask.to(torch.float32)
        mask = None
        getitem_276 = mask_1[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        mask_1 = None
        causal_4d_mask = getitem_276.expand(1, 1, 8, 8)
        getitem_276 = None
        getitem_277 = l_attention_mask_[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        l_attention_mask_ = None
        expand_2 = getitem_277.expand(1, 1, 8, 8)
        getitem_277 = None
        expanded_mask = expand_2.to(torch.float32)
        expand_2 = None
        tensor = torch.tensor(1.0, dtype=torch.float32)
        inverted_mask = tensor - expanded_mask
        tensor = expanded_mask = None
        to_52 = inverted_mask.to(torch.bool)
        attention_mask = inverted_mask.masked_fill(to_52, -3.4028234663852886e38)
        inverted_mask = to_52 = None
        item_129 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_eps.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_eps = (
            None
        )
        hidden_states_225 = torch.nn.functional.layer_norm(
            embeddings_2,
            (768,),
            l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_bias_,
            item_129,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_bias_ = (item_129) = (
            None
        )
        queries_98 = torch._C._nn.linear(
            hidden_states_225,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_98 = torch._C._nn.linear(
            hidden_states_225,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_98 = torch._C._nn.linear(
            hidden_states_225,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_225 = l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_199 = queries_98.view(1, 8, 12, 64)
        queries_98 = None
        queries_99 = view_199.transpose(1, 2)
        view_199 = None
        view_200 = keys_98.view(1, 8, 12, 64)
        keys_98 = None
        keys_99 = view_200.transpose(1, 2)
        view_200 = None
        view_201 = values_98.view(1, 8, 12, 64)
        values_98 = None
        values_99 = view_201.transpose(1, 2)
        view_201 = None
        attention_mask_1 = attention_mask + causal_4d_mask
        transpose_249 = keys_99.transpose(-1, -2)
        keys_99 = None
        matmul_99 = torch.matmul(queries_99, transpose_249)
        queries_99 = transpose_249 = None
        item_130 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_scale.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_scale = (
            None
        )
        attn_weights_147 = matmul_99 * item_130
        matmul_99 = item_130 = None
        attn_weights_148 = attn_weights_147 + attention_mask_1
        attn_weights_147 = attention_mask_1 = None
        softmax_49 = torch.nn.functional.softmax(
            attn_weights_148, dim=-1, dtype=torch.float32
        )
        attn_weights_148 = None
        attn_weights_149 = softmax_49.to(torch.float32)
        softmax_49 = None
        attn_weights_150 = torch.nn.functional.dropout(
            attn_weights_149, p=0.0, training=False
        )
        attn_weights_149 = None
        attn_output_196 = torch.matmul(attn_weights_150, values_99)
        attn_weights_150 = values_99 = None
        transpose_250 = attn_output_196.transpose(1, 2)
        attn_output_196 = None
        attn_output_197 = transpose_250.contiguous()
        transpose_250 = None
        reshape_50 = attn_output_197.reshape(1, 8, 768)
        attn_output_197 = None
        attn_output_198 = reshape_50.contiguous()
        reshape_50 = None
        attn_output_199 = torch._C._nn.linear(
            attn_output_198,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_198 = l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_226 = embeddings_2 + attn_output_199
        embeddings_2 = attn_output_199 = None
        item_131 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_eps.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_eps = (
            None
        )
        hidden_states_227 = torch.nn.functional.layer_norm(
            hidden_states_226,
            (768,),
            l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_bias_,
            item_131,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_bias_ = (item_131) = (
            None
        )
        hidden_states_228 = torch._C._nn.linear(
            hidden_states_227,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_227 = l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_100 = 1.702 * hidden_states_228
        sigmoid_25 = torch.sigmoid(mul_100)
        mul_100 = None
        hidden_states_229 = hidden_states_228 * sigmoid_25
        hidden_states_228 = sigmoid_25 = None
        hidden_states_230 = torch._C._nn.linear(
            hidden_states_229,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_229 = l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_231 = hidden_states_226 + hidden_states_230
        hidden_states_226 = hidden_states_230 = None
        item_132 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_eps.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_eps = (
            None
        )
        hidden_states_232 = torch.nn.functional.layer_norm(
            hidden_states_231,
            (768,),
            l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_bias_,
            item_132,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_bias_ = (item_132) = (
            None
        )
        queries_100 = torch._C._nn.linear(
            hidden_states_232,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_100 = torch._C._nn.linear(
            hidden_states_232,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_100 = torch._C._nn.linear(
            hidden_states_232,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_232 = l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_202 = queries_100.view(1, 8, 12, 64)
        queries_100 = None
        queries_101 = view_202.transpose(1, 2)
        view_202 = None
        view_203 = keys_100.view(1, 8, 12, 64)
        keys_100 = None
        keys_101 = view_203.transpose(1, 2)
        view_203 = None
        view_204 = values_100.view(1, 8, 12, 64)
        values_100 = None
        values_101 = view_204.transpose(1, 2)
        view_204 = None
        attention_mask_2 = attention_mask + causal_4d_mask
        transpose_254 = keys_101.transpose(-1, -2)
        keys_101 = None
        matmul_101 = torch.matmul(queries_101, transpose_254)
        queries_101 = transpose_254 = None
        item_133 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_scale.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_scale = (
            None
        )
        attn_weights_151 = matmul_101 * item_133
        matmul_101 = item_133 = None
        attn_weights_152 = attn_weights_151 + attention_mask_2
        attn_weights_151 = attention_mask_2 = None
        softmax_50 = torch.nn.functional.softmax(
            attn_weights_152, dim=-1, dtype=torch.float32
        )
        attn_weights_152 = None
        attn_weights_153 = softmax_50.to(torch.float32)
        softmax_50 = None
        attn_weights_154 = torch.nn.functional.dropout(
            attn_weights_153, p=0.0, training=False
        )
        attn_weights_153 = None
        attn_output_200 = torch.matmul(attn_weights_154, values_101)
        attn_weights_154 = values_101 = None
        transpose_255 = attn_output_200.transpose(1, 2)
        attn_output_200 = None
        attn_output_201 = transpose_255.contiguous()
        transpose_255 = None
        reshape_51 = attn_output_201.reshape(1, 8, 768)
        attn_output_201 = None
        attn_output_202 = reshape_51.contiguous()
        reshape_51 = None
        attn_output_203 = torch._C._nn.linear(
            attn_output_202,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_202 = l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_233 = hidden_states_231 + attn_output_203
        hidden_states_231 = attn_output_203 = None
        item_134 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_eps.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_eps = (
            None
        )
        hidden_states_234 = torch.nn.functional.layer_norm(
            hidden_states_233,
            (768,),
            l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_bias_,
            item_134,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_bias_ = (item_134) = (
            None
        )
        hidden_states_235 = torch._C._nn.linear(
            hidden_states_234,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_234 = l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_103 = 1.702 * hidden_states_235
        sigmoid_26 = torch.sigmoid(mul_103)
        mul_103 = None
        hidden_states_236 = hidden_states_235 * sigmoid_26
        hidden_states_235 = sigmoid_26 = None
        hidden_states_237 = torch._C._nn.linear(
            hidden_states_236,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_236 = l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_238 = hidden_states_233 + hidden_states_237
        hidden_states_233 = hidden_states_237 = None
        item_135 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_eps.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_eps = (
            None
        )
        hidden_states_239 = torch.nn.functional.layer_norm(
            hidden_states_238,
            (768,),
            l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_bias_,
            item_135,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_bias_ = (item_135) = (
            None
        )
        queries_102 = torch._C._nn.linear(
            hidden_states_239,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_102 = torch._C._nn.linear(
            hidden_states_239,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_102 = torch._C._nn.linear(
            hidden_states_239,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_239 = l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_205 = queries_102.view(1, 8, 12, 64)
        queries_102 = None
        queries_103 = view_205.transpose(1, 2)
        view_205 = None
        view_206 = keys_102.view(1, 8, 12, 64)
        keys_102 = None
        keys_103 = view_206.transpose(1, 2)
        view_206 = None
        view_207 = values_102.view(1, 8, 12, 64)
        values_102 = None
        values_103 = view_207.transpose(1, 2)
        view_207 = None
        attention_mask_3 = attention_mask + causal_4d_mask
        transpose_259 = keys_103.transpose(-1, -2)
        keys_103 = None
        matmul_103 = torch.matmul(queries_103, transpose_259)
        queries_103 = transpose_259 = None
        item_136 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_scale.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_scale = (
            None
        )
        attn_weights_155 = matmul_103 * item_136
        matmul_103 = item_136 = None
        attn_weights_156 = attn_weights_155 + attention_mask_3
        attn_weights_155 = attention_mask_3 = None
        softmax_51 = torch.nn.functional.softmax(
            attn_weights_156, dim=-1, dtype=torch.float32
        )
        attn_weights_156 = None
        attn_weights_157 = softmax_51.to(torch.float32)
        softmax_51 = None
        attn_weights_158 = torch.nn.functional.dropout(
            attn_weights_157, p=0.0, training=False
        )
        attn_weights_157 = None
        attn_output_204 = torch.matmul(attn_weights_158, values_103)
        attn_weights_158 = values_103 = None
        transpose_260 = attn_output_204.transpose(1, 2)
        attn_output_204 = None
        attn_output_205 = transpose_260.contiguous()
        transpose_260 = None
        reshape_52 = attn_output_205.reshape(1, 8, 768)
        attn_output_205 = None
        attn_output_206 = reshape_52.contiguous()
        reshape_52 = None
        attn_output_207 = torch._C._nn.linear(
            attn_output_206,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_206 = l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_240 = hidden_states_238 + attn_output_207
        hidden_states_238 = attn_output_207 = None
        item_137 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_eps.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_eps = (
            None
        )
        hidden_states_241 = torch.nn.functional.layer_norm(
            hidden_states_240,
            (768,),
            l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_bias_,
            item_137,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_bias_ = (item_137) = (
            None
        )
        hidden_states_242 = torch._C._nn.linear(
            hidden_states_241,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_241 = l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_106 = 1.702 * hidden_states_242
        sigmoid_27 = torch.sigmoid(mul_106)
        mul_106 = None
        hidden_states_243 = hidden_states_242 * sigmoid_27
        hidden_states_242 = sigmoid_27 = None
        hidden_states_244 = torch._C._nn.linear(
            hidden_states_243,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_243 = l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_245 = hidden_states_240 + hidden_states_244
        hidden_states_240 = hidden_states_244 = None
        item_138 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_eps.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_eps = (
            None
        )
        hidden_states_246 = torch.nn.functional.layer_norm(
            hidden_states_245,
            (768,),
            l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_bias_,
            item_138,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_bias_ = (item_138) = (
            None
        )
        queries_104 = torch._C._nn.linear(
            hidden_states_246,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_104 = torch._C._nn.linear(
            hidden_states_246,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_104 = torch._C._nn.linear(
            hidden_states_246,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_246 = l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_208 = queries_104.view(1, 8, 12, 64)
        queries_104 = None
        queries_105 = view_208.transpose(1, 2)
        view_208 = None
        view_209 = keys_104.view(1, 8, 12, 64)
        keys_104 = None
        keys_105 = view_209.transpose(1, 2)
        view_209 = None
        view_210 = values_104.view(1, 8, 12, 64)
        values_104 = None
        values_105 = view_210.transpose(1, 2)
        view_210 = None
        attention_mask_4 = attention_mask + causal_4d_mask
        transpose_264 = keys_105.transpose(-1, -2)
        keys_105 = None
        matmul_105 = torch.matmul(queries_105, transpose_264)
        queries_105 = transpose_264 = None
        item_139 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_scale.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_scale = (
            None
        )
        attn_weights_159 = matmul_105 * item_139
        matmul_105 = item_139 = None
        attn_weights_160 = attn_weights_159 + attention_mask_4
        attn_weights_159 = attention_mask_4 = None
        softmax_52 = torch.nn.functional.softmax(
            attn_weights_160, dim=-1, dtype=torch.float32
        )
        attn_weights_160 = None
        attn_weights_161 = softmax_52.to(torch.float32)
        softmax_52 = None
        attn_weights_162 = torch.nn.functional.dropout(
            attn_weights_161, p=0.0, training=False
        )
        attn_weights_161 = None
        attn_output_208 = torch.matmul(attn_weights_162, values_105)
        attn_weights_162 = values_105 = None
        transpose_265 = attn_output_208.transpose(1, 2)
        attn_output_208 = None
        attn_output_209 = transpose_265.contiguous()
        transpose_265 = None
        reshape_53 = attn_output_209.reshape(1, 8, 768)
        attn_output_209 = None
        attn_output_210 = reshape_53.contiguous()
        reshape_53 = None
        attn_output_211 = torch._C._nn.linear(
            attn_output_210,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_210 = l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_247 = hidden_states_245 + attn_output_211
        hidden_states_245 = attn_output_211 = None
        item_140 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_eps.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_eps = (
            None
        )
        hidden_states_248 = torch.nn.functional.layer_norm(
            hidden_states_247,
            (768,),
            l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_bias_,
            item_140,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_bias_ = (item_140) = (
            None
        )
        hidden_states_249 = torch._C._nn.linear(
            hidden_states_248,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_248 = l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_109 = 1.702 * hidden_states_249
        sigmoid_28 = torch.sigmoid(mul_109)
        mul_109 = None
        hidden_states_250 = hidden_states_249 * sigmoid_28
        hidden_states_249 = sigmoid_28 = None
        hidden_states_251 = torch._C._nn.linear(
            hidden_states_250,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_250 = l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_252 = hidden_states_247 + hidden_states_251
        hidden_states_247 = hidden_states_251 = None
        item_141 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_eps.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_eps = (
            None
        )
        hidden_states_253 = torch.nn.functional.layer_norm(
            hidden_states_252,
            (768,),
            l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_bias_,
            item_141,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_bias_ = (item_141) = (
            None
        )
        queries_106 = torch._C._nn.linear(
            hidden_states_253,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_106 = torch._C._nn.linear(
            hidden_states_253,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_106 = torch._C._nn.linear(
            hidden_states_253,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_253 = l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_211 = queries_106.view(1, 8, 12, 64)
        queries_106 = None
        queries_107 = view_211.transpose(1, 2)
        view_211 = None
        view_212 = keys_106.view(1, 8, 12, 64)
        keys_106 = None
        keys_107 = view_212.transpose(1, 2)
        view_212 = None
        view_213 = values_106.view(1, 8, 12, 64)
        values_106 = None
        values_107 = view_213.transpose(1, 2)
        view_213 = None
        attention_mask_5 = attention_mask + causal_4d_mask
        transpose_269 = keys_107.transpose(-1, -2)
        keys_107 = None
        matmul_107 = torch.matmul(queries_107, transpose_269)
        queries_107 = transpose_269 = None
        item_142 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_scale.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_scale = (
            None
        )
        attn_weights_163 = matmul_107 * item_142
        matmul_107 = item_142 = None
        attn_weights_164 = attn_weights_163 + attention_mask_5
        attn_weights_163 = attention_mask_5 = None
        softmax_53 = torch.nn.functional.softmax(
            attn_weights_164, dim=-1, dtype=torch.float32
        )
        attn_weights_164 = None
        attn_weights_165 = softmax_53.to(torch.float32)
        softmax_53 = None
        attn_weights_166 = torch.nn.functional.dropout(
            attn_weights_165, p=0.0, training=False
        )
        attn_weights_165 = None
        attn_output_212 = torch.matmul(attn_weights_166, values_107)
        attn_weights_166 = values_107 = None
        transpose_270 = attn_output_212.transpose(1, 2)
        attn_output_212 = None
        attn_output_213 = transpose_270.contiguous()
        transpose_270 = None
        reshape_54 = attn_output_213.reshape(1, 8, 768)
        attn_output_213 = None
        attn_output_214 = reshape_54.contiguous()
        reshape_54 = None
        attn_output_215 = torch._C._nn.linear(
            attn_output_214,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_214 = l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_254 = hidden_states_252 + attn_output_215
        hidden_states_252 = attn_output_215 = None
        item_143 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_eps.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_eps = (
            None
        )
        hidden_states_255 = torch.nn.functional.layer_norm(
            hidden_states_254,
            (768,),
            l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_bias_,
            item_143,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_bias_ = (item_143) = (
            None
        )
        hidden_states_256 = torch._C._nn.linear(
            hidden_states_255,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_255 = l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_112 = 1.702 * hidden_states_256
        sigmoid_29 = torch.sigmoid(mul_112)
        mul_112 = None
        hidden_states_257 = hidden_states_256 * sigmoid_29
        hidden_states_256 = sigmoid_29 = None
        hidden_states_258 = torch._C._nn.linear(
            hidden_states_257,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_257 = l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_259 = hidden_states_254 + hidden_states_258
        hidden_states_254 = hidden_states_258 = None
        item_144 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_eps.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_eps = (
            None
        )
        hidden_states_260 = torch.nn.functional.layer_norm(
            hidden_states_259,
            (768,),
            l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_bias_,
            item_144,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_bias_ = (item_144) = (
            None
        )
        queries_108 = torch._C._nn.linear(
            hidden_states_260,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_108 = torch._C._nn.linear(
            hidden_states_260,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_108 = torch._C._nn.linear(
            hidden_states_260,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_260 = l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_214 = queries_108.view(1, 8, 12, 64)
        queries_108 = None
        queries_109 = view_214.transpose(1, 2)
        view_214 = None
        view_215 = keys_108.view(1, 8, 12, 64)
        keys_108 = None
        keys_109 = view_215.transpose(1, 2)
        view_215 = None
        view_216 = values_108.view(1, 8, 12, 64)
        values_108 = None
        values_109 = view_216.transpose(1, 2)
        view_216 = None
        attention_mask_6 = attention_mask + causal_4d_mask
        transpose_274 = keys_109.transpose(-1, -2)
        keys_109 = None
        matmul_109 = torch.matmul(queries_109, transpose_274)
        queries_109 = transpose_274 = None
        item_145 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_scale.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_scale = (
            None
        )
        attn_weights_167 = matmul_109 * item_145
        matmul_109 = item_145 = None
        attn_weights_168 = attn_weights_167 + attention_mask_6
        attn_weights_167 = attention_mask_6 = None
        softmax_54 = torch.nn.functional.softmax(
            attn_weights_168, dim=-1, dtype=torch.float32
        )
        attn_weights_168 = None
        attn_weights_169 = softmax_54.to(torch.float32)
        softmax_54 = None
        attn_weights_170 = torch.nn.functional.dropout(
            attn_weights_169, p=0.0, training=False
        )
        attn_weights_169 = None
        attn_output_216 = torch.matmul(attn_weights_170, values_109)
        attn_weights_170 = values_109 = None
        transpose_275 = attn_output_216.transpose(1, 2)
        attn_output_216 = None
        attn_output_217 = transpose_275.contiguous()
        transpose_275 = None
        reshape_55 = attn_output_217.reshape(1, 8, 768)
        attn_output_217 = None
        attn_output_218 = reshape_55.contiguous()
        reshape_55 = None
        attn_output_219 = torch._C._nn.linear(
            attn_output_218,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_218 = l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_261 = hidden_states_259 + attn_output_219
        hidden_states_259 = attn_output_219 = None
        item_146 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_eps.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_eps = (
            None
        )
        hidden_states_262 = torch.nn.functional.layer_norm(
            hidden_states_261,
            (768,),
            l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_bias_,
            item_146,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_bias_ = (item_146) = (
            None
        )
        hidden_states_263 = torch._C._nn.linear(
            hidden_states_262,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_262 = l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_115 = 1.702 * hidden_states_263
        sigmoid_30 = torch.sigmoid(mul_115)
        mul_115 = None
        hidden_states_264 = hidden_states_263 * sigmoid_30
        hidden_states_263 = sigmoid_30 = None
        hidden_states_265 = torch._C._nn.linear(
            hidden_states_264,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_264 = l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_266 = hidden_states_261 + hidden_states_265
        hidden_states_261 = hidden_states_265 = None
        item_147 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_eps.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_eps = (
            None
        )
        hidden_states_267 = torch.nn.functional.layer_norm(
            hidden_states_266,
            (768,),
            l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_bias_,
            item_147,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_bias_ = (item_147) = (
            None
        )
        queries_110 = torch._C._nn.linear(
            hidden_states_267,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_110 = torch._C._nn.linear(
            hidden_states_267,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_110 = torch._C._nn.linear(
            hidden_states_267,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_267 = l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_217 = queries_110.view(1, 8, 12, 64)
        queries_110 = None
        queries_111 = view_217.transpose(1, 2)
        view_217 = None
        view_218 = keys_110.view(1, 8, 12, 64)
        keys_110 = None
        keys_111 = view_218.transpose(1, 2)
        view_218 = None
        view_219 = values_110.view(1, 8, 12, 64)
        values_110 = None
        values_111 = view_219.transpose(1, 2)
        view_219 = None
        attention_mask_7 = attention_mask + causal_4d_mask
        transpose_279 = keys_111.transpose(-1, -2)
        keys_111 = None
        matmul_111 = torch.matmul(queries_111, transpose_279)
        queries_111 = transpose_279 = None
        item_148 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_scale.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_scale = (
            None
        )
        attn_weights_171 = matmul_111 * item_148
        matmul_111 = item_148 = None
        attn_weights_172 = attn_weights_171 + attention_mask_7
        attn_weights_171 = attention_mask_7 = None
        softmax_55 = torch.nn.functional.softmax(
            attn_weights_172, dim=-1, dtype=torch.float32
        )
        attn_weights_172 = None
        attn_weights_173 = softmax_55.to(torch.float32)
        softmax_55 = None
        attn_weights_174 = torch.nn.functional.dropout(
            attn_weights_173, p=0.0, training=False
        )
        attn_weights_173 = None
        attn_output_220 = torch.matmul(attn_weights_174, values_111)
        attn_weights_174 = values_111 = None
        transpose_280 = attn_output_220.transpose(1, 2)
        attn_output_220 = None
        attn_output_221 = transpose_280.contiguous()
        transpose_280 = None
        reshape_56 = attn_output_221.reshape(1, 8, 768)
        attn_output_221 = None
        attn_output_222 = reshape_56.contiguous()
        reshape_56 = None
        attn_output_223 = torch._C._nn.linear(
            attn_output_222,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_222 = l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_268 = hidden_states_266 + attn_output_223
        hidden_states_266 = attn_output_223 = None
        item_149 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_eps.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_eps = (
            None
        )
        hidden_states_269 = torch.nn.functional.layer_norm(
            hidden_states_268,
            (768,),
            l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_bias_,
            item_149,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_bias_ = (item_149) = (
            None
        )
        hidden_states_270 = torch._C._nn.linear(
            hidden_states_269,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_269 = l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_118 = 1.702 * hidden_states_270
        sigmoid_31 = torch.sigmoid(mul_118)
        mul_118 = None
        hidden_states_271 = hidden_states_270 * sigmoid_31
        hidden_states_270 = sigmoid_31 = None
        hidden_states_272 = torch._C._nn.linear(
            hidden_states_271,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_271 = l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_273 = hidden_states_268 + hidden_states_272
        hidden_states_268 = hidden_states_272 = None
        item_150 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_eps.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_eps = (
            None
        )
        hidden_states_274 = torch.nn.functional.layer_norm(
            hidden_states_273,
            (768,),
            l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_bias_,
            item_150,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_bias_ = (item_150) = (
            None
        )
        queries_112 = torch._C._nn.linear(
            hidden_states_274,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_112 = torch._C._nn.linear(
            hidden_states_274,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_112 = torch._C._nn.linear(
            hidden_states_274,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_274 = l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_220 = queries_112.view(1, 8, 12, 64)
        queries_112 = None
        queries_113 = view_220.transpose(1, 2)
        view_220 = None
        view_221 = keys_112.view(1, 8, 12, 64)
        keys_112 = None
        keys_113 = view_221.transpose(1, 2)
        view_221 = None
        view_222 = values_112.view(1, 8, 12, 64)
        values_112 = None
        values_113 = view_222.transpose(1, 2)
        view_222 = None
        attention_mask_8 = attention_mask + causal_4d_mask
        transpose_284 = keys_113.transpose(-1, -2)
        keys_113 = None
        matmul_113 = torch.matmul(queries_113, transpose_284)
        queries_113 = transpose_284 = None
        item_151 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_scale.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_scale = (
            None
        )
        attn_weights_175 = matmul_113 * item_151
        matmul_113 = item_151 = None
        attn_weights_176 = attn_weights_175 + attention_mask_8
        attn_weights_175 = attention_mask_8 = None
        softmax_56 = torch.nn.functional.softmax(
            attn_weights_176, dim=-1, dtype=torch.float32
        )
        attn_weights_176 = None
        attn_weights_177 = softmax_56.to(torch.float32)
        softmax_56 = None
        attn_weights_178 = torch.nn.functional.dropout(
            attn_weights_177, p=0.0, training=False
        )
        attn_weights_177 = None
        attn_output_224 = torch.matmul(attn_weights_178, values_113)
        attn_weights_178 = values_113 = None
        transpose_285 = attn_output_224.transpose(1, 2)
        attn_output_224 = None
        attn_output_225 = transpose_285.contiguous()
        transpose_285 = None
        reshape_57 = attn_output_225.reshape(1, 8, 768)
        attn_output_225 = None
        attn_output_226 = reshape_57.contiguous()
        reshape_57 = None
        attn_output_227 = torch._C._nn.linear(
            attn_output_226,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_226 = l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_275 = hidden_states_273 + attn_output_227
        hidden_states_273 = attn_output_227 = None
        item_152 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_eps.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_eps = (
            None
        )
        hidden_states_276 = torch.nn.functional.layer_norm(
            hidden_states_275,
            (768,),
            l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_bias_,
            item_152,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_bias_ = (item_152) = (
            None
        )
        hidden_states_277 = torch._C._nn.linear(
            hidden_states_276,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_276 = l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_121 = 1.702 * hidden_states_277
        sigmoid_32 = torch.sigmoid(mul_121)
        mul_121 = None
        hidden_states_278 = hidden_states_277 * sigmoid_32
        hidden_states_277 = sigmoid_32 = None
        hidden_states_279 = torch._C._nn.linear(
            hidden_states_278,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_278 = l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_280 = hidden_states_275 + hidden_states_279
        hidden_states_275 = hidden_states_279 = None
        item_153 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_eps.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_eps = (
            None
        )
        hidden_states_281 = torch.nn.functional.layer_norm(
            hidden_states_280,
            (768,),
            l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_bias_,
            item_153,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_bias_ = (item_153) = (
            None
        )
        queries_114 = torch._C._nn.linear(
            hidden_states_281,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_114 = torch._C._nn.linear(
            hidden_states_281,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_114 = torch._C._nn.linear(
            hidden_states_281,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_281 = l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_223 = queries_114.view(1, 8, 12, 64)
        queries_114 = None
        queries_115 = view_223.transpose(1, 2)
        view_223 = None
        view_224 = keys_114.view(1, 8, 12, 64)
        keys_114 = None
        keys_115 = view_224.transpose(1, 2)
        view_224 = None
        view_225 = values_114.view(1, 8, 12, 64)
        values_114 = None
        values_115 = view_225.transpose(1, 2)
        view_225 = None
        attention_mask_9 = attention_mask + causal_4d_mask
        transpose_289 = keys_115.transpose(-1, -2)
        keys_115 = None
        matmul_115 = torch.matmul(queries_115, transpose_289)
        queries_115 = transpose_289 = None
        item_154 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_scale.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_scale = (
            None
        )
        attn_weights_179 = matmul_115 * item_154
        matmul_115 = item_154 = None
        attn_weights_180 = attn_weights_179 + attention_mask_9
        attn_weights_179 = attention_mask_9 = None
        softmax_57 = torch.nn.functional.softmax(
            attn_weights_180, dim=-1, dtype=torch.float32
        )
        attn_weights_180 = None
        attn_weights_181 = softmax_57.to(torch.float32)
        softmax_57 = None
        attn_weights_182 = torch.nn.functional.dropout(
            attn_weights_181, p=0.0, training=False
        )
        attn_weights_181 = None
        attn_output_228 = torch.matmul(attn_weights_182, values_115)
        attn_weights_182 = values_115 = None
        transpose_290 = attn_output_228.transpose(1, 2)
        attn_output_228 = None
        attn_output_229 = transpose_290.contiguous()
        transpose_290 = None
        reshape_58 = attn_output_229.reshape(1, 8, 768)
        attn_output_229 = None
        attn_output_230 = reshape_58.contiguous()
        reshape_58 = None
        attn_output_231 = torch._C._nn.linear(
            attn_output_230,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_230 = l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_282 = hidden_states_280 + attn_output_231
        hidden_states_280 = attn_output_231 = None
        item_155 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_eps.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_eps = (
            None
        )
        hidden_states_283 = torch.nn.functional.layer_norm(
            hidden_states_282,
            (768,),
            l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_bias_,
            item_155,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_bias_ = (item_155) = (
            None
        )
        hidden_states_284 = torch._C._nn.linear(
            hidden_states_283,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_283 = l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_124 = 1.702 * hidden_states_284
        sigmoid_33 = torch.sigmoid(mul_124)
        mul_124 = None
        hidden_states_285 = hidden_states_284 * sigmoid_33
        hidden_states_284 = sigmoid_33 = None
        hidden_states_286 = torch._C._nn.linear(
            hidden_states_285,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_285 = l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_287 = hidden_states_282 + hidden_states_286
        hidden_states_282 = hidden_states_286 = None
        item_156 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_eps.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_eps = (
            None
        )
        hidden_states_288 = torch.nn.functional.layer_norm(
            hidden_states_287,
            (768,),
            l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_bias_,
            item_156,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_bias_ = (item_156) = (
            None
        )
        queries_116 = torch._C._nn.linear(
            hidden_states_288,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_116 = torch._C._nn.linear(
            hidden_states_288,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_116 = torch._C._nn.linear(
            hidden_states_288,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_288 = l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_226 = queries_116.view(1, 8, 12, 64)
        queries_116 = None
        queries_117 = view_226.transpose(1, 2)
        view_226 = None
        view_227 = keys_116.view(1, 8, 12, 64)
        keys_116 = None
        keys_117 = view_227.transpose(1, 2)
        view_227 = None
        view_228 = values_116.view(1, 8, 12, 64)
        values_116 = None
        values_117 = view_228.transpose(1, 2)
        view_228 = None
        attention_mask_10 = attention_mask + causal_4d_mask
        transpose_294 = keys_117.transpose(-1, -2)
        keys_117 = None
        matmul_117 = torch.matmul(queries_117, transpose_294)
        queries_117 = transpose_294 = None
        item_157 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_scale.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_scale = (
            None
        )
        attn_weights_183 = matmul_117 * item_157
        matmul_117 = item_157 = None
        attn_weights_184 = attn_weights_183 + attention_mask_10
        attn_weights_183 = attention_mask_10 = None
        softmax_58 = torch.nn.functional.softmax(
            attn_weights_184, dim=-1, dtype=torch.float32
        )
        attn_weights_184 = None
        attn_weights_185 = softmax_58.to(torch.float32)
        softmax_58 = None
        attn_weights_186 = torch.nn.functional.dropout(
            attn_weights_185, p=0.0, training=False
        )
        attn_weights_185 = None
        attn_output_232 = torch.matmul(attn_weights_186, values_117)
        attn_weights_186 = values_117 = None
        transpose_295 = attn_output_232.transpose(1, 2)
        attn_output_232 = None
        attn_output_233 = transpose_295.contiguous()
        transpose_295 = None
        reshape_59 = attn_output_233.reshape(1, 8, 768)
        attn_output_233 = None
        attn_output_234 = reshape_59.contiguous()
        reshape_59 = None
        attn_output_235 = torch._C._nn.linear(
            attn_output_234,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_234 = l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_289 = hidden_states_287 + attn_output_235
        hidden_states_287 = attn_output_235 = None
        item_158 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_eps.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_eps = (
            None
        )
        hidden_states_290 = torch.nn.functional.layer_norm(
            hidden_states_289,
            (768,),
            l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_bias_,
            item_158,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_bias_ = (item_158) = (
            None
        )
        hidden_states_291 = torch._C._nn.linear(
            hidden_states_290,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_290 = l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_127 = 1.702 * hidden_states_291
        sigmoid_34 = torch.sigmoid(mul_127)
        mul_127 = None
        hidden_states_292 = hidden_states_291 * sigmoid_34
        hidden_states_291 = sigmoid_34 = None
        hidden_states_293 = torch._C._nn.linear(
            hidden_states_292,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_292 = l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_294 = hidden_states_289 + hidden_states_293
        hidden_states_289 = hidden_states_293 = None
        item_159 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_eps.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_eps = (
            None
        )
        hidden_states_295 = torch.nn.functional.layer_norm(
            hidden_states_294,
            (768,),
            l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_bias_,
            item_159,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_bias_ = (item_159) = (
            None
        )
        queries_118 = torch._C._nn.linear(
            hidden_states_295,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_118 = torch._C._nn.linear(
            hidden_states_295,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_118 = torch._C._nn.linear(
            hidden_states_295,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_295 = l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_229 = queries_118.view(1, 8, 12, 64)
        queries_118 = None
        queries_119 = view_229.transpose(1, 2)
        view_229 = None
        view_230 = keys_118.view(1, 8, 12, 64)
        keys_118 = None
        keys_119 = view_230.transpose(1, 2)
        view_230 = None
        view_231 = values_118.view(1, 8, 12, 64)
        values_118 = None
        values_119 = view_231.transpose(1, 2)
        view_231 = None
        attention_mask_11 = attention_mask + causal_4d_mask
        transpose_299 = keys_119.transpose(-1, -2)
        keys_119 = None
        matmul_119 = torch.matmul(queries_119, transpose_299)
        queries_119 = transpose_299 = None
        item_160 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_scale.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_scale = (
            None
        )
        attn_weights_187 = matmul_119 * item_160
        matmul_119 = item_160 = None
        attn_weights_188 = attn_weights_187 + attention_mask_11
        attn_weights_187 = attention_mask_11 = None
        softmax_59 = torch.nn.functional.softmax(
            attn_weights_188, dim=-1, dtype=torch.float32
        )
        attn_weights_188 = None
        attn_weights_189 = softmax_59.to(torch.float32)
        softmax_59 = None
        attn_weights_190 = torch.nn.functional.dropout(
            attn_weights_189, p=0.0, training=False
        )
        attn_weights_189 = None
        attn_output_236 = torch.matmul(attn_weights_190, values_119)
        attn_weights_190 = values_119 = None
        transpose_300 = attn_output_236.transpose(1, 2)
        attn_output_236 = None
        attn_output_237 = transpose_300.contiguous()
        transpose_300 = None
        reshape_60 = attn_output_237.reshape(1, 8, 768)
        attn_output_237 = None
        attn_output_238 = reshape_60.contiguous()
        reshape_60 = None
        attn_output_239 = torch._C._nn.linear(
            attn_output_238,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_238 = l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_296 = hidden_states_294 + attn_output_239
        hidden_states_294 = attn_output_239 = None
        item_161 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_eps.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_eps = (
            None
        )
        hidden_states_297 = torch.nn.functional.layer_norm(
            hidden_states_296,
            (768,),
            l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_bias_,
            item_161,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_bias_ = (item_161) = (
            None
        )
        hidden_states_298 = torch._C._nn.linear(
            hidden_states_297,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_297 = l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_130 = 1.702 * hidden_states_298
        sigmoid_35 = torch.sigmoid(mul_130)
        mul_130 = None
        hidden_states_299 = hidden_states_298 * sigmoid_35
        hidden_states_298 = sigmoid_35 = None
        hidden_states_300 = torch._C._nn.linear(
            hidden_states_299,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_299 = l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_301 = hidden_states_296 + hidden_states_300
        hidden_states_296 = hidden_states_300 = None
        item_162 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_eps.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_eps = (
            None
        )
        hidden_states_302 = torch.nn.functional.layer_norm(
            hidden_states_301,
            (768,),
            l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_bias_,
            item_162,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_bias_ = (item_162) = (
            None
        )
        queries_120 = torch._C._nn.linear(
            hidden_states_302,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        keys_120 = torch._C._nn.linear(
            hidden_states_302,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        values_120 = torch._C._nn.linear(
            hidden_states_302,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_302 = l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_232 = queries_120.view(1, 8, 12, 64)
        queries_120 = None
        queries_121 = view_232.transpose(1, 2)
        view_232 = None
        view_233 = keys_120.view(1, 8, 12, 64)
        keys_120 = None
        keys_121 = view_233.transpose(1, 2)
        view_233 = None
        view_234 = values_120.view(1, 8, 12, 64)
        values_120 = None
        values_121 = view_234.transpose(1, 2)
        view_234 = None
        attention_mask_12 = attention_mask + causal_4d_mask
        attention_mask = causal_4d_mask = None
        transpose_304 = keys_121.transpose(-1, -2)
        keys_121 = None
        matmul_121 = torch.matmul(queries_121, transpose_304)
        queries_121 = transpose_304 = None
        item_163 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_scale.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_scale = (
            None
        )
        attn_weights_191 = matmul_121 * item_163
        matmul_121 = item_163 = None
        attn_weights_192 = attn_weights_191 + attention_mask_12
        attn_weights_191 = attention_mask_12 = None
        softmax_60 = torch.nn.functional.softmax(
            attn_weights_192, dim=-1, dtype=torch.float32
        )
        attn_weights_192 = None
        attn_weights_193 = softmax_60.to(torch.float32)
        softmax_60 = None
        attn_weights_194 = torch.nn.functional.dropout(
            attn_weights_193, p=0.0, training=False
        )
        attn_weights_193 = None
        attn_output_240 = torch.matmul(attn_weights_194, values_121)
        attn_weights_194 = values_121 = None
        transpose_305 = attn_output_240.transpose(1, 2)
        attn_output_240 = None
        attn_output_241 = transpose_305.contiguous()
        transpose_305 = None
        reshape_61 = attn_output_241.reshape(1, 8, 768)
        attn_output_241 = None
        attn_output_242 = reshape_61.contiguous()
        reshape_61 = None
        attn_output_243 = torch._C._nn.linear(
            attn_output_242,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_242 = l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_303 = hidden_states_301 + attn_output_243
        hidden_states_301 = attn_output_243 = None
        item_164 = (
            l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_eps.item()
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_eps = (
            None
        )
        hidden_states_304 = torch.nn.functional.layer_norm(
            hidden_states_303,
            (768,),
            l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_bias_,
            item_164,
        )
        l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_bias_ = (item_164) = (
            None
        )
        hidden_states_305 = torch._C._nn.linear(
            hidden_states_304,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_bias_,
        )
        hidden_states_304 = l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_bias_ = (None)
        mul_133 = 1.702 * hidden_states_305
        sigmoid_36 = torch.sigmoid(mul_133)
        mul_133 = None
        hidden_states_306 = hidden_states_305 * sigmoid_36
        hidden_states_305 = sigmoid_36 = None
        hidden_states_307 = torch._C._nn.linear(
            hidden_states_306,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_bias_,
        )
        hidden_states_306 = l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_bias_ = (None)
        hidden_states_308 = hidden_states_303 + hidden_states_307
        hidden_states_303 = hidden_states_307 = None
        item_165 = l_self_modules_text_model_modules_final_layer_norm_eps.item()
        l_self_modules_text_model_modules_final_layer_norm_eps = None
        last_hidden_state_1 = torch.nn.functional.layer_norm(
            hidden_states_308,
            (768,),
            l_self_modules_text_model_modules_final_layer_norm_parameters_weight_,
            l_self_modules_text_model_modules_final_layer_norm_parameters_bias_,
            item_165,
        )
        hidden_states_308 = (
            l_self_modules_text_model_modules_final_layer_norm_parameters_weight_
        ) = (
            l_self_modules_text_model_modules_final_layer_norm_parameters_bias_
        ) = item_165 = None
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
        item_166 = l_self_modules_prompts_generator_modules_layernorm_eps.item()
        l_self_modules_prompts_generator_modules_layernorm_eps = None
        visual = torch.nn.functional.layer_norm(
            img_features_4,
            (768,),
            l_self_modules_prompts_generator_modules_layernorm_parameters_weight_,
            l_self_modules_prompts_generator_modules_layernorm_parameters_bias_,
            item_166,
        )
        img_features_4 = (
            l_self_modules_prompts_generator_modules_layernorm_parameters_weight_
        ) = (
            l_self_modules_prompts_generator_modules_layernorm_parameters_bias_
        ) = item_166 = None
        item_167 = (
            l_self_modules_prompts_generator_modules_decoder_modules_0_modules_norm1_eps.item()
        )
        l_self_modules_prompts_generator_modules_decoder_modules_0_modules_norm1_eps = (
            None
        )
        layer_norm_103 = torch.nn.functional.layer_norm(
            text_embeds_1,
            (768,),
            l_self_modules_prompts_generator_modules_decoder_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_prompts_generator_modules_decoder_modules_0_modules_norm1_parameters_bias_,
            item_167,
        )
        l_self_modules_prompts_generator_modules_decoder_modules_0_modules_norm1_parameters_weight_ = l_self_modules_prompts_generator_modules_decoder_modules_0_modules_norm1_parameters_bias_ = (item_167) = (
            None
        )
        linear_344 = torch._C._nn.linear(
            layer_norm_103,
            l_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_q_proj_parameters_weight_,
            None,
        )
        layer_norm_103 = l_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_q_proj_parameters_weight_ = (None)
        reshape_62 = linear_344.reshape(1, 1, 8, 96)
        linear_344 = None
        queries_122 = reshape_62.permute(0, 2, 1, 3)
        reshape_62 = None
        linear_345 = torch._C._nn.linear(
            visual,
            l_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        reshape_63 = linear_345.reshape(1, 256, 8, 96)
        linear_345 = None
        keys_122 = reshape_63.permute(0, 2, 1, 3)
        reshape_63 = None
        linear_346 = torch._C._nn.linear(
            visual,
            l_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_v_proj_parameters_weight_ = (
            None
        )
        reshape_64 = linear_346.reshape(1, 256, 8, 96)
        linear_346 = None
        values_122 = reshape_64.permute(0, 2, 1, 3)
        reshape_64 = None
        transpose_306 = keys_122.transpose(-2, -1)
        keys_122 = None
        matmul_123 = queries_122 @ transpose_306
        queries_122 = transpose_306 = None
        item_168 = (
            l_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_scale.item()
        )
        l_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_scale = (
            None
        )
        attn = matmul_123 * item_168
        matmul_123 = item_168 = None
        attn_1 = attn.softmax(dim=-1)
        attn = None
        item_169 = (
            l_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_attn_drop_p.item()
        )
        l_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_attn_drop_p = (
            None
        )
        attn_2 = torch.nn.functional.dropout(attn_1, item_169, False, False)
        attn_1 = item_169 = None
        matmul_124 = attn_2 @ values_122
        attn_2 = values_122 = None
        transpose_307 = matmul_124.transpose(1, 2)
        matmul_124 = None
        x = transpose_307.reshape(1, 1, 768)
        transpose_307 = None
        x_1 = torch._C._nn.linear(
            x,
            l_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_proj_parameters_weight_,
            l_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_proj_parameters_bias_,
        )
        x = l_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_proj_parameters_weight_ = l_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_proj_parameters_bias_ = (None)
        item_170 = (
            l_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_proj_drop_p.item()
        )
        l_self_modules_prompts_generator_modules_decoder_modules_0_modules_cross_attn_modules_proj_drop_p = (
            None
        )
        x_2 = torch.nn.functional.dropout(x_1, item_170, False, False)
        x_1 = item_170 = None
        x_3 = text_embeds_1 + x_2
        x_2 = None
        item_171 = (
            l_self_modules_prompts_generator_modules_decoder_modules_0_modules_norm3_eps.item()
        )
        l_self_modules_prompts_generator_modules_decoder_modules_0_modules_norm3_eps = (
            None
        )
        layer_norm_104 = torch.nn.functional.layer_norm(
            x_3,
            (768,),
            l_self_modules_prompts_generator_modules_decoder_modules_0_modules_norm3_parameters_weight_,
            l_self_modules_prompts_generator_modules_decoder_modules_0_modules_norm3_parameters_bias_,
            item_171,
        )
        l_self_modules_prompts_generator_modules_decoder_modules_0_modules_norm3_parameters_weight_ = l_self_modules_prompts_generator_modules_decoder_modules_0_modules_norm3_parameters_bias_ = (item_171) = (
            None
        )
        input_1 = torch._C._nn.linear(
            layer_norm_104,
            l_self_modules_prompts_generator_modules_decoder_modules_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_prompts_generator_modules_decoder_modules_0_modules_mlp_modules_0_parameters_bias_,
        )
        layer_norm_104 = l_self_modules_prompts_generator_modules_decoder_modules_0_modules_mlp_modules_0_parameters_weight_ = l_self_modules_prompts_generator_modules_decoder_modules_0_modules_mlp_modules_0_parameters_bias_ = (None)
        mul_136 = 1.702 * input_1
        sigmoid_37 = torch.sigmoid(mul_136)
        mul_136 = None
        input_2 = input_1 * sigmoid_37
        input_1 = sigmoid_37 = None
        item_172 = (
            l_self_modules_prompts_generator_modules_decoder_modules_0_modules_mlp_modules_2_p.item()
        )
        l_self_modules_prompts_generator_modules_decoder_modules_0_modules_mlp_modules_2_p = (
            None
        )
        input_3 = torch.nn.functional.dropout(input_2, item_172, False, False)
        input_2 = item_172 = None
        input_4 = torch._C._nn.linear(
            input_3,
            l_self_modules_prompts_generator_modules_decoder_modules_0_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_prompts_generator_modules_decoder_modules_0_modules_mlp_modules_3_parameters_bias_,
        )
        input_3 = l_self_modules_prompts_generator_modules_decoder_modules_0_modules_mlp_modules_3_parameters_weight_ = l_self_modules_prompts_generator_modules_decoder_modules_0_modules_mlp_modules_3_parameters_bias_ = (None)
        x_4 = x_3 + input_4
        x_3 = input_4 = None
        item_173 = (
            l_self_modules_prompts_generator_modules_decoder_modules_1_modules_norm1_eps.item()
        )
        l_self_modules_prompts_generator_modules_decoder_modules_1_modules_norm1_eps = (
            None
        )
        layer_norm_105 = torch.nn.functional.layer_norm(
            x_4,
            (768,),
            l_self_modules_prompts_generator_modules_decoder_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_prompts_generator_modules_decoder_modules_1_modules_norm1_parameters_bias_,
            item_173,
        )
        l_self_modules_prompts_generator_modules_decoder_modules_1_modules_norm1_parameters_weight_ = l_self_modules_prompts_generator_modules_decoder_modules_1_modules_norm1_parameters_bias_ = (item_173) = (
            None
        )
        linear_350 = torch._C._nn.linear(
            layer_norm_105,
            l_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_q_proj_parameters_weight_,
            None,
        )
        layer_norm_105 = l_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_q_proj_parameters_weight_ = (None)
        reshape_66 = linear_350.reshape(1, 1, 8, 96)
        linear_350 = None
        queries_123 = reshape_66.permute(0, 2, 1, 3)
        reshape_66 = None
        linear_351 = torch._C._nn.linear(
            visual,
            l_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        reshape_67 = linear_351.reshape(1, 256, 8, 96)
        linear_351 = None
        keys_123 = reshape_67.permute(0, 2, 1, 3)
        reshape_67 = None
        linear_352 = torch._C._nn.linear(
            visual,
            l_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_v_proj_parameters_weight_,
            None,
        )
        visual = l_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_v_proj_parameters_weight_ = (None)
        reshape_68 = linear_352.reshape(1, 256, 8, 96)
        linear_352 = None
        values_123 = reshape_68.permute(0, 2, 1, 3)
        reshape_68 = None
        transpose_308 = keys_123.transpose(-2, -1)
        keys_123 = None
        matmul_125 = queries_123 @ transpose_308
        queries_123 = transpose_308 = None
        item_174 = (
            l_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_scale.item()
        )
        l_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_scale = (
            None
        )
        attn_3 = matmul_125 * item_174
        matmul_125 = item_174 = None
        attn_4 = attn_3.softmax(dim=-1)
        attn_3 = None
        item_175 = (
            l_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_attn_drop_p.item()
        )
        l_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_attn_drop_p = (
            None
        )
        attn_5 = torch.nn.functional.dropout(attn_4, item_175, False, False)
        attn_4 = item_175 = None
        matmul_126 = attn_5 @ values_123
        attn_5 = values_123 = None
        transpose_309 = matmul_126.transpose(1, 2)
        matmul_126 = None
        x_5 = transpose_309.reshape(1, 1, 768)
        transpose_309 = None
        x_6 = torch._C._nn.linear(
            x_5,
            l_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_proj_parameters_weight_,
            l_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_proj_parameters_bias_,
        )
        x_5 = l_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_proj_parameters_weight_ = l_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_proj_parameters_bias_ = (None)
        item_176 = (
            l_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_proj_drop_p.item()
        )
        l_self_modules_prompts_generator_modules_decoder_modules_1_modules_cross_attn_modules_proj_drop_p = (
            None
        )
        x_7 = torch.nn.functional.dropout(x_6, item_176, False, False)
        x_6 = item_176 = None
        x_8 = x_4 + x_7
        x_4 = x_7 = None
        item_177 = (
            l_self_modules_prompts_generator_modules_decoder_modules_1_modules_norm3_eps.item()
        )
        l_self_modules_prompts_generator_modules_decoder_modules_1_modules_norm3_eps = (
            None
        )
        layer_norm_106 = torch.nn.functional.layer_norm(
            x_8,
            (768,),
            l_self_modules_prompts_generator_modules_decoder_modules_1_modules_norm3_parameters_weight_,
            l_self_modules_prompts_generator_modules_decoder_modules_1_modules_norm3_parameters_bias_,
            item_177,
        )
        l_self_modules_prompts_generator_modules_decoder_modules_1_modules_norm3_parameters_weight_ = l_self_modules_prompts_generator_modules_decoder_modules_1_modules_norm3_parameters_bias_ = (item_177) = (
            None
        )
        input_5 = torch._C._nn.linear(
            layer_norm_106,
            l_self_modules_prompts_generator_modules_decoder_modules_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_prompts_generator_modules_decoder_modules_1_modules_mlp_modules_0_parameters_bias_,
        )
        layer_norm_106 = l_self_modules_prompts_generator_modules_decoder_modules_1_modules_mlp_modules_0_parameters_weight_ = l_self_modules_prompts_generator_modules_decoder_modules_1_modules_mlp_modules_0_parameters_bias_ = (None)
        mul_139 = 1.702 * input_5
        sigmoid_38 = torch.sigmoid(mul_139)
        mul_139 = None
        input_6 = input_5 * sigmoid_38
        input_5 = sigmoid_38 = None
        item_178 = (
            l_self_modules_prompts_generator_modules_decoder_modules_1_modules_mlp_modules_2_p.item()
        )
        l_self_modules_prompts_generator_modules_decoder_modules_1_modules_mlp_modules_2_p = (
            None
        )
        input_7 = torch.nn.functional.dropout(input_6, item_178, False, False)
        input_6 = item_178 = None
        input_8 = torch._C._nn.linear(
            input_7,
            l_self_modules_prompts_generator_modules_decoder_modules_1_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_prompts_generator_modules_decoder_modules_1_modules_mlp_modules_3_parameters_bias_,
        )
        input_7 = l_self_modules_prompts_generator_modules_decoder_modules_1_modules_mlp_modules_3_parameters_weight_ = l_self_modules_prompts_generator_modules_decoder_modules_1_modules_mlp_modules_3_parameters_bias_ = (None)
        x_9 = x_8 + input_8
        x_8 = input_8 = None
        mul_141 = l_self_modules_prompts_generator_parameters_alpha_ * x_9
        l_self_modules_prompts_generator_parameters_alpha_ = x_9 = None
        text_embeds_2 = text_embeds_1 + mul_141
        text_embeds_1 = mul_141 = None
        norm = pooled_output_2.norm(p=2, dim=-1, keepdim=True)
        video_embeds_1 = pooled_output_2 / norm
        norm = None
        norm_1 = text_embeds_2.norm(p=2, dim=-1, keepdim=True)
        text_embeds_3 = text_embeds_2 / norm_1
        text_embeds_2 = norm_1 = None
        logit_scale = l_self_parameters_logit_scale_.exp()
        l_self_parameters_logit_scale_ = None
        mul_142 = logit_scale * text_embeds_3
        logit_scale = None
        logits_per_video = torch.functional.einsum(
            "bd,bkd->bk", video_embeds_1, mul_142
        )
        mul_142 = None
        logits_per_text = logits_per_video.T
        return (
            hidden_states_216,
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
