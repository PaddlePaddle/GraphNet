import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_input_ids_: torch.Tensor,
        L_self_modules_model_modules_encoder_modules_embed_tokens_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_embed_positions_parameters_weight_: torch.nn.parameter.Parameter,
        L_attention_mask_: torch.Tensor,
        L_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_0_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_0_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_0_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_0_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_1_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_1_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_1_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_1_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_3_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_3_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_3_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_3_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_4_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_4_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_4_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_4_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_5_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_5_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_5_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_5_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_encoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_decoder_input_ids_: torch.Tensor,
        L_self_modules_model_modules_decoder_modules_embed_positions_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_buffers_final_logits_bias_: torch.Tensor,
    ):
        l_input_ids_ = L_input_ids_
        l_self_modules_model_modules_encoder_modules_embed_tokens_parameters_weight_ = (
            L_self_modules_model_modules_encoder_modules_embed_tokens_parameters_weight_
        )
        l_self_modules_model_modules_encoder_modules_embed_positions_parameters_weight_ = L_self_modules_model_modules_encoder_modules_embed_positions_parameters_weight_
        l_attention_mask_ = L_attention_mask_
        l_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_encoder_modules_layers_modules_0_modules_fc1_parameters_weight_ = L_self_modules_model_modules_encoder_modules_layers_modules_0_modules_fc1_parameters_weight_
        l_self_modules_model_modules_encoder_modules_layers_modules_0_modules_fc1_parameters_bias_ = L_self_modules_model_modules_encoder_modules_layers_modules_0_modules_fc1_parameters_bias_
        l_self_modules_model_modules_encoder_modules_layers_modules_0_modules_fc2_parameters_weight_ = L_self_modules_model_modules_encoder_modules_layers_modules_0_modules_fc2_parameters_weight_
        l_self_modules_model_modules_encoder_modules_layers_modules_0_modules_fc2_parameters_bias_ = L_self_modules_model_modules_encoder_modules_layers_modules_0_modules_fc2_parameters_bias_
        l_self_modules_model_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_encoder_modules_layers_modules_1_modules_fc1_parameters_weight_ = L_self_modules_model_modules_encoder_modules_layers_modules_1_modules_fc1_parameters_weight_
        l_self_modules_model_modules_encoder_modules_layers_modules_1_modules_fc1_parameters_bias_ = L_self_modules_model_modules_encoder_modules_layers_modules_1_modules_fc1_parameters_bias_
        l_self_modules_model_modules_encoder_modules_layers_modules_1_modules_fc2_parameters_weight_ = L_self_modules_model_modules_encoder_modules_layers_modules_1_modules_fc2_parameters_weight_
        l_self_modules_model_modules_encoder_modules_layers_modules_1_modules_fc2_parameters_bias_ = L_self_modules_model_modules_encoder_modules_layers_modules_1_modules_fc2_parameters_bias_
        l_self_modules_model_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_encoder_modules_layers_modules_2_modules_fc1_parameters_weight_ = L_self_modules_model_modules_encoder_modules_layers_modules_2_modules_fc1_parameters_weight_
        l_self_modules_model_modules_encoder_modules_layers_modules_2_modules_fc1_parameters_bias_ = L_self_modules_model_modules_encoder_modules_layers_modules_2_modules_fc1_parameters_bias_
        l_self_modules_model_modules_encoder_modules_layers_modules_2_modules_fc2_parameters_weight_ = L_self_modules_model_modules_encoder_modules_layers_modules_2_modules_fc2_parameters_weight_
        l_self_modules_model_modules_encoder_modules_layers_modules_2_modules_fc2_parameters_bias_ = L_self_modules_model_modules_encoder_modules_layers_modules_2_modules_fc2_parameters_bias_
        l_self_modules_model_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_encoder_modules_layers_modules_3_modules_fc1_parameters_weight_ = L_self_modules_model_modules_encoder_modules_layers_modules_3_modules_fc1_parameters_weight_
        l_self_modules_model_modules_encoder_modules_layers_modules_3_modules_fc1_parameters_bias_ = L_self_modules_model_modules_encoder_modules_layers_modules_3_modules_fc1_parameters_bias_
        l_self_modules_model_modules_encoder_modules_layers_modules_3_modules_fc2_parameters_weight_ = L_self_modules_model_modules_encoder_modules_layers_modules_3_modules_fc2_parameters_weight_
        l_self_modules_model_modules_encoder_modules_layers_modules_3_modules_fc2_parameters_bias_ = L_self_modules_model_modules_encoder_modules_layers_modules_3_modules_fc2_parameters_bias_
        l_self_modules_model_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_encoder_modules_layers_modules_4_modules_fc1_parameters_weight_ = L_self_modules_model_modules_encoder_modules_layers_modules_4_modules_fc1_parameters_weight_
        l_self_modules_model_modules_encoder_modules_layers_modules_4_modules_fc1_parameters_bias_ = L_self_modules_model_modules_encoder_modules_layers_modules_4_modules_fc1_parameters_bias_
        l_self_modules_model_modules_encoder_modules_layers_modules_4_modules_fc2_parameters_weight_ = L_self_modules_model_modules_encoder_modules_layers_modules_4_modules_fc2_parameters_weight_
        l_self_modules_model_modules_encoder_modules_layers_modules_4_modules_fc2_parameters_bias_ = L_self_modules_model_modules_encoder_modules_layers_modules_4_modules_fc2_parameters_bias_
        l_self_modules_model_modules_encoder_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_encoder_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_encoder_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_encoder_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_encoder_modules_layers_modules_5_modules_fc1_parameters_weight_ = L_self_modules_model_modules_encoder_modules_layers_modules_5_modules_fc1_parameters_weight_
        l_self_modules_model_modules_encoder_modules_layers_modules_5_modules_fc1_parameters_bias_ = L_self_modules_model_modules_encoder_modules_layers_modules_5_modules_fc1_parameters_bias_
        l_self_modules_model_modules_encoder_modules_layers_modules_5_modules_fc2_parameters_weight_ = L_self_modules_model_modules_encoder_modules_layers_modules_5_modules_fc2_parameters_weight_
        l_self_modules_model_modules_encoder_modules_layers_modules_5_modules_fc2_parameters_bias_ = L_self_modules_model_modules_encoder_modules_layers_modules_5_modules_fc2_parameters_bias_
        l_self_modules_model_modules_encoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_encoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_encoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_encoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_
        l_decoder_input_ids_ = L_decoder_input_ids_
        l_self_modules_model_modules_decoder_modules_embed_positions_parameters_weight_ = L_self_modules_model_modules_decoder_modules_embed_positions_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_
        l_self_buffers_final_logits_bias_ = L_self_buffers_final_logits_bias_
        input_ids = l_input_ids_.view(-1, 21)
        l_input_ids_ = None
        embedding = torch.nn.functional.embedding(
            input_ids,
            l_self_modules_model_modules_encoder_modules_embed_tokens_parameters_weight_,
            52996,
            None,
            2.0,
            False,
            False,
        )
        input_ids = None
        inputs_embeds = embedding * 22.627416997969522
        embedding = None
        position_ids = torch.arange(0, 21, dtype=torch.int64, device=device(type="cpu"))
        embed_pos = torch.nn.functional.embedding(
            position_ids,
            l_self_modules_model_modules_encoder_modules_embed_positions_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        position_ids = l_self_modules_model_modules_encoder_modules_embed_positions_parameters_weight_ = (None)
        hidden_states = inputs_embeds + embed_pos
        inputs_embeds = embed_pos = None
        hidden_states_1 = torch.nn.functional.dropout(
            hidden_states, p=0.1, training=False
        )
        hidden_states = None
        getitem = l_attention_mask_[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        expand = getitem.expand(1, 1, 21, 21)
        getitem = None
        expanded_mask = expand.to(torch.float32)
        expand = None
        tensor = torch.tensor(1.0, dtype=torch.float32)
        inverted_mask = tensor - expanded_mask
        tensor = expanded_mask = None
        to_1 = inverted_mask.to(torch.bool)
        attention_mask = inverted_mask.masked_fill(to_1, -3.4028234663852886e38)
        inverted_mask = to_1 = None
        linear = torch._C._nn.linear(
            hidden_states_1,
            l_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_1 = linear.view(1, 21, -1, 64)
        linear = None
        query_states = view_1.transpose(1, 2)
        view_1 = None
        key_states = torch._C._nn.linear(
            hidden_states_1,
            l_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states = torch._C._nn.linear(
            hidden_states_1,
            l_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_2 = key_states.view(1, 21, -1, 64)
        key_states = None
        key_states_1 = view_2.transpose(1, 2)
        view_2 = None
        view_3 = value_states.view(1, 21, -1, 64)
        value_states = None
        value_states_1 = view_3.transpose(1, 2)
        view_3 = None
        attention_mask_1 = attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 21, None),
            )
        ]
        query = query_states.contiguous()
        query_states = None
        key = key_states_1.contiguous()
        key_states_1 = None
        value = value_states_1.contiguous()
        value_states_1 = None
        attn_output = torch._C._nn.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask_1,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query = key = value = attention_mask_1 = None
        transpose_3 = attn_output.transpose(1, 2)
        attn_output = None
        attn_output_1 = transpose_3.contiguous()
        transpose_3 = None
        reshape = attn_output_1.reshape(1, 21, -1)
        attn_output_1 = None
        attn_output_2 = reshape.contiguous()
        reshape = None
        attn_output_3 = torch._C._nn.linear(
            attn_output_2,
            l_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_2 = l_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_2 = torch.nn.functional.dropout(
            attn_output_3, p=0.1, training=False
        )
        attn_output_3 = None
        hidden_states_3 = hidden_states_1 + hidden_states_2
        hidden_states_1 = hidden_states_2 = None
        hidden_states_4 = torch.nn.functional.layer_norm(
            hidden_states_3,
            (512,),
            l_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_3 = l_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_4 = torch._C._nn.linear(
            hidden_states_4,
            l_self_modules_model_modules_encoder_modules_layers_modules_0_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_encoder_modules_layers_modules_0_modules_fc1_parameters_bias_,
        )
        l_self_modules_model_modules_encoder_modules_layers_modules_0_modules_fc1_parameters_weight_ = l_self_modules_model_modules_encoder_modules_layers_modules_0_modules_fc1_parameters_bias_ = (None)
        hidden_states_5 = torch.nn.functional.silu(linear_4, inplace=False)
        linear_4 = None
        hidden_states_6 = torch.nn.functional.dropout(
            hidden_states_5, p=0.0, training=False
        )
        hidden_states_5 = None
        hidden_states_7 = torch._C._nn.linear(
            hidden_states_6,
            l_self_modules_model_modules_encoder_modules_layers_modules_0_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_encoder_modules_layers_modules_0_modules_fc2_parameters_bias_,
        )
        hidden_states_6 = l_self_modules_model_modules_encoder_modules_layers_modules_0_modules_fc2_parameters_weight_ = l_self_modules_model_modules_encoder_modules_layers_modules_0_modules_fc2_parameters_bias_ = (None)
        hidden_states_8 = torch.nn.functional.dropout(
            hidden_states_7, p=0.1, training=False
        )
        hidden_states_7 = None
        hidden_states_9 = hidden_states_4 + hidden_states_8
        hidden_states_4 = hidden_states_8 = None
        hidden_states_10 = torch.nn.functional.layer_norm(
            hidden_states_9,
            (512,),
            l_self_modules_model_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_9 = l_self_modules_model_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_ = (None)
        linear_6 = torch._C._nn.linear(
            hidden_states_10,
            l_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_4 = linear_6.view(1, 21, -1, 64)
        linear_6 = None
        query_states_1 = view_4.transpose(1, 2)
        view_4 = None
        key_states_2 = torch._C._nn.linear(
            hidden_states_10,
            l_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_2 = torch._C._nn.linear(
            hidden_states_10,
            l_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_5 = key_states_2.view(1, 21, -1, 64)
        key_states_2 = None
        key_states_3 = view_5.transpose(1, 2)
        view_5 = None
        view_6 = value_states_2.view(1, 21, -1, 64)
        value_states_2 = None
        value_states_3 = view_6.transpose(1, 2)
        view_6 = None
        attention_mask_2 = attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 21, None),
            )
        ]
        query_1 = query_states_1.contiguous()
        query_states_1 = None
        key_1 = key_states_3.contiguous()
        key_states_3 = None
        value_1 = value_states_3.contiguous()
        value_states_3 = None
        attn_output_4 = torch._C._nn.scaled_dot_product_attention(
            query_1,
            key_1,
            value_1,
            attn_mask=attention_mask_2,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_1 = key_1 = value_1 = attention_mask_2 = None
        transpose_7 = attn_output_4.transpose(1, 2)
        attn_output_4 = None
        attn_output_5 = transpose_7.contiguous()
        transpose_7 = None
        reshape_1 = attn_output_5.reshape(1, 21, -1)
        attn_output_5 = None
        attn_output_6 = reshape_1.contiguous()
        reshape_1 = None
        attn_output_7 = torch._C._nn.linear(
            attn_output_6,
            l_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_6 = l_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_11 = torch.nn.functional.dropout(
            attn_output_7, p=0.1, training=False
        )
        attn_output_7 = None
        hidden_states_12 = hidden_states_10 + hidden_states_11
        hidden_states_10 = hidden_states_11 = None
        hidden_states_13 = torch.nn.functional.layer_norm(
            hidden_states_12,
            (512,),
            l_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_12 = l_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_10 = torch._C._nn.linear(
            hidden_states_13,
            l_self_modules_model_modules_encoder_modules_layers_modules_1_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_encoder_modules_layers_modules_1_modules_fc1_parameters_bias_,
        )
        l_self_modules_model_modules_encoder_modules_layers_modules_1_modules_fc1_parameters_weight_ = l_self_modules_model_modules_encoder_modules_layers_modules_1_modules_fc1_parameters_bias_ = (None)
        hidden_states_14 = torch.nn.functional.silu(linear_10, inplace=False)
        linear_10 = None
        hidden_states_15 = torch.nn.functional.dropout(
            hidden_states_14, p=0.0, training=False
        )
        hidden_states_14 = None
        hidden_states_16 = torch._C._nn.linear(
            hidden_states_15,
            l_self_modules_model_modules_encoder_modules_layers_modules_1_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_encoder_modules_layers_modules_1_modules_fc2_parameters_bias_,
        )
        hidden_states_15 = l_self_modules_model_modules_encoder_modules_layers_modules_1_modules_fc2_parameters_weight_ = l_self_modules_model_modules_encoder_modules_layers_modules_1_modules_fc2_parameters_bias_ = (None)
        hidden_states_17 = torch.nn.functional.dropout(
            hidden_states_16, p=0.1, training=False
        )
        hidden_states_16 = None
        hidden_states_18 = hidden_states_13 + hidden_states_17
        hidden_states_13 = hidden_states_17 = None
        hidden_states_19 = torch.nn.functional.layer_norm(
            hidden_states_18,
            (512,),
            l_self_modules_model_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_18 = l_self_modules_model_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_ = (None)
        linear_12 = torch._C._nn.linear(
            hidden_states_19,
            l_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_7 = linear_12.view(1, 21, -1, 64)
        linear_12 = None
        query_states_2 = view_7.transpose(1, 2)
        view_7 = None
        key_states_4 = torch._C._nn.linear(
            hidden_states_19,
            l_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_4 = torch._C._nn.linear(
            hidden_states_19,
            l_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_8 = key_states_4.view(1, 21, -1, 64)
        key_states_4 = None
        key_states_5 = view_8.transpose(1, 2)
        view_8 = None
        view_9 = value_states_4.view(1, 21, -1, 64)
        value_states_4 = None
        value_states_5 = view_9.transpose(1, 2)
        view_9 = None
        attention_mask_3 = attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 21, None),
            )
        ]
        query_2 = query_states_2.contiguous()
        query_states_2 = None
        key_2 = key_states_5.contiguous()
        key_states_5 = None
        value_2 = value_states_5.contiguous()
        value_states_5 = None
        attn_output_8 = torch._C._nn.scaled_dot_product_attention(
            query_2,
            key_2,
            value_2,
            attn_mask=attention_mask_3,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_2 = key_2 = value_2 = attention_mask_3 = None
        transpose_11 = attn_output_8.transpose(1, 2)
        attn_output_8 = None
        attn_output_9 = transpose_11.contiguous()
        transpose_11 = None
        reshape_2 = attn_output_9.reshape(1, 21, -1)
        attn_output_9 = None
        attn_output_10 = reshape_2.contiguous()
        reshape_2 = None
        attn_output_11 = torch._C._nn.linear(
            attn_output_10,
            l_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_10 = l_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_20 = torch.nn.functional.dropout(
            attn_output_11, p=0.1, training=False
        )
        attn_output_11 = None
        hidden_states_21 = hidden_states_19 + hidden_states_20
        hidden_states_19 = hidden_states_20 = None
        hidden_states_22 = torch.nn.functional.layer_norm(
            hidden_states_21,
            (512,),
            l_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_21 = l_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_16 = torch._C._nn.linear(
            hidden_states_22,
            l_self_modules_model_modules_encoder_modules_layers_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_encoder_modules_layers_modules_2_modules_fc1_parameters_bias_,
        )
        l_self_modules_model_modules_encoder_modules_layers_modules_2_modules_fc1_parameters_weight_ = l_self_modules_model_modules_encoder_modules_layers_modules_2_modules_fc1_parameters_bias_ = (None)
        hidden_states_23 = torch.nn.functional.silu(linear_16, inplace=False)
        linear_16 = None
        hidden_states_24 = torch.nn.functional.dropout(
            hidden_states_23, p=0.0, training=False
        )
        hidden_states_23 = None
        hidden_states_25 = torch._C._nn.linear(
            hidden_states_24,
            l_self_modules_model_modules_encoder_modules_layers_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_encoder_modules_layers_modules_2_modules_fc2_parameters_bias_,
        )
        hidden_states_24 = l_self_modules_model_modules_encoder_modules_layers_modules_2_modules_fc2_parameters_weight_ = l_self_modules_model_modules_encoder_modules_layers_modules_2_modules_fc2_parameters_bias_ = (None)
        hidden_states_26 = torch.nn.functional.dropout(
            hidden_states_25, p=0.1, training=False
        )
        hidden_states_25 = None
        hidden_states_27 = hidden_states_22 + hidden_states_26
        hidden_states_22 = hidden_states_26 = None
        hidden_states_28 = torch.nn.functional.layer_norm(
            hidden_states_27,
            (512,),
            l_self_modules_model_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_27 = l_self_modules_model_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_ = (None)
        linear_18 = torch._C._nn.linear(
            hidden_states_28,
            l_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_10 = linear_18.view(1, 21, -1, 64)
        linear_18 = None
        query_states_3 = view_10.transpose(1, 2)
        view_10 = None
        key_states_6 = torch._C._nn.linear(
            hidden_states_28,
            l_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_6 = torch._C._nn.linear(
            hidden_states_28,
            l_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_11 = key_states_6.view(1, 21, -1, 64)
        key_states_6 = None
        key_states_7 = view_11.transpose(1, 2)
        view_11 = None
        view_12 = value_states_6.view(1, 21, -1, 64)
        value_states_6 = None
        value_states_7 = view_12.transpose(1, 2)
        view_12 = None
        attention_mask_4 = attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 21, None),
            )
        ]
        query_3 = query_states_3.contiguous()
        query_states_3 = None
        key_3 = key_states_7.contiguous()
        key_states_7 = None
        value_3 = value_states_7.contiguous()
        value_states_7 = None
        attn_output_12 = torch._C._nn.scaled_dot_product_attention(
            query_3,
            key_3,
            value_3,
            attn_mask=attention_mask_4,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_3 = key_3 = value_3 = attention_mask_4 = None
        transpose_15 = attn_output_12.transpose(1, 2)
        attn_output_12 = None
        attn_output_13 = transpose_15.contiguous()
        transpose_15 = None
        reshape_3 = attn_output_13.reshape(1, 21, -1)
        attn_output_13 = None
        attn_output_14 = reshape_3.contiguous()
        reshape_3 = None
        attn_output_15 = torch._C._nn.linear(
            attn_output_14,
            l_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_14 = l_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_29 = torch.nn.functional.dropout(
            attn_output_15, p=0.1, training=False
        )
        attn_output_15 = None
        hidden_states_30 = hidden_states_28 + hidden_states_29
        hidden_states_28 = hidden_states_29 = None
        hidden_states_31 = torch.nn.functional.layer_norm(
            hidden_states_30,
            (512,),
            l_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_30 = l_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_22 = torch._C._nn.linear(
            hidden_states_31,
            l_self_modules_model_modules_encoder_modules_layers_modules_3_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_encoder_modules_layers_modules_3_modules_fc1_parameters_bias_,
        )
        l_self_modules_model_modules_encoder_modules_layers_modules_3_modules_fc1_parameters_weight_ = l_self_modules_model_modules_encoder_modules_layers_modules_3_modules_fc1_parameters_bias_ = (None)
        hidden_states_32 = torch.nn.functional.silu(linear_22, inplace=False)
        linear_22 = None
        hidden_states_33 = torch.nn.functional.dropout(
            hidden_states_32, p=0.0, training=False
        )
        hidden_states_32 = None
        hidden_states_34 = torch._C._nn.linear(
            hidden_states_33,
            l_self_modules_model_modules_encoder_modules_layers_modules_3_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_encoder_modules_layers_modules_3_modules_fc2_parameters_bias_,
        )
        hidden_states_33 = l_self_modules_model_modules_encoder_modules_layers_modules_3_modules_fc2_parameters_weight_ = l_self_modules_model_modules_encoder_modules_layers_modules_3_modules_fc2_parameters_bias_ = (None)
        hidden_states_35 = torch.nn.functional.dropout(
            hidden_states_34, p=0.1, training=False
        )
        hidden_states_34 = None
        hidden_states_36 = hidden_states_31 + hidden_states_35
        hidden_states_31 = hidden_states_35 = None
        hidden_states_37 = torch.nn.functional.layer_norm(
            hidden_states_36,
            (512,),
            l_self_modules_model_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_36 = l_self_modules_model_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_ = (None)
        linear_24 = torch._C._nn.linear(
            hidden_states_37,
            l_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_13 = linear_24.view(1, 21, -1, 64)
        linear_24 = None
        query_states_4 = view_13.transpose(1, 2)
        view_13 = None
        key_states_8 = torch._C._nn.linear(
            hidden_states_37,
            l_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_8 = torch._C._nn.linear(
            hidden_states_37,
            l_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_14 = key_states_8.view(1, 21, -1, 64)
        key_states_8 = None
        key_states_9 = view_14.transpose(1, 2)
        view_14 = None
        view_15 = value_states_8.view(1, 21, -1, 64)
        value_states_8 = None
        value_states_9 = view_15.transpose(1, 2)
        view_15 = None
        attention_mask_5 = attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 21, None),
            )
        ]
        query_4 = query_states_4.contiguous()
        query_states_4 = None
        key_4 = key_states_9.contiguous()
        key_states_9 = None
        value_4 = value_states_9.contiguous()
        value_states_9 = None
        attn_output_16 = torch._C._nn.scaled_dot_product_attention(
            query_4,
            key_4,
            value_4,
            attn_mask=attention_mask_5,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_4 = key_4 = value_4 = attention_mask_5 = None
        transpose_19 = attn_output_16.transpose(1, 2)
        attn_output_16 = None
        attn_output_17 = transpose_19.contiguous()
        transpose_19 = None
        reshape_4 = attn_output_17.reshape(1, 21, -1)
        attn_output_17 = None
        attn_output_18 = reshape_4.contiguous()
        reshape_4 = None
        attn_output_19 = torch._C._nn.linear(
            attn_output_18,
            l_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_18 = l_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_38 = torch.nn.functional.dropout(
            attn_output_19, p=0.1, training=False
        )
        attn_output_19 = None
        hidden_states_39 = hidden_states_37 + hidden_states_38
        hidden_states_37 = hidden_states_38 = None
        hidden_states_40 = torch.nn.functional.layer_norm(
            hidden_states_39,
            (512,),
            l_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_39 = l_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_28 = torch._C._nn.linear(
            hidden_states_40,
            l_self_modules_model_modules_encoder_modules_layers_modules_4_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_encoder_modules_layers_modules_4_modules_fc1_parameters_bias_,
        )
        l_self_modules_model_modules_encoder_modules_layers_modules_4_modules_fc1_parameters_weight_ = l_self_modules_model_modules_encoder_modules_layers_modules_4_modules_fc1_parameters_bias_ = (None)
        hidden_states_41 = torch.nn.functional.silu(linear_28, inplace=False)
        linear_28 = None
        hidden_states_42 = torch.nn.functional.dropout(
            hidden_states_41, p=0.0, training=False
        )
        hidden_states_41 = None
        hidden_states_43 = torch._C._nn.linear(
            hidden_states_42,
            l_self_modules_model_modules_encoder_modules_layers_modules_4_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_encoder_modules_layers_modules_4_modules_fc2_parameters_bias_,
        )
        hidden_states_42 = l_self_modules_model_modules_encoder_modules_layers_modules_4_modules_fc2_parameters_weight_ = l_self_modules_model_modules_encoder_modules_layers_modules_4_modules_fc2_parameters_bias_ = (None)
        hidden_states_44 = torch.nn.functional.dropout(
            hidden_states_43, p=0.1, training=False
        )
        hidden_states_43 = None
        hidden_states_45 = hidden_states_40 + hidden_states_44
        hidden_states_40 = hidden_states_44 = None
        hidden_states_46 = torch.nn.functional.layer_norm(
            hidden_states_45,
            (512,),
            l_self_modules_model_modules_encoder_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_encoder_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_45 = l_self_modules_model_modules_encoder_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_encoder_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_ = (None)
        linear_30 = torch._C._nn.linear(
            hidden_states_46,
            l_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_16 = linear_30.view(1, 21, -1, 64)
        linear_30 = None
        query_states_5 = view_16.transpose(1, 2)
        view_16 = None
        key_states_10 = torch._C._nn.linear(
            hidden_states_46,
            l_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_10 = torch._C._nn.linear(
            hidden_states_46,
            l_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_17 = key_states_10.view(1, 21, -1, 64)
        key_states_10 = None
        key_states_11 = view_17.transpose(1, 2)
        view_17 = None
        view_18 = value_states_10.view(1, 21, -1, 64)
        value_states_10 = None
        value_states_11 = view_18.transpose(1, 2)
        view_18 = None
        attention_mask_6 = attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 21, None),
            )
        ]
        attention_mask = None
        query_5 = query_states_5.contiguous()
        query_states_5 = None
        key_5 = key_states_11.contiguous()
        key_states_11 = None
        value_5 = value_states_11.contiguous()
        value_states_11 = None
        attn_output_20 = torch._C._nn.scaled_dot_product_attention(
            query_5,
            key_5,
            value_5,
            attn_mask=attention_mask_6,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_5 = key_5 = value_5 = attention_mask_6 = None
        transpose_23 = attn_output_20.transpose(1, 2)
        attn_output_20 = None
        attn_output_21 = transpose_23.contiguous()
        transpose_23 = None
        reshape_5 = attn_output_21.reshape(1, 21, -1)
        attn_output_21 = None
        attn_output_22 = reshape_5.contiguous()
        reshape_5 = None
        attn_output_23 = torch._C._nn.linear(
            attn_output_22,
            l_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_22 = l_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_47 = torch.nn.functional.dropout(
            attn_output_23, p=0.1, training=False
        )
        attn_output_23 = None
        hidden_states_48 = hidden_states_46 + hidden_states_47
        hidden_states_46 = hidden_states_47 = None
        hidden_states_49 = torch.nn.functional.layer_norm(
            hidden_states_48,
            (512,),
            l_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_48 = l_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_34 = torch._C._nn.linear(
            hidden_states_49,
            l_self_modules_model_modules_encoder_modules_layers_modules_5_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_encoder_modules_layers_modules_5_modules_fc1_parameters_bias_,
        )
        l_self_modules_model_modules_encoder_modules_layers_modules_5_modules_fc1_parameters_weight_ = l_self_modules_model_modules_encoder_modules_layers_modules_5_modules_fc1_parameters_bias_ = (None)
        hidden_states_50 = torch.nn.functional.silu(linear_34, inplace=False)
        linear_34 = None
        hidden_states_51 = torch.nn.functional.dropout(
            hidden_states_50, p=0.0, training=False
        )
        hidden_states_50 = None
        hidden_states_52 = torch._C._nn.linear(
            hidden_states_51,
            l_self_modules_model_modules_encoder_modules_layers_modules_5_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_encoder_modules_layers_modules_5_modules_fc2_parameters_bias_,
        )
        hidden_states_51 = l_self_modules_model_modules_encoder_modules_layers_modules_5_modules_fc2_parameters_weight_ = l_self_modules_model_modules_encoder_modules_layers_modules_5_modules_fc2_parameters_bias_ = (None)
        hidden_states_53 = torch.nn.functional.dropout(
            hidden_states_52, p=0.1, training=False
        )
        hidden_states_52 = None
        hidden_states_54 = hidden_states_49 + hidden_states_53
        hidden_states_49 = hidden_states_53 = None
        hidden_states_55 = torch.nn.functional.layer_norm(
            hidden_states_54,
            (512,),
            l_self_modules_model_modules_encoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_encoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_54 = l_self_modules_model_modules_encoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_encoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_ = (None)
        input_ids_1 = l_decoder_input_ids_.view(-1, 1)
        input_ids_1 = None
        inputs_embeds_1 = torch.nn.functional.embedding(
            l_decoder_input_ids_,
            l_self_modules_model_modules_encoder_modules_embed_tokens_parameters_weight_,
            52996,
            None,
            2.0,
            False,
            False,
        )
        l_decoder_input_ids_ = None
        inputs_embeds_2 = inputs_embeds_1 * 22.627416997969522
        inputs_embeds_1 = None
        cache_position = torch.arange(0, 1, device=device(type="cpu"))
        causal_mask = torch.full(
            (1, 2),
            fill_value=-3.4028234663852886e38,
            dtype=torch.float32,
            device=device(type="cpu"),
        )
        arange_2 = torch.arange(2, device=device(type="cpu"))
        reshape_6 = cache_position.reshape(-1, 1)
        gt = arange_2 > reshape_6
        arange_2 = reshape_6 = None
        causal_mask *= gt
        causal_mask_1 = causal_mask
        causal_mask = gt = None
        getitem_7 = causal_mask_1[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        causal_mask_1 = None
        causal_mask_2 = getitem_7.expand(1, 1, -1, -1)
        getitem_7 = None
        getitem_8 = l_attention_mask_[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        l_attention_mask_ = None
        expand_2 = getitem_8.expand(1, 1, 1, 21)
        getitem_8 = None
        expanded_mask_1 = expand_2.to(torch.float32)
        expand_2 = None
        tensor_1 = torch.tensor(1.0, dtype=torch.float32)
        inverted_mask_1 = tensor_1 - expanded_mask_1
        tensor_1 = expanded_mask_1 = None
        to_3 = inverted_mask_1.to(torch.bool)
        encoder_attention_mask = inverted_mask_1.masked_fill(
            to_3, -3.4028234663852886e38
        )
        inverted_mask_1 = to_3 = None
        position_ids_1 = torch.nn.functional.embedding(
            cache_position,
            l_self_modules_model_modules_decoder_modules_embed_positions_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        cache_position = l_self_modules_model_modules_decoder_modules_embed_positions_parameters_weight_ = (None)
        hidden_states_56 = inputs_embeds_2 + position_ids_1
        inputs_embeds_2 = position_ids_1 = None
        hidden_states_57 = torch.nn.functional.dropout(
            hidden_states_56, p=0.1, training=False
        )
        hidden_states_56 = None
        linear_36 = torch._C._nn.linear(
            hidden_states_57,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_20 = linear_36.view(1, 1, -1, 64)
        linear_36 = None
        query_states_6 = view_20.transpose(1, 2)
        view_20 = None
        key_states_12 = torch._C._nn.linear(
            hidden_states_57,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_12 = torch._C._nn.linear(
            hidden_states_57,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_21 = key_states_12.view(1, 1, -1, 64)
        key_states_12 = None
        key_states_13 = view_21.transpose(1, 2)
        view_21 = None
        view_22 = value_states_12.view(1, 1, -1, 64)
        value_states_12 = None
        value_states_13 = view_22.transpose(1, 2)
        view_22 = None
        attention_mask_7 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
            )
        ]
        query_6 = query_states_6.contiguous()
        query_states_6 = None
        key_6 = key_states_13.contiguous()
        key_states_13 = None
        value_6 = value_states_13.contiguous()
        value_states_13 = None
        attn_output_24 = torch._C._nn.scaled_dot_product_attention(
            query_6,
            key_6,
            value_6,
            attn_mask=attention_mask_7,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_6 = key_6 = value_6 = attention_mask_7 = None
        transpose_27 = attn_output_24.transpose(1, 2)
        attn_output_24 = None
        attn_output_25 = transpose_27.contiguous()
        transpose_27 = None
        reshape_7 = attn_output_25.reshape(1, 1, -1)
        attn_output_25 = None
        attn_output_26 = reshape_7.contiguous()
        reshape_7 = None
        attn_output_27 = torch._C._nn.linear(
            attn_output_26,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_26 = l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_58 = torch.nn.functional.dropout(
            attn_output_27, p=0.1, training=False
        )
        attn_output_27 = None
        hidden_states_59 = hidden_states_57 + hidden_states_58
        hidden_states_57 = hidden_states_58 = None
        hidden_states_60 = torch.nn.functional.layer_norm(
            hidden_states_59,
            (512,),
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_59 = l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_40 = torch._C._nn.linear(
            hidden_states_60,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        view_23 = linear_40.view(1, 1, -1, 64)
        linear_40 = None
        query_states_7 = view_23.transpose(1, 2)
        view_23 = None
        key_states_14 = torch._C._nn.linear(
            hidden_states_55,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_14 = torch._C._nn.linear(
            hidden_states_55,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_24 = key_states_14.view(1, 21, -1, 64)
        key_states_14 = None
        key_states_15 = view_24.transpose(1, 2)
        view_24 = None
        view_25 = value_states_14.view(1, 21, -1, 64)
        value_states_14 = None
        value_states_15 = view_25.transpose(1, 2)
        view_25 = None
        attention_mask_8 = encoder_attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 21, None),
            )
        ]
        query_7 = query_states_7.contiguous()
        query_states_7 = None
        key_7 = key_states_15.contiguous()
        key_states_15 = None
        value_7 = value_states_15.contiguous()
        value_states_15 = None
        attn_output_28 = torch._C._nn.scaled_dot_product_attention(
            query_7,
            key_7,
            value_7,
            attn_mask=attention_mask_8,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_7 = key_7 = value_7 = attention_mask_8 = None
        transpose_31 = attn_output_28.transpose(1, 2)
        attn_output_28 = None
        attn_output_29 = transpose_31.contiguous()
        transpose_31 = None
        reshape_8 = attn_output_29.reshape(1, 1, -1)
        attn_output_29 = None
        attn_output_30 = reshape_8.contiguous()
        reshape_8 = None
        attn_output_31 = torch._C._nn.linear(
            attn_output_30,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_30 = l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_61 = torch.nn.functional.dropout(
            attn_output_31, p=0.1, training=False
        )
        attn_output_31 = None
        hidden_states_62 = hidden_states_60 + hidden_states_61
        hidden_states_60 = hidden_states_61 = None
        hidden_states_63 = torch.nn.functional.layer_norm(
            hidden_states_62,
            (512,),
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_62 = l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_44 = torch._C._nn.linear(
            hidden_states_63,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_bias_ = (None)
        hidden_states_64 = torch.nn.functional.silu(linear_44, inplace=False)
        linear_44 = None
        hidden_states_65 = torch.nn.functional.dropout(
            hidden_states_64, p=0.0, training=False
        )
        hidden_states_64 = None
        hidden_states_66 = torch._C._nn.linear(
            hidden_states_65,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_bias_,
        )
        hidden_states_65 = l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_bias_ = (None)
        hidden_states_67 = torch.nn.functional.dropout(
            hidden_states_66, p=0.1, training=False
        )
        hidden_states_66 = None
        hidden_states_68 = hidden_states_63 + hidden_states_67
        hidden_states_63 = hidden_states_67 = None
        hidden_states_69 = torch.nn.functional.layer_norm(
            hidden_states_68,
            (512,),
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_68 = l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_ = (None)
        linear_46 = torch._C._nn.linear(
            hidden_states_69,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_26 = linear_46.view(1, 1, -1, 64)
        linear_46 = None
        query_states_8 = view_26.transpose(1, 2)
        view_26 = None
        key_states_16 = torch._C._nn.linear(
            hidden_states_69,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_16 = torch._C._nn.linear(
            hidden_states_69,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_27 = key_states_16.view(1, 1, -1, 64)
        key_states_16 = None
        key_states_17 = view_27.transpose(1, 2)
        view_27 = None
        view_28 = value_states_16.view(1, 1, -1, 64)
        value_states_16 = None
        value_states_17 = view_28.transpose(1, 2)
        view_28 = None
        attention_mask_9 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
            )
        ]
        query_8 = query_states_8.contiguous()
        query_states_8 = None
        key_8 = key_states_17.contiguous()
        key_states_17 = None
        value_8 = value_states_17.contiguous()
        value_states_17 = None
        attn_output_32 = torch._C._nn.scaled_dot_product_attention(
            query_8,
            key_8,
            value_8,
            attn_mask=attention_mask_9,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_8 = key_8 = value_8 = attention_mask_9 = None
        transpose_35 = attn_output_32.transpose(1, 2)
        attn_output_32 = None
        attn_output_33 = transpose_35.contiguous()
        transpose_35 = None
        reshape_9 = attn_output_33.reshape(1, 1, -1)
        attn_output_33 = None
        attn_output_34 = reshape_9.contiguous()
        reshape_9 = None
        attn_output_35 = torch._C._nn.linear(
            attn_output_34,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_34 = l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_70 = torch.nn.functional.dropout(
            attn_output_35, p=0.1, training=False
        )
        attn_output_35 = None
        hidden_states_71 = hidden_states_69 + hidden_states_70
        hidden_states_69 = hidden_states_70 = None
        hidden_states_72 = torch.nn.functional.layer_norm(
            hidden_states_71,
            (512,),
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_71 = l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_50 = torch._C._nn.linear(
            hidden_states_72,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        view_29 = linear_50.view(1, 1, -1, 64)
        linear_50 = None
        query_states_9 = view_29.transpose(1, 2)
        view_29 = None
        key_states_18 = torch._C._nn.linear(
            hidden_states_55,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_18 = torch._C._nn.linear(
            hidden_states_55,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_30 = key_states_18.view(1, 21, -1, 64)
        key_states_18 = None
        key_states_19 = view_30.transpose(1, 2)
        view_30 = None
        view_31 = value_states_18.view(1, 21, -1, 64)
        value_states_18 = None
        value_states_19 = view_31.transpose(1, 2)
        view_31 = None
        attention_mask_10 = encoder_attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 21, None),
            )
        ]
        query_9 = query_states_9.contiguous()
        query_states_9 = None
        key_9 = key_states_19.contiguous()
        key_states_19 = None
        value_9 = value_states_19.contiguous()
        value_states_19 = None
        attn_output_36 = torch._C._nn.scaled_dot_product_attention(
            query_9,
            key_9,
            value_9,
            attn_mask=attention_mask_10,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_9 = key_9 = value_9 = attention_mask_10 = None
        transpose_39 = attn_output_36.transpose(1, 2)
        attn_output_36 = None
        attn_output_37 = transpose_39.contiguous()
        transpose_39 = None
        reshape_10 = attn_output_37.reshape(1, 1, -1)
        attn_output_37 = None
        attn_output_38 = reshape_10.contiguous()
        reshape_10 = None
        attn_output_39 = torch._C._nn.linear(
            attn_output_38,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_38 = l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_73 = torch.nn.functional.dropout(
            attn_output_39, p=0.1, training=False
        )
        attn_output_39 = None
        hidden_states_74 = hidden_states_72 + hidden_states_73
        hidden_states_72 = hidden_states_73 = None
        hidden_states_75 = torch.nn.functional.layer_norm(
            hidden_states_74,
            (512,),
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_74 = l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_54 = torch._C._nn.linear(
            hidden_states_75,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_bias_ = (None)
        hidden_states_76 = torch.nn.functional.silu(linear_54, inplace=False)
        linear_54 = None
        hidden_states_77 = torch.nn.functional.dropout(
            hidden_states_76, p=0.0, training=False
        )
        hidden_states_76 = None
        hidden_states_78 = torch._C._nn.linear(
            hidden_states_77,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_bias_,
        )
        hidden_states_77 = l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_bias_ = (None)
        hidden_states_79 = torch.nn.functional.dropout(
            hidden_states_78, p=0.1, training=False
        )
        hidden_states_78 = None
        hidden_states_80 = hidden_states_75 + hidden_states_79
        hidden_states_75 = hidden_states_79 = None
        hidden_states_81 = torch.nn.functional.layer_norm(
            hidden_states_80,
            (512,),
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_80 = l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_ = (None)
        linear_56 = torch._C._nn.linear(
            hidden_states_81,
            l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_32 = linear_56.view(1, 1, -1, 64)
        linear_56 = None
        query_states_10 = view_32.transpose(1, 2)
        view_32 = None
        key_states_20 = torch._C._nn.linear(
            hidden_states_81,
            l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_20 = torch._C._nn.linear(
            hidden_states_81,
            l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_33 = key_states_20.view(1, 1, -1, 64)
        key_states_20 = None
        key_states_21 = view_33.transpose(1, 2)
        view_33 = None
        view_34 = value_states_20.view(1, 1, -1, 64)
        value_states_20 = None
        value_states_21 = view_34.transpose(1, 2)
        view_34 = None
        attention_mask_11 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
            )
        ]
        query_10 = query_states_10.contiguous()
        query_states_10 = None
        key_10 = key_states_21.contiguous()
        key_states_21 = None
        value_10 = value_states_21.contiguous()
        value_states_21 = None
        attn_output_40 = torch._C._nn.scaled_dot_product_attention(
            query_10,
            key_10,
            value_10,
            attn_mask=attention_mask_11,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_10 = key_10 = value_10 = attention_mask_11 = None
        transpose_43 = attn_output_40.transpose(1, 2)
        attn_output_40 = None
        attn_output_41 = transpose_43.contiguous()
        transpose_43 = None
        reshape_11 = attn_output_41.reshape(1, 1, -1)
        attn_output_41 = None
        attn_output_42 = reshape_11.contiguous()
        reshape_11 = None
        attn_output_43 = torch._C._nn.linear(
            attn_output_42,
            l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_42 = l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_82 = torch.nn.functional.dropout(
            attn_output_43, p=0.1, training=False
        )
        attn_output_43 = None
        hidden_states_83 = hidden_states_81 + hidden_states_82
        hidden_states_81 = hidden_states_82 = None
        hidden_states_84 = torch.nn.functional.layer_norm(
            hidden_states_83,
            (512,),
            l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_83 = l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_60 = torch._C._nn.linear(
            hidden_states_84,
            l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        view_35 = linear_60.view(1, 1, -1, 64)
        linear_60 = None
        query_states_11 = view_35.transpose(1, 2)
        view_35 = None
        key_states_22 = torch._C._nn.linear(
            hidden_states_55,
            l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_22 = torch._C._nn.linear(
            hidden_states_55,
            l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_36 = key_states_22.view(1, 21, -1, 64)
        key_states_22 = None
        key_states_23 = view_36.transpose(1, 2)
        view_36 = None
        view_37 = value_states_22.view(1, 21, -1, 64)
        value_states_22 = None
        value_states_23 = view_37.transpose(1, 2)
        view_37 = None
        attention_mask_12 = encoder_attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 21, None),
            )
        ]
        query_11 = query_states_11.contiguous()
        query_states_11 = None
        key_11 = key_states_23.contiguous()
        key_states_23 = None
        value_11 = value_states_23.contiguous()
        value_states_23 = None
        attn_output_44 = torch._C._nn.scaled_dot_product_attention(
            query_11,
            key_11,
            value_11,
            attn_mask=attention_mask_12,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_11 = key_11 = value_11 = attention_mask_12 = None
        transpose_47 = attn_output_44.transpose(1, 2)
        attn_output_44 = None
        attn_output_45 = transpose_47.contiguous()
        transpose_47 = None
        reshape_12 = attn_output_45.reshape(1, 1, -1)
        attn_output_45 = None
        attn_output_46 = reshape_12.contiguous()
        reshape_12 = None
        attn_output_47 = torch._C._nn.linear(
            attn_output_46,
            l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_46 = l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_85 = torch.nn.functional.dropout(
            attn_output_47, p=0.1, training=False
        )
        attn_output_47 = None
        hidden_states_86 = hidden_states_84 + hidden_states_85
        hidden_states_84 = hidden_states_85 = None
        hidden_states_87 = torch.nn.functional.layer_norm(
            hidden_states_86,
            (512,),
            l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_86 = l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_64 = torch._C._nn.linear(
            hidden_states_87,
            l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_bias_ = (None)
        hidden_states_88 = torch.nn.functional.silu(linear_64, inplace=False)
        linear_64 = None
        hidden_states_89 = torch.nn.functional.dropout(
            hidden_states_88, p=0.0, training=False
        )
        hidden_states_88 = None
        hidden_states_90 = torch._C._nn.linear(
            hidden_states_89,
            l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_bias_,
        )
        hidden_states_89 = l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_bias_ = (None)
        hidden_states_91 = torch.nn.functional.dropout(
            hidden_states_90, p=0.1, training=False
        )
        hidden_states_90 = None
        hidden_states_92 = hidden_states_87 + hidden_states_91
        hidden_states_87 = hidden_states_91 = None
        hidden_states_93 = torch.nn.functional.layer_norm(
            hidden_states_92,
            (512,),
            l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_92 = l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_ = (None)
        linear_66 = torch._C._nn.linear(
            hidden_states_93,
            l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_38 = linear_66.view(1, 1, -1, 64)
        linear_66 = None
        query_states_12 = view_38.transpose(1, 2)
        view_38 = None
        key_states_24 = torch._C._nn.linear(
            hidden_states_93,
            l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_24 = torch._C._nn.linear(
            hidden_states_93,
            l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_39 = key_states_24.view(1, 1, -1, 64)
        key_states_24 = None
        key_states_25 = view_39.transpose(1, 2)
        view_39 = None
        view_40 = value_states_24.view(1, 1, -1, 64)
        value_states_24 = None
        value_states_25 = view_40.transpose(1, 2)
        view_40 = None
        attention_mask_13 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
            )
        ]
        query_12 = query_states_12.contiguous()
        query_states_12 = None
        key_12 = key_states_25.contiguous()
        key_states_25 = None
        value_12 = value_states_25.contiguous()
        value_states_25 = None
        attn_output_48 = torch._C._nn.scaled_dot_product_attention(
            query_12,
            key_12,
            value_12,
            attn_mask=attention_mask_13,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_12 = key_12 = value_12 = attention_mask_13 = None
        transpose_51 = attn_output_48.transpose(1, 2)
        attn_output_48 = None
        attn_output_49 = transpose_51.contiguous()
        transpose_51 = None
        reshape_13 = attn_output_49.reshape(1, 1, -1)
        attn_output_49 = None
        attn_output_50 = reshape_13.contiguous()
        reshape_13 = None
        attn_output_51 = torch._C._nn.linear(
            attn_output_50,
            l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_50 = l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_94 = torch.nn.functional.dropout(
            attn_output_51, p=0.1, training=False
        )
        attn_output_51 = None
        hidden_states_95 = hidden_states_93 + hidden_states_94
        hidden_states_93 = hidden_states_94 = None
        hidden_states_96 = torch.nn.functional.layer_norm(
            hidden_states_95,
            (512,),
            l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_95 = l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_70 = torch._C._nn.linear(
            hidden_states_96,
            l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        view_41 = linear_70.view(1, 1, -1, 64)
        linear_70 = None
        query_states_13 = view_41.transpose(1, 2)
        view_41 = None
        key_states_26 = torch._C._nn.linear(
            hidden_states_55,
            l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_26 = torch._C._nn.linear(
            hidden_states_55,
            l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_42 = key_states_26.view(1, 21, -1, 64)
        key_states_26 = None
        key_states_27 = view_42.transpose(1, 2)
        view_42 = None
        view_43 = value_states_26.view(1, 21, -1, 64)
        value_states_26 = None
        value_states_27 = view_43.transpose(1, 2)
        view_43 = None
        attention_mask_14 = encoder_attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 21, None),
            )
        ]
        query_13 = query_states_13.contiguous()
        query_states_13 = None
        key_13 = key_states_27.contiguous()
        key_states_27 = None
        value_13 = value_states_27.contiguous()
        value_states_27 = None
        attn_output_52 = torch._C._nn.scaled_dot_product_attention(
            query_13,
            key_13,
            value_13,
            attn_mask=attention_mask_14,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_13 = key_13 = value_13 = attention_mask_14 = None
        transpose_55 = attn_output_52.transpose(1, 2)
        attn_output_52 = None
        attn_output_53 = transpose_55.contiguous()
        transpose_55 = None
        reshape_14 = attn_output_53.reshape(1, 1, -1)
        attn_output_53 = None
        attn_output_54 = reshape_14.contiguous()
        reshape_14 = None
        attn_output_55 = torch._C._nn.linear(
            attn_output_54,
            l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_54 = l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_97 = torch.nn.functional.dropout(
            attn_output_55, p=0.1, training=False
        )
        attn_output_55 = None
        hidden_states_98 = hidden_states_96 + hidden_states_97
        hidden_states_96 = hidden_states_97 = None
        hidden_states_99 = torch.nn.functional.layer_norm(
            hidden_states_98,
            (512,),
            l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_98 = l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_74 = torch._C._nn.linear(
            hidden_states_99,
            l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_bias_ = (None)
        hidden_states_100 = torch.nn.functional.silu(linear_74, inplace=False)
        linear_74 = None
        hidden_states_101 = torch.nn.functional.dropout(
            hidden_states_100, p=0.0, training=False
        )
        hidden_states_100 = None
        hidden_states_102 = torch._C._nn.linear(
            hidden_states_101,
            l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_bias_,
        )
        hidden_states_101 = l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_bias_ = (None)
        hidden_states_103 = torch.nn.functional.dropout(
            hidden_states_102, p=0.1, training=False
        )
        hidden_states_102 = None
        hidden_states_104 = hidden_states_99 + hidden_states_103
        hidden_states_99 = hidden_states_103 = None
        hidden_states_105 = torch.nn.functional.layer_norm(
            hidden_states_104,
            (512,),
            l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_104 = l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_ = (None)
        linear_76 = torch._C._nn.linear(
            hidden_states_105,
            l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_44 = linear_76.view(1, 1, -1, 64)
        linear_76 = None
        query_states_14 = view_44.transpose(1, 2)
        view_44 = None
        key_states_28 = torch._C._nn.linear(
            hidden_states_105,
            l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_28 = torch._C._nn.linear(
            hidden_states_105,
            l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_45 = key_states_28.view(1, 1, -1, 64)
        key_states_28 = None
        key_states_29 = view_45.transpose(1, 2)
        view_45 = None
        view_46 = value_states_28.view(1, 1, -1, 64)
        value_states_28 = None
        value_states_29 = view_46.transpose(1, 2)
        view_46 = None
        attention_mask_15 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
            )
        ]
        query_14 = query_states_14.contiguous()
        query_states_14 = None
        key_14 = key_states_29.contiguous()
        key_states_29 = None
        value_14 = value_states_29.contiguous()
        value_states_29 = None
        attn_output_56 = torch._C._nn.scaled_dot_product_attention(
            query_14,
            key_14,
            value_14,
            attn_mask=attention_mask_15,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_14 = key_14 = value_14 = attention_mask_15 = None
        transpose_59 = attn_output_56.transpose(1, 2)
        attn_output_56 = None
        attn_output_57 = transpose_59.contiguous()
        transpose_59 = None
        reshape_15 = attn_output_57.reshape(1, 1, -1)
        attn_output_57 = None
        attn_output_58 = reshape_15.contiguous()
        reshape_15 = None
        attn_output_59 = torch._C._nn.linear(
            attn_output_58,
            l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_58 = l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_106 = torch.nn.functional.dropout(
            attn_output_59, p=0.1, training=False
        )
        attn_output_59 = None
        hidden_states_107 = hidden_states_105 + hidden_states_106
        hidden_states_105 = hidden_states_106 = None
        hidden_states_108 = torch.nn.functional.layer_norm(
            hidden_states_107,
            (512,),
            l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_107 = l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_80 = torch._C._nn.linear(
            hidden_states_108,
            l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        view_47 = linear_80.view(1, 1, -1, 64)
        linear_80 = None
        query_states_15 = view_47.transpose(1, 2)
        view_47 = None
        key_states_30 = torch._C._nn.linear(
            hidden_states_55,
            l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_30 = torch._C._nn.linear(
            hidden_states_55,
            l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_48 = key_states_30.view(1, 21, -1, 64)
        key_states_30 = None
        key_states_31 = view_48.transpose(1, 2)
        view_48 = None
        view_49 = value_states_30.view(1, 21, -1, 64)
        value_states_30 = None
        value_states_31 = view_49.transpose(1, 2)
        view_49 = None
        attention_mask_16 = encoder_attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 21, None),
            )
        ]
        query_15 = query_states_15.contiguous()
        query_states_15 = None
        key_15 = key_states_31.contiguous()
        key_states_31 = None
        value_15 = value_states_31.contiguous()
        value_states_31 = None
        attn_output_60 = torch._C._nn.scaled_dot_product_attention(
            query_15,
            key_15,
            value_15,
            attn_mask=attention_mask_16,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_15 = key_15 = value_15 = attention_mask_16 = None
        transpose_63 = attn_output_60.transpose(1, 2)
        attn_output_60 = None
        attn_output_61 = transpose_63.contiguous()
        transpose_63 = None
        reshape_16 = attn_output_61.reshape(1, 1, -1)
        attn_output_61 = None
        attn_output_62 = reshape_16.contiguous()
        reshape_16 = None
        attn_output_63 = torch._C._nn.linear(
            attn_output_62,
            l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_62 = l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_109 = torch.nn.functional.dropout(
            attn_output_63, p=0.1, training=False
        )
        attn_output_63 = None
        hidden_states_110 = hidden_states_108 + hidden_states_109
        hidden_states_108 = hidden_states_109 = None
        hidden_states_111 = torch.nn.functional.layer_norm(
            hidden_states_110,
            (512,),
            l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_110 = l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_84 = torch._C._nn.linear(
            hidden_states_111,
            l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_bias_ = (None)
        hidden_states_112 = torch.nn.functional.silu(linear_84, inplace=False)
        linear_84 = None
        hidden_states_113 = torch.nn.functional.dropout(
            hidden_states_112, p=0.0, training=False
        )
        hidden_states_112 = None
        hidden_states_114 = torch._C._nn.linear(
            hidden_states_113,
            l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_bias_,
        )
        hidden_states_113 = l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_bias_ = (None)
        hidden_states_115 = torch.nn.functional.dropout(
            hidden_states_114, p=0.1, training=False
        )
        hidden_states_114 = None
        hidden_states_116 = hidden_states_111 + hidden_states_115
        hidden_states_111 = hidden_states_115 = None
        hidden_states_117 = torch.nn.functional.layer_norm(
            hidden_states_116,
            (512,),
            l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_116 = l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_ = (None)
        linear_86 = torch._C._nn.linear(
            hidden_states_117,
            l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_50 = linear_86.view(1, 1, -1, 64)
        linear_86 = None
        query_states_16 = view_50.transpose(1, 2)
        view_50 = None
        key_states_32 = torch._C._nn.linear(
            hidden_states_117,
            l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_32 = torch._C._nn.linear(
            hidden_states_117,
            l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_51 = key_states_32.view(1, 1, -1, 64)
        key_states_32 = None
        key_states_33 = view_51.transpose(1, 2)
        view_51 = None
        view_52 = value_states_32.view(1, 1, -1, 64)
        value_states_32 = None
        value_states_33 = view_52.transpose(1, 2)
        view_52 = None
        attention_mask_17 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
            )
        ]
        causal_mask_2 = None
        query_16 = query_states_16.contiguous()
        query_states_16 = None
        key_16 = key_states_33.contiguous()
        key_states_33 = None
        value_16 = value_states_33.contiguous()
        value_states_33 = None
        attn_output_64 = torch._C._nn.scaled_dot_product_attention(
            query_16,
            key_16,
            value_16,
            attn_mask=attention_mask_17,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_16 = key_16 = value_16 = attention_mask_17 = None
        transpose_67 = attn_output_64.transpose(1, 2)
        attn_output_64 = None
        attn_output_65 = transpose_67.contiguous()
        transpose_67 = None
        reshape_17 = attn_output_65.reshape(1, 1, -1)
        attn_output_65 = None
        attn_output_66 = reshape_17.contiguous()
        reshape_17 = None
        attn_output_67 = torch._C._nn.linear(
            attn_output_66,
            l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_66 = l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_118 = torch.nn.functional.dropout(
            attn_output_67, p=0.1, training=False
        )
        attn_output_67 = None
        hidden_states_119 = hidden_states_117 + hidden_states_118
        hidden_states_117 = hidden_states_118 = None
        hidden_states_120 = torch.nn.functional.layer_norm(
            hidden_states_119,
            (512,),
            l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_119 = l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_90 = torch._C._nn.linear(
            hidden_states_120,
            l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        view_53 = linear_90.view(1, 1, -1, 64)
        linear_90 = None
        query_states_17 = view_53.transpose(1, 2)
        view_53 = None
        key_states_34 = torch._C._nn.linear(
            hidden_states_55,
            l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_34 = torch._C._nn.linear(
            hidden_states_55,
            l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_54 = key_states_34.view(1, 21, -1, 64)
        key_states_34 = None
        key_states_35 = view_54.transpose(1, 2)
        view_54 = None
        view_55 = value_states_34.view(1, 21, -1, 64)
        value_states_34 = None
        value_states_35 = view_55.transpose(1, 2)
        view_55 = None
        attention_mask_18 = encoder_attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 21, None),
            )
        ]
        encoder_attention_mask = None
        query_17 = query_states_17.contiguous()
        query_states_17 = None
        key_17 = key_states_35.contiguous()
        key_states_35 = None
        value_17 = value_states_35.contiguous()
        value_states_35 = None
        attn_output_68 = torch._C._nn.scaled_dot_product_attention(
            query_17,
            key_17,
            value_17,
            attn_mask=attention_mask_18,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_17 = key_17 = value_17 = attention_mask_18 = None
        transpose_71 = attn_output_68.transpose(1, 2)
        attn_output_68 = None
        attn_output_69 = transpose_71.contiguous()
        transpose_71 = None
        reshape_18 = attn_output_69.reshape(1, 1, -1)
        attn_output_69 = None
        attn_output_70 = reshape_18.contiguous()
        reshape_18 = None
        attn_output_71 = torch._C._nn.linear(
            attn_output_70,
            l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_70 = l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_121 = torch.nn.functional.dropout(
            attn_output_71, p=0.1, training=False
        )
        attn_output_71 = None
        hidden_states_122 = hidden_states_120 + hidden_states_121
        hidden_states_120 = hidden_states_121 = None
        hidden_states_123 = torch.nn.functional.layer_norm(
            hidden_states_122,
            (512,),
            l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_122 = l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_94 = torch._C._nn.linear(
            hidden_states_123,
            l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_bias_ = (None)
        hidden_states_124 = torch.nn.functional.silu(linear_94, inplace=False)
        linear_94 = None
        hidden_states_125 = torch.nn.functional.dropout(
            hidden_states_124, p=0.0, training=False
        )
        hidden_states_124 = None
        hidden_states_126 = torch._C._nn.linear(
            hidden_states_125,
            l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_bias_,
        )
        hidden_states_125 = l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_bias_ = (None)
        hidden_states_127 = torch.nn.functional.dropout(
            hidden_states_126, p=0.1, training=False
        )
        hidden_states_126 = None
        hidden_states_128 = hidden_states_123 + hidden_states_127
        hidden_states_123 = hidden_states_127 = None
        hidden_states_129 = torch.nn.functional.layer_norm(
            hidden_states_128,
            (512,),
            l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_128 = l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_ = (None)
        linear_96 = torch._C._nn.linear(
            hidden_states_129,
            l_self_modules_model_modules_encoder_modules_embed_tokens_parameters_weight_,
            None,
        )
        hidden_states_129 = (
            l_self_modules_model_modules_encoder_modules_embed_tokens_parameters_weight_
        ) = None
        lm_logits = linear_96 + l_self_buffers_final_logits_bias_
        linear_96 = l_self_buffers_final_logits_bias_ = None
        return (lm_logits, hidden_states_55)
