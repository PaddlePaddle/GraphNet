import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_input_ids_: torch.Tensor,
        L_self_modules_encoder_modules_embed_tokens_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_embed_positions_parameters_weight_: torch.nn.parameter.Parameter,
        L_attention_mask_: torch.Tensor,
        L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_0_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_0_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_0_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_0_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_1_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_1_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_1_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_1_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_3_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_3_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_3_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_3_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_4_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_4_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_4_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_4_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_5_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_5_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_5_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_5_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_6_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_6_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_6_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_6_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_6_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_6_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_7_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_7_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_7_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_7_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_7_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_7_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_8_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_8_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_8_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_8_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_8_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_8_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_9_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_9_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_9_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_9_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_9_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_9_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_10_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_10_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_10_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_10_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_10_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_10_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_11_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_11_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_11_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_11_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_11_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_11_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_12_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_12_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_12_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_12_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_12_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_12_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_13_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_13_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_13_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_13_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_13_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_13_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_14_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_14_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_14_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_14_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_14_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_14_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_15_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_15_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_15_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_15_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_15_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_15_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_decoder_input_ids_: torch.Tensor,
        L_self_modules_decoder_modules_embed_positions_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_6_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_6_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_6_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_6_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_6_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_6_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_7_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_7_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_7_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_7_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_7_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_7_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_8_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_8_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_8_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_8_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_8_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_8_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_9_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_9_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_9_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_9_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_9_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_9_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_10_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_10_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_10_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_10_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_10_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_10_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_11_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_11_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_11_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_11_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_11_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_11_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_12_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_12_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_12_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_12_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_12_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_12_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_13_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_13_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_13_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_13_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_13_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_13_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_14_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_14_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_14_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_14_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_14_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_14_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_15_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_15_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_15_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_15_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_15_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_15_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_input_ids_ = L_input_ids_
        l_self_modules_encoder_modules_embed_tokens_parameters_weight_ = (
            L_self_modules_encoder_modules_embed_tokens_parameters_weight_
        )
        l_self_modules_encoder_modules_embed_positions_parameters_weight_ = (
            L_self_modules_encoder_modules_embed_positions_parameters_weight_
        )
        l_attention_mask_ = L_attention_mask_
        l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_0_modules_fc1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_0_modules_fc1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_0_modules_fc1_parameters_bias_ = (
            L_self_modules_encoder_modules_layers_modules_0_modules_fc1_parameters_bias_
        )
        l_self_modules_encoder_modules_layers_modules_0_modules_fc2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_0_modules_fc2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_0_modules_fc2_parameters_bias_ = (
            L_self_modules_encoder_modules_layers_modules_0_modules_fc2_parameters_bias_
        )
        l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_1_modules_fc1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_1_modules_fc1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_1_modules_fc1_parameters_bias_ = (
            L_self_modules_encoder_modules_layers_modules_1_modules_fc1_parameters_bias_
        )
        l_self_modules_encoder_modules_layers_modules_1_modules_fc2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_1_modules_fc2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_1_modules_fc2_parameters_bias_ = (
            L_self_modules_encoder_modules_layers_modules_1_modules_fc2_parameters_bias_
        )
        l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_2_modules_fc1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_2_modules_fc1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_2_modules_fc1_parameters_bias_ = (
            L_self_modules_encoder_modules_layers_modules_2_modules_fc1_parameters_bias_
        )
        l_self_modules_encoder_modules_layers_modules_2_modules_fc2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_2_modules_fc2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_2_modules_fc2_parameters_bias_ = (
            L_self_modules_encoder_modules_layers_modules_2_modules_fc2_parameters_bias_
        )
        l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_3_modules_fc1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_3_modules_fc1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_3_modules_fc1_parameters_bias_ = (
            L_self_modules_encoder_modules_layers_modules_3_modules_fc1_parameters_bias_
        )
        l_self_modules_encoder_modules_layers_modules_3_modules_fc2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_3_modules_fc2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_3_modules_fc2_parameters_bias_ = (
            L_self_modules_encoder_modules_layers_modules_3_modules_fc2_parameters_bias_
        )
        l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_4_modules_fc1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_4_modules_fc1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_4_modules_fc1_parameters_bias_ = (
            L_self_modules_encoder_modules_layers_modules_4_modules_fc1_parameters_bias_
        )
        l_self_modules_encoder_modules_layers_modules_4_modules_fc2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_4_modules_fc2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_4_modules_fc2_parameters_bias_ = (
            L_self_modules_encoder_modules_layers_modules_4_modules_fc2_parameters_bias_
        )
        l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_5_modules_fc1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_5_modules_fc1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_5_modules_fc1_parameters_bias_ = (
            L_self_modules_encoder_modules_layers_modules_5_modules_fc1_parameters_bias_
        )
        l_self_modules_encoder_modules_layers_modules_5_modules_fc2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_5_modules_fc2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_5_modules_fc2_parameters_bias_ = (
            L_self_modules_encoder_modules_layers_modules_5_modules_fc2_parameters_bias_
        )
        l_self_modules_encoder_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_6_modules_final_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_6_modules_final_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_6_modules_final_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_6_modules_final_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_6_modules_fc1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_6_modules_fc1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_6_modules_fc1_parameters_bias_ = (
            L_self_modules_encoder_modules_layers_modules_6_modules_fc1_parameters_bias_
        )
        l_self_modules_encoder_modules_layers_modules_6_modules_fc2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_6_modules_fc2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_6_modules_fc2_parameters_bias_ = (
            L_self_modules_encoder_modules_layers_modules_6_modules_fc2_parameters_bias_
        )
        l_self_modules_encoder_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_7_modules_final_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_7_modules_final_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_7_modules_final_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_7_modules_final_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_7_modules_fc1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_7_modules_fc1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_7_modules_fc1_parameters_bias_ = (
            L_self_modules_encoder_modules_layers_modules_7_modules_fc1_parameters_bias_
        )
        l_self_modules_encoder_modules_layers_modules_7_modules_fc2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_7_modules_fc2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_7_modules_fc2_parameters_bias_ = (
            L_self_modules_encoder_modules_layers_modules_7_modules_fc2_parameters_bias_
        )
        l_self_modules_encoder_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_8_modules_final_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_8_modules_final_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_8_modules_final_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_8_modules_final_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_8_modules_fc1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_8_modules_fc1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_8_modules_fc1_parameters_bias_ = (
            L_self_modules_encoder_modules_layers_modules_8_modules_fc1_parameters_bias_
        )
        l_self_modules_encoder_modules_layers_modules_8_modules_fc2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_8_modules_fc2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_8_modules_fc2_parameters_bias_ = (
            L_self_modules_encoder_modules_layers_modules_8_modules_fc2_parameters_bias_
        )
        l_self_modules_encoder_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_9_modules_final_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_9_modules_final_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_9_modules_final_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_9_modules_final_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_9_modules_fc1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_9_modules_fc1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_9_modules_fc1_parameters_bias_ = (
            L_self_modules_encoder_modules_layers_modules_9_modules_fc1_parameters_bias_
        )
        l_self_modules_encoder_modules_layers_modules_9_modules_fc2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_9_modules_fc2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_9_modules_fc2_parameters_bias_ = (
            L_self_modules_encoder_modules_layers_modules_9_modules_fc2_parameters_bias_
        )
        l_self_modules_encoder_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_10_modules_final_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_10_modules_final_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_10_modules_final_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_10_modules_final_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_10_modules_fc1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_10_modules_fc1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_10_modules_fc1_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_10_modules_fc1_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_10_modules_fc2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_10_modules_fc2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_10_modules_fc2_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_10_modules_fc2_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_11_modules_final_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_11_modules_final_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_11_modules_final_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_11_modules_final_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_11_modules_fc1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_11_modules_fc1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_11_modules_fc1_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_11_modules_fc1_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_11_modules_fc2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_11_modules_fc2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_11_modules_fc2_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_11_modules_fc2_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_12_modules_final_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_12_modules_final_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_12_modules_final_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_12_modules_final_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_12_modules_fc1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_12_modules_fc1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_12_modules_fc1_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_12_modules_fc1_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_12_modules_fc2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_12_modules_fc2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_12_modules_fc2_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_12_modules_fc2_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_13_modules_final_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_13_modules_final_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_13_modules_final_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_13_modules_final_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_13_modules_fc1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_13_modules_fc1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_13_modules_fc1_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_13_modules_fc1_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_13_modules_fc2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_13_modules_fc2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_13_modules_fc2_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_13_modules_fc2_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_14_modules_final_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_14_modules_final_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_14_modules_final_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_14_modules_final_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_14_modules_fc1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_14_modules_fc1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_14_modules_fc1_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_14_modules_fc1_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_14_modules_fc2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_14_modules_fc2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_14_modules_fc2_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_14_modules_fc2_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_15_modules_final_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_15_modules_final_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_15_modules_final_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_15_modules_final_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_15_modules_fc1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_15_modules_fc1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_15_modules_fc1_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_15_modules_fc1_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_15_modules_fc2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_15_modules_fc2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_15_modules_fc2_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_15_modules_fc2_parameters_bias_
        l_self_modules_encoder_modules_layer_norm_parameters_weight_ = (
            L_self_modules_encoder_modules_layer_norm_parameters_weight_
        )
        l_self_modules_encoder_modules_layer_norm_parameters_bias_ = (
            L_self_modules_encoder_modules_layer_norm_parameters_bias_
        )
        l_decoder_input_ids_ = L_decoder_input_ids_
        l_self_modules_decoder_modules_embed_positions_parameters_weight_ = (
            L_self_modules_decoder_modules_embed_positions_parameters_weight_
        )
        l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_bias_ = (
            L_self_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_bias_
        )
        l_self_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_bias_ = (
            L_self_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_bias_
        )
        l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_bias_ = (
            L_self_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_bias_
        )
        l_self_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_bias_ = (
            L_self_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_bias_
        )
        l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_bias_ = (
            L_self_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_bias_
        )
        l_self_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_bias_ = (
            L_self_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_bias_
        )
        l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_bias_ = (
            L_self_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_bias_
        )
        l_self_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_bias_ = (
            L_self_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_bias_
        )
        l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_bias_ = (
            L_self_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_bias_
        )
        l_self_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_bias_ = (
            L_self_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_bias_
        )
        l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_bias_ = (
            L_self_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_bias_
        )
        l_self_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_bias_ = (
            L_self_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_bias_
        )
        l_self_modules_decoder_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_6_modules_final_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_6_modules_final_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_6_modules_final_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_6_modules_final_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_6_modules_fc1_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_6_modules_fc1_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_6_modules_fc1_parameters_bias_ = (
            L_self_modules_decoder_modules_layers_modules_6_modules_fc1_parameters_bias_
        )
        l_self_modules_decoder_modules_layers_modules_6_modules_fc2_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_6_modules_fc2_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_6_modules_fc2_parameters_bias_ = (
            L_self_modules_decoder_modules_layers_modules_6_modules_fc2_parameters_bias_
        )
        l_self_modules_decoder_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_7_modules_final_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_7_modules_final_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_7_modules_final_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_7_modules_final_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_7_modules_fc1_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_7_modules_fc1_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_7_modules_fc1_parameters_bias_ = (
            L_self_modules_decoder_modules_layers_modules_7_modules_fc1_parameters_bias_
        )
        l_self_modules_decoder_modules_layers_modules_7_modules_fc2_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_7_modules_fc2_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_7_modules_fc2_parameters_bias_ = (
            L_self_modules_decoder_modules_layers_modules_7_modules_fc2_parameters_bias_
        )
        l_self_modules_decoder_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_8_modules_final_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_8_modules_final_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_8_modules_final_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_8_modules_final_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_8_modules_fc1_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_8_modules_fc1_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_8_modules_fc1_parameters_bias_ = (
            L_self_modules_decoder_modules_layers_modules_8_modules_fc1_parameters_bias_
        )
        l_self_modules_decoder_modules_layers_modules_8_modules_fc2_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_8_modules_fc2_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_8_modules_fc2_parameters_bias_ = (
            L_self_modules_decoder_modules_layers_modules_8_modules_fc2_parameters_bias_
        )
        l_self_modules_decoder_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_9_modules_final_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_9_modules_final_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_9_modules_final_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_9_modules_final_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_9_modules_fc1_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_9_modules_fc1_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_9_modules_fc1_parameters_bias_ = (
            L_self_modules_decoder_modules_layers_modules_9_modules_fc1_parameters_bias_
        )
        l_self_modules_decoder_modules_layers_modules_9_modules_fc2_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_9_modules_fc2_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_9_modules_fc2_parameters_bias_ = (
            L_self_modules_decoder_modules_layers_modules_9_modules_fc2_parameters_bias_
        )
        l_self_modules_decoder_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_10_modules_final_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_10_modules_final_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_10_modules_final_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_10_modules_final_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_10_modules_fc1_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_10_modules_fc1_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_10_modules_fc1_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_10_modules_fc1_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_10_modules_fc2_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_10_modules_fc2_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_10_modules_fc2_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_10_modules_fc2_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_11_modules_final_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_11_modules_final_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_11_modules_final_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_11_modules_final_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_11_modules_fc1_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_11_modules_fc1_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_11_modules_fc1_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_11_modules_fc1_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_11_modules_fc2_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_11_modules_fc2_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_11_modules_fc2_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_11_modules_fc2_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_12_modules_final_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_12_modules_final_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_12_modules_final_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_12_modules_final_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_12_modules_fc1_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_12_modules_fc1_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_12_modules_fc1_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_12_modules_fc1_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_12_modules_fc2_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_12_modules_fc2_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_12_modules_fc2_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_12_modules_fc2_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_13_modules_final_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_13_modules_final_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_13_modules_final_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_13_modules_final_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_13_modules_fc1_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_13_modules_fc1_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_13_modules_fc1_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_13_modules_fc1_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_13_modules_fc2_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_13_modules_fc2_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_13_modules_fc2_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_13_modules_fc2_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_14_modules_final_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_14_modules_final_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_14_modules_final_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_14_modules_final_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_14_modules_fc1_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_14_modules_fc1_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_14_modules_fc1_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_14_modules_fc1_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_14_modules_fc2_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_14_modules_fc2_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_14_modules_fc2_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_14_modules_fc2_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_15_modules_final_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_15_modules_final_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_15_modules_final_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_15_modules_final_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_15_modules_fc1_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_15_modules_fc1_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_15_modules_fc1_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_15_modules_fc1_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_15_modules_fc2_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_15_modules_fc2_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_15_modules_fc2_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_15_modules_fc2_parameters_bias_
        l_self_modules_decoder_modules_layer_norm_parameters_weight_ = (
            L_self_modules_decoder_modules_layer_norm_parameters_weight_
        )
        l_self_modules_decoder_modules_layer_norm_parameters_bias_ = (
            L_self_modules_decoder_modules_layer_norm_parameters_bias_
        )
        input_ids = l_input_ids_.view(-1, 10)
        l_input_ids_ = None
        embedding = torch.nn.functional.embedding(
            input_ids,
            l_self_modules_encoder_modules_embed_tokens_parameters_weight_,
            0,
            None,
            2.0,
            False,
            False,
        )
        input_ids = None
        inputs_embeds = embedding * 32.0
        embedding = None
        position_ids = torch.arange(
            0, 10, dtype=torch.int64, device=device(type="cuda", index=0)
        )
        embed_pos = torch.nn.functional.embedding(
            position_ids,
            l_self_modules_encoder_modules_embed_positions_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        position_ids = (
            l_self_modules_encoder_modules_embed_positions_parameters_weight_
        ) = None
        hidden_states = inputs_embeds + embed_pos
        inputs_embeds = embed_pos = None
        hidden_states_1 = torch.nn.functional.dropout(
            hidden_states, p=0.1, training=False
        )
        hidden_states = None
        getitem = l_attention_mask_[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        expand = getitem.expand(1, 1, 10, 10)
        getitem = None
        expanded_mask = expand.to(torch.bfloat16)
        expand = None
        tensor = torch.tensor(1.0, dtype=torch.bfloat16)
        inverted_mask = tensor - expanded_mask
        tensor = expanded_mask = None
        to_1 = inverted_mask.to(torch.bool)
        attention_mask = inverted_mask.masked_fill(to_1, -3.3895313892515355e38)
        inverted_mask = to_1 = None
        hidden_states_2 = torch.nn.functional.layer_norm(
            hidden_states_1,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_1 = linear.view(1, 10, -1, 64)
        linear = None
        query_states = view_1.transpose(1, 2)
        view_1 = None
        key_states = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_2 = l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_2 = key_states.view(1, 10, -1, 64)
        key_states = None
        key_states_1 = view_2.transpose(1, 2)
        view_2 = None
        view_3 = value_states.view(1, 10, -1, 64)
        value_states = None
        value_states_1 = view_3.transpose(1, 2)
        view_3 = None
        attention_mask_1 = attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
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
        reshape = attn_output_1.reshape(1, 10, -1)
        attn_output_1 = None
        attn_output_2 = reshape.contiguous()
        reshape = None
        attn_output_3 = torch._C._nn.linear(
            attn_output_2,
            l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_2 = l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_3 = torch.nn.functional.dropout(
            attn_output_3, p=0.1, training=False
        )
        attn_output_3 = None
        hidden_states_4 = hidden_states_1 + hidden_states_3
        hidden_states_1 = hidden_states_3 = None
        hidden_states_5 = torch.nn.functional.layer_norm(
            hidden_states_4,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_ = (None)
        linear_4 = torch._C._nn.linear(
            hidden_states_5,
            l_self_modules_encoder_modules_layers_modules_0_modules_fc1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_0_modules_fc1_parameters_bias_,
        )
        hidden_states_5 = l_self_modules_encoder_modules_layers_modules_0_modules_fc1_parameters_weight_ = (
            l_self_modules_encoder_modules_layers_modules_0_modules_fc1_parameters_bias_
        ) = None
        hidden_states_6 = torch.nn.functional.relu(linear_4, inplace=False)
        linear_4 = None
        hidden_states_7 = torch.nn.functional.dropout(
            hidden_states_6, p=0.1, training=False
        )
        hidden_states_6 = None
        hidden_states_8 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_encoder_modules_layers_modules_0_modules_fc2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_0_modules_fc2_parameters_bias_,
        )
        hidden_states_7 = l_self_modules_encoder_modules_layers_modules_0_modules_fc2_parameters_weight_ = (
            l_self_modules_encoder_modules_layers_modules_0_modules_fc2_parameters_bias_
        ) = None
        hidden_states_9 = torch.nn.functional.dropout(
            hidden_states_8, p=0.1, training=False
        )
        hidden_states_8 = None
        hidden_states_10 = hidden_states_4 + hidden_states_9
        hidden_states_4 = hidden_states_9 = None
        hidden_states_11 = torch.nn.functional.layer_norm(
            hidden_states_10,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_6 = torch._C._nn.linear(
            hidden_states_11,
            l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_4 = linear_6.view(1, 10, -1, 64)
        linear_6 = None
        query_states_1 = view_4.transpose(1, 2)
        view_4 = None
        key_states_2 = torch._C._nn.linear(
            hidden_states_11,
            l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_2 = torch._C._nn.linear(
            hidden_states_11,
            l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_11 = l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_5 = key_states_2.view(1, 10, -1, 64)
        key_states_2 = None
        key_states_3 = view_5.transpose(1, 2)
        view_5 = None
        view_6 = value_states_2.view(1, 10, -1, 64)
        value_states_2 = None
        value_states_3 = view_6.transpose(1, 2)
        view_6 = None
        attention_mask_2 = attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
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
        reshape_1 = attn_output_5.reshape(1, 10, -1)
        attn_output_5 = None
        attn_output_6 = reshape_1.contiguous()
        reshape_1 = None
        attn_output_7 = torch._C._nn.linear(
            attn_output_6,
            l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_6 = l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_12 = torch.nn.functional.dropout(
            attn_output_7, p=0.1, training=False
        )
        attn_output_7 = None
        hidden_states_13 = hidden_states_10 + hidden_states_12
        hidden_states_10 = hidden_states_12 = None
        hidden_states_14 = torch.nn.functional.layer_norm(
            hidden_states_13,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_ = (None)
        linear_10 = torch._C._nn.linear(
            hidden_states_14,
            l_self_modules_encoder_modules_layers_modules_1_modules_fc1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_1_modules_fc1_parameters_bias_,
        )
        hidden_states_14 = l_self_modules_encoder_modules_layers_modules_1_modules_fc1_parameters_weight_ = (
            l_self_modules_encoder_modules_layers_modules_1_modules_fc1_parameters_bias_
        ) = None
        hidden_states_15 = torch.nn.functional.relu(linear_10, inplace=False)
        linear_10 = None
        hidden_states_16 = torch.nn.functional.dropout(
            hidden_states_15, p=0.1, training=False
        )
        hidden_states_15 = None
        hidden_states_17 = torch._C._nn.linear(
            hidden_states_16,
            l_self_modules_encoder_modules_layers_modules_1_modules_fc2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_1_modules_fc2_parameters_bias_,
        )
        hidden_states_16 = l_self_modules_encoder_modules_layers_modules_1_modules_fc2_parameters_weight_ = (
            l_self_modules_encoder_modules_layers_modules_1_modules_fc2_parameters_bias_
        ) = None
        hidden_states_18 = torch.nn.functional.dropout(
            hidden_states_17, p=0.1, training=False
        )
        hidden_states_17 = None
        hidden_states_19 = hidden_states_13 + hidden_states_18
        hidden_states_13 = hidden_states_18 = None
        hidden_states_20 = torch.nn.functional.layer_norm(
            hidden_states_19,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_12 = torch._C._nn.linear(
            hidden_states_20,
            l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_7 = linear_12.view(1, 10, -1, 64)
        linear_12 = None
        query_states_2 = view_7.transpose(1, 2)
        view_7 = None
        key_states_4 = torch._C._nn.linear(
            hidden_states_20,
            l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_4 = torch._C._nn.linear(
            hidden_states_20,
            l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_20 = l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_8 = key_states_4.view(1, 10, -1, 64)
        key_states_4 = None
        key_states_5 = view_8.transpose(1, 2)
        view_8 = None
        view_9 = value_states_4.view(1, 10, -1, 64)
        value_states_4 = None
        value_states_5 = view_9.transpose(1, 2)
        view_9 = None
        attention_mask_3 = attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
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
        reshape_2 = attn_output_9.reshape(1, 10, -1)
        attn_output_9 = None
        attn_output_10 = reshape_2.contiguous()
        reshape_2 = None
        attn_output_11 = torch._C._nn.linear(
            attn_output_10,
            l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_10 = l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_21 = torch.nn.functional.dropout(
            attn_output_11, p=0.1, training=False
        )
        attn_output_11 = None
        hidden_states_22 = hidden_states_19 + hidden_states_21
        hidden_states_19 = hidden_states_21 = None
        hidden_states_23 = torch.nn.functional.layer_norm(
            hidden_states_22,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_ = (None)
        linear_16 = torch._C._nn.linear(
            hidden_states_23,
            l_self_modules_encoder_modules_layers_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_2_modules_fc1_parameters_bias_,
        )
        hidden_states_23 = l_self_modules_encoder_modules_layers_modules_2_modules_fc1_parameters_weight_ = (
            l_self_modules_encoder_modules_layers_modules_2_modules_fc1_parameters_bias_
        ) = None
        hidden_states_24 = torch.nn.functional.relu(linear_16, inplace=False)
        linear_16 = None
        hidden_states_25 = torch.nn.functional.dropout(
            hidden_states_24, p=0.1, training=False
        )
        hidden_states_24 = None
        hidden_states_26 = torch._C._nn.linear(
            hidden_states_25,
            l_self_modules_encoder_modules_layers_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_2_modules_fc2_parameters_bias_,
        )
        hidden_states_25 = l_self_modules_encoder_modules_layers_modules_2_modules_fc2_parameters_weight_ = (
            l_self_modules_encoder_modules_layers_modules_2_modules_fc2_parameters_bias_
        ) = None
        hidden_states_27 = torch.nn.functional.dropout(
            hidden_states_26, p=0.1, training=False
        )
        hidden_states_26 = None
        hidden_states_28 = hidden_states_22 + hidden_states_27
        hidden_states_22 = hidden_states_27 = None
        hidden_states_29 = torch.nn.functional.layer_norm(
            hidden_states_28,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_18 = torch._C._nn.linear(
            hidden_states_29,
            l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_10 = linear_18.view(1, 10, -1, 64)
        linear_18 = None
        query_states_3 = view_10.transpose(1, 2)
        view_10 = None
        key_states_6 = torch._C._nn.linear(
            hidden_states_29,
            l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_6 = torch._C._nn.linear(
            hidden_states_29,
            l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_29 = l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_11 = key_states_6.view(1, 10, -1, 64)
        key_states_6 = None
        key_states_7 = view_11.transpose(1, 2)
        view_11 = None
        view_12 = value_states_6.view(1, 10, -1, 64)
        value_states_6 = None
        value_states_7 = view_12.transpose(1, 2)
        view_12 = None
        attention_mask_4 = attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
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
        reshape_3 = attn_output_13.reshape(1, 10, -1)
        attn_output_13 = None
        attn_output_14 = reshape_3.contiguous()
        reshape_3 = None
        attn_output_15 = torch._C._nn.linear(
            attn_output_14,
            l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_14 = l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_30 = torch.nn.functional.dropout(
            attn_output_15, p=0.1, training=False
        )
        attn_output_15 = None
        hidden_states_31 = hidden_states_28 + hidden_states_30
        hidden_states_28 = hidden_states_30 = None
        hidden_states_32 = torch.nn.functional.layer_norm(
            hidden_states_31,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_ = (None)
        linear_22 = torch._C._nn.linear(
            hidden_states_32,
            l_self_modules_encoder_modules_layers_modules_3_modules_fc1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_3_modules_fc1_parameters_bias_,
        )
        hidden_states_32 = l_self_modules_encoder_modules_layers_modules_3_modules_fc1_parameters_weight_ = (
            l_self_modules_encoder_modules_layers_modules_3_modules_fc1_parameters_bias_
        ) = None
        hidden_states_33 = torch.nn.functional.relu(linear_22, inplace=False)
        linear_22 = None
        hidden_states_34 = torch.nn.functional.dropout(
            hidden_states_33, p=0.1, training=False
        )
        hidden_states_33 = None
        hidden_states_35 = torch._C._nn.linear(
            hidden_states_34,
            l_self_modules_encoder_modules_layers_modules_3_modules_fc2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_3_modules_fc2_parameters_bias_,
        )
        hidden_states_34 = l_self_modules_encoder_modules_layers_modules_3_modules_fc2_parameters_weight_ = (
            l_self_modules_encoder_modules_layers_modules_3_modules_fc2_parameters_bias_
        ) = None
        hidden_states_36 = torch.nn.functional.dropout(
            hidden_states_35, p=0.1, training=False
        )
        hidden_states_35 = None
        hidden_states_37 = hidden_states_31 + hidden_states_36
        hidden_states_31 = hidden_states_36 = None
        hidden_states_38 = torch.nn.functional.layer_norm(
            hidden_states_37,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_24 = torch._C._nn.linear(
            hidden_states_38,
            l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_13 = linear_24.view(1, 10, -1, 64)
        linear_24 = None
        query_states_4 = view_13.transpose(1, 2)
        view_13 = None
        key_states_8 = torch._C._nn.linear(
            hidden_states_38,
            l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_8 = torch._C._nn.linear(
            hidden_states_38,
            l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_38 = l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_14 = key_states_8.view(1, 10, -1, 64)
        key_states_8 = None
        key_states_9 = view_14.transpose(1, 2)
        view_14 = None
        view_15 = value_states_8.view(1, 10, -1, 64)
        value_states_8 = None
        value_states_9 = view_15.transpose(1, 2)
        view_15 = None
        attention_mask_5 = attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
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
        reshape_4 = attn_output_17.reshape(1, 10, -1)
        attn_output_17 = None
        attn_output_18 = reshape_4.contiguous()
        reshape_4 = None
        attn_output_19 = torch._C._nn.linear(
            attn_output_18,
            l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_18 = l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_39 = torch.nn.functional.dropout(
            attn_output_19, p=0.1, training=False
        )
        attn_output_19 = None
        hidden_states_40 = hidden_states_37 + hidden_states_39
        hidden_states_37 = hidden_states_39 = None
        hidden_states_41 = torch.nn.functional.layer_norm(
            hidden_states_40,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_ = (None)
        linear_28 = torch._C._nn.linear(
            hidden_states_41,
            l_self_modules_encoder_modules_layers_modules_4_modules_fc1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_4_modules_fc1_parameters_bias_,
        )
        hidden_states_41 = l_self_modules_encoder_modules_layers_modules_4_modules_fc1_parameters_weight_ = (
            l_self_modules_encoder_modules_layers_modules_4_modules_fc1_parameters_bias_
        ) = None
        hidden_states_42 = torch.nn.functional.relu(linear_28, inplace=False)
        linear_28 = None
        hidden_states_43 = torch.nn.functional.dropout(
            hidden_states_42, p=0.1, training=False
        )
        hidden_states_42 = None
        hidden_states_44 = torch._C._nn.linear(
            hidden_states_43,
            l_self_modules_encoder_modules_layers_modules_4_modules_fc2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_4_modules_fc2_parameters_bias_,
        )
        hidden_states_43 = l_self_modules_encoder_modules_layers_modules_4_modules_fc2_parameters_weight_ = (
            l_self_modules_encoder_modules_layers_modules_4_modules_fc2_parameters_bias_
        ) = None
        hidden_states_45 = torch.nn.functional.dropout(
            hidden_states_44, p=0.1, training=False
        )
        hidden_states_44 = None
        hidden_states_46 = hidden_states_40 + hidden_states_45
        hidden_states_40 = hidden_states_45 = None
        hidden_states_47 = torch.nn.functional.layer_norm(
            hidden_states_46,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_30 = torch._C._nn.linear(
            hidden_states_47,
            l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_16 = linear_30.view(1, 10, -1, 64)
        linear_30 = None
        query_states_5 = view_16.transpose(1, 2)
        view_16 = None
        key_states_10 = torch._C._nn.linear(
            hidden_states_47,
            l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_10 = torch._C._nn.linear(
            hidden_states_47,
            l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_47 = l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_17 = key_states_10.view(1, 10, -1, 64)
        key_states_10 = None
        key_states_11 = view_17.transpose(1, 2)
        view_17 = None
        view_18 = value_states_10.view(1, 10, -1, 64)
        value_states_10 = None
        value_states_11 = view_18.transpose(1, 2)
        view_18 = None
        attention_mask_6 = attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
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
        reshape_5 = attn_output_21.reshape(1, 10, -1)
        attn_output_21 = None
        attn_output_22 = reshape_5.contiguous()
        reshape_5 = None
        attn_output_23 = torch._C._nn.linear(
            attn_output_22,
            l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_22 = l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_48 = torch.nn.functional.dropout(
            attn_output_23, p=0.1, training=False
        )
        attn_output_23 = None
        hidden_states_49 = hidden_states_46 + hidden_states_48
        hidden_states_46 = hidden_states_48 = None
        hidden_states_50 = torch.nn.functional.layer_norm(
            hidden_states_49,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_ = (None)
        linear_34 = torch._C._nn.linear(
            hidden_states_50,
            l_self_modules_encoder_modules_layers_modules_5_modules_fc1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_5_modules_fc1_parameters_bias_,
        )
        hidden_states_50 = l_self_modules_encoder_modules_layers_modules_5_modules_fc1_parameters_weight_ = (
            l_self_modules_encoder_modules_layers_modules_5_modules_fc1_parameters_bias_
        ) = None
        hidden_states_51 = torch.nn.functional.relu(linear_34, inplace=False)
        linear_34 = None
        hidden_states_52 = torch.nn.functional.dropout(
            hidden_states_51, p=0.1, training=False
        )
        hidden_states_51 = None
        hidden_states_53 = torch._C._nn.linear(
            hidden_states_52,
            l_self_modules_encoder_modules_layers_modules_5_modules_fc2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_5_modules_fc2_parameters_bias_,
        )
        hidden_states_52 = l_self_modules_encoder_modules_layers_modules_5_modules_fc2_parameters_weight_ = (
            l_self_modules_encoder_modules_layers_modules_5_modules_fc2_parameters_bias_
        ) = None
        hidden_states_54 = torch.nn.functional.dropout(
            hidden_states_53, p=0.1, training=False
        )
        hidden_states_53 = None
        hidden_states_55 = hidden_states_49 + hidden_states_54
        hidden_states_49 = hidden_states_54 = None
        hidden_states_56 = torch.nn.functional.layer_norm(
            hidden_states_55,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_36 = torch._C._nn.linear(
            hidden_states_56,
            l_self_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_19 = linear_36.view(1, 10, -1, 64)
        linear_36 = None
        query_states_6 = view_19.transpose(1, 2)
        view_19 = None
        key_states_12 = torch._C._nn.linear(
            hidden_states_56,
            l_self_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_12 = torch._C._nn.linear(
            hidden_states_56,
            l_self_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_56 = l_self_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_20 = key_states_12.view(1, 10, -1, 64)
        key_states_12 = None
        key_states_13 = view_20.transpose(1, 2)
        view_20 = None
        view_21 = value_states_12.view(1, 10, -1, 64)
        value_states_12 = None
        value_states_13 = view_21.transpose(1, 2)
        view_21 = None
        attention_mask_7 = attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
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
        reshape_6 = attn_output_25.reshape(1, 10, -1)
        attn_output_25 = None
        attn_output_26 = reshape_6.contiguous()
        reshape_6 = None
        attn_output_27 = torch._C._nn.linear(
            attn_output_26,
            l_self_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_26 = l_self_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_57 = torch.nn.functional.dropout(
            attn_output_27, p=0.1, training=False
        )
        attn_output_27 = None
        hidden_states_58 = hidden_states_55 + hidden_states_57
        hidden_states_55 = hidden_states_57 = None
        hidden_states_59 = torch.nn.functional.layer_norm(
            hidden_states_58,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_6_modules_final_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_6_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layers_modules_6_modules_final_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_6_modules_final_layer_norm_parameters_bias_ = (None)
        linear_40 = torch._C._nn.linear(
            hidden_states_59,
            l_self_modules_encoder_modules_layers_modules_6_modules_fc1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_6_modules_fc1_parameters_bias_,
        )
        hidden_states_59 = l_self_modules_encoder_modules_layers_modules_6_modules_fc1_parameters_weight_ = (
            l_self_modules_encoder_modules_layers_modules_6_modules_fc1_parameters_bias_
        ) = None
        hidden_states_60 = torch.nn.functional.relu(linear_40, inplace=False)
        linear_40 = None
        hidden_states_61 = torch.nn.functional.dropout(
            hidden_states_60, p=0.1, training=False
        )
        hidden_states_60 = None
        hidden_states_62 = torch._C._nn.linear(
            hidden_states_61,
            l_self_modules_encoder_modules_layers_modules_6_modules_fc2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_6_modules_fc2_parameters_bias_,
        )
        hidden_states_61 = l_self_modules_encoder_modules_layers_modules_6_modules_fc2_parameters_weight_ = (
            l_self_modules_encoder_modules_layers_modules_6_modules_fc2_parameters_bias_
        ) = None
        hidden_states_63 = torch.nn.functional.dropout(
            hidden_states_62, p=0.1, training=False
        )
        hidden_states_62 = None
        hidden_states_64 = hidden_states_58 + hidden_states_63
        hidden_states_58 = hidden_states_63 = None
        hidden_states_65 = torch.nn.functional.layer_norm(
            hidden_states_64,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_42 = torch._C._nn.linear(
            hidden_states_65,
            l_self_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_22 = linear_42.view(1, 10, -1, 64)
        linear_42 = None
        query_states_7 = view_22.transpose(1, 2)
        view_22 = None
        key_states_14 = torch._C._nn.linear(
            hidden_states_65,
            l_self_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_14 = torch._C._nn.linear(
            hidden_states_65,
            l_self_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_65 = l_self_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_23 = key_states_14.view(1, 10, -1, 64)
        key_states_14 = None
        key_states_15 = view_23.transpose(1, 2)
        view_23 = None
        view_24 = value_states_14.view(1, 10, -1, 64)
        value_states_14 = None
        value_states_15 = view_24.transpose(1, 2)
        view_24 = None
        attention_mask_8 = attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
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
        reshape_7 = attn_output_29.reshape(1, 10, -1)
        attn_output_29 = None
        attn_output_30 = reshape_7.contiguous()
        reshape_7 = None
        attn_output_31 = torch._C._nn.linear(
            attn_output_30,
            l_self_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_30 = l_self_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_66 = torch.nn.functional.dropout(
            attn_output_31, p=0.1, training=False
        )
        attn_output_31 = None
        hidden_states_67 = hidden_states_64 + hidden_states_66
        hidden_states_64 = hidden_states_66 = None
        hidden_states_68 = torch.nn.functional.layer_norm(
            hidden_states_67,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_7_modules_final_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_7_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layers_modules_7_modules_final_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_7_modules_final_layer_norm_parameters_bias_ = (None)
        linear_46 = torch._C._nn.linear(
            hidden_states_68,
            l_self_modules_encoder_modules_layers_modules_7_modules_fc1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_7_modules_fc1_parameters_bias_,
        )
        hidden_states_68 = l_self_modules_encoder_modules_layers_modules_7_modules_fc1_parameters_weight_ = (
            l_self_modules_encoder_modules_layers_modules_7_modules_fc1_parameters_bias_
        ) = None
        hidden_states_69 = torch.nn.functional.relu(linear_46, inplace=False)
        linear_46 = None
        hidden_states_70 = torch.nn.functional.dropout(
            hidden_states_69, p=0.1, training=False
        )
        hidden_states_69 = None
        hidden_states_71 = torch._C._nn.linear(
            hidden_states_70,
            l_self_modules_encoder_modules_layers_modules_7_modules_fc2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_7_modules_fc2_parameters_bias_,
        )
        hidden_states_70 = l_self_modules_encoder_modules_layers_modules_7_modules_fc2_parameters_weight_ = (
            l_self_modules_encoder_modules_layers_modules_7_modules_fc2_parameters_bias_
        ) = None
        hidden_states_72 = torch.nn.functional.dropout(
            hidden_states_71, p=0.1, training=False
        )
        hidden_states_71 = None
        hidden_states_73 = hidden_states_67 + hidden_states_72
        hidden_states_67 = hidden_states_72 = None
        hidden_states_74 = torch.nn.functional.layer_norm(
            hidden_states_73,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_48 = torch._C._nn.linear(
            hidden_states_74,
            l_self_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_25 = linear_48.view(1, 10, -1, 64)
        linear_48 = None
        query_states_8 = view_25.transpose(1, 2)
        view_25 = None
        key_states_16 = torch._C._nn.linear(
            hidden_states_74,
            l_self_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_16 = torch._C._nn.linear(
            hidden_states_74,
            l_self_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_74 = l_self_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_26 = key_states_16.view(1, 10, -1, 64)
        key_states_16 = None
        key_states_17 = view_26.transpose(1, 2)
        view_26 = None
        view_27 = value_states_16.view(1, 10, -1, 64)
        value_states_16 = None
        value_states_17 = view_27.transpose(1, 2)
        view_27 = None
        attention_mask_9 = attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
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
        reshape_8 = attn_output_33.reshape(1, 10, -1)
        attn_output_33 = None
        attn_output_34 = reshape_8.contiguous()
        reshape_8 = None
        attn_output_35 = torch._C._nn.linear(
            attn_output_34,
            l_self_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_34 = l_self_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_75 = torch.nn.functional.dropout(
            attn_output_35, p=0.1, training=False
        )
        attn_output_35 = None
        hidden_states_76 = hidden_states_73 + hidden_states_75
        hidden_states_73 = hidden_states_75 = None
        hidden_states_77 = torch.nn.functional.layer_norm(
            hidden_states_76,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_8_modules_final_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_8_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layers_modules_8_modules_final_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_8_modules_final_layer_norm_parameters_bias_ = (None)
        linear_52 = torch._C._nn.linear(
            hidden_states_77,
            l_self_modules_encoder_modules_layers_modules_8_modules_fc1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_8_modules_fc1_parameters_bias_,
        )
        hidden_states_77 = l_self_modules_encoder_modules_layers_modules_8_modules_fc1_parameters_weight_ = (
            l_self_modules_encoder_modules_layers_modules_8_modules_fc1_parameters_bias_
        ) = None
        hidden_states_78 = torch.nn.functional.relu(linear_52, inplace=False)
        linear_52 = None
        hidden_states_79 = torch.nn.functional.dropout(
            hidden_states_78, p=0.1, training=False
        )
        hidden_states_78 = None
        hidden_states_80 = torch._C._nn.linear(
            hidden_states_79,
            l_self_modules_encoder_modules_layers_modules_8_modules_fc2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_8_modules_fc2_parameters_bias_,
        )
        hidden_states_79 = l_self_modules_encoder_modules_layers_modules_8_modules_fc2_parameters_weight_ = (
            l_self_modules_encoder_modules_layers_modules_8_modules_fc2_parameters_bias_
        ) = None
        hidden_states_81 = torch.nn.functional.dropout(
            hidden_states_80, p=0.1, training=False
        )
        hidden_states_80 = None
        hidden_states_82 = hidden_states_76 + hidden_states_81
        hidden_states_76 = hidden_states_81 = None
        hidden_states_83 = torch.nn.functional.layer_norm(
            hidden_states_82,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_54 = torch._C._nn.linear(
            hidden_states_83,
            l_self_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_28 = linear_54.view(1, 10, -1, 64)
        linear_54 = None
        query_states_9 = view_28.transpose(1, 2)
        view_28 = None
        key_states_18 = torch._C._nn.linear(
            hidden_states_83,
            l_self_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_18 = torch._C._nn.linear(
            hidden_states_83,
            l_self_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_83 = l_self_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_29 = key_states_18.view(1, 10, -1, 64)
        key_states_18 = None
        key_states_19 = view_29.transpose(1, 2)
        view_29 = None
        view_30 = value_states_18.view(1, 10, -1, 64)
        value_states_18 = None
        value_states_19 = view_30.transpose(1, 2)
        view_30 = None
        attention_mask_10 = attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
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
        reshape_9 = attn_output_37.reshape(1, 10, -1)
        attn_output_37 = None
        attn_output_38 = reshape_9.contiguous()
        reshape_9 = None
        attn_output_39 = torch._C._nn.linear(
            attn_output_38,
            l_self_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_38 = l_self_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_84 = torch.nn.functional.dropout(
            attn_output_39, p=0.1, training=False
        )
        attn_output_39 = None
        hidden_states_85 = hidden_states_82 + hidden_states_84
        hidden_states_82 = hidden_states_84 = None
        hidden_states_86 = torch.nn.functional.layer_norm(
            hidden_states_85,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_9_modules_final_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_9_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layers_modules_9_modules_final_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_9_modules_final_layer_norm_parameters_bias_ = (None)
        linear_58 = torch._C._nn.linear(
            hidden_states_86,
            l_self_modules_encoder_modules_layers_modules_9_modules_fc1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_9_modules_fc1_parameters_bias_,
        )
        hidden_states_86 = l_self_modules_encoder_modules_layers_modules_9_modules_fc1_parameters_weight_ = (
            l_self_modules_encoder_modules_layers_modules_9_modules_fc1_parameters_bias_
        ) = None
        hidden_states_87 = torch.nn.functional.relu(linear_58, inplace=False)
        linear_58 = None
        hidden_states_88 = torch.nn.functional.dropout(
            hidden_states_87, p=0.1, training=False
        )
        hidden_states_87 = None
        hidden_states_89 = torch._C._nn.linear(
            hidden_states_88,
            l_self_modules_encoder_modules_layers_modules_9_modules_fc2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_9_modules_fc2_parameters_bias_,
        )
        hidden_states_88 = l_self_modules_encoder_modules_layers_modules_9_modules_fc2_parameters_weight_ = (
            l_self_modules_encoder_modules_layers_modules_9_modules_fc2_parameters_bias_
        ) = None
        hidden_states_90 = torch.nn.functional.dropout(
            hidden_states_89, p=0.1, training=False
        )
        hidden_states_89 = None
        hidden_states_91 = hidden_states_85 + hidden_states_90
        hidden_states_85 = hidden_states_90 = None
        hidden_states_92 = torch.nn.functional.layer_norm(
            hidden_states_91,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_60 = torch._C._nn.linear(
            hidden_states_92,
            l_self_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_31 = linear_60.view(1, 10, -1, 64)
        linear_60 = None
        query_states_10 = view_31.transpose(1, 2)
        view_31 = None
        key_states_20 = torch._C._nn.linear(
            hidden_states_92,
            l_self_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_20 = torch._C._nn.linear(
            hidden_states_92,
            l_self_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_92 = l_self_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_32 = key_states_20.view(1, 10, -1, 64)
        key_states_20 = None
        key_states_21 = view_32.transpose(1, 2)
        view_32 = None
        view_33 = value_states_20.view(1, 10, -1, 64)
        value_states_20 = None
        value_states_21 = view_33.transpose(1, 2)
        view_33 = None
        attention_mask_11 = attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
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
        reshape_10 = attn_output_41.reshape(1, 10, -1)
        attn_output_41 = None
        attn_output_42 = reshape_10.contiguous()
        reshape_10 = None
        attn_output_43 = torch._C._nn.linear(
            attn_output_42,
            l_self_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_42 = l_self_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_93 = torch.nn.functional.dropout(
            attn_output_43, p=0.1, training=False
        )
        attn_output_43 = None
        hidden_states_94 = hidden_states_91 + hidden_states_93
        hidden_states_91 = hidden_states_93 = None
        hidden_states_95 = torch.nn.functional.layer_norm(
            hidden_states_94,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_10_modules_final_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_10_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layers_modules_10_modules_final_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_10_modules_final_layer_norm_parameters_bias_ = (None)
        linear_64 = torch._C._nn.linear(
            hidden_states_95,
            l_self_modules_encoder_modules_layers_modules_10_modules_fc1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_10_modules_fc1_parameters_bias_,
        )
        hidden_states_95 = l_self_modules_encoder_modules_layers_modules_10_modules_fc1_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_10_modules_fc1_parameters_bias_ = (None)
        hidden_states_96 = torch.nn.functional.relu(linear_64, inplace=False)
        linear_64 = None
        hidden_states_97 = torch.nn.functional.dropout(
            hidden_states_96, p=0.1, training=False
        )
        hidden_states_96 = None
        hidden_states_98 = torch._C._nn.linear(
            hidden_states_97,
            l_self_modules_encoder_modules_layers_modules_10_modules_fc2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_10_modules_fc2_parameters_bias_,
        )
        hidden_states_97 = l_self_modules_encoder_modules_layers_modules_10_modules_fc2_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_10_modules_fc2_parameters_bias_ = (None)
        hidden_states_99 = torch.nn.functional.dropout(
            hidden_states_98, p=0.1, training=False
        )
        hidden_states_98 = None
        hidden_states_100 = hidden_states_94 + hidden_states_99
        hidden_states_94 = hidden_states_99 = None
        hidden_states_101 = torch.nn.functional.layer_norm(
            hidden_states_100,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_66 = torch._C._nn.linear(
            hidden_states_101,
            l_self_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_34 = linear_66.view(1, 10, -1, 64)
        linear_66 = None
        query_states_11 = view_34.transpose(1, 2)
        view_34 = None
        key_states_22 = torch._C._nn.linear(
            hidden_states_101,
            l_self_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_22 = torch._C._nn.linear(
            hidden_states_101,
            l_self_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_101 = l_self_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_35 = key_states_22.view(1, 10, -1, 64)
        key_states_22 = None
        key_states_23 = view_35.transpose(1, 2)
        view_35 = None
        view_36 = value_states_22.view(1, 10, -1, 64)
        value_states_22 = None
        value_states_23 = view_36.transpose(1, 2)
        view_36 = None
        attention_mask_12 = attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
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
        reshape_11 = attn_output_45.reshape(1, 10, -1)
        attn_output_45 = None
        attn_output_46 = reshape_11.contiguous()
        reshape_11 = None
        attn_output_47 = torch._C._nn.linear(
            attn_output_46,
            l_self_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_46 = l_self_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_102 = torch.nn.functional.dropout(
            attn_output_47, p=0.1, training=False
        )
        attn_output_47 = None
        hidden_states_103 = hidden_states_100 + hidden_states_102
        hidden_states_100 = hidden_states_102 = None
        hidden_states_104 = torch.nn.functional.layer_norm(
            hidden_states_103,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_11_modules_final_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_11_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layers_modules_11_modules_final_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_11_modules_final_layer_norm_parameters_bias_ = (None)
        linear_70 = torch._C._nn.linear(
            hidden_states_104,
            l_self_modules_encoder_modules_layers_modules_11_modules_fc1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_11_modules_fc1_parameters_bias_,
        )
        hidden_states_104 = l_self_modules_encoder_modules_layers_modules_11_modules_fc1_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_11_modules_fc1_parameters_bias_ = (None)
        hidden_states_105 = torch.nn.functional.relu(linear_70, inplace=False)
        linear_70 = None
        hidden_states_106 = torch.nn.functional.dropout(
            hidden_states_105, p=0.1, training=False
        )
        hidden_states_105 = None
        hidden_states_107 = torch._C._nn.linear(
            hidden_states_106,
            l_self_modules_encoder_modules_layers_modules_11_modules_fc2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_11_modules_fc2_parameters_bias_,
        )
        hidden_states_106 = l_self_modules_encoder_modules_layers_modules_11_modules_fc2_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_11_modules_fc2_parameters_bias_ = (None)
        hidden_states_108 = torch.nn.functional.dropout(
            hidden_states_107, p=0.1, training=False
        )
        hidden_states_107 = None
        hidden_states_109 = hidden_states_103 + hidden_states_108
        hidden_states_103 = hidden_states_108 = None
        hidden_states_110 = torch.nn.functional.layer_norm(
            hidden_states_109,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_72 = torch._C._nn.linear(
            hidden_states_110,
            l_self_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_37 = linear_72.view(1, 10, -1, 64)
        linear_72 = None
        query_states_12 = view_37.transpose(1, 2)
        view_37 = None
        key_states_24 = torch._C._nn.linear(
            hidden_states_110,
            l_self_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_24 = torch._C._nn.linear(
            hidden_states_110,
            l_self_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_110 = l_self_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_38 = key_states_24.view(1, 10, -1, 64)
        key_states_24 = None
        key_states_25 = view_38.transpose(1, 2)
        view_38 = None
        view_39 = value_states_24.view(1, 10, -1, 64)
        value_states_24 = None
        value_states_25 = view_39.transpose(1, 2)
        view_39 = None
        attention_mask_13 = attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
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
        reshape_12 = attn_output_49.reshape(1, 10, -1)
        attn_output_49 = None
        attn_output_50 = reshape_12.contiguous()
        reshape_12 = None
        attn_output_51 = torch._C._nn.linear(
            attn_output_50,
            l_self_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_50 = l_self_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_111 = torch.nn.functional.dropout(
            attn_output_51, p=0.1, training=False
        )
        attn_output_51 = None
        hidden_states_112 = hidden_states_109 + hidden_states_111
        hidden_states_109 = hidden_states_111 = None
        hidden_states_113 = torch.nn.functional.layer_norm(
            hidden_states_112,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_12_modules_final_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_12_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layers_modules_12_modules_final_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_12_modules_final_layer_norm_parameters_bias_ = (None)
        linear_76 = torch._C._nn.linear(
            hidden_states_113,
            l_self_modules_encoder_modules_layers_modules_12_modules_fc1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_12_modules_fc1_parameters_bias_,
        )
        hidden_states_113 = l_self_modules_encoder_modules_layers_modules_12_modules_fc1_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_12_modules_fc1_parameters_bias_ = (None)
        hidden_states_114 = torch.nn.functional.relu(linear_76, inplace=False)
        linear_76 = None
        hidden_states_115 = torch.nn.functional.dropout(
            hidden_states_114, p=0.1, training=False
        )
        hidden_states_114 = None
        hidden_states_116 = torch._C._nn.linear(
            hidden_states_115,
            l_self_modules_encoder_modules_layers_modules_12_modules_fc2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_12_modules_fc2_parameters_bias_,
        )
        hidden_states_115 = l_self_modules_encoder_modules_layers_modules_12_modules_fc2_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_12_modules_fc2_parameters_bias_ = (None)
        hidden_states_117 = torch.nn.functional.dropout(
            hidden_states_116, p=0.1, training=False
        )
        hidden_states_116 = None
        hidden_states_118 = hidden_states_112 + hidden_states_117
        hidden_states_112 = hidden_states_117 = None
        hidden_states_119 = torch.nn.functional.layer_norm(
            hidden_states_118,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_78 = torch._C._nn.linear(
            hidden_states_119,
            l_self_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_40 = linear_78.view(1, 10, -1, 64)
        linear_78 = None
        query_states_13 = view_40.transpose(1, 2)
        view_40 = None
        key_states_26 = torch._C._nn.linear(
            hidden_states_119,
            l_self_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_26 = torch._C._nn.linear(
            hidden_states_119,
            l_self_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_119 = l_self_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_41 = key_states_26.view(1, 10, -1, 64)
        key_states_26 = None
        key_states_27 = view_41.transpose(1, 2)
        view_41 = None
        view_42 = value_states_26.view(1, 10, -1, 64)
        value_states_26 = None
        value_states_27 = view_42.transpose(1, 2)
        view_42 = None
        attention_mask_14 = attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
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
        reshape_13 = attn_output_53.reshape(1, 10, -1)
        attn_output_53 = None
        attn_output_54 = reshape_13.contiguous()
        reshape_13 = None
        attn_output_55 = torch._C._nn.linear(
            attn_output_54,
            l_self_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_54 = l_self_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_120 = torch.nn.functional.dropout(
            attn_output_55, p=0.1, training=False
        )
        attn_output_55 = None
        hidden_states_121 = hidden_states_118 + hidden_states_120
        hidden_states_118 = hidden_states_120 = None
        hidden_states_122 = torch.nn.functional.layer_norm(
            hidden_states_121,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_13_modules_final_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_13_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layers_modules_13_modules_final_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_13_modules_final_layer_norm_parameters_bias_ = (None)
        linear_82 = torch._C._nn.linear(
            hidden_states_122,
            l_self_modules_encoder_modules_layers_modules_13_modules_fc1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_13_modules_fc1_parameters_bias_,
        )
        hidden_states_122 = l_self_modules_encoder_modules_layers_modules_13_modules_fc1_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_13_modules_fc1_parameters_bias_ = (None)
        hidden_states_123 = torch.nn.functional.relu(linear_82, inplace=False)
        linear_82 = None
        hidden_states_124 = torch.nn.functional.dropout(
            hidden_states_123, p=0.1, training=False
        )
        hidden_states_123 = None
        hidden_states_125 = torch._C._nn.linear(
            hidden_states_124,
            l_self_modules_encoder_modules_layers_modules_13_modules_fc2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_13_modules_fc2_parameters_bias_,
        )
        hidden_states_124 = l_self_modules_encoder_modules_layers_modules_13_modules_fc2_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_13_modules_fc2_parameters_bias_ = (None)
        hidden_states_126 = torch.nn.functional.dropout(
            hidden_states_125, p=0.1, training=False
        )
        hidden_states_125 = None
        hidden_states_127 = hidden_states_121 + hidden_states_126
        hidden_states_121 = hidden_states_126 = None
        hidden_states_128 = torch.nn.functional.layer_norm(
            hidden_states_127,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_84 = torch._C._nn.linear(
            hidden_states_128,
            l_self_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_43 = linear_84.view(1, 10, -1, 64)
        linear_84 = None
        query_states_14 = view_43.transpose(1, 2)
        view_43 = None
        key_states_28 = torch._C._nn.linear(
            hidden_states_128,
            l_self_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_28 = torch._C._nn.linear(
            hidden_states_128,
            l_self_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_128 = l_self_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_44 = key_states_28.view(1, 10, -1, 64)
        key_states_28 = None
        key_states_29 = view_44.transpose(1, 2)
        view_44 = None
        view_45 = value_states_28.view(1, 10, -1, 64)
        value_states_28 = None
        value_states_29 = view_45.transpose(1, 2)
        view_45 = None
        attention_mask_15 = attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
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
        reshape_14 = attn_output_57.reshape(1, 10, -1)
        attn_output_57 = None
        attn_output_58 = reshape_14.contiguous()
        reshape_14 = None
        attn_output_59 = torch._C._nn.linear(
            attn_output_58,
            l_self_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_58 = l_self_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_129 = torch.nn.functional.dropout(
            attn_output_59, p=0.1, training=False
        )
        attn_output_59 = None
        hidden_states_130 = hidden_states_127 + hidden_states_129
        hidden_states_127 = hidden_states_129 = None
        hidden_states_131 = torch.nn.functional.layer_norm(
            hidden_states_130,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_14_modules_final_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_14_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layers_modules_14_modules_final_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_14_modules_final_layer_norm_parameters_bias_ = (None)
        linear_88 = torch._C._nn.linear(
            hidden_states_131,
            l_self_modules_encoder_modules_layers_modules_14_modules_fc1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_14_modules_fc1_parameters_bias_,
        )
        hidden_states_131 = l_self_modules_encoder_modules_layers_modules_14_modules_fc1_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_14_modules_fc1_parameters_bias_ = (None)
        hidden_states_132 = torch.nn.functional.relu(linear_88, inplace=False)
        linear_88 = None
        hidden_states_133 = torch.nn.functional.dropout(
            hidden_states_132, p=0.1, training=False
        )
        hidden_states_132 = None
        hidden_states_134 = torch._C._nn.linear(
            hidden_states_133,
            l_self_modules_encoder_modules_layers_modules_14_modules_fc2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_14_modules_fc2_parameters_bias_,
        )
        hidden_states_133 = l_self_modules_encoder_modules_layers_modules_14_modules_fc2_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_14_modules_fc2_parameters_bias_ = (None)
        hidden_states_135 = torch.nn.functional.dropout(
            hidden_states_134, p=0.1, training=False
        )
        hidden_states_134 = None
        hidden_states_136 = hidden_states_130 + hidden_states_135
        hidden_states_130 = hidden_states_135 = None
        hidden_states_137 = torch.nn.functional.layer_norm(
            hidden_states_136,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_90 = torch._C._nn.linear(
            hidden_states_137,
            l_self_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_46 = linear_90.view(1, 10, -1, 64)
        linear_90 = None
        query_states_15 = view_46.transpose(1, 2)
        view_46 = None
        key_states_30 = torch._C._nn.linear(
            hidden_states_137,
            l_self_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_30 = torch._C._nn.linear(
            hidden_states_137,
            l_self_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_137 = l_self_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_47 = key_states_30.view(1, 10, -1, 64)
        key_states_30 = None
        key_states_31 = view_47.transpose(1, 2)
        view_47 = None
        view_48 = value_states_30.view(1, 10, -1, 64)
        value_states_30 = None
        value_states_31 = view_48.transpose(1, 2)
        view_48 = None
        attention_mask_16 = attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        attention_mask = None
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
        reshape_15 = attn_output_61.reshape(1, 10, -1)
        attn_output_61 = None
        attn_output_62 = reshape_15.contiguous()
        reshape_15 = None
        attn_output_63 = torch._C._nn.linear(
            attn_output_62,
            l_self_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_62 = l_self_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_138 = torch.nn.functional.dropout(
            attn_output_63, p=0.1, training=False
        )
        attn_output_63 = None
        hidden_states_139 = hidden_states_136 + hidden_states_138
        hidden_states_136 = hidden_states_138 = None
        hidden_states_140 = torch.nn.functional.layer_norm(
            hidden_states_139,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_15_modules_final_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_15_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layers_modules_15_modules_final_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_15_modules_final_layer_norm_parameters_bias_ = (None)
        linear_94 = torch._C._nn.linear(
            hidden_states_140,
            l_self_modules_encoder_modules_layers_modules_15_modules_fc1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_15_modules_fc1_parameters_bias_,
        )
        hidden_states_140 = l_self_modules_encoder_modules_layers_modules_15_modules_fc1_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_15_modules_fc1_parameters_bias_ = (None)
        hidden_states_141 = torch.nn.functional.relu(linear_94, inplace=False)
        linear_94 = None
        hidden_states_142 = torch.nn.functional.dropout(
            hidden_states_141, p=0.1, training=False
        )
        hidden_states_141 = None
        hidden_states_143 = torch._C._nn.linear(
            hidden_states_142,
            l_self_modules_encoder_modules_layers_modules_15_modules_fc2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_15_modules_fc2_parameters_bias_,
        )
        hidden_states_142 = l_self_modules_encoder_modules_layers_modules_15_modules_fc2_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_15_modules_fc2_parameters_bias_ = (None)
        hidden_states_144 = torch.nn.functional.dropout(
            hidden_states_143, p=0.1, training=False
        )
        hidden_states_143 = None
        hidden_states_145 = hidden_states_139 + hidden_states_144
        hidden_states_139 = hidden_states_144 = None
        hidden_states_146 = torch.nn.functional.layer_norm(
            hidden_states_145,
            (1024,),
            l_self_modules_encoder_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_145 = (
            l_self_modules_encoder_modules_layer_norm_parameters_weight_
        ) = l_self_modules_encoder_modules_layer_norm_parameters_bias_ = None
        input_ids_1 = l_decoder_input_ids_.view(-1, 10)
        input_ids_1 = None
        inputs_embeds_1 = torch.nn.functional.embedding(
            l_decoder_input_ids_,
            l_self_modules_encoder_modules_embed_tokens_parameters_weight_,
            0,
            None,
            2.0,
            False,
            False,
        )
        l_decoder_input_ids_ = (
            l_self_modules_encoder_modules_embed_tokens_parameters_weight_
        ) = None
        inputs_embeds_2 = inputs_embeds_1 * 32.0
        inputs_embeds_1 = None
        cache_position = torch.arange(0, 10, device=device(type="cuda", index=0))
        causal_mask = torch.full(
            (10, 11),
            fill_value=-3.3895313892515355e38,
            dtype=torch.bfloat16,
            device=device(type="cuda", index=0),
        )
        causal_mask_1 = torch.triu(causal_mask, diagonal=1)
        causal_mask = None
        arange_2 = torch.arange(11, device=device(type="cuda", index=0))
        reshape_16 = cache_position.reshape(-1, 1)
        gt = arange_2 > reshape_16
        arange_2 = reshape_16 = None
        causal_mask_1 *= gt
        causal_mask_2 = causal_mask_1
        causal_mask_1 = gt = None
        getitem_17 = causal_mask_2[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        causal_mask_2 = None
        causal_mask_3 = getitem_17.expand(1, 1, -1, -1)
        getitem_17 = None
        getitem_18 = l_attention_mask_[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        l_attention_mask_ = None
        expand_2 = getitem_18.expand(1, 1, 10, 10)
        getitem_18 = None
        expanded_mask_1 = expand_2.to(torch.bfloat16)
        expand_2 = None
        tensor_1 = torch.tensor(1.0, dtype=torch.bfloat16)
        inverted_mask_1 = tensor_1 - expanded_mask_1
        tensor_1 = expanded_mask_1 = None
        to_3 = inverted_mask_1.to(torch.bool)
        encoder_attention_mask = inverted_mask_1.masked_fill(
            to_3, -3.3895313892515355e38
        )
        inverted_mask_1 = to_3 = None
        positions = torch.nn.functional.embedding(
            cache_position,
            l_self_modules_decoder_modules_embed_positions_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        cache_position = (
            l_self_modules_decoder_modules_embed_positions_parameters_weight_
        ) = None
        hidden_states_147 = inputs_embeds_2 + positions
        inputs_embeds_2 = positions = None
        hidden_states_148 = torch.nn.functional.dropout(
            hidden_states_147, p=0.1, training=False
        )
        hidden_states_147 = None
        hidden_states_149 = torch.nn.functional.layer_norm(
            hidden_states_148,
            (1024,),
            l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_96 = torch._C._nn.linear(
            hidden_states_149,
            l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_50 = linear_96.view(1, 10, -1, 64)
        linear_96 = None
        query_states_16 = view_50.transpose(1, 2)
        view_50 = None
        key_states_32 = torch._C._nn.linear(
            hidden_states_149,
            l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_32 = torch._C._nn.linear(
            hidden_states_149,
            l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_149 = l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_51 = key_states_32.view(1, 10, -1, 64)
        key_states_32 = None
        key_states_33 = view_51.transpose(1, 2)
        view_51 = None
        view_52 = value_states_32.view(1, 10, -1, 64)
        value_states_32 = None
        value_states_33 = view_52.transpose(1, 2)
        view_52 = None
        attention_mask_17 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
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
        reshape_17 = attn_output_65.reshape(1, 10, -1)
        attn_output_65 = None
        attn_output_66 = reshape_17.contiguous()
        reshape_17 = None
        attn_output_67 = torch._C._nn.linear(
            attn_output_66,
            l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_66 = l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_150 = torch.nn.functional.dropout(
            attn_output_67, p=0.1, training=False
        )
        attn_output_67 = None
        hidden_states_151 = hidden_states_148 + hidden_states_150
        hidden_states_148 = hidden_states_150 = None
        hidden_states_152 = torch.nn.functional.layer_norm(
            hidden_states_151,
            (1024,),
            l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_100 = torch._C._nn.linear(
            hidden_states_152,
            l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        hidden_states_152 = l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        view_53 = linear_100.view(1, 10, -1, 64)
        linear_100 = None
        query_states_17 = view_53.transpose(1, 2)
        view_53 = None
        key_states_34 = torch._C._nn.linear(
            hidden_states_146,
            l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_34 = torch._C._nn.linear(
            hidden_states_146,
            l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_54 = key_states_34.view(1, 10, -1, 64)
        key_states_34 = None
        key_states_35 = view_54.transpose(1, 2)
        view_54 = None
        view_55 = value_states_34.view(1, 10, -1, 64)
        value_states_34 = None
        value_states_35 = view_55.transpose(1, 2)
        view_55 = None
        attention_mask_18 = encoder_attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
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
        reshape_18 = attn_output_69.reshape(1, 10, -1)
        attn_output_69 = None
        attn_output_70 = reshape_18.contiguous()
        reshape_18 = None
        attn_output_71 = torch._C._nn.linear(
            attn_output_70,
            l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_70 = l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_153 = torch.nn.functional.dropout(
            attn_output_71, p=0.1, training=False
        )
        attn_output_71 = None
        hidden_states_154 = hidden_states_151 + hidden_states_153
        hidden_states_151 = hidden_states_153 = None
        hidden_states_155 = torch.nn.functional.layer_norm(
            hidden_states_154,
            (1024,),
            l_self_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_ = (None)
        linear_104 = torch._C._nn.linear(
            hidden_states_155,
            l_self_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_bias_,
        )
        hidden_states_155 = l_self_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_weight_ = (
            l_self_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_bias_
        ) = None
        hidden_states_156 = torch.nn.functional.relu(linear_104, inplace=False)
        linear_104 = None
        hidden_states_157 = torch.nn.functional.dropout(
            hidden_states_156, p=0.1, training=False
        )
        hidden_states_156 = None
        hidden_states_158 = torch._C._nn.linear(
            hidden_states_157,
            l_self_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_bias_,
        )
        hidden_states_157 = l_self_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_weight_ = (
            l_self_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_bias_
        ) = None
        hidden_states_159 = torch.nn.functional.dropout(
            hidden_states_158, p=0.1, training=False
        )
        hidden_states_158 = None
        hidden_states_160 = hidden_states_154 + hidden_states_159
        hidden_states_154 = hidden_states_159 = None
        hidden_states_161 = torch.nn.functional.layer_norm(
            hidden_states_160,
            (1024,),
            l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_106 = torch._C._nn.linear(
            hidden_states_161,
            l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_56 = linear_106.view(1, 10, -1, 64)
        linear_106 = None
        query_states_18 = view_56.transpose(1, 2)
        view_56 = None
        key_states_36 = torch._C._nn.linear(
            hidden_states_161,
            l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_36 = torch._C._nn.linear(
            hidden_states_161,
            l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_161 = l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_57 = key_states_36.view(1, 10, -1, 64)
        key_states_36 = None
        key_states_37 = view_57.transpose(1, 2)
        view_57 = None
        view_58 = value_states_36.view(1, 10, -1, 64)
        value_states_36 = None
        value_states_37 = view_58.transpose(1, 2)
        view_58 = None
        attention_mask_19 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_18 = query_states_18.contiguous()
        query_states_18 = None
        key_18 = key_states_37.contiguous()
        key_states_37 = None
        value_18 = value_states_37.contiguous()
        value_states_37 = None
        attn_output_72 = torch._C._nn.scaled_dot_product_attention(
            query_18,
            key_18,
            value_18,
            attn_mask=attention_mask_19,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_18 = key_18 = value_18 = attention_mask_19 = None
        transpose_75 = attn_output_72.transpose(1, 2)
        attn_output_72 = None
        attn_output_73 = transpose_75.contiguous()
        transpose_75 = None
        reshape_19 = attn_output_73.reshape(1, 10, -1)
        attn_output_73 = None
        attn_output_74 = reshape_19.contiguous()
        reshape_19 = None
        attn_output_75 = torch._C._nn.linear(
            attn_output_74,
            l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_74 = l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_162 = torch.nn.functional.dropout(
            attn_output_75, p=0.1, training=False
        )
        attn_output_75 = None
        hidden_states_163 = hidden_states_160 + hidden_states_162
        hidden_states_160 = hidden_states_162 = None
        hidden_states_164 = torch.nn.functional.layer_norm(
            hidden_states_163,
            (1024,),
            l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_110 = torch._C._nn.linear(
            hidden_states_164,
            l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        hidden_states_164 = l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        view_59 = linear_110.view(1, 10, -1, 64)
        linear_110 = None
        query_states_19 = view_59.transpose(1, 2)
        view_59 = None
        key_states_38 = torch._C._nn.linear(
            hidden_states_146,
            l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_38 = torch._C._nn.linear(
            hidden_states_146,
            l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_60 = key_states_38.view(1, 10, -1, 64)
        key_states_38 = None
        key_states_39 = view_60.transpose(1, 2)
        view_60 = None
        view_61 = value_states_38.view(1, 10, -1, 64)
        value_states_38 = None
        value_states_39 = view_61.transpose(1, 2)
        view_61 = None
        attention_mask_20 = encoder_attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_19 = query_states_19.contiguous()
        query_states_19 = None
        key_19 = key_states_39.contiguous()
        key_states_39 = None
        value_19 = value_states_39.contiguous()
        value_states_39 = None
        attn_output_76 = torch._C._nn.scaled_dot_product_attention(
            query_19,
            key_19,
            value_19,
            attn_mask=attention_mask_20,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_19 = key_19 = value_19 = attention_mask_20 = None
        transpose_79 = attn_output_76.transpose(1, 2)
        attn_output_76 = None
        attn_output_77 = transpose_79.contiguous()
        transpose_79 = None
        reshape_20 = attn_output_77.reshape(1, 10, -1)
        attn_output_77 = None
        attn_output_78 = reshape_20.contiguous()
        reshape_20 = None
        attn_output_79 = torch._C._nn.linear(
            attn_output_78,
            l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_78 = l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_165 = torch.nn.functional.dropout(
            attn_output_79, p=0.1, training=False
        )
        attn_output_79 = None
        hidden_states_166 = hidden_states_163 + hidden_states_165
        hidden_states_163 = hidden_states_165 = None
        hidden_states_167 = torch.nn.functional.layer_norm(
            hidden_states_166,
            (1024,),
            l_self_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_ = (None)
        linear_114 = torch._C._nn.linear(
            hidden_states_167,
            l_self_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_bias_,
        )
        hidden_states_167 = l_self_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_weight_ = (
            l_self_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_bias_
        ) = None
        hidden_states_168 = torch.nn.functional.relu(linear_114, inplace=False)
        linear_114 = None
        hidden_states_169 = torch.nn.functional.dropout(
            hidden_states_168, p=0.1, training=False
        )
        hidden_states_168 = None
        hidden_states_170 = torch._C._nn.linear(
            hidden_states_169,
            l_self_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_bias_,
        )
        hidden_states_169 = l_self_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_weight_ = (
            l_self_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_bias_
        ) = None
        hidden_states_171 = torch.nn.functional.dropout(
            hidden_states_170, p=0.1, training=False
        )
        hidden_states_170 = None
        hidden_states_172 = hidden_states_166 + hidden_states_171
        hidden_states_166 = hidden_states_171 = None
        hidden_states_173 = torch.nn.functional.layer_norm(
            hidden_states_172,
            (1024,),
            l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_116 = torch._C._nn.linear(
            hidden_states_173,
            l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_62 = linear_116.view(1, 10, -1, 64)
        linear_116 = None
        query_states_20 = view_62.transpose(1, 2)
        view_62 = None
        key_states_40 = torch._C._nn.linear(
            hidden_states_173,
            l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_40 = torch._C._nn.linear(
            hidden_states_173,
            l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_173 = l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_63 = key_states_40.view(1, 10, -1, 64)
        key_states_40 = None
        key_states_41 = view_63.transpose(1, 2)
        view_63 = None
        view_64 = value_states_40.view(1, 10, -1, 64)
        value_states_40 = None
        value_states_41 = view_64.transpose(1, 2)
        view_64 = None
        attention_mask_21 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_20 = query_states_20.contiguous()
        query_states_20 = None
        key_20 = key_states_41.contiguous()
        key_states_41 = None
        value_20 = value_states_41.contiguous()
        value_states_41 = None
        attn_output_80 = torch._C._nn.scaled_dot_product_attention(
            query_20,
            key_20,
            value_20,
            attn_mask=attention_mask_21,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_20 = key_20 = value_20 = attention_mask_21 = None
        transpose_83 = attn_output_80.transpose(1, 2)
        attn_output_80 = None
        attn_output_81 = transpose_83.contiguous()
        transpose_83 = None
        reshape_21 = attn_output_81.reshape(1, 10, -1)
        attn_output_81 = None
        attn_output_82 = reshape_21.contiguous()
        reshape_21 = None
        attn_output_83 = torch._C._nn.linear(
            attn_output_82,
            l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_82 = l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_174 = torch.nn.functional.dropout(
            attn_output_83, p=0.1, training=False
        )
        attn_output_83 = None
        hidden_states_175 = hidden_states_172 + hidden_states_174
        hidden_states_172 = hidden_states_174 = None
        hidden_states_176 = torch.nn.functional.layer_norm(
            hidden_states_175,
            (1024,),
            l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_120 = torch._C._nn.linear(
            hidden_states_176,
            l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        hidden_states_176 = l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        view_65 = linear_120.view(1, 10, -1, 64)
        linear_120 = None
        query_states_21 = view_65.transpose(1, 2)
        view_65 = None
        key_states_42 = torch._C._nn.linear(
            hidden_states_146,
            l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_42 = torch._C._nn.linear(
            hidden_states_146,
            l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_66 = key_states_42.view(1, 10, -1, 64)
        key_states_42 = None
        key_states_43 = view_66.transpose(1, 2)
        view_66 = None
        view_67 = value_states_42.view(1, 10, -1, 64)
        value_states_42 = None
        value_states_43 = view_67.transpose(1, 2)
        view_67 = None
        attention_mask_22 = encoder_attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_21 = query_states_21.contiguous()
        query_states_21 = None
        key_21 = key_states_43.contiguous()
        key_states_43 = None
        value_21 = value_states_43.contiguous()
        value_states_43 = None
        attn_output_84 = torch._C._nn.scaled_dot_product_attention(
            query_21,
            key_21,
            value_21,
            attn_mask=attention_mask_22,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_21 = key_21 = value_21 = attention_mask_22 = None
        transpose_87 = attn_output_84.transpose(1, 2)
        attn_output_84 = None
        attn_output_85 = transpose_87.contiguous()
        transpose_87 = None
        reshape_22 = attn_output_85.reshape(1, 10, -1)
        attn_output_85 = None
        attn_output_86 = reshape_22.contiguous()
        reshape_22 = None
        attn_output_87 = torch._C._nn.linear(
            attn_output_86,
            l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_86 = l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_177 = torch.nn.functional.dropout(
            attn_output_87, p=0.1, training=False
        )
        attn_output_87 = None
        hidden_states_178 = hidden_states_175 + hidden_states_177
        hidden_states_175 = hidden_states_177 = None
        hidden_states_179 = torch.nn.functional.layer_norm(
            hidden_states_178,
            (1024,),
            l_self_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_ = (None)
        linear_124 = torch._C._nn.linear(
            hidden_states_179,
            l_self_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_bias_,
        )
        hidden_states_179 = l_self_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_weight_ = (
            l_self_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_bias_
        ) = None
        hidden_states_180 = torch.nn.functional.relu(linear_124, inplace=False)
        linear_124 = None
        hidden_states_181 = torch.nn.functional.dropout(
            hidden_states_180, p=0.1, training=False
        )
        hidden_states_180 = None
        hidden_states_182 = torch._C._nn.linear(
            hidden_states_181,
            l_self_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_bias_,
        )
        hidden_states_181 = l_self_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_weight_ = (
            l_self_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_bias_
        ) = None
        hidden_states_183 = torch.nn.functional.dropout(
            hidden_states_182, p=0.1, training=False
        )
        hidden_states_182 = None
        hidden_states_184 = hidden_states_178 + hidden_states_183
        hidden_states_178 = hidden_states_183 = None
        hidden_states_185 = torch.nn.functional.layer_norm(
            hidden_states_184,
            (1024,),
            l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_126 = torch._C._nn.linear(
            hidden_states_185,
            l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_68 = linear_126.view(1, 10, -1, 64)
        linear_126 = None
        query_states_22 = view_68.transpose(1, 2)
        view_68 = None
        key_states_44 = torch._C._nn.linear(
            hidden_states_185,
            l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_44 = torch._C._nn.linear(
            hidden_states_185,
            l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_185 = l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_69 = key_states_44.view(1, 10, -1, 64)
        key_states_44 = None
        key_states_45 = view_69.transpose(1, 2)
        view_69 = None
        view_70 = value_states_44.view(1, 10, -1, 64)
        value_states_44 = None
        value_states_45 = view_70.transpose(1, 2)
        view_70 = None
        attention_mask_23 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_22 = query_states_22.contiguous()
        query_states_22 = None
        key_22 = key_states_45.contiguous()
        key_states_45 = None
        value_22 = value_states_45.contiguous()
        value_states_45 = None
        attn_output_88 = torch._C._nn.scaled_dot_product_attention(
            query_22,
            key_22,
            value_22,
            attn_mask=attention_mask_23,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_22 = key_22 = value_22 = attention_mask_23 = None
        transpose_91 = attn_output_88.transpose(1, 2)
        attn_output_88 = None
        attn_output_89 = transpose_91.contiguous()
        transpose_91 = None
        reshape_23 = attn_output_89.reshape(1, 10, -1)
        attn_output_89 = None
        attn_output_90 = reshape_23.contiguous()
        reshape_23 = None
        attn_output_91 = torch._C._nn.linear(
            attn_output_90,
            l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_90 = l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_186 = torch.nn.functional.dropout(
            attn_output_91, p=0.1, training=False
        )
        attn_output_91 = None
        hidden_states_187 = hidden_states_184 + hidden_states_186
        hidden_states_184 = hidden_states_186 = None
        hidden_states_188 = torch.nn.functional.layer_norm(
            hidden_states_187,
            (1024,),
            l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_130 = torch._C._nn.linear(
            hidden_states_188,
            l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        hidden_states_188 = l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        view_71 = linear_130.view(1, 10, -1, 64)
        linear_130 = None
        query_states_23 = view_71.transpose(1, 2)
        view_71 = None
        key_states_46 = torch._C._nn.linear(
            hidden_states_146,
            l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_46 = torch._C._nn.linear(
            hidden_states_146,
            l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_72 = key_states_46.view(1, 10, -1, 64)
        key_states_46 = None
        key_states_47 = view_72.transpose(1, 2)
        view_72 = None
        view_73 = value_states_46.view(1, 10, -1, 64)
        value_states_46 = None
        value_states_47 = view_73.transpose(1, 2)
        view_73 = None
        attention_mask_24 = encoder_attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_23 = query_states_23.contiguous()
        query_states_23 = None
        key_23 = key_states_47.contiguous()
        key_states_47 = None
        value_23 = value_states_47.contiguous()
        value_states_47 = None
        attn_output_92 = torch._C._nn.scaled_dot_product_attention(
            query_23,
            key_23,
            value_23,
            attn_mask=attention_mask_24,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_23 = key_23 = value_23 = attention_mask_24 = None
        transpose_95 = attn_output_92.transpose(1, 2)
        attn_output_92 = None
        attn_output_93 = transpose_95.contiguous()
        transpose_95 = None
        reshape_24 = attn_output_93.reshape(1, 10, -1)
        attn_output_93 = None
        attn_output_94 = reshape_24.contiguous()
        reshape_24 = None
        attn_output_95 = torch._C._nn.linear(
            attn_output_94,
            l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_94 = l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_189 = torch.nn.functional.dropout(
            attn_output_95, p=0.1, training=False
        )
        attn_output_95 = None
        hidden_states_190 = hidden_states_187 + hidden_states_189
        hidden_states_187 = hidden_states_189 = None
        hidden_states_191 = torch.nn.functional.layer_norm(
            hidden_states_190,
            (1024,),
            l_self_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_ = (None)
        linear_134 = torch._C._nn.linear(
            hidden_states_191,
            l_self_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_bias_,
        )
        hidden_states_191 = l_self_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_weight_ = (
            l_self_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_bias_
        ) = None
        hidden_states_192 = torch.nn.functional.relu(linear_134, inplace=False)
        linear_134 = None
        hidden_states_193 = torch.nn.functional.dropout(
            hidden_states_192, p=0.1, training=False
        )
        hidden_states_192 = None
        hidden_states_194 = torch._C._nn.linear(
            hidden_states_193,
            l_self_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_bias_,
        )
        hidden_states_193 = l_self_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_weight_ = (
            l_self_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_bias_
        ) = None
        hidden_states_195 = torch.nn.functional.dropout(
            hidden_states_194, p=0.1, training=False
        )
        hidden_states_194 = None
        hidden_states_196 = hidden_states_190 + hidden_states_195
        hidden_states_190 = hidden_states_195 = None
        hidden_states_197 = torch.nn.functional.layer_norm(
            hidden_states_196,
            (1024,),
            l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_136 = torch._C._nn.linear(
            hidden_states_197,
            l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_74 = linear_136.view(1, 10, -1, 64)
        linear_136 = None
        query_states_24 = view_74.transpose(1, 2)
        view_74 = None
        key_states_48 = torch._C._nn.linear(
            hidden_states_197,
            l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_48 = torch._C._nn.linear(
            hidden_states_197,
            l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_197 = l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_75 = key_states_48.view(1, 10, -1, 64)
        key_states_48 = None
        key_states_49 = view_75.transpose(1, 2)
        view_75 = None
        view_76 = value_states_48.view(1, 10, -1, 64)
        value_states_48 = None
        value_states_49 = view_76.transpose(1, 2)
        view_76 = None
        attention_mask_25 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_24 = query_states_24.contiguous()
        query_states_24 = None
        key_24 = key_states_49.contiguous()
        key_states_49 = None
        value_24 = value_states_49.contiguous()
        value_states_49 = None
        attn_output_96 = torch._C._nn.scaled_dot_product_attention(
            query_24,
            key_24,
            value_24,
            attn_mask=attention_mask_25,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_24 = key_24 = value_24 = attention_mask_25 = None
        transpose_99 = attn_output_96.transpose(1, 2)
        attn_output_96 = None
        attn_output_97 = transpose_99.contiguous()
        transpose_99 = None
        reshape_25 = attn_output_97.reshape(1, 10, -1)
        attn_output_97 = None
        attn_output_98 = reshape_25.contiguous()
        reshape_25 = None
        attn_output_99 = torch._C._nn.linear(
            attn_output_98,
            l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_98 = l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_198 = torch.nn.functional.dropout(
            attn_output_99, p=0.1, training=False
        )
        attn_output_99 = None
        hidden_states_199 = hidden_states_196 + hidden_states_198
        hidden_states_196 = hidden_states_198 = None
        hidden_states_200 = torch.nn.functional.layer_norm(
            hidden_states_199,
            (1024,),
            l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_140 = torch._C._nn.linear(
            hidden_states_200,
            l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        hidden_states_200 = l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        view_77 = linear_140.view(1, 10, -1, 64)
        linear_140 = None
        query_states_25 = view_77.transpose(1, 2)
        view_77 = None
        key_states_50 = torch._C._nn.linear(
            hidden_states_146,
            l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_50 = torch._C._nn.linear(
            hidden_states_146,
            l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_78 = key_states_50.view(1, 10, -1, 64)
        key_states_50 = None
        key_states_51 = view_78.transpose(1, 2)
        view_78 = None
        view_79 = value_states_50.view(1, 10, -1, 64)
        value_states_50 = None
        value_states_51 = view_79.transpose(1, 2)
        view_79 = None
        attention_mask_26 = encoder_attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_25 = query_states_25.contiguous()
        query_states_25 = None
        key_25 = key_states_51.contiguous()
        key_states_51 = None
        value_25 = value_states_51.contiguous()
        value_states_51 = None
        attn_output_100 = torch._C._nn.scaled_dot_product_attention(
            query_25,
            key_25,
            value_25,
            attn_mask=attention_mask_26,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_25 = key_25 = value_25 = attention_mask_26 = None
        transpose_103 = attn_output_100.transpose(1, 2)
        attn_output_100 = None
        attn_output_101 = transpose_103.contiguous()
        transpose_103 = None
        reshape_26 = attn_output_101.reshape(1, 10, -1)
        attn_output_101 = None
        attn_output_102 = reshape_26.contiguous()
        reshape_26 = None
        attn_output_103 = torch._C._nn.linear(
            attn_output_102,
            l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_102 = l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_201 = torch.nn.functional.dropout(
            attn_output_103, p=0.1, training=False
        )
        attn_output_103 = None
        hidden_states_202 = hidden_states_199 + hidden_states_201
        hidden_states_199 = hidden_states_201 = None
        hidden_states_203 = torch.nn.functional.layer_norm(
            hidden_states_202,
            (1024,),
            l_self_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_ = (None)
        linear_144 = torch._C._nn.linear(
            hidden_states_203,
            l_self_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_bias_,
        )
        hidden_states_203 = l_self_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_weight_ = (
            l_self_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_bias_
        ) = None
        hidden_states_204 = torch.nn.functional.relu(linear_144, inplace=False)
        linear_144 = None
        hidden_states_205 = torch.nn.functional.dropout(
            hidden_states_204, p=0.1, training=False
        )
        hidden_states_204 = None
        hidden_states_206 = torch._C._nn.linear(
            hidden_states_205,
            l_self_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_bias_,
        )
        hidden_states_205 = l_self_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_weight_ = (
            l_self_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_bias_
        ) = None
        hidden_states_207 = torch.nn.functional.dropout(
            hidden_states_206, p=0.1, training=False
        )
        hidden_states_206 = None
        hidden_states_208 = hidden_states_202 + hidden_states_207
        hidden_states_202 = hidden_states_207 = None
        hidden_states_209 = torch.nn.functional.layer_norm(
            hidden_states_208,
            (1024,),
            l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_146 = torch._C._nn.linear(
            hidden_states_209,
            l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_80 = linear_146.view(1, 10, -1, 64)
        linear_146 = None
        query_states_26 = view_80.transpose(1, 2)
        view_80 = None
        key_states_52 = torch._C._nn.linear(
            hidden_states_209,
            l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_52 = torch._C._nn.linear(
            hidden_states_209,
            l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_209 = l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_81 = key_states_52.view(1, 10, -1, 64)
        key_states_52 = None
        key_states_53 = view_81.transpose(1, 2)
        view_81 = None
        view_82 = value_states_52.view(1, 10, -1, 64)
        value_states_52 = None
        value_states_53 = view_82.transpose(1, 2)
        view_82 = None
        attention_mask_27 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_26 = query_states_26.contiguous()
        query_states_26 = None
        key_26 = key_states_53.contiguous()
        key_states_53 = None
        value_26 = value_states_53.contiguous()
        value_states_53 = None
        attn_output_104 = torch._C._nn.scaled_dot_product_attention(
            query_26,
            key_26,
            value_26,
            attn_mask=attention_mask_27,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_26 = key_26 = value_26 = attention_mask_27 = None
        transpose_107 = attn_output_104.transpose(1, 2)
        attn_output_104 = None
        attn_output_105 = transpose_107.contiguous()
        transpose_107 = None
        reshape_27 = attn_output_105.reshape(1, 10, -1)
        attn_output_105 = None
        attn_output_106 = reshape_27.contiguous()
        reshape_27 = None
        attn_output_107 = torch._C._nn.linear(
            attn_output_106,
            l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_106 = l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_210 = torch.nn.functional.dropout(
            attn_output_107, p=0.1, training=False
        )
        attn_output_107 = None
        hidden_states_211 = hidden_states_208 + hidden_states_210
        hidden_states_208 = hidden_states_210 = None
        hidden_states_212 = torch.nn.functional.layer_norm(
            hidden_states_211,
            (1024,),
            l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_150 = torch._C._nn.linear(
            hidden_states_212,
            l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        hidden_states_212 = l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        view_83 = linear_150.view(1, 10, -1, 64)
        linear_150 = None
        query_states_27 = view_83.transpose(1, 2)
        view_83 = None
        key_states_54 = torch._C._nn.linear(
            hidden_states_146,
            l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_54 = torch._C._nn.linear(
            hidden_states_146,
            l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_84 = key_states_54.view(1, 10, -1, 64)
        key_states_54 = None
        key_states_55 = view_84.transpose(1, 2)
        view_84 = None
        view_85 = value_states_54.view(1, 10, -1, 64)
        value_states_54 = None
        value_states_55 = view_85.transpose(1, 2)
        view_85 = None
        attention_mask_28 = encoder_attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_27 = query_states_27.contiguous()
        query_states_27 = None
        key_27 = key_states_55.contiguous()
        key_states_55 = None
        value_27 = value_states_55.contiguous()
        value_states_55 = None
        attn_output_108 = torch._C._nn.scaled_dot_product_attention(
            query_27,
            key_27,
            value_27,
            attn_mask=attention_mask_28,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_27 = key_27 = value_27 = attention_mask_28 = None
        transpose_111 = attn_output_108.transpose(1, 2)
        attn_output_108 = None
        attn_output_109 = transpose_111.contiguous()
        transpose_111 = None
        reshape_28 = attn_output_109.reshape(1, 10, -1)
        attn_output_109 = None
        attn_output_110 = reshape_28.contiguous()
        reshape_28 = None
        attn_output_111 = torch._C._nn.linear(
            attn_output_110,
            l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_110 = l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_213 = torch.nn.functional.dropout(
            attn_output_111, p=0.1, training=False
        )
        attn_output_111 = None
        hidden_states_214 = hidden_states_211 + hidden_states_213
        hidden_states_211 = hidden_states_213 = None
        hidden_states_215 = torch.nn.functional.layer_norm(
            hidden_states_214,
            (1024,),
            l_self_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_ = (None)
        linear_154 = torch._C._nn.linear(
            hidden_states_215,
            l_self_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_bias_,
        )
        hidden_states_215 = l_self_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_weight_ = (
            l_self_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_bias_
        ) = None
        hidden_states_216 = torch.nn.functional.relu(linear_154, inplace=False)
        linear_154 = None
        hidden_states_217 = torch.nn.functional.dropout(
            hidden_states_216, p=0.1, training=False
        )
        hidden_states_216 = None
        hidden_states_218 = torch._C._nn.linear(
            hidden_states_217,
            l_self_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_bias_,
        )
        hidden_states_217 = l_self_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_weight_ = (
            l_self_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_bias_
        ) = None
        hidden_states_219 = torch.nn.functional.dropout(
            hidden_states_218, p=0.1, training=False
        )
        hidden_states_218 = None
        hidden_states_220 = hidden_states_214 + hidden_states_219
        hidden_states_214 = hidden_states_219 = None
        hidden_states_221 = torch.nn.functional.layer_norm(
            hidden_states_220,
            (1024,),
            l_self_modules_decoder_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_156 = torch._C._nn.linear(
            hidden_states_221,
            l_self_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_86 = linear_156.view(1, 10, -1, 64)
        linear_156 = None
        query_states_28 = view_86.transpose(1, 2)
        view_86 = None
        key_states_56 = torch._C._nn.linear(
            hidden_states_221,
            l_self_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_56 = torch._C._nn.linear(
            hidden_states_221,
            l_self_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_221 = l_self_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_87 = key_states_56.view(1, 10, -1, 64)
        key_states_56 = None
        key_states_57 = view_87.transpose(1, 2)
        view_87 = None
        view_88 = value_states_56.view(1, 10, -1, 64)
        value_states_56 = None
        value_states_57 = view_88.transpose(1, 2)
        view_88 = None
        attention_mask_29 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_28 = query_states_28.contiguous()
        query_states_28 = None
        key_28 = key_states_57.contiguous()
        key_states_57 = None
        value_28 = value_states_57.contiguous()
        value_states_57 = None
        attn_output_112 = torch._C._nn.scaled_dot_product_attention(
            query_28,
            key_28,
            value_28,
            attn_mask=attention_mask_29,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_28 = key_28 = value_28 = attention_mask_29 = None
        transpose_115 = attn_output_112.transpose(1, 2)
        attn_output_112 = None
        attn_output_113 = transpose_115.contiguous()
        transpose_115 = None
        reshape_29 = attn_output_113.reshape(1, 10, -1)
        attn_output_113 = None
        attn_output_114 = reshape_29.contiguous()
        reshape_29 = None
        attn_output_115 = torch._C._nn.linear(
            attn_output_114,
            l_self_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_114 = l_self_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_222 = torch.nn.functional.dropout(
            attn_output_115, p=0.1, training=False
        )
        attn_output_115 = None
        hidden_states_223 = hidden_states_220 + hidden_states_222
        hidden_states_220 = hidden_states_222 = None
        hidden_states_224 = torch.nn.functional.layer_norm(
            hidden_states_223,
            (1024,),
            l_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_160 = torch._C._nn.linear(
            hidden_states_224,
            l_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        hidden_states_224 = l_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        view_89 = linear_160.view(1, 10, -1, 64)
        linear_160 = None
        query_states_29 = view_89.transpose(1, 2)
        view_89 = None
        key_states_58 = torch._C._nn.linear(
            hidden_states_146,
            l_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_58 = torch._C._nn.linear(
            hidden_states_146,
            l_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_90 = key_states_58.view(1, 10, -1, 64)
        key_states_58 = None
        key_states_59 = view_90.transpose(1, 2)
        view_90 = None
        view_91 = value_states_58.view(1, 10, -1, 64)
        value_states_58 = None
        value_states_59 = view_91.transpose(1, 2)
        view_91 = None
        attention_mask_30 = encoder_attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_29 = query_states_29.contiguous()
        query_states_29 = None
        key_29 = key_states_59.contiguous()
        key_states_59 = None
        value_29 = value_states_59.contiguous()
        value_states_59 = None
        attn_output_116 = torch._C._nn.scaled_dot_product_attention(
            query_29,
            key_29,
            value_29,
            attn_mask=attention_mask_30,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_29 = key_29 = value_29 = attention_mask_30 = None
        transpose_119 = attn_output_116.transpose(1, 2)
        attn_output_116 = None
        attn_output_117 = transpose_119.contiguous()
        transpose_119 = None
        reshape_30 = attn_output_117.reshape(1, 10, -1)
        attn_output_117 = None
        attn_output_118 = reshape_30.contiguous()
        reshape_30 = None
        attn_output_119 = torch._C._nn.linear(
            attn_output_118,
            l_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_118 = l_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_225 = torch.nn.functional.dropout(
            attn_output_119, p=0.1, training=False
        )
        attn_output_119 = None
        hidden_states_226 = hidden_states_223 + hidden_states_225
        hidden_states_223 = hidden_states_225 = None
        hidden_states_227 = torch.nn.functional.layer_norm(
            hidden_states_226,
            (1024,),
            l_self_modules_decoder_modules_layers_modules_6_modules_final_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_6_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_layers_modules_6_modules_final_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_6_modules_final_layer_norm_parameters_bias_ = (None)
        linear_164 = torch._C._nn.linear(
            hidden_states_227,
            l_self_modules_decoder_modules_layers_modules_6_modules_fc1_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_6_modules_fc1_parameters_bias_,
        )
        hidden_states_227 = l_self_modules_decoder_modules_layers_modules_6_modules_fc1_parameters_weight_ = (
            l_self_modules_decoder_modules_layers_modules_6_modules_fc1_parameters_bias_
        ) = None
        hidden_states_228 = torch.nn.functional.relu(linear_164, inplace=False)
        linear_164 = None
        hidden_states_229 = torch.nn.functional.dropout(
            hidden_states_228, p=0.1, training=False
        )
        hidden_states_228 = None
        hidden_states_230 = torch._C._nn.linear(
            hidden_states_229,
            l_self_modules_decoder_modules_layers_modules_6_modules_fc2_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_6_modules_fc2_parameters_bias_,
        )
        hidden_states_229 = l_self_modules_decoder_modules_layers_modules_6_modules_fc2_parameters_weight_ = (
            l_self_modules_decoder_modules_layers_modules_6_modules_fc2_parameters_bias_
        ) = None
        hidden_states_231 = torch.nn.functional.dropout(
            hidden_states_230, p=0.1, training=False
        )
        hidden_states_230 = None
        hidden_states_232 = hidden_states_226 + hidden_states_231
        hidden_states_226 = hidden_states_231 = None
        hidden_states_233 = torch.nn.functional.layer_norm(
            hidden_states_232,
            (1024,),
            l_self_modules_decoder_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_166 = torch._C._nn.linear(
            hidden_states_233,
            l_self_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_92 = linear_166.view(1, 10, -1, 64)
        linear_166 = None
        query_states_30 = view_92.transpose(1, 2)
        view_92 = None
        key_states_60 = torch._C._nn.linear(
            hidden_states_233,
            l_self_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_60 = torch._C._nn.linear(
            hidden_states_233,
            l_self_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_233 = l_self_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_93 = key_states_60.view(1, 10, -1, 64)
        key_states_60 = None
        key_states_61 = view_93.transpose(1, 2)
        view_93 = None
        view_94 = value_states_60.view(1, 10, -1, 64)
        value_states_60 = None
        value_states_61 = view_94.transpose(1, 2)
        view_94 = None
        attention_mask_31 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_30 = query_states_30.contiguous()
        query_states_30 = None
        key_30 = key_states_61.contiguous()
        key_states_61 = None
        value_30 = value_states_61.contiguous()
        value_states_61 = None
        attn_output_120 = torch._C._nn.scaled_dot_product_attention(
            query_30,
            key_30,
            value_30,
            attn_mask=attention_mask_31,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_30 = key_30 = value_30 = attention_mask_31 = None
        transpose_123 = attn_output_120.transpose(1, 2)
        attn_output_120 = None
        attn_output_121 = transpose_123.contiguous()
        transpose_123 = None
        reshape_31 = attn_output_121.reshape(1, 10, -1)
        attn_output_121 = None
        attn_output_122 = reshape_31.contiguous()
        reshape_31 = None
        attn_output_123 = torch._C._nn.linear(
            attn_output_122,
            l_self_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_122 = l_self_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_234 = torch.nn.functional.dropout(
            attn_output_123, p=0.1, training=False
        )
        attn_output_123 = None
        hidden_states_235 = hidden_states_232 + hidden_states_234
        hidden_states_232 = hidden_states_234 = None
        hidden_states_236 = torch.nn.functional.layer_norm(
            hidden_states_235,
            (1024,),
            l_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_170 = torch._C._nn.linear(
            hidden_states_236,
            l_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        hidden_states_236 = l_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        view_95 = linear_170.view(1, 10, -1, 64)
        linear_170 = None
        query_states_31 = view_95.transpose(1, 2)
        view_95 = None
        key_states_62 = torch._C._nn.linear(
            hidden_states_146,
            l_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_62 = torch._C._nn.linear(
            hidden_states_146,
            l_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_96 = key_states_62.view(1, 10, -1, 64)
        key_states_62 = None
        key_states_63 = view_96.transpose(1, 2)
        view_96 = None
        view_97 = value_states_62.view(1, 10, -1, 64)
        value_states_62 = None
        value_states_63 = view_97.transpose(1, 2)
        view_97 = None
        attention_mask_32 = encoder_attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_31 = query_states_31.contiguous()
        query_states_31 = None
        key_31 = key_states_63.contiguous()
        key_states_63 = None
        value_31 = value_states_63.contiguous()
        value_states_63 = None
        attn_output_124 = torch._C._nn.scaled_dot_product_attention(
            query_31,
            key_31,
            value_31,
            attn_mask=attention_mask_32,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_31 = key_31 = value_31 = attention_mask_32 = None
        transpose_127 = attn_output_124.transpose(1, 2)
        attn_output_124 = None
        attn_output_125 = transpose_127.contiguous()
        transpose_127 = None
        reshape_32 = attn_output_125.reshape(1, 10, -1)
        attn_output_125 = None
        attn_output_126 = reshape_32.contiguous()
        reshape_32 = None
        attn_output_127 = torch._C._nn.linear(
            attn_output_126,
            l_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_126 = l_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_237 = torch.nn.functional.dropout(
            attn_output_127, p=0.1, training=False
        )
        attn_output_127 = None
        hidden_states_238 = hidden_states_235 + hidden_states_237
        hidden_states_235 = hidden_states_237 = None
        hidden_states_239 = torch.nn.functional.layer_norm(
            hidden_states_238,
            (1024,),
            l_self_modules_decoder_modules_layers_modules_7_modules_final_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_7_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_layers_modules_7_modules_final_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_7_modules_final_layer_norm_parameters_bias_ = (None)
        linear_174 = torch._C._nn.linear(
            hidden_states_239,
            l_self_modules_decoder_modules_layers_modules_7_modules_fc1_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_7_modules_fc1_parameters_bias_,
        )
        hidden_states_239 = l_self_modules_decoder_modules_layers_modules_7_modules_fc1_parameters_weight_ = (
            l_self_modules_decoder_modules_layers_modules_7_modules_fc1_parameters_bias_
        ) = None
        hidden_states_240 = torch.nn.functional.relu(linear_174, inplace=False)
        linear_174 = None
        hidden_states_241 = torch.nn.functional.dropout(
            hidden_states_240, p=0.1, training=False
        )
        hidden_states_240 = None
        hidden_states_242 = torch._C._nn.linear(
            hidden_states_241,
            l_self_modules_decoder_modules_layers_modules_7_modules_fc2_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_7_modules_fc2_parameters_bias_,
        )
        hidden_states_241 = l_self_modules_decoder_modules_layers_modules_7_modules_fc2_parameters_weight_ = (
            l_self_modules_decoder_modules_layers_modules_7_modules_fc2_parameters_bias_
        ) = None
        hidden_states_243 = torch.nn.functional.dropout(
            hidden_states_242, p=0.1, training=False
        )
        hidden_states_242 = None
        hidden_states_244 = hidden_states_238 + hidden_states_243
        hidden_states_238 = hidden_states_243 = None
        hidden_states_245 = torch.nn.functional.layer_norm(
            hidden_states_244,
            (1024,),
            l_self_modules_decoder_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_176 = torch._C._nn.linear(
            hidden_states_245,
            l_self_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_98 = linear_176.view(1, 10, -1, 64)
        linear_176 = None
        query_states_32 = view_98.transpose(1, 2)
        view_98 = None
        key_states_64 = torch._C._nn.linear(
            hidden_states_245,
            l_self_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_64 = torch._C._nn.linear(
            hidden_states_245,
            l_self_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_245 = l_self_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_99 = key_states_64.view(1, 10, -1, 64)
        key_states_64 = None
        key_states_65 = view_99.transpose(1, 2)
        view_99 = None
        view_100 = value_states_64.view(1, 10, -1, 64)
        value_states_64 = None
        value_states_65 = view_100.transpose(1, 2)
        view_100 = None
        attention_mask_33 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_32 = query_states_32.contiguous()
        query_states_32 = None
        key_32 = key_states_65.contiguous()
        key_states_65 = None
        value_32 = value_states_65.contiguous()
        value_states_65 = None
        attn_output_128 = torch._C._nn.scaled_dot_product_attention(
            query_32,
            key_32,
            value_32,
            attn_mask=attention_mask_33,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_32 = key_32 = value_32 = attention_mask_33 = None
        transpose_131 = attn_output_128.transpose(1, 2)
        attn_output_128 = None
        attn_output_129 = transpose_131.contiguous()
        transpose_131 = None
        reshape_33 = attn_output_129.reshape(1, 10, -1)
        attn_output_129 = None
        attn_output_130 = reshape_33.contiguous()
        reshape_33 = None
        attn_output_131 = torch._C._nn.linear(
            attn_output_130,
            l_self_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_130 = l_self_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_246 = torch.nn.functional.dropout(
            attn_output_131, p=0.1, training=False
        )
        attn_output_131 = None
        hidden_states_247 = hidden_states_244 + hidden_states_246
        hidden_states_244 = hidden_states_246 = None
        hidden_states_248 = torch.nn.functional.layer_norm(
            hidden_states_247,
            (1024,),
            l_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_180 = torch._C._nn.linear(
            hidden_states_248,
            l_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        hidden_states_248 = l_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        view_101 = linear_180.view(1, 10, -1, 64)
        linear_180 = None
        query_states_33 = view_101.transpose(1, 2)
        view_101 = None
        key_states_66 = torch._C._nn.linear(
            hidden_states_146,
            l_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_66 = torch._C._nn.linear(
            hidden_states_146,
            l_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_102 = key_states_66.view(1, 10, -1, 64)
        key_states_66 = None
        key_states_67 = view_102.transpose(1, 2)
        view_102 = None
        view_103 = value_states_66.view(1, 10, -1, 64)
        value_states_66 = None
        value_states_67 = view_103.transpose(1, 2)
        view_103 = None
        attention_mask_34 = encoder_attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_33 = query_states_33.contiguous()
        query_states_33 = None
        key_33 = key_states_67.contiguous()
        key_states_67 = None
        value_33 = value_states_67.contiguous()
        value_states_67 = None
        attn_output_132 = torch._C._nn.scaled_dot_product_attention(
            query_33,
            key_33,
            value_33,
            attn_mask=attention_mask_34,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_33 = key_33 = value_33 = attention_mask_34 = None
        transpose_135 = attn_output_132.transpose(1, 2)
        attn_output_132 = None
        attn_output_133 = transpose_135.contiguous()
        transpose_135 = None
        reshape_34 = attn_output_133.reshape(1, 10, -1)
        attn_output_133 = None
        attn_output_134 = reshape_34.contiguous()
        reshape_34 = None
        attn_output_135 = torch._C._nn.linear(
            attn_output_134,
            l_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_134 = l_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_249 = torch.nn.functional.dropout(
            attn_output_135, p=0.1, training=False
        )
        attn_output_135 = None
        hidden_states_250 = hidden_states_247 + hidden_states_249
        hidden_states_247 = hidden_states_249 = None
        hidden_states_251 = torch.nn.functional.layer_norm(
            hidden_states_250,
            (1024,),
            l_self_modules_decoder_modules_layers_modules_8_modules_final_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_8_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_layers_modules_8_modules_final_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_8_modules_final_layer_norm_parameters_bias_ = (None)
        linear_184 = torch._C._nn.linear(
            hidden_states_251,
            l_self_modules_decoder_modules_layers_modules_8_modules_fc1_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_8_modules_fc1_parameters_bias_,
        )
        hidden_states_251 = l_self_modules_decoder_modules_layers_modules_8_modules_fc1_parameters_weight_ = (
            l_self_modules_decoder_modules_layers_modules_8_modules_fc1_parameters_bias_
        ) = None
        hidden_states_252 = torch.nn.functional.relu(linear_184, inplace=False)
        linear_184 = None
        hidden_states_253 = torch.nn.functional.dropout(
            hidden_states_252, p=0.1, training=False
        )
        hidden_states_252 = None
        hidden_states_254 = torch._C._nn.linear(
            hidden_states_253,
            l_self_modules_decoder_modules_layers_modules_8_modules_fc2_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_8_modules_fc2_parameters_bias_,
        )
        hidden_states_253 = l_self_modules_decoder_modules_layers_modules_8_modules_fc2_parameters_weight_ = (
            l_self_modules_decoder_modules_layers_modules_8_modules_fc2_parameters_bias_
        ) = None
        hidden_states_255 = torch.nn.functional.dropout(
            hidden_states_254, p=0.1, training=False
        )
        hidden_states_254 = None
        hidden_states_256 = hidden_states_250 + hidden_states_255
        hidden_states_250 = hidden_states_255 = None
        hidden_states_257 = torch.nn.functional.layer_norm(
            hidden_states_256,
            (1024,),
            l_self_modules_decoder_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_186 = torch._C._nn.linear(
            hidden_states_257,
            l_self_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_104 = linear_186.view(1, 10, -1, 64)
        linear_186 = None
        query_states_34 = view_104.transpose(1, 2)
        view_104 = None
        key_states_68 = torch._C._nn.linear(
            hidden_states_257,
            l_self_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_68 = torch._C._nn.linear(
            hidden_states_257,
            l_self_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_257 = l_self_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_105 = key_states_68.view(1, 10, -1, 64)
        key_states_68 = None
        key_states_69 = view_105.transpose(1, 2)
        view_105 = None
        view_106 = value_states_68.view(1, 10, -1, 64)
        value_states_68 = None
        value_states_69 = view_106.transpose(1, 2)
        view_106 = None
        attention_mask_35 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_34 = query_states_34.contiguous()
        query_states_34 = None
        key_34 = key_states_69.contiguous()
        key_states_69 = None
        value_34 = value_states_69.contiguous()
        value_states_69 = None
        attn_output_136 = torch._C._nn.scaled_dot_product_attention(
            query_34,
            key_34,
            value_34,
            attn_mask=attention_mask_35,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_34 = key_34 = value_34 = attention_mask_35 = None
        transpose_139 = attn_output_136.transpose(1, 2)
        attn_output_136 = None
        attn_output_137 = transpose_139.contiguous()
        transpose_139 = None
        reshape_35 = attn_output_137.reshape(1, 10, -1)
        attn_output_137 = None
        attn_output_138 = reshape_35.contiguous()
        reshape_35 = None
        attn_output_139 = torch._C._nn.linear(
            attn_output_138,
            l_self_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_138 = l_self_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_258 = torch.nn.functional.dropout(
            attn_output_139, p=0.1, training=False
        )
        attn_output_139 = None
        hidden_states_259 = hidden_states_256 + hidden_states_258
        hidden_states_256 = hidden_states_258 = None
        hidden_states_260 = torch.nn.functional.layer_norm(
            hidden_states_259,
            (1024,),
            l_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_190 = torch._C._nn.linear(
            hidden_states_260,
            l_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        hidden_states_260 = l_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        view_107 = linear_190.view(1, 10, -1, 64)
        linear_190 = None
        query_states_35 = view_107.transpose(1, 2)
        view_107 = None
        key_states_70 = torch._C._nn.linear(
            hidden_states_146,
            l_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_70 = torch._C._nn.linear(
            hidden_states_146,
            l_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_108 = key_states_70.view(1, 10, -1, 64)
        key_states_70 = None
        key_states_71 = view_108.transpose(1, 2)
        view_108 = None
        view_109 = value_states_70.view(1, 10, -1, 64)
        value_states_70 = None
        value_states_71 = view_109.transpose(1, 2)
        view_109 = None
        attention_mask_36 = encoder_attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_35 = query_states_35.contiguous()
        query_states_35 = None
        key_35 = key_states_71.contiguous()
        key_states_71 = None
        value_35 = value_states_71.contiguous()
        value_states_71 = None
        attn_output_140 = torch._C._nn.scaled_dot_product_attention(
            query_35,
            key_35,
            value_35,
            attn_mask=attention_mask_36,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_35 = key_35 = value_35 = attention_mask_36 = None
        transpose_143 = attn_output_140.transpose(1, 2)
        attn_output_140 = None
        attn_output_141 = transpose_143.contiguous()
        transpose_143 = None
        reshape_36 = attn_output_141.reshape(1, 10, -1)
        attn_output_141 = None
        attn_output_142 = reshape_36.contiguous()
        reshape_36 = None
        attn_output_143 = torch._C._nn.linear(
            attn_output_142,
            l_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_142 = l_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_261 = torch.nn.functional.dropout(
            attn_output_143, p=0.1, training=False
        )
        attn_output_143 = None
        hidden_states_262 = hidden_states_259 + hidden_states_261
        hidden_states_259 = hidden_states_261 = None
        hidden_states_263 = torch.nn.functional.layer_norm(
            hidden_states_262,
            (1024,),
            l_self_modules_decoder_modules_layers_modules_9_modules_final_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_9_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_layers_modules_9_modules_final_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_9_modules_final_layer_norm_parameters_bias_ = (None)
        linear_194 = torch._C._nn.linear(
            hidden_states_263,
            l_self_modules_decoder_modules_layers_modules_9_modules_fc1_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_9_modules_fc1_parameters_bias_,
        )
        hidden_states_263 = l_self_modules_decoder_modules_layers_modules_9_modules_fc1_parameters_weight_ = (
            l_self_modules_decoder_modules_layers_modules_9_modules_fc1_parameters_bias_
        ) = None
        hidden_states_264 = torch.nn.functional.relu(linear_194, inplace=False)
        linear_194 = None
        hidden_states_265 = torch.nn.functional.dropout(
            hidden_states_264, p=0.1, training=False
        )
        hidden_states_264 = None
        hidden_states_266 = torch._C._nn.linear(
            hidden_states_265,
            l_self_modules_decoder_modules_layers_modules_9_modules_fc2_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_9_modules_fc2_parameters_bias_,
        )
        hidden_states_265 = l_self_modules_decoder_modules_layers_modules_9_modules_fc2_parameters_weight_ = (
            l_self_modules_decoder_modules_layers_modules_9_modules_fc2_parameters_bias_
        ) = None
        hidden_states_267 = torch.nn.functional.dropout(
            hidden_states_266, p=0.1, training=False
        )
        hidden_states_266 = None
        hidden_states_268 = hidden_states_262 + hidden_states_267
        hidden_states_262 = hidden_states_267 = None
        hidden_states_269 = torch.nn.functional.layer_norm(
            hidden_states_268,
            (1024,),
            l_self_modules_decoder_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_196 = torch._C._nn.linear(
            hidden_states_269,
            l_self_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_110 = linear_196.view(1, 10, -1, 64)
        linear_196 = None
        query_states_36 = view_110.transpose(1, 2)
        view_110 = None
        key_states_72 = torch._C._nn.linear(
            hidden_states_269,
            l_self_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_72 = torch._C._nn.linear(
            hidden_states_269,
            l_self_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_269 = l_self_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_111 = key_states_72.view(1, 10, -1, 64)
        key_states_72 = None
        key_states_73 = view_111.transpose(1, 2)
        view_111 = None
        view_112 = value_states_72.view(1, 10, -1, 64)
        value_states_72 = None
        value_states_73 = view_112.transpose(1, 2)
        view_112 = None
        attention_mask_37 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_36 = query_states_36.contiguous()
        query_states_36 = None
        key_36 = key_states_73.contiguous()
        key_states_73 = None
        value_36 = value_states_73.contiguous()
        value_states_73 = None
        attn_output_144 = torch._C._nn.scaled_dot_product_attention(
            query_36,
            key_36,
            value_36,
            attn_mask=attention_mask_37,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_36 = key_36 = value_36 = attention_mask_37 = None
        transpose_147 = attn_output_144.transpose(1, 2)
        attn_output_144 = None
        attn_output_145 = transpose_147.contiguous()
        transpose_147 = None
        reshape_37 = attn_output_145.reshape(1, 10, -1)
        attn_output_145 = None
        attn_output_146 = reshape_37.contiguous()
        reshape_37 = None
        attn_output_147 = torch._C._nn.linear(
            attn_output_146,
            l_self_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_146 = l_self_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_270 = torch.nn.functional.dropout(
            attn_output_147, p=0.1, training=False
        )
        attn_output_147 = None
        hidden_states_271 = hidden_states_268 + hidden_states_270
        hidden_states_268 = hidden_states_270 = None
        hidden_states_272 = torch.nn.functional.layer_norm(
            hidden_states_271,
            (1024,),
            l_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_200 = torch._C._nn.linear(
            hidden_states_272,
            l_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        hidden_states_272 = l_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        view_113 = linear_200.view(1, 10, -1, 64)
        linear_200 = None
        query_states_37 = view_113.transpose(1, 2)
        view_113 = None
        key_states_74 = torch._C._nn.linear(
            hidden_states_146,
            l_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_74 = torch._C._nn.linear(
            hidden_states_146,
            l_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_114 = key_states_74.view(1, 10, -1, 64)
        key_states_74 = None
        key_states_75 = view_114.transpose(1, 2)
        view_114 = None
        view_115 = value_states_74.view(1, 10, -1, 64)
        value_states_74 = None
        value_states_75 = view_115.transpose(1, 2)
        view_115 = None
        attention_mask_38 = encoder_attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_37 = query_states_37.contiguous()
        query_states_37 = None
        key_37 = key_states_75.contiguous()
        key_states_75 = None
        value_37 = value_states_75.contiguous()
        value_states_75 = None
        attn_output_148 = torch._C._nn.scaled_dot_product_attention(
            query_37,
            key_37,
            value_37,
            attn_mask=attention_mask_38,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_37 = key_37 = value_37 = attention_mask_38 = None
        transpose_151 = attn_output_148.transpose(1, 2)
        attn_output_148 = None
        attn_output_149 = transpose_151.contiguous()
        transpose_151 = None
        reshape_38 = attn_output_149.reshape(1, 10, -1)
        attn_output_149 = None
        attn_output_150 = reshape_38.contiguous()
        reshape_38 = None
        attn_output_151 = torch._C._nn.linear(
            attn_output_150,
            l_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_150 = l_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_273 = torch.nn.functional.dropout(
            attn_output_151, p=0.1, training=False
        )
        attn_output_151 = None
        hidden_states_274 = hidden_states_271 + hidden_states_273
        hidden_states_271 = hidden_states_273 = None
        hidden_states_275 = torch.nn.functional.layer_norm(
            hidden_states_274,
            (1024,),
            l_self_modules_decoder_modules_layers_modules_10_modules_final_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_10_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_layers_modules_10_modules_final_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_10_modules_final_layer_norm_parameters_bias_ = (None)
        linear_204 = torch._C._nn.linear(
            hidden_states_275,
            l_self_modules_decoder_modules_layers_modules_10_modules_fc1_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_10_modules_fc1_parameters_bias_,
        )
        hidden_states_275 = l_self_modules_decoder_modules_layers_modules_10_modules_fc1_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_10_modules_fc1_parameters_bias_ = (None)
        hidden_states_276 = torch.nn.functional.relu(linear_204, inplace=False)
        linear_204 = None
        hidden_states_277 = torch.nn.functional.dropout(
            hidden_states_276, p=0.1, training=False
        )
        hidden_states_276 = None
        hidden_states_278 = torch._C._nn.linear(
            hidden_states_277,
            l_self_modules_decoder_modules_layers_modules_10_modules_fc2_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_10_modules_fc2_parameters_bias_,
        )
        hidden_states_277 = l_self_modules_decoder_modules_layers_modules_10_modules_fc2_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_10_modules_fc2_parameters_bias_ = (None)
        hidden_states_279 = torch.nn.functional.dropout(
            hidden_states_278, p=0.1, training=False
        )
        hidden_states_278 = None
        hidden_states_280 = hidden_states_274 + hidden_states_279
        hidden_states_274 = hidden_states_279 = None
        hidden_states_281 = torch.nn.functional.layer_norm(
            hidden_states_280,
            (1024,),
            l_self_modules_decoder_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_206 = torch._C._nn.linear(
            hidden_states_281,
            l_self_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_116 = linear_206.view(1, 10, -1, 64)
        linear_206 = None
        query_states_38 = view_116.transpose(1, 2)
        view_116 = None
        key_states_76 = torch._C._nn.linear(
            hidden_states_281,
            l_self_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_76 = torch._C._nn.linear(
            hidden_states_281,
            l_self_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_281 = l_self_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_117 = key_states_76.view(1, 10, -1, 64)
        key_states_76 = None
        key_states_77 = view_117.transpose(1, 2)
        view_117 = None
        view_118 = value_states_76.view(1, 10, -1, 64)
        value_states_76 = None
        value_states_77 = view_118.transpose(1, 2)
        view_118 = None
        attention_mask_39 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_38 = query_states_38.contiguous()
        query_states_38 = None
        key_38 = key_states_77.contiguous()
        key_states_77 = None
        value_38 = value_states_77.contiguous()
        value_states_77 = None
        attn_output_152 = torch._C._nn.scaled_dot_product_attention(
            query_38,
            key_38,
            value_38,
            attn_mask=attention_mask_39,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_38 = key_38 = value_38 = attention_mask_39 = None
        transpose_155 = attn_output_152.transpose(1, 2)
        attn_output_152 = None
        attn_output_153 = transpose_155.contiguous()
        transpose_155 = None
        reshape_39 = attn_output_153.reshape(1, 10, -1)
        attn_output_153 = None
        attn_output_154 = reshape_39.contiguous()
        reshape_39 = None
        attn_output_155 = torch._C._nn.linear(
            attn_output_154,
            l_self_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_154 = l_self_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_282 = torch.nn.functional.dropout(
            attn_output_155, p=0.1, training=False
        )
        attn_output_155 = None
        hidden_states_283 = hidden_states_280 + hidden_states_282
        hidden_states_280 = hidden_states_282 = None
        hidden_states_284 = torch.nn.functional.layer_norm(
            hidden_states_283,
            (1024,),
            l_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_210 = torch._C._nn.linear(
            hidden_states_284,
            l_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        hidden_states_284 = l_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        view_119 = linear_210.view(1, 10, -1, 64)
        linear_210 = None
        query_states_39 = view_119.transpose(1, 2)
        view_119 = None
        key_states_78 = torch._C._nn.linear(
            hidden_states_146,
            l_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_78 = torch._C._nn.linear(
            hidden_states_146,
            l_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_120 = key_states_78.view(1, 10, -1, 64)
        key_states_78 = None
        key_states_79 = view_120.transpose(1, 2)
        view_120 = None
        view_121 = value_states_78.view(1, 10, -1, 64)
        value_states_78 = None
        value_states_79 = view_121.transpose(1, 2)
        view_121 = None
        attention_mask_40 = encoder_attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_39 = query_states_39.contiguous()
        query_states_39 = None
        key_39 = key_states_79.contiguous()
        key_states_79 = None
        value_39 = value_states_79.contiguous()
        value_states_79 = None
        attn_output_156 = torch._C._nn.scaled_dot_product_attention(
            query_39,
            key_39,
            value_39,
            attn_mask=attention_mask_40,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_39 = key_39 = value_39 = attention_mask_40 = None
        transpose_159 = attn_output_156.transpose(1, 2)
        attn_output_156 = None
        attn_output_157 = transpose_159.contiguous()
        transpose_159 = None
        reshape_40 = attn_output_157.reshape(1, 10, -1)
        attn_output_157 = None
        attn_output_158 = reshape_40.contiguous()
        reshape_40 = None
        attn_output_159 = torch._C._nn.linear(
            attn_output_158,
            l_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_158 = l_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_285 = torch.nn.functional.dropout(
            attn_output_159, p=0.1, training=False
        )
        attn_output_159 = None
        hidden_states_286 = hidden_states_283 + hidden_states_285
        hidden_states_283 = hidden_states_285 = None
        hidden_states_287 = torch.nn.functional.layer_norm(
            hidden_states_286,
            (1024,),
            l_self_modules_decoder_modules_layers_modules_11_modules_final_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_11_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_layers_modules_11_modules_final_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_11_modules_final_layer_norm_parameters_bias_ = (None)
        linear_214 = torch._C._nn.linear(
            hidden_states_287,
            l_self_modules_decoder_modules_layers_modules_11_modules_fc1_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_11_modules_fc1_parameters_bias_,
        )
        hidden_states_287 = l_self_modules_decoder_modules_layers_modules_11_modules_fc1_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_11_modules_fc1_parameters_bias_ = (None)
        hidden_states_288 = torch.nn.functional.relu(linear_214, inplace=False)
        linear_214 = None
        hidden_states_289 = torch.nn.functional.dropout(
            hidden_states_288, p=0.1, training=False
        )
        hidden_states_288 = None
        hidden_states_290 = torch._C._nn.linear(
            hidden_states_289,
            l_self_modules_decoder_modules_layers_modules_11_modules_fc2_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_11_modules_fc2_parameters_bias_,
        )
        hidden_states_289 = l_self_modules_decoder_modules_layers_modules_11_modules_fc2_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_11_modules_fc2_parameters_bias_ = (None)
        hidden_states_291 = torch.nn.functional.dropout(
            hidden_states_290, p=0.1, training=False
        )
        hidden_states_290 = None
        hidden_states_292 = hidden_states_286 + hidden_states_291
        hidden_states_286 = hidden_states_291 = None
        hidden_states_293 = torch.nn.functional.layer_norm(
            hidden_states_292,
            (1024,),
            l_self_modules_decoder_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_216 = torch._C._nn.linear(
            hidden_states_293,
            l_self_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_122 = linear_216.view(1, 10, -1, 64)
        linear_216 = None
        query_states_40 = view_122.transpose(1, 2)
        view_122 = None
        key_states_80 = torch._C._nn.linear(
            hidden_states_293,
            l_self_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_80 = torch._C._nn.linear(
            hidden_states_293,
            l_self_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_293 = l_self_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_123 = key_states_80.view(1, 10, -1, 64)
        key_states_80 = None
        key_states_81 = view_123.transpose(1, 2)
        view_123 = None
        view_124 = value_states_80.view(1, 10, -1, 64)
        value_states_80 = None
        value_states_81 = view_124.transpose(1, 2)
        view_124 = None
        attention_mask_41 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_40 = query_states_40.contiguous()
        query_states_40 = None
        key_40 = key_states_81.contiguous()
        key_states_81 = None
        value_40 = value_states_81.contiguous()
        value_states_81 = None
        attn_output_160 = torch._C._nn.scaled_dot_product_attention(
            query_40,
            key_40,
            value_40,
            attn_mask=attention_mask_41,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_40 = key_40 = value_40 = attention_mask_41 = None
        transpose_163 = attn_output_160.transpose(1, 2)
        attn_output_160 = None
        attn_output_161 = transpose_163.contiguous()
        transpose_163 = None
        reshape_41 = attn_output_161.reshape(1, 10, -1)
        attn_output_161 = None
        attn_output_162 = reshape_41.contiguous()
        reshape_41 = None
        attn_output_163 = torch._C._nn.linear(
            attn_output_162,
            l_self_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_162 = l_self_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_294 = torch.nn.functional.dropout(
            attn_output_163, p=0.1, training=False
        )
        attn_output_163 = None
        hidden_states_295 = hidden_states_292 + hidden_states_294
        hidden_states_292 = hidden_states_294 = None
        hidden_states_296 = torch.nn.functional.layer_norm(
            hidden_states_295,
            (1024,),
            l_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_220 = torch._C._nn.linear(
            hidden_states_296,
            l_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        hidden_states_296 = l_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        view_125 = linear_220.view(1, 10, -1, 64)
        linear_220 = None
        query_states_41 = view_125.transpose(1, 2)
        view_125 = None
        key_states_82 = torch._C._nn.linear(
            hidden_states_146,
            l_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_82 = torch._C._nn.linear(
            hidden_states_146,
            l_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_126 = key_states_82.view(1, 10, -1, 64)
        key_states_82 = None
        key_states_83 = view_126.transpose(1, 2)
        view_126 = None
        view_127 = value_states_82.view(1, 10, -1, 64)
        value_states_82 = None
        value_states_83 = view_127.transpose(1, 2)
        view_127 = None
        attention_mask_42 = encoder_attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_41 = query_states_41.contiguous()
        query_states_41 = None
        key_41 = key_states_83.contiguous()
        key_states_83 = None
        value_41 = value_states_83.contiguous()
        value_states_83 = None
        attn_output_164 = torch._C._nn.scaled_dot_product_attention(
            query_41,
            key_41,
            value_41,
            attn_mask=attention_mask_42,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_41 = key_41 = value_41 = attention_mask_42 = None
        transpose_167 = attn_output_164.transpose(1, 2)
        attn_output_164 = None
        attn_output_165 = transpose_167.contiguous()
        transpose_167 = None
        reshape_42 = attn_output_165.reshape(1, 10, -1)
        attn_output_165 = None
        attn_output_166 = reshape_42.contiguous()
        reshape_42 = None
        attn_output_167 = torch._C._nn.linear(
            attn_output_166,
            l_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_166 = l_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_12_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_297 = torch.nn.functional.dropout(
            attn_output_167, p=0.1, training=False
        )
        attn_output_167 = None
        hidden_states_298 = hidden_states_295 + hidden_states_297
        hidden_states_295 = hidden_states_297 = None
        hidden_states_299 = torch.nn.functional.layer_norm(
            hidden_states_298,
            (1024,),
            l_self_modules_decoder_modules_layers_modules_12_modules_final_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_12_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_layers_modules_12_modules_final_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_12_modules_final_layer_norm_parameters_bias_ = (None)
        linear_224 = torch._C._nn.linear(
            hidden_states_299,
            l_self_modules_decoder_modules_layers_modules_12_modules_fc1_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_12_modules_fc1_parameters_bias_,
        )
        hidden_states_299 = l_self_modules_decoder_modules_layers_modules_12_modules_fc1_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_12_modules_fc1_parameters_bias_ = (None)
        hidden_states_300 = torch.nn.functional.relu(linear_224, inplace=False)
        linear_224 = None
        hidden_states_301 = torch.nn.functional.dropout(
            hidden_states_300, p=0.1, training=False
        )
        hidden_states_300 = None
        hidden_states_302 = torch._C._nn.linear(
            hidden_states_301,
            l_self_modules_decoder_modules_layers_modules_12_modules_fc2_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_12_modules_fc2_parameters_bias_,
        )
        hidden_states_301 = l_self_modules_decoder_modules_layers_modules_12_modules_fc2_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_12_modules_fc2_parameters_bias_ = (None)
        hidden_states_303 = torch.nn.functional.dropout(
            hidden_states_302, p=0.1, training=False
        )
        hidden_states_302 = None
        hidden_states_304 = hidden_states_298 + hidden_states_303
        hidden_states_298 = hidden_states_303 = None
        hidden_states_305 = torch.nn.functional.layer_norm(
            hidden_states_304,
            (1024,),
            l_self_modules_decoder_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_226 = torch._C._nn.linear(
            hidden_states_305,
            l_self_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_128 = linear_226.view(1, 10, -1, 64)
        linear_226 = None
        query_states_42 = view_128.transpose(1, 2)
        view_128 = None
        key_states_84 = torch._C._nn.linear(
            hidden_states_305,
            l_self_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_84 = torch._C._nn.linear(
            hidden_states_305,
            l_self_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_305 = l_self_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_129 = key_states_84.view(1, 10, -1, 64)
        key_states_84 = None
        key_states_85 = view_129.transpose(1, 2)
        view_129 = None
        view_130 = value_states_84.view(1, 10, -1, 64)
        value_states_84 = None
        value_states_85 = view_130.transpose(1, 2)
        view_130 = None
        attention_mask_43 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_42 = query_states_42.contiguous()
        query_states_42 = None
        key_42 = key_states_85.contiguous()
        key_states_85 = None
        value_42 = value_states_85.contiguous()
        value_states_85 = None
        attn_output_168 = torch._C._nn.scaled_dot_product_attention(
            query_42,
            key_42,
            value_42,
            attn_mask=attention_mask_43,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_42 = key_42 = value_42 = attention_mask_43 = None
        transpose_171 = attn_output_168.transpose(1, 2)
        attn_output_168 = None
        attn_output_169 = transpose_171.contiguous()
        transpose_171 = None
        reshape_43 = attn_output_169.reshape(1, 10, -1)
        attn_output_169 = None
        attn_output_170 = reshape_43.contiguous()
        reshape_43 = None
        attn_output_171 = torch._C._nn.linear(
            attn_output_170,
            l_self_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_170 = l_self_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_306 = torch.nn.functional.dropout(
            attn_output_171, p=0.1, training=False
        )
        attn_output_171 = None
        hidden_states_307 = hidden_states_304 + hidden_states_306
        hidden_states_304 = hidden_states_306 = None
        hidden_states_308 = torch.nn.functional.layer_norm(
            hidden_states_307,
            (1024,),
            l_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_230 = torch._C._nn.linear(
            hidden_states_308,
            l_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        hidden_states_308 = l_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        view_131 = linear_230.view(1, 10, -1, 64)
        linear_230 = None
        query_states_43 = view_131.transpose(1, 2)
        view_131 = None
        key_states_86 = torch._C._nn.linear(
            hidden_states_146,
            l_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_86 = torch._C._nn.linear(
            hidden_states_146,
            l_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_132 = key_states_86.view(1, 10, -1, 64)
        key_states_86 = None
        key_states_87 = view_132.transpose(1, 2)
        view_132 = None
        view_133 = value_states_86.view(1, 10, -1, 64)
        value_states_86 = None
        value_states_87 = view_133.transpose(1, 2)
        view_133 = None
        attention_mask_44 = encoder_attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_43 = query_states_43.contiguous()
        query_states_43 = None
        key_43 = key_states_87.contiguous()
        key_states_87 = None
        value_43 = value_states_87.contiguous()
        value_states_87 = None
        attn_output_172 = torch._C._nn.scaled_dot_product_attention(
            query_43,
            key_43,
            value_43,
            attn_mask=attention_mask_44,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_43 = key_43 = value_43 = attention_mask_44 = None
        transpose_175 = attn_output_172.transpose(1, 2)
        attn_output_172 = None
        attn_output_173 = transpose_175.contiguous()
        transpose_175 = None
        reshape_44 = attn_output_173.reshape(1, 10, -1)
        attn_output_173 = None
        attn_output_174 = reshape_44.contiguous()
        reshape_44 = None
        attn_output_175 = torch._C._nn.linear(
            attn_output_174,
            l_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_174 = l_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_13_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_309 = torch.nn.functional.dropout(
            attn_output_175, p=0.1, training=False
        )
        attn_output_175 = None
        hidden_states_310 = hidden_states_307 + hidden_states_309
        hidden_states_307 = hidden_states_309 = None
        hidden_states_311 = torch.nn.functional.layer_norm(
            hidden_states_310,
            (1024,),
            l_self_modules_decoder_modules_layers_modules_13_modules_final_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_13_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_layers_modules_13_modules_final_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_13_modules_final_layer_norm_parameters_bias_ = (None)
        linear_234 = torch._C._nn.linear(
            hidden_states_311,
            l_self_modules_decoder_modules_layers_modules_13_modules_fc1_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_13_modules_fc1_parameters_bias_,
        )
        hidden_states_311 = l_self_modules_decoder_modules_layers_modules_13_modules_fc1_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_13_modules_fc1_parameters_bias_ = (None)
        hidden_states_312 = torch.nn.functional.relu(linear_234, inplace=False)
        linear_234 = None
        hidden_states_313 = torch.nn.functional.dropout(
            hidden_states_312, p=0.1, training=False
        )
        hidden_states_312 = None
        hidden_states_314 = torch._C._nn.linear(
            hidden_states_313,
            l_self_modules_decoder_modules_layers_modules_13_modules_fc2_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_13_modules_fc2_parameters_bias_,
        )
        hidden_states_313 = l_self_modules_decoder_modules_layers_modules_13_modules_fc2_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_13_modules_fc2_parameters_bias_ = (None)
        hidden_states_315 = torch.nn.functional.dropout(
            hidden_states_314, p=0.1, training=False
        )
        hidden_states_314 = None
        hidden_states_316 = hidden_states_310 + hidden_states_315
        hidden_states_310 = hidden_states_315 = None
        hidden_states_317 = torch.nn.functional.layer_norm(
            hidden_states_316,
            (1024,),
            l_self_modules_decoder_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_236 = torch._C._nn.linear(
            hidden_states_317,
            l_self_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_134 = linear_236.view(1, 10, -1, 64)
        linear_236 = None
        query_states_44 = view_134.transpose(1, 2)
        view_134 = None
        key_states_88 = torch._C._nn.linear(
            hidden_states_317,
            l_self_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_88 = torch._C._nn.linear(
            hidden_states_317,
            l_self_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_317 = l_self_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_135 = key_states_88.view(1, 10, -1, 64)
        key_states_88 = None
        key_states_89 = view_135.transpose(1, 2)
        view_135 = None
        view_136 = value_states_88.view(1, 10, -1, 64)
        value_states_88 = None
        value_states_89 = view_136.transpose(1, 2)
        view_136 = None
        attention_mask_45 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_44 = query_states_44.contiguous()
        query_states_44 = None
        key_44 = key_states_89.contiguous()
        key_states_89 = None
        value_44 = value_states_89.contiguous()
        value_states_89 = None
        attn_output_176 = torch._C._nn.scaled_dot_product_attention(
            query_44,
            key_44,
            value_44,
            attn_mask=attention_mask_45,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_44 = key_44 = value_44 = attention_mask_45 = None
        transpose_179 = attn_output_176.transpose(1, 2)
        attn_output_176 = None
        attn_output_177 = transpose_179.contiguous()
        transpose_179 = None
        reshape_45 = attn_output_177.reshape(1, 10, -1)
        attn_output_177 = None
        attn_output_178 = reshape_45.contiguous()
        reshape_45 = None
        attn_output_179 = torch._C._nn.linear(
            attn_output_178,
            l_self_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_178 = l_self_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_318 = torch.nn.functional.dropout(
            attn_output_179, p=0.1, training=False
        )
        attn_output_179 = None
        hidden_states_319 = hidden_states_316 + hidden_states_318
        hidden_states_316 = hidden_states_318 = None
        hidden_states_320 = torch.nn.functional.layer_norm(
            hidden_states_319,
            (1024,),
            l_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_240 = torch._C._nn.linear(
            hidden_states_320,
            l_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        hidden_states_320 = l_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        view_137 = linear_240.view(1, 10, -1, 64)
        linear_240 = None
        query_states_45 = view_137.transpose(1, 2)
        view_137 = None
        key_states_90 = torch._C._nn.linear(
            hidden_states_146,
            l_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_90 = torch._C._nn.linear(
            hidden_states_146,
            l_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_138 = key_states_90.view(1, 10, -1, 64)
        key_states_90 = None
        key_states_91 = view_138.transpose(1, 2)
        view_138 = None
        view_139 = value_states_90.view(1, 10, -1, 64)
        value_states_90 = None
        value_states_91 = view_139.transpose(1, 2)
        view_139 = None
        attention_mask_46 = encoder_attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_45 = query_states_45.contiguous()
        query_states_45 = None
        key_45 = key_states_91.contiguous()
        key_states_91 = None
        value_45 = value_states_91.contiguous()
        value_states_91 = None
        attn_output_180 = torch._C._nn.scaled_dot_product_attention(
            query_45,
            key_45,
            value_45,
            attn_mask=attention_mask_46,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_45 = key_45 = value_45 = attention_mask_46 = None
        transpose_183 = attn_output_180.transpose(1, 2)
        attn_output_180 = None
        attn_output_181 = transpose_183.contiguous()
        transpose_183 = None
        reshape_46 = attn_output_181.reshape(1, 10, -1)
        attn_output_181 = None
        attn_output_182 = reshape_46.contiguous()
        reshape_46 = None
        attn_output_183 = torch._C._nn.linear(
            attn_output_182,
            l_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_182 = l_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_14_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_321 = torch.nn.functional.dropout(
            attn_output_183, p=0.1, training=False
        )
        attn_output_183 = None
        hidden_states_322 = hidden_states_319 + hidden_states_321
        hidden_states_319 = hidden_states_321 = None
        hidden_states_323 = torch.nn.functional.layer_norm(
            hidden_states_322,
            (1024,),
            l_self_modules_decoder_modules_layers_modules_14_modules_final_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_14_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_layers_modules_14_modules_final_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_14_modules_final_layer_norm_parameters_bias_ = (None)
        linear_244 = torch._C._nn.linear(
            hidden_states_323,
            l_self_modules_decoder_modules_layers_modules_14_modules_fc1_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_14_modules_fc1_parameters_bias_,
        )
        hidden_states_323 = l_self_modules_decoder_modules_layers_modules_14_modules_fc1_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_14_modules_fc1_parameters_bias_ = (None)
        hidden_states_324 = torch.nn.functional.relu(linear_244, inplace=False)
        linear_244 = None
        hidden_states_325 = torch.nn.functional.dropout(
            hidden_states_324, p=0.1, training=False
        )
        hidden_states_324 = None
        hidden_states_326 = torch._C._nn.linear(
            hidden_states_325,
            l_self_modules_decoder_modules_layers_modules_14_modules_fc2_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_14_modules_fc2_parameters_bias_,
        )
        hidden_states_325 = l_self_modules_decoder_modules_layers_modules_14_modules_fc2_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_14_modules_fc2_parameters_bias_ = (None)
        hidden_states_327 = torch.nn.functional.dropout(
            hidden_states_326, p=0.1, training=False
        )
        hidden_states_326 = None
        hidden_states_328 = hidden_states_322 + hidden_states_327
        hidden_states_322 = hidden_states_327 = None
        hidden_states_329 = torch.nn.functional.layer_norm(
            hidden_states_328,
            (1024,),
            l_self_modules_decoder_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_246 = torch._C._nn.linear(
            hidden_states_329,
            l_self_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_140 = linear_246.view(1, 10, -1, 64)
        linear_246 = None
        query_states_46 = view_140.transpose(1, 2)
        view_140 = None
        key_states_92 = torch._C._nn.linear(
            hidden_states_329,
            l_self_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_92 = torch._C._nn.linear(
            hidden_states_329,
            l_self_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_329 = l_self_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_141 = key_states_92.view(1, 10, -1, 64)
        key_states_92 = None
        key_states_93 = view_141.transpose(1, 2)
        view_141 = None
        view_142 = value_states_92.view(1, 10, -1, 64)
        value_states_92 = None
        value_states_93 = view_142.transpose(1, 2)
        view_142 = None
        attention_mask_47 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        causal_mask_3 = None
        query_46 = query_states_46.contiguous()
        query_states_46 = None
        key_46 = key_states_93.contiguous()
        key_states_93 = None
        value_46 = value_states_93.contiguous()
        value_states_93 = None
        attn_output_184 = torch._C._nn.scaled_dot_product_attention(
            query_46,
            key_46,
            value_46,
            attn_mask=attention_mask_47,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_46 = key_46 = value_46 = attention_mask_47 = None
        transpose_187 = attn_output_184.transpose(1, 2)
        attn_output_184 = None
        attn_output_185 = transpose_187.contiguous()
        transpose_187 = None
        reshape_47 = attn_output_185.reshape(1, 10, -1)
        attn_output_185 = None
        attn_output_186 = reshape_47.contiguous()
        reshape_47 = None
        attn_output_187 = torch._C._nn.linear(
            attn_output_186,
            l_self_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_186 = l_self_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_330 = torch.nn.functional.dropout(
            attn_output_187, p=0.1, training=False
        )
        attn_output_187 = None
        hidden_states_331 = hidden_states_328 + hidden_states_330
        hidden_states_328 = hidden_states_330 = None
        hidden_states_332 = torch.nn.functional.layer_norm(
            hidden_states_331,
            (1024,),
            l_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_250 = torch._C._nn.linear(
            hidden_states_332,
            l_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        hidden_states_332 = l_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        view_143 = linear_250.view(1, 10, -1, 64)
        linear_250 = None
        query_states_47 = view_143.transpose(1, 2)
        view_143 = None
        key_states_94 = torch._C._nn.linear(
            hidden_states_146,
            l_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_94 = torch._C._nn.linear(
            hidden_states_146,
            l_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_144 = key_states_94.view(1, 10, -1, 64)
        key_states_94 = None
        key_states_95 = view_144.transpose(1, 2)
        view_144 = None
        view_145 = value_states_94.view(1, 10, -1, 64)
        value_states_94 = None
        value_states_95 = view_145.transpose(1, 2)
        view_145 = None
        attention_mask_48 = encoder_attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        encoder_attention_mask = None
        query_47 = query_states_47.contiguous()
        query_states_47 = None
        key_47 = key_states_95.contiguous()
        key_states_95 = None
        value_47 = value_states_95.contiguous()
        value_states_95 = None
        attn_output_188 = torch._C._nn.scaled_dot_product_attention(
            query_47,
            key_47,
            value_47,
            attn_mask=attention_mask_48,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_47 = key_47 = value_47 = attention_mask_48 = None
        transpose_191 = attn_output_188.transpose(1, 2)
        attn_output_188 = None
        attn_output_189 = transpose_191.contiguous()
        transpose_191 = None
        reshape_48 = attn_output_189.reshape(1, 10, -1)
        attn_output_189 = None
        attn_output_190 = reshape_48.contiguous()
        reshape_48 = None
        attn_output_191 = torch._C._nn.linear(
            attn_output_190,
            l_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_190 = l_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_15_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_333 = torch.nn.functional.dropout(
            attn_output_191, p=0.1, training=False
        )
        attn_output_191 = None
        hidden_states_334 = hidden_states_331 + hidden_states_333
        hidden_states_331 = hidden_states_333 = None
        hidden_states_335 = torch.nn.functional.layer_norm(
            hidden_states_334,
            (1024,),
            l_self_modules_decoder_modules_layers_modules_15_modules_final_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_15_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_layers_modules_15_modules_final_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_15_modules_final_layer_norm_parameters_bias_ = (None)
        linear_254 = torch._C._nn.linear(
            hidden_states_335,
            l_self_modules_decoder_modules_layers_modules_15_modules_fc1_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_15_modules_fc1_parameters_bias_,
        )
        hidden_states_335 = l_self_modules_decoder_modules_layers_modules_15_modules_fc1_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_15_modules_fc1_parameters_bias_ = (None)
        hidden_states_336 = torch.nn.functional.relu(linear_254, inplace=False)
        linear_254 = None
        hidden_states_337 = torch.nn.functional.dropout(
            hidden_states_336, p=0.1, training=False
        )
        hidden_states_336 = None
        hidden_states_338 = torch._C._nn.linear(
            hidden_states_337,
            l_self_modules_decoder_modules_layers_modules_15_modules_fc2_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_15_modules_fc2_parameters_bias_,
        )
        hidden_states_337 = l_self_modules_decoder_modules_layers_modules_15_modules_fc2_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_15_modules_fc2_parameters_bias_ = (None)
        hidden_states_339 = torch.nn.functional.dropout(
            hidden_states_338, p=0.1, training=False
        )
        hidden_states_338 = None
        hidden_states_340 = hidden_states_334 + hidden_states_339
        hidden_states_334 = hidden_states_339 = None
        hidden_states_341 = torch.nn.functional.layer_norm(
            hidden_states_340,
            (1024,),
            l_self_modules_decoder_modules_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_340 = (
            l_self_modules_decoder_modules_layer_norm_parameters_weight_
        ) = l_self_modules_decoder_modules_layer_norm_parameters_bias_ = None
        return (hidden_states_341, hidden_states_146)
