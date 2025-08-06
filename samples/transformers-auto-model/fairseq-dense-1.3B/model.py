import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_input_ids_: torch.Tensor,
        L_self_modules_model_modules_embed_tokens_parameters_weight_: torch.nn.parameter.Parameter,
        L_attention_mask_: torch.Tensor,
        L_self_modules_model_modules_embed_positions_buffers_weights_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_input_ids_ = L_input_ids_
        l_self_modules_model_modules_embed_tokens_parameters_weight_ = (
            L_self_modules_model_modules_embed_tokens_parameters_weight_
        )
        l_attention_mask_ = L_attention_mask_
        l_self_modules_model_modules_embed_positions_buffers_weights_ = (
            L_self_modules_model_modules_embed_positions_buffers_weights_
        )
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_layers_modules_0_modules_fc1_parameters_weight_ = (
            L_self_modules_model_modules_layers_modules_0_modules_fc1_parameters_weight_
        )
        l_self_modules_model_modules_layers_modules_0_modules_fc1_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_0_modules_fc1_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_0_modules_fc2_parameters_weight_ = (
            L_self_modules_model_modules_layers_modules_0_modules_fc2_parameters_weight_
        )
        l_self_modules_model_modules_layers_modules_0_modules_fc2_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_0_modules_fc2_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_layers_modules_1_modules_fc1_parameters_weight_ = (
            L_self_modules_model_modules_layers_modules_1_modules_fc1_parameters_weight_
        )
        l_self_modules_model_modules_layers_modules_1_modules_fc1_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_1_modules_fc1_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_1_modules_fc2_parameters_weight_ = (
            L_self_modules_model_modules_layers_modules_1_modules_fc2_parameters_weight_
        )
        l_self_modules_model_modules_layers_modules_1_modules_fc2_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_1_modules_fc2_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_layers_modules_2_modules_fc1_parameters_weight_ = (
            L_self_modules_model_modules_layers_modules_2_modules_fc1_parameters_weight_
        )
        l_self_modules_model_modules_layers_modules_2_modules_fc1_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_2_modules_fc1_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_2_modules_fc2_parameters_weight_ = (
            L_self_modules_model_modules_layers_modules_2_modules_fc2_parameters_weight_
        )
        l_self_modules_model_modules_layers_modules_2_modules_fc2_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_2_modules_fc2_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_layers_modules_3_modules_fc1_parameters_weight_ = (
            L_self_modules_model_modules_layers_modules_3_modules_fc1_parameters_weight_
        )
        l_self_modules_model_modules_layers_modules_3_modules_fc1_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_3_modules_fc1_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_3_modules_fc2_parameters_weight_ = (
            L_self_modules_model_modules_layers_modules_3_modules_fc2_parameters_weight_
        )
        l_self_modules_model_modules_layers_modules_3_modules_fc2_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_3_modules_fc2_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_layers_modules_4_modules_fc1_parameters_weight_ = (
            L_self_modules_model_modules_layers_modules_4_modules_fc1_parameters_weight_
        )
        l_self_modules_model_modules_layers_modules_4_modules_fc1_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_4_modules_fc1_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_4_modules_fc2_parameters_weight_ = (
            L_self_modules_model_modules_layers_modules_4_modules_fc2_parameters_weight_
        )
        l_self_modules_model_modules_layers_modules_4_modules_fc2_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_4_modules_fc2_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_layers_modules_5_modules_fc1_parameters_weight_ = (
            L_self_modules_model_modules_layers_modules_5_modules_fc1_parameters_weight_
        )
        l_self_modules_model_modules_layers_modules_5_modules_fc1_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_5_modules_fc1_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_5_modules_fc2_parameters_weight_ = (
            L_self_modules_model_modules_layers_modules_5_modules_fc2_parameters_weight_
        )
        l_self_modules_model_modules_layers_modules_5_modules_fc2_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_5_modules_fc2_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_6_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_layers_modules_6_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_layers_modules_6_modules_fc1_parameters_weight_ = (
            L_self_modules_model_modules_layers_modules_6_modules_fc1_parameters_weight_
        )
        l_self_modules_model_modules_layers_modules_6_modules_fc1_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_6_modules_fc1_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_6_modules_fc2_parameters_weight_ = (
            L_self_modules_model_modules_layers_modules_6_modules_fc2_parameters_weight_
        )
        l_self_modules_model_modules_layers_modules_6_modules_fc2_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_6_modules_fc2_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_7_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_layers_modules_7_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_layers_modules_7_modules_fc1_parameters_weight_ = (
            L_self_modules_model_modules_layers_modules_7_modules_fc1_parameters_weight_
        )
        l_self_modules_model_modules_layers_modules_7_modules_fc1_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_7_modules_fc1_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_7_modules_fc2_parameters_weight_ = (
            L_self_modules_model_modules_layers_modules_7_modules_fc2_parameters_weight_
        )
        l_self_modules_model_modules_layers_modules_7_modules_fc2_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_7_modules_fc2_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_8_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_layers_modules_8_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_layers_modules_8_modules_fc1_parameters_weight_ = (
            L_self_modules_model_modules_layers_modules_8_modules_fc1_parameters_weight_
        )
        l_self_modules_model_modules_layers_modules_8_modules_fc1_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_8_modules_fc1_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_8_modules_fc2_parameters_weight_ = (
            L_self_modules_model_modules_layers_modules_8_modules_fc2_parameters_weight_
        )
        l_self_modules_model_modules_layers_modules_8_modules_fc2_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_8_modules_fc2_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_9_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_layers_modules_9_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_layers_modules_9_modules_fc1_parameters_weight_ = (
            L_self_modules_model_modules_layers_modules_9_modules_fc1_parameters_weight_
        )
        l_self_modules_model_modules_layers_modules_9_modules_fc1_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_9_modules_fc1_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_9_modules_fc2_parameters_weight_ = (
            L_self_modules_model_modules_layers_modules_9_modules_fc2_parameters_weight_
        )
        l_self_modules_model_modules_layers_modules_9_modules_fc2_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_9_modules_fc2_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_10_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_layers_modules_10_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_layers_modules_10_modules_fc1_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_fc1_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_fc1_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_10_modules_fc1_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_10_modules_fc2_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_fc2_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_fc2_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_10_modules_fc2_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_11_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_layers_modules_11_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_layers_modules_11_modules_fc1_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_fc1_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_fc1_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_11_modules_fc1_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_11_modules_fc2_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_fc2_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_fc2_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_11_modules_fc2_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_12_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_layers_modules_12_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_layers_modules_12_modules_fc1_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_fc1_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_fc1_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_12_modules_fc1_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_12_modules_fc2_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_fc2_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_fc2_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_12_modules_fc2_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_13_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_layers_modules_13_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_layers_modules_13_modules_fc1_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_fc1_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_fc1_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_13_modules_fc1_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_13_modules_fc2_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_fc2_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_fc2_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_13_modules_fc2_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_14_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_layers_modules_14_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_layers_modules_14_modules_fc1_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_fc1_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_fc1_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_14_modules_fc1_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_14_modules_fc2_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_fc2_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_fc2_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_14_modules_fc2_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_15_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_layers_modules_15_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_layers_modules_15_modules_fc1_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_fc1_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_fc1_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_15_modules_fc1_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_15_modules_fc2_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_fc2_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_fc2_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_15_modules_fc2_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_layers_modules_16_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_16_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_layers_modules_16_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_layers_modules_16_modules_fc1_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_fc1_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_fc1_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_16_modules_fc1_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_16_modules_fc2_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_fc2_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_fc2_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_16_modules_fc2_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_layers_modules_17_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_17_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_layers_modules_17_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_layers_modules_17_modules_fc1_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_fc1_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_fc1_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_17_modules_fc1_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_17_modules_fc2_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_fc2_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_fc2_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_17_modules_fc2_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_layers_modules_18_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_18_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_layers_modules_18_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_layers_modules_18_modules_fc1_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_fc1_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_fc1_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_18_modules_fc1_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_18_modules_fc2_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_fc2_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_fc2_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_18_modules_fc2_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_layers_modules_19_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_19_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_layers_modules_19_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_layers_modules_19_modules_fc1_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_fc1_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_fc1_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_19_modules_fc1_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_19_modules_fc2_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_fc2_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_fc2_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_19_modules_fc2_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_layers_modules_20_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_20_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_layers_modules_20_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_layers_modules_20_modules_fc1_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_fc1_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_fc1_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_20_modules_fc1_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_20_modules_fc2_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_fc2_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_fc2_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_20_modules_fc2_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_layers_modules_21_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_21_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_layers_modules_21_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_layers_modules_21_modules_fc1_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_fc1_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_fc1_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_21_modules_fc1_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_21_modules_fc2_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_fc2_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_fc2_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_21_modules_fc2_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_layers_modules_22_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_22_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_layers_modules_22_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_layers_modules_22_modules_fc1_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_fc1_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_fc1_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_22_modules_fc1_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_22_modules_fc2_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_fc2_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_fc2_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_22_modules_fc2_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_layers_modules_23_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_23_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_layers_modules_23_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_layers_modules_23_modules_fc1_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_fc1_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_fc1_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_23_modules_fc1_parameters_bias_
        )
        l_self_modules_model_modules_layers_modules_23_modules_fc2_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_fc2_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_fc2_parameters_bias_ = (
            L_self_modules_model_modules_layers_modules_23_modules_fc2_parameters_bias_
        )
        l_self_modules_model_modules_layer_norm_parameters_weight_ = (
            L_self_modules_model_modules_layer_norm_parameters_weight_
        )
        l_self_modules_model_modules_layer_norm_parameters_bias_ = (
            L_self_modules_model_modules_layer_norm_parameters_bias_
        )
        input_ids = l_input_ids_.view(-1, 19)
        l_input_ids_ = None
        embedding = torch.nn.functional.embedding(
            input_ids,
            l_self_modules_model_modules_embed_tokens_parameters_weight_,
            1,
            None,
            2.0,
            False,
            False,
        )
        input_ids = None
        inputs_embeds = embedding * 45.254833995939045
        embedding = None
        mask = torch.full((19, 19), -65504.0, device=device(type="cpu"))
        mask_cond = torch.arange(19, device=device(type="cpu"))
        add = mask_cond + 1
        view_1 = add.view(19, 1)
        add = None
        lt = mask_cond < view_1
        mask_cond = view_1 = None
        masked_fill_ = mask.masked_fill_(lt, 0)
        lt = masked_fill_ = None
        mask_1 = mask.to(torch.float16)
        mask = None
        getitem = mask_1[(None, None, slice(None, None, None), slice(None, None, None))]
        mask_1 = None
        causal_4d_mask = getitem.expand(1, 1, 19, 19)
        getitem = None
        getitem_1 = l_attention_mask_[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        l_attention_mask_ = None
        expand_1 = getitem_1.expand(1, 1, 19, 19)
        getitem_1 = None
        expanded_mask = expand_1.to(torch.float16)
        expand_1 = None
        tensor = torch.tensor(1.0, dtype=torch.float16)
        inverted_mask = tensor - expanded_mask
        tensor = expanded_mask = None
        to_2 = inverted_mask.to(torch.bool)
        masked_fill = inverted_mask.masked_fill(to_2, -65504.0)
        inverted_mask = to_2 = None
        expanded_attn_mask = masked_fill.to(device(type="cpu"))
        masked_fill = None
        bool_1 = expanded_attn_mask.bool()
        expanded_attn_mask = None
        expanded_attn_mask_1 = causal_4d_mask.masked_fill(bool_1, -65504.0)
        causal_4d_mask = bool_1 = None
        position_ids = torch.arange(0, 19, dtype=torch.int64, device=device(type="cpu"))
        position_ids_1 = position_ids.unsqueeze(0)
        position_ids = None
        position_ids_1 += 2
        position_ids_2 = position_ids_1
        position_ids_1 = None
        view_2 = position_ids_2.view(-1)
        position_ids_2 = None
        index_select = (
            l_self_modules_model_modules_embed_positions_buffers_weights_.index_select(
                0, view_2
            )
        )
        l_self_modules_model_modules_embed_positions_buffers_weights_ = view_2 = None
        view_3 = index_select.view(1, 19, 2048)
        index_select = None
        detach = view_3.detach()
        view_3 = None
        to_4 = detach.to(device(type="cpu"))
        detach = None
        hidden_states = inputs_embeds + to_4
        inputs_embeds = to_4 = None
        hidden_states_1 = torch.nn.functional.dropout(
            hidden_states, p=0.1, training=False
        )
        hidden_states = None
        hidden_states_2 = torch.nn.functional.layer_norm(
            hidden_states_1,
            (2048,),
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states = linear * 0.125
        linear = None
        key_states = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_2 = l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_4 = key_states.view(1, 19, -1, 64)
        key_states = None
        key_states_1 = view_4.transpose(1, 2)
        view_4 = None
        view_5 = value_states.view(1, 19, -1, 64)
        value_states = None
        value_states_1 = view_5.transpose(1, 2)
        view_5 = None
        view_6 = query_states.view(1, 19, 32, 64)
        query_states = None
        query_states_1 = view_6.transpose(1, 2)
        view_6 = None
        query_states_2 = query_states_1.reshape(32, -1, 64)
        query_states_1 = None
        key_states_2 = key_states_1.reshape(32, -1, 64)
        key_states_1 = None
        value_states_2 = value_states_1.reshape(32, -1, 64)
        value_states_1 = None
        transpose_3 = key_states_2.transpose(1, 2)
        key_states_2 = None
        attn_weights = torch.bmm(query_states_2, transpose_3)
        query_states_2 = transpose_3 = None
        view_7 = attn_weights.view(1, 32, 19, 19)
        attn_weights = None
        attn_weights_1 = view_7 + expanded_attn_mask_1
        view_7 = None
        tensor_1 = torch.tensor(-65504.0, device=device(type="cpu"))
        attn_weights_2 = torch.max(attn_weights_1, tensor_1)
        attn_weights_1 = tensor_1 = None
        attn_weights_3 = attn_weights_2.view(32, 19, 19)
        attn_weights_2 = None
        softmax = torch.nn.functional.softmax(
            attn_weights_3, dim=-1, dtype=torch.float32
        )
        attn_weights_3 = None
        attn_weights_4 = softmax.to(torch.float16)
        softmax = None
        attn_probs = torch.nn.functional.dropout(attn_weights_4, p=0.1, training=False)
        attn_weights_4 = None
        attn_output = torch.bmm(attn_probs, value_states_2)
        attn_probs = value_states_2 = None
        attn_output_1 = attn_output.view(1, 32, 19, 64)
        attn_output = None
        attn_output_2 = attn_output_1.transpose(1, 2)
        attn_output_1 = None
        attn_output_3 = attn_output_2.reshape(1, 19, 2048)
        attn_output_2 = None
        attn_output_4 = torch._C._nn.linear(
            attn_output_3,
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_3 = l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_3 = torch.nn.functional.dropout(
            attn_output_4, p=0.1, training=False
        )
        attn_output_4 = None
        hidden_states_4 = hidden_states_1 + hidden_states_3
        hidden_states_1 = hidden_states_3 = None
        hidden_states_5 = torch.nn.functional.layer_norm(
            hidden_states_4,
            (2048,),
            l_self_modules_model_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_ = (None)
        linear_4 = torch._C._nn.linear(
            hidden_states_5,
            l_self_modules_model_modules_layers_modules_0_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_layers_modules_0_modules_fc1_parameters_bias_,
        )
        hidden_states_5 = (
            l_self_modules_model_modules_layers_modules_0_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_model_modules_layers_modules_0_modules_fc1_parameters_bias_
        ) = None
        hidden_states_6 = torch._C._nn.gelu(linear_4)
        linear_4 = None
        hidden_states_7 = torch.nn.functional.dropout(
            hidden_states_6, p=0.0, training=False
        )
        hidden_states_6 = None
        hidden_states_8 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_model_modules_layers_modules_0_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_layers_modules_0_modules_fc2_parameters_bias_,
        )
        hidden_states_7 = (
            l_self_modules_model_modules_layers_modules_0_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_model_modules_layers_modules_0_modules_fc2_parameters_bias_
        ) = None
        hidden_states_9 = torch.nn.functional.dropout(
            hidden_states_8, p=0.1, training=False
        )
        hidden_states_8 = None
        hidden_states_10 = hidden_states_4 + hidden_states_9
        hidden_states_4 = hidden_states_9 = None
        hidden_states_11 = torch.nn.functional.layer_norm(
            hidden_states_10,
            (2048,),
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_6 = torch._C._nn.linear(
            hidden_states_11,
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_3 = linear_6 * 0.125
        linear_6 = None
        key_states_3 = torch._C._nn.linear(
            hidden_states_11,
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_3 = torch._C._nn.linear(
            hidden_states_11,
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_11 = l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_10 = key_states_3.view(1, 19, -1, 64)
        key_states_3 = None
        key_states_4 = view_10.transpose(1, 2)
        view_10 = None
        view_11 = value_states_3.view(1, 19, -1, 64)
        value_states_3 = None
        value_states_4 = view_11.transpose(1, 2)
        view_11 = None
        view_12 = query_states_3.view(1, 19, 32, 64)
        query_states_3 = None
        query_states_4 = view_12.transpose(1, 2)
        view_12 = None
        query_states_5 = query_states_4.reshape(32, -1, 64)
        query_states_4 = None
        key_states_5 = key_states_4.reshape(32, -1, 64)
        key_states_4 = None
        value_states_5 = value_states_4.reshape(32, -1, 64)
        value_states_4 = None
        transpose_8 = key_states_5.transpose(1, 2)
        key_states_5 = None
        attn_weights_5 = torch.bmm(query_states_5, transpose_8)
        query_states_5 = transpose_8 = None
        view_13 = attn_weights_5.view(1, 32, 19, 19)
        attn_weights_5 = None
        attn_weights_6 = view_13 + expanded_attn_mask_1
        view_13 = None
        tensor_2 = torch.tensor(-65504.0, device=device(type="cpu"))
        attn_weights_7 = torch.max(attn_weights_6, tensor_2)
        attn_weights_6 = tensor_2 = None
        attn_weights_8 = attn_weights_7.view(32, 19, 19)
        attn_weights_7 = None
        softmax_1 = torch.nn.functional.softmax(
            attn_weights_8, dim=-1, dtype=torch.float32
        )
        attn_weights_8 = None
        attn_weights_9 = softmax_1.to(torch.float16)
        softmax_1 = None
        attn_probs_1 = torch.nn.functional.dropout(
            attn_weights_9, p=0.1, training=False
        )
        attn_weights_9 = None
        attn_output_5 = torch.bmm(attn_probs_1, value_states_5)
        attn_probs_1 = value_states_5 = None
        attn_output_6 = attn_output_5.view(1, 32, 19, 64)
        attn_output_5 = None
        attn_output_7 = attn_output_6.transpose(1, 2)
        attn_output_6 = None
        attn_output_8 = attn_output_7.reshape(1, 19, 2048)
        attn_output_7 = None
        attn_output_9 = torch._C._nn.linear(
            attn_output_8,
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_8 = l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_12 = torch.nn.functional.dropout(
            attn_output_9, p=0.1, training=False
        )
        attn_output_9 = None
        hidden_states_13 = hidden_states_10 + hidden_states_12
        hidden_states_10 = hidden_states_12 = None
        hidden_states_14 = torch.nn.functional.layer_norm(
            hidden_states_13,
            (2048,),
            l_self_modules_model_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_ = (None)
        linear_10 = torch._C._nn.linear(
            hidden_states_14,
            l_self_modules_model_modules_layers_modules_1_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_layers_modules_1_modules_fc1_parameters_bias_,
        )
        hidden_states_14 = (
            l_self_modules_model_modules_layers_modules_1_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_model_modules_layers_modules_1_modules_fc1_parameters_bias_
        ) = None
        hidden_states_15 = torch._C._nn.gelu(linear_10)
        linear_10 = None
        hidden_states_16 = torch.nn.functional.dropout(
            hidden_states_15, p=0.0, training=False
        )
        hidden_states_15 = None
        hidden_states_17 = torch._C._nn.linear(
            hidden_states_16,
            l_self_modules_model_modules_layers_modules_1_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_layers_modules_1_modules_fc2_parameters_bias_,
        )
        hidden_states_16 = (
            l_self_modules_model_modules_layers_modules_1_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_model_modules_layers_modules_1_modules_fc2_parameters_bias_
        ) = None
        hidden_states_18 = torch.nn.functional.dropout(
            hidden_states_17, p=0.1, training=False
        )
        hidden_states_17 = None
        hidden_states_19 = hidden_states_13 + hidden_states_18
        hidden_states_13 = hidden_states_18 = None
        hidden_states_20 = torch.nn.functional.layer_norm(
            hidden_states_19,
            (2048,),
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_12 = torch._C._nn.linear(
            hidden_states_20,
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_6 = linear_12 * 0.125
        linear_12 = None
        key_states_6 = torch._C._nn.linear(
            hidden_states_20,
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_6 = torch._C._nn.linear(
            hidden_states_20,
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_20 = l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_16 = key_states_6.view(1, 19, -1, 64)
        key_states_6 = None
        key_states_7 = view_16.transpose(1, 2)
        view_16 = None
        view_17 = value_states_6.view(1, 19, -1, 64)
        value_states_6 = None
        value_states_7 = view_17.transpose(1, 2)
        view_17 = None
        view_18 = query_states_6.view(1, 19, 32, 64)
        query_states_6 = None
        query_states_7 = view_18.transpose(1, 2)
        view_18 = None
        query_states_8 = query_states_7.reshape(32, -1, 64)
        query_states_7 = None
        key_states_8 = key_states_7.reshape(32, -1, 64)
        key_states_7 = None
        value_states_8 = value_states_7.reshape(32, -1, 64)
        value_states_7 = None
        transpose_13 = key_states_8.transpose(1, 2)
        key_states_8 = None
        attn_weights_10 = torch.bmm(query_states_8, transpose_13)
        query_states_8 = transpose_13 = None
        view_19 = attn_weights_10.view(1, 32, 19, 19)
        attn_weights_10 = None
        attn_weights_11 = view_19 + expanded_attn_mask_1
        view_19 = None
        tensor_3 = torch.tensor(-65504.0, device=device(type="cpu"))
        attn_weights_12 = torch.max(attn_weights_11, tensor_3)
        attn_weights_11 = tensor_3 = None
        attn_weights_13 = attn_weights_12.view(32, 19, 19)
        attn_weights_12 = None
        softmax_2 = torch.nn.functional.softmax(
            attn_weights_13, dim=-1, dtype=torch.float32
        )
        attn_weights_13 = None
        attn_weights_14 = softmax_2.to(torch.float16)
        softmax_2 = None
        attn_probs_2 = torch.nn.functional.dropout(
            attn_weights_14, p=0.1, training=False
        )
        attn_weights_14 = None
        attn_output_10 = torch.bmm(attn_probs_2, value_states_8)
        attn_probs_2 = value_states_8 = None
        attn_output_11 = attn_output_10.view(1, 32, 19, 64)
        attn_output_10 = None
        attn_output_12 = attn_output_11.transpose(1, 2)
        attn_output_11 = None
        attn_output_13 = attn_output_12.reshape(1, 19, 2048)
        attn_output_12 = None
        attn_output_14 = torch._C._nn.linear(
            attn_output_13,
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_13 = l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_21 = torch.nn.functional.dropout(
            attn_output_14, p=0.1, training=False
        )
        attn_output_14 = None
        hidden_states_22 = hidden_states_19 + hidden_states_21
        hidden_states_19 = hidden_states_21 = None
        hidden_states_23 = torch.nn.functional.layer_norm(
            hidden_states_22,
            (2048,),
            l_self_modules_model_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_ = (None)
        linear_16 = torch._C._nn.linear(
            hidden_states_23,
            l_self_modules_model_modules_layers_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_layers_modules_2_modules_fc1_parameters_bias_,
        )
        hidden_states_23 = (
            l_self_modules_model_modules_layers_modules_2_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_model_modules_layers_modules_2_modules_fc1_parameters_bias_
        ) = None
        hidden_states_24 = torch._C._nn.gelu(linear_16)
        linear_16 = None
        hidden_states_25 = torch.nn.functional.dropout(
            hidden_states_24, p=0.0, training=False
        )
        hidden_states_24 = None
        hidden_states_26 = torch._C._nn.linear(
            hidden_states_25,
            l_self_modules_model_modules_layers_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_layers_modules_2_modules_fc2_parameters_bias_,
        )
        hidden_states_25 = (
            l_self_modules_model_modules_layers_modules_2_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_model_modules_layers_modules_2_modules_fc2_parameters_bias_
        ) = None
        hidden_states_27 = torch.nn.functional.dropout(
            hidden_states_26, p=0.1, training=False
        )
        hidden_states_26 = None
        hidden_states_28 = hidden_states_22 + hidden_states_27
        hidden_states_22 = hidden_states_27 = None
        hidden_states_29 = torch.nn.functional.layer_norm(
            hidden_states_28,
            (2048,),
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_18 = torch._C._nn.linear(
            hidden_states_29,
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_9 = linear_18 * 0.125
        linear_18 = None
        key_states_9 = torch._C._nn.linear(
            hidden_states_29,
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_9 = torch._C._nn.linear(
            hidden_states_29,
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_29 = l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_22 = key_states_9.view(1, 19, -1, 64)
        key_states_9 = None
        key_states_10 = view_22.transpose(1, 2)
        view_22 = None
        view_23 = value_states_9.view(1, 19, -1, 64)
        value_states_9 = None
        value_states_10 = view_23.transpose(1, 2)
        view_23 = None
        view_24 = query_states_9.view(1, 19, 32, 64)
        query_states_9 = None
        query_states_10 = view_24.transpose(1, 2)
        view_24 = None
        query_states_11 = query_states_10.reshape(32, -1, 64)
        query_states_10 = None
        key_states_11 = key_states_10.reshape(32, -1, 64)
        key_states_10 = None
        value_states_11 = value_states_10.reshape(32, -1, 64)
        value_states_10 = None
        transpose_18 = key_states_11.transpose(1, 2)
        key_states_11 = None
        attn_weights_15 = torch.bmm(query_states_11, transpose_18)
        query_states_11 = transpose_18 = None
        view_25 = attn_weights_15.view(1, 32, 19, 19)
        attn_weights_15 = None
        attn_weights_16 = view_25 + expanded_attn_mask_1
        view_25 = None
        tensor_4 = torch.tensor(-65504.0, device=device(type="cpu"))
        attn_weights_17 = torch.max(attn_weights_16, tensor_4)
        attn_weights_16 = tensor_4 = None
        attn_weights_18 = attn_weights_17.view(32, 19, 19)
        attn_weights_17 = None
        softmax_3 = torch.nn.functional.softmax(
            attn_weights_18, dim=-1, dtype=torch.float32
        )
        attn_weights_18 = None
        attn_weights_19 = softmax_3.to(torch.float16)
        softmax_3 = None
        attn_probs_3 = torch.nn.functional.dropout(
            attn_weights_19, p=0.1, training=False
        )
        attn_weights_19 = None
        attn_output_15 = torch.bmm(attn_probs_3, value_states_11)
        attn_probs_3 = value_states_11 = None
        attn_output_16 = attn_output_15.view(1, 32, 19, 64)
        attn_output_15 = None
        attn_output_17 = attn_output_16.transpose(1, 2)
        attn_output_16 = None
        attn_output_18 = attn_output_17.reshape(1, 19, 2048)
        attn_output_17 = None
        attn_output_19 = torch._C._nn.linear(
            attn_output_18,
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_18 = l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_30 = torch.nn.functional.dropout(
            attn_output_19, p=0.1, training=False
        )
        attn_output_19 = None
        hidden_states_31 = hidden_states_28 + hidden_states_30
        hidden_states_28 = hidden_states_30 = None
        hidden_states_32 = torch.nn.functional.layer_norm(
            hidden_states_31,
            (2048,),
            l_self_modules_model_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_ = (None)
        linear_22 = torch._C._nn.linear(
            hidden_states_32,
            l_self_modules_model_modules_layers_modules_3_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_layers_modules_3_modules_fc1_parameters_bias_,
        )
        hidden_states_32 = (
            l_self_modules_model_modules_layers_modules_3_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_model_modules_layers_modules_3_modules_fc1_parameters_bias_
        ) = None
        hidden_states_33 = torch._C._nn.gelu(linear_22)
        linear_22 = None
        hidden_states_34 = torch.nn.functional.dropout(
            hidden_states_33, p=0.0, training=False
        )
        hidden_states_33 = None
        hidden_states_35 = torch._C._nn.linear(
            hidden_states_34,
            l_self_modules_model_modules_layers_modules_3_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_layers_modules_3_modules_fc2_parameters_bias_,
        )
        hidden_states_34 = (
            l_self_modules_model_modules_layers_modules_3_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_model_modules_layers_modules_3_modules_fc2_parameters_bias_
        ) = None
        hidden_states_36 = torch.nn.functional.dropout(
            hidden_states_35, p=0.1, training=False
        )
        hidden_states_35 = None
        hidden_states_37 = hidden_states_31 + hidden_states_36
        hidden_states_31 = hidden_states_36 = None
        hidden_states_38 = torch.nn.functional.layer_norm(
            hidden_states_37,
            (2048,),
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_24 = torch._C._nn.linear(
            hidden_states_38,
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_12 = linear_24 * 0.125
        linear_24 = None
        key_states_12 = torch._C._nn.linear(
            hidden_states_38,
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_12 = torch._C._nn.linear(
            hidden_states_38,
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_38 = l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_28 = key_states_12.view(1, 19, -1, 64)
        key_states_12 = None
        key_states_13 = view_28.transpose(1, 2)
        view_28 = None
        view_29 = value_states_12.view(1, 19, -1, 64)
        value_states_12 = None
        value_states_13 = view_29.transpose(1, 2)
        view_29 = None
        view_30 = query_states_12.view(1, 19, 32, 64)
        query_states_12 = None
        query_states_13 = view_30.transpose(1, 2)
        view_30 = None
        query_states_14 = query_states_13.reshape(32, -1, 64)
        query_states_13 = None
        key_states_14 = key_states_13.reshape(32, -1, 64)
        key_states_13 = None
        value_states_14 = value_states_13.reshape(32, -1, 64)
        value_states_13 = None
        transpose_23 = key_states_14.transpose(1, 2)
        key_states_14 = None
        attn_weights_20 = torch.bmm(query_states_14, transpose_23)
        query_states_14 = transpose_23 = None
        view_31 = attn_weights_20.view(1, 32, 19, 19)
        attn_weights_20 = None
        attn_weights_21 = view_31 + expanded_attn_mask_1
        view_31 = None
        tensor_5 = torch.tensor(-65504.0, device=device(type="cpu"))
        attn_weights_22 = torch.max(attn_weights_21, tensor_5)
        attn_weights_21 = tensor_5 = None
        attn_weights_23 = attn_weights_22.view(32, 19, 19)
        attn_weights_22 = None
        softmax_4 = torch.nn.functional.softmax(
            attn_weights_23, dim=-1, dtype=torch.float32
        )
        attn_weights_23 = None
        attn_weights_24 = softmax_4.to(torch.float16)
        softmax_4 = None
        attn_probs_4 = torch.nn.functional.dropout(
            attn_weights_24, p=0.1, training=False
        )
        attn_weights_24 = None
        attn_output_20 = torch.bmm(attn_probs_4, value_states_14)
        attn_probs_4 = value_states_14 = None
        attn_output_21 = attn_output_20.view(1, 32, 19, 64)
        attn_output_20 = None
        attn_output_22 = attn_output_21.transpose(1, 2)
        attn_output_21 = None
        attn_output_23 = attn_output_22.reshape(1, 19, 2048)
        attn_output_22 = None
        attn_output_24 = torch._C._nn.linear(
            attn_output_23,
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_23 = l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_39 = torch.nn.functional.dropout(
            attn_output_24, p=0.1, training=False
        )
        attn_output_24 = None
        hidden_states_40 = hidden_states_37 + hidden_states_39
        hidden_states_37 = hidden_states_39 = None
        hidden_states_41 = torch.nn.functional.layer_norm(
            hidden_states_40,
            (2048,),
            l_self_modules_model_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_ = (None)
        linear_28 = torch._C._nn.linear(
            hidden_states_41,
            l_self_modules_model_modules_layers_modules_4_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_layers_modules_4_modules_fc1_parameters_bias_,
        )
        hidden_states_41 = (
            l_self_modules_model_modules_layers_modules_4_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_model_modules_layers_modules_4_modules_fc1_parameters_bias_
        ) = None
        hidden_states_42 = torch._C._nn.gelu(linear_28)
        linear_28 = None
        hidden_states_43 = torch.nn.functional.dropout(
            hidden_states_42, p=0.0, training=False
        )
        hidden_states_42 = None
        hidden_states_44 = torch._C._nn.linear(
            hidden_states_43,
            l_self_modules_model_modules_layers_modules_4_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_layers_modules_4_modules_fc2_parameters_bias_,
        )
        hidden_states_43 = (
            l_self_modules_model_modules_layers_modules_4_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_model_modules_layers_modules_4_modules_fc2_parameters_bias_
        ) = None
        hidden_states_45 = torch.nn.functional.dropout(
            hidden_states_44, p=0.1, training=False
        )
        hidden_states_44 = None
        hidden_states_46 = hidden_states_40 + hidden_states_45
        hidden_states_40 = hidden_states_45 = None
        hidden_states_47 = torch.nn.functional.layer_norm(
            hidden_states_46,
            (2048,),
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_30 = torch._C._nn.linear(
            hidden_states_47,
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_15 = linear_30 * 0.125
        linear_30 = None
        key_states_15 = torch._C._nn.linear(
            hidden_states_47,
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_15 = torch._C._nn.linear(
            hidden_states_47,
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_47 = l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_34 = key_states_15.view(1, 19, -1, 64)
        key_states_15 = None
        key_states_16 = view_34.transpose(1, 2)
        view_34 = None
        view_35 = value_states_15.view(1, 19, -1, 64)
        value_states_15 = None
        value_states_16 = view_35.transpose(1, 2)
        view_35 = None
        view_36 = query_states_15.view(1, 19, 32, 64)
        query_states_15 = None
        query_states_16 = view_36.transpose(1, 2)
        view_36 = None
        query_states_17 = query_states_16.reshape(32, -1, 64)
        query_states_16 = None
        key_states_17 = key_states_16.reshape(32, -1, 64)
        key_states_16 = None
        value_states_17 = value_states_16.reshape(32, -1, 64)
        value_states_16 = None
        transpose_28 = key_states_17.transpose(1, 2)
        key_states_17 = None
        attn_weights_25 = torch.bmm(query_states_17, transpose_28)
        query_states_17 = transpose_28 = None
        view_37 = attn_weights_25.view(1, 32, 19, 19)
        attn_weights_25 = None
        attn_weights_26 = view_37 + expanded_attn_mask_1
        view_37 = None
        tensor_6 = torch.tensor(-65504.0, device=device(type="cpu"))
        attn_weights_27 = torch.max(attn_weights_26, tensor_6)
        attn_weights_26 = tensor_6 = None
        attn_weights_28 = attn_weights_27.view(32, 19, 19)
        attn_weights_27 = None
        softmax_5 = torch.nn.functional.softmax(
            attn_weights_28, dim=-1, dtype=torch.float32
        )
        attn_weights_28 = None
        attn_weights_29 = softmax_5.to(torch.float16)
        softmax_5 = None
        attn_probs_5 = torch.nn.functional.dropout(
            attn_weights_29, p=0.1, training=False
        )
        attn_weights_29 = None
        attn_output_25 = torch.bmm(attn_probs_5, value_states_17)
        attn_probs_5 = value_states_17 = None
        attn_output_26 = attn_output_25.view(1, 32, 19, 64)
        attn_output_25 = None
        attn_output_27 = attn_output_26.transpose(1, 2)
        attn_output_26 = None
        attn_output_28 = attn_output_27.reshape(1, 19, 2048)
        attn_output_27 = None
        attn_output_29 = torch._C._nn.linear(
            attn_output_28,
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_28 = l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_48 = torch.nn.functional.dropout(
            attn_output_29, p=0.1, training=False
        )
        attn_output_29 = None
        hidden_states_49 = hidden_states_46 + hidden_states_48
        hidden_states_46 = hidden_states_48 = None
        hidden_states_50 = torch.nn.functional.layer_norm(
            hidden_states_49,
            (2048,),
            l_self_modules_model_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_ = (None)
        linear_34 = torch._C._nn.linear(
            hidden_states_50,
            l_self_modules_model_modules_layers_modules_5_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_layers_modules_5_modules_fc1_parameters_bias_,
        )
        hidden_states_50 = (
            l_self_modules_model_modules_layers_modules_5_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_model_modules_layers_modules_5_modules_fc1_parameters_bias_
        ) = None
        hidden_states_51 = torch._C._nn.gelu(linear_34)
        linear_34 = None
        hidden_states_52 = torch.nn.functional.dropout(
            hidden_states_51, p=0.0, training=False
        )
        hidden_states_51 = None
        hidden_states_53 = torch._C._nn.linear(
            hidden_states_52,
            l_self_modules_model_modules_layers_modules_5_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_layers_modules_5_modules_fc2_parameters_bias_,
        )
        hidden_states_52 = (
            l_self_modules_model_modules_layers_modules_5_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_model_modules_layers_modules_5_modules_fc2_parameters_bias_
        ) = None
        hidden_states_54 = torch.nn.functional.dropout(
            hidden_states_53, p=0.1, training=False
        )
        hidden_states_53 = None
        hidden_states_55 = hidden_states_49 + hidden_states_54
        hidden_states_49 = hidden_states_54 = None
        hidden_states_56 = torch.nn.functional.layer_norm(
            hidden_states_55,
            (2048,),
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_36 = torch._C._nn.linear(
            hidden_states_56,
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_18 = linear_36 * 0.125
        linear_36 = None
        key_states_18 = torch._C._nn.linear(
            hidden_states_56,
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_18 = torch._C._nn.linear(
            hidden_states_56,
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_56 = l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_40 = key_states_18.view(1, 19, -1, 64)
        key_states_18 = None
        key_states_19 = view_40.transpose(1, 2)
        view_40 = None
        view_41 = value_states_18.view(1, 19, -1, 64)
        value_states_18 = None
        value_states_19 = view_41.transpose(1, 2)
        view_41 = None
        view_42 = query_states_18.view(1, 19, 32, 64)
        query_states_18 = None
        query_states_19 = view_42.transpose(1, 2)
        view_42 = None
        query_states_20 = query_states_19.reshape(32, -1, 64)
        query_states_19 = None
        key_states_20 = key_states_19.reshape(32, -1, 64)
        key_states_19 = None
        value_states_20 = value_states_19.reshape(32, -1, 64)
        value_states_19 = None
        transpose_33 = key_states_20.transpose(1, 2)
        key_states_20 = None
        attn_weights_30 = torch.bmm(query_states_20, transpose_33)
        query_states_20 = transpose_33 = None
        view_43 = attn_weights_30.view(1, 32, 19, 19)
        attn_weights_30 = None
        attn_weights_31 = view_43 + expanded_attn_mask_1
        view_43 = None
        tensor_7 = torch.tensor(-65504.0, device=device(type="cpu"))
        attn_weights_32 = torch.max(attn_weights_31, tensor_7)
        attn_weights_31 = tensor_7 = None
        attn_weights_33 = attn_weights_32.view(32, 19, 19)
        attn_weights_32 = None
        softmax_6 = torch.nn.functional.softmax(
            attn_weights_33, dim=-1, dtype=torch.float32
        )
        attn_weights_33 = None
        attn_weights_34 = softmax_6.to(torch.float16)
        softmax_6 = None
        attn_probs_6 = torch.nn.functional.dropout(
            attn_weights_34, p=0.1, training=False
        )
        attn_weights_34 = None
        attn_output_30 = torch.bmm(attn_probs_6, value_states_20)
        attn_probs_6 = value_states_20 = None
        attn_output_31 = attn_output_30.view(1, 32, 19, 64)
        attn_output_30 = None
        attn_output_32 = attn_output_31.transpose(1, 2)
        attn_output_31 = None
        attn_output_33 = attn_output_32.reshape(1, 19, 2048)
        attn_output_32 = None
        attn_output_34 = torch._C._nn.linear(
            attn_output_33,
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_33 = l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_57 = torch.nn.functional.dropout(
            attn_output_34, p=0.1, training=False
        )
        attn_output_34 = None
        hidden_states_58 = hidden_states_55 + hidden_states_57
        hidden_states_55 = hidden_states_57 = None
        hidden_states_59 = torch.nn.functional.layer_norm(
            hidden_states_58,
            (2048,),
            l_self_modules_model_modules_layers_modules_6_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_6_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_6_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_6_modules_final_layer_norm_parameters_bias_ = (None)
        linear_40 = torch._C._nn.linear(
            hidden_states_59,
            l_self_modules_model_modules_layers_modules_6_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_layers_modules_6_modules_fc1_parameters_bias_,
        )
        hidden_states_59 = (
            l_self_modules_model_modules_layers_modules_6_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_model_modules_layers_modules_6_modules_fc1_parameters_bias_
        ) = None
        hidden_states_60 = torch._C._nn.gelu(linear_40)
        linear_40 = None
        hidden_states_61 = torch.nn.functional.dropout(
            hidden_states_60, p=0.0, training=False
        )
        hidden_states_60 = None
        hidden_states_62 = torch._C._nn.linear(
            hidden_states_61,
            l_self_modules_model_modules_layers_modules_6_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_layers_modules_6_modules_fc2_parameters_bias_,
        )
        hidden_states_61 = (
            l_self_modules_model_modules_layers_modules_6_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_model_modules_layers_modules_6_modules_fc2_parameters_bias_
        ) = None
        hidden_states_63 = torch.nn.functional.dropout(
            hidden_states_62, p=0.1, training=False
        )
        hidden_states_62 = None
        hidden_states_64 = hidden_states_58 + hidden_states_63
        hidden_states_58 = hidden_states_63 = None
        hidden_states_65 = torch.nn.functional.layer_norm(
            hidden_states_64,
            (2048,),
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_42 = torch._C._nn.linear(
            hidden_states_65,
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_21 = linear_42 * 0.125
        linear_42 = None
        key_states_21 = torch._C._nn.linear(
            hidden_states_65,
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_21 = torch._C._nn.linear(
            hidden_states_65,
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_65 = l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_46 = key_states_21.view(1, 19, -1, 64)
        key_states_21 = None
        key_states_22 = view_46.transpose(1, 2)
        view_46 = None
        view_47 = value_states_21.view(1, 19, -1, 64)
        value_states_21 = None
        value_states_22 = view_47.transpose(1, 2)
        view_47 = None
        view_48 = query_states_21.view(1, 19, 32, 64)
        query_states_21 = None
        query_states_22 = view_48.transpose(1, 2)
        view_48 = None
        query_states_23 = query_states_22.reshape(32, -1, 64)
        query_states_22 = None
        key_states_23 = key_states_22.reshape(32, -1, 64)
        key_states_22 = None
        value_states_23 = value_states_22.reshape(32, -1, 64)
        value_states_22 = None
        transpose_38 = key_states_23.transpose(1, 2)
        key_states_23 = None
        attn_weights_35 = torch.bmm(query_states_23, transpose_38)
        query_states_23 = transpose_38 = None
        view_49 = attn_weights_35.view(1, 32, 19, 19)
        attn_weights_35 = None
        attn_weights_36 = view_49 + expanded_attn_mask_1
        view_49 = None
        tensor_8 = torch.tensor(-65504.0, device=device(type="cpu"))
        attn_weights_37 = torch.max(attn_weights_36, tensor_8)
        attn_weights_36 = tensor_8 = None
        attn_weights_38 = attn_weights_37.view(32, 19, 19)
        attn_weights_37 = None
        softmax_7 = torch.nn.functional.softmax(
            attn_weights_38, dim=-1, dtype=torch.float32
        )
        attn_weights_38 = None
        attn_weights_39 = softmax_7.to(torch.float16)
        softmax_7 = None
        attn_probs_7 = torch.nn.functional.dropout(
            attn_weights_39, p=0.1, training=False
        )
        attn_weights_39 = None
        attn_output_35 = torch.bmm(attn_probs_7, value_states_23)
        attn_probs_7 = value_states_23 = None
        attn_output_36 = attn_output_35.view(1, 32, 19, 64)
        attn_output_35 = None
        attn_output_37 = attn_output_36.transpose(1, 2)
        attn_output_36 = None
        attn_output_38 = attn_output_37.reshape(1, 19, 2048)
        attn_output_37 = None
        attn_output_39 = torch._C._nn.linear(
            attn_output_38,
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_38 = l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_66 = torch.nn.functional.dropout(
            attn_output_39, p=0.1, training=False
        )
        attn_output_39 = None
        hidden_states_67 = hidden_states_64 + hidden_states_66
        hidden_states_64 = hidden_states_66 = None
        hidden_states_68 = torch.nn.functional.layer_norm(
            hidden_states_67,
            (2048,),
            l_self_modules_model_modules_layers_modules_7_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_7_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_7_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_7_modules_final_layer_norm_parameters_bias_ = (None)
        linear_46 = torch._C._nn.linear(
            hidden_states_68,
            l_self_modules_model_modules_layers_modules_7_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_layers_modules_7_modules_fc1_parameters_bias_,
        )
        hidden_states_68 = (
            l_self_modules_model_modules_layers_modules_7_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_model_modules_layers_modules_7_modules_fc1_parameters_bias_
        ) = None
        hidden_states_69 = torch._C._nn.gelu(linear_46)
        linear_46 = None
        hidden_states_70 = torch.nn.functional.dropout(
            hidden_states_69, p=0.0, training=False
        )
        hidden_states_69 = None
        hidden_states_71 = torch._C._nn.linear(
            hidden_states_70,
            l_self_modules_model_modules_layers_modules_7_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_layers_modules_7_modules_fc2_parameters_bias_,
        )
        hidden_states_70 = (
            l_self_modules_model_modules_layers_modules_7_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_model_modules_layers_modules_7_modules_fc2_parameters_bias_
        ) = None
        hidden_states_72 = torch.nn.functional.dropout(
            hidden_states_71, p=0.1, training=False
        )
        hidden_states_71 = None
        hidden_states_73 = hidden_states_67 + hidden_states_72
        hidden_states_67 = hidden_states_72 = None
        hidden_states_74 = torch.nn.functional.layer_norm(
            hidden_states_73,
            (2048,),
            l_self_modules_model_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_48 = torch._C._nn.linear(
            hidden_states_74,
            l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_24 = linear_48 * 0.125
        linear_48 = None
        key_states_24 = torch._C._nn.linear(
            hidden_states_74,
            l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_24 = torch._C._nn.linear(
            hidden_states_74,
            l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_74 = l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_52 = key_states_24.view(1, 19, -1, 64)
        key_states_24 = None
        key_states_25 = view_52.transpose(1, 2)
        view_52 = None
        view_53 = value_states_24.view(1, 19, -1, 64)
        value_states_24 = None
        value_states_25 = view_53.transpose(1, 2)
        view_53 = None
        view_54 = query_states_24.view(1, 19, 32, 64)
        query_states_24 = None
        query_states_25 = view_54.transpose(1, 2)
        view_54 = None
        query_states_26 = query_states_25.reshape(32, -1, 64)
        query_states_25 = None
        key_states_26 = key_states_25.reshape(32, -1, 64)
        key_states_25 = None
        value_states_26 = value_states_25.reshape(32, -1, 64)
        value_states_25 = None
        transpose_43 = key_states_26.transpose(1, 2)
        key_states_26 = None
        attn_weights_40 = torch.bmm(query_states_26, transpose_43)
        query_states_26 = transpose_43 = None
        view_55 = attn_weights_40.view(1, 32, 19, 19)
        attn_weights_40 = None
        attn_weights_41 = view_55 + expanded_attn_mask_1
        view_55 = None
        tensor_9 = torch.tensor(-65504.0, device=device(type="cpu"))
        attn_weights_42 = torch.max(attn_weights_41, tensor_9)
        attn_weights_41 = tensor_9 = None
        attn_weights_43 = attn_weights_42.view(32, 19, 19)
        attn_weights_42 = None
        softmax_8 = torch.nn.functional.softmax(
            attn_weights_43, dim=-1, dtype=torch.float32
        )
        attn_weights_43 = None
        attn_weights_44 = softmax_8.to(torch.float16)
        softmax_8 = None
        attn_probs_8 = torch.nn.functional.dropout(
            attn_weights_44, p=0.1, training=False
        )
        attn_weights_44 = None
        attn_output_40 = torch.bmm(attn_probs_8, value_states_26)
        attn_probs_8 = value_states_26 = None
        attn_output_41 = attn_output_40.view(1, 32, 19, 64)
        attn_output_40 = None
        attn_output_42 = attn_output_41.transpose(1, 2)
        attn_output_41 = None
        attn_output_43 = attn_output_42.reshape(1, 19, 2048)
        attn_output_42 = None
        attn_output_44 = torch._C._nn.linear(
            attn_output_43,
            l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_43 = l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_75 = torch.nn.functional.dropout(
            attn_output_44, p=0.1, training=False
        )
        attn_output_44 = None
        hidden_states_76 = hidden_states_73 + hidden_states_75
        hidden_states_73 = hidden_states_75 = None
        hidden_states_77 = torch.nn.functional.layer_norm(
            hidden_states_76,
            (2048,),
            l_self_modules_model_modules_layers_modules_8_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_8_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_8_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_8_modules_final_layer_norm_parameters_bias_ = (None)
        linear_52 = torch._C._nn.linear(
            hidden_states_77,
            l_self_modules_model_modules_layers_modules_8_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_layers_modules_8_modules_fc1_parameters_bias_,
        )
        hidden_states_77 = (
            l_self_modules_model_modules_layers_modules_8_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_model_modules_layers_modules_8_modules_fc1_parameters_bias_
        ) = None
        hidden_states_78 = torch._C._nn.gelu(linear_52)
        linear_52 = None
        hidden_states_79 = torch.nn.functional.dropout(
            hidden_states_78, p=0.0, training=False
        )
        hidden_states_78 = None
        hidden_states_80 = torch._C._nn.linear(
            hidden_states_79,
            l_self_modules_model_modules_layers_modules_8_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_layers_modules_8_modules_fc2_parameters_bias_,
        )
        hidden_states_79 = (
            l_self_modules_model_modules_layers_modules_8_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_model_modules_layers_modules_8_modules_fc2_parameters_bias_
        ) = None
        hidden_states_81 = torch.nn.functional.dropout(
            hidden_states_80, p=0.1, training=False
        )
        hidden_states_80 = None
        hidden_states_82 = hidden_states_76 + hidden_states_81
        hidden_states_76 = hidden_states_81 = None
        hidden_states_83 = torch.nn.functional.layer_norm(
            hidden_states_82,
            (2048,),
            l_self_modules_model_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_54 = torch._C._nn.linear(
            hidden_states_83,
            l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_27 = linear_54 * 0.125
        linear_54 = None
        key_states_27 = torch._C._nn.linear(
            hidden_states_83,
            l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_27 = torch._C._nn.linear(
            hidden_states_83,
            l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_83 = l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_58 = key_states_27.view(1, 19, -1, 64)
        key_states_27 = None
        key_states_28 = view_58.transpose(1, 2)
        view_58 = None
        view_59 = value_states_27.view(1, 19, -1, 64)
        value_states_27 = None
        value_states_28 = view_59.transpose(1, 2)
        view_59 = None
        view_60 = query_states_27.view(1, 19, 32, 64)
        query_states_27 = None
        query_states_28 = view_60.transpose(1, 2)
        view_60 = None
        query_states_29 = query_states_28.reshape(32, -1, 64)
        query_states_28 = None
        key_states_29 = key_states_28.reshape(32, -1, 64)
        key_states_28 = None
        value_states_29 = value_states_28.reshape(32, -1, 64)
        value_states_28 = None
        transpose_48 = key_states_29.transpose(1, 2)
        key_states_29 = None
        attn_weights_45 = torch.bmm(query_states_29, transpose_48)
        query_states_29 = transpose_48 = None
        view_61 = attn_weights_45.view(1, 32, 19, 19)
        attn_weights_45 = None
        attn_weights_46 = view_61 + expanded_attn_mask_1
        view_61 = None
        tensor_10 = torch.tensor(-65504.0, device=device(type="cpu"))
        attn_weights_47 = torch.max(attn_weights_46, tensor_10)
        attn_weights_46 = tensor_10 = None
        attn_weights_48 = attn_weights_47.view(32, 19, 19)
        attn_weights_47 = None
        softmax_9 = torch.nn.functional.softmax(
            attn_weights_48, dim=-1, dtype=torch.float32
        )
        attn_weights_48 = None
        attn_weights_49 = softmax_9.to(torch.float16)
        softmax_9 = None
        attn_probs_9 = torch.nn.functional.dropout(
            attn_weights_49, p=0.1, training=False
        )
        attn_weights_49 = None
        attn_output_45 = torch.bmm(attn_probs_9, value_states_29)
        attn_probs_9 = value_states_29 = None
        attn_output_46 = attn_output_45.view(1, 32, 19, 64)
        attn_output_45 = None
        attn_output_47 = attn_output_46.transpose(1, 2)
        attn_output_46 = None
        attn_output_48 = attn_output_47.reshape(1, 19, 2048)
        attn_output_47 = None
        attn_output_49 = torch._C._nn.linear(
            attn_output_48,
            l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_48 = l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_84 = torch.nn.functional.dropout(
            attn_output_49, p=0.1, training=False
        )
        attn_output_49 = None
        hidden_states_85 = hidden_states_82 + hidden_states_84
        hidden_states_82 = hidden_states_84 = None
        hidden_states_86 = torch.nn.functional.layer_norm(
            hidden_states_85,
            (2048,),
            l_self_modules_model_modules_layers_modules_9_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_9_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_9_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_9_modules_final_layer_norm_parameters_bias_ = (None)
        linear_58 = torch._C._nn.linear(
            hidden_states_86,
            l_self_modules_model_modules_layers_modules_9_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_layers_modules_9_modules_fc1_parameters_bias_,
        )
        hidden_states_86 = (
            l_self_modules_model_modules_layers_modules_9_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_model_modules_layers_modules_9_modules_fc1_parameters_bias_
        ) = None
        hidden_states_87 = torch._C._nn.gelu(linear_58)
        linear_58 = None
        hidden_states_88 = torch.nn.functional.dropout(
            hidden_states_87, p=0.0, training=False
        )
        hidden_states_87 = None
        hidden_states_89 = torch._C._nn.linear(
            hidden_states_88,
            l_self_modules_model_modules_layers_modules_9_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_layers_modules_9_modules_fc2_parameters_bias_,
        )
        hidden_states_88 = (
            l_self_modules_model_modules_layers_modules_9_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_model_modules_layers_modules_9_modules_fc2_parameters_bias_
        ) = None
        hidden_states_90 = torch.nn.functional.dropout(
            hidden_states_89, p=0.1, training=False
        )
        hidden_states_89 = None
        hidden_states_91 = hidden_states_85 + hidden_states_90
        hidden_states_85 = hidden_states_90 = None
        hidden_states_92 = torch.nn.functional.layer_norm(
            hidden_states_91,
            (2048,),
            l_self_modules_model_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_60 = torch._C._nn.linear(
            hidden_states_92,
            l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_30 = linear_60 * 0.125
        linear_60 = None
        key_states_30 = torch._C._nn.linear(
            hidden_states_92,
            l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_30 = torch._C._nn.linear(
            hidden_states_92,
            l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_92 = l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_64 = key_states_30.view(1, 19, -1, 64)
        key_states_30 = None
        key_states_31 = view_64.transpose(1, 2)
        view_64 = None
        view_65 = value_states_30.view(1, 19, -1, 64)
        value_states_30 = None
        value_states_31 = view_65.transpose(1, 2)
        view_65 = None
        view_66 = query_states_30.view(1, 19, 32, 64)
        query_states_30 = None
        query_states_31 = view_66.transpose(1, 2)
        view_66 = None
        query_states_32 = query_states_31.reshape(32, -1, 64)
        query_states_31 = None
        key_states_32 = key_states_31.reshape(32, -1, 64)
        key_states_31 = None
        value_states_32 = value_states_31.reshape(32, -1, 64)
        value_states_31 = None
        transpose_53 = key_states_32.transpose(1, 2)
        key_states_32 = None
        attn_weights_50 = torch.bmm(query_states_32, transpose_53)
        query_states_32 = transpose_53 = None
        view_67 = attn_weights_50.view(1, 32, 19, 19)
        attn_weights_50 = None
        attn_weights_51 = view_67 + expanded_attn_mask_1
        view_67 = None
        tensor_11 = torch.tensor(-65504.0, device=device(type="cpu"))
        attn_weights_52 = torch.max(attn_weights_51, tensor_11)
        attn_weights_51 = tensor_11 = None
        attn_weights_53 = attn_weights_52.view(32, 19, 19)
        attn_weights_52 = None
        softmax_10 = torch.nn.functional.softmax(
            attn_weights_53, dim=-1, dtype=torch.float32
        )
        attn_weights_53 = None
        attn_weights_54 = softmax_10.to(torch.float16)
        softmax_10 = None
        attn_probs_10 = torch.nn.functional.dropout(
            attn_weights_54, p=0.1, training=False
        )
        attn_weights_54 = None
        attn_output_50 = torch.bmm(attn_probs_10, value_states_32)
        attn_probs_10 = value_states_32 = None
        attn_output_51 = attn_output_50.view(1, 32, 19, 64)
        attn_output_50 = None
        attn_output_52 = attn_output_51.transpose(1, 2)
        attn_output_51 = None
        attn_output_53 = attn_output_52.reshape(1, 19, 2048)
        attn_output_52 = None
        attn_output_54 = torch._C._nn.linear(
            attn_output_53,
            l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_53 = l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_93 = torch.nn.functional.dropout(
            attn_output_54, p=0.1, training=False
        )
        attn_output_54 = None
        hidden_states_94 = hidden_states_91 + hidden_states_93
        hidden_states_91 = hidden_states_93 = None
        hidden_states_95 = torch.nn.functional.layer_norm(
            hidden_states_94,
            (2048,),
            l_self_modules_model_modules_layers_modules_10_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_10_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_10_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_10_modules_final_layer_norm_parameters_bias_ = (None)
        linear_64 = torch._C._nn.linear(
            hidden_states_95,
            l_self_modules_model_modules_layers_modules_10_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_layers_modules_10_modules_fc1_parameters_bias_,
        )
        hidden_states_95 = l_self_modules_model_modules_layers_modules_10_modules_fc1_parameters_weight_ = (
            l_self_modules_model_modules_layers_modules_10_modules_fc1_parameters_bias_
        ) = None
        hidden_states_96 = torch._C._nn.gelu(linear_64)
        linear_64 = None
        hidden_states_97 = torch.nn.functional.dropout(
            hidden_states_96, p=0.0, training=False
        )
        hidden_states_96 = None
        hidden_states_98 = torch._C._nn.linear(
            hidden_states_97,
            l_self_modules_model_modules_layers_modules_10_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_layers_modules_10_modules_fc2_parameters_bias_,
        )
        hidden_states_97 = l_self_modules_model_modules_layers_modules_10_modules_fc2_parameters_weight_ = (
            l_self_modules_model_modules_layers_modules_10_modules_fc2_parameters_bias_
        ) = None
        hidden_states_99 = torch.nn.functional.dropout(
            hidden_states_98, p=0.1, training=False
        )
        hidden_states_98 = None
        hidden_states_100 = hidden_states_94 + hidden_states_99
        hidden_states_94 = hidden_states_99 = None
        hidden_states_101 = torch.nn.functional.layer_norm(
            hidden_states_100,
            (2048,),
            l_self_modules_model_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_66 = torch._C._nn.linear(
            hidden_states_101,
            l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_33 = linear_66 * 0.125
        linear_66 = None
        key_states_33 = torch._C._nn.linear(
            hidden_states_101,
            l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_33 = torch._C._nn.linear(
            hidden_states_101,
            l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_101 = l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_70 = key_states_33.view(1, 19, -1, 64)
        key_states_33 = None
        key_states_34 = view_70.transpose(1, 2)
        view_70 = None
        view_71 = value_states_33.view(1, 19, -1, 64)
        value_states_33 = None
        value_states_34 = view_71.transpose(1, 2)
        view_71 = None
        view_72 = query_states_33.view(1, 19, 32, 64)
        query_states_33 = None
        query_states_34 = view_72.transpose(1, 2)
        view_72 = None
        query_states_35 = query_states_34.reshape(32, -1, 64)
        query_states_34 = None
        key_states_35 = key_states_34.reshape(32, -1, 64)
        key_states_34 = None
        value_states_35 = value_states_34.reshape(32, -1, 64)
        value_states_34 = None
        transpose_58 = key_states_35.transpose(1, 2)
        key_states_35 = None
        attn_weights_55 = torch.bmm(query_states_35, transpose_58)
        query_states_35 = transpose_58 = None
        view_73 = attn_weights_55.view(1, 32, 19, 19)
        attn_weights_55 = None
        attn_weights_56 = view_73 + expanded_attn_mask_1
        view_73 = None
        tensor_12 = torch.tensor(-65504.0, device=device(type="cpu"))
        attn_weights_57 = torch.max(attn_weights_56, tensor_12)
        attn_weights_56 = tensor_12 = None
        attn_weights_58 = attn_weights_57.view(32, 19, 19)
        attn_weights_57 = None
        softmax_11 = torch.nn.functional.softmax(
            attn_weights_58, dim=-1, dtype=torch.float32
        )
        attn_weights_58 = None
        attn_weights_59 = softmax_11.to(torch.float16)
        softmax_11 = None
        attn_probs_11 = torch.nn.functional.dropout(
            attn_weights_59, p=0.1, training=False
        )
        attn_weights_59 = None
        attn_output_55 = torch.bmm(attn_probs_11, value_states_35)
        attn_probs_11 = value_states_35 = None
        attn_output_56 = attn_output_55.view(1, 32, 19, 64)
        attn_output_55 = None
        attn_output_57 = attn_output_56.transpose(1, 2)
        attn_output_56 = None
        attn_output_58 = attn_output_57.reshape(1, 19, 2048)
        attn_output_57 = None
        attn_output_59 = torch._C._nn.linear(
            attn_output_58,
            l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_58 = l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_102 = torch.nn.functional.dropout(
            attn_output_59, p=0.1, training=False
        )
        attn_output_59 = None
        hidden_states_103 = hidden_states_100 + hidden_states_102
        hidden_states_100 = hidden_states_102 = None
        hidden_states_104 = torch.nn.functional.layer_norm(
            hidden_states_103,
            (2048,),
            l_self_modules_model_modules_layers_modules_11_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_11_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_11_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_11_modules_final_layer_norm_parameters_bias_ = (None)
        linear_70 = torch._C._nn.linear(
            hidden_states_104,
            l_self_modules_model_modules_layers_modules_11_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_layers_modules_11_modules_fc1_parameters_bias_,
        )
        hidden_states_104 = l_self_modules_model_modules_layers_modules_11_modules_fc1_parameters_weight_ = (
            l_self_modules_model_modules_layers_modules_11_modules_fc1_parameters_bias_
        ) = None
        hidden_states_105 = torch._C._nn.gelu(linear_70)
        linear_70 = None
        hidden_states_106 = torch.nn.functional.dropout(
            hidden_states_105, p=0.0, training=False
        )
        hidden_states_105 = None
        hidden_states_107 = torch._C._nn.linear(
            hidden_states_106,
            l_self_modules_model_modules_layers_modules_11_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_layers_modules_11_modules_fc2_parameters_bias_,
        )
        hidden_states_106 = l_self_modules_model_modules_layers_modules_11_modules_fc2_parameters_weight_ = (
            l_self_modules_model_modules_layers_modules_11_modules_fc2_parameters_bias_
        ) = None
        hidden_states_108 = torch.nn.functional.dropout(
            hidden_states_107, p=0.1, training=False
        )
        hidden_states_107 = None
        hidden_states_109 = hidden_states_103 + hidden_states_108
        hidden_states_103 = hidden_states_108 = None
        hidden_states_110 = torch.nn.functional.layer_norm(
            hidden_states_109,
            (2048,),
            l_self_modules_model_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_72 = torch._C._nn.linear(
            hidden_states_110,
            l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_36 = linear_72 * 0.125
        linear_72 = None
        key_states_36 = torch._C._nn.linear(
            hidden_states_110,
            l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_36 = torch._C._nn.linear(
            hidden_states_110,
            l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_110 = l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_76 = key_states_36.view(1, 19, -1, 64)
        key_states_36 = None
        key_states_37 = view_76.transpose(1, 2)
        view_76 = None
        view_77 = value_states_36.view(1, 19, -1, 64)
        value_states_36 = None
        value_states_37 = view_77.transpose(1, 2)
        view_77 = None
        view_78 = query_states_36.view(1, 19, 32, 64)
        query_states_36 = None
        query_states_37 = view_78.transpose(1, 2)
        view_78 = None
        query_states_38 = query_states_37.reshape(32, -1, 64)
        query_states_37 = None
        key_states_38 = key_states_37.reshape(32, -1, 64)
        key_states_37 = None
        value_states_38 = value_states_37.reshape(32, -1, 64)
        value_states_37 = None
        transpose_63 = key_states_38.transpose(1, 2)
        key_states_38 = None
        attn_weights_60 = torch.bmm(query_states_38, transpose_63)
        query_states_38 = transpose_63 = None
        view_79 = attn_weights_60.view(1, 32, 19, 19)
        attn_weights_60 = None
        attn_weights_61 = view_79 + expanded_attn_mask_1
        view_79 = None
        tensor_13 = torch.tensor(-65504.0, device=device(type="cpu"))
        attn_weights_62 = torch.max(attn_weights_61, tensor_13)
        attn_weights_61 = tensor_13 = None
        attn_weights_63 = attn_weights_62.view(32, 19, 19)
        attn_weights_62 = None
        softmax_12 = torch.nn.functional.softmax(
            attn_weights_63, dim=-1, dtype=torch.float32
        )
        attn_weights_63 = None
        attn_weights_64 = softmax_12.to(torch.float16)
        softmax_12 = None
        attn_probs_12 = torch.nn.functional.dropout(
            attn_weights_64, p=0.1, training=False
        )
        attn_weights_64 = None
        attn_output_60 = torch.bmm(attn_probs_12, value_states_38)
        attn_probs_12 = value_states_38 = None
        attn_output_61 = attn_output_60.view(1, 32, 19, 64)
        attn_output_60 = None
        attn_output_62 = attn_output_61.transpose(1, 2)
        attn_output_61 = None
        attn_output_63 = attn_output_62.reshape(1, 19, 2048)
        attn_output_62 = None
        attn_output_64 = torch._C._nn.linear(
            attn_output_63,
            l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_63 = l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_111 = torch.nn.functional.dropout(
            attn_output_64, p=0.1, training=False
        )
        attn_output_64 = None
        hidden_states_112 = hidden_states_109 + hidden_states_111
        hidden_states_109 = hidden_states_111 = None
        hidden_states_113 = torch.nn.functional.layer_norm(
            hidden_states_112,
            (2048,),
            l_self_modules_model_modules_layers_modules_12_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_12_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_12_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_12_modules_final_layer_norm_parameters_bias_ = (None)
        linear_76 = torch._C._nn.linear(
            hidden_states_113,
            l_self_modules_model_modules_layers_modules_12_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_layers_modules_12_modules_fc1_parameters_bias_,
        )
        hidden_states_113 = l_self_modules_model_modules_layers_modules_12_modules_fc1_parameters_weight_ = (
            l_self_modules_model_modules_layers_modules_12_modules_fc1_parameters_bias_
        ) = None
        hidden_states_114 = torch._C._nn.gelu(linear_76)
        linear_76 = None
        hidden_states_115 = torch.nn.functional.dropout(
            hidden_states_114, p=0.0, training=False
        )
        hidden_states_114 = None
        hidden_states_116 = torch._C._nn.linear(
            hidden_states_115,
            l_self_modules_model_modules_layers_modules_12_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_layers_modules_12_modules_fc2_parameters_bias_,
        )
        hidden_states_115 = l_self_modules_model_modules_layers_modules_12_modules_fc2_parameters_weight_ = (
            l_self_modules_model_modules_layers_modules_12_modules_fc2_parameters_bias_
        ) = None
        hidden_states_117 = torch.nn.functional.dropout(
            hidden_states_116, p=0.1, training=False
        )
        hidden_states_116 = None
        hidden_states_118 = hidden_states_112 + hidden_states_117
        hidden_states_112 = hidden_states_117 = None
        hidden_states_119 = torch.nn.functional.layer_norm(
            hidden_states_118,
            (2048,),
            l_self_modules_model_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_78 = torch._C._nn.linear(
            hidden_states_119,
            l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_39 = linear_78 * 0.125
        linear_78 = None
        key_states_39 = torch._C._nn.linear(
            hidden_states_119,
            l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_39 = torch._C._nn.linear(
            hidden_states_119,
            l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_119 = l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_82 = key_states_39.view(1, 19, -1, 64)
        key_states_39 = None
        key_states_40 = view_82.transpose(1, 2)
        view_82 = None
        view_83 = value_states_39.view(1, 19, -1, 64)
        value_states_39 = None
        value_states_40 = view_83.transpose(1, 2)
        view_83 = None
        view_84 = query_states_39.view(1, 19, 32, 64)
        query_states_39 = None
        query_states_40 = view_84.transpose(1, 2)
        view_84 = None
        query_states_41 = query_states_40.reshape(32, -1, 64)
        query_states_40 = None
        key_states_41 = key_states_40.reshape(32, -1, 64)
        key_states_40 = None
        value_states_41 = value_states_40.reshape(32, -1, 64)
        value_states_40 = None
        transpose_68 = key_states_41.transpose(1, 2)
        key_states_41 = None
        attn_weights_65 = torch.bmm(query_states_41, transpose_68)
        query_states_41 = transpose_68 = None
        view_85 = attn_weights_65.view(1, 32, 19, 19)
        attn_weights_65 = None
        attn_weights_66 = view_85 + expanded_attn_mask_1
        view_85 = None
        tensor_14 = torch.tensor(-65504.0, device=device(type="cpu"))
        attn_weights_67 = torch.max(attn_weights_66, tensor_14)
        attn_weights_66 = tensor_14 = None
        attn_weights_68 = attn_weights_67.view(32, 19, 19)
        attn_weights_67 = None
        softmax_13 = torch.nn.functional.softmax(
            attn_weights_68, dim=-1, dtype=torch.float32
        )
        attn_weights_68 = None
        attn_weights_69 = softmax_13.to(torch.float16)
        softmax_13 = None
        attn_probs_13 = torch.nn.functional.dropout(
            attn_weights_69, p=0.1, training=False
        )
        attn_weights_69 = None
        attn_output_65 = torch.bmm(attn_probs_13, value_states_41)
        attn_probs_13 = value_states_41 = None
        attn_output_66 = attn_output_65.view(1, 32, 19, 64)
        attn_output_65 = None
        attn_output_67 = attn_output_66.transpose(1, 2)
        attn_output_66 = None
        attn_output_68 = attn_output_67.reshape(1, 19, 2048)
        attn_output_67 = None
        attn_output_69 = torch._C._nn.linear(
            attn_output_68,
            l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_68 = l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_120 = torch.nn.functional.dropout(
            attn_output_69, p=0.1, training=False
        )
        attn_output_69 = None
        hidden_states_121 = hidden_states_118 + hidden_states_120
        hidden_states_118 = hidden_states_120 = None
        hidden_states_122 = torch.nn.functional.layer_norm(
            hidden_states_121,
            (2048,),
            l_self_modules_model_modules_layers_modules_13_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_13_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_13_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_13_modules_final_layer_norm_parameters_bias_ = (None)
        linear_82 = torch._C._nn.linear(
            hidden_states_122,
            l_self_modules_model_modules_layers_modules_13_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_layers_modules_13_modules_fc1_parameters_bias_,
        )
        hidden_states_122 = l_self_modules_model_modules_layers_modules_13_modules_fc1_parameters_weight_ = (
            l_self_modules_model_modules_layers_modules_13_modules_fc1_parameters_bias_
        ) = None
        hidden_states_123 = torch._C._nn.gelu(linear_82)
        linear_82 = None
        hidden_states_124 = torch.nn.functional.dropout(
            hidden_states_123, p=0.0, training=False
        )
        hidden_states_123 = None
        hidden_states_125 = torch._C._nn.linear(
            hidden_states_124,
            l_self_modules_model_modules_layers_modules_13_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_layers_modules_13_modules_fc2_parameters_bias_,
        )
        hidden_states_124 = l_self_modules_model_modules_layers_modules_13_modules_fc2_parameters_weight_ = (
            l_self_modules_model_modules_layers_modules_13_modules_fc2_parameters_bias_
        ) = None
        hidden_states_126 = torch.nn.functional.dropout(
            hidden_states_125, p=0.1, training=False
        )
        hidden_states_125 = None
        hidden_states_127 = hidden_states_121 + hidden_states_126
        hidden_states_121 = hidden_states_126 = None
        hidden_states_128 = torch.nn.functional.layer_norm(
            hidden_states_127,
            (2048,),
            l_self_modules_model_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_84 = torch._C._nn.linear(
            hidden_states_128,
            l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_42 = linear_84 * 0.125
        linear_84 = None
        key_states_42 = torch._C._nn.linear(
            hidden_states_128,
            l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_42 = torch._C._nn.linear(
            hidden_states_128,
            l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_128 = l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_88 = key_states_42.view(1, 19, -1, 64)
        key_states_42 = None
        key_states_43 = view_88.transpose(1, 2)
        view_88 = None
        view_89 = value_states_42.view(1, 19, -1, 64)
        value_states_42 = None
        value_states_43 = view_89.transpose(1, 2)
        view_89 = None
        view_90 = query_states_42.view(1, 19, 32, 64)
        query_states_42 = None
        query_states_43 = view_90.transpose(1, 2)
        view_90 = None
        query_states_44 = query_states_43.reshape(32, -1, 64)
        query_states_43 = None
        key_states_44 = key_states_43.reshape(32, -1, 64)
        key_states_43 = None
        value_states_44 = value_states_43.reshape(32, -1, 64)
        value_states_43 = None
        transpose_73 = key_states_44.transpose(1, 2)
        key_states_44 = None
        attn_weights_70 = torch.bmm(query_states_44, transpose_73)
        query_states_44 = transpose_73 = None
        view_91 = attn_weights_70.view(1, 32, 19, 19)
        attn_weights_70 = None
        attn_weights_71 = view_91 + expanded_attn_mask_1
        view_91 = None
        tensor_15 = torch.tensor(-65504.0, device=device(type="cpu"))
        attn_weights_72 = torch.max(attn_weights_71, tensor_15)
        attn_weights_71 = tensor_15 = None
        attn_weights_73 = attn_weights_72.view(32, 19, 19)
        attn_weights_72 = None
        softmax_14 = torch.nn.functional.softmax(
            attn_weights_73, dim=-1, dtype=torch.float32
        )
        attn_weights_73 = None
        attn_weights_74 = softmax_14.to(torch.float16)
        softmax_14 = None
        attn_probs_14 = torch.nn.functional.dropout(
            attn_weights_74, p=0.1, training=False
        )
        attn_weights_74 = None
        attn_output_70 = torch.bmm(attn_probs_14, value_states_44)
        attn_probs_14 = value_states_44 = None
        attn_output_71 = attn_output_70.view(1, 32, 19, 64)
        attn_output_70 = None
        attn_output_72 = attn_output_71.transpose(1, 2)
        attn_output_71 = None
        attn_output_73 = attn_output_72.reshape(1, 19, 2048)
        attn_output_72 = None
        attn_output_74 = torch._C._nn.linear(
            attn_output_73,
            l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_73 = l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_129 = torch.nn.functional.dropout(
            attn_output_74, p=0.1, training=False
        )
        attn_output_74 = None
        hidden_states_130 = hidden_states_127 + hidden_states_129
        hidden_states_127 = hidden_states_129 = None
        hidden_states_131 = torch.nn.functional.layer_norm(
            hidden_states_130,
            (2048,),
            l_self_modules_model_modules_layers_modules_14_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_14_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_14_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_14_modules_final_layer_norm_parameters_bias_ = (None)
        linear_88 = torch._C._nn.linear(
            hidden_states_131,
            l_self_modules_model_modules_layers_modules_14_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_layers_modules_14_modules_fc1_parameters_bias_,
        )
        hidden_states_131 = l_self_modules_model_modules_layers_modules_14_modules_fc1_parameters_weight_ = (
            l_self_modules_model_modules_layers_modules_14_modules_fc1_parameters_bias_
        ) = None
        hidden_states_132 = torch._C._nn.gelu(linear_88)
        linear_88 = None
        hidden_states_133 = torch.nn.functional.dropout(
            hidden_states_132, p=0.0, training=False
        )
        hidden_states_132 = None
        hidden_states_134 = torch._C._nn.linear(
            hidden_states_133,
            l_self_modules_model_modules_layers_modules_14_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_layers_modules_14_modules_fc2_parameters_bias_,
        )
        hidden_states_133 = l_self_modules_model_modules_layers_modules_14_modules_fc2_parameters_weight_ = (
            l_self_modules_model_modules_layers_modules_14_modules_fc2_parameters_bias_
        ) = None
        hidden_states_135 = torch.nn.functional.dropout(
            hidden_states_134, p=0.1, training=False
        )
        hidden_states_134 = None
        hidden_states_136 = hidden_states_130 + hidden_states_135
        hidden_states_130 = hidden_states_135 = None
        hidden_states_137 = torch.nn.functional.layer_norm(
            hidden_states_136,
            (2048,),
            l_self_modules_model_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_90 = torch._C._nn.linear(
            hidden_states_137,
            l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_45 = linear_90 * 0.125
        linear_90 = None
        key_states_45 = torch._C._nn.linear(
            hidden_states_137,
            l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_45 = torch._C._nn.linear(
            hidden_states_137,
            l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_137 = l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_94 = key_states_45.view(1, 19, -1, 64)
        key_states_45 = None
        key_states_46 = view_94.transpose(1, 2)
        view_94 = None
        view_95 = value_states_45.view(1, 19, -1, 64)
        value_states_45 = None
        value_states_46 = view_95.transpose(1, 2)
        view_95 = None
        view_96 = query_states_45.view(1, 19, 32, 64)
        query_states_45 = None
        query_states_46 = view_96.transpose(1, 2)
        view_96 = None
        query_states_47 = query_states_46.reshape(32, -1, 64)
        query_states_46 = None
        key_states_47 = key_states_46.reshape(32, -1, 64)
        key_states_46 = None
        value_states_47 = value_states_46.reshape(32, -1, 64)
        value_states_46 = None
        transpose_78 = key_states_47.transpose(1, 2)
        key_states_47 = None
        attn_weights_75 = torch.bmm(query_states_47, transpose_78)
        query_states_47 = transpose_78 = None
        view_97 = attn_weights_75.view(1, 32, 19, 19)
        attn_weights_75 = None
        attn_weights_76 = view_97 + expanded_attn_mask_1
        view_97 = None
        tensor_16 = torch.tensor(-65504.0, device=device(type="cpu"))
        attn_weights_77 = torch.max(attn_weights_76, tensor_16)
        attn_weights_76 = tensor_16 = None
        attn_weights_78 = attn_weights_77.view(32, 19, 19)
        attn_weights_77 = None
        softmax_15 = torch.nn.functional.softmax(
            attn_weights_78, dim=-1, dtype=torch.float32
        )
        attn_weights_78 = None
        attn_weights_79 = softmax_15.to(torch.float16)
        softmax_15 = None
        attn_probs_15 = torch.nn.functional.dropout(
            attn_weights_79, p=0.1, training=False
        )
        attn_weights_79 = None
        attn_output_75 = torch.bmm(attn_probs_15, value_states_47)
        attn_probs_15 = value_states_47 = None
        attn_output_76 = attn_output_75.view(1, 32, 19, 64)
        attn_output_75 = None
        attn_output_77 = attn_output_76.transpose(1, 2)
        attn_output_76 = None
        attn_output_78 = attn_output_77.reshape(1, 19, 2048)
        attn_output_77 = None
        attn_output_79 = torch._C._nn.linear(
            attn_output_78,
            l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_78 = l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_138 = torch.nn.functional.dropout(
            attn_output_79, p=0.1, training=False
        )
        attn_output_79 = None
        hidden_states_139 = hidden_states_136 + hidden_states_138
        hidden_states_136 = hidden_states_138 = None
        hidden_states_140 = torch.nn.functional.layer_norm(
            hidden_states_139,
            (2048,),
            l_self_modules_model_modules_layers_modules_15_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_15_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_15_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_15_modules_final_layer_norm_parameters_bias_ = (None)
        linear_94 = torch._C._nn.linear(
            hidden_states_140,
            l_self_modules_model_modules_layers_modules_15_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_layers_modules_15_modules_fc1_parameters_bias_,
        )
        hidden_states_140 = l_self_modules_model_modules_layers_modules_15_modules_fc1_parameters_weight_ = (
            l_self_modules_model_modules_layers_modules_15_modules_fc1_parameters_bias_
        ) = None
        hidden_states_141 = torch._C._nn.gelu(linear_94)
        linear_94 = None
        hidden_states_142 = torch.nn.functional.dropout(
            hidden_states_141, p=0.0, training=False
        )
        hidden_states_141 = None
        hidden_states_143 = torch._C._nn.linear(
            hidden_states_142,
            l_self_modules_model_modules_layers_modules_15_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_layers_modules_15_modules_fc2_parameters_bias_,
        )
        hidden_states_142 = l_self_modules_model_modules_layers_modules_15_modules_fc2_parameters_weight_ = (
            l_self_modules_model_modules_layers_modules_15_modules_fc2_parameters_bias_
        ) = None
        hidden_states_144 = torch.nn.functional.dropout(
            hidden_states_143, p=0.1, training=False
        )
        hidden_states_143 = None
        hidden_states_145 = hidden_states_139 + hidden_states_144
        hidden_states_139 = hidden_states_144 = None
        hidden_states_146 = torch.nn.functional.layer_norm(
            hidden_states_145,
            (2048,),
            l_self_modules_model_modules_layers_modules_16_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_16_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_16_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_96 = torch._C._nn.linear(
            hidden_states_146,
            l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_48 = linear_96 * 0.125
        linear_96 = None
        key_states_48 = torch._C._nn.linear(
            hidden_states_146,
            l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_48 = torch._C._nn.linear(
            hidden_states_146,
            l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_146 = l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_100 = key_states_48.view(1, 19, -1, 64)
        key_states_48 = None
        key_states_49 = view_100.transpose(1, 2)
        view_100 = None
        view_101 = value_states_48.view(1, 19, -1, 64)
        value_states_48 = None
        value_states_49 = view_101.transpose(1, 2)
        view_101 = None
        view_102 = query_states_48.view(1, 19, 32, 64)
        query_states_48 = None
        query_states_49 = view_102.transpose(1, 2)
        view_102 = None
        query_states_50 = query_states_49.reshape(32, -1, 64)
        query_states_49 = None
        key_states_50 = key_states_49.reshape(32, -1, 64)
        key_states_49 = None
        value_states_50 = value_states_49.reshape(32, -1, 64)
        value_states_49 = None
        transpose_83 = key_states_50.transpose(1, 2)
        key_states_50 = None
        attn_weights_80 = torch.bmm(query_states_50, transpose_83)
        query_states_50 = transpose_83 = None
        view_103 = attn_weights_80.view(1, 32, 19, 19)
        attn_weights_80 = None
        attn_weights_81 = view_103 + expanded_attn_mask_1
        view_103 = None
        tensor_17 = torch.tensor(-65504.0, device=device(type="cpu"))
        attn_weights_82 = torch.max(attn_weights_81, tensor_17)
        attn_weights_81 = tensor_17 = None
        attn_weights_83 = attn_weights_82.view(32, 19, 19)
        attn_weights_82 = None
        softmax_16 = torch.nn.functional.softmax(
            attn_weights_83, dim=-1, dtype=torch.float32
        )
        attn_weights_83 = None
        attn_weights_84 = softmax_16.to(torch.float16)
        softmax_16 = None
        attn_probs_16 = torch.nn.functional.dropout(
            attn_weights_84, p=0.1, training=False
        )
        attn_weights_84 = None
        attn_output_80 = torch.bmm(attn_probs_16, value_states_50)
        attn_probs_16 = value_states_50 = None
        attn_output_81 = attn_output_80.view(1, 32, 19, 64)
        attn_output_80 = None
        attn_output_82 = attn_output_81.transpose(1, 2)
        attn_output_81 = None
        attn_output_83 = attn_output_82.reshape(1, 19, 2048)
        attn_output_82 = None
        attn_output_84 = torch._C._nn.linear(
            attn_output_83,
            l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_83 = l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_147 = torch.nn.functional.dropout(
            attn_output_84, p=0.1, training=False
        )
        attn_output_84 = None
        hidden_states_148 = hidden_states_145 + hidden_states_147
        hidden_states_145 = hidden_states_147 = None
        hidden_states_149 = torch.nn.functional.layer_norm(
            hidden_states_148,
            (2048,),
            l_self_modules_model_modules_layers_modules_16_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_16_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_16_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_16_modules_final_layer_norm_parameters_bias_ = (None)
        linear_100 = torch._C._nn.linear(
            hidden_states_149,
            l_self_modules_model_modules_layers_modules_16_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_layers_modules_16_modules_fc1_parameters_bias_,
        )
        hidden_states_149 = l_self_modules_model_modules_layers_modules_16_modules_fc1_parameters_weight_ = (
            l_self_modules_model_modules_layers_modules_16_modules_fc1_parameters_bias_
        ) = None
        hidden_states_150 = torch._C._nn.gelu(linear_100)
        linear_100 = None
        hidden_states_151 = torch.nn.functional.dropout(
            hidden_states_150, p=0.0, training=False
        )
        hidden_states_150 = None
        hidden_states_152 = torch._C._nn.linear(
            hidden_states_151,
            l_self_modules_model_modules_layers_modules_16_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_layers_modules_16_modules_fc2_parameters_bias_,
        )
        hidden_states_151 = l_self_modules_model_modules_layers_modules_16_modules_fc2_parameters_weight_ = (
            l_self_modules_model_modules_layers_modules_16_modules_fc2_parameters_bias_
        ) = None
        hidden_states_153 = torch.nn.functional.dropout(
            hidden_states_152, p=0.1, training=False
        )
        hidden_states_152 = None
        hidden_states_154 = hidden_states_148 + hidden_states_153
        hidden_states_148 = hidden_states_153 = None
        hidden_states_155 = torch.nn.functional.layer_norm(
            hidden_states_154,
            (2048,),
            l_self_modules_model_modules_layers_modules_17_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_17_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_17_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_102 = torch._C._nn.linear(
            hidden_states_155,
            l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_51 = linear_102 * 0.125
        linear_102 = None
        key_states_51 = torch._C._nn.linear(
            hidden_states_155,
            l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_51 = torch._C._nn.linear(
            hidden_states_155,
            l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_155 = l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_106 = key_states_51.view(1, 19, -1, 64)
        key_states_51 = None
        key_states_52 = view_106.transpose(1, 2)
        view_106 = None
        view_107 = value_states_51.view(1, 19, -1, 64)
        value_states_51 = None
        value_states_52 = view_107.transpose(1, 2)
        view_107 = None
        view_108 = query_states_51.view(1, 19, 32, 64)
        query_states_51 = None
        query_states_52 = view_108.transpose(1, 2)
        view_108 = None
        query_states_53 = query_states_52.reshape(32, -1, 64)
        query_states_52 = None
        key_states_53 = key_states_52.reshape(32, -1, 64)
        key_states_52 = None
        value_states_53 = value_states_52.reshape(32, -1, 64)
        value_states_52 = None
        transpose_88 = key_states_53.transpose(1, 2)
        key_states_53 = None
        attn_weights_85 = torch.bmm(query_states_53, transpose_88)
        query_states_53 = transpose_88 = None
        view_109 = attn_weights_85.view(1, 32, 19, 19)
        attn_weights_85 = None
        attn_weights_86 = view_109 + expanded_attn_mask_1
        view_109 = None
        tensor_18 = torch.tensor(-65504.0, device=device(type="cpu"))
        attn_weights_87 = torch.max(attn_weights_86, tensor_18)
        attn_weights_86 = tensor_18 = None
        attn_weights_88 = attn_weights_87.view(32, 19, 19)
        attn_weights_87 = None
        softmax_17 = torch.nn.functional.softmax(
            attn_weights_88, dim=-1, dtype=torch.float32
        )
        attn_weights_88 = None
        attn_weights_89 = softmax_17.to(torch.float16)
        softmax_17 = None
        attn_probs_17 = torch.nn.functional.dropout(
            attn_weights_89, p=0.1, training=False
        )
        attn_weights_89 = None
        attn_output_85 = torch.bmm(attn_probs_17, value_states_53)
        attn_probs_17 = value_states_53 = None
        attn_output_86 = attn_output_85.view(1, 32, 19, 64)
        attn_output_85 = None
        attn_output_87 = attn_output_86.transpose(1, 2)
        attn_output_86 = None
        attn_output_88 = attn_output_87.reshape(1, 19, 2048)
        attn_output_87 = None
        attn_output_89 = torch._C._nn.linear(
            attn_output_88,
            l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_88 = l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_156 = torch.nn.functional.dropout(
            attn_output_89, p=0.1, training=False
        )
        attn_output_89 = None
        hidden_states_157 = hidden_states_154 + hidden_states_156
        hidden_states_154 = hidden_states_156 = None
        hidden_states_158 = torch.nn.functional.layer_norm(
            hidden_states_157,
            (2048,),
            l_self_modules_model_modules_layers_modules_17_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_17_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_17_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_17_modules_final_layer_norm_parameters_bias_ = (None)
        linear_106 = torch._C._nn.linear(
            hidden_states_158,
            l_self_modules_model_modules_layers_modules_17_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_layers_modules_17_modules_fc1_parameters_bias_,
        )
        hidden_states_158 = l_self_modules_model_modules_layers_modules_17_modules_fc1_parameters_weight_ = (
            l_self_modules_model_modules_layers_modules_17_modules_fc1_parameters_bias_
        ) = None
        hidden_states_159 = torch._C._nn.gelu(linear_106)
        linear_106 = None
        hidden_states_160 = torch.nn.functional.dropout(
            hidden_states_159, p=0.0, training=False
        )
        hidden_states_159 = None
        hidden_states_161 = torch._C._nn.linear(
            hidden_states_160,
            l_self_modules_model_modules_layers_modules_17_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_layers_modules_17_modules_fc2_parameters_bias_,
        )
        hidden_states_160 = l_self_modules_model_modules_layers_modules_17_modules_fc2_parameters_weight_ = (
            l_self_modules_model_modules_layers_modules_17_modules_fc2_parameters_bias_
        ) = None
        hidden_states_162 = torch.nn.functional.dropout(
            hidden_states_161, p=0.1, training=False
        )
        hidden_states_161 = None
        hidden_states_163 = hidden_states_157 + hidden_states_162
        hidden_states_157 = hidden_states_162 = None
        hidden_states_164 = torch.nn.functional.layer_norm(
            hidden_states_163,
            (2048,),
            l_self_modules_model_modules_layers_modules_18_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_18_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_18_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_108 = torch._C._nn.linear(
            hidden_states_164,
            l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_54 = linear_108 * 0.125
        linear_108 = None
        key_states_54 = torch._C._nn.linear(
            hidden_states_164,
            l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_54 = torch._C._nn.linear(
            hidden_states_164,
            l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_164 = l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_112 = key_states_54.view(1, 19, -1, 64)
        key_states_54 = None
        key_states_55 = view_112.transpose(1, 2)
        view_112 = None
        view_113 = value_states_54.view(1, 19, -1, 64)
        value_states_54 = None
        value_states_55 = view_113.transpose(1, 2)
        view_113 = None
        view_114 = query_states_54.view(1, 19, 32, 64)
        query_states_54 = None
        query_states_55 = view_114.transpose(1, 2)
        view_114 = None
        query_states_56 = query_states_55.reshape(32, -1, 64)
        query_states_55 = None
        key_states_56 = key_states_55.reshape(32, -1, 64)
        key_states_55 = None
        value_states_56 = value_states_55.reshape(32, -1, 64)
        value_states_55 = None
        transpose_93 = key_states_56.transpose(1, 2)
        key_states_56 = None
        attn_weights_90 = torch.bmm(query_states_56, transpose_93)
        query_states_56 = transpose_93 = None
        view_115 = attn_weights_90.view(1, 32, 19, 19)
        attn_weights_90 = None
        attn_weights_91 = view_115 + expanded_attn_mask_1
        view_115 = None
        tensor_19 = torch.tensor(-65504.0, device=device(type="cpu"))
        attn_weights_92 = torch.max(attn_weights_91, tensor_19)
        attn_weights_91 = tensor_19 = None
        attn_weights_93 = attn_weights_92.view(32, 19, 19)
        attn_weights_92 = None
        softmax_18 = torch.nn.functional.softmax(
            attn_weights_93, dim=-1, dtype=torch.float32
        )
        attn_weights_93 = None
        attn_weights_94 = softmax_18.to(torch.float16)
        softmax_18 = None
        attn_probs_18 = torch.nn.functional.dropout(
            attn_weights_94, p=0.1, training=False
        )
        attn_weights_94 = None
        attn_output_90 = torch.bmm(attn_probs_18, value_states_56)
        attn_probs_18 = value_states_56 = None
        attn_output_91 = attn_output_90.view(1, 32, 19, 64)
        attn_output_90 = None
        attn_output_92 = attn_output_91.transpose(1, 2)
        attn_output_91 = None
        attn_output_93 = attn_output_92.reshape(1, 19, 2048)
        attn_output_92 = None
        attn_output_94 = torch._C._nn.linear(
            attn_output_93,
            l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_93 = l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_165 = torch.nn.functional.dropout(
            attn_output_94, p=0.1, training=False
        )
        attn_output_94 = None
        hidden_states_166 = hidden_states_163 + hidden_states_165
        hidden_states_163 = hidden_states_165 = None
        hidden_states_167 = torch.nn.functional.layer_norm(
            hidden_states_166,
            (2048,),
            l_self_modules_model_modules_layers_modules_18_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_18_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_18_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_18_modules_final_layer_norm_parameters_bias_ = (None)
        linear_112 = torch._C._nn.linear(
            hidden_states_167,
            l_self_modules_model_modules_layers_modules_18_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_layers_modules_18_modules_fc1_parameters_bias_,
        )
        hidden_states_167 = l_self_modules_model_modules_layers_modules_18_modules_fc1_parameters_weight_ = (
            l_self_modules_model_modules_layers_modules_18_modules_fc1_parameters_bias_
        ) = None
        hidden_states_168 = torch._C._nn.gelu(linear_112)
        linear_112 = None
        hidden_states_169 = torch.nn.functional.dropout(
            hidden_states_168, p=0.0, training=False
        )
        hidden_states_168 = None
        hidden_states_170 = torch._C._nn.linear(
            hidden_states_169,
            l_self_modules_model_modules_layers_modules_18_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_layers_modules_18_modules_fc2_parameters_bias_,
        )
        hidden_states_169 = l_self_modules_model_modules_layers_modules_18_modules_fc2_parameters_weight_ = (
            l_self_modules_model_modules_layers_modules_18_modules_fc2_parameters_bias_
        ) = None
        hidden_states_171 = torch.nn.functional.dropout(
            hidden_states_170, p=0.1, training=False
        )
        hidden_states_170 = None
        hidden_states_172 = hidden_states_166 + hidden_states_171
        hidden_states_166 = hidden_states_171 = None
        hidden_states_173 = torch.nn.functional.layer_norm(
            hidden_states_172,
            (2048,),
            l_self_modules_model_modules_layers_modules_19_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_19_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_19_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_114 = torch._C._nn.linear(
            hidden_states_173,
            l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_57 = linear_114 * 0.125
        linear_114 = None
        key_states_57 = torch._C._nn.linear(
            hidden_states_173,
            l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_57 = torch._C._nn.linear(
            hidden_states_173,
            l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_173 = l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_118 = key_states_57.view(1, 19, -1, 64)
        key_states_57 = None
        key_states_58 = view_118.transpose(1, 2)
        view_118 = None
        view_119 = value_states_57.view(1, 19, -1, 64)
        value_states_57 = None
        value_states_58 = view_119.transpose(1, 2)
        view_119 = None
        view_120 = query_states_57.view(1, 19, 32, 64)
        query_states_57 = None
        query_states_58 = view_120.transpose(1, 2)
        view_120 = None
        query_states_59 = query_states_58.reshape(32, -1, 64)
        query_states_58 = None
        key_states_59 = key_states_58.reshape(32, -1, 64)
        key_states_58 = None
        value_states_59 = value_states_58.reshape(32, -1, 64)
        value_states_58 = None
        transpose_98 = key_states_59.transpose(1, 2)
        key_states_59 = None
        attn_weights_95 = torch.bmm(query_states_59, transpose_98)
        query_states_59 = transpose_98 = None
        view_121 = attn_weights_95.view(1, 32, 19, 19)
        attn_weights_95 = None
        attn_weights_96 = view_121 + expanded_attn_mask_1
        view_121 = None
        tensor_20 = torch.tensor(-65504.0, device=device(type="cpu"))
        attn_weights_97 = torch.max(attn_weights_96, tensor_20)
        attn_weights_96 = tensor_20 = None
        attn_weights_98 = attn_weights_97.view(32, 19, 19)
        attn_weights_97 = None
        softmax_19 = torch.nn.functional.softmax(
            attn_weights_98, dim=-1, dtype=torch.float32
        )
        attn_weights_98 = None
        attn_weights_99 = softmax_19.to(torch.float16)
        softmax_19 = None
        attn_probs_19 = torch.nn.functional.dropout(
            attn_weights_99, p=0.1, training=False
        )
        attn_weights_99 = None
        attn_output_95 = torch.bmm(attn_probs_19, value_states_59)
        attn_probs_19 = value_states_59 = None
        attn_output_96 = attn_output_95.view(1, 32, 19, 64)
        attn_output_95 = None
        attn_output_97 = attn_output_96.transpose(1, 2)
        attn_output_96 = None
        attn_output_98 = attn_output_97.reshape(1, 19, 2048)
        attn_output_97 = None
        attn_output_99 = torch._C._nn.linear(
            attn_output_98,
            l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_98 = l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_174 = torch.nn.functional.dropout(
            attn_output_99, p=0.1, training=False
        )
        attn_output_99 = None
        hidden_states_175 = hidden_states_172 + hidden_states_174
        hidden_states_172 = hidden_states_174 = None
        hidden_states_176 = torch.nn.functional.layer_norm(
            hidden_states_175,
            (2048,),
            l_self_modules_model_modules_layers_modules_19_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_19_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_19_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_19_modules_final_layer_norm_parameters_bias_ = (None)
        linear_118 = torch._C._nn.linear(
            hidden_states_176,
            l_self_modules_model_modules_layers_modules_19_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_layers_modules_19_modules_fc1_parameters_bias_,
        )
        hidden_states_176 = l_self_modules_model_modules_layers_modules_19_modules_fc1_parameters_weight_ = (
            l_self_modules_model_modules_layers_modules_19_modules_fc1_parameters_bias_
        ) = None
        hidden_states_177 = torch._C._nn.gelu(linear_118)
        linear_118 = None
        hidden_states_178 = torch.nn.functional.dropout(
            hidden_states_177, p=0.0, training=False
        )
        hidden_states_177 = None
        hidden_states_179 = torch._C._nn.linear(
            hidden_states_178,
            l_self_modules_model_modules_layers_modules_19_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_layers_modules_19_modules_fc2_parameters_bias_,
        )
        hidden_states_178 = l_self_modules_model_modules_layers_modules_19_modules_fc2_parameters_weight_ = (
            l_self_modules_model_modules_layers_modules_19_modules_fc2_parameters_bias_
        ) = None
        hidden_states_180 = torch.nn.functional.dropout(
            hidden_states_179, p=0.1, training=False
        )
        hidden_states_179 = None
        hidden_states_181 = hidden_states_175 + hidden_states_180
        hidden_states_175 = hidden_states_180 = None
        hidden_states_182 = torch.nn.functional.layer_norm(
            hidden_states_181,
            (2048,),
            l_self_modules_model_modules_layers_modules_20_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_20_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_20_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_120 = torch._C._nn.linear(
            hidden_states_182,
            l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_60 = linear_120 * 0.125
        linear_120 = None
        key_states_60 = torch._C._nn.linear(
            hidden_states_182,
            l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_60 = torch._C._nn.linear(
            hidden_states_182,
            l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_182 = l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_124 = key_states_60.view(1, 19, -1, 64)
        key_states_60 = None
        key_states_61 = view_124.transpose(1, 2)
        view_124 = None
        view_125 = value_states_60.view(1, 19, -1, 64)
        value_states_60 = None
        value_states_61 = view_125.transpose(1, 2)
        view_125 = None
        view_126 = query_states_60.view(1, 19, 32, 64)
        query_states_60 = None
        query_states_61 = view_126.transpose(1, 2)
        view_126 = None
        query_states_62 = query_states_61.reshape(32, -1, 64)
        query_states_61 = None
        key_states_62 = key_states_61.reshape(32, -1, 64)
        key_states_61 = None
        value_states_62 = value_states_61.reshape(32, -1, 64)
        value_states_61 = None
        transpose_103 = key_states_62.transpose(1, 2)
        key_states_62 = None
        attn_weights_100 = torch.bmm(query_states_62, transpose_103)
        query_states_62 = transpose_103 = None
        view_127 = attn_weights_100.view(1, 32, 19, 19)
        attn_weights_100 = None
        attn_weights_101 = view_127 + expanded_attn_mask_1
        view_127 = None
        tensor_21 = torch.tensor(-65504.0, device=device(type="cpu"))
        attn_weights_102 = torch.max(attn_weights_101, tensor_21)
        attn_weights_101 = tensor_21 = None
        attn_weights_103 = attn_weights_102.view(32, 19, 19)
        attn_weights_102 = None
        softmax_20 = torch.nn.functional.softmax(
            attn_weights_103, dim=-1, dtype=torch.float32
        )
        attn_weights_103 = None
        attn_weights_104 = softmax_20.to(torch.float16)
        softmax_20 = None
        attn_probs_20 = torch.nn.functional.dropout(
            attn_weights_104, p=0.1, training=False
        )
        attn_weights_104 = None
        attn_output_100 = torch.bmm(attn_probs_20, value_states_62)
        attn_probs_20 = value_states_62 = None
        attn_output_101 = attn_output_100.view(1, 32, 19, 64)
        attn_output_100 = None
        attn_output_102 = attn_output_101.transpose(1, 2)
        attn_output_101 = None
        attn_output_103 = attn_output_102.reshape(1, 19, 2048)
        attn_output_102 = None
        attn_output_104 = torch._C._nn.linear(
            attn_output_103,
            l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_103 = l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_183 = torch.nn.functional.dropout(
            attn_output_104, p=0.1, training=False
        )
        attn_output_104 = None
        hidden_states_184 = hidden_states_181 + hidden_states_183
        hidden_states_181 = hidden_states_183 = None
        hidden_states_185 = torch.nn.functional.layer_norm(
            hidden_states_184,
            (2048,),
            l_self_modules_model_modules_layers_modules_20_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_20_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_20_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_20_modules_final_layer_norm_parameters_bias_ = (None)
        linear_124 = torch._C._nn.linear(
            hidden_states_185,
            l_self_modules_model_modules_layers_modules_20_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_layers_modules_20_modules_fc1_parameters_bias_,
        )
        hidden_states_185 = l_self_modules_model_modules_layers_modules_20_modules_fc1_parameters_weight_ = (
            l_self_modules_model_modules_layers_modules_20_modules_fc1_parameters_bias_
        ) = None
        hidden_states_186 = torch._C._nn.gelu(linear_124)
        linear_124 = None
        hidden_states_187 = torch.nn.functional.dropout(
            hidden_states_186, p=0.0, training=False
        )
        hidden_states_186 = None
        hidden_states_188 = torch._C._nn.linear(
            hidden_states_187,
            l_self_modules_model_modules_layers_modules_20_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_layers_modules_20_modules_fc2_parameters_bias_,
        )
        hidden_states_187 = l_self_modules_model_modules_layers_modules_20_modules_fc2_parameters_weight_ = (
            l_self_modules_model_modules_layers_modules_20_modules_fc2_parameters_bias_
        ) = None
        hidden_states_189 = torch.nn.functional.dropout(
            hidden_states_188, p=0.1, training=False
        )
        hidden_states_188 = None
        hidden_states_190 = hidden_states_184 + hidden_states_189
        hidden_states_184 = hidden_states_189 = None
        hidden_states_191 = torch.nn.functional.layer_norm(
            hidden_states_190,
            (2048,),
            l_self_modules_model_modules_layers_modules_21_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_21_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_21_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_126 = torch._C._nn.linear(
            hidden_states_191,
            l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_63 = linear_126 * 0.125
        linear_126 = None
        key_states_63 = torch._C._nn.linear(
            hidden_states_191,
            l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_63 = torch._C._nn.linear(
            hidden_states_191,
            l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_191 = l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_130 = key_states_63.view(1, 19, -1, 64)
        key_states_63 = None
        key_states_64 = view_130.transpose(1, 2)
        view_130 = None
        view_131 = value_states_63.view(1, 19, -1, 64)
        value_states_63 = None
        value_states_64 = view_131.transpose(1, 2)
        view_131 = None
        view_132 = query_states_63.view(1, 19, 32, 64)
        query_states_63 = None
        query_states_64 = view_132.transpose(1, 2)
        view_132 = None
        query_states_65 = query_states_64.reshape(32, -1, 64)
        query_states_64 = None
        key_states_65 = key_states_64.reshape(32, -1, 64)
        key_states_64 = None
        value_states_65 = value_states_64.reshape(32, -1, 64)
        value_states_64 = None
        transpose_108 = key_states_65.transpose(1, 2)
        key_states_65 = None
        attn_weights_105 = torch.bmm(query_states_65, transpose_108)
        query_states_65 = transpose_108 = None
        view_133 = attn_weights_105.view(1, 32, 19, 19)
        attn_weights_105 = None
        attn_weights_106 = view_133 + expanded_attn_mask_1
        view_133 = None
        tensor_22 = torch.tensor(-65504.0, device=device(type="cpu"))
        attn_weights_107 = torch.max(attn_weights_106, tensor_22)
        attn_weights_106 = tensor_22 = None
        attn_weights_108 = attn_weights_107.view(32, 19, 19)
        attn_weights_107 = None
        softmax_21 = torch.nn.functional.softmax(
            attn_weights_108, dim=-1, dtype=torch.float32
        )
        attn_weights_108 = None
        attn_weights_109 = softmax_21.to(torch.float16)
        softmax_21 = None
        attn_probs_21 = torch.nn.functional.dropout(
            attn_weights_109, p=0.1, training=False
        )
        attn_weights_109 = None
        attn_output_105 = torch.bmm(attn_probs_21, value_states_65)
        attn_probs_21 = value_states_65 = None
        attn_output_106 = attn_output_105.view(1, 32, 19, 64)
        attn_output_105 = None
        attn_output_107 = attn_output_106.transpose(1, 2)
        attn_output_106 = None
        attn_output_108 = attn_output_107.reshape(1, 19, 2048)
        attn_output_107 = None
        attn_output_109 = torch._C._nn.linear(
            attn_output_108,
            l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_108 = l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_192 = torch.nn.functional.dropout(
            attn_output_109, p=0.1, training=False
        )
        attn_output_109 = None
        hidden_states_193 = hidden_states_190 + hidden_states_192
        hidden_states_190 = hidden_states_192 = None
        hidden_states_194 = torch.nn.functional.layer_norm(
            hidden_states_193,
            (2048,),
            l_self_modules_model_modules_layers_modules_21_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_21_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_21_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_21_modules_final_layer_norm_parameters_bias_ = (None)
        linear_130 = torch._C._nn.linear(
            hidden_states_194,
            l_self_modules_model_modules_layers_modules_21_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_layers_modules_21_modules_fc1_parameters_bias_,
        )
        hidden_states_194 = l_self_modules_model_modules_layers_modules_21_modules_fc1_parameters_weight_ = (
            l_self_modules_model_modules_layers_modules_21_modules_fc1_parameters_bias_
        ) = None
        hidden_states_195 = torch._C._nn.gelu(linear_130)
        linear_130 = None
        hidden_states_196 = torch.nn.functional.dropout(
            hidden_states_195, p=0.0, training=False
        )
        hidden_states_195 = None
        hidden_states_197 = torch._C._nn.linear(
            hidden_states_196,
            l_self_modules_model_modules_layers_modules_21_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_layers_modules_21_modules_fc2_parameters_bias_,
        )
        hidden_states_196 = l_self_modules_model_modules_layers_modules_21_modules_fc2_parameters_weight_ = (
            l_self_modules_model_modules_layers_modules_21_modules_fc2_parameters_bias_
        ) = None
        hidden_states_198 = torch.nn.functional.dropout(
            hidden_states_197, p=0.1, training=False
        )
        hidden_states_197 = None
        hidden_states_199 = hidden_states_193 + hidden_states_198
        hidden_states_193 = hidden_states_198 = None
        hidden_states_200 = torch.nn.functional.layer_norm(
            hidden_states_199,
            (2048,),
            l_self_modules_model_modules_layers_modules_22_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_22_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_22_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_132 = torch._C._nn.linear(
            hidden_states_200,
            l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_66 = linear_132 * 0.125
        linear_132 = None
        key_states_66 = torch._C._nn.linear(
            hidden_states_200,
            l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_66 = torch._C._nn.linear(
            hidden_states_200,
            l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_200 = l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_136 = key_states_66.view(1, 19, -1, 64)
        key_states_66 = None
        key_states_67 = view_136.transpose(1, 2)
        view_136 = None
        view_137 = value_states_66.view(1, 19, -1, 64)
        value_states_66 = None
        value_states_67 = view_137.transpose(1, 2)
        view_137 = None
        view_138 = query_states_66.view(1, 19, 32, 64)
        query_states_66 = None
        query_states_67 = view_138.transpose(1, 2)
        view_138 = None
        query_states_68 = query_states_67.reshape(32, -1, 64)
        query_states_67 = None
        key_states_68 = key_states_67.reshape(32, -1, 64)
        key_states_67 = None
        value_states_68 = value_states_67.reshape(32, -1, 64)
        value_states_67 = None
        transpose_113 = key_states_68.transpose(1, 2)
        key_states_68 = None
        attn_weights_110 = torch.bmm(query_states_68, transpose_113)
        query_states_68 = transpose_113 = None
        view_139 = attn_weights_110.view(1, 32, 19, 19)
        attn_weights_110 = None
        attn_weights_111 = view_139 + expanded_attn_mask_1
        view_139 = None
        tensor_23 = torch.tensor(-65504.0, device=device(type="cpu"))
        attn_weights_112 = torch.max(attn_weights_111, tensor_23)
        attn_weights_111 = tensor_23 = None
        attn_weights_113 = attn_weights_112.view(32, 19, 19)
        attn_weights_112 = None
        softmax_22 = torch.nn.functional.softmax(
            attn_weights_113, dim=-1, dtype=torch.float32
        )
        attn_weights_113 = None
        attn_weights_114 = softmax_22.to(torch.float16)
        softmax_22 = None
        attn_probs_22 = torch.nn.functional.dropout(
            attn_weights_114, p=0.1, training=False
        )
        attn_weights_114 = None
        attn_output_110 = torch.bmm(attn_probs_22, value_states_68)
        attn_probs_22 = value_states_68 = None
        attn_output_111 = attn_output_110.view(1, 32, 19, 64)
        attn_output_110 = None
        attn_output_112 = attn_output_111.transpose(1, 2)
        attn_output_111 = None
        attn_output_113 = attn_output_112.reshape(1, 19, 2048)
        attn_output_112 = None
        attn_output_114 = torch._C._nn.linear(
            attn_output_113,
            l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_113 = l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_201 = torch.nn.functional.dropout(
            attn_output_114, p=0.1, training=False
        )
        attn_output_114 = None
        hidden_states_202 = hidden_states_199 + hidden_states_201
        hidden_states_199 = hidden_states_201 = None
        hidden_states_203 = torch.nn.functional.layer_norm(
            hidden_states_202,
            (2048,),
            l_self_modules_model_modules_layers_modules_22_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_22_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_22_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_22_modules_final_layer_norm_parameters_bias_ = (None)
        linear_136 = torch._C._nn.linear(
            hidden_states_203,
            l_self_modules_model_modules_layers_modules_22_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_layers_modules_22_modules_fc1_parameters_bias_,
        )
        hidden_states_203 = l_self_modules_model_modules_layers_modules_22_modules_fc1_parameters_weight_ = (
            l_self_modules_model_modules_layers_modules_22_modules_fc1_parameters_bias_
        ) = None
        hidden_states_204 = torch._C._nn.gelu(linear_136)
        linear_136 = None
        hidden_states_205 = torch.nn.functional.dropout(
            hidden_states_204, p=0.0, training=False
        )
        hidden_states_204 = None
        hidden_states_206 = torch._C._nn.linear(
            hidden_states_205,
            l_self_modules_model_modules_layers_modules_22_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_layers_modules_22_modules_fc2_parameters_bias_,
        )
        hidden_states_205 = l_self_modules_model_modules_layers_modules_22_modules_fc2_parameters_weight_ = (
            l_self_modules_model_modules_layers_modules_22_modules_fc2_parameters_bias_
        ) = None
        hidden_states_207 = torch.nn.functional.dropout(
            hidden_states_206, p=0.1, training=False
        )
        hidden_states_206 = None
        hidden_states_208 = hidden_states_202 + hidden_states_207
        hidden_states_202 = hidden_states_207 = None
        hidden_states_209 = torch.nn.functional.layer_norm(
            hidden_states_208,
            (2048,),
            l_self_modules_model_modules_layers_modules_23_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_23_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_23_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_138 = torch._C._nn.linear(
            hidden_states_209,
            l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_69 = linear_138 * 0.125
        linear_138 = None
        key_states_69 = torch._C._nn.linear(
            hidden_states_209,
            l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_69 = torch._C._nn.linear(
            hidden_states_209,
            l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_209 = l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_142 = key_states_69.view(1, 19, -1, 64)
        key_states_69 = None
        key_states_70 = view_142.transpose(1, 2)
        view_142 = None
        view_143 = value_states_69.view(1, 19, -1, 64)
        value_states_69 = None
        value_states_70 = view_143.transpose(1, 2)
        view_143 = None
        view_144 = query_states_69.view(1, 19, 32, 64)
        query_states_69 = None
        query_states_70 = view_144.transpose(1, 2)
        view_144 = None
        query_states_71 = query_states_70.reshape(32, -1, 64)
        query_states_70 = None
        key_states_71 = key_states_70.reshape(32, -1, 64)
        key_states_70 = None
        value_states_71 = value_states_70.reshape(32, -1, 64)
        value_states_70 = None
        transpose_118 = key_states_71.transpose(1, 2)
        key_states_71 = None
        attn_weights_115 = torch.bmm(query_states_71, transpose_118)
        query_states_71 = transpose_118 = None
        view_145 = attn_weights_115.view(1, 32, 19, 19)
        attn_weights_115 = None
        attn_weights_116 = view_145 + expanded_attn_mask_1
        view_145 = expanded_attn_mask_1 = None
        tensor_24 = torch.tensor(-65504.0, device=device(type="cpu"))
        attn_weights_117 = torch.max(attn_weights_116, tensor_24)
        attn_weights_116 = tensor_24 = None
        attn_weights_118 = attn_weights_117.view(32, 19, 19)
        attn_weights_117 = None
        softmax_23 = torch.nn.functional.softmax(
            attn_weights_118, dim=-1, dtype=torch.float32
        )
        attn_weights_118 = None
        attn_weights_119 = softmax_23.to(torch.float16)
        softmax_23 = None
        attn_probs_23 = torch.nn.functional.dropout(
            attn_weights_119, p=0.1, training=False
        )
        attn_weights_119 = None
        attn_output_115 = torch.bmm(attn_probs_23, value_states_71)
        attn_probs_23 = value_states_71 = None
        attn_output_116 = attn_output_115.view(1, 32, 19, 64)
        attn_output_115 = None
        attn_output_117 = attn_output_116.transpose(1, 2)
        attn_output_116 = None
        attn_output_118 = attn_output_117.reshape(1, 19, 2048)
        attn_output_117 = None
        attn_output_119 = torch._C._nn.linear(
            attn_output_118,
            l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_118 = l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_210 = torch.nn.functional.dropout(
            attn_output_119, p=0.1, training=False
        )
        attn_output_119 = None
        hidden_states_211 = hidden_states_208 + hidden_states_210
        hidden_states_208 = hidden_states_210 = None
        hidden_states_212 = torch.nn.functional.layer_norm(
            hidden_states_211,
            (2048,),
            l_self_modules_model_modules_layers_modules_23_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_23_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_23_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_layers_modules_23_modules_final_layer_norm_parameters_bias_ = (None)
        linear_142 = torch._C._nn.linear(
            hidden_states_212,
            l_self_modules_model_modules_layers_modules_23_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_layers_modules_23_modules_fc1_parameters_bias_,
        )
        hidden_states_212 = l_self_modules_model_modules_layers_modules_23_modules_fc1_parameters_weight_ = (
            l_self_modules_model_modules_layers_modules_23_modules_fc1_parameters_bias_
        ) = None
        hidden_states_213 = torch._C._nn.gelu(linear_142)
        linear_142 = None
        hidden_states_214 = torch.nn.functional.dropout(
            hidden_states_213, p=0.0, training=False
        )
        hidden_states_213 = None
        hidden_states_215 = torch._C._nn.linear(
            hidden_states_214,
            l_self_modules_model_modules_layers_modules_23_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_layers_modules_23_modules_fc2_parameters_bias_,
        )
        hidden_states_214 = l_self_modules_model_modules_layers_modules_23_modules_fc2_parameters_weight_ = (
            l_self_modules_model_modules_layers_modules_23_modules_fc2_parameters_bias_
        ) = None
        hidden_states_216 = torch.nn.functional.dropout(
            hidden_states_215, p=0.1, training=False
        )
        hidden_states_215 = None
        hidden_states_217 = hidden_states_211 + hidden_states_216
        hidden_states_211 = hidden_states_216 = None
        hidden_states_218 = torch.nn.functional.layer_norm(
            hidden_states_217,
            (2048,),
            l_self_modules_model_modules_layer_norm_parameters_weight_,
            l_self_modules_model_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_217 = (
            l_self_modules_model_modules_layer_norm_parameters_weight_
        ) = l_self_modules_model_modules_layer_norm_parameters_bias_ = None
        logits = torch._C._nn.linear(
            hidden_states_218,
            l_self_modules_model_modules_embed_tokens_parameters_weight_,
            None,
        )
        hidden_states_218 = (
            l_self_modules_model_modules_embed_tokens_parameters_weight_
        ) = None
        return (logits,)
