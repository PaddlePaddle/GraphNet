import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_inputs_embeds_: torch.Tensor,
        L_attention_mask_: torch.Tensor,
        L_self_modules_embed_positions_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_inputs_embeds_ = L_inputs_embeds_
        l_attention_mask_ = L_attention_mask_
        l_self_modules_embed_positions_parameters_weight_ = (
            L_self_modules_embed_positions_parameters_weight_
        )
        l_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_
        )
        l_self_modules_layers_modules_0_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_0_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_0_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_0_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_0_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_0_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_0_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_0_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_
        )
        l_self_modules_layers_modules_1_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_1_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_1_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_1_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_1_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_1_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_1_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_1_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_
        )
        l_self_modules_layers_modules_2_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_2_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_2_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_2_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_2_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_2_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_2_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_2_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_
        )
        l_self_modules_layers_modules_3_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_3_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_3_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_3_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_3_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_3_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_3_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_3_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_
        )
        l_self_modules_layers_modules_4_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_4_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_4_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_4_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_4_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_4_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_4_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_4_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_
        )
        l_self_modules_layers_modules_5_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_5_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_5_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_5_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_5_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_5_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_5_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_5_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_6_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_layers_modules_6_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_layers_modules_6_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_layers_modules_6_modules_final_layer_norm_parameters_bias_
        )
        l_self_modules_layers_modules_6_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_6_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_6_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_6_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_6_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_6_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_6_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_6_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_7_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_layers_modules_7_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_layers_modules_7_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_layers_modules_7_modules_final_layer_norm_parameters_bias_
        )
        l_self_modules_layers_modules_7_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_7_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_7_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_7_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_7_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_7_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_7_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_7_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_8_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_layers_modules_8_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_layers_modules_8_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_layers_modules_8_modules_final_layer_norm_parameters_bias_
        )
        l_self_modules_layers_modules_8_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_8_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_8_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_8_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_8_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_8_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_8_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_8_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_9_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_layers_modules_9_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_layers_modules_9_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_layers_modules_9_modules_final_layer_norm_parameters_bias_
        )
        l_self_modules_layers_modules_9_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_9_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_9_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_9_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_9_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_9_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_9_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_9_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_10_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_layers_modules_10_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_layers_modules_10_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_layers_modules_10_modules_final_layer_norm_parameters_bias_
        )
        l_self_modules_layers_modules_10_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_10_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_10_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_10_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_10_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_10_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_10_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_10_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_11_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_layers_modules_11_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_layers_modules_11_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_layers_modules_11_modules_final_layer_norm_parameters_bias_
        )
        l_self_modules_layers_modules_11_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_11_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_11_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_11_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_11_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_11_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_11_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_11_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_12_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_layers_modules_12_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_layers_modules_12_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_layers_modules_12_modules_final_layer_norm_parameters_bias_
        )
        l_self_modules_layers_modules_12_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_12_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_12_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_12_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_12_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_12_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_12_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_12_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_13_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_layers_modules_13_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_layers_modules_13_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_layers_modules_13_modules_final_layer_norm_parameters_bias_
        )
        l_self_modules_layers_modules_13_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_13_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_13_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_13_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_13_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_13_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_13_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_13_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_14_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_layers_modules_14_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_layers_modules_14_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_layers_modules_14_modules_final_layer_norm_parameters_bias_
        )
        l_self_modules_layers_modules_14_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_14_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_14_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_14_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_14_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_14_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_14_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_14_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_15_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_layers_modules_15_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_layers_modules_15_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_layers_modules_15_modules_final_layer_norm_parameters_bias_
        )
        l_self_modules_layers_modules_15_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_15_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_15_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_15_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_15_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_15_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_15_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_15_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_16_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_16_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_16_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_16_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_16_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_layers_modules_16_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_layers_modules_16_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_layers_modules_16_modules_final_layer_norm_parameters_bias_
        )
        l_self_modules_layers_modules_16_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_16_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_16_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_16_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_16_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_16_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_16_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_16_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_17_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_17_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_17_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_17_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_17_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_layers_modules_17_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_layers_modules_17_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_layers_modules_17_modules_final_layer_norm_parameters_bias_
        )
        l_self_modules_layers_modules_17_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_17_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_17_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_17_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_17_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_17_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_17_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_17_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_18_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_18_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_18_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_18_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_18_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_layers_modules_18_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_layers_modules_18_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_layers_modules_18_modules_final_layer_norm_parameters_bias_
        )
        l_self_modules_layers_modules_18_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_18_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_18_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_18_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_18_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_18_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_18_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_18_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_19_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_19_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_19_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_19_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_19_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_layers_modules_19_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_layers_modules_19_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_layers_modules_19_modules_final_layer_norm_parameters_bias_
        )
        l_self_modules_layers_modules_19_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_19_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_19_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_19_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_19_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_19_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_19_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_19_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_20_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_20_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_20_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_20_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_20_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_layers_modules_20_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_layers_modules_20_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_layers_modules_20_modules_final_layer_norm_parameters_bias_
        )
        l_self_modules_layers_modules_20_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_20_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_20_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_20_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_20_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_20_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_20_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_20_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_21_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_21_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_21_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_21_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_21_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_layers_modules_21_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_layers_modules_21_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_layers_modules_21_modules_final_layer_norm_parameters_bias_
        )
        l_self_modules_layers_modules_21_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_21_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_21_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_21_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_21_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_21_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_21_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_21_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_22_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_22_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_22_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_22_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_22_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_layers_modules_22_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_layers_modules_22_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_layers_modules_22_modules_final_layer_norm_parameters_bias_
        )
        l_self_modules_layers_modules_22_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_22_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_22_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_22_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_22_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_22_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_22_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_22_modules_fc2_parameters_bias_
        )
        l_self_modules_layers_modules_23_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_23_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_23_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_23_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_23_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_layers_modules_23_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_layers_modules_23_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_layers_modules_23_modules_final_layer_norm_parameters_bias_
        )
        l_self_modules_layers_modules_23_modules_fc1_parameters_weight_ = (
            L_self_modules_layers_modules_23_modules_fc1_parameters_weight_
        )
        l_self_modules_layers_modules_23_modules_fc1_parameters_bias_ = (
            L_self_modules_layers_modules_23_modules_fc1_parameters_bias_
        )
        l_self_modules_layers_modules_23_modules_fc2_parameters_weight_ = (
            L_self_modules_layers_modules_23_modules_fc2_parameters_weight_
        )
        l_self_modules_layers_modules_23_modules_fc2_parameters_bias_ = (
            L_self_modules_layers_modules_23_modules_fc2_parameters_bias_
        )
        l_self_modules_layer_norm_parameters_weight_ = (
            L_self_modules_layer_norm_parameters_weight_
        )
        l_self_modules_layer_norm_parameters_bias_ = (
            L_self_modules_layer_norm_parameters_bias_
        )
        cache_position = torch.arange(0, 10, device=device(type="cuda", index=0))
        causal_mask = torch.full(
            (10, 10),
            fill_value=-3.4028234663852886e38,
            dtype=torch.float32,
            device=device(type="cuda", index=0),
        )
        causal_mask_1 = torch.triu(causal_mask, diagonal=1)
        causal_mask = None
        arange_1 = torch.arange(10, device=device(type="cuda", index=0))
        reshape = cache_position.reshape(-1, 1)
        cache_position = None
        gt = arange_1 > reshape
        arange_1 = reshape = None
        causal_mask_1 *= gt
        causal_mask_2 = causal_mask_1
        causal_mask_1 = gt = None
        getitem = causal_mask_2[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        causal_mask_2 = None
        causal_mask_3 = getitem.expand(1, 1, -1, -1)
        getitem = None
        causal_mask_4 = causal_mask_3.clone()
        causal_mask_3 = None
        getitem_1 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        getitem_2 = l_attention_mask_[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        to = getitem_2.to(device(type="cuda", index=0))
        getitem_2 = None
        padding_mask = getitem_1 + to
        getitem_1 = to = None
        padding_mask_1 = padding_mask.__eq__(0)
        padding_mask = None
        getitem_3 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        masked_fill = getitem_3.masked_fill(padding_mask_1, -3.4028234663852886e38)
        getitem_3 = padding_mask_1 = None
        causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ] = masked_fill
        setitem = causal_mask_4
        masked_fill = setitem = None
        eq_1 = causal_mask_4.__eq__(-3.4028234663852886e38)
        all_1 = torch.all(eq_1, dim=-1, keepdim=True)
        eq_1 = None
        invert = ~all_1
        all_1 = None
        causal_mask_5 = causal_mask_4.mul(invert)
        causal_mask_4 = invert = None
        position_ids = torch.cumsum(l_attention_mask_, dim=1)
        mul_1 = position_ids * l_attention_mask_
        position_ids = l_attention_mask_ = None
        sub = mul_1 - 1
        mul_1 = None
        position_ids_1 = sub.long()
        sub = None
        position_ids_2 = position_ids_1[(slice(None, None, None), slice(0, None, None))]
        position_ids_1 = None
        add_1 = position_ids_2 + 2
        position_ids_2 = None
        positions = torch.nn.functional.embedding(
            add_1,
            l_self_modules_embed_positions_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        add_1 = l_self_modules_embed_positions_parameters_weight_ = None
        hidden_states = l_inputs_embeds_ + positions
        l_inputs_embeds_ = positions = None
        hidden_states_1 = torch.nn.functional.dropout(
            hidden_states, p=0.1, training=False
        )
        hidden_states = None
        hidden_states_2 = torch.nn.functional.layer_norm(
            hidden_states_1,
            (1024,),
            l_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view = linear.view(1, 10, -1, 64)
        linear = None
        query_states = view.transpose(1, 2)
        view = None
        key_states = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_2 = l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_1 = key_states.view(1, 10, -1, 64)
        key_states = None
        key_states_1 = view_1.transpose(1, 2)
        view_1 = None
        view_2 = value_states.view(1, 10, -1, 64)
        value_states = None
        value_states_1 = view_2.transpose(1, 2)
        view_2 = None
        attention_mask = causal_mask_5[
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
        value = value_states_1.contiguous()
        attn_output = torch._C._nn.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query = key = value = attention_mask = None
        transpose_3 = attn_output.transpose(1, 2)
        attn_output = None
        attn_output_1 = transpose_3.contiguous()
        transpose_3 = None
        reshape_1 = attn_output_1.reshape(1, 10, -1)
        attn_output_1 = None
        attn_output_2 = reshape_1.contiguous()
        reshape_1 = None
        attn_output_3 = torch._C._nn.linear(
            attn_output_2,
            l_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_2 = l_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_3 = torch.nn.functional.dropout(
            attn_output_3, p=0.1, training=False
        )
        attn_output_3 = None
        hidden_states_4 = hidden_states_1 + hidden_states_3
        hidden_states_1 = hidden_states_3 = None
        hidden_states_5 = torch.nn.functional.layer_norm(
            hidden_states_4,
            (1024,),
            l_self_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_ = (
            l_self_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_
        ) = None
        hidden_states_6 = torch._C._nn.linear(
            hidden_states_5,
            l_self_modules_layers_modules_0_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_0_modules_fc1_parameters_bias_,
        )
        hidden_states_5 = (
            l_self_modules_layers_modules_0_modules_fc1_parameters_weight_
        ) = l_self_modules_layers_modules_0_modules_fc1_parameters_bias_ = None
        hidden_states_7 = torch._C._nn.gelu(hidden_states_6)
        hidden_states_6 = None
        hidden_states_8 = torch.nn.functional.dropout(
            hidden_states_7, p=0.0, training=False
        )
        hidden_states_7 = None
        hidden_states_9 = torch._C._nn.linear(
            hidden_states_8,
            l_self_modules_layers_modules_0_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_0_modules_fc2_parameters_bias_,
        )
        hidden_states_8 = (
            l_self_modules_layers_modules_0_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_0_modules_fc2_parameters_bias_ = None
        hidden_states_10 = torch.nn.functional.dropout(
            hidden_states_9, p=0.1, training=False
        )
        hidden_states_9 = None
        hidden_states_11 = hidden_states_4 + hidden_states_10
        hidden_states_4 = hidden_states_10 = None
        hidden_states_12 = torch.nn.functional.layer_norm(
            hidden_states_11,
            (1024,),
            l_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_6 = torch._C._nn.linear(
            hidden_states_12,
            l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_3 = linear_6.view(1, 10, -1, 64)
        linear_6 = None
        query_states_1 = view_3.transpose(1, 2)
        view_3 = None
        key_states_2 = torch._C._nn.linear(
            hidden_states_12,
            l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_2 = torch._C._nn.linear(
            hidden_states_12,
            l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_12 = l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_4 = key_states_2.view(1, 10, -1, 64)
        key_states_2 = None
        key_states_3 = view_4.transpose(1, 2)
        view_4 = None
        view_5 = value_states_2.view(1, 10, -1, 64)
        value_states_2 = None
        value_states_3 = view_5.transpose(1, 2)
        view_5 = None
        attention_mask_1 = causal_mask_5[
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
        value_1 = value_states_3.contiguous()
        attn_output_4 = torch._C._nn.scaled_dot_product_attention(
            query_1,
            key_1,
            value_1,
            attn_mask=attention_mask_1,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_1 = key_1 = value_1 = attention_mask_1 = None
        transpose_7 = attn_output_4.transpose(1, 2)
        attn_output_4 = None
        attn_output_5 = transpose_7.contiguous()
        transpose_7 = None
        reshape_2 = attn_output_5.reshape(1, 10, -1)
        attn_output_5 = None
        attn_output_6 = reshape_2.contiguous()
        reshape_2 = None
        attn_output_7 = torch._C._nn.linear(
            attn_output_6,
            l_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_6 = l_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_13 = torch.nn.functional.dropout(
            attn_output_7, p=0.1, training=False
        )
        attn_output_7 = None
        hidden_states_14 = hidden_states_11 + hidden_states_13
        hidden_states_11 = hidden_states_13 = None
        hidden_states_15 = torch.nn.functional.layer_norm(
            hidden_states_14,
            (1024,),
            l_self_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_ = (
            l_self_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_
        ) = None
        hidden_states_16 = torch._C._nn.linear(
            hidden_states_15,
            l_self_modules_layers_modules_1_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_1_modules_fc1_parameters_bias_,
        )
        hidden_states_15 = (
            l_self_modules_layers_modules_1_modules_fc1_parameters_weight_
        ) = l_self_modules_layers_modules_1_modules_fc1_parameters_bias_ = None
        hidden_states_17 = torch._C._nn.gelu(hidden_states_16)
        hidden_states_16 = None
        hidden_states_18 = torch.nn.functional.dropout(
            hidden_states_17, p=0.0, training=False
        )
        hidden_states_17 = None
        hidden_states_19 = torch._C._nn.linear(
            hidden_states_18,
            l_self_modules_layers_modules_1_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_1_modules_fc2_parameters_bias_,
        )
        hidden_states_18 = (
            l_self_modules_layers_modules_1_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_1_modules_fc2_parameters_bias_ = None
        hidden_states_20 = torch.nn.functional.dropout(
            hidden_states_19, p=0.1, training=False
        )
        hidden_states_19 = None
        hidden_states_21 = hidden_states_14 + hidden_states_20
        hidden_states_14 = hidden_states_20 = None
        hidden_states_22 = torch.nn.functional.layer_norm(
            hidden_states_21,
            (1024,),
            l_self_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_12 = torch._C._nn.linear(
            hidden_states_22,
            l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_6 = linear_12.view(1, 10, -1, 64)
        linear_12 = None
        query_states_2 = view_6.transpose(1, 2)
        view_6 = None
        key_states_4 = torch._C._nn.linear(
            hidden_states_22,
            l_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_4 = torch._C._nn.linear(
            hidden_states_22,
            l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_22 = l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_7 = key_states_4.view(1, 10, -1, 64)
        key_states_4 = None
        key_states_5 = view_7.transpose(1, 2)
        view_7 = None
        view_8 = value_states_4.view(1, 10, -1, 64)
        value_states_4 = None
        value_states_5 = view_8.transpose(1, 2)
        view_8 = None
        attention_mask_2 = causal_mask_5[
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
        value_2 = value_states_5.contiguous()
        attn_output_8 = torch._C._nn.scaled_dot_product_attention(
            query_2,
            key_2,
            value_2,
            attn_mask=attention_mask_2,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_2 = key_2 = value_2 = attention_mask_2 = None
        transpose_11 = attn_output_8.transpose(1, 2)
        attn_output_8 = None
        attn_output_9 = transpose_11.contiguous()
        transpose_11 = None
        reshape_3 = attn_output_9.reshape(1, 10, -1)
        attn_output_9 = None
        attn_output_10 = reshape_3.contiguous()
        reshape_3 = None
        attn_output_11 = torch._C._nn.linear(
            attn_output_10,
            l_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_10 = l_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_23 = torch.nn.functional.dropout(
            attn_output_11, p=0.1, training=False
        )
        attn_output_11 = None
        hidden_states_24 = hidden_states_21 + hidden_states_23
        hidden_states_21 = hidden_states_23 = None
        hidden_states_25 = torch.nn.functional.layer_norm(
            hidden_states_24,
            (1024,),
            l_self_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_ = (
            l_self_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_
        ) = None
        hidden_states_26 = torch._C._nn.linear(
            hidden_states_25,
            l_self_modules_layers_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_fc1_parameters_bias_,
        )
        hidden_states_25 = (
            l_self_modules_layers_modules_2_modules_fc1_parameters_weight_
        ) = l_self_modules_layers_modules_2_modules_fc1_parameters_bias_ = None
        hidden_states_27 = torch._C._nn.gelu(hidden_states_26)
        hidden_states_26 = None
        hidden_states_28 = torch.nn.functional.dropout(
            hidden_states_27, p=0.0, training=False
        )
        hidden_states_27 = None
        hidden_states_29 = torch._C._nn.linear(
            hidden_states_28,
            l_self_modules_layers_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_fc2_parameters_bias_,
        )
        hidden_states_28 = (
            l_self_modules_layers_modules_2_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_2_modules_fc2_parameters_bias_ = None
        hidden_states_30 = torch.nn.functional.dropout(
            hidden_states_29, p=0.1, training=False
        )
        hidden_states_29 = None
        hidden_states_31 = hidden_states_24 + hidden_states_30
        hidden_states_24 = hidden_states_30 = None
        hidden_states_32 = torch.nn.functional.layer_norm(
            hidden_states_31,
            (1024,),
            l_self_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_18 = torch._C._nn.linear(
            hidden_states_32,
            l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_9 = linear_18.view(1, 10, -1, 64)
        linear_18 = None
        query_states_3 = view_9.transpose(1, 2)
        view_9 = None
        key_states_6 = torch._C._nn.linear(
            hidden_states_32,
            l_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_6 = torch._C._nn.linear(
            hidden_states_32,
            l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_32 = l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_10 = key_states_6.view(1, 10, -1, 64)
        key_states_6 = None
        key_states_7 = view_10.transpose(1, 2)
        view_10 = None
        view_11 = value_states_6.view(1, 10, -1, 64)
        value_states_6 = None
        value_states_7 = view_11.transpose(1, 2)
        view_11 = None
        attention_mask_3 = causal_mask_5[
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
        value_3 = value_states_7.contiguous()
        attn_output_12 = torch._C._nn.scaled_dot_product_attention(
            query_3,
            key_3,
            value_3,
            attn_mask=attention_mask_3,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_3 = key_3 = value_3 = attention_mask_3 = None
        transpose_15 = attn_output_12.transpose(1, 2)
        attn_output_12 = None
        attn_output_13 = transpose_15.contiguous()
        transpose_15 = None
        reshape_4 = attn_output_13.reshape(1, 10, -1)
        attn_output_13 = None
        attn_output_14 = reshape_4.contiguous()
        reshape_4 = None
        attn_output_15 = torch._C._nn.linear(
            attn_output_14,
            l_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_14 = l_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_33 = torch.nn.functional.dropout(
            attn_output_15, p=0.1, training=False
        )
        attn_output_15 = None
        hidden_states_34 = hidden_states_31 + hidden_states_33
        hidden_states_31 = hidden_states_33 = None
        hidden_states_35 = torch.nn.functional.layer_norm(
            hidden_states_34,
            (1024,),
            l_self_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_ = (
            l_self_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_
        ) = None
        hidden_states_36 = torch._C._nn.linear(
            hidden_states_35,
            l_self_modules_layers_modules_3_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_3_modules_fc1_parameters_bias_,
        )
        hidden_states_35 = (
            l_self_modules_layers_modules_3_modules_fc1_parameters_weight_
        ) = l_self_modules_layers_modules_3_modules_fc1_parameters_bias_ = None
        hidden_states_37 = torch._C._nn.gelu(hidden_states_36)
        hidden_states_36 = None
        hidden_states_38 = torch.nn.functional.dropout(
            hidden_states_37, p=0.0, training=False
        )
        hidden_states_37 = None
        hidden_states_39 = torch._C._nn.linear(
            hidden_states_38,
            l_self_modules_layers_modules_3_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_3_modules_fc2_parameters_bias_,
        )
        hidden_states_38 = (
            l_self_modules_layers_modules_3_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_3_modules_fc2_parameters_bias_ = None
        hidden_states_40 = torch.nn.functional.dropout(
            hidden_states_39, p=0.1, training=False
        )
        hidden_states_39 = None
        hidden_states_41 = hidden_states_34 + hidden_states_40
        hidden_states_34 = hidden_states_40 = None
        hidden_states_42 = torch.nn.functional.layer_norm(
            hidden_states_41,
            (1024,),
            l_self_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_24 = torch._C._nn.linear(
            hidden_states_42,
            l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_12 = linear_24.view(1, 10, -1, 64)
        linear_24 = None
        query_states_4 = view_12.transpose(1, 2)
        view_12 = None
        key_states_8 = torch._C._nn.linear(
            hidden_states_42,
            l_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_8 = torch._C._nn.linear(
            hidden_states_42,
            l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_42 = l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_13 = key_states_8.view(1, 10, -1, 64)
        key_states_8 = None
        key_states_9 = view_13.transpose(1, 2)
        view_13 = None
        view_14 = value_states_8.view(1, 10, -1, 64)
        value_states_8 = None
        value_states_9 = view_14.transpose(1, 2)
        view_14 = None
        attention_mask_4 = causal_mask_5[
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
        value_4 = value_states_9.contiguous()
        attn_output_16 = torch._C._nn.scaled_dot_product_attention(
            query_4,
            key_4,
            value_4,
            attn_mask=attention_mask_4,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_4 = key_4 = value_4 = attention_mask_4 = None
        transpose_19 = attn_output_16.transpose(1, 2)
        attn_output_16 = None
        attn_output_17 = transpose_19.contiguous()
        transpose_19 = None
        reshape_5 = attn_output_17.reshape(1, 10, -1)
        attn_output_17 = None
        attn_output_18 = reshape_5.contiguous()
        reshape_5 = None
        attn_output_19 = torch._C._nn.linear(
            attn_output_18,
            l_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_18 = l_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_43 = torch.nn.functional.dropout(
            attn_output_19, p=0.1, training=False
        )
        attn_output_19 = None
        hidden_states_44 = hidden_states_41 + hidden_states_43
        hidden_states_41 = hidden_states_43 = None
        hidden_states_45 = torch.nn.functional.layer_norm(
            hidden_states_44,
            (1024,),
            l_self_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_ = (
            l_self_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_
        ) = None
        hidden_states_46 = torch._C._nn.linear(
            hidden_states_45,
            l_self_modules_layers_modules_4_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_4_modules_fc1_parameters_bias_,
        )
        hidden_states_45 = (
            l_self_modules_layers_modules_4_modules_fc1_parameters_weight_
        ) = l_self_modules_layers_modules_4_modules_fc1_parameters_bias_ = None
        hidden_states_47 = torch._C._nn.gelu(hidden_states_46)
        hidden_states_46 = None
        hidden_states_48 = torch.nn.functional.dropout(
            hidden_states_47, p=0.0, training=False
        )
        hidden_states_47 = None
        hidden_states_49 = torch._C._nn.linear(
            hidden_states_48,
            l_self_modules_layers_modules_4_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_4_modules_fc2_parameters_bias_,
        )
        hidden_states_48 = (
            l_self_modules_layers_modules_4_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_4_modules_fc2_parameters_bias_ = None
        hidden_states_50 = torch.nn.functional.dropout(
            hidden_states_49, p=0.1, training=False
        )
        hidden_states_49 = None
        hidden_states_51 = hidden_states_44 + hidden_states_50
        hidden_states_44 = hidden_states_50 = None
        hidden_states_52 = torch.nn.functional.layer_norm(
            hidden_states_51,
            (1024,),
            l_self_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_30 = torch._C._nn.linear(
            hidden_states_52,
            l_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_15 = linear_30.view(1, 10, -1, 64)
        linear_30 = None
        query_states_5 = view_15.transpose(1, 2)
        view_15 = None
        key_states_10 = torch._C._nn.linear(
            hidden_states_52,
            l_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_10 = torch._C._nn.linear(
            hidden_states_52,
            l_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_52 = l_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_16 = key_states_10.view(1, 10, -1, 64)
        key_states_10 = None
        key_states_11 = view_16.transpose(1, 2)
        view_16 = None
        view_17 = value_states_10.view(1, 10, -1, 64)
        value_states_10 = None
        value_states_11 = view_17.transpose(1, 2)
        view_17 = None
        attention_mask_5 = causal_mask_5[
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
        value_5 = value_states_11.contiguous()
        attn_output_20 = torch._C._nn.scaled_dot_product_attention(
            query_5,
            key_5,
            value_5,
            attn_mask=attention_mask_5,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_5 = key_5 = value_5 = attention_mask_5 = None
        transpose_23 = attn_output_20.transpose(1, 2)
        attn_output_20 = None
        attn_output_21 = transpose_23.contiguous()
        transpose_23 = None
        reshape_6 = attn_output_21.reshape(1, 10, -1)
        attn_output_21 = None
        attn_output_22 = reshape_6.contiguous()
        reshape_6 = None
        attn_output_23 = torch._C._nn.linear(
            attn_output_22,
            l_self_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_22 = l_self_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_53 = torch.nn.functional.dropout(
            attn_output_23, p=0.1, training=False
        )
        attn_output_23 = None
        hidden_states_54 = hidden_states_51 + hidden_states_53
        hidden_states_51 = hidden_states_53 = None
        hidden_states_55 = torch.nn.functional.layer_norm(
            hidden_states_54,
            (1024,),
            l_self_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_ = (
            l_self_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_
        ) = None
        hidden_states_56 = torch._C._nn.linear(
            hidden_states_55,
            l_self_modules_layers_modules_5_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_5_modules_fc1_parameters_bias_,
        )
        hidden_states_55 = (
            l_self_modules_layers_modules_5_modules_fc1_parameters_weight_
        ) = l_self_modules_layers_modules_5_modules_fc1_parameters_bias_ = None
        hidden_states_57 = torch._C._nn.gelu(hidden_states_56)
        hidden_states_56 = None
        hidden_states_58 = torch.nn.functional.dropout(
            hidden_states_57, p=0.0, training=False
        )
        hidden_states_57 = None
        hidden_states_59 = torch._C._nn.linear(
            hidden_states_58,
            l_self_modules_layers_modules_5_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_5_modules_fc2_parameters_bias_,
        )
        hidden_states_58 = (
            l_self_modules_layers_modules_5_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_5_modules_fc2_parameters_bias_ = None
        hidden_states_60 = torch.nn.functional.dropout(
            hidden_states_59, p=0.1, training=False
        )
        hidden_states_59 = None
        hidden_states_61 = hidden_states_54 + hidden_states_60
        hidden_states_54 = hidden_states_60 = None
        hidden_states_62 = torch.nn.functional.layer_norm(
            hidden_states_61,
            (1024,),
            l_self_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_36 = torch._C._nn.linear(
            hidden_states_62,
            l_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_18 = linear_36.view(1, 10, -1, 64)
        linear_36 = None
        query_states_6 = view_18.transpose(1, 2)
        view_18 = None
        key_states_12 = torch._C._nn.linear(
            hidden_states_62,
            l_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_12 = torch._C._nn.linear(
            hidden_states_62,
            l_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_62 = l_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_19 = key_states_12.view(1, 10, -1, 64)
        key_states_12 = None
        key_states_13 = view_19.transpose(1, 2)
        view_19 = None
        view_20 = value_states_12.view(1, 10, -1, 64)
        value_states_12 = None
        value_states_13 = view_20.transpose(1, 2)
        view_20 = None
        attention_mask_6 = causal_mask_5[
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
        value_6 = value_states_13.contiguous()
        attn_output_24 = torch._C._nn.scaled_dot_product_attention(
            query_6,
            key_6,
            value_6,
            attn_mask=attention_mask_6,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_6 = key_6 = value_6 = attention_mask_6 = None
        transpose_27 = attn_output_24.transpose(1, 2)
        attn_output_24 = None
        attn_output_25 = transpose_27.contiguous()
        transpose_27 = None
        reshape_7 = attn_output_25.reshape(1, 10, -1)
        attn_output_25 = None
        attn_output_26 = reshape_7.contiguous()
        reshape_7 = None
        attn_output_27 = torch._C._nn.linear(
            attn_output_26,
            l_self_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_26 = l_self_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_63 = torch.nn.functional.dropout(
            attn_output_27, p=0.1, training=False
        )
        attn_output_27 = None
        hidden_states_64 = hidden_states_61 + hidden_states_63
        hidden_states_61 = hidden_states_63 = None
        hidden_states_65 = torch.nn.functional.layer_norm(
            hidden_states_64,
            (1024,),
            l_self_modules_layers_modules_6_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_6_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_6_modules_final_layer_norm_parameters_weight_ = (
            l_self_modules_layers_modules_6_modules_final_layer_norm_parameters_bias_
        ) = None
        hidden_states_66 = torch._C._nn.linear(
            hidden_states_65,
            l_self_modules_layers_modules_6_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_6_modules_fc1_parameters_bias_,
        )
        hidden_states_65 = (
            l_self_modules_layers_modules_6_modules_fc1_parameters_weight_
        ) = l_self_modules_layers_modules_6_modules_fc1_parameters_bias_ = None
        hidden_states_67 = torch._C._nn.gelu(hidden_states_66)
        hidden_states_66 = None
        hidden_states_68 = torch.nn.functional.dropout(
            hidden_states_67, p=0.0, training=False
        )
        hidden_states_67 = None
        hidden_states_69 = torch._C._nn.linear(
            hidden_states_68,
            l_self_modules_layers_modules_6_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_6_modules_fc2_parameters_bias_,
        )
        hidden_states_68 = (
            l_self_modules_layers_modules_6_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_6_modules_fc2_parameters_bias_ = None
        hidden_states_70 = torch.nn.functional.dropout(
            hidden_states_69, p=0.1, training=False
        )
        hidden_states_69 = None
        hidden_states_71 = hidden_states_64 + hidden_states_70
        hidden_states_64 = hidden_states_70 = None
        hidden_states_72 = torch.nn.functional.layer_norm(
            hidden_states_71,
            (1024,),
            l_self_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_42 = torch._C._nn.linear(
            hidden_states_72,
            l_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_21 = linear_42.view(1, 10, -1, 64)
        linear_42 = None
        query_states_7 = view_21.transpose(1, 2)
        view_21 = None
        key_states_14 = torch._C._nn.linear(
            hidden_states_72,
            l_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_14 = torch._C._nn.linear(
            hidden_states_72,
            l_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_72 = l_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_22 = key_states_14.view(1, 10, -1, 64)
        key_states_14 = None
        key_states_15 = view_22.transpose(1, 2)
        view_22 = None
        view_23 = value_states_14.view(1, 10, -1, 64)
        value_states_14 = None
        value_states_15 = view_23.transpose(1, 2)
        view_23 = None
        attention_mask_7 = causal_mask_5[
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
        value_7 = value_states_15.contiguous()
        attn_output_28 = torch._C._nn.scaled_dot_product_attention(
            query_7,
            key_7,
            value_7,
            attn_mask=attention_mask_7,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_7 = key_7 = value_7 = attention_mask_7 = None
        transpose_31 = attn_output_28.transpose(1, 2)
        attn_output_28 = None
        attn_output_29 = transpose_31.contiguous()
        transpose_31 = None
        reshape_8 = attn_output_29.reshape(1, 10, -1)
        attn_output_29 = None
        attn_output_30 = reshape_8.contiguous()
        reshape_8 = None
        attn_output_31 = torch._C._nn.linear(
            attn_output_30,
            l_self_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_30 = l_self_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_73 = torch.nn.functional.dropout(
            attn_output_31, p=0.1, training=False
        )
        attn_output_31 = None
        hidden_states_74 = hidden_states_71 + hidden_states_73
        hidden_states_71 = hidden_states_73 = None
        hidden_states_75 = torch.nn.functional.layer_norm(
            hidden_states_74,
            (1024,),
            l_self_modules_layers_modules_7_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_7_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_7_modules_final_layer_norm_parameters_weight_ = (
            l_self_modules_layers_modules_7_modules_final_layer_norm_parameters_bias_
        ) = None
        hidden_states_76 = torch._C._nn.linear(
            hidden_states_75,
            l_self_modules_layers_modules_7_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_7_modules_fc1_parameters_bias_,
        )
        hidden_states_75 = (
            l_self_modules_layers_modules_7_modules_fc1_parameters_weight_
        ) = l_self_modules_layers_modules_7_modules_fc1_parameters_bias_ = None
        hidden_states_77 = torch._C._nn.gelu(hidden_states_76)
        hidden_states_76 = None
        hidden_states_78 = torch.nn.functional.dropout(
            hidden_states_77, p=0.0, training=False
        )
        hidden_states_77 = None
        hidden_states_79 = torch._C._nn.linear(
            hidden_states_78,
            l_self_modules_layers_modules_7_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_7_modules_fc2_parameters_bias_,
        )
        hidden_states_78 = (
            l_self_modules_layers_modules_7_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_7_modules_fc2_parameters_bias_ = None
        hidden_states_80 = torch.nn.functional.dropout(
            hidden_states_79, p=0.1, training=False
        )
        hidden_states_79 = None
        hidden_states_81 = hidden_states_74 + hidden_states_80
        hidden_states_74 = hidden_states_80 = None
        hidden_states_82 = torch.nn.functional.layer_norm(
            hidden_states_81,
            (1024,),
            l_self_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_48 = torch._C._nn.linear(
            hidden_states_82,
            l_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_24 = linear_48.view(1, 10, -1, 64)
        linear_48 = None
        query_states_8 = view_24.transpose(1, 2)
        view_24 = None
        key_states_16 = torch._C._nn.linear(
            hidden_states_82,
            l_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_16 = torch._C._nn.linear(
            hidden_states_82,
            l_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_82 = l_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_25 = key_states_16.view(1, 10, -1, 64)
        key_states_16 = None
        key_states_17 = view_25.transpose(1, 2)
        view_25 = None
        view_26 = value_states_16.view(1, 10, -1, 64)
        value_states_16 = None
        value_states_17 = view_26.transpose(1, 2)
        view_26 = None
        attention_mask_8 = causal_mask_5[
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
        value_8 = value_states_17.contiguous()
        attn_output_32 = torch._C._nn.scaled_dot_product_attention(
            query_8,
            key_8,
            value_8,
            attn_mask=attention_mask_8,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_8 = key_8 = value_8 = attention_mask_8 = None
        transpose_35 = attn_output_32.transpose(1, 2)
        attn_output_32 = None
        attn_output_33 = transpose_35.contiguous()
        transpose_35 = None
        reshape_9 = attn_output_33.reshape(1, 10, -1)
        attn_output_33 = None
        attn_output_34 = reshape_9.contiguous()
        reshape_9 = None
        attn_output_35 = torch._C._nn.linear(
            attn_output_34,
            l_self_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_34 = l_self_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_83 = torch.nn.functional.dropout(
            attn_output_35, p=0.1, training=False
        )
        attn_output_35 = None
        hidden_states_84 = hidden_states_81 + hidden_states_83
        hidden_states_81 = hidden_states_83 = None
        hidden_states_85 = torch.nn.functional.layer_norm(
            hidden_states_84,
            (1024,),
            l_self_modules_layers_modules_8_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_8_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_8_modules_final_layer_norm_parameters_weight_ = (
            l_self_modules_layers_modules_8_modules_final_layer_norm_parameters_bias_
        ) = None
        hidden_states_86 = torch._C._nn.linear(
            hidden_states_85,
            l_self_modules_layers_modules_8_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_8_modules_fc1_parameters_bias_,
        )
        hidden_states_85 = (
            l_self_modules_layers_modules_8_modules_fc1_parameters_weight_
        ) = l_self_modules_layers_modules_8_modules_fc1_parameters_bias_ = None
        hidden_states_87 = torch._C._nn.gelu(hidden_states_86)
        hidden_states_86 = None
        hidden_states_88 = torch.nn.functional.dropout(
            hidden_states_87, p=0.0, training=False
        )
        hidden_states_87 = None
        hidden_states_89 = torch._C._nn.linear(
            hidden_states_88,
            l_self_modules_layers_modules_8_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_8_modules_fc2_parameters_bias_,
        )
        hidden_states_88 = (
            l_self_modules_layers_modules_8_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_8_modules_fc2_parameters_bias_ = None
        hidden_states_90 = torch.nn.functional.dropout(
            hidden_states_89, p=0.1, training=False
        )
        hidden_states_89 = None
        hidden_states_91 = hidden_states_84 + hidden_states_90
        hidden_states_84 = hidden_states_90 = None
        hidden_states_92 = torch.nn.functional.layer_norm(
            hidden_states_91,
            (1024,),
            l_self_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_54 = torch._C._nn.linear(
            hidden_states_92,
            l_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_27 = linear_54.view(1, 10, -1, 64)
        linear_54 = None
        query_states_9 = view_27.transpose(1, 2)
        view_27 = None
        key_states_18 = torch._C._nn.linear(
            hidden_states_92,
            l_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_18 = torch._C._nn.linear(
            hidden_states_92,
            l_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_92 = l_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_28 = key_states_18.view(1, 10, -1, 64)
        key_states_18 = None
        key_states_19 = view_28.transpose(1, 2)
        view_28 = None
        view_29 = value_states_18.view(1, 10, -1, 64)
        value_states_18 = None
        value_states_19 = view_29.transpose(1, 2)
        view_29 = None
        attention_mask_9 = causal_mask_5[
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
        value_9 = value_states_19.contiguous()
        attn_output_36 = torch._C._nn.scaled_dot_product_attention(
            query_9,
            key_9,
            value_9,
            attn_mask=attention_mask_9,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_9 = key_9 = value_9 = attention_mask_9 = None
        transpose_39 = attn_output_36.transpose(1, 2)
        attn_output_36 = None
        attn_output_37 = transpose_39.contiguous()
        transpose_39 = None
        reshape_10 = attn_output_37.reshape(1, 10, -1)
        attn_output_37 = None
        attn_output_38 = reshape_10.contiguous()
        reshape_10 = None
        attn_output_39 = torch._C._nn.linear(
            attn_output_38,
            l_self_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_38 = l_self_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_93 = torch.nn.functional.dropout(
            attn_output_39, p=0.1, training=False
        )
        attn_output_39 = None
        hidden_states_94 = hidden_states_91 + hidden_states_93
        hidden_states_91 = hidden_states_93 = None
        hidden_states_95 = torch.nn.functional.layer_norm(
            hidden_states_94,
            (1024,),
            l_self_modules_layers_modules_9_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_9_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_9_modules_final_layer_norm_parameters_weight_ = (
            l_self_modules_layers_modules_9_modules_final_layer_norm_parameters_bias_
        ) = None
        hidden_states_96 = torch._C._nn.linear(
            hidden_states_95,
            l_self_modules_layers_modules_9_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_9_modules_fc1_parameters_bias_,
        )
        hidden_states_95 = (
            l_self_modules_layers_modules_9_modules_fc1_parameters_weight_
        ) = l_self_modules_layers_modules_9_modules_fc1_parameters_bias_ = None
        hidden_states_97 = torch._C._nn.gelu(hidden_states_96)
        hidden_states_96 = None
        hidden_states_98 = torch.nn.functional.dropout(
            hidden_states_97, p=0.0, training=False
        )
        hidden_states_97 = None
        hidden_states_99 = torch._C._nn.linear(
            hidden_states_98,
            l_self_modules_layers_modules_9_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_9_modules_fc2_parameters_bias_,
        )
        hidden_states_98 = (
            l_self_modules_layers_modules_9_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_9_modules_fc2_parameters_bias_ = None
        hidden_states_100 = torch.nn.functional.dropout(
            hidden_states_99, p=0.1, training=False
        )
        hidden_states_99 = None
        hidden_states_101 = hidden_states_94 + hidden_states_100
        hidden_states_94 = hidden_states_100 = None
        hidden_states_102 = torch.nn.functional.layer_norm(
            hidden_states_101,
            (1024,),
            l_self_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_60 = torch._C._nn.linear(
            hidden_states_102,
            l_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_30 = linear_60.view(1, 10, -1, 64)
        linear_60 = None
        query_states_10 = view_30.transpose(1, 2)
        view_30 = None
        key_states_20 = torch._C._nn.linear(
            hidden_states_102,
            l_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_20 = torch._C._nn.linear(
            hidden_states_102,
            l_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_102 = l_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_31 = key_states_20.view(1, 10, -1, 64)
        key_states_20 = None
        key_states_21 = view_31.transpose(1, 2)
        view_31 = None
        view_32 = value_states_20.view(1, 10, -1, 64)
        value_states_20 = None
        value_states_21 = view_32.transpose(1, 2)
        view_32 = None
        attention_mask_10 = causal_mask_5[
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
        value_10 = value_states_21.contiguous()
        attn_output_40 = torch._C._nn.scaled_dot_product_attention(
            query_10,
            key_10,
            value_10,
            attn_mask=attention_mask_10,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_10 = key_10 = value_10 = attention_mask_10 = None
        transpose_43 = attn_output_40.transpose(1, 2)
        attn_output_40 = None
        attn_output_41 = transpose_43.contiguous()
        transpose_43 = None
        reshape_11 = attn_output_41.reshape(1, 10, -1)
        attn_output_41 = None
        attn_output_42 = reshape_11.contiguous()
        reshape_11 = None
        attn_output_43 = torch._C._nn.linear(
            attn_output_42,
            l_self_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_42 = l_self_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_103 = torch.nn.functional.dropout(
            attn_output_43, p=0.1, training=False
        )
        attn_output_43 = None
        hidden_states_104 = hidden_states_101 + hidden_states_103
        hidden_states_101 = hidden_states_103 = None
        hidden_states_105 = torch.nn.functional.layer_norm(
            hidden_states_104,
            (1024,),
            l_self_modules_layers_modules_10_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_10_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_10_modules_final_layer_norm_parameters_weight_ = (
            l_self_modules_layers_modules_10_modules_final_layer_norm_parameters_bias_
        ) = None
        hidden_states_106 = torch._C._nn.linear(
            hidden_states_105,
            l_self_modules_layers_modules_10_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_10_modules_fc1_parameters_bias_,
        )
        hidden_states_105 = (
            l_self_modules_layers_modules_10_modules_fc1_parameters_weight_
        ) = l_self_modules_layers_modules_10_modules_fc1_parameters_bias_ = None
        hidden_states_107 = torch._C._nn.gelu(hidden_states_106)
        hidden_states_106 = None
        hidden_states_108 = torch.nn.functional.dropout(
            hidden_states_107, p=0.0, training=False
        )
        hidden_states_107 = None
        hidden_states_109 = torch._C._nn.linear(
            hidden_states_108,
            l_self_modules_layers_modules_10_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_10_modules_fc2_parameters_bias_,
        )
        hidden_states_108 = (
            l_self_modules_layers_modules_10_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_10_modules_fc2_parameters_bias_ = None
        hidden_states_110 = torch.nn.functional.dropout(
            hidden_states_109, p=0.1, training=False
        )
        hidden_states_109 = None
        hidden_states_111 = hidden_states_104 + hidden_states_110
        hidden_states_104 = hidden_states_110 = None
        hidden_states_112 = torch.nn.functional.layer_norm(
            hidden_states_111,
            (1024,),
            l_self_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_66 = torch._C._nn.linear(
            hidden_states_112,
            l_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_33 = linear_66.view(1, 10, -1, 64)
        linear_66 = None
        query_states_11 = view_33.transpose(1, 2)
        view_33 = None
        key_states_22 = torch._C._nn.linear(
            hidden_states_112,
            l_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_22 = torch._C._nn.linear(
            hidden_states_112,
            l_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_112 = l_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_34 = key_states_22.view(1, 10, -1, 64)
        key_states_22 = None
        key_states_23 = view_34.transpose(1, 2)
        view_34 = None
        view_35 = value_states_22.view(1, 10, -1, 64)
        value_states_22 = None
        value_states_23 = view_35.transpose(1, 2)
        view_35 = None
        attention_mask_11 = causal_mask_5[
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
        value_11 = value_states_23.contiguous()
        attn_output_44 = torch._C._nn.scaled_dot_product_attention(
            query_11,
            key_11,
            value_11,
            attn_mask=attention_mask_11,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_11 = key_11 = value_11 = attention_mask_11 = None
        transpose_47 = attn_output_44.transpose(1, 2)
        attn_output_44 = None
        attn_output_45 = transpose_47.contiguous()
        transpose_47 = None
        reshape_12 = attn_output_45.reshape(1, 10, -1)
        attn_output_45 = None
        attn_output_46 = reshape_12.contiguous()
        reshape_12 = None
        attn_output_47 = torch._C._nn.linear(
            attn_output_46,
            l_self_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_46 = l_self_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_113 = torch.nn.functional.dropout(
            attn_output_47, p=0.1, training=False
        )
        attn_output_47 = None
        hidden_states_114 = hidden_states_111 + hidden_states_113
        hidden_states_111 = hidden_states_113 = None
        hidden_states_115 = torch.nn.functional.layer_norm(
            hidden_states_114,
            (1024,),
            l_self_modules_layers_modules_11_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_11_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_11_modules_final_layer_norm_parameters_weight_ = (
            l_self_modules_layers_modules_11_modules_final_layer_norm_parameters_bias_
        ) = None
        hidden_states_116 = torch._C._nn.linear(
            hidden_states_115,
            l_self_modules_layers_modules_11_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_11_modules_fc1_parameters_bias_,
        )
        hidden_states_115 = (
            l_self_modules_layers_modules_11_modules_fc1_parameters_weight_
        ) = l_self_modules_layers_modules_11_modules_fc1_parameters_bias_ = None
        hidden_states_117 = torch._C._nn.gelu(hidden_states_116)
        hidden_states_116 = None
        hidden_states_118 = torch.nn.functional.dropout(
            hidden_states_117, p=0.0, training=False
        )
        hidden_states_117 = None
        hidden_states_119 = torch._C._nn.linear(
            hidden_states_118,
            l_self_modules_layers_modules_11_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_11_modules_fc2_parameters_bias_,
        )
        hidden_states_118 = (
            l_self_modules_layers_modules_11_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_11_modules_fc2_parameters_bias_ = None
        hidden_states_120 = torch.nn.functional.dropout(
            hidden_states_119, p=0.1, training=False
        )
        hidden_states_119 = None
        hidden_states_121 = hidden_states_114 + hidden_states_120
        hidden_states_114 = hidden_states_120 = None
        hidden_states_122 = torch.nn.functional.layer_norm(
            hidden_states_121,
            (1024,),
            l_self_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_72 = torch._C._nn.linear(
            hidden_states_122,
            l_self_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_36 = linear_72.view(1, 10, -1, 64)
        linear_72 = None
        query_states_12 = view_36.transpose(1, 2)
        view_36 = None
        key_states_24 = torch._C._nn.linear(
            hidden_states_122,
            l_self_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_24 = torch._C._nn.linear(
            hidden_states_122,
            l_self_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_122 = l_self_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_37 = key_states_24.view(1, 10, -1, 64)
        key_states_24 = None
        key_states_25 = view_37.transpose(1, 2)
        view_37 = None
        view_38 = value_states_24.view(1, 10, -1, 64)
        value_states_24 = None
        value_states_25 = view_38.transpose(1, 2)
        view_38 = None
        attention_mask_12 = causal_mask_5[
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
        value_12 = value_states_25.contiguous()
        attn_output_48 = torch._C._nn.scaled_dot_product_attention(
            query_12,
            key_12,
            value_12,
            attn_mask=attention_mask_12,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_12 = key_12 = value_12 = attention_mask_12 = None
        transpose_51 = attn_output_48.transpose(1, 2)
        attn_output_48 = None
        attn_output_49 = transpose_51.contiguous()
        transpose_51 = None
        reshape_13 = attn_output_49.reshape(1, 10, -1)
        attn_output_49 = None
        attn_output_50 = reshape_13.contiguous()
        reshape_13 = None
        attn_output_51 = torch._C._nn.linear(
            attn_output_50,
            l_self_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_50 = l_self_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_123 = torch.nn.functional.dropout(
            attn_output_51, p=0.1, training=False
        )
        attn_output_51 = None
        hidden_states_124 = hidden_states_121 + hidden_states_123
        hidden_states_121 = hidden_states_123 = None
        hidden_states_125 = torch.nn.functional.layer_norm(
            hidden_states_124,
            (1024,),
            l_self_modules_layers_modules_12_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_12_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_12_modules_final_layer_norm_parameters_weight_ = (
            l_self_modules_layers_modules_12_modules_final_layer_norm_parameters_bias_
        ) = None
        hidden_states_126 = torch._C._nn.linear(
            hidden_states_125,
            l_self_modules_layers_modules_12_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_12_modules_fc1_parameters_bias_,
        )
        hidden_states_125 = (
            l_self_modules_layers_modules_12_modules_fc1_parameters_weight_
        ) = l_self_modules_layers_modules_12_modules_fc1_parameters_bias_ = None
        hidden_states_127 = torch._C._nn.gelu(hidden_states_126)
        hidden_states_126 = None
        hidden_states_128 = torch.nn.functional.dropout(
            hidden_states_127, p=0.0, training=False
        )
        hidden_states_127 = None
        hidden_states_129 = torch._C._nn.linear(
            hidden_states_128,
            l_self_modules_layers_modules_12_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_12_modules_fc2_parameters_bias_,
        )
        hidden_states_128 = (
            l_self_modules_layers_modules_12_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_12_modules_fc2_parameters_bias_ = None
        hidden_states_130 = torch.nn.functional.dropout(
            hidden_states_129, p=0.1, training=False
        )
        hidden_states_129 = None
        hidden_states_131 = hidden_states_124 + hidden_states_130
        hidden_states_124 = hidden_states_130 = None
        hidden_states_132 = torch.nn.functional.layer_norm(
            hidden_states_131,
            (1024,),
            l_self_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_78 = torch._C._nn.linear(
            hidden_states_132,
            l_self_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_39 = linear_78.view(1, 10, -1, 64)
        linear_78 = None
        query_states_13 = view_39.transpose(1, 2)
        view_39 = None
        key_states_26 = torch._C._nn.linear(
            hidden_states_132,
            l_self_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_26 = torch._C._nn.linear(
            hidden_states_132,
            l_self_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_132 = l_self_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_40 = key_states_26.view(1, 10, -1, 64)
        key_states_26 = None
        key_states_27 = view_40.transpose(1, 2)
        view_40 = None
        view_41 = value_states_26.view(1, 10, -1, 64)
        value_states_26 = None
        value_states_27 = view_41.transpose(1, 2)
        view_41 = None
        attention_mask_13 = causal_mask_5[
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
        value_13 = value_states_27.contiguous()
        attn_output_52 = torch._C._nn.scaled_dot_product_attention(
            query_13,
            key_13,
            value_13,
            attn_mask=attention_mask_13,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_13 = key_13 = value_13 = attention_mask_13 = None
        transpose_55 = attn_output_52.transpose(1, 2)
        attn_output_52 = None
        attn_output_53 = transpose_55.contiguous()
        transpose_55 = None
        reshape_14 = attn_output_53.reshape(1, 10, -1)
        attn_output_53 = None
        attn_output_54 = reshape_14.contiguous()
        reshape_14 = None
        attn_output_55 = torch._C._nn.linear(
            attn_output_54,
            l_self_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_54 = l_self_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_133 = torch.nn.functional.dropout(
            attn_output_55, p=0.1, training=False
        )
        attn_output_55 = None
        hidden_states_134 = hidden_states_131 + hidden_states_133
        hidden_states_131 = hidden_states_133 = None
        hidden_states_135 = torch.nn.functional.layer_norm(
            hidden_states_134,
            (1024,),
            l_self_modules_layers_modules_13_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_13_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_13_modules_final_layer_norm_parameters_weight_ = (
            l_self_modules_layers_modules_13_modules_final_layer_norm_parameters_bias_
        ) = None
        hidden_states_136 = torch._C._nn.linear(
            hidden_states_135,
            l_self_modules_layers_modules_13_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_13_modules_fc1_parameters_bias_,
        )
        hidden_states_135 = (
            l_self_modules_layers_modules_13_modules_fc1_parameters_weight_
        ) = l_self_modules_layers_modules_13_modules_fc1_parameters_bias_ = None
        hidden_states_137 = torch._C._nn.gelu(hidden_states_136)
        hidden_states_136 = None
        hidden_states_138 = torch.nn.functional.dropout(
            hidden_states_137, p=0.0, training=False
        )
        hidden_states_137 = None
        hidden_states_139 = torch._C._nn.linear(
            hidden_states_138,
            l_self_modules_layers_modules_13_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_13_modules_fc2_parameters_bias_,
        )
        hidden_states_138 = (
            l_self_modules_layers_modules_13_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_13_modules_fc2_parameters_bias_ = None
        hidden_states_140 = torch.nn.functional.dropout(
            hidden_states_139, p=0.1, training=False
        )
        hidden_states_139 = None
        hidden_states_141 = hidden_states_134 + hidden_states_140
        hidden_states_134 = hidden_states_140 = None
        hidden_states_142 = torch.nn.functional.layer_norm(
            hidden_states_141,
            (1024,),
            l_self_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_84 = torch._C._nn.linear(
            hidden_states_142,
            l_self_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_42 = linear_84.view(1, 10, -1, 64)
        linear_84 = None
        query_states_14 = view_42.transpose(1, 2)
        view_42 = None
        key_states_28 = torch._C._nn.linear(
            hidden_states_142,
            l_self_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_28 = torch._C._nn.linear(
            hidden_states_142,
            l_self_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_142 = l_self_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_43 = key_states_28.view(1, 10, -1, 64)
        key_states_28 = None
        key_states_29 = view_43.transpose(1, 2)
        view_43 = None
        view_44 = value_states_28.view(1, 10, -1, 64)
        value_states_28 = None
        value_states_29 = view_44.transpose(1, 2)
        view_44 = None
        attention_mask_14 = causal_mask_5[
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
        value_14 = value_states_29.contiguous()
        attn_output_56 = torch._C._nn.scaled_dot_product_attention(
            query_14,
            key_14,
            value_14,
            attn_mask=attention_mask_14,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_14 = key_14 = value_14 = attention_mask_14 = None
        transpose_59 = attn_output_56.transpose(1, 2)
        attn_output_56 = None
        attn_output_57 = transpose_59.contiguous()
        transpose_59 = None
        reshape_15 = attn_output_57.reshape(1, 10, -1)
        attn_output_57 = None
        attn_output_58 = reshape_15.contiguous()
        reshape_15 = None
        attn_output_59 = torch._C._nn.linear(
            attn_output_58,
            l_self_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_58 = l_self_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_143 = torch.nn.functional.dropout(
            attn_output_59, p=0.1, training=False
        )
        attn_output_59 = None
        hidden_states_144 = hidden_states_141 + hidden_states_143
        hidden_states_141 = hidden_states_143 = None
        hidden_states_145 = torch.nn.functional.layer_norm(
            hidden_states_144,
            (1024,),
            l_self_modules_layers_modules_14_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_14_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_14_modules_final_layer_norm_parameters_weight_ = (
            l_self_modules_layers_modules_14_modules_final_layer_norm_parameters_bias_
        ) = None
        hidden_states_146 = torch._C._nn.linear(
            hidden_states_145,
            l_self_modules_layers_modules_14_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_14_modules_fc1_parameters_bias_,
        )
        hidden_states_145 = (
            l_self_modules_layers_modules_14_modules_fc1_parameters_weight_
        ) = l_self_modules_layers_modules_14_modules_fc1_parameters_bias_ = None
        hidden_states_147 = torch._C._nn.gelu(hidden_states_146)
        hidden_states_146 = None
        hidden_states_148 = torch.nn.functional.dropout(
            hidden_states_147, p=0.0, training=False
        )
        hidden_states_147 = None
        hidden_states_149 = torch._C._nn.linear(
            hidden_states_148,
            l_self_modules_layers_modules_14_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_14_modules_fc2_parameters_bias_,
        )
        hidden_states_148 = (
            l_self_modules_layers_modules_14_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_14_modules_fc2_parameters_bias_ = None
        hidden_states_150 = torch.nn.functional.dropout(
            hidden_states_149, p=0.1, training=False
        )
        hidden_states_149 = None
        hidden_states_151 = hidden_states_144 + hidden_states_150
        hidden_states_144 = hidden_states_150 = None
        hidden_states_152 = torch.nn.functional.layer_norm(
            hidden_states_151,
            (1024,),
            l_self_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_90 = torch._C._nn.linear(
            hidden_states_152,
            l_self_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_45 = linear_90.view(1, 10, -1, 64)
        linear_90 = None
        query_states_15 = view_45.transpose(1, 2)
        view_45 = None
        key_states_30 = torch._C._nn.linear(
            hidden_states_152,
            l_self_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_30 = torch._C._nn.linear(
            hidden_states_152,
            l_self_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_152 = l_self_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_46 = key_states_30.view(1, 10, -1, 64)
        key_states_30 = None
        key_states_31 = view_46.transpose(1, 2)
        view_46 = None
        view_47 = value_states_30.view(1, 10, -1, 64)
        value_states_30 = None
        value_states_31 = view_47.transpose(1, 2)
        view_47 = None
        attention_mask_15 = causal_mask_5[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_15 = query_states_15.contiguous()
        query_states_15 = None
        key_15 = key_states_31.contiguous()
        value_15 = value_states_31.contiguous()
        attn_output_60 = torch._C._nn.scaled_dot_product_attention(
            query_15,
            key_15,
            value_15,
            attn_mask=attention_mask_15,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_15 = key_15 = value_15 = attention_mask_15 = None
        transpose_63 = attn_output_60.transpose(1, 2)
        attn_output_60 = None
        attn_output_61 = transpose_63.contiguous()
        transpose_63 = None
        reshape_16 = attn_output_61.reshape(1, 10, -1)
        attn_output_61 = None
        attn_output_62 = reshape_16.contiguous()
        reshape_16 = None
        attn_output_63 = torch._C._nn.linear(
            attn_output_62,
            l_self_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_62 = l_self_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_153 = torch.nn.functional.dropout(
            attn_output_63, p=0.1, training=False
        )
        attn_output_63 = None
        hidden_states_154 = hidden_states_151 + hidden_states_153
        hidden_states_151 = hidden_states_153 = None
        hidden_states_155 = torch.nn.functional.layer_norm(
            hidden_states_154,
            (1024,),
            l_self_modules_layers_modules_15_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_15_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_15_modules_final_layer_norm_parameters_weight_ = (
            l_self_modules_layers_modules_15_modules_final_layer_norm_parameters_bias_
        ) = None
        hidden_states_156 = torch._C._nn.linear(
            hidden_states_155,
            l_self_modules_layers_modules_15_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_15_modules_fc1_parameters_bias_,
        )
        hidden_states_155 = (
            l_self_modules_layers_modules_15_modules_fc1_parameters_weight_
        ) = l_self_modules_layers_modules_15_modules_fc1_parameters_bias_ = None
        hidden_states_157 = torch._C._nn.gelu(hidden_states_156)
        hidden_states_156 = None
        hidden_states_158 = torch.nn.functional.dropout(
            hidden_states_157, p=0.0, training=False
        )
        hidden_states_157 = None
        hidden_states_159 = torch._C._nn.linear(
            hidden_states_158,
            l_self_modules_layers_modules_15_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_15_modules_fc2_parameters_bias_,
        )
        hidden_states_158 = (
            l_self_modules_layers_modules_15_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_15_modules_fc2_parameters_bias_ = None
        hidden_states_160 = torch.nn.functional.dropout(
            hidden_states_159, p=0.1, training=False
        )
        hidden_states_159 = None
        hidden_states_161 = hidden_states_154 + hidden_states_160
        hidden_states_154 = hidden_states_160 = None
        hidden_states_162 = torch.nn.functional.layer_norm(
            hidden_states_161,
            (1024,),
            l_self_modules_layers_modules_16_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_16_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_16_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_16_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_96 = torch._C._nn.linear(
            hidden_states_162,
            l_self_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_48 = linear_96.view(1, 10, -1, 64)
        linear_96 = None
        query_states_16 = view_48.transpose(1, 2)
        view_48 = None
        key_states_32 = torch._C._nn.linear(
            hidden_states_162,
            l_self_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_32 = torch._C._nn.linear(
            hidden_states_162,
            l_self_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_162 = l_self_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_49 = key_states_32.view(1, 10, -1, 64)
        key_states_32 = None
        key_states_33 = view_49.transpose(1, 2)
        view_49 = None
        view_50 = value_states_32.view(1, 10, -1, 64)
        value_states_32 = None
        value_states_33 = view_50.transpose(1, 2)
        view_50 = None
        attention_mask_16 = causal_mask_5[
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
        value_16 = value_states_33.contiguous()
        attn_output_64 = torch._C._nn.scaled_dot_product_attention(
            query_16,
            key_16,
            value_16,
            attn_mask=attention_mask_16,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_16 = key_16 = value_16 = attention_mask_16 = None
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
            l_self_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_66 = l_self_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_163 = torch.nn.functional.dropout(
            attn_output_67, p=0.1, training=False
        )
        attn_output_67 = None
        hidden_states_164 = hidden_states_161 + hidden_states_163
        hidden_states_161 = hidden_states_163 = None
        hidden_states_165 = torch.nn.functional.layer_norm(
            hidden_states_164,
            (1024,),
            l_self_modules_layers_modules_16_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_16_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_16_modules_final_layer_norm_parameters_weight_ = (
            l_self_modules_layers_modules_16_modules_final_layer_norm_parameters_bias_
        ) = None
        hidden_states_166 = torch._C._nn.linear(
            hidden_states_165,
            l_self_modules_layers_modules_16_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_16_modules_fc1_parameters_bias_,
        )
        hidden_states_165 = (
            l_self_modules_layers_modules_16_modules_fc1_parameters_weight_
        ) = l_self_modules_layers_modules_16_modules_fc1_parameters_bias_ = None
        hidden_states_167 = torch._C._nn.gelu(hidden_states_166)
        hidden_states_166 = None
        hidden_states_168 = torch.nn.functional.dropout(
            hidden_states_167, p=0.0, training=False
        )
        hidden_states_167 = None
        hidden_states_169 = torch._C._nn.linear(
            hidden_states_168,
            l_self_modules_layers_modules_16_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_16_modules_fc2_parameters_bias_,
        )
        hidden_states_168 = (
            l_self_modules_layers_modules_16_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_16_modules_fc2_parameters_bias_ = None
        hidden_states_170 = torch.nn.functional.dropout(
            hidden_states_169, p=0.1, training=False
        )
        hidden_states_169 = None
        hidden_states_171 = hidden_states_164 + hidden_states_170
        hidden_states_164 = hidden_states_170 = None
        hidden_states_172 = torch.nn.functional.layer_norm(
            hidden_states_171,
            (1024,),
            l_self_modules_layers_modules_17_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_17_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_17_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_17_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_102 = torch._C._nn.linear(
            hidden_states_172,
            l_self_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_51 = linear_102.view(1, 10, -1, 64)
        linear_102 = None
        query_states_17 = view_51.transpose(1, 2)
        view_51 = None
        key_states_34 = torch._C._nn.linear(
            hidden_states_172,
            l_self_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_34 = torch._C._nn.linear(
            hidden_states_172,
            l_self_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_172 = l_self_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_52 = key_states_34.view(1, 10, -1, 64)
        key_states_34 = None
        key_states_35 = view_52.transpose(1, 2)
        view_52 = None
        view_53 = value_states_34.view(1, 10, -1, 64)
        value_states_34 = None
        value_states_35 = view_53.transpose(1, 2)
        view_53 = None
        attention_mask_17 = causal_mask_5[
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
        value_17 = value_states_35.contiguous()
        attn_output_68 = torch._C._nn.scaled_dot_product_attention(
            query_17,
            key_17,
            value_17,
            attn_mask=attention_mask_17,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_17 = key_17 = value_17 = attention_mask_17 = None
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
            l_self_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_70 = l_self_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_173 = torch.nn.functional.dropout(
            attn_output_71, p=0.1, training=False
        )
        attn_output_71 = None
        hidden_states_174 = hidden_states_171 + hidden_states_173
        hidden_states_171 = hidden_states_173 = None
        hidden_states_175 = torch.nn.functional.layer_norm(
            hidden_states_174,
            (1024,),
            l_self_modules_layers_modules_17_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_17_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_17_modules_final_layer_norm_parameters_weight_ = (
            l_self_modules_layers_modules_17_modules_final_layer_norm_parameters_bias_
        ) = None
        hidden_states_176 = torch._C._nn.linear(
            hidden_states_175,
            l_self_modules_layers_modules_17_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_17_modules_fc1_parameters_bias_,
        )
        hidden_states_175 = (
            l_self_modules_layers_modules_17_modules_fc1_parameters_weight_
        ) = l_self_modules_layers_modules_17_modules_fc1_parameters_bias_ = None
        hidden_states_177 = torch._C._nn.gelu(hidden_states_176)
        hidden_states_176 = None
        hidden_states_178 = torch.nn.functional.dropout(
            hidden_states_177, p=0.0, training=False
        )
        hidden_states_177 = None
        hidden_states_179 = torch._C._nn.linear(
            hidden_states_178,
            l_self_modules_layers_modules_17_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_17_modules_fc2_parameters_bias_,
        )
        hidden_states_178 = (
            l_self_modules_layers_modules_17_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_17_modules_fc2_parameters_bias_ = None
        hidden_states_180 = torch.nn.functional.dropout(
            hidden_states_179, p=0.1, training=False
        )
        hidden_states_179 = None
        hidden_states_181 = hidden_states_174 + hidden_states_180
        hidden_states_174 = hidden_states_180 = None
        hidden_states_182 = torch.nn.functional.layer_norm(
            hidden_states_181,
            (1024,),
            l_self_modules_layers_modules_18_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_18_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_18_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_18_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_108 = torch._C._nn.linear(
            hidden_states_182,
            l_self_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_54 = linear_108.view(1, 10, -1, 64)
        linear_108 = None
        query_states_18 = view_54.transpose(1, 2)
        view_54 = None
        key_states_36 = torch._C._nn.linear(
            hidden_states_182,
            l_self_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_36 = torch._C._nn.linear(
            hidden_states_182,
            l_self_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_182 = l_self_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_55 = key_states_36.view(1, 10, -1, 64)
        key_states_36 = None
        key_states_37 = view_55.transpose(1, 2)
        view_55 = None
        view_56 = value_states_36.view(1, 10, -1, 64)
        value_states_36 = None
        value_states_37 = view_56.transpose(1, 2)
        view_56 = None
        attention_mask_18 = causal_mask_5[
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
        value_18 = value_states_37.contiguous()
        attn_output_72 = torch._C._nn.scaled_dot_product_attention(
            query_18,
            key_18,
            value_18,
            attn_mask=attention_mask_18,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_18 = key_18 = value_18 = attention_mask_18 = None
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
            l_self_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_74 = l_self_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_183 = torch.nn.functional.dropout(
            attn_output_75, p=0.1, training=False
        )
        attn_output_75 = None
        hidden_states_184 = hidden_states_181 + hidden_states_183
        hidden_states_181 = hidden_states_183 = None
        hidden_states_185 = torch.nn.functional.layer_norm(
            hidden_states_184,
            (1024,),
            l_self_modules_layers_modules_18_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_18_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_18_modules_final_layer_norm_parameters_weight_ = (
            l_self_modules_layers_modules_18_modules_final_layer_norm_parameters_bias_
        ) = None
        hidden_states_186 = torch._C._nn.linear(
            hidden_states_185,
            l_self_modules_layers_modules_18_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_18_modules_fc1_parameters_bias_,
        )
        hidden_states_185 = (
            l_self_modules_layers_modules_18_modules_fc1_parameters_weight_
        ) = l_self_modules_layers_modules_18_modules_fc1_parameters_bias_ = None
        hidden_states_187 = torch._C._nn.gelu(hidden_states_186)
        hidden_states_186 = None
        hidden_states_188 = torch.nn.functional.dropout(
            hidden_states_187, p=0.0, training=False
        )
        hidden_states_187 = None
        hidden_states_189 = torch._C._nn.linear(
            hidden_states_188,
            l_self_modules_layers_modules_18_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_18_modules_fc2_parameters_bias_,
        )
        hidden_states_188 = (
            l_self_modules_layers_modules_18_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_18_modules_fc2_parameters_bias_ = None
        hidden_states_190 = torch.nn.functional.dropout(
            hidden_states_189, p=0.1, training=False
        )
        hidden_states_189 = None
        hidden_states_191 = hidden_states_184 + hidden_states_190
        hidden_states_184 = hidden_states_190 = None
        hidden_states_192 = torch.nn.functional.layer_norm(
            hidden_states_191,
            (1024,),
            l_self_modules_layers_modules_19_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_19_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_19_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_19_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_114 = torch._C._nn.linear(
            hidden_states_192,
            l_self_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_57 = linear_114.view(1, 10, -1, 64)
        linear_114 = None
        query_states_19 = view_57.transpose(1, 2)
        view_57 = None
        key_states_38 = torch._C._nn.linear(
            hidden_states_192,
            l_self_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_38 = torch._C._nn.linear(
            hidden_states_192,
            l_self_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_192 = l_self_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_58 = key_states_38.view(1, 10, -1, 64)
        key_states_38 = None
        key_states_39 = view_58.transpose(1, 2)
        view_58 = None
        view_59 = value_states_38.view(1, 10, -1, 64)
        value_states_38 = None
        value_states_39 = view_59.transpose(1, 2)
        view_59 = None
        attention_mask_19 = causal_mask_5[
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
        value_19 = value_states_39.contiguous()
        attn_output_76 = torch._C._nn.scaled_dot_product_attention(
            query_19,
            key_19,
            value_19,
            attn_mask=attention_mask_19,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_19 = key_19 = value_19 = attention_mask_19 = None
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
            l_self_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_78 = l_self_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_193 = torch.nn.functional.dropout(
            attn_output_79, p=0.1, training=False
        )
        attn_output_79 = None
        hidden_states_194 = hidden_states_191 + hidden_states_193
        hidden_states_191 = hidden_states_193 = None
        hidden_states_195 = torch.nn.functional.layer_norm(
            hidden_states_194,
            (1024,),
            l_self_modules_layers_modules_19_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_19_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_19_modules_final_layer_norm_parameters_weight_ = (
            l_self_modules_layers_modules_19_modules_final_layer_norm_parameters_bias_
        ) = None
        hidden_states_196 = torch._C._nn.linear(
            hidden_states_195,
            l_self_modules_layers_modules_19_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_19_modules_fc1_parameters_bias_,
        )
        hidden_states_195 = (
            l_self_modules_layers_modules_19_modules_fc1_parameters_weight_
        ) = l_self_modules_layers_modules_19_modules_fc1_parameters_bias_ = None
        hidden_states_197 = torch._C._nn.gelu(hidden_states_196)
        hidden_states_196 = None
        hidden_states_198 = torch.nn.functional.dropout(
            hidden_states_197, p=0.0, training=False
        )
        hidden_states_197 = None
        hidden_states_199 = torch._C._nn.linear(
            hidden_states_198,
            l_self_modules_layers_modules_19_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_19_modules_fc2_parameters_bias_,
        )
        hidden_states_198 = (
            l_self_modules_layers_modules_19_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_19_modules_fc2_parameters_bias_ = None
        hidden_states_200 = torch.nn.functional.dropout(
            hidden_states_199, p=0.1, training=False
        )
        hidden_states_199 = None
        hidden_states_201 = hidden_states_194 + hidden_states_200
        hidden_states_194 = hidden_states_200 = None
        hidden_states_202 = torch.nn.functional.layer_norm(
            hidden_states_201,
            (1024,),
            l_self_modules_layers_modules_20_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_20_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_20_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_20_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_120 = torch._C._nn.linear(
            hidden_states_202,
            l_self_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_60 = linear_120.view(1, 10, -1, 64)
        linear_120 = None
        query_states_20 = view_60.transpose(1, 2)
        view_60 = None
        key_states_40 = torch._C._nn.linear(
            hidden_states_202,
            l_self_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_40 = torch._C._nn.linear(
            hidden_states_202,
            l_self_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_202 = l_self_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_61 = key_states_40.view(1, 10, -1, 64)
        key_states_40 = None
        key_states_41 = view_61.transpose(1, 2)
        view_61 = None
        view_62 = value_states_40.view(1, 10, -1, 64)
        value_states_40 = None
        value_states_41 = view_62.transpose(1, 2)
        view_62 = None
        attention_mask_20 = causal_mask_5[
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
        value_20 = value_states_41.contiguous()
        attn_output_80 = torch._C._nn.scaled_dot_product_attention(
            query_20,
            key_20,
            value_20,
            attn_mask=attention_mask_20,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_20 = key_20 = value_20 = attention_mask_20 = None
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
            l_self_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_82 = l_self_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_203 = torch.nn.functional.dropout(
            attn_output_83, p=0.1, training=False
        )
        attn_output_83 = None
        hidden_states_204 = hidden_states_201 + hidden_states_203
        hidden_states_201 = hidden_states_203 = None
        hidden_states_205 = torch.nn.functional.layer_norm(
            hidden_states_204,
            (1024,),
            l_self_modules_layers_modules_20_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_20_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_20_modules_final_layer_norm_parameters_weight_ = (
            l_self_modules_layers_modules_20_modules_final_layer_norm_parameters_bias_
        ) = None
        hidden_states_206 = torch._C._nn.linear(
            hidden_states_205,
            l_self_modules_layers_modules_20_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_20_modules_fc1_parameters_bias_,
        )
        hidden_states_205 = (
            l_self_modules_layers_modules_20_modules_fc1_parameters_weight_
        ) = l_self_modules_layers_modules_20_modules_fc1_parameters_bias_ = None
        hidden_states_207 = torch._C._nn.gelu(hidden_states_206)
        hidden_states_206 = None
        hidden_states_208 = torch.nn.functional.dropout(
            hidden_states_207, p=0.0, training=False
        )
        hidden_states_207 = None
        hidden_states_209 = torch._C._nn.linear(
            hidden_states_208,
            l_self_modules_layers_modules_20_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_20_modules_fc2_parameters_bias_,
        )
        hidden_states_208 = (
            l_self_modules_layers_modules_20_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_20_modules_fc2_parameters_bias_ = None
        hidden_states_210 = torch.nn.functional.dropout(
            hidden_states_209, p=0.1, training=False
        )
        hidden_states_209 = None
        hidden_states_211 = hidden_states_204 + hidden_states_210
        hidden_states_204 = hidden_states_210 = None
        hidden_states_212 = torch.nn.functional.layer_norm(
            hidden_states_211,
            (1024,),
            l_self_modules_layers_modules_21_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_21_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_21_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_21_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_126 = torch._C._nn.linear(
            hidden_states_212,
            l_self_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_63 = linear_126.view(1, 10, -1, 64)
        linear_126 = None
        query_states_21 = view_63.transpose(1, 2)
        view_63 = None
        key_states_42 = torch._C._nn.linear(
            hidden_states_212,
            l_self_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_42 = torch._C._nn.linear(
            hidden_states_212,
            l_self_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_212 = l_self_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_64 = key_states_42.view(1, 10, -1, 64)
        key_states_42 = None
        key_states_43 = view_64.transpose(1, 2)
        view_64 = None
        view_65 = value_states_42.view(1, 10, -1, 64)
        value_states_42 = None
        value_states_43 = view_65.transpose(1, 2)
        view_65 = None
        attention_mask_21 = causal_mask_5[
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
        value_21 = value_states_43.contiguous()
        attn_output_84 = torch._C._nn.scaled_dot_product_attention(
            query_21,
            key_21,
            value_21,
            attn_mask=attention_mask_21,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_21 = key_21 = value_21 = attention_mask_21 = None
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
            l_self_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_86 = l_self_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_213 = torch.nn.functional.dropout(
            attn_output_87, p=0.1, training=False
        )
        attn_output_87 = None
        hidden_states_214 = hidden_states_211 + hidden_states_213
        hidden_states_211 = hidden_states_213 = None
        hidden_states_215 = torch.nn.functional.layer_norm(
            hidden_states_214,
            (1024,),
            l_self_modules_layers_modules_21_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_21_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_21_modules_final_layer_norm_parameters_weight_ = (
            l_self_modules_layers_modules_21_modules_final_layer_norm_parameters_bias_
        ) = None
        hidden_states_216 = torch._C._nn.linear(
            hidden_states_215,
            l_self_modules_layers_modules_21_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_21_modules_fc1_parameters_bias_,
        )
        hidden_states_215 = (
            l_self_modules_layers_modules_21_modules_fc1_parameters_weight_
        ) = l_self_modules_layers_modules_21_modules_fc1_parameters_bias_ = None
        hidden_states_217 = torch._C._nn.gelu(hidden_states_216)
        hidden_states_216 = None
        hidden_states_218 = torch.nn.functional.dropout(
            hidden_states_217, p=0.0, training=False
        )
        hidden_states_217 = None
        hidden_states_219 = torch._C._nn.linear(
            hidden_states_218,
            l_self_modules_layers_modules_21_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_21_modules_fc2_parameters_bias_,
        )
        hidden_states_218 = (
            l_self_modules_layers_modules_21_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_21_modules_fc2_parameters_bias_ = None
        hidden_states_220 = torch.nn.functional.dropout(
            hidden_states_219, p=0.1, training=False
        )
        hidden_states_219 = None
        hidden_states_221 = hidden_states_214 + hidden_states_220
        hidden_states_214 = hidden_states_220 = None
        hidden_states_222 = torch.nn.functional.layer_norm(
            hidden_states_221,
            (1024,),
            l_self_modules_layers_modules_22_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_22_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_22_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_22_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_132 = torch._C._nn.linear(
            hidden_states_222,
            l_self_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_66 = linear_132.view(1, 10, -1, 64)
        linear_132 = None
        query_states_22 = view_66.transpose(1, 2)
        view_66 = None
        key_states_44 = torch._C._nn.linear(
            hidden_states_222,
            l_self_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_44 = torch._C._nn.linear(
            hidden_states_222,
            l_self_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_222 = l_self_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_67 = key_states_44.view(1, 10, -1, 64)
        key_states_44 = None
        key_states_45 = view_67.transpose(1, 2)
        view_67 = None
        view_68 = value_states_44.view(1, 10, -1, 64)
        value_states_44 = None
        value_states_45 = view_68.transpose(1, 2)
        view_68 = None
        attention_mask_22 = causal_mask_5[
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
        value_22 = value_states_45.contiguous()
        attn_output_88 = torch._C._nn.scaled_dot_product_attention(
            query_22,
            key_22,
            value_22,
            attn_mask=attention_mask_22,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_22 = key_22 = value_22 = attention_mask_22 = None
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
            l_self_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_90 = l_self_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_223 = torch.nn.functional.dropout(
            attn_output_91, p=0.1, training=False
        )
        attn_output_91 = None
        hidden_states_224 = hidden_states_221 + hidden_states_223
        hidden_states_221 = hidden_states_223 = None
        hidden_states_225 = torch.nn.functional.layer_norm(
            hidden_states_224,
            (1024,),
            l_self_modules_layers_modules_22_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_22_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_22_modules_final_layer_norm_parameters_weight_ = (
            l_self_modules_layers_modules_22_modules_final_layer_norm_parameters_bias_
        ) = None
        hidden_states_226 = torch._C._nn.linear(
            hidden_states_225,
            l_self_modules_layers_modules_22_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_22_modules_fc1_parameters_bias_,
        )
        hidden_states_225 = (
            l_self_modules_layers_modules_22_modules_fc1_parameters_weight_
        ) = l_self_modules_layers_modules_22_modules_fc1_parameters_bias_ = None
        hidden_states_227 = torch._C._nn.gelu(hidden_states_226)
        hidden_states_226 = None
        hidden_states_228 = torch.nn.functional.dropout(
            hidden_states_227, p=0.0, training=False
        )
        hidden_states_227 = None
        hidden_states_229 = torch._C._nn.linear(
            hidden_states_228,
            l_self_modules_layers_modules_22_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_22_modules_fc2_parameters_bias_,
        )
        hidden_states_228 = (
            l_self_modules_layers_modules_22_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_22_modules_fc2_parameters_bias_ = None
        hidden_states_230 = torch.nn.functional.dropout(
            hidden_states_229, p=0.1, training=False
        )
        hidden_states_229 = None
        hidden_states_231 = hidden_states_224 + hidden_states_230
        hidden_states_224 = hidden_states_230 = None
        hidden_states_232 = torch.nn.functional.layer_norm(
            hidden_states_231,
            (1024,),
            l_self_modules_layers_modules_23_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_23_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_23_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_23_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_138 = torch._C._nn.linear(
            hidden_states_232,
            l_self_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_69 = linear_138.view(1, 10, -1, 64)
        linear_138 = None
        query_states_23 = view_69.transpose(1, 2)
        view_69 = None
        key_states_46 = torch._C._nn.linear(
            hidden_states_232,
            l_self_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_46 = torch._C._nn.linear(
            hidden_states_232,
            l_self_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_232 = l_self_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_70 = key_states_46.view(1, 10, -1, 64)
        key_states_46 = None
        key_states_47 = view_70.transpose(1, 2)
        view_70 = None
        view_71 = value_states_46.view(1, 10, -1, 64)
        value_states_46 = None
        value_states_47 = view_71.transpose(1, 2)
        view_71 = None
        attention_mask_23 = causal_mask_5[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        causal_mask_5 = None
        query_23 = query_states_23.contiguous()
        query_states_23 = None
        key_23 = key_states_47.contiguous()
        value_23 = value_states_47.contiguous()
        attn_output_92 = torch._C._nn.scaled_dot_product_attention(
            query_23,
            key_23,
            value_23,
            attn_mask=attention_mask_23,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_23 = key_23 = value_23 = attention_mask_23 = None
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
            l_self_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_94 = l_self_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_233 = torch.nn.functional.dropout(
            attn_output_95, p=0.1, training=False
        )
        attn_output_95 = None
        hidden_states_234 = hidden_states_231 + hidden_states_233
        hidden_states_231 = hidden_states_233 = None
        hidden_states_235 = torch.nn.functional.layer_norm(
            hidden_states_234,
            (1024,),
            l_self_modules_layers_modules_23_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_23_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_23_modules_final_layer_norm_parameters_weight_ = (
            l_self_modules_layers_modules_23_modules_final_layer_norm_parameters_bias_
        ) = None
        hidden_states_236 = torch._C._nn.linear(
            hidden_states_235,
            l_self_modules_layers_modules_23_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_23_modules_fc1_parameters_bias_,
        )
        hidden_states_235 = (
            l_self_modules_layers_modules_23_modules_fc1_parameters_weight_
        ) = l_self_modules_layers_modules_23_modules_fc1_parameters_bias_ = None
        hidden_states_237 = torch._C._nn.gelu(hidden_states_236)
        hidden_states_236 = None
        hidden_states_238 = torch.nn.functional.dropout(
            hidden_states_237, p=0.0, training=False
        )
        hidden_states_237 = None
        hidden_states_239 = torch._C._nn.linear(
            hidden_states_238,
            l_self_modules_layers_modules_23_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_23_modules_fc2_parameters_bias_,
        )
        hidden_states_238 = (
            l_self_modules_layers_modules_23_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_23_modules_fc2_parameters_bias_ = None
        hidden_states_240 = torch.nn.functional.dropout(
            hidden_states_239, p=0.1, training=False
        )
        hidden_states_239 = None
        hidden_states_241 = hidden_states_234 + hidden_states_240
        hidden_states_234 = hidden_states_240 = None
        hidden_states_242 = torch.nn.functional.layer_norm(
            hidden_states_241,
            (1024,),
            l_self_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_241 = (
            l_self_modules_layer_norm_parameters_weight_
        ) = l_self_modules_layer_norm_parameters_bias_ = None
        return (
            value_states_1,
            key_states_1,
            value_states_3,
            key_states_3,
            value_states_5,
            key_states_5,
            value_states_7,
            key_states_7,
            value_states_9,
            key_states_9,
            value_states_11,
            key_states_11,
            value_states_13,
            key_states_13,
            value_states_15,
            key_states_15,
            value_states_17,
            key_states_17,
            value_states_19,
            key_states_19,
            value_states_21,
            key_states_21,
            value_states_23,
            key_states_23,
            value_states_25,
            key_states_25,
            value_states_27,
            key_states_27,
            value_states_29,
            key_states_29,
            value_states_31,
            key_states_31,
            value_states_33,
            key_states_33,
            value_states_35,
            key_states_35,
            value_states_37,
            key_states_37,
            value_states_39,
            key_states_39,
            value_states_41,
            key_states_41,
            value_states_43,
            key_states_43,
            value_states_45,
            key_states_45,
            value_states_47,
            key_states_47,
            hidden_states_242,
        )
