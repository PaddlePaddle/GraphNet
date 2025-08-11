import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_inputs_embeds_: torch.Tensor,
        L_encoder_hidden_states_: torch.Tensor,
        L_encoder_attention_mask_: torch.Tensor,
        L_self_modules_embed_positions_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layernorm_embedding_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layernorm_embedding_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_inputs_embeds_ = L_inputs_embeds_
        l_encoder_hidden_states_ = L_encoder_hidden_states_
        l_encoder_attention_mask_ = L_encoder_attention_mask_
        l_self_modules_embed_positions_parameters_weight_ = (
            L_self_modules_embed_positions_parameters_weight_
        )
        l_self_modules_layernorm_embedding_parameters_weight_ = (
            L_self_modules_layernorm_embedding_parameters_weight_
        )
        l_self_modules_layernorm_embedding_parameters_bias_ = (
            L_self_modules_layernorm_embedding_parameters_bias_
        )
        l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_bias_
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
        l_self_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_
        )
        l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_bias_
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
        l_self_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_
        )
        l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_bias_
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
        l_self_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_
        )
        l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_bias_
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
        l_self_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_
        )
        l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_bias_
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
        l_self_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_
        )
        l_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_bias_
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
        l_self_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_
        )
        l_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_6_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_6_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_6_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_6_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_6_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_6_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_6_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_6_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_6_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_6_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_6_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_6_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_6_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_6_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_6_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_6_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_6_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_6_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_6_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_6_modules_encoder_attn_layer_norm_parameters_bias_
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
        l_self_modules_layers_modules_6_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_layers_modules_6_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_layers_modules_6_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_layers_modules_6_modules_final_layer_norm_parameters_bias_
        )
        l_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_7_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_7_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_7_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_7_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_7_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_7_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_7_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_7_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_7_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_7_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_7_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_7_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_7_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_7_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_7_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_7_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_7_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_7_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_7_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_7_modules_encoder_attn_layer_norm_parameters_bias_
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
        l_self_modules_layers_modules_7_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_layers_modules_7_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_layers_modules_7_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_layers_modules_7_modules_final_layer_norm_parameters_bias_
        )
        l_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_8_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_8_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_8_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_8_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_8_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_8_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_8_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_8_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_8_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_8_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_8_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_8_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_8_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_8_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_8_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_8_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_8_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_8_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_8_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_8_modules_encoder_attn_layer_norm_parameters_bias_
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
        l_self_modules_layers_modules_8_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_layers_modules_8_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_layers_modules_8_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_layers_modules_8_modules_final_layer_norm_parameters_bias_
        )
        l_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_9_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_9_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_9_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_9_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_9_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_9_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_9_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_9_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_9_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_9_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_9_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_9_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_9_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_9_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_9_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_9_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_9_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_9_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_9_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_9_modules_encoder_attn_layer_norm_parameters_bias_
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
        l_self_modules_layers_modules_9_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_layers_modules_9_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_layers_modules_9_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_layers_modules_9_modules_final_layer_norm_parameters_bias_
        )
        l_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_10_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_10_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_10_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_10_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_10_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_10_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_10_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_10_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_10_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_10_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_10_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_10_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_10_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_10_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_10_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_10_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_10_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_10_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_10_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_10_modules_encoder_attn_layer_norm_parameters_bias_
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
        l_self_modules_layers_modules_10_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_layers_modules_10_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_layers_modules_10_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_layers_modules_10_modules_final_layer_norm_parameters_bias_
        )
        l_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_11_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_11_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_11_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_11_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_11_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_11_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_11_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_11_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_11_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_11_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_11_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_11_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_11_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_11_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_11_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_11_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_11_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_11_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_11_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_11_modules_encoder_attn_layer_norm_parameters_bias_
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
        l_self_modules_layers_modules_11_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_layers_modules_11_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_layers_modules_11_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_layers_modules_11_modules_final_layer_norm_parameters_bias_
        )
        cache_position = torch.arange(0, 11, device=device(type="cuda", index=0))
        causal_mask = torch.full(
            (11, 12),
            fill_value=-3.4028234663852886e38,
            dtype=torch.float32,
            device=device(type="cuda", index=0),
        )
        causal_mask_1 = torch.triu(causal_mask, diagonal=1)
        causal_mask = None
        arange_1 = torch.arange(12, device=device(type="cuda", index=0))
        reshape = cache_position.reshape(-1, 1)
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
        getitem_1 = l_encoder_attention_mask_[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        l_encoder_attention_mask_ = None
        expand_1 = getitem_1.expand(1, 1, 11, 11)
        getitem_1 = None
        expanded_mask = expand_1.to(torch.float32)
        expand_1 = None
        tensor = torch.tensor(1.0, dtype=torch.float32)
        inverted_mask = tensor - expanded_mask
        tensor = expanded_mask = None
        to_1 = inverted_mask.to(torch.bool)
        encoder_attention_mask = inverted_mask.masked_fill(to_1, -3.4028234663852886e38)
        inverted_mask = to_1 = None
        position_ids = cache_position.unsqueeze(0)
        cache_position = None
        add = position_ids + 2
        position_ids = None
        positions = torch.nn.functional.embedding(
            add,
            l_self_modules_embed_positions_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        add = l_self_modules_embed_positions_parameters_weight_ = None
        positions_1 = positions.to(device(type="cuda", index=0))
        positions = None
        hidden_states = l_inputs_embeds_ + positions_1
        l_inputs_embeds_ = positions_1 = None
        hidden_states_1 = torch.nn.functional.layer_norm(
            hidden_states,
            (1024,),
            l_self_modules_layernorm_embedding_parameters_weight_,
            l_self_modules_layernorm_embedding_parameters_bias_,
            1e-05,
        )
        hidden_states = (
            l_self_modules_layernorm_embedding_parameters_weight_
        ) = l_self_modules_layernorm_embedding_parameters_bias_ = None
        hidden_states_2 = torch.nn.functional.dropout(
            hidden_states_1, p=0.1, training=False
        )
        hidden_states_1 = None
        linear = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view = linear.view(1, 11, -1, 64)
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
        l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_1 = key_states.view(1, 11, -1, 64)
        key_states = None
        key_states_1 = view_1.transpose(1, 2)
        view_1 = None
        view_2 = value_states.view(1, 11, -1, 64)
        value_states = None
        value_states_1 = view_2.transpose(1, 2)
        view_2 = None
        attention_mask = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 11, None),
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
        reshape_1 = attn_output_1.reshape(1, 11, -1)
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
        hidden_states_4 = hidden_states_2 + hidden_states_3
        hidden_states_2 = hidden_states_3 = None
        hidden_states_5 = torch.nn.functional.layer_norm(
            hidden_states_4,
            (1024,),
            l_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_4 = l_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_4 = torch._C._nn.linear(
            hidden_states_5,
            l_self_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        view_3 = linear_4.view(1, 11, -1, 64)
        linear_4 = None
        query_states_1 = view_3.transpose(1, 2)
        view_3 = None
        key_states_2 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_2 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_4 = key_states_2.view(1, 11, -1, 64)
        key_states_2 = None
        key_states_3 = view_4.transpose(1, 2)
        view_4 = None
        view_5 = value_states_2.view(1, 11, -1, 64)
        value_states_2 = None
        value_states_3 = view_5.transpose(1, 2)
        view_5 = None
        attention_mask_1 = encoder_attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 11, None),
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
        reshape_2 = attn_output_5.reshape(1, 11, -1)
        attn_output_5 = None
        attn_output_6 = reshape_2.contiguous()
        reshape_2 = None
        attn_output_7 = torch._C._nn.linear(
            attn_output_6,
            l_self_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_6 = l_self_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_6 = torch.nn.functional.dropout(
            attn_output_7, p=0.1, training=False
        )
        attn_output_7 = None
        hidden_states_7 = hidden_states_5 + hidden_states_6
        hidden_states_5 = hidden_states_6 = None
        hidden_states_8 = torch.nn.functional.layer_norm(
            hidden_states_7,
            (1024,),
            l_self_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_7 = l_self_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_8 = torch._C._nn.linear(
            hidden_states_8,
            l_self_modules_layers_modules_0_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_0_modules_fc1_parameters_bias_,
        )
        l_self_modules_layers_modules_0_modules_fc1_parameters_weight_ = (
            l_self_modules_layers_modules_0_modules_fc1_parameters_bias_
        ) = None
        hidden_states_9 = torch._C._nn.gelu(linear_8)
        linear_8 = None
        hidden_states_10 = torch.nn.functional.dropout(
            hidden_states_9, p=0, training=False
        )
        hidden_states_9 = None
        hidden_states_11 = torch._C._nn.linear(
            hidden_states_10,
            l_self_modules_layers_modules_0_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_0_modules_fc2_parameters_bias_,
        )
        hidden_states_10 = (
            l_self_modules_layers_modules_0_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_0_modules_fc2_parameters_bias_ = None
        hidden_states_12 = torch.nn.functional.dropout(
            hidden_states_11, p=0.1, training=False
        )
        hidden_states_11 = None
        hidden_states_13 = hidden_states_8 + hidden_states_12
        hidden_states_8 = hidden_states_12 = None
        hidden_states_14 = torch.nn.functional.layer_norm(
            hidden_states_13,
            (1024,),
            l_self_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_13 = (
            l_self_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_
        ) = (
            l_self_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_
        ) = None
        linear_10 = torch._C._nn.linear(
            hidden_states_14,
            l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_6 = linear_10.view(1, 11, -1, 64)
        linear_10 = None
        query_states_2 = view_6.transpose(1, 2)
        view_6 = None
        key_states_4 = torch._C._nn.linear(
            hidden_states_14,
            l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_4 = torch._C._nn.linear(
            hidden_states_14,
            l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_7 = key_states_4.view(1, 11, -1, 64)
        key_states_4 = None
        key_states_5 = view_7.transpose(1, 2)
        view_7 = None
        view_8 = value_states_4.view(1, 11, -1, 64)
        value_states_4 = None
        value_states_5 = view_8.transpose(1, 2)
        view_8 = None
        attention_mask_2 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 11, None),
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
        reshape_3 = attn_output_9.reshape(1, 11, -1)
        attn_output_9 = None
        attn_output_10 = reshape_3.contiguous()
        reshape_3 = None
        attn_output_11 = torch._C._nn.linear(
            attn_output_10,
            l_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_10 = l_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_15 = torch.nn.functional.dropout(
            attn_output_11, p=0.1, training=False
        )
        attn_output_11 = None
        hidden_states_16 = hidden_states_14 + hidden_states_15
        hidden_states_14 = hidden_states_15 = None
        hidden_states_17 = torch.nn.functional.layer_norm(
            hidden_states_16,
            (1024,),
            l_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_16 = l_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_14 = torch._C._nn.linear(
            hidden_states_17,
            l_self_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        view_9 = linear_14.view(1, 11, -1, 64)
        linear_14 = None
        query_states_3 = view_9.transpose(1, 2)
        view_9 = None
        key_states_6 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_6 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_10 = key_states_6.view(1, 11, -1, 64)
        key_states_6 = None
        key_states_7 = view_10.transpose(1, 2)
        view_10 = None
        view_11 = value_states_6.view(1, 11, -1, 64)
        value_states_6 = None
        value_states_7 = view_11.transpose(1, 2)
        view_11 = None
        attention_mask_3 = encoder_attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 11, None),
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
        reshape_4 = attn_output_13.reshape(1, 11, -1)
        attn_output_13 = None
        attn_output_14 = reshape_4.contiguous()
        reshape_4 = None
        attn_output_15 = torch._C._nn.linear(
            attn_output_14,
            l_self_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_14 = l_self_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_18 = torch.nn.functional.dropout(
            attn_output_15, p=0.1, training=False
        )
        attn_output_15 = None
        hidden_states_19 = hidden_states_17 + hidden_states_18
        hidden_states_17 = hidden_states_18 = None
        hidden_states_20 = torch.nn.functional.layer_norm(
            hidden_states_19,
            (1024,),
            l_self_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_19 = l_self_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_18 = torch._C._nn.linear(
            hidden_states_20,
            l_self_modules_layers_modules_1_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_1_modules_fc1_parameters_bias_,
        )
        l_self_modules_layers_modules_1_modules_fc1_parameters_weight_ = (
            l_self_modules_layers_modules_1_modules_fc1_parameters_bias_
        ) = None
        hidden_states_21 = torch._C._nn.gelu(linear_18)
        linear_18 = None
        hidden_states_22 = torch.nn.functional.dropout(
            hidden_states_21, p=0, training=False
        )
        hidden_states_21 = None
        hidden_states_23 = torch._C._nn.linear(
            hidden_states_22,
            l_self_modules_layers_modules_1_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_1_modules_fc2_parameters_bias_,
        )
        hidden_states_22 = (
            l_self_modules_layers_modules_1_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_1_modules_fc2_parameters_bias_ = None
        hidden_states_24 = torch.nn.functional.dropout(
            hidden_states_23, p=0.1, training=False
        )
        hidden_states_23 = None
        hidden_states_25 = hidden_states_20 + hidden_states_24
        hidden_states_20 = hidden_states_24 = None
        hidden_states_26 = torch.nn.functional.layer_norm(
            hidden_states_25,
            (1024,),
            l_self_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_25 = (
            l_self_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_
        ) = (
            l_self_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_
        ) = None
        linear_20 = torch._C._nn.linear(
            hidden_states_26,
            l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_12 = linear_20.view(1, 11, -1, 64)
        linear_20 = None
        query_states_4 = view_12.transpose(1, 2)
        view_12 = None
        key_states_8 = torch._C._nn.linear(
            hidden_states_26,
            l_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_8 = torch._C._nn.linear(
            hidden_states_26,
            l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_13 = key_states_8.view(1, 11, -1, 64)
        key_states_8 = None
        key_states_9 = view_13.transpose(1, 2)
        view_13 = None
        view_14 = value_states_8.view(1, 11, -1, 64)
        value_states_8 = None
        value_states_9 = view_14.transpose(1, 2)
        view_14 = None
        attention_mask_4 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 11, None),
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
        reshape_5 = attn_output_17.reshape(1, 11, -1)
        attn_output_17 = None
        attn_output_18 = reshape_5.contiguous()
        reshape_5 = None
        attn_output_19 = torch._C._nn.linear(
            attn_output_18,
            l_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_18 = l_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_27 = torch.nn.functional.dropout(
            attn_output_19, p=0.1, training=False
        )
        attn_output_19 = None
        hidden_states_28 = hidden_states_26 + hidden_states_27
        hidden_states_26 = hidden_states_27 = None
        hidden_states_29 = torch.nn.functional.layer_norm(
            hidden_states_28,
            (1024,),
            l_self_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_28 = l_self_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_24 = torch._C._nn.linear(
            hidden_states_29,
            l_self_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        view_15 = linear_24.view(1, 11, -1, 64)
        linear_24 = None
        query_states_5 = view_15.transpose(1, 2)
        view_15 = None
        key_states_10 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_10 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_16 = key_states_10.view(1, 11, -1, 64)
        key_states_10 = None
        key_states_11 = view_16.transpose(1, 2)
        view_16 = None
        view_17 = value_states_10.view(1, 11, -1, 64)
        value_states_10 = None
        value_states_11 = view_17.transpose(1, 2)
        view_17 = None
        attention_mask_5 = encoder_attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 11, None),
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
        reshape_6 = attn_output_21.reshape(1, 11, -1)
        attn_output_21 = None
        attn_output_22 = reshape_6.contiguous()
        reshape_6 = None
        attn_output_23 = torch._C._nn.linear(
            attn_output_22,
            l_self_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_22 = l_self_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_30 = torch.nn.functional.dropout(
            attn_output_23, p=0.1, training=False
        )
        attn_output_23 = None
        hidden_states_31 = hidden_states_29 + hidden_states_30
        hidden_states_29 = hidden_states_30 = None
        hidden_states_32 = torch.nn.functional.layer_norm(
            hidden_states_31,
            (1024,),
            l_self_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_31 = l_self_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_28 = torch._C._nn.linear(
            hidden_states_32,
            l_self_modules_layers_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_fc1_parameters_bias_,
        )
        l_self_modules_layers_modules_2_modules_fc1_parameters_weight_ = (
            l_self_modules_layers_modules_2_modules_fc1_parameters_bias_
        ) = None
        hidden_states_33 = torch._C._nn.gelu(linear_28)
        linear_28 = None
        hidden_states_34 = torch.nn.functional.dropout(
            hidden_states_33, p=0, training=False
        )
        hidden_states_33 = None
        hidden_states_35 = torch._C._nn.linear(
            hidden_states_34,
            l_self_modules_layers_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_fc2_parameters_bias_,
        )
        hidden_states_34 = (
            l_self_modules_layers_modules_2_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_2_modules_fc2_parameters_bias_ = None
        hidden_states_36 = torch.nn.functional.dropout(
            hidden_states_35, p=0.1, training=False
        )
        hidden_states_35 = None
        hidden_states_37 = hidden_states_32 + hidden_states_36
        hidden_states_32 = hidden_states_36 = None
        hidden_states_38 = torch.nn.functional.layer_norm(
            hidden_states_37,
            (1024,),
            l_self_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_37 = (
            l_self_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_
        ) = (
            l_self_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_
        ) = None
        linear_30 = torch._C._nn.linear(
            hidden_states_38,
            l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_18 = linear_30.view(1, 11, -1, 64)
        linear_30 = None
        query_states_6 = view_18.transpose(1, 2)
        view_18 = None
        key_states_12 = torch._C._nn.linear(
            hidden_states_38,
            l_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_12 = torch._C._nn.linear(
            hidden_states_38,
            l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_19 = key_states_12.view(1, 11, -1, 64)
        key_states_12 = None
        key_states_13 = view_19.transpose(1, 2)
        view_19 = None
        view_20 = value_states_12.view(1, 11, -1, 64)
        value_states_12 = None
        value_states_13 = view_20.transpose(1, 2)
        view_20 = None
        attention_mask_6 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 11, None),
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
        reshape_7 = attn_output_25.reshape(1, 11, -1)
        attn_output_25 = None
        attn_output_26 = reshape_7.contiguous()
        reshape_7 = None
        attn_output_27 = torch._C._nn.linear(
            attn_output_26,
            l_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_26 = l_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_39 = torch.nn.functional.dropout(
            attn_output_27, p=0.1, training=False
        )
        attn_output_27 = None
        hidden_states_40 = hidden_states_38 + hidden_states_39
        hidden_states_38 = hidden_states_39 = None
        hidden_states_41 = torch.nn.functional.layer_norm(
            hidden_states_40,
            (1024,),
            l_self_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_40 = l_self_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_34 = torch._C._nn.linear(
            hidden_states_41,
            l_self_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        view_21 = linear_34.view(1, 11, -1, 64)
        linear_34 = None
        query_states_7 = view_21.transpose(1, 2)
        view_21 = None
        key_states_14 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_14 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_22 = key_states_14.view(1, 11, -1, 64)
        key_states_14 = None
        key_states_15 = view_22.transpose(1, 2)
        view_22 = None
        view_23 = value_states_14.view(1, 11, -1, 64)
        value_states_14 = None
        value_states_15 = view_23.transpose(1, 2)
        view_23 = None
        attention_mask_7 = encoder_attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 11, None),
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
        reshape_8 = attn_output_29.reshape(1, 11, -1)
        attn_output_29 = None
        attn_output_30 = reshape_8.contiguous()
        reshape_8 = None
        attn_output_31 = torch._C._nn.linear(
            attn_output_30,
            l_self_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_30 = l_self_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_42 = torch.nn.functional.dropout(
            attn_output_31, p=0.1, training=False
        )
        attn_output_31 = None
        hidden_states_43 = hidden_states_41 + hidden_states_42
        hidden_states_41 = hidden_states_42 = None
        hidden_states_44 = torch.nn.functional.layer_norm(
            hidden_states_43,
            (1024,),
            l_self_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_43 = l_self_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_38 = torch._C._nn.linear(
            hidden_states_44,
            l_self_modules_layers_modules_3_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_3_modules_fc1_parameters_bias_,
        )
        l_self_modules_layers_modules_3_modules_fc1_parameters_weight_ = (
            l_self_modules_layers_modules_3_modules_fc1_parameters_bias_
        ) = None
        hidden_states_45 = torch._C._nn.gelu(linear_38)
        linear_38 = None
        hidden_states_46 = torch.nn.functional.dropout(
            hidden_states_45, p=0, training=False
        )
        hidden_states_45 = None
        hidden_states_47 = torch._C._nn.linear(
            hidden_states_46,
            l_self_modules_layers_modules_3_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_3_modules_fc2_parameters_bias_,
        )
        hidden_states_46 = (
            l_self_modules_layers_modules_3_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_3_modules_fc2_parameters_bias_ = None
        hidden_states_48 = torch.nn.functional.dropout(
            hidden_states_47, p=0.1, training=False
        )
        hidden_states_47 = None
        hidden_states_49 = hidden_states_44 + hidden_states_48
        hidden_states_44 = hidden_states_48 = None
        hidden_states_50 = torch.nn.functional.layer_norm(
            hidden_states_49,
            (1024,),
            l_self_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_49 = (
            l_self_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_
        ) = (
            l_self_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_
        ) = None
        linear_40 = torch._C._nn.linear(
            hidden_states_50,
            l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_24 = linear_40.view(1, 11, -1, 64)
        linear_40 = None
        query_states_8 = view_24.transpose(1, 2)
        view_24 = None
        key_states_16 = torch._C._nn.linear(
            hidden_states_50,
            l_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_16 = torch._C._nn.linear(
            hidden_states_50,
            l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_25 = key_states_16.view(1, 11, -1, 64)
        key_states_16 = None
        key_states_17 = view_25.transpose(1, 2)
        view_25 = None
        view_26 = value_states_16.view(1, 11, -1, 64)
        value_states_16 = None
        value_states_17 = view_26.transpose(1, 2)
        view_26 = None
        attention_mask_8 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 11, None),
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
        reshape_9 = attn_output_33.reshape(1, 11, -1)
        attn_output_33 = None
        attn_output_34 = reshape_9.contiguous()
        reshape_9 = None
        attn_output_35 = torch._C._nn.linear(
            attn_output_34,
            l_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_34 = l_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_51 = torch.nn.functional.dropout(
            attn_output_35, p=0.1, training=False
        )
        attn_output_35 = None
        hidden_states_52 = hidden_states_50 + hidden_states_51
        hidden_states_50 = hidden_states_51 = None
        hidden_states_53 = torch.nn.functional.layer_norm(
            hidden_states_52,
            (1024,),
            l_self_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_52 = l_self_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_44 = torch._C._nn.linear(
            hidden_states_53,
            l_self_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        view_27 = linear_44.view(1, 11, -1, 64)
        linear_44 = None
        query_states_9 = view_27.transpose(1, 2)
        view_27 = None
        key_states_18 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_18 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_28 = key_states_18.view(1, 11, -1, 64)
        key_states_18 = None
        key_states_19 = view_28.transpose(1, 2)
        view_28 = None
        view_29 = value_states_18.view(1, 11, -1, 64)
        value_states_18 = None
        value_states_19 = view_29.transpose(1, 2)
        view_29 = None
        attention_mask_9 = encoder_attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 11, None),
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
        reshape_10 = attn_output_37.reshape(1, 11, -1)
        attn_output_37 = None
        attn_output_38 = reshape_10.contiguous()
        reshape_10 = None
        attn_output_39 = torch._C._nn.linear(
            attn_output_38,
            l_self_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_38 = l_self_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_54 = torch.nn.functional.dropout(
            attn_output_39, p=0.1, training=False
        )
        attn_output_39 = None
        hidden_states_55 = hidden_states_53 + hidden_states_54
        hidden_states_53 = hidden_states_54 = None
        hidden_states_56 = torch.nn.functional.layer_norm(
            hidden_states_55,
            (1024,),
            l_self_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_55 = l_self_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_48 = torch._C._nn.linear(
            hidden_states_56,
            l_self_modules_layers_modules_4_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_4_modules_fc1_parameters_bias_,
        )
        l_self_modules_layers_modules_4_modules_fc1_parameters_weight_ = (
            l_self_modules_layers_modules_4_modules_fc1_parameters_bias_
        ) = None
        hidden_states_57 = torch._C._nn.gelu(linear_48)
        linear_48 = None
        hidden_states_58 = torch.nn.functional.dropout(
            hidden_states_57, p=0, training=False
        )
        hidden_states_57 = None
        hidden_states_59 = torch._C._nn.linear(
            hidden_states_58,
            l_self_modules_layers_modules_4_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_4_modules_fc2_parameters_bias_,
        )
        hidden_states_58 = (
            l_self_modules_layers_modules_4_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_4_modules_fc2_parameters_bias_ = None
        hidden_states_60 = torch.nn.functional.dropout(
            hidden_states_59, p=0.1, training=False
        )
        hidden_states_59 = None
        hidden_states_61 = hidden_states_56 + hidden_states_60
        hidden_states_56 = hidden_states_60 = None
        hidden_states_62 = torch.nn.functional.layer_norm(
            hidden_states_61,
            (1024,),
            l_self_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_61 = (
            l_self_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_
        ) = (
            l_self_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_
        ) = None
        linear_50 = torch._C._nn.linear(
            hidden_states_62,
            l_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_30 = linear_50.view(1, 11, -1, 64)
        linear_50 = None
        query_states_10 = view_30.transpose(1, 2)
        view_30 = None
        key_states_20 = torch._C._nn.linear(
            hidden_states_62,
            l_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_20 = torch._C._nn.linear(
            hidden_states_62,
            l_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_31 = key_states_20.view(1, 11, -1, 64)
        key_states_20 = None
        key_states_21 = view_31.transpose(1, 2)
        view_31 = None
        view_32 = value_states_20.view(1, 11, -1, 64)
        value_states_20 = None
        value_states_21 = view_32.transpose(1, 2)
        view_32 = None
        attention_mask_10 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 11, None),
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
        reshape_11 = attn_output_41.reshape(1, 11, -1)
        attn_output_41 = None
        attn_output_42 = reshape_11.contiguous()
        reshape_11 = None
        attn_output_43 = torch._C._nn.linear(
            attn_output_42,
            l_self_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_42 = l_self_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_63 = torch.nn.functional.dropout(
            attn_output_43, p=0.1, training=False
        )
        attn_output_43 = None
        hidden_states_64 = hidden_states_62 + hidden_states_63
        hidden_states_62 = hidden_states_63 = None
        hidden_states_65 = torch.nn.functional.layer_norm(
            hidden_states_64,
            (1024,),
            l_self_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_64 = l_self_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_54 = torch._C._nn.linear(
            hidden_states_65,
            l_self_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        view_33 = linear_54.view(1, 11, -1, 64)
        linear_54 = None
        query_states_11 = view_33.transpose(1, 2)
        view_33 = None
        key_states_22 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_22 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_34 = key_states_22.view(1, 11, -1, 64)
        key_states_22 = None
        key_states_23 = view_34.transpose(1, 2)
        view_34 = None
        view_35 = value_states_22.view(1, 11, -1, 64)
        value_states_22 = None
        value_states_23 = view_35.transpose(1, 2)
        view_35 = None
        attention_mask_11 = encoder_attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 11, None),
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
        reshape_12 = attn_output_45.reshape(1, 11, -1)
        attn_output_45 = None
        attn_output_46 = reshape_12.contiguous()
        reshape_12 = None
        attn_output_47 = torch._C._nn.linear(
            attn_output_46,
            l_self_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_46 = l_self_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_66 = torch.nn.functional.dropout(
            attn_output_47, p=0.1, training=False
        )
        attn_output_47 = None
        hidden_states_67 = hidden_states_65 + hidden_states_66
        hidden_states_65 = hidden_states_66 = None
        hidden_states_68 = torch.nn.functional.layer_norm(
            hidden_states_67,
            (1024,),
            l_self_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_67 = l_self_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_58 = torch._C._nn.linear(
            hidden_states_68,
            l_self_modules_layers_modules_5_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_5_modules_fc1_parameters_bias_,
        )
        l_self_modules_layers_modules_5_modules_fc1_parameters_weight_ = (
            l_self_modules_layers_modules_5_modules_fc1_parameters_bias_
        ) = None
        hidden_states_69 = torch._C._nn.gelu(linear_58)
        linear_58 = None
        hidden_states_70 = torch.nn.functional.dropout(
            hidden_states_69, p=0, training=False
        )
        hidden_states_69 = None
        hidden_states_71 = torch._C._nn.linear(
            hidden_states_70,
            l_self_modules_layers_modules_5_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_5_modules_fc2_parameters_bias_,
        )
        hidden_states_70 = (
            l_self_modules_layers_modules_5_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_5_modules_fc2_parameters_bias_ = None
        hidden_states_72 = torch.nn.functional.dropout(
            hidden_states_71, p=0.1, training=False
        )
        hidden_states_71 = None
        hidden_states_73 = hidden_states_68 + hidden_states_72
        hidden_states_68 = hidden_states_72 = None
        hidden_states_74 = torch.nn.functional.layer_norm(
            hidden_states_73,
            (1024,),
            l_self_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_73 = (
            l_self_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_
        ) = (
            l_self_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_
        ) = None
        linear_60 = torch._C._nn.linear(
            hidden_states_74,
            l_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_36 = linear_60.view(1, 11, -1, 64)
        linear_60 = None
        query_states_12 = view_36.transpose(1, 2)
        view_36 = None
        key_states_24 = torch._C._nn.linear(
            hidden_states_74,
            l_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_24 = torch._C._nn.linear(
            hidden_states_74,
            l_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_37 = key_states_24.view(1, 11, -1, 64)
        key_states_24 = None
        key_states_25 = view_37.transpose(1, 2)
        view_37 = None
        view_38 = value_states_24.view(1, 11, -1, 64)
        value_states_24 = None
        value_states_25 = view_38.transpose(1, 2)
        view_38 = None
        attention_mask_12 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 11, None),
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
        reshape_13 = attn_output_49.reshape(1, 11, -1)
        attn_output_49 = None
        attn_output_50 = reshape_13.contiguous()
        reshape_13 = None
        attn_output_51 = torch._C._nn.linear(
            attn_output_50,
            l_self_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_50 = l_self_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_75 = torch.nn.functional.dropout(
            attn_output_51, p=0.1, training=False
        )
        attn_output_51 = None
        hidden_states_76 = hidden_states_74 + hidden_states_75
        hidden_states_74 = hidden_states_75 = None
        hidden_states_77 = torch.nn.functional.layer_norm(
            hidden_states_76,
            (1024,),
            l_self_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_76 = l_self_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_64 = torch._C._nn.linear(
            hidden_states_77,
            l_self_modules_layers_modules_6_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_6_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_6_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_6_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        view_39 = linear_64.view(1, 11, -1, 64)
        linear_64 = None
        query_states_13 = view_39.transpose(1, 2)
        view_39 = None
        key_states_26 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_layers_modules_6_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_6_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_6_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_6_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_26 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_layers_modules_6_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_6_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_6_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_6_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_40 = key_states_26.view(1, 11, -1, 64)
        key_states_26 = None
        key_states_27 = view_40.transpose(1, 2)
        view_40 = None
        view_41 = value_states_26.view(1, 11, -1, 64)
        value_states_26 = None
        value_states_27 = view_41.transpose(1, 2)
        view_41 = None
        attention_mask_13 = encoder_attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 11, None),
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
        reshape_14 = attn_output_53.reshape(1, 11, -1)
        attn_output_53 = None
        attn_output_54 = reshape_14.contiguous()
        reshape_14 = None
        attn_output_55 = torch._C._nn.linear(
            attn_output_54,
            l_self_modules_layers_modules_6_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_6_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_54 = l_self_modules_layers_modules_6_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_6_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_78 = torch.nn.functional.dropout(
            attn_output_55, p=0.1, training=False
        )
        attn_output_55 = None
        hidden_states_79 = hidden_states_77 + hidden_states_78
        hidden_states_77 = hidden_states_78 = None
        hidden_states_80 = torch.nn.functional.layer_norm(
            hidden_states_79,
            (1024,),
            l_self_modules_layers_modules_6_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_6_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_79 = l_self_modules_layers_modules_6_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_6_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_68 = torch._C._nn.linear(
            hidden_states_80,
            l_self_modules_layers_modules_6_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_6_modules_fc1_parameters_bias_,
        )
        l_self_modules_layers_modules_6_modules_fc1_parameters_weight_ = (
            l_self_modules_layers_modules_6_modules_fc1_parameters_bias_
        ) = None
        hidden_states_81 = torch._C._nn.gelu(linear_68)
        linear_68 = None
        hidden_states_82 = torch.nn.functional.dropout(
            hidden_states_81, p=0, training=False
        )
        hidden_states_81 = None
        hidden_states_83 = torch._C._nn.linear(
            hidden_states_82,
            l_self_modules_layers_modules_6_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_6_modules_fc2_parameters_bias_,
        )
        hidden_states_82 = (
            l_self_modules_layers_modules_6_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_6_modules_fc2_parameters_bias_ = None
        hidden_states_84 = torch.nn.functional.dropout(
            hidden_states_83, p=0.1, training=False
        )
        hidden_states_83 = None
        hidden_states_85 = hidden_states_80 + hidden_states_84
        hidden_states_80 = hidden_states_84 = None
        hidden_states_86 = torch.nn.functional.layer_norm(
            hidden_states_85,
            (1024,),
            l_self_modules_layers_modules_6_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_6_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_85 = (
            l_self_modules_layers_modules_6_modules_final_layer_norm_parameters_weight_
        ) = (
            l_self_modules_layers_modules_6_modules_final_layer_norm_parameters_bias_
        ) = None
        linear_70 = torch._C._nn.linear(
            hidden_states_86,
            l_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_42 = linear_70.view(1, 11, -1, 64)
        linear_70 = None
        query_states_14 = view_42.transpose(1, 2)
        view_42 = None
        key_states_28 = torch._C._nn.linear(
            hidden_states_86,
            l_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_28 = torch._C._nn.linear(
            hidden_states_86,
            l_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_43 = key_states_28.view(1, 11, -1, 64)
        key_states_28 = None
        key_states_29 = view_43.transpose(1, 2)
        view_43 = None
        view_44 = value_states_28.view(1, 11, -1, 64)
        value_states_28 = None
        value_states_29 = view_44.transpose(1, 2)
        view_44 = None
        attention_mask_14 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 11, None),
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
        reshape_15 = attn_output_57.reshape(1, 11, -1)
        attn_output_57 = None
        attn_output_58 = reshape_15.contiguous()
        reshape_15 = None
        attn_output_59 = torch._C._nn.linear(
            attn_output_58,
            l_self_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_58 = l_self_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_87 = torch.nn.functional.dropout(
            attn_output_59, p=0.1, training=False
        )
        attn_output_59 = None
        hidden_states_88 = hidden_states_86 + hidden_states_87
        hidden_states_86 = hidden_states_87 = None
        hidden_states_89 = torch.nn.functional.layer_norm(
            hidden_states_88,
            (1024,),
            l_self_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_88 = l_self_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_74 = torch._C._nn.linear(
            hidden_states_89,
            l_self_modules_layers_modules_7_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_7_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_7_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_7_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        view_45 = linear_74.view(1, 11, -1, 64)
        linear_74 = None
        query_states_15 = view_45.transpose(1, 2)
        view_45 = None
        key_states_30 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_layers_modules_7_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_7_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_7_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_7_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_30 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_layers_modules_7_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_7_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_7_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_7_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_46 = key_states_30.view(1, 11, -1, 64)
        key_states_30 = None
        key_states_31 = view_46.transpose(1, 2)
        view_46 = None
        view_47 = value_states_30.view(1, 11, -1, 64)
        value_states_30 = None
        value_states_31 = view_47.transpose(1, 2)
        view_47 = None
        attention_mask_15 = encoder_attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 11, None),
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
        reshape_16 = attn_output_61.reshape(1, 11, -1)
        attn_output_61 = None
        attn_output_62 = reshape_16.contiguous()
        reshape_16 = None
        attn_output_63 = torch._C._nn.linear(
            attn_output_62,
            l_self_modules_layers_modules_7_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_7_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_62 = l_self_modules_layers_modules_7_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_7_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_90 = torch.nn.functional.dropout(
            attn_output_63, p=0.1, training=False
        )
        attn_output_63 = None
        hidden_states_91 = hidden_states_89 + hidden_states_90
        hidden_states_89 = hidden_states_90 = None
        hidden_states_92 = torch.nn.functional.layer_norm(
            hidden_states_91,
            (1024,),
            l_self_modules_layers_modules_7_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_7_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_91 = l_self_modules_layers_modules_7_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_7_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_78 = torch._C._nn.linear(
            hidden_states_92,
            l_self_modules_layers_modules_7_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_7_modules_fc1_parameters_bias_,
        )
        l_self_modules_layers_modules_7_modules_fc1_parameters_weight_ = (
            l_self_modules_layers_modules_7_modules_fc1_parameters_bias_
        ) = None
        hidden_states_93 = torch._C._nn.gelu(linear_78)
        linear_78 = None
        hidden_states_94 = torch.nn.functional.dropout(
            hidden_states_93, p=0, training=False
        )
        hidden_states_93 = None
        hidden_states_95 = torch._C._nn.linear(
            hidden_states_94,
            l_self_modules_layers_modules_7_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_7_modules_fc2_parameters_bias_,
        )
        hidden_states_94 = (
            l_self_modules_layers_modules_7_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_7_modules_fc2_parameters_bias_ = None
        hidden_states_96 = torch.nn.functional.dropout(
            hidden_states_95, p=0.1, training=False
        )
        hidden_states_95 = None
        hidden_states_97 = hidden_states_92 + hidden_states_96
        hidden_states_92 = hidden_states_96 = None
        hidden_states_98 = torch.nn.functional.layer_norm(
            hidden_states_97,
            (1024,),
            l_self_modules_layers_modules_7_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_7_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_97 = (
            l_self_modules_layers_modules_7_modules_final_layer_norm_parameters_weight_
        ) = (
            l_self_modules_layers_modules_7_modules_final_layer_norm_parameters_bias_
        ) = None
        linear_80 = torch._C._nn.linear(
            hidden_states_98,
            l_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_48 = linear_80.view(1, 11, -1, 64)
        linear_80 = None
        query_states_16 = view_48.transpose(1, 2)
        view_48 = None
        key_states_32 = torch._C._nn.linear(
            hidden_states_98,
            l_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_32 = torch._C._nn.linear(
            hidden_states_98,
            l_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_49 = key_states_32.view(1, 11, -1, 64)
        key_states_32 = None
        key_states_33 = view_49.transpose(1, 2)
        view_49 = None
        view_50 = value_states_32.view(1, 11, -1, 64)
        value_states_32 = None
        value_states_33 = view_50.transpose(1, 2)
        view_50 = None
        attention_mask_16 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 11, None),
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
        reshape_17 = attn_output_65.reshape(1, 11, -1)
        attn_output_65 = None
        attn_output_66 = reshape_17.contiguous()
        reshape_17 = None
        attn_output_67 = torch._C._nn.linear(
            attn_output_66,
            l_self_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_66 = l_self_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_99 = torch.nn.functional.dropout(
            attn_output_67, p=0.1, training=False
        )
        attn_output_67 = None
        hidden_states_100 = hidden_states_98 + hidden_states_99
        hidden_states_98 = hidden_states_99 = None
        hidden_states_101 = torch.nn.functional.layer_norm(
            hidden_states_100,
            (1024,),
            l_self_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_100 = l_self_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_84 = torch._C._nn.linear(
            hidden_states_101,
            l_self_modules_layers_modules_8_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_8_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_8_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_8_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        view_51 = linear_84.view(1, 11, -1, 64)
        linear_84 = None
        query_states_17 = view_51.transpose(1, 2)
        view_51 = None
        key_states_34 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_layers_modules_8_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_8_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_8_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_8_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_34 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_layers_modules_8_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_8_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_8_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_8_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_52 = key_states_34.view(1, 11, -1, 64)
        key_states_34 = None
        key_states_35 = view_52.transpose(1, 2)
        view_52 = None
        view_53 = value_states_34.view(1, 11, -1, 64)
        value_states_34 = None
        value_states_35 = view_53.transpose(1, 2)
        view_53 = None
        attention_mask_17 = encoder_attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 11, None),
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
        reshape_18 = attn_output_69.reshape(1, 11, -1)
        attn_output_69 = None
        attn_output_70 = reshape_18.contiguous()
        reshape_18 = None
        attn_output_71 = torch._C._nn.linear(
            attn_output_70,
            l_self_modules_layers_modules_8_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_8_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_70 = l_self_modules_layers_modules_8_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_8_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_102 = torch.nn.functional.dropout(
            attn_output_71, p=0.1, training=False
        )
        attn_output_71 = None
        hidden_states_103 = hidden_states_101 + hidden_states_102
        hidden_states_101 = hidden_states_102 = None
        hidden_states_104 = torch.nn.functional.layer_norm(
            hidden_states_103,
            (1024,),
            l_self_modules_layers_modules_8_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_8_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_103 = l_self_modules_layers_modules_8_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_8_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_88 = torch._C._nn.linear(
            hidden_states_104,
            l_self_modules_layers_modules_8_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_8_modules_fc1_parameters_bias_,
        )
        l_self_modules_layers_modules_8_modules_fc1_parameters_weight_ = (
            l_self_modules_layers_modules_8_modules_fc1_parameters_bias_
        ) = None
        hidden_states_105 = torch._C._nn.gelu(linear_88)
        linear_88 = None
        hidden_states_106 = torch.nn.functional.dropout(
            hidden_states_105, p=0, training=False
        )
        hidden_states_105 = None
        hidden_states_107 = torch._C._nn.linear(
            hidden_states_106,
            l_self_modules_layers_modules_8_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_8_modules_fc2_parameters_bias_,
        )
        hidden_states_106 = (
            l_self_modules_layers_modules_8_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_8_modules_fc2_parameters_bias_ = None
        hidden_states_108 = torch.nn.functional.dropout(
            hidden_states_107, p=0.1, training=False
        )
        hidden_states_107 = None
        hidden_states_109 = hidden_states_104 + hidden_states_108
        hidden_states_104 = hidden_states_108 = None
        hidden_states_110 = torch.nn.functional.layer_norm(
            hidden_states_109,
            (1024,),
            l_self_modules_layers_modules_8_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_8_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_109 = (
            l_self_modules_layers_modules_8_modules_final_layer_norm_parameters_weight_
        ) = (
            l_self_modules_layers_modules_8_modules_final_layer_norm_parameters_bias_
        ) = None
        linear_90 = torch._C._nn.linear(
            hidden_states_110,
            l_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_54 = linear_90.view(1, 11, -1, 64)
        linear_90 = None
        query_states_18 = view_54.transpose(1, 2)
        view_54 = None
        key_states_36 = torch._C._nn.linear(
            hidden_states_110,
            l_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_36 = torch._C._nn.linear(
            hidden_states_110,
            l_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_55 = key_states_36.view(1, 11, -1, 64)
        key_states_36 = None
        key_states_37 = view_55.transpose(1, 2)
        view_55 = None
        view_56 = value_states_36.view(1, 11, -1, 64)
        value_states_36 = None
        value_states_37 = view_56.transpose(1, 2)
        view_56 = None
        attention_mask_18 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 11, None),
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
        reshape_19 = attn_output_73.reshape(1, 11, -1)
        attn_output_73 = None
        attn_output_74 = reshape_19.contiguous()
        reshape_19 = None
        attn_output_75 = torch._C._nn.linear(
            attn_output_74,
            l_self_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_74 = l_self_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_111 = torch.nn.functional.dropout(
            attn_output_75, p=0.1, training=False
        )
        attn_output_75 = None
        hidden_states_112 = hidden_states_110 + hidden_states_111
        hidden_states_110 = hidden_states_111 = None
        hidden_states_113 = torch.nn.functional.layer_norm(
            hidden_states_112,
            (1024,),
            l_self_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_112 = l_self_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_94 = torch._C._nn.linear(
            hidden_states_113,
            l_self_modules_layers_modules_9_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_9_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_9_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_9_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        view_57 = linear_94.view(1, 11, -1, 64)
        linear_94 = None
        query_states_19 = view_57.transpose(1, 2)
        view_57 = None
        key_states_38 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_layers_modules_9_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_9_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_9_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_9_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_38 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_layers_modules_9_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_9_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_9_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_9_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_58 = key_states_38.view(1, 11, -1, 64)
        key_states_38 = None
        key_states_39 = view_58.transpose(1, 2)
        view_58 = None
        view_59 = value_states_38.view(1, 11, -1, 64)
        value_states_38 = None
        value_states_39 = view_59.transpose(1, 2)
        view_59 = None
        attention_mask_19 = encoder_attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 11, None),
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
        reshape_20 = attn_output_77.reshape(1, 11, -1)
        attn_output_77 = None
        attn_output_78 = reshape_20.contiguous()
        reshape_20 = None
        attn_output_79 = torch._C._nn.linear(
            attn_output_78,
            l_self_modules_layers_modules_9_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_9_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_78 = l_self_modules_layers_modules_9_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_9_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_114 = torch.nn.functional.dropout(
            attn_output_79, p=0.1, training=False
        )
        attn_output_79 = None
        hidden_states_115 = hidden_states_113 + hidden_states_114
        hidden_states_113 = hidden_states_114 = None
        hidden_states_116 = torch.nn.functional.layer_norm(
            hidden_states_115,
            (1024,),
            l_self_modules_layers_modules_9_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_9_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_115 = l_self_modules_layers_modules_9_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_9_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_98 = torch._C._nn.linear(
            hidden_states_116,
            l_self_modules_layers_modules_9_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_9_modules_fc1_parameters_bias_,
        )
        l_self_modules_layers_modules_9_modules_fc1_parameters_weight_ = (
            l_self_modules_layers_modules_9_modules_fc1_parameters_bias_
        ) = None
        hidden_states_117 = torch._C._nn.gelu(linear_98)
        linear_98 = None
        hidden_states_118 = torch.nn.functional.dropout(
            hidden_states_117, p=0, training=False
        )
        hidden_states_117 = None
        hidden_states_119 = torch._C._nn.linear(
            hidden_states_118,
            l_self_modules_layers_modules_9_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_9_modules_fc2_parameters_bias_,
        )
        hidden_states_118 = (
            l_self_modules_layers_modules_9_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_9_modules_fc2_parameters_bias_ = None
        hidden_states_120 = torch.nn.functional.dropout(
            hidden_states_119, p=0.1, training=False
        )
        hidden_states_119 = None
        hidden_states_121 = hidden_states_116 + hidden_states_120
        hidden_states_116 = hidden_states_120 = None
        hidden_states_122 = torch.nn.functional.layer_norm(
            hidden_states_121,
            (1024,),
            l_self_modules_layers_modules_9_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_9_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_121 = (
            l_self_modules_layers_modules_9_modules_final_layer_norm_parameters_weight_
        ) = (
            l_self_modules_layers_modules_9_modules_final_layer_norm_parameters_bias_
        ) = None
        linear_100 = torch._C._nn.linear(
            hidden_states_122,
            l_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_60 = linear_100.view(1, 11, -1, 64)
        linear_100 = None
        query_states_20 = view_60.transpose(1, 2)
        view_60 = None
        key_states_40 = torch._C._nn.linear(
            hidden_states_122,
            l_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_40 = torch._C._nn.linear(
            hidden_states_122,
            l_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_61 = key_states_40.view(1, 11, -1, 64)
        key_states_40 = None
        key_states_41 = view_61.transpose(1, 2)
        view_61 = None
        view_62 = value_states_40.view(1, 11, -1, 64)
        value_states_40 = None
        value_states_41 = view_62.transpose(1, 2)
        view_62 = None
        attention_mask_20 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 11, None),
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
        reshape_21 = attn_output_81.reshape(1, 11, -1)
        attn_output_81 = None
        attn_output_82 = reshape_21.contiguous()
        reshape_21 = None
        attn_output_83 = torch._C._nn.linear(
            attn_output_82,
            l_self_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_82 = l_self_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_123 = torch.nn.functional.dropout(
            attn_output_83, p=0.1, training=False
        )
        attn_output_83 = None
        hidden_states_124 = hidden_states_122 + hidden_states_123
        hidden_states_122 = hidden_states_123 = None
        hidden_states_125 = torch.nn.functional.layer_norm(
            hidden_states_124,
            (1024,),
            l_self_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_124 = l_self_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_104 = torch._C._nn.linear(
            hidden_states_125,
            l_self_modules_layers_modules_10_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_10_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_10_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_10_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        view_63 = linear_104.view(1, 11, -1, 64)
        linear_104 = None
        query_states_21 = view_63.transpose(1, 2)
        view_63 = None
        key_states_42 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_layers_modules_10_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_10_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_10_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_10_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_42 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_layers_modules_10_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_10_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_10_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_10_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_64 = key_states_42.view(1, 11, -1, 64)
        key_states_42 = None
        key_states_43 = view_64.transpose(1, 2)
        view_64 = None
        view_65 = value_states_42.view(1, 11, -1, 64)
        value_states_42 = None
        value_states_43 = view_65.transpose(1, 2)
        view_65 = None
        attention_mask_21 = encoder_attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 11, None),
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
        reshape_22 = attn_output_85.reshape(1, 11, -1)
        attn_output_85 = None
        attn_output_86 = reshape_22.contiguous()
        reshape_22 = None
        attn_output_87 = torch._C._nn.linear(
            attn_output_86,
            l_self_modules_layers_modules_10_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_10_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_86 = l_self_modules_layers_modules_10_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_10_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_126 = torch.nn.functional.dropout(
            attn_output_87, p=0.1, training=False
        )
        attn_output_87 = None
        hidden_states_127 = hidden_states_125 + hidden_states_126
        hidden_states_125 = hidden_states_126 = None
        hidden_states_128 = torch.nn.functional.layer_norm(
            hidden_states_127,
            (1024,),
            l_self_modules_layers_modules_10_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_10_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_127 = l_self_modules_layers_modules_10_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_10_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_108 = torch._C._nn.linear(
            hidden_states_128,
            l_self_modules_layers_modules_10_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_10_modules_fc1_parameters_bias_,
        )
        l_self_modules_layers_modules_10_modules_fc1_parameters_weight_ = (
            l_self_modules_layers_modules_10_modules_fc1_parameters_bias_
        ) = None
        hidden_states_129 = torch._C._nn.gelu(linear_108)
        linear_108 = None
        hidden_states_130 = torch.nn.functional.dropout(
            hidden_states_129, p=0, training=False
        )
        hidden_states_129 = None
        hidden_states_131 = torch._C._nn.linear(
            hidden_states_130,
            l_self_modules_layers_modules_10_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_10_modules_fc2_parameters_bias_,
        )
        hidden_states_130 = (
            l_self_modules_layers_modules_10_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_10_modules_fc2_parameters_bias_ = None
        hidden_states_132 = torch.nn.functional.dropout(
            hidden_states_131, p=0.1, training=False
        )
        hidden_states_131 = None
        hidden_states_133 = hidden_states_128 + hidden_states_132
        hidden_states_128 = hidden_states_132 = None
        hidden_states_134 = torch.nn.functional.layer_norm(
            hidden_states_133,
            (1024,),
            l_self_modules_layers_modules_10_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_10_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_133 = (
            l_self_modules_layers_modules_10_modules_final_layer_norm_parameters_weight_
        ) = (
            l_self_modules_layers_modules_10_modules_final_layer_norm_parameters_bias_
        ) = None
        linear_110 = torch._C._nn.linear(
            hidden_states_134,
            l_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_66 = linear_110.view(1, 11, -1, 64)
        linear_110 = None
        query_states_22 = view_66.transpose(1, 2)
        view_66 = None
        key_states_44 = torch._C._nn.linear(
            hidden_states_134,
            l_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_44 = torch._C._nn.linear(
            hidden_states_134,
            l_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_67 = key_states_44.view(1, 11, -1, 64)
        key_states_44 = None
        key_states_45 = view_67.transpose(1, 2)
        view_67 = None
        view_68 = value_states_44.view(1, 11, -1, 64)
        value_states_44 = None
        value_states_45 = view_68.transpose(1, 2)
        view_68 = None
        attention_mask_22 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 11, None),
            )
        ]
        causal_mask_3 = None
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
        reshape_23 = attn_output_89.reshape(1, 11, -1)
        attn_output_89 = None
        attn_output_90 = reshape_23.contiguous()
        reshape_23 = None
        attn_output_91 = torch._C._nn.linear(
            attn_output_90,
            l_self_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_90 = l_self_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_135 = torch.nn.functional.dropout(
            attn_output_91, p=0.1, training=False
        )
        attn_output_91 = None
        hidden_states_136 = hidden_states_134 + hidden_states_135
        hidden_states_134 = hidden_states_135 = None
        hidden_states_137 = torch.nn.functional.layer_norm(
            hidden_states_136,
            (1024,),
            l_self_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_136 = l_self_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_114 = torch._C._nn.linear(
            hidden_states_137,
            l_self_modules_layers_modules_11_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_11_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_11_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_11_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        view_69 = linear_114.view(1, 11, -1, 64)
        linear_114 = None
        query_states_23 = view_69.transpose(1, 2)
        view_69 = None
        key_states_46 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_layers_modules_11_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_11_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_11_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_11_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_46 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_layers_modules_11_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_11_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_encoder_hidden_states_ = l_self_modules_layers_modules_11_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_11_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_70 = key_states_46.view(1, 11, -1, 64)
        key_states_46 = None
        key_states_47 = view_70.transpose(1, 2)
        view_70 = None
        view_71 = value_states_46.view(1, 11, -1, 64)
        value_states_46 = None
        value_states_47 = view_71.transpose(1, 2)
        view_71 = None
        attention_mask_23 = encoder_attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 11, None),
            )
        ]
        encoder_attention_mask = None
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
        reshape_24 = attn_output_93.reshape(1, 11, -1)
        attn_output_93 = None
        attn_output_94 = reshape_24.contiguous()
        reshape_24 = None
        attn_output_95 = torch._C._nn.linear(
            attn_output_94,
            l_self_modules_layers_modules_11_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_11_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_94 = l_self_modules_layers_modules_11_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_11_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_138 = torch.nn.functional.dropout(
            attn_output_95, p=0.1, training=False
        )
        attn_output_95 = None
        hidden_states_139 = hidden_states_137 + hidden_states_138
        hidden_states_137 = hidden_states_138 = None
        hidden_states_140 = torch.nn.functional.layer_norm(
            hidden_states_139,
            (1024,),
            l_self_modules_layers_modules_11_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_11_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_139 = l_self_modules_layers_modules_11_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_11_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_118 = torch._C._nn.linear(
            hidden_states_140,
            l_self_modules_layers_modules_11_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_11_modules_fc1_parameters_bias_,
        )
        l_self_modules_layers_modules_11_modules_fc1_parameters_weight_ = (
            l_self_modules_layers_modules_11_modules_fc1_parameters_bias_
        ) = None
        hidden_states_141 = torch._C._nn.gelu(linear_118)
        linear_118 = None
        hidden_states_142 = torch.nn.functional.dropout(
            hidden_states_141, p=0, training=False
        )
        hidden_states_141 = None
        hidden_states_143 = torch._C._nn.linear(
            hidden_states_142,
            l_self_modules_layers_modules_11_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_11_modules_fc2_parameters_bias_,
        )
        hidden_states_142 = (
            l_self_modules_layers_modules_11_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_11_modules_fc2_parameters_bias_ = None
        hidden_states_144 = torch.nn.functional.dropout(
            hidden_states_143, p=0.1, training=False
        )
        hidden_states_143 = None
        hidden_states_145 = hidden_states_140 + hidden_states_144
        hidden_states_140 = hidden_states_144 = None
        hidden_states_146 = torch.nn.functional.layer_norm(
            hidden_states_145,
            (1024,),
            l_self_modules_layers_modules_11_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_11_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_145 = (
            l_self_modules_layers_modules_11_modules_final_layer_norm_parameters_weight_
        ) = (
            l_self_modules_layers_modules_11_modules_final_layer_norm_parameters_bias_
        ) = None
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
            hidden_states_146,
        )
