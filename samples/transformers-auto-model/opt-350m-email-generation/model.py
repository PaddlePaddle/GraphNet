import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_kwargs_input_ids_: torch.Tensor,
        L_self_modules_model_modules_decoder_modules_embed_tokens_parameters_weight_: torch.nn.parameter.Parameter,
        L_kwargs_attention_mask_: torch.Tensor,
        L_self_modules_model_modules_decoder_modules_embed_positions_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_project_in_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_6_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_6_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_6_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_6_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_6_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_6_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_7_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_7_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_7_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_7_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_7_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_7_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_8_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_8_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_8_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_8_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_8_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_8_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_9_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_9_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_9_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_9_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_9_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_9_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_10_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_10_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_10_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_10_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_10_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_10_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_11_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_11_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_11_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_11_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_11_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_11_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_12_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_12_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_12_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_12_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_12_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_12_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_13_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_13_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_13_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_13_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_13_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_13_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_14_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_14_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_14_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_14_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_14_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_14_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_15_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_15_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_15_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_15_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_15_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_15_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_16_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_16_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_16_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_16_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_16_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_16_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_17_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_17_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_17_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_17_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_17_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_17_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_18_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_18_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_18_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_18_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_18_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_18_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_19_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_19_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_19_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_19_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_19_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_19_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_20_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_20_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_20_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_20_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_20_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_20_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_21_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_21_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_21_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_21_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_21_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_21_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_22_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_22_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_22_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_22_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_22_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_22_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_23_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_23_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_23_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_23_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_23_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_layers_modules_23_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_decoder_modules_project_out_parameters_weight_: torch.nn.parameter.Parameter,
    ):
        l_kwargs_input_ids_ = L_kwargs_input_ids_
        l_self_modules_model_modules_decoder_modules_embed_tokens_parameters_weight_ = (
            L_self_modules_model_modules_decoder_modules_embed_tokens_parameters_weight_
        )
        l_kwargs_attention_mask_ = L_kwargs_attention_mask_
        l_self_modules_model_modules_decoder_modules_embed_positions_parameters_weight_ = L_self_modules_model_modules_decoder_modules_embed_positions_parameters_weight_
        l_self_modules_model_modules_decoder_modules_project_in_parameters_weight_ = (
            L_self_modules_model_modules_decoder_modules_project_in_parameters_weight_
        )
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
        l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_6_modules_fc1_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_6_modules_fc1_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_6_modules_fc1_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_6_modules_fc1_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_6_modules_fc2_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_6_modules_fc2_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_6_modules_fc2_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_6_modules_fc2_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_6_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_6_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_6_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_6_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_7_modules_fc1_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_7_modules_fc1_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_7_modules_fc1_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_7_modules_fc1_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_7_modules_fc2_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_7_modules_fc2_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_7_modules_fc2_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_7_modules_fc2_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_7_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_7_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_7_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_7_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_8_modules_fc1_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_8_modules_fc1_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_8_modules_fc1_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_8_modules_fc1_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_8_modules_fc2_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_8_modules_fc2_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_8_modules_fc2_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_8_modules_fc2_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_8_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_8_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_8_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_8_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_9_modules_fc1_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_9_modules_fc1_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_9_modules_fc1_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_9_modules_fc1_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_9_modules_fc2_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_9_modules_fc2_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_9_modules_fc2_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_9_modules_fc2_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_9_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_9_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_9_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_9_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_10_modules_fc1_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_10_modules_fc1_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_10_modules_fc1_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_10_modules_fc1_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_10_modules_fc2_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_10_modules_fc2_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_10_modules_fc2_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_10_modules_fc2_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_10_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_10_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_10_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_10_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_11_modules_fc1_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_11_modules_fc1_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_11_modules_fc1_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_11_modules_fc1_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_11_modules_fc2_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_11_modules_fc2_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_11_modules_fc2_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_11_modules_fc2_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_11_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_11_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_11_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_11_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_12_modules_fc1_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_12_modules_fc1_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_12_modules_fc1_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_12_modules_fc1_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_12_modules_fc2_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_12_modules_fc2_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_12_modules_fc2_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_12_modules_fc2_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_12_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_12_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_12_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_12_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_13_modules_fc1_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_13_modules_fc1_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_13_modules_fc1_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_13_modules_fc1_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_13_modules_fc2_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_13_modules_fc2_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_13_modules_fc2_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_13_modules_fc2_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_13_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_13_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_13_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_13_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_14_modules_fc1_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_14_modules_fc1_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_14_modules_fc1_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_14_modules_fc1_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_14_modules_fc2_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_14_modules_fc2_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_14_modules_fc2_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_14_modules_fc2_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_14_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_14_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_14_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_14_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_15_modules_fc1_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_15_modules_fc1_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_15_modules_fc1_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_15_modules_fc1_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_15_modules_fc2_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_15_modules_fc2_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_15_modules_fc2_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_15_modules_fc2_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_15_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_15_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_15_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_15_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_16_modules_fc1_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_16_modules_fc1_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_16_modules_fc1_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_16_modules_fc1_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_16_modules_fc2_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_16_modules_fc2_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_16_modules_fc2_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_16_modules_fc2_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_16_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_16_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_16_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_16_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_17_modules_fc1_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_17_modules_fc1_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_17_modules_fc1_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_17_modules_fc1_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_17_modules_fc2_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_17_modules_fc2_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_17_modules_fc2_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_17_modules_fc2_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_17_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_17_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_17_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_17_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_18_modules_fc1_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_18_modules_fc1_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_18_modules_fc1_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_18_modules_fc1_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_18_modules_fc2_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_18_modules_fc2_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_18_modules_fc2_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_18_modules_fc2_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_18_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_18_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_18_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_18_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_19_modules_fc1_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_19_modules_fc1_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_19_modules_fc1_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_19_modules_fc1_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_19_modules_fc2_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_19_modules_fc2_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_19_modules_fc2_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_19_modules_fc2_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_19_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_19_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_19_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_19_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_20_modules_fc1_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_20_modules_fc1_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_20_modules_fc1_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_20_modules_fc1_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_20_modules_fc2_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_20_modules_fc2_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_20_modules_fc2_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_20_modules_fc2_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_20_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_20_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_20_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_20_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_21_modules_fc1_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_21_modules_fc1_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_21_modules_fc1_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_21_modules_fc1_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_21_modules_fc2_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_21_modules_fc2_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_21_modules_fc2_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_21_modules_fc2_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_21_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_21_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_21_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_21_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_22_modules_fc1_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_22_modules_fc1_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_22_modules_fc1_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_22_modules_fc1_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_22_modules_fc2_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_22_modules_fc2_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_22_modules_fc2_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_22_modules_fc2_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_22_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_22_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_22_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_22_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_23_modules_fc1_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_23_modules_fc1_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_23_modules_fc1_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_23_modules_fc1_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_23_modules_fc2_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_23_modules_fc2_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_23_modules_fc2_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_23_modules_fc2_parameters_bias_
        l_self_modules_model_modules_decoder_modules_layers_modules_23_modules_final_layer_norm_parameters_weight_ = L_self_modules_model_modules_decoder_modules_layers_modules_23_modules_final_layer_norm_parameters_weight_
        l_self_modules_model_modules_decoder_modules_layers_modules_23_modules_final_layer_norm_parameters_bias_ = L_self_modules_model_modules_decoder_modules_layers_modules_23_modules_final_layer_norm_parameters_bias_
        l_self_modules_model_modules_decoder_modules_project_out_parameters_weight_ = (
            L_self_modules_model_modules_decoder_modules_project_out_parameters_weight_
        )
        input_ids = l_kwargs_input_ids_.view(-1, 19)
        l_kwargs_input_ids_ = None
        inputs_embeds = torch.nn.functional.embedding(
            input_ids,
            l_self_modules_model_modules_decoder_modules_embed_tokens_parameters_weight_,
            1,
            None,
            2.0,
            False,
            False,
        )
        input_ids = None
        cache_position = torch.arange(0, 19, device=device(type="cpu"))
        causal_mask = torch.full(
            (19, 19),
            fill_value=-65504.0,
            dtype=torch.float16,
            device=device(type="cpu"),
        )
        causal_mask_1 = torch.triu(causal_mask, diagonal=1)
        causal_mask = None
        arange_1 = torch.arange(19, device=device(type="cpu"))
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
                slice(None, 19, None),
            )
        ]
        getitem_2 = l_kwargs_attention_mask_[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        to = getitem_2.to(device(type="cpu"))
        getitem_2 = None
        padding_mask = getitem_1 + to
        getitem_1 = to = None
        padding_mask_1 = padding_mask == 0
        padding_mask = None
        getitem_3 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        masked_fill = getitem_3.masked_fill(padding_mask_1, -65504.0)
        getitem_3 = padding_mask_1 = None
        causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ] = masked_fill
        setitem = causal_mask_4
        masked_fill = setitem = None
        position_ids = torch.cumsum(l_kwargs_attention_mask_, dim=1)
        mul = position_ids * l_kwargs_attention_mask_
        position_ids = l_kwargs_attention_mask_ = None
        sub = mul - 1
        mul = None
        position_ids_1 = sub.long()
        sub = None
        position_ids_2 = position_ids_1[(slice(None, None, None), slice(0, None, None))]
        position_ids_1 = None
        add_1 = position_ids_2 + 2
        position_ids_2 = None
        pos_embeds = torch.nn.functional.embedding(
            add_1,
            l_self_modules_model_modules_decoder_modules_embed_positions_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        add_1 = l_self_modules_model_modules_decoder_modules_embed_positions_parameters_weight_ = (None)
        inputs_embeds_1 = torch._C._nn.linear(
            inputs_embeds,
            l_self_modules_model_modules_decoder_modules_project_in_parameters_weight_,
            None,
        )
        inputs_embeds = (
            l_self_modules_model_modules_decoder_modules_project_in_parameters_weight_
        ) = None
        to_1 = pos_embeds.to(device(type="cpu"))
        pos_embeds = None
        hidden_states = inputs_embeds_1 + to_1
        inputs_embeds_1 = to_1 = None
        linear_1 = torch._C._nn.linear(
            hidden_states,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states = linear_1 * 0.125
        linear_1 = None
        view_1 = query_states.view(1, -1, 16, 64)
        query_states = None
        query_states_1 = view_1.transpose(1, 2)
        view_1 = None
        key_states = torch._C._nn.linear(
            hidden_states,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states = torch._C._nn.linear(
            hidden_states,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_2 = key_states.view(1, -1, 16, 64)
        key_states = None
        key_states_1 = view_2.transpose(1, 2)
        view_2 = None
        view_3 = value_states.view(1, -1, 16, 64)
        value_states = None
        value_states_1 = view_3.transpose(1, 2)
        view_3 = None
        attention_mask = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query = query_states_1.contiguous()
        query_states_1 = None
        key = key_states_1.contiguous()
        key_states_1 = None
        value = value_states_1.contiguous()
        value_states_1 = None
        attn_output = torch._C._nn.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            scale=1.0,
            is_causal=False,
        )
        query = key = value = attention_mask = None
        transpose_3 = attn_output.transpose(1, 2)
        attn_output = None
        attn_output_1 = transpose_3.contiguous()
        transpose_3 = None
        reshape_1 = attn_output_1.reshape(1, 19, -1)
        attn_output_1 = None
        attn_output_2 = reshape_1.contiguous()
        reshape_1 = None
        attn_output_3 = torch._C._nn.linear(
            attn_output_2,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_2 = l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_1 = torch.nn.functional.dropout(
            attn_output_3, p=0.1, training=False
        )
        attn_output_3 = None
        hidden_states_2 = hidden_states + hidden_states_1
        hidden_states = hidden_states_1 = None
        hidden_states_3 = torch.nn.functional.layer_norm(
            hidden_states_2,
            (1024,),
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_2 = l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_ = (None)
        hidden_states_4 = hidden_states_3.reshape(-1, 1024)
        hidden_states_3 = None
        hidden_states_5 = torch._C._nn.linear(
            hidden_states_4,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_bias_ = (None)
        hidden_states_6 = torch.nn.functional.relu(hidden_states_5, inplace=False)
        hidden_states_5 = None
        hidden_states_7 = torch._C._nn.linear(
            hidden_states_6,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_bias_,
        )
        hidden_states_6 = l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_bias_ = (None)
        hidden_states_8 = torch.nn.functional.dropout(
            hidden_states_7, p=0.1, training=False
        )
        hidden_states_7 = None
        add_4 = hidden_states_4 + hidden_states_8
        hidden_states_4 = hidden_states_8 = None
        hidden_states_9 = add_4.view((1, 19, 1024))
        add_4 = None
        hidden_states_10 = torch.nn.functional.layer_norm(
            hidden_states_9,
            (1024,),
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_9 = l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_ = (None)
        linear_7 = torch._C._nn.linear(
            hidden_states_10,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_2 = linear_7 * 0.125
        linear_7 = None
        view_5 = query_states_2.view(1, -1, 16, 64)
        query_states_2 = None
        query_states_3 = view_5.transpose(1, 2)
        view_5 = None
        key_states_2 = torch._C._nn.linear(
            hidden_states_10,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_2 = torch._C._nn.linear(
            hidden_states_10,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_6 = key_states_2.view(1, -1, 16, 64)
        key_states_2 = None
        key_states_3 = view_6.transpose(1, 2)
        view_6 = None
        view_7 = value_states_2.view(1, -1, 16, 64)
        value_states_2 = None
        value_states_3 = view_7.transpose(1, 2)
        view_7 = None
        attention_mask_1 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_1 = query_states_3.contiguous()
        query_states_3 = None
        key_1 = key_states_3.contiguous()
        key_states_3 = None
        value_1 = value_states_3.contiguous()
        value_states_3 = None
        attn_output_4 = torch._C._nn.scaled_dot_product_attention(
            query_1,
            key_1,
            value_1,
            attn_mask=attention_mask_1,
            dropout_p=0.0,
            scale=1.0,
            is_causal=False,
        )
        query_1 = key_1 = value_1 = attention_mask_1 = None
        transpose_7 = attn_output_4.transpose(1, 2)
        attn_output_4 = None
        attn_output_5 = transpose_7.contiguous()
        transpose_7 = None
        reshape_3 = attn_output_5.reshape(1, 19, -1)
        attn_output_5 = None
        attn_output_6 = reshape_3.contiguous()
        reshape_3 = None
        attn_output_7 = torch._C._nn.linear(
            attn_output_6,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_6 = l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_11 = torch.nn.functional.dropout(
            attn_output_7, p=0.1, training=False
        )
        attn_output_7 = None
        hidden_states_12 = hidden_states_10 + hidden_states_11
        hidden_states_10 = hidden_states_11 = None
        hidden_states_13 = torch.nn.functional.layer_norm(
            hidden_states_12,
            (1024,),
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_12 = l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_ = (None)
        hidden_states_14 = hidden_states_13.reshape(-1, 1024)
        hidden_states_13 = None
        hidden_states_15 = torch._C._nn.linear(
            hidden_states_14,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_bias_ = (None)
        hidden_states_16 = torch.nn.functional.relu(hidden_states_15, inplace=False)
        hidden_states_15 = None
        hidden_states_17 = torch._C._nn.linear(
            hidden_states_16,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_bias_,
        )
        hidden_states_16 = l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_bias_ = (None)
        hidden_states_18 = torch.nn.functional.dropout(
            hidden_states_17, p=0.1, training=False
        )
        hidden_states_17 = None
        add_6 = hidden_states_14 + hidden_states_18
        hidden_states_14 = hidden_states_18 = None
        hidden_states_19 = add_6.view((1, 19, 1024))
        add_6 = None
        hidden_states_20 = torch.nn.functional.layer_norm(
            hidden_states_19,
            (1024,),
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_19 = l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_ = (None)
        linear_13 = torch._C._nn.linear(
            hidden_states_20,
            l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_4 = linear_13 * 0.125
        linear_13 = None
        view_9 = query_states_4.view(1, -1, 16, 64)
        query_states_4 = None
        query_states_5 = view_9.transpose(1, 2)
        view_9 = None
        key_states_4 = torch._C._nn.linear(
            hidden_states_20,
            l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_4 = torch._C._nn.linear(
            hidden_states_20,
            l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_10 = key_states_4.view(1, -1, 16, 64)
        key_states_4 = None
        key_states_5 = view_10.transpose(1, 2)
        view_10 = None
        view_11 = value_states_4.view(1, -1, 16, 64)
        value_states_4 = None
        value_states_5 = view_11.transpose(1, 2)
        view_11 = None
        attention_mask_2 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_2 = query_states_5.contiguous()
        query_states_5 = None
        key_2 = key_states_5.contiguous()
        key_states_5 = None
        value_2 = value_states_5.contiguous()
        value_states_5 = None
        attn_output_8 = torch._C._nn.scaled_dot_product_attention(
            query_2,
            key_2,
            value_2,
            attn_mask=attention_mask_2,
            dropout_p=0.0,
            scale=1.0,
            is_causal=False,
        )
        query_2 = key_2 = value_2 = attention_mask_2 = None
        transpose_11 = attn_output_8.transpose(1, 2)
        attn_output_8 = None
        attn_output_9 = transpose_11.contiguous()
        transpose_11 = None
        reshape_5 = attn_output_9.reshape(1, 19, -1)
        attn_output_9 = None
        attn_output_10 = reshape_5.contiguous()
        reshape_5 = None
        attn_output_11 = torch._C._nn.linear(
            attn_output_10,
            l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_10 = l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_21 = torch.nn.functional.dropout(
            attn_output_11, p=0.1, training=False
        )
        attn_output_11 = None
        hidden_states_22 = hidden_states_20 + hidden_states_21
        hidden_states_20 = hidden_states_21 = None
        hidden_states_23 = torch.nn.functional.layer_norm(
            hidden_states_22,
            (1024,),
            l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_22 = l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_ = (None)
        hidden_states_24 = hidden_states_23.reshape(-1, 1024)
        hidden_states_23 = None
        hidden_states_25 = torch._C._nn.linear(
            hidden_states_24,
            l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_bias_ = (None)
        hidden_states_26 = torch.nn.functional.relu(hidden_states_25, inplace=False)
        hidden_states_25 = None
        hidden_states_27 = torch._C._nn.linear(
            hidden_states_26,
            l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_bias_,
        )
        hidden_states_26 = l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_bias_ = (None)
        hidden_states_28 = torch.nn.functional.dropout(
            hidden_states_27, p=0.1, training=False
        )
        hidden_states_27 = None
        add_8 = hidden_states_24 + hidden_states_28
        hidden_states_24 = hidden_states_28 = None
        hidden_states_29 = add_8.view((1, 19, 1024))
        add_8 = None
        hidden_states_30 = torch.nn.functional.layer_norm(
            hidden_states_29,
            (1024,),
            l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_29 = l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_ = (None)
        linear_19 = torch._C._nn.linear(
            hidden_states_30,
            l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_6 = linear_19 * 0.125
        linear_19 = None
        view_13 = query_states_6.view(1, -1, 16, 64)
        query_states_6 = None
        query_states_7 = view_13.transpose(1, 2)
        view_13 = None
        key_states_6 = torch._C._nn.linear(
            hidden_states_30,
            l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_6 = torch._C._nn.linear(
            hidden_states_30,
            l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_14 = key_states_6.view(1, -1, 16, 64)
        key_states_6 = None
        key_states_7 = view_14.transpose(1, 2)
        view_14 = None
        view_15 = value_states_6.view(1, -1, 16, 64)
        value_states_6 = None
        value_states_7 = view_15.transpose(1, 2)
        view_15 = None
        attention_mask_3 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_3 = query_states_7.contiguous()
        query_states_7 = None
        key_3 = key_states_7.contiguous()
        key_states_7 = None
        value_3 = value_states_7.contiguous()
        value_states_7 = None
        attn_output_12 = torch._C._nn.scaled_dot_product_attention(
            query_3,
            key_3,
            value_3,
            attn_mask=attention_mask_3,
            dropout_p=0.0,
            scale=1.0,
            is_causal=False,
        )
        query_3 = key_3 = value_3 = attention_mask_3 = None
        transpose_15 = attn_output_12.transpose(1, 2)
        attn_output_12 = None
        attn_output_13 = transpose_15.contiguous()
        transpose_15 = None
        reshape_7 = attn_output_13.reshape(1, 19, -1)
        attn_output_13 = None
        attn_output_14 = reshape_7.contiguous()
        reshape_7 = None
        attn_output_15 = torch._C._nn.linear(
            attn_output_14,
            l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_14 = l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_31 = torch.nn.functional.dropout(
            attn_output_15, p=0.1, training=False
        )
        attn_output_15 = None
        hidden_states_32 = hidden_states_30 + hidden_states_31
        hidden_states_30 = hidden_states_31 = None
        hidden_states_33 = torch.nn.functional.layer_norm(
            hidden_states_32,
            (1024,),
            l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_32 = l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_ = (None)
        hidden_states_34 = hidden_states_33.reshape(-1, 1024)
        hidden_states_33 = None
        hidden_states_35 = torch._C._nn.linear(
            hidden_states_34,
            l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_bias_ = (None)
        hidden_states_36 = torch.nn.functional.relu(hidden_states_35, inplace=False)
        hidden_states_35 = None
        hidden_states_37 = torch._C._nn.linear(
            hidden_states_36,
            l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_bias_,
        )
        hidden_states_36 = l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_bias_ = (None)
        hidden_states_38 = torch.nn.functional.dropout(
            hidden_states_37, p=0.1, training=False
        )
        hidden_states_37 = None
        add_10 = hidden_states_34 + hidden_states_38
        hidden_states_34 = hidden_states_38 = None
        hidden_states_39 = add_10.view((1, 19, 1024))
        add_10 = None
        hidden_states_40 = torch.nn.functional.layer_norm(
            hidden_states_39,
            (1024,),
            l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_39 = l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_ = (None)
        linear_25 = torch._C._nn.linear(
            hidden_states_40,
            l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_8 = linear_25 * 0.125
        linear_25 = None
        view_17 = query_states_8.view(1, -1, 16, 64)
        query_states_8 = None
        query_states_9 = view_17.transpose(1, 2)
        view_17 = None
        key_states_8 = torch._C._nn.linear(
            hidden_states_40,
            l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_8 = torch._C._nn.linear(
            hidden_states_40,
            l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_18 = key_states_8.view(1, -1, 16, 64)
        key_states_8 = None
        key_states_9 = view_18.transpose(1, 2)
        view_18 = None
        view_19 = value_states_8.view(1, -1, 16, 64)
        value_states_8 = None
        value_states_9 = view_19.transpose(1, 2)
        view_19 = None
        attention_mask_4 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_4 = query_states_9.contiguous()
        query_states_9 = None
        key_4 = key_states_9.contiguous()
        key_states_9 = None
        value_4 = value_states_9.contiguous()
        value_states_9 = None
        attn_output_16 = torch._C._nn.scaled_dot_product_attention(
            query_4,
            key_4,
            value_4,
            attn_mask=attention_mask_4,
            dropout_p=0.0,
            scale=1.0,
            is_causal=False,
        )
        query_4 = key_4 = value_4 = attention_mask_4 = None
        transpose_19 = attn_output_16.transpose(1, 2)
        attn_output_16 = None
        attn_output_17 = transpose_19.contiguous()
        transpose_19 = None
        reshape_9 = attn_output_17.reshape(1, 19, -1)
        attn_output_17 = None
        attn_output_18 = reshape_9.contiguous()
        reshape_9 = None
        attn_output_19 = torch._C._nn.linear(
            attn_output_18,
            l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_18 = l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_41 = torch.nn.functional.dropout(
            attn_output_19, p=0.1, training=False
        )
        attn_output_19 = None
        hidden_states_42 = hidden_states_40 + hidden_states_41
        hidden_states_40 = hidden_states_41 = None
        hidden_states_43 = torch.nn.functional.layer_norm(
            hidden_states_42,
            (1024,),
            l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_42 = l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_ = (None)
        hidden_states_44 = hidden_states_43.reshape(-1, 1024)
        hidden_states_43 = None
        hidden_states_45 = torch._C._nn.linear(
            hidden_states_44,
            l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_bias_ = (None)
        hidden_states_46 = torch.nn.functional.relu(hidden_states_45, inplace=False)
        hidden_states_45 = None
        hidden_states_47 = torch._C._nn.linear(
            hidden_states_46,
            l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_bias_,
        )
        hidden_states_46 = l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_bias_ = (None)
        hidden_states_48 = torch.nn.functional.dropout(
            hidden_states_47, p=0.1, training=False
        )
        hidden_states_47 = None
        add_12 = hidden_states_44 + hidden_states_48
        hidden_states_44 = hidden_states_48 = None
        hidden_states_49 = add_12.view((1, 19, 1024))
        add_12 = None
        hidden_states_50 = torch.nn.functional.layer_norm(
            hidden_states_49,
            (1024,),
            l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_49 = l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_ = (None)
        linear_31 = torch._C._nn.linear(
            hidden_states_50,
            l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_10 = linear_31 * 0.125
        linear_31 = None
        view_21 = query_states_10.view(1, -1, 16, 64)
        query_states_10 = None
        query_states_11 = view_21.transpose(1, 2)
        view_21 = None
        key_states_10 = torch._C._nn.linear(
            hidden_states_50,
            l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_10 = torch._C._nn.linear(
            hidden_states_50,
            l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_22 = key_states_10.view(1, -1, 16, 64)
        key_states_10 = None
        key_states_11 = view_22.transpose(1, 2)
        view_22 = None
        view_23 = value_states_10.view(1, -1, 16, 64)
        value_states_10 = None
        value_states_11 = view_23.transpose(1, 2)
        view_23 = None
        attention_mask_5 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_5 = query_states_11.contiguous()
        query_states_11 = None
        key_5 = key_states_11.contiguous()
        key_states_11 = None
        value_5 = value_states_11.contiguous()
        value_states_11 = None
        attn_output_20 = torch._C._nn.scaled_dot_product_attention(
            query_5,
            key_5,
            value_5,
            attn_mask=attention_mask_5,
            dropout_p=0.0,
            scale=1.0,
            is_causal=False,
        )
        query_5 = key_5 = value_5 = attention_mask_5 = None
        transpose_23 = attn_output_20.transpose(1, 2)
        attn_output_20 = None
        attn_output_21 = transpose_23.contiguous()
        transpose_23 = None
        reshape_11 = attn_output_21.reshape(1, 19, -1)
        attn_output_21 = None
        attn_output_22 = reshape_11.contiguous()
        reshape_11 = None
        attn_output_23 = torch._C._nn.linear(
            attn_output_22,
            l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_22 = l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_51 = torch.nn.functional.dropout(
            attn_output_23, p=0.1, training=False
        )
        attn_output_23 = None
        hidden_states_52 = hidden_states_50 + hidden_states_51
        hidden_states_50 = hidden_states_51 = None
        hidden_states_53 = torch.nn.functional.layer_norm(
            hidden_states_52,
            (1024,),
            l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_52 = l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_ = (None)
        hidden_states_54 = hidden_states_53.reshape(-1, 1024)
        hidden_states_53 = None
        hidden_states_55 = torch._C._nn.linear(
            hidden_states_54,
            l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_bias_ = (None)
        hidden_states_56 = torch.nn.functional.relu(hidden_states_55, inplace=False)
        hidden_states_55 = None
        hidden_states_57 = torch._C._nn.linear(
            hidden_states_56,
            l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_bias_,
        )
        hidden_states_56 = l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_bias_ = (None)
        hidden_states_58 = torch.nn.functional.dropout(
            hidden_states_57, p=0.1, training=False
        )
        hidden_states_57 = None
        add_14 = hidden_states_54 + hidden_states_58
        hidden_states_54 = hidden_states_58 = None
        hidden_states_59 = add_14.view((1, 19, 1024))
        add_14 = None
        hidden_states_60 = torch.nn.functional.layer_norm(
            hidden_states_59,
            (1024,),
            l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_59 = l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_ = (None)
        linear_37 = torch._C._nn.linear(
            hidden_states_60,
            l_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_12 = linear_37 * 0.125
        linear_37 = None
        view_25 = query_states_12.view(1, -1, 16, 64)
        query_states_12 = None
        query_states_13 = view_25.transpose(1, 2)
        view_25 = None
        key_states_12 = torch._C._nn.linear(
            hidden_states_60,
            l_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_12 = torch._C._nn.linear(
            hidden_states_60,
            l_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_26 = key_states_12.view(1, -1, 16, 64)
        key_states_12 = None
        key_states_13 = view_26.transpose(1, 2)
        view_26 = None
        view_27 = value_states_12.view(1, -1, 16, 64)
        value_states_12 = None
        value_states_13 = view_27.transpose(1, 2)
        view_27 = None
        attention_mask_6 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_6 = query_states_13.contiguous()
        query_states_13 = None
        key_6 = key_states_13.contiguous()
        key_states_13 = None
        value_6 = value_states_13.contiguous()
        value_states_13 = None
        attn_output_24 = torch._C._nn.scaled_dot_product_attention(
            query_6,
            key_6,
            value_6,
            attn_mask=attention_mask_6,
            dropout_p=0.0,
            scale=1.0,
            is_causal=False,
        )
        query_6 = key_6 = value_6 = attention_mask_6 = None
        transpose_27 = attn_output_24.transpose(1, 2)
        attn_output_24 = None
        attn_output_25 = transpose_27.contiguous()
        transpose_27 = None
        reshape_13 = attn_output_25.reshape(1, 19, -1)
        attn_output_25 = None
        attn_output_26 = reshape_13.contiguous()
        reshape_13 = None
        attn_output_27 = torch._C._nn.linear(
            attn_output_26,
            l_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_26 = l_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_61 = torch.nn.functional.dropout(
            attn_output_27, p=0.1, training=False
        )
        attn_output_27 = None
        hidden_states_62 = hidden_states_60 + hidden_states_61
        hidden_states_60 = hidden_states_61 = None
        hidden_states_63 = torch.nn.functional.layer_norm(
            hidden_states_62,
            (1024,),
            l_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_62 = l_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_bias_ = (None)
        hidden_states_64 = hidden_states_63.reshape(-1, 1024)
        hidden_states_63 = None
        hidden_states_65 = torch._C._nn.linear(
            hidden_states_64,
            l_self_modules_model_modules_decoder_modules_layers_modules_6_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_6_modules_fc1_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_6_modules_fc1_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_6_modules_fc1_parameters_bias_ = (None)
        hidden_states_66 = torch.nn.functional.relu(hidden_states_65, inplace=False)
        hidden_states_65 = None
        hidden_states_67 = torch._C._nn.linear(
            hidden_states_66,
            l_self_modules_model_modules_decoder_modules_layers_modules_6_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_6_modules_fc2_parameters_bias_,
        )
        hidden_states_66 = l_self_modules_model_modules_decoder_modules_layers_modules_6_modules_fc2_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_6_modules_fc2_parameters_bias_ = (None)
        hidden_states_68 = torch.nn.functional.dropout(
            hidden_states_67, p=0.1, training=False
        )
        hidden_states_67 = None
        add_16 = hidden_states_64 + hidden_states_68
        hidden_states_64 = hidden_states_68 = None
        hidden_states_69 = add_16.view((1, 19, 1024))
        add_16 = None
        hidden_states_70 = torch.nn.functional.layer_norm(
            hidden_states_69,
            (1024,),
            l_self_modules_model_modules_decoder_modules_layers_modules_6_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_6_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_69 = l_self_modules_model_modules_decoder_modules_layers_modules_6_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_6_modules_final_layer_norm_parameters_bias_ = (None)
        linear_43 = torch._C._nn.linear(
            hidden_states_70,
            l_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_14 = linear_43 * 0.125
        linear_43 = None
        view_29 = query_states_14.view(1, -1, 16, 64)
        query_states_14 = None
        query_states_15 = view_29.transpose(1, 2)
        view_29 = None
        key_states_14 = torch._C._nn.linear(
            hidden_states_70,
            l_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_14 = torch._C._nn.linear(
            hidden_states_70,
            l_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_30 = key_states_14.view(1, -1, 16, 64)
        key_states_14 = None
        key_states_15 = view_30.transpose(1, 2)
        view_30 = None
        view_31 = value_states_14.view(1, -1, 16, 64)
        value_states_14 = None
        value_states_15 = view_31.transpose(1, 2)
        view_31 = None
        attention_mask_7 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_7 = query_states_15.contiguous()
        query_states_15 = None
        key_7 = key_states_15.contiguous()
        key_states_15 = None
        value_7 = value_states_15.contiguous()
        value_states_15 = None
        attn_output_28 = torch._C._nn.scaled_dot_product_attention(
            query_7,
            key_7,
            value_7,
            attn_mask=attention_mask_7,
            dropout_p=0.0,
            scale=1.0,
            is_causal=False,
        )
        query_7 = key_7 = value_7 = attention_mask_7 = None
        transpose_31 = attn_output_28.transpose(1, 2)
        attn_output_28 = None
        attn_output_29 = transpose_31.contiguous()
        transpose_31 = None
        reshape_15 = attn_output_29.reshape(1, 19, -1)
        attn_output_29 = None
        attn_output_30 = reshape_15.contiguous()
        reshape_15 = None
        attn_output_31 = torch._C._nn.linear(
            attn_output_30,
            l_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_30 = l_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_71 = torch.nn.functional.dropout(
            attn_output_31, p=0.1, training=False
        )
        attn_output_31 = None
        hidden_states_72 = hidden_states_70 + hidden_states_71
        hidden_states_70 = hidden_states_71 = None
        hidden_states_73 = torch.nn.functional.layer_norm(
            hidden_states_72,
            (1024,),
            l_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_72 = l_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_bias_ = (None)
        hidden_states_74 = hidden_states_73.reshape(-1, 1024)
        hidden_states_73 = None
        hidden_states_75 = torch._C._nn.linear(
            hidden_states_74,
            l_self_modules_model_modules_decoder_modules_layers_modules_7_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_7_modules_fc1_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_7_modules_fc1_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_7_modules_fc1_parameters_bias_ = (None)
        hidden_states_76 = torch.nn.functional.relu(hidden_states_75, inplace=False)
        hidden_states_75 = None
        hidden_states_77 = torch._C._nn.linear(
            hidden_states_76,
            l_self_modules_model_modules_decoder_modules_layers_modules_7_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_7_modules_fc2_parameters_bias_,
        )
        hidden_states_76 = l_self_modules_model_modules_decoder_modules_layers_modules_7_modules_fc2_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_7_modules_fc2_parameters_bias_ = (None)
        hidden_states_78 = torch.nn.functional.dropout(
            hidden_states_77, p=0.1, training=False
        )
        hidden_states_77 = None
        add_18 = hidden_states_74 + hidden_states_78
        hidden_states_74 = hidden_states_78 = None
        hidden_states_79 = add_18.view((1, 19, 1024))
        add_18 = None
        hidden_states_80 = torch.nn.functional.layer_norm(
            hidden_states_79,
            (1024,),
            l_self_modules_model_modules_decoder_modules_layers_modules_7_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_7_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_79 = l_self_modules_model_modules_decoder_modules_layers_modules_7_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_7_modules_final_layer_norm_parameters_bias_ = (None)
        linear_49 = torch._C._nn.linear(
            hidden_states_80,
            l_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_16 = linear_49 * 0.125
        linear_49 = None
        view_33 = query_states_16.view(1, -1, 16, 64)
        query_states_16 = None
        query_states_17 = view_33.transpose(1, 2)
        view_33 = None
        key_states_16 = torch._C._nn.linear(
            hidden_states_80,
            l_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_16 = torch._C._nn.linear(
            hidden_states_80,
            l_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_34 = key_states_16.view(1, -1, 16, 64)
        key_states_16 = None
        key_states_17 = view_34.transpose(1, 2)
        view_34 = None
        view_35 = value_states_16.view(1, -1, 16, 64)
        value_states_16 = None
        value_states_17 = view_35.transpose(1, 2)
        view_35 = None
        attention_mask_8 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_8 = query_states_17.contiguous()
        query_states_17 = None
        key_8 = key_states_17.contiguous()
        key_states_17 = None
        value_8 = value_states_17.contiguous()
        value_states_17 = None
        attn_output_32 = torch._C._nn.scaled_dot_product_attention(
            query_8,
            key_8,
            value_8,
            attn_mask=attention_mask_8,
            dropout_p=0.0,
            scale=1.0,
            is_causal=False,
        )
        query_8 = key_8 = value_8 = attention_mask_8 = None
        transpose_35 = attn_output_32.transpose(1, 2)
        attn_output_32 = None
        attn_output_33 = transpose_35.contiguous()
        transpose_35 = None
        reshape_17 = attn_output_33.reshape(1, 19, -1)
        attn_output_33 = None
        attn_output_34 = reshape_17.contiguous()
        reshape_17 = None
        attn_output_35 = torch._C._nn.linear(
            attn_output_34,
            l_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_34 = l_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_81 = torch.nn.functional.dropout(
            attn_output_35, p=0.1, training=False
        )
        attn_output_35 = None
        hidden_states_82 = hidden_states_80 + hidden_states_81
        hidden_states_80 = hidden_states_81 = None
        hidden_states_83 = torch.nn.functional.layer_norm(
            hidden_states_82,
            (1024,),
            l_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_82 = l_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_bias_ = (None)
        hidden_states_84 = hidden_states_83.reshape(-1, 1024)
        hidden_states_83 = None
        hidden_states_85 = torch._C._nn.linear(
            hidden_states_84,
            l_self_modules_model_modules_decoder_modules_layers_modules_8_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_8_modules_fc1_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_8_modules_fc1_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_8_modules_fc1_parameters_bias_ = (None)
        hidden_states_86 = torch.nn.functional.relu(hidden_states_85, inplace=False)
        hidden_states_85 = None
        hidden_states_87 = torch._C._nn.linear(
            hidden_states_86,
            l_self_modules_model_modules_decoder_modules_layers_modules_8_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_8_modules_fc2_parameters_bias_,
        )
        hidden_states_86 = l_self_modules_model_modules_decoder_modules_layers_modules_8_modules_fc2_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_8_modules_fc2_parameters_bias_ = (None)
        hidden_states_88 = torch.nn.functional.dropout(
            hidden_states_87, p=0.1, training=False
        )
        hidden_states_87 = None
        add_20 = hidden_states_84 + hidden_states_88
        hidden_states_84 = hidden_states_88 = None
        hidden_states_89 = add_20.view((1, 19, 1024))
        add_20 = None
        hidden_states_90 = torch.nn.functional.layer_norm(
            hidden_states_89,
            (1024,),
            l_self_modules_model_modules_decoder_modules_layers_modules_8_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_8_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_89 = l_self_modules_model_modules_decoder_modules_layers_modules_8_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_8_modules_final_layer_norm_parameters_bias_ = (None)
        linear_55 = torch._C._nn.linear(
            hidden_states_90,
            l_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_18 = linear_55 * 0.125
        linear_55 = None
        view_37 = query_states_18.view(1, -1, 16, 64)
        query_states_18 = None
        query_states_19 = view_37.transpose(1, 2)
        view_37 = None
        key_states_18 = torch._C._nn.linear(
            hidden_states_90,
            l_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_18 = torch._C._nn.linear(
            hidden_states_90,
            l_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_38 = key_states_18.view(1, -1, 16, 64)
        key_states_18 = None
        key_states_19 = view_38.transpose(1, 2)
        view_38 = None
        view_39 = value_states_18.view(1, -1, 16, 64)
        value_states_18 = None
        value_states_19 = view_39.transpose(1, 2)
        view_39 = None
        attention_mask_9 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_9 = query_states_19.contiguous()
        query_states_19 = None
        key_9 = key_states_19.contiguous()
        key_states_19 = None
        value_9 = value_states_19.contiguous()
        value_states_19 = None
        attn_output_36 = torch._C._nn.scaled_dot_product_attention(
            query_9,
            key_9,
            value_9,
            attn_mask=attention_mask_9,
            dropout_p=0.0,
            scale=1.0,
            is_causal=False,
        )
        query_9 = key_9 = value_9 = attention_mask_9 = None
        transpose_39 = attn_output_36.transpose(1, 2)
        attn_output_36 = None
        attn_output_37 = transpose_39.contiguous()
        transpose_39 = None
        reshape_19 = attn_output_37.reshape(1, 19, -1)
        attn_output_37 = None
        attn_output_38 = reshape_19.contiguous()
        reshape_19 = None
        attn_output_39 = torch._C._nn.linear(
            attn_output_38,
            l_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_38 = l_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_91 = torch.nn.functional.dropout(
            attn_output_39, p=0.1, training=False
        )
        attn_output_39 = None
        hidden_states_92 = hidden_states_90 + hidden_states_91
        hidden_states_90 = hidden_states_91 = None
        hidden_states_93 = torch.nn.functional.layer_norm(
            hidden_states_92,
            (1024,),
            l_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_92 = l_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_bias_ = (None)
        hidden_states_94 = hidden_states_93.reshape(-1, 1024)
        hidden_states_93 = None
        hidden_states_95 = torch._C._nn.linear(
            hidden_states_94,
            l_self_modules_model_modules_decoder_modules_layers_modules_9_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_9_modules_fc1_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_9_modules_fc1_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_9_modules_fc1_parameters_bias_ = (None)
        hidden_states_96 = torch.nn.functional.relu(hidden_states_95, inplace=False)
        hidden_states_95 = None
        hidden_states_97 = torch._C._nn.linear(
            hidden_states_96,
            l_self_modules_model_modules_decoder_modules_layers_modules_9_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_9_modules_fc2_parameters_bias_,
        )
        hidden_states_96 = l_self_modules_model_modules_decoder_modules_layers_modules_9_modules_fc2_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_9_modules_fc2_parameters_bias_ = (None)
        hidden_states_98 = torch.nn.functional.dropout(
            hidden_states_97, p=0.1, training=False
        )
        hidden_states_97 = None
        add_22 = hidden_states_94 + hidden_states_98
        hidden_states_94 = hidden_states_98 = None
        hidden_states_99 = add_22.view((1, 19, 1024))
        add_22 = None
        hidden_states_100 = torch.nn.functional.layer_norm(
            hidden_states_99,
            (1024,),
            l_self_modules_model_modules_decoder_modules_layers_modules_9_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_9_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_99 = l_self_modules_model_modules_decoder_modules_layers_modules_9_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_9_modules_final_layer_norm_parameters_bias_ = (None)
        linear_61 = torch._C._nn.linear(
            hidden_states_100,
            l_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_20 = linear_61 * 0.125
        linear_61 = None
        view_41 = query_states_20.view(1, -1, 16, 64)
        query_states_20 = None
        query_states_21 = view_41.transpose(1, 2)
        view_41 = None
        key_states_20 = torch._C._nn.linear(
            hidden_states_100,
            l_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_20 = torch._C._nn.linear(
            hidden_states_100,
            l_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_42 = key_states_20.view(1, -1, 16, 64)
        key_states_20 = None
        key_states_21 = view_42.transpose(1, 2)
        view_42 = None
        view_43 = value_states_20.view(1, -1, 16, 64)
        value_states_20 = None
        value_states_21 = view_43.transpose(1, 2)
        view_43 = None
        attention_mask_10 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_10 = query_states_21.contiguous()
        query_states_21 = None
        key_10 = key_states_21.contiguous()
        key_states_21 = None
        value_10 = value_states_21.contiguous()
        value_states_21 = None
        attn_output_40 = torch._C._nn.scaled_dot_product_attention(
            query_10,
            key_10,
            value_10,
            attn_mask=attention_mask_10,
            dropout_p=0.0,
            scale=1.0,
            is_causal=False,
        )
        query_10 = key_10 = value_10 = attention_mask_10 = None
        transpose_43 = attn_output_40.transpose(1, 2)
        attn_output_40 = None
        attn_output_41 = transpose_43.contiguous()
        transpose_43 = None
        reshape_21 = attn_output_41.reshape(1, 19, -1)
        attn_output_41 = None
        attn_output_42 = reshape_21.contiguous()
        reshape_21 = None
        attn_output_43 = torch._C._nn.linear(
            attn_output_42,
            l_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_42 = l_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_101 = torch.nn.functional.dropout(
            attn_output_43, p=0.1, training=False
        )
        attn_output_43 = None
        hidden_states_102 = hidden_states_100 + hidden_states_101
        hidden_states_100 = hidden_states_101 = None
        hidden_states_103 = torch.nn.functional.layer_norm(
            hidden_states_102,
            (1024,),
            l_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_102 = l_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_bias_ = (None)
        hidden_states_104 = hidden_states_103.reshape(-1, 1024)
        hidden_states_103 = None
        hidden_states_105 = torch._C._nn.linear(
            hidden_states_104,
            l_self_modules_model_modules_decoder_modules_layers_modules_10_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_10_modules_fc1_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_10_modules_fc1_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_10_modules_fc1_parameters_bias_ = (None)
        hidden_states_106 = torch.nn.functional.relu(hidden_states_105, inplace=False)
        hidden_states_105 = None
        hidden_states_107 = torch._C._nn.linear(
            hidden_states_106,
            l_self_modules_model_modules_decoder_modules_layers_modules_10_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_10_modules_fc2_parameters_bias_,
        )
        hidden_states_106 = l_self_modules_model_modules_decoder_modules_layers_modules_10_modules_fc2_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_10_modules_fc2_parameters_bias_ = (None)
        hidden_states_108 = torch.nn.functional.dropout(
            hidden_states_107, p=0.1, training=False
        )
        hidden_states_107 = None
        add_24 = hidden_states_104 + hidden_states_108
        hidden_states_104 = hidden_states_108 = None
        hidden_states_109 = add_24.view((1, 19, 1024))
        add_24 = None
        hidden_states_110 = torch.nn.functional.layer_norm(
            hidden_states_109,
            (1024,),
            l_self_modules_model_modules_decoder_modules_layers_modules_10_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_10_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_109 = l_self_modules_model_modules_decoder_modules_layers_modules_10_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_10_modules_final_layer_norm_parameters_bias_ = (None)
        linear_67 = torch._C._nn.linear(
            hidden_states_110,
            l_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_22 = linear_67 * 0.125
        linear_67 = None
        view_45 = query_states_22.view(1, -1, 16, 64)
        query_states_22 = None
        query_states_23 = view_45.transpose(1, 2)
        view_45 = None
        key_states_22 = torch._C._nn.linear(
            hidden_states_110,
            l_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_22 = torch._C._nn.linear(
            hidden_states_110,
            l_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_46 = key_states_22.view(1, -1, 16, 64)
        key_states_22 = None
        key_states_23 = view_46.transpose(1, 2)
        view_46 = None
        view_47 = value_states_22.view(1, -1, 16, 64)
        value_states_22 = None
        value_states_23 = view_47.transpose(1, 2)
        view_47 = None
        attention_mask_11 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_11 = query_states_23.contiguous()
        query_states_23 = None
        key_11 = key_states_23.contiguous()
        key_states_23 = None
        value_11 = value_states_23.contiguous()
        value_states_23 = None
        attn_output_44 = torch._C._nn.scaled_dot_product_attention(
            query_11,
            key_11,
            value_11,
            attn_mask=attention_mask_11,
            dropout_p=0.0,
            scale=1.0,
            is_causal=False,
        )
        query_11 = key_11 = value_11 = attention_mask_11 = None
        transpose_47 = attn_output_44.transpose(1, 2)
        attn_output_44 = None
        attn_output_45 = transpose_47.contiguous()
        transpose_47 = None
        reshape_23 = attn_output_45.reshape(1, 19, -1)
        attn_output_45 = None
        attn_output_46 = reshape_23.contiguous()
        reshape_23 = None
        attn_output_47 = torch._C._nn.linear(
            attn_output_46,
            l_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_46 = l_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_111 = torch.nn.functional.dropout(
            attn_output_47, p=0.1, training=False
        )
        attn_output_47 = None
        hidden_states_112 = hidden_states_110 + hidden_states_111
        hidden_states_110 = hidden_states_111 = None
        hidden_states_113 = torch.nn.functional.layer_norm(
            hidden_states_112,
            (1024,),
            l_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_112 = l_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_bias_ = (None)
        hidden_states_114 = hidden_states_113.reshape(-1, 1024)
        hidden_states_113 = None
        hidden_states_115 = torch._C._nn.linear(
            hidden_states_114,
            l_self_modules_model_modules_decoder_modules_layers_modules_11_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_11_modules_fc1_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_11_modules_fc1_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_11_modules_fc1_parameters_bias_ = (None)
        hidden_states_116 = torch.nn.functional.relu(hidden_states_115, inplace=False)
        hidden_states_115 = None
        hidden_states_117 = torch._C._nn.linear(
            hidden_states_116,
            l_self_modules_model_modules_decoder_modules_layers_modules_11_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_11_modules_fc2_parameters_bias_,
        )
        hidden_states_116 = l_self_modules_model_modules_decoder_modules_layers_modules_11_modules_fc2_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_11_modules_fc2_parameters_bias_ = (None)
        hidden_states_118 = torch.nn.functional.dropout(
            hidden_states_117, p=0.1, training=False
        )
        hidden_states_117 = None
        add_26 = hidden_states_114 + hidden_states_118
        hidden_states_114 = hidden_states_118 = None
        hidden_states_119 = add_26.view((1, 19, 1024))
        add_26 = None
        hidden_states_120 = torch.nn.functional.layer_norm(
            hidden_states_119,
            (1024,),
            l_self_modules_model_modules_decoder_modules_layers_modules_11_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_11_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_119 = l_self_modules_model_modules_decoder_modules_layers_modules_11_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_11_modules_final_layer_norm_parameters_bias_ = (None)
        linear_73 = torch._C._nn.linear(
            hidden_states_120,
            l_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_24 = linear_73 * 0.125
        linear_73 = None
        view_49 = query_states_24.view(1, -1, 16, 64)
        query_states_24 = None
        query_states_25 = view_49.transpose(1, 2)
        view_49 = None
        key_states_24 = torch._C._nn.linear(
            hidden_states_120,
            l_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_24 = torch._C._nn.linear(
            hidden_states_120,
            l_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_50 = key_states_24.view(1, -1, 16, 64)
        key_states_24 = None
        key_states_25 = view_50.transpose(1, 2)
        view_50 = None
        view_51 = value_states_24.view(1, -1, 16, 64)
        value_states_24 = None
        value_states_25 = view_51.transpose(1, 2)
        view_51 = None
        attention_mask_12 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_12 = query_states_25.contiguous()
        query_states_25 = None
        key_12 = key_states_25.contiguous()
        key_states_25 = None
        value_12 = value_states_25.contiguous()
        value_states_25 = None
        attn_output_48 = torch._C._nn.scaled_dot_product_attention(
            query_12,
            key_12,
            value_12,
            attn_mask=attention_mask_12,
            dropout_p=0.0,
            scale=1.0,
            is_causal=False,
        )
        query_12 = key_12 = value_12 = attention_mask_12 = None
        transpose_51 = attn_output_48.transpose(1, 2)
        attn_output_48 = None
        attn_output_49 = transpose_51.contiguous()
        transpose_51 = None
        reshape_25 = attn_output_49.reshape(1, 19, -1)
        attn_output_49 = None
        attn_output_50 = reshape_25.contiguous()
        reshape_25 = None
        attn_output_51 = torch._C._nn.linear(
            attn_output_50,
            l_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_50 = l_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_121 = torch.nn.functional.dropout(
            attn_output_51, p=0.1, training=False
        )
        attn_output_51 = None
        hidden_states_122 = hidden_states_120 + hidden_states_121
        hidden_states_120 = hidden_states_121 = None
        hidden_states_123 = torch.nn.functional.layer_norm(
            hidden_states_122,
            (1024,),
            l_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_122 = l_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_bias_ = (None)
        hidden_states_124 = hidden_states_123.reshape(-1, 1024)
        hidden_states_123 = None
        hidden_states_125 = torch._C._nn.linear(
            hidden_states_124,
            l_self_modules_model_modules_decoder_modules_layers_modules_12_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_12_modules_fc1_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_12_modules_fc1_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_12_modules_fc1_parameters_bias_ = (None)
        hidden_states_126 = torch.nn.functional.relu(hidden_states_125, inplace=False)
        hidden_states_125 = None
        hidden_states_127 = torch._C._nn.linear(
            hidden_states_126,
            l_self_modules_model_modules_decoder_modules_layers_modules_12_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_12_modules_fc2_parameters_bias_,
        )
        hidden_states_126 = l_self_modules_model_modules_decoder_modules_layers_modules_12_modules_fc2_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_12_modules_fc2_parameters_bias_ = (None)
        hidden_states_128 = torch.nn.functional.dropout(
            hidden_states_127, p=0.1, training=False
        )
        hidden_states_127 = None
        add_28 = hidden_states_124 + hidden_states_128
        hidden_states_124 = hidden_states_128 = None
        hidden_states_129 = add_28.view((1, 19, 1024))
        add_28 = None
        hidden_states_130 = torch.nn.functional.layer_norm(
            hidden_states_129,
            (1024,),
            l_self_modules_model_modules_decoder_modules_layers_modules_12_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_12_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_129 = l_self_modules_model_modules_decoder_modules_layers_modules_12_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_12_modules_final_layer_norm_parameters_bias_ = (None)
        linear_79 = torch._C._nn.linear(
            hidden_states_130,
            l_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_26 = linear_79 * 0.125
        linear_79 = None
        view_53 = query_states_26.view(1, -1, 16, 64)
        query_states_26 = None
        query_states_27 = view_53.transpose(1, 2)
        view_53 = None
        key_states_26 = torch._C._nn.linear(
            hidden_states_130,
            l_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_26 = torch._C._nn.linear(
            hidden_states_130,
            l_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_54 = key_states_26.view(1, -1, 16, 64)
        key_states_26 = None
        key_states_27 = view_54.transpose(1, 2)
        view_54 = None
        view_55 = value_states_26.view(1, -1, 16, 64)
        value_states_26 = None
        value_states_27 = view_55.transpose(1, 2)
        view_55 = None
        attention_mask_13 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_13 = query_states_27.contiguous()
        query_states_27 = None
        key_13 = key_states_27.contiguous()
        key_states_27 = None
        value_13 = value_states_27.contiguous()
        value_states_27 = None
        attn_output_52 = torch._C._nn.scaled_dot_product_attention(
            query_13,
            key_13,
            value_13,
            attn_mask=attention_mask_13,
            dropout_p=0.0,
            scale=1.0,
            is_causal=False,
        )
        query_13 = key_13 = value_13 = attention_mask_13 = None
        transpose_55 = attn_output_52.transpose(1, 2)
        attn_output_52 = None
        attn_output_53 = transpose_55.contiguous()
        transpose_55 = None
        reshape_27 = attn_output_53.reshape(1, 19, -1)
        attn_output_53 = None
        attn_output_54 = reshape_27.contiguous()
        reshape_27 = None
        attn_output_55 = torch._C._nn.linear(
            attn_output_54,
            l_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_54 = l_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_131 = torch.nn.functional.dropout(
            attn_output_55, p=0.1, training=False
        )
        attn_output_55 = None
        hidden_states_132 = hidden_states_130 + hidden_states_131
        hidden_states_130 = hidden_states_131 = None
        hidden_states_133 = torch.nn.functional.layer_norm(
            hidden_states_132,
            (1024,),
            l_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_132 = l_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_bias_ = (None)
        hidden_states_134 = hidden_states_133.reshape(-1, 1024)
        hidden_states_133 = None
        hidden_states_135 = torch._C._nn.linear(
            hidden_states_134,
            l_self_modules_model_modules_decoder_modules_layers_modules_13_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_13_modules_fc1_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_13_modules_fc1_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_13_modules_fc1_parameters_bias_ = (None)
        hidden_states_136 = torch.nn.functional.relu(hidden_states_135, inplace=False)
        hidden_states_135 = None
        hidden_states_137 = torch._C._nn.linear(
            hidden_states_136,
            l_self_modules_model_modules_decoder_modules_layers_modules_13_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_13_modules_fc2_parameters_bias_,
        )
        hidden_states_136 = l_self_modules_model_modules_decoder_modules_layers_modules_13_modules_fc2_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_13_modules_fc2_parameters_bias_ = (None)
        hidden_states_138 = torch.nn.functional.dropout(
            hidden_states_137, p=0.1, training=False
        )
        hidden_states_137 = None
        add_30 = hidden_states_134 + hidden_states_138
        hidden_states_134 = hidden_states_138 = None
        hidden_states_139 = add_30.view((1, 19, 1024))
        add_30 = None
        hidden_states_140 = torch.nn.functional.layer_norm(
            hidden_states_139,
            (1024,),
            l_self_modules_model_modules_decoder_modules_layers_modules_13_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_13_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_139 = l_self_modules_model_modules_decoder_modules_layers_modules_13_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_13_modules_final_layer_norm_parameters_bias_ = (None)
        linear_85 = torch._C._nn.linear(
            hidden_states_140,
            l_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_28 = linear_85 * 0.125
        linear_85 = None
        view_57 = query_states_28.view(1, -1, 16, 64)
        query_states_28 = None
        query_states_29 = view_57.transpose(1, 2)
        view_57 = None
        key_states_28 = torch._C._nn.linear(
            hidden_states_140,
            l_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_28 = torch._C._nn.linear(
            hidden_states_140,
            l_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_58 = key_states_28.view(1, -1, 16, 64)
        key_states_28 = None
        key_states_29 = view_58.transpose(1, 2)
        view_58 = None
        view_59 = value_states_28.view(1, -1, 16, 64)
        value_states_28 = None
        value_states_29 = view_59.transpose(1, 2)
        view_59 = None
        attention_mask_14 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_14 = query_states_29.contiguous()
        query_states_29 = None
        key_14 = key_states_29.contiguous()
        key_states_29 = None
        value_14 = value_states_29.contiguous()
        value_states_29 = None
        attn_output_56 = torch._C._nn.scaled_dot_product_attention(
            query_14,
            key_14,
            value_14,
            attn_mask=attention_mask_14,
            dropout_p=0.0,
            scale=1.0,
            is_causal=False,
        )
        query_14 = key_14 = value_14 = attention_mask_14 = None
        transpose_59 = attn_output_56.transpose(1, 2)
        attn_output_56 = None
        attn_output_57 = transpose_59.contiguous()
        transpose_59 = None
        reshape_29 = attn_output_57.reshape(1, 19, -1)
        attn_output_57 = None
        attn_output_58 = reshape_29.contiguous()
        reshape_29 = None
        attn_output_59 = torch._C._nn.linear(
            attn_output_58,
            l_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_58 = l_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_141 = torch.nn.functional.dropout(
            attn_output_59, p=0.1, training=False
        )
        attn_output_59 = None
        hidden_states_142 = hidden_states_140 + hidden_states_141
        hidden_states_140 = hidden_states_141 = None
        hidden_states_143 = torch.nn.functional.layer_norm(
            hidden_states_142,
            (1024,),
            l_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_142 = l_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_bias_ = (None)
        hidden_states_144 = hidden_states_143.reshape(-1, 1024)
        hidden_states_143 = None
        hidden_states_145 = torch._C._nn.linear(
            hidden_states_144,
            l_self_modules_model_modules_decoder_modules_layers_modules_14_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_14_modules_fc1_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_14_modules_fc1_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_14_modules_fc1_parameters_bias_ = (None)
        hidden_states_146 = torch.nn.functional.relu(hidden_states_145, inplace=False)
        hidden_states_145 = None
        hidden_states_147 = torch._C._nn.linear(
            hidden_states_146,
            l_self_modules_model_modules_decoder_modules_layers_modules_14_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_14_modules_fc2_parameters_bias_,
        )
        hidden_states_146 = l_self_modules_model_modules_decoder_modules_layers_modules_14_modules_fc2_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_14_modules_fc2_parameters_bias_ = (None)
        hidden_states_148 = torch.nn.functional.dropout(
            hidden_states_147, p=0.1, training=False
        )
        hidden_states_147 = None
        add_32 = hidden_states_144 + hidden_states_148
        hidden_states_144 = hidden_states_148 = None
        hidden_states_149 = add_32.view((1, 19, 1024))
        add_32 = None
        hidden_states_150 = torch.nn.functional.layer_norm(
            hidden_states_149,
            (1024,),
            l_self_modules_model_modules_decoder_modules_layers_modules_14_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_14_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_149 = l_self_modules_model_modules_decoder_modules_layers_modules_14_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_14_modules_final_layer_norm_parameters_bias_ = (None)
        linear_91 = torch._C._nn.linear(
            hidden_states_150,
            l_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_30 = linear_91 * 0.125
        linear_91 = None
        view_61 = query_states_30.view(1, -1, 16, 64)
        query_states_30 = None
        query_states_31 = view_61.transpose(1, 2)
        view_61 = None
        key_states_30 = torch._C._nn.linear(
            hidden_states_150,
            l_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_30 = torch._C._nn.linear(
            hidden_states_150,
            l_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_62 = key_states_30.view(1, -1, 16, 64)
        key_states_30 = None
        key_states_31 = view_62.transpose(1, 2)
        view_62 = None
        view_63 = value_states_30.view(1, -1, 16, 64)
        value_states_30 = None
        value_states_31 = view_63.transpose(1, 2)
        view_63 = None
        attention_mask_15 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_15 = query_states_31.contiguous()
        query_states_31 = None
        key_15 = key_states_31.contiguous()
        key_states_31 = None
        value_15 = value_states_31.contiguous()
        value_states_31 = None
        attn_output_60 = torch._C._nn.scaled_dot_product_attention(
            query_15,
            key_15,
            value_15,
            attn_mask=attention_mask_15,
            dropout_p=0.0,
            scale=1.0,
            is_causal=False,
        )
        query_15 = key_15 = value_15 = attention_mask_15 = None
        transpose_63 = attn_output_60.transpose(1, 2)
        attn_output_60 = None
        attn_output_61 = transpose_63.contiguous()
        transpose_63 = None
        reshape_31 = attn_output_61.reshape(1, 19, -1)
        attn_output_61 = None
        attn_output_62 = reshape_31.contiguous()
        reshape_31 = None
        attn_output_63 = torch._C._nn.linear(
            attn_output_62,
            l_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_62 = l_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_151 = torch.nn.functional.dropout(
            attn_output_63, p=0.1, training=False
        )
        attn_output_63 = None
        hidden_states_152 = hidden_states_150 + hidden_states_151
        hidden_states_150 = hidden_states_151 = None
        hidden_states_153 = torch.nn.functional.layer_norm(
            hidden_states_152,
            (1024,),
            l_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_152 = l_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_bias_ = (None)
        hidden_states_154 = hidden_states_153.reshape(-1, 1024)
        hidden_states_153 = None
        hidden_states_155 = torch._C._nn.linear(
            hidden_states_154,
            l_self_modules_model_modules_decoder_modules_layers_modules_15_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_15_modules_fc1_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_15_modules_fc1_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_15_modules_fc1_parameters_bias_ = (None)
        hidden_states_156 = torch.nn.functional.relu(hidden_states_155, inplace=False)
        hidden_states_155 = None
        hidden_states_157 = torch._C._nn.linear(
            hidden_states_156,
            l_self_modules_model_modules_decoder_modules_layers_modules_15_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_15_modules_fc2_parameters_bias_,
        )
        hidden_states_156 = l_self_modules_model_modules_decoder_modules_layers_modules_15_modules_fc2_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_15_modules_fc2_parameters_bias_ = (None)
        hidden_states_158 = torch.nn.functional.dropout(
            hidden_states_157, p=0.1, training=False
        )
        hidden_states_157 = None
        add_34 = hidden_states_154 + hidden_states_158
        hidden_states_154 = hidden_states_158 = None
        hidden_states_159 = add_34.view((1, 19, 1024))
        add_34 = None
        hidden_states_160 = torch.nn.functional.layer_norm(
            hidden_states_159,
            (1024,),
            l_self_modules_model_modules_decoder_modules_layers_modules_15_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_15_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_159 = l_self_modules_model_modules_decoder_modules_layers_modules_15_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_15_modules_final_layer_norm_parameters_bias_ = (None)
        linear_97 = torch._C._nn.linear(
            hidden_states_160,
            l_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_32 = linear_97 * 0.125
        linear_97 = None
        view_65 = query_states_32.view(1, -1, 16, 64)
        query_states_32 = None
        query_states_33 = view_65.transpose(1, 2)
        view_65 = None
        key_states_32 = torch._C._nn.linear(
            hidden_states_160,
            l_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_32 = torch._C._nn.linear(
            hidden_states_160,
            l_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_66 = key_states_32.view(1, -1, 16, 64)
        key_states_32 = None
        key_states_33 = view_66.transpose(1, 2)
        view_66 = None
        view_67 = value_states_32.view(1, -1, 16, 64)
        value_states_32 = None
        value_states_33 = view_67.transpose(1, 2)
        view_67 = None
        attention_mask_16 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_16 = query_states_33.contiguous()
        query_states_33 = None
        key_16 = key_states_33.contiguous()
        key_states_33 = None
        value_16 = value_states_33.contiguous()
        value_states_33 = None
        attn_output_64 = torch._C._nn.scaled_dot_product_attention(
            query_16,
            key_16,
            value_16,
            attn_mask=attention_mask_16,
            dropout_p=0.0,
            scale=1.0,
            is_causal=False,
        )
        query_16 = key_16 = value_16 = attention_mask_16 = None
        transpose_67 = attn_output_64.transpose(1, 2)
        attn_output_64 = None
        attn_output_65 = transpose_67.contiguous()
        transpose_67 = None
        reshape_33 = attn_output_65.reshape(1, 19, -1)
        attn_output_65 = None
        attn_output_66 = reshape_33.contiguous()
        reshape_33 = None
        attn_output_67 = torch._C._nn.linear(
            attn_output_66,
            l_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_66 = l_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_161 = torch.nn.functional.dropout(
            attn_output_67, p=0.1, training=False
        )
        attn_output_67 = None
        hidden_states_162 = hidden_states_160 + hidden_states_161
        hidden_states_160 = hidden_states_161 = None
        hidden_states_163 = torch.nn.functional.layer_norm(
            hidden_states_162,
            (1024,),
            l_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_162 = l_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_16_modules_self_attn_layer_norm_parameters_bias_ = (None)
        hidden_states_164 = hidden_states_163.reshape(-1, 1024)
        hidden_states_163 = None
        hidden_states_165 = torch._C._nn.linear(
            hidden_states_164,
            l_self_modules_model_modules_decoder_modules_layers_modules_16_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_16_modules_fc1_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_16_modules_fc1_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_16_modules_fc1_parameters_bias_ = (None)
        hidden_states_166 = torch.nn.functional.relu(hidden_states_165, inplace=False)
        hidden_states_165 = None
        hidden_states_167 = torch._C._nn.linear(
            hidden_states_166,
            l_self_modules_model_modules_decoder_modules_layers_modules_16_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_16_modules_fc2_parameters_bias_,
        )
        hidden_states_166 = l_self_modules_model_modules_decoder_modules_layers_modules_16_modules_fc2_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_16_modules_fc2_parameters_bias_ = (None)
        hidden_states_168 = torch.nn.functional.dropout(
            hidden_states_167, p=0.1, training=False
        )
        hidden_states_167 = None
        add_36 = hidden_states_164 + hidden_states_168
        hidden_states_164 = hidden_states_168 = None
        hidden_states_169 = add_36.view((1, 19, 1024))
        add_36 = None
        hidden_states_170 = torch.nn.functional.layer_norm(
            hidden_states_169,
            (1024,),
            l_self_modules_model_modules_decoder_modules_layers_modules_16_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_16_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_169 = l_self_modules_model_modules_decoder_modules_layers_modules_16_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_16_modules_final_layer_norm_parameters_bias_ = (None)
        linear_103 = torch._C._nn.linear(
            hidden_states_170,
            l_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_34 = linear_103 * 0.125
        linear_103 = None
        view_69 = query_states_34.view(1, -1, 16, 64)
        query_states_34 = None
        query_states_35 = view_69.transpose(1, 2)
        view_69 = None
        key_states_34 = torch._C._nn.linear(
            hidden_states_170,
            l_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_34 = torch._C._nn.linear(
            hidden_states_170,
            l_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_70 = key_states_34.view(1, -1, 16, 64)
        key_states_34 = None
        key_states_35 = view_70.transpose(1, 2)
        view_70 = None
        view_71 = value_states_34.view(1, -1, 16, 64)
        value_states_34 = None
        value_states_35 = view_71.transpose(1, 2)
        view_71 = None
        attention_mask_17 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_17 = query_states_35.contiguous()
        query_states_35 = None
        key_17 = key_states_35.contiguous()
        key_states_35 = None
        value_17 = value_states_35.contiguous()
        value_states_35 = None
        attn_output_68 = torch._C._nn.scaled_dot_product_attention(
            query_17,
            key_17,
            value_17,
            attn_mask=attention_mask_17,
            dropout_p=0.0,
            scale=1.0,
            is_causal=False,
        )
        query_17 = key_17 = value_17 = attention_mask_17 = None
        transpose_71 = attn_output_68.transpose(1, 2)
        attn_output_68 = None
        attn_output_69 = transpose_71.contiguous()
        transpose_71 = None
        reshape_35 = attn_output_69.reshape(1, 19, -1)
        attn_output_69 = None
        attn_output_70 = reshape_35.contiguous()
        reshape_35 = None
        attn_output_71 = torch._C._nn.linear(
            attn_output_70,
            l_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_70 = l_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_171 = torch.nn.functional.dropout(
            attn_output_71, p=0.1, training=False
        )
        attn_output_71 = None
        hidden_states_172 = hidden_states_170 + hidden_states_171
        hidden_states_170 = hidden_states_171 = None
        hidden_states_173 = torch.nn.functional.layer_norm(
            hidden_states_172,
            (1024,),
            l_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_172 = l_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_17_modules_self_attn_layer_norm_parameters_bias_ = (None)
        hidden_states_174 = hidden_states_173.reshape(-1, 1024)
        hidden_states_173 = None
        hidden_states_175 = torch._C._nn.linear(
            hidden_states_174,
            l_self_modules_model_modules_decoder_modules_layers_modules_17_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_17_modules_fc1_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_17_modules_fc1_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_17_modules_fc1_parameters_bias_ = (None)
        hidden_states_176 = torch.nn.functional.relu(hidden_states_175, inplace=False)
        hidden_states_175 = None
        hidden_states_177 = torch._C._nn.linear(
            hidden_states_176,
            l_self_modules_model_modules_decoder_modules_layers_modules_17_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_17_modules_fc2_parameters_bias_,
        )
        hidden_states_176 = l_self_modules_model_modules_decoder_modules_layers_modules_17_modules_fc2_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_17_modules_fc2_parameters_bias_ = (None)
        hidden_states_178 = torch.nn.functional.dropout(
            hidden_states_177, p=0.1, training=False
        )
        hidden_states_177 = None
        add_38 = hidden_states_174 + hidden_states_178
        hidden_states_174 = hidden_states_178 = None
        hidden_states_179 = add_38.view((1, 19, 1024))
        add_38 = None
        hidden_states_180 = torch.nn.functional.layer_norm(
            hidden_states_179,
            (1024,),
            l_self_modules_model_modules_decoder_modules_layers_modules_17_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_17_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_179 = l_self_modules_model_modules_decoder_modules_layers_modules_17_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_17_modules_final_layer_norm_parameters_bias_ = (None)
        linear_109 = torch._C._nn.linear(
            hidden_states_180,
            l_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_36 = linear_109 * 0.125
        linear_109 = None
        view_73 = query_states_36.view(1, -1, 16, 64)
        query_states_36 = None
        query_states_37 = view_73.transpose(1, 2)
        view_73 = None
        key_states_36 = torch._C._nn.linear(
            hidden_states_180,
            l_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_36 = torch._C._nn.linear(
            hidden_states_180,
            l_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_74 = key_states_36.view(1, -1, 16, 64)
        key_states_36 = None
        key_states_37 = view_74.transpose(1, 2)
        view_74 = None
        view_75 = value_states_36.view(1, -1, 16, 64)
        value_states_36 = None
        value_states_37 = view_75.transpose(1, 2)
        view_75 = None
        attention_mask_18 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_18 = query_states_37.contiguous()
        query_states_37 = None
        key_18 = key_states_37.contiguous()
        key_states_37 = None
        value_18 = value_states_37.contiguous()
        value_states_37 = None
        attn_output_72 = torch._C._nn.scaled_dot_product_attention(
            query_18,
            key_18,
            value_18,
            attn_mask=attention_mask_18,
            dropout_p=0.0,
            scale=1.0,
            is_causal=False,
        )
        query_18 = key_18 = value_18 = attention_mask_18 = None
        transpose_75 = attn_output_72.transpose(1, 2)
        attn_output_72 = None
        attn_output_73 = transpose_75.contiguous()
        transpose_75 = None
        reshape_37 = attn_output_73.reshape(1, 19, -1)
        attn_output_73 = None
        attn_output_74 = reshape_37.contiguous()
        reshape_37 = None
        attn_output_75 = torch._C._nn.linear(
            attn_output_74,
            l_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_74 = l_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_181 = torch.nn.functional.dropout(
            attn_output_75, p=0.1, training=False
        )
        attn_output_75 = None
        hidden_states_182 = hidden_states_180 + hidden_states_181
        hidden_states_180 = hidden_states_181 = None
        hidden_states_183 = torch.nn.functional.layer_norm(
            hidden_states_182,
            (1024,),
            l_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_182 = l_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_18_modules_self_attn_layer_norm_parameters_bias_ = (None)
        hidden_states_184 = hidden_states_183.reshape(-1, 1024)
        hidden_states_183 = None
        hidden_states_185 = torch._C._nn.linear(
            hidden_states_184,
            l_self_modules_model_modules_decoder_modules_layers_modules_18_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_18_modules_fc1_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_18_modules_fc1_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_18_modules_fc1_parameters_bias_ = (None)
        hidden_states_186 = torch.nn.functional.relu(hidden_states_185, inplace=False)
        hidden_states_185 = None
        hidden_states_187 = torch._C._nn.linear(
            hidden_states_186,
            l_self_modules_model_modules_decoder_modules_layers_modules_18_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_18_modules_fc2_parameters_bias_,
        )
        hidden_states_186 = l_self_modules_model_modules_decoder_modules_layers_modules_18_modules_fc2_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_18_modules_fc2_parameters_bias_ = (None)
        hidden_states_188 = torch.nn.functional.dropout(
            hidden_states_187, p=0.1, training=False
        )
        hidden_states_187 = None
        add_40 = hidden_states_184 + hidden_states_188
        hidden_states_184 = hidden_states_188 = None
        hidden_states_189 = add_40.view((1, 19, 1024))
        add_40 = None
        hidden_states_190 = torch.nn.functional.layer_norm(
            hidden_states_189,
            (1024,),
            l_self_modules_model_modules_decoder_modules_layers_modules_18_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_18_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_189 = l_self_modules_model_modules_decoder_modules_layers_modules_18_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_18_modules_final_layer_norm_parameters_bias_ = (None)
        linear_115 = torch._C._nn.linear(
            hidden_states_190,
            l_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_38 = linear_115 * 0.125
        linear_115 = None
        view_77 = query_states_38.view(1, -1, 16, 64)
        query_states_38 = None
        query_states_39 = view_77.transpose(1, 2)
        view_77 = None
        key_states_38 = torch._C._nn.linear(
            hidden_states_190,
            l_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_38 = torch._C._nn.linear(
            hidden_states_190,
            l_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_78 = key_states_38.view(1, -1, 16, 64)
        key_states_38 = None
        key_states_39 = view_78.transpose(1, 2)
        view_78 = None
        view_79 = value_states_38.view(1, -1, 16, 64)
        value_states_38 = None
        value_states_39 = view_79.transpose(1, 2)
        view_79 = None
        attention_mask_19 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_19 = query_states_39.contiguous()
        query_states_39 = None
        key_19 = key_states_39.contiguous()
        key_states_39 = None
        value_19 = value_states_39.contiguous()
        value_states_39 = None
        attn_output_76 = torch._C._nn.scaled_dot_product_attention(
            query_19,
            key_19,
            value_19,
            attn_mask=attention_mask_19,
            dropout_p=0.0,
            scale=1.0,
            is_causal=False,
        )
        query_19 = key_19 = value_19 = attention_mask_19 = None
        transpose_79 = attn_output_76.transpose(1, 2)
        attn_output_76 = None
        attn_output_77 = transpose_79.contiguous()
        transpose_79 = None
        reshape_39 = attn_output_77.reshape(1, 19, -1)
        attn_output_77 = None
        attn_output_78 = reshape_39.contiguous()
        reshape_39 = None
        attn_output_79 = torch._C._nn.linear(
            attn_output_78,
            l_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_78 = l_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_191 = torch.nn.functional.dropout(
            attn_output_79, p=0.1, training=False
        )
        attn_output_79 = None
        hidden_states_192 = hidden_states_190 + hidden_states_191
        hidden_states_190 = hidden_states_191 = None
        hidden_states_193 = torch.nn.functional.layer_norm(
            hidden_states_192,
            (1024,),
            l_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_192 = l_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_19_modules_self_attn_layer_norm_parameters_bias_ = (None)
        hidden_states_194 = hidden_states_193.reshape(-1, 1024)
        hidden_states_193 = None
        hidden_states_195 = torch._C._nn.linear(
            hidden_states_194,
            l_self_modules_model_modules_decoder_modules_layers_modules_19_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_19_modules_fc1_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_19_modules_fc1_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_19_modules_fc1_parameters_bias_ = (None)
        hidden_states_196 = torch.nn.functional.relu(hidden_states_195, inplace=False)
        hidden_states_195 = None
        hidden_states_197 = torch._C._nn.linear(
            hidden_states_196,
            l_self_modules_model_modules_decoder_modules_layers_modules_19_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_19_modules_fc2_parameters_bias_,
        )
        hidden_states_196 = l_self_modules_model_modules_decoder_modules_layers_modules_19_modules_fc2_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_19_modules_fc2_parameters_bias_ = (None)
        hidden_states_198 = torch.nn.functional.dropout(
            hidden_states_197, p=0.1, training=False
        )
        hidden_states_197 = None
        add_42 = hidden_states_194 + hidden_states_198
        hidden_states_194 = hidden_states_198 = None
        hidden_states_199 = add_42.view((1, 19, 1024))
        add_42 = None
        hidden_states_200 = torch.nn.functional.layer_norm(
            hidden_states_199,
            (1024,),
            l_self_modules_model_modules_decoder_modules_layers_modules_19_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_19_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_199 = l_self_modules_model_modules_decoder_modules_layers_modules_19_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_19_modules_final_layer_norm_parameters_bias_ = (None)
        linear_121 = torch._C._nn.linear(
            hidden_states_200,
            l_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_40 = linear_121 * 0.125
        linear_121 = None
        view_81 = query_states_40.view(1, -1, 16, 64)
        query_states_40 = None
        query_states_41 = view_81.transpose(1, 2)
        view_81 = None
        key_states_40 = torch._C._nn.linear(
            hidden_states_200,
            l_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_40 = torch._C._nn.linear(
            hidden_states_200,
            l_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_82 = key_states_40.view(1, -1, 16, 64)
        key_states_40 = None
        key_states_41 = view_82.transpose(1, 2)
        view_82 = None
        view_83 = value_states_40.view(1, -1, 16, 64)
        value_states_40 = None
        value_states_41 = view_83.transpose(1, 2)
        view_83 = None
        attention_mask_20 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_20 = query_states_41.contiguous()
        query_states_41 = None
        key_20 = key_states_41.contiguous()
        key_states_41 = None
        value_20 = value_states_41.contiguous()
        value_states_41 = None
        attn_output_80 = torch._C._nn.scaled_dot_product_attention(
            query_20,
            key_20,
            value_20,
            attn_mask=attention_mask_20,
            dropout_p=0.0,
            scale=1.0,
            is_causal=False,
        )
        query_20 = key_20 = value_20 = attention_mask_20 = None
        transpose_83 = attn_output_80.transpose(1, 2)
        attn_output_80 = None
        attn_output_81 = transpose_83.contiguous()
        transpose_83 = None
        reshape_41 = attn_output_81.reshape(1, 19, -1)
        attn_output_81 = None
        attn_output_82 = reshape_41.contiguous()
        reshape_41 = None
        attn_output_83 = torch._C._nn.linear(
            attn_output_82,
            l_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_82 = l_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_201 = torch.nn.functional.dropout(
            attn_output_83, p=0.1, training=False
        )
        attn_output_83 = None
        hidden_states_202 = hidden_states_200 + hidden_states_201
        hidden_states_200 = hidden_states_201 = None
        hidden_states_203 = torch.nn.functional.layer_norm(
            hidden_states_202,
            (1024,),
            l_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_202 = l_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_20_modules_self_attn_layer_norm_parameters_bias_ = (None)
        hidden_states_204 = hidden_states_203.reshape(-1, 1024)
        hidden_states_203 = None
        hidden_states_205 = torch._C._nn.linear(
            hidden_states_204,
            l_self_modules_model_modules_decoder_modules_layers_modules_20_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_20_modules_fc1_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_20_modules_fc1_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_20_modules_fc1_parameters_bias_ = (None)
        hidden_states_206 = torch.nn.functional.relu(hidden_states_205, inplace=False)
        hidden_states_205 = None
        hidden_states_207 = torch._C._nn.linear(
            hidden_states_206,
            l_self_modules_model_modules_decoder_modules_layers_modules_20_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_20_modules_fc2_parameters_bias_,
        )
        hidden_states_206 = l_self_modules_model_modules_decoder_modules_layers_modules_20_modules_fc2_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_20_modules_fc2_parameters_bias_ = (None)
        hidden_states_208 = torch.nn.functional.dropout(
            hidden_states_207, p=0.1, training=False
        )
        hidden_states_207 = None
        add_44 = hidden_states_204 + hidden_states_208
        hidden_states_204 = hidden_states_208 = None
        hidden_states_209 = add_44.view((1, 19, 1024))
        add_44 = None
        hidden_states_210 = torch.nn.functional.layer_norm(
            hidden_states_209,
            (1024,),
            l_self_modules_model_modules_decoder_modules_layers_modules_20_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_20_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_209 = l_self_modules_model_modules_decoder_modules_layers_modules_20_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_20_modules_final_layer_norm_parameters_bias_ = (None)
        linear_127 = torch._C._nn.linear(
            hidden_states_210,
            l_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_42 = linear_127 * 0.125
        linear_127 = None
        view_85 = query_states_42.view(1, -1, 16, 64)
        query_states_42 = None
        query_states_43 = view_85.transpose(1, 2)
        view_85 = None
        key_states_42 = torch._C._nn.linear(
            hidden_states_210,
            l_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_42 = torch._C._nn.linear(
            hidden_states_210,
            l_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_86 = key_states_42.view(1, -1, 16, 64)
        key_states_42 = None
        key_states_43 = view_86.transpose(1, 2)
        view_86 = None
        view_87 = value_states_42.view(1, -1, 16, 64)
        value_states_42 = None
        value_states_43 = view_87.transpose(1, 2)
        view_87 = None
        attention_mask_21 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_21 = query_states_43.contiguous()
        query_states_43 = None
        key_21 = key_states_43.contiguous()
        key_states_43 = None
        value_21 = value_states_43.contiguous()
        value_states_43 = None
        attn_output_84 = torch._C._nn.scaled_dot_product_attention(
            query_21,
            key_21,
            value_21,
            attn_mask=attention_mask_21,
            dropout_p=0.0,
            scale=1.0,
            is_causal=False,
        )
        query_21 = key_21 = value_21 = attention_mask_21 = None
        transpose_87 = attn_output_84.transpose(1, 2)
        attn_output_84 = None
        attn_output_85 = transpose_87.contiguous()
        transpose_87 = None
        reshape_43 = attn_output_85.reshape(1, 19, -1)
        attn_output_85 = None
        attn_output_86 = reshape_43.contiguous()
        reshape_43 = None
        attn_output_87 = torch._C._nn.linear(
            attn_output_86,
            l_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_86 = l_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_211 = torch.nn.functional.dropout(
            attn_output_87, p=0.1, training=False
        )
        attn_output_87 = None
        hidden_states_212 = hidden_states_210 + hidden_states_211
        hidden_states_210 = hidden_states_211 = None
        hidden_states_213 = torch.nn.functional.layer_norm(
            hidden_states_212,
            (1024,),
            l_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_212 = l_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_21_modules_self_attn_layer_norm_parameters_bias_ = (None)
        hidden_states_214 = hidden_states_213.reshape(-1, 1024)
        hidden_states_213 = None
        hidden_states_215 = torch._C._nn.linear(
            hidden_states_214,
            l_self_modules_model_modules_decoder_modules_layers_modules_21_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_21_modules_fc1_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_21_modules_fc1_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_21_modules_fc1_parameters_bias_ = (None)
        hidden_states_216 = torch.nn.functional.relu(hidden_states_215, inplace=False)
        hidden_states_215 = None
        hidden_states_217 = torch._C._nn.linear(
            hidden_states_216,
            l_self_modules_model_modules_decoder_modules_layers_modules_21_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_21_modules_fc2_parameters_bias_,
        )
        hidden_states_216 = l_self_modules_model_modules_decoder_modules_layers_modules_21_modules_fc2_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_21_modules_fc2_parameters_bias_ = (None)
        hidden_states_218 = torch.nn.functional.dropout(
            hidden_states_217, p=0.1, training=False
        )
        hidden_states_217 = None
        add_46 = hidden_states_214 + hidden_states_218
        hidden_states_214 = hidden_states_218 = None
        hidden_states_219 = add_46.view((1, 19, 1024))
        add_46 = None
        hidden_states_220 = torch.nn.functional.layer_norm(
            hidden_states_219,
            (1024,),
            l_self_modules_model_modules_decoder_modules_layers_modules_21_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_21_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_219 = l_self_modules_model_modules_decoder_modules_layers_modules_21_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_21_modules_final_layer_norm_parameters_bias_ = (None)
        linear_133 = torch._C._nn.linear(
            hidden_states_220,
            l_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_44 = linear_133 * 0.125
        linear_133 = None
        view_89 = query_states_44.view(1, -1, 16, 64)
        query_states_44 = None
        query_states_45 = view_89.transpose(1, 2)
        view_89 = None
        key_states_44 = torch._C._nn.linear(
            hidden_states_220,
            l_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_44 = torch._C._nn.linear(
            hidden_states_220,
            l_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_90 = key_states_44.view(1, -1, 16, 64)
        key_states_44 = None
        key_states_45 = view_90.transpose(1, 2)
        view_90 = None
        view_91 = value_states_44.view(1, -1, 16, 64)
        value_states_44 = None
        value_states_45 = view_91.transpose(1, 2)
        view_91 = None
        attention_mask_22 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_22 = query_states_45.contiguous()
        query_states_45 = None
        key_22 = key_states_45.contiguous()
        key_states_45 = None
        value_22 = value_states_45.contiguous()
        value_states_45 = None
        attn_output_88 = torch._C._nn.scaled_dot_product_attention(
            query_22,
            key_22,
            value_22,
            attn_mask=attention_mask_22,
            dropout_p=0.0,
            scale=1.0,
            is_causal=False,
        )
        query_22 = key_22 = value_22 = attention_mask_22 = None
        transpose_91 = attn_output_88.transpose(1, 2)
        attn_output_88 = None
        attn_output_89 = transpose_91.contiguous()
        transpose_91 = None
        reshape_45 = attn_output_89.reshape(1, 19, -1)
        attn_output_89 = None
        attn_output_90 = reshape_45.contiguous()
        reshape_45 = None
        attn_output_91 = torch._C._nn.linear(
            attn_output_90,
            l_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_90 = l_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_221 = torch.nn.functional.dropout(
            attn_output_91, p=0.1, training=False
        )
        attn_output_91 = None
        hidden_states_222 = hidden_states_220 + hidden_states_221
        hidden_states_220 = hidden_states_221 = None
        hidden_states_223 = torch.nn.functional.layer_norm(
            hidden_states_222,
            (1024,),
            l_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_222 = l_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_22_modules_self_attn_layer_norm_parameters_bias_ = (None)
        hidden_states_224 = hidden_states_223.reshape(-1, 1024)
        hidden_states_223 = None
        hidden_states_225 = torch._C._nn.linear(
            hidden_states_224,
            l_self_modules_model_modules_decoder_modules_layers_modules_22_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_22_modules_fc1_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_22_modules_fc1_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_22_modules_fc1_parameters_bias_ = (None)
        hidden_states_226 = torch.nn.functional.relu(hidden_states_225, inplace=False)
        hidden_states_225 = None
        hidden_states_227 = torch._C._nn.linear(
            hidden_states_226,
            l_self_modules_model_modules_decoder_modules_layers_modules_22_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_22_modules_fc2_parameters_bias_,
        )
        hidden_states_226 = l_self_modules_model_modules_decoder_modules_layers_modules_22_modules_fc2_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_22_modules_fc2_parameters_bias_ = (None)
        hidden_states_228 = torch.nn.functional.dropout(
            hidden_states_227, p=0.1, training=False
        )
        hidden_states_227 = None
        add_48 = hidden_states_224 + hidden_states_228
        hidden_states_224 = hidden_states_228 = None
        hidden_states_229 = add_48.view((1, 19, 1024))
        add_48 = None
        hidden_states_230 = torch.nn.functional.layer_norm(
            hidden_states_229,
            (1024,),
            l_self_modules_model_modules_decoder_modules_layers_modules_22_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_22_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_229 = l_self_modules_model_modules_decoder_modules_layers_modules_22_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_22_modules_final_layer_norm_parameters_bias_ = (None)
        linear_139 = torch._C._nn.linear(
            hidden_states_230,
            l_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_46 = linear_139 * 0.125
        linear_139 = None
        view_93 = query_states_46.view(1, -1, 16, 64)
        query_states_46 = None
        query_states_47 = view_93.transpose(1, 2)
        view_93 = None
        key_states_46 = torch._C._nn.linear(
            hidden_states_230,
            l_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_46 = torch._C._nn.linear(
            hidden_states_230,
            l_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_94 = key_states_46.view(1, -1, 16, 64)
        key_states_46 = None
        key_states_47 = view_94.transpose(1, 2)
        view_94 = None
        view_95 = value_states_46.view(1, -1, 16, 64)
        value_states_46 = None
        value_states_47 = view_95.transpose(1, 2)
        view_95 = None
        attention_mask_23 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        causal_mask_4 = None
        query_23 = query_states_47.contiguous()
        query_states_47 = None
        key_23 = key_states_47.contiguous()
        key_states_47 = None
        value_23 = value_states_47.contiguous()
        value_states_47 = None
        attn_output_92 = torch._C._nn.scaled_dot_product_attention(
            query_23,
            key_23,
            value_23,
            attn_mask=attention_mask_23,
            dropout_p=0.0,
            scale=1.0,
            is_causal=False,
        )
        query_23 = key_23 = value_23 = attention_mask_23 = None
        transpose_95 = attn_output_92.transpose(1, 2)
        attn_output_92 = None
        attn_output_93 = transpose_95.contiguous()
        transpose_95 = None
        reshape_47 = attn_output_93.reshape(1, 19, -1)
        attn_output_93 = None
        attn_output_94 = reshape_47.contiguous()
        reshape_47 = None
        attn_output_95 = torch._C._nn.linear(
            attn_output_94,
            l_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_94 = l_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_231 = torch.nn.functional.dropout(
            attn_output_95, p=0.1, training=False
        )
        attn_output_95 = None
        hidden_states_232 = hidden_states_230 + hidden_states_231
        hidden_states_230 = hidden_states_231 = None
        hidden_states_233 = torch.nn.functional.layer_norm(
            hidden_states_232,
            (1024,),
            l_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_232 = l_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_23_modules_self_attn_layer_norm_parameters_bias_ = (None)
        hidden_states_234 = hidden_states_233.reshape(-1, 1024)
        hidden_states_233 = None
        hidden_states_235 = torch._C._nn.linear(
            hidden_states_234,
            l_self_modules_model_modules_decoder_modules_layers_modules_23_modules_fc1_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_23_modules_fc1_parameters_bias_,
        )
        l_self_modules_model_modules_decoder_modules_layers_modules_23_modules_fc1_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_23_modules_fc1_parameters_bias_ = (None)
        hidden_states_236 = torch.nn.functional.relu(hidden_states_235, inplace=False)
        hidden_states_235 = None
        hidden_states_237 = torch._C._nn.linear(
            hidden_states_236,
            l_self_modules_model_modules_decoder_modules_layers_modules_23_modules_fc2_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_23_modules_fc2_parameters_bias_,
        )
        hidden_states_236 = l_self_modules_model_modules_decoder_modules_layers_modules_23_modules_fc2_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_23_modules_fc2_parameters_bias_ = (None)
        hidden_states_238 = torch.nn.functional.dropout(
            hidden_states_237, p=0.1, training=False
        )
        hidden_states_237 = None
        add_50 = hidden_states_234 + hidden_states_238
        hidden_states_234 = hidden_states_238 = None
        hidden_states_239 = add_50.view((1, 19, 1024))
        add_50 = None
        hidden_states_240 = torch.nn.functional.layer_norm(
            hidden_states_239,
            (1024,),
            l_self_modules_model_modules_decoder_modules_layers_modules_23_modules_final_layer_norm_parameters_weight_,
            l_self_modules_model_modules_decoder_modules_layers_modules_23_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_239 = l_self_modules_model_modules_decoder_modules_layers_modules_23_modules_final_layer_norm_parameters_weight_ = l_self_modules_model_modules_decoder_modules_layers_modules_23_modules_final_layer_norm_parameters_bias_ = (None)
        hidden_states_241 = torch._C._nn.linear(
            hidden_states_240,
            l_self_modules_model_modules_decoder_modules_project_out_parameters_weight_,
            None,
        )
        hidden_states_240 = (
            l_self_modules_model_modules_decoder_modules_project_out_parameters_weight_
        ) = None
        linear_146 = torch._C._nn.linear(
            hidden_states_241,
            l_self_modules_model_modules_decoder_modules_embed_tokens_parameters_weight_,
            None,
        )
        hidden_states_241 = (
            l_self_modules_model_modules_decoder_modules_embed_tokens_parameters_weight_
        ) = None
        logits = linear_146.contiguous()
        linear_146 = None
        return (logits,)
