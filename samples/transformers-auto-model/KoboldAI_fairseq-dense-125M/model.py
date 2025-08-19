import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_attention_mask_: torch.Tensor,
        L_inputs_embeds_: torch.Tensor,
        L_self_modules_embed_positions_buffers_weights_: torch.Tensor,
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
        L_self_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_attention_mask_ = L_attention_mask_
        l_inputs_embeds_ = L_inputs_embeds_
        l_self_modules_embed_positions_buffers_weights_ = (
            L_self_modules_embed_positions_buffers_weights_
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
        l_self_modules_layer_norm_parameters_weight_ = (
            L_self_modules_layer_norm_parameters_weight_
        )
        l_self_modules_layer_norm_parameters_bias_ = (
            L_self_modules_layer_norm_parameters_bias_
        )
        mask = torch.full(
            (9, 9), -3.4028234663852886e38, device=device(type="cuda", index=0)
        )
        mask_cond = torch.arange(9, device=device(type="cuda", index=0))
        add = mask_cond + 1
        view = add.view(9, 1)
        add = None
        lt = mask_cond < view
        mask_cond = view = None
        masked_fill_ = mask.masked_fill_(lt, 0)
        lt = masked_fill_ = None
        mask_1 = mask.to(torch.float32)
        mask = None
        getitem = mask_1[(None, None, slice(None, None, None), slice(None, None, None))]
        mask_1 = None
        causal_4d_mask = getitem.expand(1, 1, 9, 9)
        getitem = None
        getitem_1 = l_attention_mask_[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        l_attention_mask_ = None
        expand_1 = getitem_1.expand(1, 1, 9, 9)
        getitem_1 = None
        expanded_mask = expand_1.to(torch.float32)
        expand_1 = None
        tensor = torch.tensor(1.0, dtype=torch.float32)
        inverted_mask = tensor - expanded_mask
        tensor = expanded_mask = None
        to_2 = inverted_mask.to(torch.bool)
        masked_fill = inverted_mask.masked_fill(to_2, -3.4028234663852886e38)
        inverted_mask = to_2 = None
        expanded_attn_mask = masked_fill.to(device(type="cuda", index=0))
        masked_fill = None
        bool_1 = expanded_attn_mask.bool()
        expanded_attn_mask = None
        expanded_attn_mask_1 = causal_4d_mask.masked_fill(
            bool_1, -3.4028234663852886e38
        )
        causal_4d_mask = bool_1 = None
        position_ids = torch.arange(
            0, 9, dtype=torch.int64, device=device(type="cuda", index=0)
        )
        position_ids_1 = position_ids.unsqueeze(0)
        position_ids = None
        position_ids_1 += 2
        position_ids_2 = position_ids_1
        position_ids_1 = None
        view_1 = position_ids_2.view(-1)
        position_ids_2 = None
        index_select = l_self_modules_embed_positions_buffers_weights_.index_select(
            0, view_1
        )
        l_self_modules_embed_positions_buffers_weights_ = view_1 = None
        view_2 = index_select.view(1, 9, 768)
        index_select = None
        detach = view_2.detach()
        view_2 = None
        to_4 = detach.to(device(type="cuda", index=0))
        detach = None
        hidden_states = l_inputs_embeds_ + to_4
        l_inputs_embeds_ = to_4 = None
        hidden_states_1 = torch.nn.functional.dropout(
            hidden_states, p=0.1, training=False
        )
        hidden_states = None
        hidden_states_2 = torch.nn.functional.layer_norm(
            hidden_states_1,
            (768,),
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
        query_states = linear * 0.125
        linear = None
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
        view_3 = key_states.view(1, 9, -1, 64)
        key_states = None
        key_states_1 = view_3.transpose(1, 2)
        view_3 = None
        view_4 = value_states.view(1, 9, -1, 64)
        value_states = None
        value_states_1 = view_4.transpose(1, 2)
        view_4 = None
        view_5 = query_states.view(1, 9, 12, 64)
        query_states = None
        query_states_1 = view_5.transpose(1, 2)
        view_5 = None
        query_states_2 = query_states_1.reshape(12, -1, 64)
        query_states_1 = None
        key_states_2 = key_states_1.reshape(12, -1, 64)
        value_states_2 = value_states_1.reshape(12, -1, 64)
        transpose_3 = key_states_2.transpose(1, 2)
        key_states_2 = None
        attn_weights = torch.bmm(query_states_2, transpose_3)
        query_states_2 = transpose_3 = None
        view_6 = attn_weights.view(1, 12, 9, 9)
        attn_weights = None
        attn_weights_1 = view_6 + expanded_attn_mask_1
        view_6 = None
        tensor_1 = torch.tensor(
            -3.4028234663852886e38, device=device(type="cuda", index=0)
        )
        attn_weights_2 = torch.max(attn_weights_1, tensor_1)
        attn_weights_1 = tensor_1 = None
        attn_weights_3 = attn_weights_2.view(12, 9, 9)
        attn_weights_2 = None
        attn_weights_4 = torch.nn.functional.softmax(attn_weights_3, dim=-1)
        attn_weights_3 = None
        attn_probs = torch.nn.functional.dropout(attn_weights_4, p=0.1, training=False)
        attn_weights_4 = None
        attn_output = torch.bmm(attn_probs, value_states_2)
        attn_probs = value_states_2 = None
        attn_output_1 = attn_output.view(1, 12, 9, 64)
        attn_output = None
        attn_output_2 = attn_output_1.transpose(1, 2)
        attn_output_1 = None
        attn_output_3 = attn_output_2.reshape(1, 9, 768)
        attn_output_2 = None
        attn_output_4 = torch._C._nn.linear(
            attn_output_3,
            l_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_3 = l_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_3 = torch.nn.functional.dropout(
            attn_output_4, p=0.1, training=False
        )
        attn_output_4 = None
        hidden_states_4 = hidden_states_1 + hidden_states_3
        hidden_states_1 = hidden_states_3 = None
        hidden_states_5 = torch.nn.functional.layer_norm(
            hidden_states_4,
            (768,),
            l_self_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_ = (
            l_self_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_
        ) = None
        linear_4 = torch._C._nn.linear(
            hidden_states_5,
            l_self_modules_layers_modules_0_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_0_modules_fc1_parameters_bias_,
        )
        hidden_states_5 = (
            l_self_modules_layers_modules_0_modules_fc1_parameters_weight_
        ) = l_self_modules_layers_modules_0_modules_fc1_parameters_bias_ = None
        hidden_states_6 = torch._C._nn.gelu(linear_4)
        linear_4 = None
        hidden_states_7 = torch.nn.functional.dropout(
            hidden_states_6, p=0.0, training=False
        )
        hidden_states_6 = None
        hidden_states_8 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_layers_modules_0_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_0_modules_fc2_parameters_bias_,
        )
        hidden_states_7 = (
            l_self_modules_layers_modules_0_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_0_modules_fc2_parameters_bias_ = None
        hidden_states_9 = torch.nn.functional.dropout(
            hidden_states_8, p=0.1, training=False
        )
        hidden_states_8 = None
        hidden_states_10 = hidden_states_4 + hidden_states_9
        hidden_states_4 = hidden_states_9 = None
        hidden_states_11 = torch.nn.functional.layer_norm(
            hidden_states_10,
            (768,),
            l_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_6 = torch._C._nn.linear(
            hidden_states_11,
            l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_3 = linear_6 * 0.125
        linear_6 = None
        key_states_3 = torch._C._nn.linear(
            hidden_states_11,
            l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_3 = torch._C._nn.linear(
            hidden_states_11,
            l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_11 = l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_9 = key_states_3.view(1, 9, -1, 64)
        key_states_3 = None
        key_states_4 = view_9.transpose(1, 2)
        view_9 = None
        view_10 = value_states_3.view(1, 9, -1, 64)
        value_states_3 = None
        value_states_4 = view_10.transpose(1, 2)
        view_10 = None
        view_11 = query_states_3.view(1, 9, 12, 64)
        query_states_3 = None
        query_states_4 = view_11.transpose(1, 2)
        view_11 = None
        query_states_5 = query_states_4.reshape(12, -1, 64)
        query_states_4 = None
        key_states_5 = key_states_4.reshape(12, -1, 64)
        value_states_5 = value_states_4.reshape(12, -1, 64)
        transpose_8 = key_states_5.transpose(1, 2)
        key_states_5 = None
        attn_weights_5 = torch.bmm(query_states_5, transpose_8)
        query_states_5 = transpose_8 = None
        view_12 = attn_weights_5.view(1, 12, 9, 9)
        attn_weights_5 = None
        attn_weights_6 = view_12 + expanded_attn_mask_1
        view_12 = None
        tensor_2 = torch.tensor(
            -3.4028234663852886e38, device=device(type="cuda", index=0)
        )
        attn_weights_7 = torch.max(attn_weights_6, tensor_2)
        attn_weights_6 = tensor_2 = None
        attn_weights_8 = attn_weights_7.view(12, 9, 9)
        attn_weights_7 = None
        attn_weights_9 = torch.nn.functional.softmax(attn_weights_8, dim=-1)
        attn_weights_8 = None
        attn_probs_1 = torch.nn.functional.dropout(
            attn_weights_9, p=0.1, training=False
        )
        attn_weights_9 = None
        attn_output_5 = torch.bmm(attn_probs_1, value_states_5)
        attn_probs_1 = value_states_5 = None
        attn_output_6 = attn_output_5.view(1, 12, 9, 64)
        attn_output_5 = None
        attn_output_7 = attn_output_6.transpose(1, 2)
        attn_output_6 = None
        attn_output_8 = attn_output_7.reshape(1, 9, 768)
        attn_output_7 = None
        attn_output_9 = torch._C._nn.linear(
            attn_output_8,
            l_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_8 = l_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_12 = torch.nn.functional.dropout(
            attn_output_9, p=0.1, training=False
        )
        attn_output_9 = None
        hidden_states_13 = hidden_states_10 + hidden_states_12
        hidden_states_10 = hidden_states_12 = None
        hidden_states_14 = torch.nn.functional.layer_norm(
            hidden_states_13,
            (768,),
            l_self_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_ = (
            l_self_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_
        ) = None
        linear_10 = torch._C._nn.linear(
            hidden_states_14,
            l_self_modules_layers_modules_1_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_1_modules_fc1_parameters_bias_,
        )
        hidden_states_14 = (
            l_self_modules_layers_modules_1_modules_fc1_parameters_weight_
        ) = l_self_modules_layers_modules_1_modules_fc1_parameters_bias_ = None
        hidden_states_15 = torch._C._nn.gelu(linear_10)
        linear_10 = None
        hidden_states_16 = torch.nn.functional.dropout(
            hidden_states_15, p=0.0, training=False
        )
        hidden_states_15 = None
        hidden_states_17 = torch._C._nn.linear(
            hidden_states_16,
            l_self_modules_layers_modules_1_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_1_modules_fc2_parameters_bias_,
        )
        hidden_states_16 = (
            l_self_modules_layers_modules_1_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_1_modules_fc2_parameters_bias_ = None
        hidden_states_18 = torch.nn.functional.dropout(
            hidden_states_17, p=0.1, training=False
        )
        hidden_states_17 = None
        hidden_states_19 = hidden_states_13 + hidden_states_18
        hidden_states_13 = hidden_states_18 = None
        hidden_states_20 = torch.nn.functional.layer_norm(
            hidden_states_19,
            (768,),
            l_self_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_12 = torch._C._nn.linear(
            hidden_states_20,
            l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_6 = linear_12 * 0.125
        linear_12 = None
        key_states_6 = torch._C._nn.linear(
            hidden_states_20,
            l_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_6 = torch._C._nn.linear(
            hidden_states_20,
            l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_20 = l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_15 = key_states_6.view(1, 9, -1, 64)
        key_states_6 = None
        key_states_7 = view_15.transpose(1, 2)
        view_15 = None
        view_16 = value_states_6.view(1, 9, -1, 64)
        value_states_6 = None
        value_states_7 = view_16.transpose(1, 2)
        view_16 = None
        view_17 = query_states_6.view(1, 9, 12, 64)
        query_states_6 = None
        query_states_7 = view_17.transpose(1, 2)
        view_17 = None
        query_states_8 = query_states_7.reshape(12, -1, 64)
        query_states_7 = None
        key_states_8 = key_states_7.reshape(12, -1, 64)
        value_states_8 = value_states_7.reshape(12, -1, 64)
        transpose_13 = key_states_8.transpose(1, 2)
        key_states_8 = None
        attn_weights_10 = torch.bmm(query_states_8, transpose_13)
        query_states_8 = transpose_13 = None
        view_18 = attn_weights_10.view(1, 12, 9, 9)
        attn_weights_10 = None
        attn_weights_11 = view_18 + expanded_attn_mask_1
        view_18 = None
        tensor_3 = torch.tensor(
            -3.4028234663852886e38, device=device(type="cuda", index=0)
        )
        attn_weights_12 = torch.max(attn_weights_11, tensor_3)
        attn_weights_11 = tensor_3 = None
        attn_weights_13 = attn_weights_12.view(12, 9, 9)
        attn_weights_12 = None
        attn_weights_14 = torch.nn.functional.softmax(attn_weights_13, dim=-1)
        attn_weights_13 = None
        attn_probs_2 = torch.nn.functional.dropout(
            attn_weights_14, p=0.1, training=False
        )
        attn_weights_14 = None
        attn_output_10 = torch.bmm(attn_probs_2, value_states_8)
        attn_probs_2 = value_states_8 = None
        attn_output_11 = attn_output_10.view(1, 12, 9, 64)
        attn_output_10 = None
        attn_output_12 = attn_output_11.transpose(1, 2)
        attn_output_11 = None
        attn_output_13 = attn_output_12.reshape(1, 9, 768)
        attn_output_12 = None
        attn_output_14 = torch._C._nn.linear(
            attn_output_13,
            l_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_13 = l_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_21 = torch.nn.functional.dropout(
            attn_output_14, p=0.1, training=False
        )
        attn_output_14 = None
        hidden_states_22 = hidden_states_19 + hidden_states_21
        hidden_states_19 = hidden_states_21 = None
        hidden_states_23 = torch.nn.functional.layer_norm(
            hidden_states_22,
            (768,),
            l_self_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_ = (
            l_self_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_
        ) = None
        linear_16 = torch._C._nn.linear(
            hidden_states_23,
            l_self_modules_layers_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_fc1_parameters_bias_,
        )
        hidden_states_23 = (
            l_self_modules_layers_modules_2_modules_fc1_parameters_weight_
        ) = l_self_modules_layers_modules_2_modules_fc1_parameters_bias_ = None
        hidden_states_24 = torch._C._nn.gelu(linear_16)
        linear_16 = None
        hidden_states_25 = torch.nn.functional.dropout(
            hidden_states_24, p=0.0, training=False
        )
        hidden_states_24 = None
        hidden_states_26 = torch._C._nn.linear(
            hidden_states_25,
            l_self_modules_layers_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_fc2_parameters_bias_,
        )
        hidden_states_25 = (
            l_self_modules_layers_modules_2_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_2_modules_fc2_parameters_bias_ = None
        hidden_states_27 = torch.nn.functional.dropout(
            hidden_states_26, p=0.1, training=False
        )
        hidden_states_26 = None
        hidden_states_28 = hidden_states_22 + hidden_states_27
        hidden_states_22 = hidden_states_27 = None
        hidden_states_29 = torch.nn.functional.layer_norm(
            hidden_states_28,
            (768,),
            l_self_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_18 = torch._C._nn.linear(
            hidden_states_29,
            l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_9 = linear_18 * 0.125
        linear_18 = None
        key_states_9 = torch._C._nn.linear(
            hidden_states_29,
            l_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_9 = torch._C._nn.linear(
            hidden_states_29,
            l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_29 = l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_21 = key_states_9.view(1, 9, -1, 64)
        key_states_9 = None
        key_states_10 = view_21.transpose(1, 2)
        view_21 = None
        view_22 = value_states_9.view(1, 9, -1, 64)
        value_states_9 = None
        value_states_10 = view_22.transpose(1, 2)
        view_22 = None
        view_23 = query_states_9.view(1, 9, 12, 64)
        query_states_9 = None
        query_states_10 = view_23.transpose(1, 2)
        view_23 = None
        query_states_11 = query_states_10.reshape(12, -1, 64)
        query_states_10 = None
        key_states_11 = key_states_10.reshape(12, -1, 64)
        value_states_11 = value_states_10.reshape(12, -1, 64)
        transpose_18 = key_states_11.transpose(1, 2)
        key_states_11 = None
        attn_weights_15 = torch.bmm(query_states_11, transpose_18)
        query_states_11 = transpose_18 = None
        view_24 = attn_weights_15.view(1, 12, 9, 9)
        attn_weights_15 = None
        attn_weights_16 = view_24 + expanded_attn_mask_1
        view_24 = None
        tensor_4 = torch.tensor(
            -3.4028234663852886e38, device=device(type="cuda", index=0)
        )
        attn_weights_17 = torch.max(attn_weights_16, tensor_4)
        attn_weights_16 = tensor_4 = None
        attn_weights_18 = attn_weights_17.view(12, 9, 9)
        attn_weights_17 = None
        attn_weights_19 = torch.nn.functional.softmax(attn_weights_18, dim=-1)
        attn_weights_18 = None
        attn_probs_3 = torch.nn.functional.dropout(
            attn_weights_19, p=0.1, training=False
        )
        attn_weights_19 = None
        attn_output_15 = torch.bmm(attn_probs_3, value_states_11)
        attn_probs_3 = value_states_11 = None
        attn_output_16 = attn_output_15.view(1, 12, 9, 64)
        attn_output_15 = None
        attn_output_17 = attn_output_16.transpose(1, 2)
        attn_output_16 = None
        attn_output_18 = attn_output_17.reshape(1, 9, 768)
        attn_output_17 = None
        attn_output_19 = torch._C._nn.linear(
            attn_output_18,
            l_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_18 = l_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_30 = torch.nn.functional.dropout(
            attn_output_19, p=0.1, training=False
        )
        attn_output_19 = None
        hidden_states_31 = hidden_states_28 + hidden_states_30
        hidden_states_28 = hidden_states_30 = None
        hidden_states_32 = torch.nn.functional.layer_norm(
            hidden_states_31,
            (768,),
            l_self_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_ = (
            l_self_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_
        ) = None
        linear_22 = torch._C._nn.linear(
            hidden_states_32,
            l_self_modules_layers_modules_3_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_3_modules_fc1_parameters_bias_,
        )
        hidden_states_32 = (
            l_self_modules_layers_modules_3_modules_fc1_parameters_weight_
        ) = l_self_modules_layers_modules_3_modules_fc1_parameters_bias_ = None
        hidden_states_33 = torch._C._nn.gelu(linear_22)
        linear_22 = None
        hidden_states_34 = torch.nn.functional.dropout(
            hidden_states_33, p=0.0, training=False
        )
        hidden_states_33 = None
        hidden_states_35 = torch._C._nn.linear(
            hidden_states_34,
            l_self_modules_layers_modules_3_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_3_modules_fc2_parameters_bias_,
        )
        hidden_states_34 = (
            l_self_modules_layers_modules_3_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_3_modules_fc2_parameters_bias_ = None
        hidden_states_36 = torch.nn.functional.dropout(
            hidden_states_35, p=0.1, training=False
        )
        hidden_states_35 = None
        hidden_states_37 = hidden_states_31 + hidden_states_36
        hidden_states_31 = hidden_states_36 = None
        hidden_states_38 = torch.nn.functional.layer_norm(
            hidden_states_37,
            (768,),
            l_self_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_24 = torch._C._nn.linear(
            hidden_states_38,
            l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_12 = linear_24 * 0.125
        linear_24 = None
        key_states_12 = torch._C._nn.linear(
            hidden_states_38,
            l_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_12 = torch._C._nn.linear(
            hidden_states_38,
            l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_38 = l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_27 = key_states_12.view(1, 9, -1, 64)
        key_states_12 = None
        key_states_13 = view_27.transpose(1, 2)
        view_27 = None
        view_28 = value_states_12.view(1, 9, -1, 64)
        value_states_12 = None
        value_states_13 = view_28.transpose(1, 2)
        view_28 = None
        view_29 = query_states_12.view(1, 9, 12, 64)
        query_states_12 = None
        query_states_13 = view_29.transpose(1, 2)
        view_29 = None
        query_states_14 = query_states_13.reshape(12, -1, 64)
        query_states_13 = None
        key_states_14 = key_states_13.reshape(12, -1, 64)
        value_states_14 = value_states_13.reshape(12, -1, 64)
        transpose_23 = key_states_14.transpose(1, 2)
        key_states_14 = None
        attn_weights_20 = torch.bmm(query_states_14, transpose_23)
        query_states_14 = transpose_23 = None
        view_30 = attn_weights_20.view(1, 12, 9, 9)
        attn_weights_20 = None
        attn_weights_21 = view_30 + expanded_attn_mask_1
        view_30 = None
        tensor_5 = torch.tensor(
            -3.4028234663852886e38, device=device(type="cuda", index=0)
        )
        attn_weights_22 = torch.max(attn_weights_21, tensor_5)
        attn_weights_21 = tensor_5 = None
        attn_weights_23 = attn_weights_22.view(12, 9, 9)
        attn_weights_22 = None
        attn_weights_24 = torch.nn.functional.softmax(attn_weights_23, dim=-1)
        attn_weights_23 = None
        attn_probs_4 = torch.nn.functional.dropout(
            attn_weights_24, p=0.1, training=False
        )
        attn_weights_24 = None
        attn_output_20 = torch.bmm(attn_probs_4, value_states_14)
        attn_probs_4 = value_states_14 = None
        attn_output_21 = attn_output_20.view(1, 12, 9, 64)
        attn_output_20 = None
        attn_output_22 = attn_output_21.transpose(1, 2)
        attn_output_21 = None
        attn_output_23 = attn_output_22.reshape(1, 9, 768)
        attn_output_22 = None
        attn_output_24 = torch._C._nn.linear(
            attn_output_23,
            l_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_23 = l_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_39 = torch.nn.functional.dropout(
            attn_output_24, p=0.1, training=False
        )
        attn_output_24 = None
        hidden_states_40 = hidden_states_37 + hidden_states_39
        hidden_states_37 = hidden_states_39 = None
        hidden_states_41 = torch.nn.functional.layer_norm(
            hidden_states_40,
            (768,),
            l_self_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_ = (
            l_self_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_
        ) = None
        linear_28 = torch._C._nn.linear(
            hidden_states_41,
            l_self_modules_layers_modules_4_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_4_modules_fc1_parameters_bias_,
        )
        hidden_states_41 = (
            l_self_modules_layers_modules_4_modules_fc1_parameters_weight_
        ) = l_self_modules_layers_modules_4_modules_fc1_parameters_bias_ = None
        hidden_states_42 = torch._C._nn.gelu(linear_28)
        linear_28 = None
        hidden_states_43 = torch.nn.functional.dropout(
            hidden_states_42, p=0.0, training=False
        )
        hidden_states_42 = None
        hidden_states_44 = torch._C._nn.linear(
            hidden_states_43,
            l_self_modules_layers_modules_4_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_4_modules_fc2_parameters_bias_,
        )
        hidden_states_43 = (
            l_self_modules_layers_modules_4_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_4_modules_fc2_parameters_bias_ = None
        hidden_states_45 = torch.nn.functional.dropout(
            hidden_states_44, p=0.1, training=False
        )
        hidden_states_44 = None
        hidden_states_46 = hidden_states_40 + hidden_states_45
        hidden_states_40 = hidden_states_45 = None
        hidden_states_47 = torch.nn.functional.layer_norm(
            hidden_states_46,
            (768,),
            l_self_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_30 = torch._C._nn.linear(
            hidden_states_47,
            l_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_15 = linear_30 * 0.125
        linear_30 = None
        key_states_15 = torch._C._nn.linear(
            hidden_states_47,
            l_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_15 = torch._C._nn.linear(
            hidden_states_47,
            l_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_47 = l_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_33 = key_states_15.view(1, 9, -1, 64)
        key_states_15 = None
        key_states_16 = view_33.transpose(1, 2)
        view_33 = None
        view_34 = value_states_15.view(1, 9, -1, 64)
        value_states_15 = None
        value_states_16 = view_34.transpose(1, 2)
        view_34 = None
        view_35 = query_states_15.view(1, 9, 12, 64)
        query_states_15 = None
        query_states_16 = view_35.transpose(1, 2)
        view_35 = None
        query_states_17 = query_states_16.reshape(12, -1, 64)
        query_states_16 = None
        key_states_17 = key_states_16.reshape(12, -1, 64)
        value_states_17 = value_states_16.reshape(12, -1, 64)
        transpose_28 = key_states_17.transpose(1, 2)
        key_states_17 = None
        attn_weights_25 = torch.bmm(query_states_17, transpose_28)
        query_states_17 = transpose_28 = None
        view_36 = attn_weights_25.view(1, 12, 9, 9)
        attn_weights_25 = None
        attn_weights_26 = view_36 + expanded_attn_mask_1
        view_36 = None
        tensor_6 = torch.tensor(
            -3.4028234663852886e38, device=device(type="cuda", index=0)
        )
        attn_weights_27 = torch.max(attn_weights_26, tensor_6)
        attn_weights_26 = tensor_6 = None
        attn_weights_28 = attn_weights_27.view(12, 9, 9)
        attn_weights_27 = None
        attn_weights_29 = torch.nn.functional.softmax(attn_weights_28, dim=-1)
        attn_weights_28 = None
        attn_probs_5 = torch.nn.functional.dropout(
            attn_weights_29, p=0.1, training=False
        )
        attn_weights_29 = None
        attn_output_25 = torch.bmm(attn_probs_5, value_states_17)
        attn_probs_5 = value_states_17 = None
        attn_output_26 = attn_output_25.view(1, 12, 9, 64)
        attn_output_25 = None
        attn_output_27 = attn_output_26.transpose(1, 2)
        attn_output_26 = None
        attn_output_28 = attn_output_27.reshape(1, 9, 768)
        attn_output_27 = None
        attn_output_29 = torch._C._nn.linear(
            attn_output_28,
            l_self_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_28 = l_self_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_48 = torch.nn.functional.dropout(
            attn_output_29, p=0.1, training=False
        )
        attn_output_29 = None
        hidden_states_49 = hidden_states_46 + hidden_states_48
        hidden_states_46 = hidden_states_48 = None
        hidden_states_50 = torch.nn.functional.layer_norm(
            hidden_states_49,
            (768,),
            l_self_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_ = (
            l_self_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_
        ) = None
        linear_34 = torch._C._nn.linear(
            hidden_states_50,
            l_self_modules_layers_modules_5_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_5_modules_fc1_parameters_bias_,
        )
        hidden_states_50 = (
            l_self_modules_layers_modules_5_modules_fc1_parameters_weight_
        ) = l_self_modules_layers_modules_5_modules_fc1_parameters_bias_ = None
        hidden_states_51 = torch._C._nn.gelu(linear_34)
        linear_34 = None
        hidden_states_52 = torch.nn.functional.dropout(
            hidden_states_51, p=0.0, training=False
        )
        hidden_states_51 = None
        hidden_states_53 = torch._C._nn.linear(
            hidden_states_52,
            l_self_modules_layers_modules_5_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_5_modules_fc2_parameters_bias_,
        )
        hidden_states_52 = (
            l_self_modules_layers_modules_5_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_5_modules_fc2_parameters_bias_ = None
        hidden_states_54 = torch.nn.functional.dropout(
            hidden_states_53, p=0.1, training=False
        )
        hidden_states_53 = None
        hidden_states_55 = hidden_states_49 + hidden_states_54
        hidden_states_49 = hidden_states_54 = None
        hidden_states_56 = torch.nn.functional.layer_norm(
            hidden_states_55,
            (768,),
            l_self_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_36 = torch._C._nn.linear(
            hidden_states_56,
            l_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_18 = linear_36 * 0.125
        linear_36 = None
        key_states_18 = torch._C._nn.linear(
            hidden_states_56,
            l_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_18 = torch._C._nn.linear(
            hidden_states_56,
            l_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_56 = l_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_39 = key_states_18.view(1, 9, -1, 64)
        key_states_18 = None
        key_states_19 = view_39.transpose(1, 2)
        view_39 = None
        view_40 = value_states_18.view(1, 9, -1, 64)
        value_states_18 = None
        value_states_19 = view_40.transpose(1, 2)
        view_40 = None
        view_41 = query_states_18.view(1, 9, 12, 64)
        query_states_18 = None
        query_states_19 = view_41.transpose(1, 2)
        view_41 = None
        query_states_20 = query_states_19.reshape(12, -1, 64)
        query_states_19 = None
        key_states_20 = key_states_19.reshape(12, -1, 64)
        value_states_20 = value_states_19.reshape(12, -1, 64)
        transpose_33 = key_states_20.transpose(1, 2)
        key_states_20 = None
        attn_weights_30 = torch.bmm(query_states_20, transpose_33)
        query_states_20 = transpose_33 = None
        view_42 = attn_weights_30.view(1, 12, 9, 9)
        attn_weights_30 = None
        attn_weights_31 = view_42 + expanded_attn_mask_1
        view_42 = None
        tensor_7 = torch.tensor(
            -3.4028234663852886e38, device=device(type="cuda", index=0)
        )
        attn_weights_32 = torch.max(attn_weights_31, tensor_7)
        attn_weights_31 = tensor_7 = None
        attn_weights_33 = attn_weights_32.view(12, 9, 9)
        attn_weights_32 = None
        attn_weights_34 = torch.nn.functional.softmax(attn_weights_33, dim=-1)
        attn_weights_33 = None
        attn_probs_6 = torch.nn.functional.dropout(
            attn_weights_34, p=0.1, training=False
        )
        attn_weights_34 = None
        attn_output_30 = torch.bmm(attn_probs_6, value_states_20)
        attn_probs_6 = value_states_20 = None
        attn_output_31 = attn_output_30.view(1, 12, 9, 64)
        attn_output_30 = None
        attn_output_32 = attn_output_31.transpose(1, 2)
        attn_output_31 = None
        attn_output_33 = attn_output_32.reshape(1, 9, 768)
        attn_output_32 = None
        attn_output_34 = torch._C._nn.linear(
            attn_output_33,
            l_self_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_33 = l_self_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_57 = torch.nn.functional.dropout(
            attn_output_34, p=0.1, training=False
        )
        attn_output_34 = None
        hidden_states_58 = hidden_states_55 + hidden_states_57
        hidden_states_55 = hidden_states_57 = None
        hidden_states_59 = torch.nn.functional.layer_norm(
            hidden_states_58,
            (768,),
            l_self_modules_layers_modules_6_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_6_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_6_modules_final_layer_norm_parameters_weight_ = (
            l_self_modules_layers_modules_6_modules_final_layer_norm_parameters_bias_
        ) = None
        linear_40 = torch._C._nn.linear(
            hidden_states_59,
            l_self_modules_layers_modules_6_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_6_modules_fc1_parameters_bias_,
        )
        hidden_states_59 = (
            l_self_modules_layers_modules_6_modules_fc1_parameters_weight_
        ) = l_self_modules_layers_modules_6_modules_fc1_parameters_bias_ = None
        hidden_states_60 = torch._C._nn.gelu(linear_40)
        linear_40 = None
        hidden_states_61 = torch.nn.functional.dropout(
            hidden_states_60, p=0.0, training=False
        )
        hidden_states_60 = None
        hidden_states_62 = torch._C._nn.linear(
            hidden_states_61,
            l_self_modules_layers_modules_6_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_6_modules_fc2_parameters_bias_,
        )
        hidden_states_61 = (
            l_self_modules_layers_modules_6_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_6_modules_fc2_parameters_bias_ = None
        hidden_states_63 = torch.nn.functional.dropout(
            hidden_states_62, p=0.1, training=False
        )
        hidden_states_62 = None
        hidden_states_64 = hidden_states_58 + hidden_states_63
        hidden_states_58 = hidden_states_63 = None
        hidden_states_65 = torch.nn.functional.layer_norm(
            hidden_states_64,
            (768,),
            l_self_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_42 = torch._C._nn.linear(
            hidden_states_65,
            l_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_21 = linear_42 * 0.125
        linear_42 = None
        key_states_21 = torch._C._nn.linear(
            hidden_states_65,
            l_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_21 = torch._C._nn.linear(
            hidden_states_65,
            l_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_65 = l_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_45 = key_states_21.view(1, 9, -1, 64)
        key_states_21 = None
        key_states_22 = view_45.transpose(1, 2)
        view_45 = None
        view_46 = value_states_21.view(1, 9, -1, 64)
        value_states_21 = None
        value_states_22 = view_46.transpose(1, 2)
        view_46 = None
        view_47 = query_states_21.view(1, 9, 12, 64)
        query_states_21 = None
        query_states_22 = view_47.transpose(1, 2)
        view_47 = None
        query_states_23 = query_states_22.reshape(12, -1, 64)
        query_states_22 = None
        key_states_23 = key_states_22.reshape(12, -1, 64)
        value_states_23 = value_states_22.reshape(12, -1, 64)
        transpose_38 = key_states_23.transpose(1, 2)
        key_states_23 = None
        attn_weights_35 = torch.bmm(query_states_23, transpose_38)
        query_states_23 = transpose_38 = None
        view_48 = attn_weights_35.view(1, 12, 9, 9)
        attn_weights_35 = None
        attn_weights_36 = view_48 + expanded_attn_mask_1
        view_48 = None
        tensor_8 = torch.tensor(
            -3.4028234663852886e38, device=device(type="cuda", index=0)
        )
        attn_weights_37 = torch.max(attn_weights_36, tensor_8)
        attn_weights_36 = tensor_8 = None
        attn_weights_38 = attn_weights_37.view(12, 9, 9)
        attn_weights_37 = None
        attn_weights_39 = torch.nn.functional.softmax(attn_weights_38, dim=-1)
        attn_weights_38 = None
        attn_probs_7 = torch.nn.functional.dropout(
            attn_weights_39, p=0.1, training=False
        )
        attn_weights_39 = None
        attn_output_35 = torch.bmm(attn_probs_7, value_states_23)
        attn_probs_7 = value_states_23 = None
        attn_output_36 = attn_output_35.view(1, 12, 9, 64)
        attn_output_35 = None
        attn_output_37 = attn_output_36.transpose(1, 2)
        attn_output_36 = None
        attn_output_38 = attn_output_37.reshape(1, 9, 768)
        attn_output_37 = None
        attn_output_39 = torch._C._nn.linear(
            attn_output_38,
            l_self_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_38 = l_self_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_66 = torch.nn.functional.dropout(
            attn_output_39, p=0.1, training=False
        )
        attn_output_39 = None
        hidden_states_67 = hidden_states_64 + hidden_states_66
        hidden_states_64 = hidden_states_66 = None
        hidden_states_68 = torch.nn.functional.layer_norm(
            hidden_states_67,
            (768,),
            l_self_modules_layers_modules_7_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_7_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_7_modules_final_layer_norm_parameters_weight_ = (
            l_self_modules_layers_modules_7_modules_final_layer_norm_parameters_bias_
        ) = None
        linear_46 = torch._C._nn.linear(
            hidden_states_68,
            l_self_modules_layers_modules_7_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_7_modules_fc1_parameters_bias_,
        )
        hidden_states_68 = (
            l_self_modules_layers_modules_7_modules_fc1_parameters_weight_
        ) = l_self_modules_layers_modules_7_modules_fc1_parameters_bias_ = None
        hidden_states_69 = torch._C._nn.gelu(linear_46)
        linear_46 = None
        hidden_states_70 = torch.nn.functional.dropout(
            hidden_states_69, p=0.0, training=False
        )
        hidden_states_69 = None
        hidden_states_71 = torch._C._nn.linear(
            hidden_states_70,
            l_self_modules_layers_modules_7_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_7_modules_fc2_parameters_bias_,
        )
        hidden_states_70 = (
            l_self_modules_layers_modules_7_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_7_modules_fc2_parameters_bias_ = None
        hidden_states_72 = torch.nn.functional.dropout(
            hidden_states_71, p=0.1, training=False
        )
        hidden_states_71 = None
        hidden_states_73 = hidden_states_67 + hidden_states_72
        hidden_states_67 = hidden_states_72 = None
        hidden_states_74 = torch.nn.functional.layer_norm(
            hidden_states_73,
            (768,),
            l_self_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_48 = torch._C._nn.linear(
            hidden_states_74,
            l_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_24 = linear_48 * 0.125
        linear_48 = None
        key_states_24 = torch._C._nn.linear(
            hidden_states_74,
            l_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_24 = torch._C._nn.linear(
            hidden_states_74,
            l_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_74 = l_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_51 = key_states_24.view(1, 9, -1, 64)
        key_states_24 = None
        key_states_25 = view_51.transpose(1, 2)
        view_51 = None
        view_52 = value_states_24.view(1, 9, -1, 64)
        value_states_24 = None
        value_states_25 = view_52.transpose(1, 2)
        view_52 = None
        view_53 = query_states_24.view(1, 9, 12, 64)
        query_states_24 = None
        query_states_25 = view_53.transpose(1, 2)
        view_53 = None
        query_states_26 = query_states_25.reshape(12, -1, 64)
        query_states_25 = None
        key_states_26 = key_states_25.reshape(12, -1, 64)
        value_states_26 = value_states_25.reshape(12, -1, 64)
        transpose_43 = key_states_26.transpose(1, 2)
        key_states_26 = None
        attn_weights_40 = torch.bmm(query_states_26, transpose_43)
        query_states_26 = transpose_43 = None
        view_54 = attn_weights_40.view(1, 12, 9, 9)
        attn_weights_40 = None
        attn_weights_41 = view_54 + expanded_attn_mask_1
        view_54 = None
        tensor_9 = torch.tensor(
            -3.4028234663852886e38, device=device(type="cuda", index=0)
        )
        attn_weights_42 = torch.max(attn_weights_41, tensor_9)
        attn_weights_41 = tensor_9 = None
        attn_weights_43 = attn_weights_42.view(12, 9, 9)
        attn_weights_42 = None
        attn_weights_44 = torch.nn.functional.softmax(attn_weights_43, dim=-1)
        attn_weights_43 = None
        attn_probs_8 = torch.nn.functional.dropout(
            attn_weights_44, p=0.1, training=False
        )
        attn_weights_44 = None
        attn_output_40 = torch.bmm(attn_probs_8, value_states_26)
        attn_probs_8 = value_states_26 = None
        attn_output_41 = attn_output_40.view(1, 12, 9, 64)
        attn_output_40 = None
        attn_output_42 = attn_output_41.transpose(1, 2)
        attn_output_41 = None
        attn_output_43 = attn_output_42.reshape(1, 9, 768)
        attn_output_42 = None
        attn_output_44 = torch._C._nn.linear(
            attn_output_43,
            l_self_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_43 = l_self_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_75 = torch.nn.functional.dropout(
            attn_output_44, p=0.1, training=False
        )
        attn_output_44 = None
        hidden_states_76 = hidden_states_73 + hidden_states_75
        hidden_states_73 = hidden_states_75 = None
        hidden_states_77 = torch.nn.functional.layer_norm(
            hidden_states_76,
            (768,),
            l_self_modules_layers_modules_8_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_8_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_8_modules_final_layer_norm_parameters_weight_ = (
            l_self_modules_layers_modules_8_modules_final_layer_norm_parameters_bias_
        ) = None
        linear_52 = torch._C._nn.linear(
            hidden_states_77,
            l_self_modules_layers_modules_8_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_8_modules_fc1_parameters_bias_,
        )
        hidden_states_77 = (
            l_self_modules_layers_modules_8_modules_fc1_parameters_weight_
        ) = l_self_modules_layers_modules_8_modules_fc1_parameters_bias_ = None
        hidden_states_78 = torch._C._nn.gelu(linear_52)
        linear_52 = None
        hidden_states_79 = torch.nn.functional.dropout(
            hidden_states_78, p=0.0, training=False
        )
        hidden_states_78 = None
        hidden_states_80 = torch._C._nn.linear(
            hidden_states_79,
            l_self_modules_layers_modules_8_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_8_modules_fc2_parameters_bias_,
        )
        hidden_states_79 = (
            l_self_modules_layers_modules_8_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_8_modules_fc2_parameters_bias_ = None
        hidden_states_81 = torch.nn.functional.dropout(
            hidden_states_80, p=0.1, training=False
        )
        hidden_states_80 = None
        hidden_states_82 = hidden_states_76 + hidden_states_81
        hidden_states_76 = hidden_states_81 = None
        hidden_states_83 = torch.nn.functional.layer_norm(
            hidden_states_82,
            (768,),
            l_self_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_54 = torch._C._nn.linear(
            hidden_states_83,
            l_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_27 = linear_54 * 0.125
        linear_54 = None
        key_states_27 = torch._C._nn.linear(
            hidden_states_83,
            l_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_27 = torch._C._nn.linear(
            hidden_states_83,
            l_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_83 = l_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_57 = key_states_27.view(1, 9, -1, 64)
        key_states_27 = None
        key_states_28 = view_57.transpose(1, 2)
        view_57 = None
        view_58 = value_states_27.view(1, 9, -1, 64)
        value_states_27 = None
        value_states_28 = view_58.transpose(1, 2)
        view_58 = None
        view_59 = query_states_27.view(1, 9, 12, 64)
        query_states_27 = None
        query_states_28 = view_59.transpose(1, 2)
        view_59 = None
        query_states_29 = query_states_28.reshape(12, -1, 64)
        query_states_28 = None
        key_states_29 = key_states_28.reshape(12, -1, 64)
        value_states_29 = value_states_28.reshape(12, -1, 64)
        transpose_48 = key_states_29.transpose(1, 2)
        key_states_29 = None
        attn_weights_45 = torch.bmm(query_states_29, transpose_48)
        query_states_29 = transpose_48 = None
        view_60 = attn_weights_45.view(1, 12, 9, 9)
        attn_weights_45 = None
        attn_weights_46 = view_60 + expanded_attn_mask_1
        view_60 = None
        tensor_10 = torch.tensor(
            -3.4028234663852886e38, device=device(type="cuda", index=0)
        )
        attn_weights_47 = torch.max(attn_weights_46, tensor_10)
        attn_weights_46 = tensor_10 = None
        attn_weights_48 = attn_weights_47.view(12, 9, 9)
        attn_weights_47 = None
        attn_weights_49 = torch.nn.functional.softmax(attn_weights_48, dim=-1)
        attn_weights_48 = None
        attn_probs_9 = torch.nn.functional.dropout(
            attn_weights_49, p=0.1, training=False
        )
        attn_weights_49 = None
        attn_output_45 = torch.bmm(attn_probs_9, value_states_29)
        attn_probs_9 = value_states_29 = None
        attn_output_46 = attn_output_45.view(1, 12, 9, 64)
        attn_output_45 = None
        attn_output_47 = attn_output_46.transpose(1, 2)
        attn_output_46 = None
        attn_output_48 = attn_output_47.reshape(1, 9, 768)
        attn_output_47 = None
        attn_output_49 = torch._C._nn.linear(
            attn_output_48,
            l_self_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_48 = l_self_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_84 = torch.nn.functional.dropout(
            attn_output_49, p=0.1, training=False
        )
        attn_output_49 = None
        hidden_states_85 = hidden_states_82 + hidden_states_84
        hidden_states_82 = hidden_states_84 = None
        hidden_states_86 = torch.nn.functional.layer_norm(
            hidden_states_85,
            (768,),
            l_self_modules_layers_modules_9_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_9_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_9_modules_final_layer_norm_parameters_weight_ = (
            l_self_modules_layers_modules_9_modules_final_layer_norm_parameters_bias_
        ) = None
        linear_58 = torch._C._nn.linear(
            hidden_states_86,
            l_self_modules_layers_modules_9_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_9_modules_fc1_parameters_bias_,
        )
        hidden_states_86 = (
            l_self_modules_layers_modules_9_modules_fc1_parameters_weight_
        ) = l_self_modules_layers_modules_9_modules_fc1_parameters_bias_ = None
        hidden_states_87 = torch._C._nn.gelu(linear_58)
        linear_58 = None
        hidden_states_88 = torch.nn.functional.dropout(
            hidden_states_87, p=0.0, training=False
        )
        hidden_states_87 = None
        hidden_states_89 = torch._C._nn.linear(
            hidden_states_88,
            l_self_modules_layers_modules_9_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_9_modules_fc2_parameters_bias_,
        )
        hidden_states_88 = (
            l_self_modules_layers_modules_9_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_9_modules_fc2_parameters_bias_ = None
        hidden_states_90 = torch.nn.functional.dropout(
            hidden_states_89, p=0.1, training=False
        )
        hidden_states_89 = None
        hidden_states_91 = hidden_states_85 + hidden_states_90
        hidden_states_85 = hidden_states_90 = None
        hidden_states_92 = torch.nn.functional.layer_norm(
            hidden_states_91,
            (768,),
            l_self_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_60 = torch._C._nn.linear(
            hidden_states_92,
            l_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_30 = linear_60 * 0.125
        linear_60 = None
        key_states_30 = torch._C._nn.linear(
            hidden_states_92,
            l_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_30 = torch._C._nn.linear(
            hidden_states_92,
            l_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_92 = l_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_63 = key_states_30.view(1, 9, -1, 64)
        key_states_30 = None
        key_states_31 = view_63.transpose(1, 2)
        view_63 = None
        view_64 = value_states_30.view(1, 9, -1, 64)
        value_states_30 = None
        value_states_31 = view_64.transpose(1, 2)
        view_64 = None
        view_65 = query_states_30.view(1, 9, 12, 64)
        query_states_30 = None
        query_states_31 = view_65.transpose(1, 2)
        view_65 = None
        query_states_32 = query_states_31.reshape(12, -1, 64)
        query_states_31 = None
        key_states_32 = key_states_31.reshape(12, -1, 64)
        value_states_32 = value_states_31.reshape(12, -1, 64)
        transpose_53 = key_states_32.transpose(1, 2)
        key_states_32 = None
        attn_weights_50 = torch.bmm(query_states_32, transpose_53)
        query_states_32 = transpose_53 = None
        view_66 = attn_weights_50.view(1, 12, 9, 9)
        attn_weights_50 = None
        attn_weights_51 = view_66 + expanded_attn_mask_1
        view_66 = None
        tensor_11 = torch.tensor(
            -3.4028234663852886e38, device=device(type="cuda", index=0)
        )
        attn_weights_52 = torch.max(attn_weights_51, tensor_11)
        attn_weights_51 = tensor_11 = None
        attn_weights_53 = attn_weights_52.view(12, 9, 9)
        attn_weights_52 = None
        attn_weights_54 = torch.nn.functional.softmax(attn_weights_53, dim=-1)
        attn_weights_53 = None
        attn_probs_10 = torch.nn.functional.dropout(
            attn_weights_54, p=0.1, training=False
        )
        attn_weights_54 = None
        attn_output_50 = torch.bmm(attn_probs_10, value_states_32)
        attn_probs_10 = value_states_32 = None
        attn_output_51 = attn_output_50.view(1, 12, 9, 64)
        attn_output_50 = None
        attn_output_52 = attn_output_51.transpose(1, 2)
        attn_output_51 = None
        attn_output_53 = attn_output_52.reshape(1, 9, 768)
        attn_output_52 = None
        attn_output_54 = torch._C._nn.linear(
            attn_output_53,
            l_self_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_53 = l_self_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_93 = torch.nn.functional.dropout(
            attn_output_54, p=0.1, training=False
        )
        attn_output_54 = None
        hidden_states_94 = hidden_states_91 + hidden_states_93
        hidden_states_91 = hidden_states_93 = None
        hidden_states_95 = torch.nn.functional.layer_norm(
            hidden_states_94,
            (768,),
            l_self_modules_layers_modules_10_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_10_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_10_modules_final_layer_norm_parameters_weight_ = (
            l_self_modules_layers_modules_10_modules_final_layer_norm_parameters_bias_
        ) = None
        linear_64 = torch._C._nn.linear(
            hidden_states_95,
            l_self_modules_layers_modules_10_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_10_modules_fc1_parameters_bias_,
        )
        hidden_states_95 = (
            l_self_modules_layers_modules_10_modules_fc1_parameters_weight_
        ) = l_self_modules_layers_modules_10_modules_fc1_parameters_bias_ = None
        hidden_states_96 = torch._C._nn.gelu(linear_64)
        linear_64 = None
        hidden_states_97 = torch.nn.functional.dropout(
            hidden_states_96, p=0.0, training=False
        )
        hidden_states_96 = None
        hidden_states_98 = torch._C._nn.linear(
            hidden_states_97,
            l_self_modules_layers_modules_10_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_10_modules_fc2_parameters_bias_,
        )
        hidden_states_97 = (
            l_self_modules_layers_modules_10_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_10_modules_fc2_parameters_bias_ = None
        hidden_states_99 = torch.nn.functional.dropout(
            hidden_states_98, p=0.1, training=False
        )
        hidden_states_98 = None
        hidden_states_100 = hidden_states_94 + hidden_states_99
        hidden_states_94 = hidden_states_99 = None
        hidden_states_101 = torch.nn.functional.layer_norm(
            hidden_states_100,
            (768,),
            l_self_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_66 = torch._C._nn.linear(
            hidden_states_101,
            l_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_33 = linear_66 * 0.125
        linear_66 = None
        key_states_33 = torch._C._nn.linear(
            hidden_states_101,
            l_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_33 = torch._C._nn.linear(
            hidden_states_101,
            l_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_101 = l_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_69 = key_states_33.view(1, 9, -1, 64)
        key_states_33 = None
        key_states_34 = view_69.transpose(1, 2)
        view_69 = None
        view_70 = value_states_33.view(1, 9, -1, 64)
        value_states_33 = None
        value_states_34 = view_70.transpose(1, 2)
        view_70 = None
        view_71 = query_states_33.view(1, 9, 12, 64)
        query_states_33 = None
        query_states_34 = view_71.transpose(1, 2)
        view_71 = None
        query_states_35 = query_states_34.reshape(12, -1, 64)
        query_states_34 = None
        key_states_35 = key_states_34.reshape(12, -1, 64)
        value_states_35 = value_states_34.reshape(12, -1, 64)
        transpose_58 = key_states_35.transpose(1, 2)
        key_states_35 = None
        attn_weights_55 = torch.bmm(query_states_35, transpose_58)
        query_states_35 = transpose_58 = None
        view_72 = attn_weights_55.view(1, 12, 9, 9)
        attn_weights_55 = None
        attn_weights_56 = view_72 + expanded_attn_mask_1
        view_72 = expanded_attn_mask_1 = None
        tensor_12 = torch.tensor(
            -3.4028234663852886e38, device=device(type="cuda", index=0)
        )
        attn_weights_57 = torch.max(attn_weights_56, tensor_12)
        attn_weights_56 = tensor_12 = None
        attn_weights_58 = attn_weights_57.view(12, 9, 9)
        attn_weights_57 = None
        attn_weights_59 = torch.nn.functional.softmax(attn_weights_58, dim=-1)
        attn_weights_58 = None
        attn_probs_11 = torch.nn.functional.dropout(
            attn_weights_59, p=0.1, training=False
        )
        attn_weights_59 = None
        attn_output_55 = torch.bmm(attn_probs_11, value_states_35)
        attn_probs_11 = value_states_35 = None
        attn_output_56 = attn_output_55.view(1, 12, 9, 64)
        attn_output_55 = None
        attn_output_57 = attn_output_56.transpose(1, 2)
        attn_output_56 = None
        attn_output_58 = attn_output_57.reshape(1, 9, 768)
        attn_output_57 = None
        attn_output_59 = torch._C._nn.linear(
            attn_output_58,
            l_self_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_58 = l_self_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_102 = torch.nn.functional.dropout(
            attn_output_59, p=0.1, training=False
        )
        attn_output_59 = None
        hidden_states_103 = hidden_states_100 + hidden_states_102
        hidden_states_100 = hidden_states_102 = None
        hidden_states_104 = torch.nn.functional.layer_norm(
            hidden_states_103,
            (768,),
            l_self_modules_layers_modules_11_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_11_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_11_modules_final_layer_norm_parameters_weight_ = (
            l_self_modules_layers_modules_11_modules_final_layer_norm_parameters_bias_
        ) = None
        linear_70 = torch._C._nn.linear(
            hidden_states_104,
            l_self_modules_layers_modules_11_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_11_modules_fc1_parameters_bias_,
        )
        hidden_states_104 = (
            l_self_modules_layers_modules_11_modules_fc1_parameters_weight_
        ) = l_self_modules_layers_modules_11_modules_fc1_parameters_bias_ = None
        hidden_states_105 = torch._C._nn.gelu(linear_70)
        linear_70 = None
        hidden_states_106 = torch.nn.functional.dropout(
            hidden_states_105, p=0.0, training=False
        )
        hidden_states_105 = None
        hidden_states_107 = torch._C._nn.linear(
            hidden_states_106,
            l_self_modules_layers_modules_11_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_11_modules_fc2_parameters_bias_,
        )
        hidden_states_106 = (
            l_self_modules_layers_modules_11_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_11_modules_fc2_parameters_bias_ = None
        hidden_states_108 = torch.nn.functional.dropout(
            hidden_states_107, p=0.1, training=False
        )
        hidden_states_107 = None
        hidden_states_109 = hidden_states_103 + hidden_states_108
        hidden_states_103 = hidden_states_108 = None
        hidden_states_110 = torch.nn.functional.layer_norm(
            hidden_states_109,
            (768,),
            l_self_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_109 = (
            l_self_modules_layer_norm_parameters_weight_
        ) = l_self_modules_layer_norm_parameters_bias_ = None
        return (
            value_states_1,
            key_states_1,
            value_states_4,
            key_states_4,
            value_states_7,
            key_states_7,
            value_states_10,
            key_states_10,
            value_states_13,
            key_states_13,
            value_states_16,
            key_states_16,
            value_states_19,
            key_states_19,
            value_states_22,
            key_states_22,
            value_states_25,
            key_states_25,
            value_states_28,
            key_states_28,
            value_states_31,
            key_states_31,
            value_states_34,
            key_states_34,
            hidden_states_110,
        )
