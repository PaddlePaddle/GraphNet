import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_inputs_embeds_: torch.Tensor,
        L_encoder_hidden_states_: torch.Tensor,
        L_encoder_attention_mask_: torch.Tensor,
        L_self_modules_embed_positions_parameters_weight_: torch.nn.parameter.Parameter,
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
    ):
        l_inputs_embeds_ = L_inputs_embeds_
        l_encoder_hidden_states_ = L_encoder_hidden_states_
        l_encoder_attention_mask_ = L_encoder_attention_mask_
        l_self_modules_embed_positions_parameters_weight_ = (
            L_self_modules_embed_positions_parameters_weight_
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
        cache_position = torch.arange(0, 16, device=device(type="cuda", index=0))
        causal_mask = torch.full(
            (16, 17),
            fill_value=-3.4028234663852886e38,
            dtype=torch.float32,
            device=device(type="cuda", index=0),
        )
        causal_mask_1 = torch.triu(causal_mask, diagonal=1)
        causal_mask = None
        arange_1 = torch.arange(17, device=device(type="cuda", index=0))
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
        expand_1 = getitem_1.expand(1, 1, 16, 16)
        getitem_1 = None
        expanded_mask = expand_1.to(torch.float32)
        expand_1 = None
        tensor = torch.tensor(1.0, dtype=torch.float32)
        inverted_mask = tensor - expanded_mask
        tensor = expanded_mask = None
        to_1 = inverted_mask.to(torch.bool)
        encoder_attention_mask = inverted_mask.masked_fill(to_1, -3.4028234663852886e38)
        inverted_mask = to_1 = None
        position_ids = torch.nn.functional.embedding(
            cache_position,
            l_self_modules_embed_positions_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        cache_position = l_self_modules_embed_positions_parameters_weight_ = None
        hidden_states = l_inputs_embeds_ + position_ids
        l_inputs_embeds_ = position_ids = None
        hidden_states_1 = torch.nn.functional.dropout(
            hidden_states, p=0.1, training=False
        )
        hidden_states = None
        linear = torch._C._nn.linear(
            hidden_states_1,
            l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view = linear.view(1, 16, -1, 64)
        linear = None
        query_states = view.transpose(1, 2)
        view = None
        key_states = torch._C._nn.linear(
            hidden_states_1,
            l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states = torch._C._nn.linear(
            hidden_states_1,
            l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_1 = key_states.view(1, 16, -1, 64)
        key_states = None
        key_states_1 = view_1.transpose(1, 2)
        view_1 = None
        view_2 = value_states.view(1, 16, -1, 64)
        value_states = None
        value_states_1 = view_2.transpose(1, 2)
        view_2 = None
        attention_mask = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 16, None),
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
        reshape_1 = attn_output_1.reshape(1, 16, -1)
        attn_output_1 = None
        attn_output_2 = reshape_1.contiguous()
        reshape_1 = None
        attn_output_3 = torch._C._nn.linear(
            attn_output_2,
            l_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_2 = l_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_2 = torch.nn.functional.dropout(
            attn_output_3, p=0.1, training=False
        )
        attn_output_3 = None
        hidden_states_3 = hidden_states_1 + hidden_states_2
        hidden_states_1 = hidden_states_2 = None
        hidden_states_4 = torch.nn.functional.layer_norm(
            hidden_states_3,
            (512,),
            l_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_3 = l_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_4 = torch._C._nn.linear(
            hidden_states_4,
            l_self_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        view_3 = linear_4.view(1, 16, -1, 64)
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
        view_4 = key_states_2.view(1, 16, -1, 64)
        key_states_2 = None
        key_states_3 = view_4.transpose(1, 2)
        view_4 = None
        view_5 = value_states_2.view(1, 16, -1, 64)
        value_states_2 = None
        value_states_3 = view_5.transpose(1, 2)
        view_5 = None
        attention_mask_1 = encoder_attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 16, None),
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
        reshape_2 = attn_output_5.reshape(1, 16, -1)
        attn_output_5 = None
        attn_output_6 = reshape_2.contiguous()
        reshape_2 = None
        attn_output_7 = torch._C._nn.linear(
            attn_output_6,
            l_self_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_6 = l_self_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_5 = torch.nn.functional.dropout(
            attn_output_7, p=0.1, training=False
        )
        attn_output_7 = None
        hidden_states_6 = hidden_states_4 + hidden_states_5
        hidden_states_4 = hidden_states_5 = None
        hidden_states_7 = torch.nn.functional.layer_norm(
            hidden_states_6,
            (512,),
            l_self_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_6 = l_self_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_8 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_layers_modules_0_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_0_modules_fc1_parameters_bias_,
        )
        l_self_modules_layers_modules_0_modules_fc1_parameters_weight_ = (
            l_self_modules_layers_modules_0_modules_fc1_parameters_bias_
        ) = None
        hidden_states_8 = torch.nn.functional.silu(linear_8, inplace=False)
        linear_8 = None
        hidden_states_9 = torch.nn.functional.dropout(
            hidden_states_8, p=0.0, training=False
        )
        hidden_states_8 = None
        hidden_states_10 = torch._C._nn.linear(
            hidden_states_9,
            l_self_modules_layers_modules_0_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_0_modules_fc2_parameters_bias_,
        )
        hidden_states_9 = (
            l_self_modules_layers_modules_0_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_0_modules_fc2_parameters_bias_ = None
        hidden_states_11 = torch.nn.functional.dropout(
            hidden_states_10, p=0.1, training=False
        )
        hidden_states_10 = None
        hidden_states_12 = hidden_states_7 + hidden_states_11
        hidden_states_7 = hidden_states_11 = None
        hidden_states_13 = torch.nn.functional.layer_norm(
            hidden_states_12,
            (512,),
            l_self_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_12 = (
            l_self_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_
        ) = (
            l_self_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_
        ) = None
        linear_10 = torch._C._nn.linear(
            hidden_states_13,
            l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_6 = linear_10.view(1, 16, -1, 64)
        linear_10 = None
        query_states_2 = view_6.transpose(1, 2)
        view_6 = None
        key_states_4 = torch._C._nn.linear(
            hidden_states_13,
            l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_4 = torch._C._nn.linear(
            hidden_states_13,
            l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_7 = key_states_4.view(1, 16, -1, 64)
        key_states_4 = None
        key_states_5 = view_7.transpose(1, 2)
        view_7 = None
        view_8 = value_states_4.view(1, 16, -1, 64)
        value_states_4 = None
        value_states_5 = view_8.transpose(1, 2)
        view_8 = None
        attention_mask_2 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 16, None),
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
        reshape_3 = attn_output_9.reshape(1, 16, -1)
        attn_output_9 = None
        attn_output_10 = reshape_3.contiguous()
        reshape_3 = None
        attn_output_11 = torch._C._nn.linear(
            attn_output_10,
            l_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_10 = l_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_14 = torch.nn.functional.dropout(
            attn_output_11, p=0.1, training=False
        )
        attn_output_11 = None
        hidden_states_15 = hidden_states_13 + hidden_states_14
        hidden_states_13 = hidden_states_14 = None
        hidden_states_16 = torch.nn.functional.layer_norm(
            hidden_states_15,
            (512,),
            l_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_15 = l_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_14 = torch._C._nn.linear(
            hidden_states_16,
            l_self_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        view_9 = linear_14.view(1, 16, -1, 64)
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
        view_10 = key_states_6.view(1, 16, -1, 64)
        key_states_6 = None
        key_states_7 = view_10.transpose(1, 2)
        view_10 = None
        view_11 = value_states_6.view(1, 16, -1, 64)
        value_states_6 = None
        value_states_7 = view_11.transpose(1, 2)
        view_11 = None
        attention_mask_3 = encoder_attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 16, None),
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
        reshape_4 = attn_output_13.reshape(1, 16, -1)
        attn_output_13 = None
        attn_output_14 = reshape_4.contiguous()
        reshape_4 = None
        attn_output_15 = torch._C._nn.linear(
            attn_output_14,
            l_self_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_14 = l_self_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_17 = torch.nn.functional.dropout(
            attn_output_15, p=0.1, training=False
        )
        attn_output_15 = None
        hidden_states_18 = hidden_states_16 + hidden_states_17
        hidden_states_16 = hidden_states_17 = None
        hidden_states_19 = torch.nn.functional.layer_norm(
            hidden_states_18,
            (512,),
            l_self_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_18 = l_self_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_18 = torch._C._nn.linear(
            hidden_states_19,
            l_self_modules_layers_modules_1_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_1_modules_fc1_parameters_bias_,
        )
        l_self_modules_layers_modules_1_modules_fc1_parameters_weight_ = (
            l_self_modules_layers_modules_1_modules_fc1_parameters_bias_
        ) = None
        hidden_states_20 = torch.nn.functional.silu(linear_18, inplace=False)
        linear_18 = None
        hidden_states_21 = torch.nn.functional.dropout(
            hidden_states_20, p=0.0, training=False
        )
        hidden_states_20 = None
        hidden_states_22 = torch._C._nn.linear(
            hidden_states_21,
            l_self_modules_layers_modules_1_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_1_modules_fc2_parameters_bias_,
        )
        hidden_states_21 = (
            l_self_modules_layers_modules_1_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_1_modules_fc2_parameters_bias_ = None
        hidden_states_23 = torch.nn.functional.dropout(
            hidden_states_22, p=0.1, training=False
        )
        hidden_states_22 = None
        hidden_states_24 = hidden_states_19 + hidden_states_23
        hidden_states_19 = hidden_states_23 = None
        hidden_states_25 = torch.nn.functional.layer_norm(
            hidden_states_24,
            (512,),
            l_self_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_24 = (
            l_self_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_
        ) = (
            l_self_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_
        ) = None
        linear_20 = torch._C._nn.linear(
            hidden_states_25,
            l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_12 = linear_20.view(1, 16, -1, 64)
        linear_20 = None
        query_states_4 = view_12.transpose(1, 2)
        view_12 = None
        key_states_8 = torch._C._nn.linear(
            hidden_states_25,
            l_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_8 = torch._C._nn.linear(
            hidden_states_25,
            l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_13 = key_states_8.view(1, 16, -1, 64)
        key_states_8 = None
        key_states_9 = view_13.transpose(1, 2)
        view_13 = None
        view_14 = value_states_8.view(1, 16, -1, 64)
        value_states_8 = None
        value_states_9 = view_14.transpose(1, 2)
        view_14 = None
        attention_mask_4 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 16, None),
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
        reshape_5 = attn_output_17.reshape(1, 16, -1)
        attn_output_17 = None
        attn_output_18 = reshape_5.contiguous()
        reshape_5 = None
        attn_output_19 = torch._C._nn.linear(
            attn_output_18,
            l_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_18 = l_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_26 = torch.nn.functional.dropout(
            attn_output_19, p=0.1, training=False
        )
        attn_output_19 = None
        hidden_states_27 = hidden_states_25 + hidden_states_26
        hidden_states_25 = hidden_states_26 = None
        hidden_states_28 = torch.nn.functional.layer_norm(
            hidden_states_27,
            (512,),
            l_self_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_27 = l_self_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_24 = torch._C._nn.linear(
            hidden_states_28,
            l_self_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        view_15 = linear_24.view(1, 16, -1, 64)
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
        view_16 = key_states_10.view(1, 16, -1, 64)
        key_states_10 = None
        key_states_11 = view_16.transpose(1, 2)
        view_16 = None
        view_17 = value_states_10.view(1, 16, -1, 64)
        value_states_10 = None
        value_states_11 = view_17.transpose(1, 2)
        view_17 = None
        attention_mask_5 = encoder_attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 16, None),
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
        reshape_6 = attn_output_21.reshape(1, 16, -1)
        attn_output_21 = None
        attn_output_22 = reshape_6.contiguous()
        reshape_6 = None
        attn_output_23 = torch._C._nn.linear(
            attn_output_22,
            l_self_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_22 = l_self_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_29 = torch.nn.functional.dropout(
            attn_output_23, p=0.1, training=False
        )
        attn_output_23 = None
        hidden_states_30 = hidden_states_28 + hidden_states_29
        hidden_states_28 = hidden_states_29 = None
        hidden_states_31 = torch.nn.functional.layer_norm(
            hidden_states_30,
            (512,),
            l_self_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_30 = l_self_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_28 = torch._C._nn.linear(
            hidden_states_31,
            l_self_modules_layers_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_fc1_parameters_bias_,
        )
        l_self_modules_layers_modules_2_modules_fc1_parameters_weight_ = (
            l_self_modules_layers_modules_2_modules_fc1_parameters_bias_
        ) = None
        hidden_states_32 = torch.nn.functional.silu(linear_28, inplace=False)
        linear_28 = None
        hidden_states_33 = torch.nn.functional.dropout(
            hidden_states_32, p=0.0, training=False
        )
        hidden_states_32 = None
        hidden_states_34 = torch._C._nn.linear(
            hidden_states_33,
            l_self_modules_layers_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_fc2_parameters_bias_,
        )
        hidden_states_33 = (
            l_self_modules_layers_modules_2_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_2_modules_fc2_parameters_bias_ = None
        hidden_states_35 = torch.nn.functional.dropout(
            hidden_states_34, p=0.1, training=False
        )
        hidden_states_34 = None
        hidden_states_36 = hidden_states_31 + hidden_states_35
        hidden_states_31 = hidden_states_35 = None
        hidden_states_37 = torch.nn.functional.layer_norm(
            hidden_states_36,
            (512,),
            l_self_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_36 = (
            l_self_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_
        ) = (
            l_self_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_
        ) = None
        linear_30 = torch._C._nn.linear(
            hidden_states_37,
            l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_18 = linear_30.view(1, 16, -1, 64)
        linear_30 = None
        query_states_6 = view_18.transpose(1, 2)
        view_18 = None
        key_states_12 = torch._C._nn.linear(
            hidden_states_37,
            l_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_12 = torch._C._nn.linear(
            hidden_states_37,
            l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_19 = key_states_12.view(1, 16, -1, 64)
        key_states_12 = None
        key_states_13 = view_19.transpose(1, 2)
        view_19 = None
        view_20 = value_states_12.view(1, 16, -1, 64)
        value_states_12 = None
        value_states_13 = view_20.transpose(1, 2)
        view_20 = None
        attention_mask_6 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 16, None),
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
        reshape_7 = attn_output_25.reshape(1, 16, -1)
        attn_output_25 = None
        attn_output_26 = reshape_7.contiguous()
        reshape_7 = None
        attn_output_27 = torch._C._nn.linear(
            attn_output_26,
            l_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_26 = l_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_38 = torch.nn.functional.dropout(
            attn_output_27, p=0.1, training=False
        )
        attn_output_27 = None
        hidden_states_39 = hidden_states_37 + hidden_states_38
        hidden_states_37 = hidden_states_38 = None
        hidden_states_40 = torch.nn.functional.layer_norm(
            hidden_states_39,
            (512,),
            l_self_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_39 = l_self_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_34 = torch._C._nn.linear(
            hidden_states_40,
            l_self_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        view_21 = linear_34.view(1, 16, -1, 64)
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
        view_22 = key_states_14.view(1, 16, -1, 64)
        key_states_14 = None
        key_states_15 = view_22.transpose(1, 2)
        view_22 = None
        view_23 = value_states_14.view(1, 16, -1, 64)
        value_states_14 = None
        value_states_15 = view_23.transpose(1, 2)
        view_23 = None
        attention_mask_7 = encoder_attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 16, None),
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
        reshape_8 = attn_output_29.reshape(1, 16, -1)
        attn_output_29 = None
        attn_output_30 = reshape_8.contiguous()
        reshape_8 = None
        attn_output_31 = torch._C._nn.linear(
            attn_output_30,
            l_self_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_30 = l_self_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_41 = torch.nn.functional.dropout(
            attn_output_31, p=0.1, training=False
        )
        attn_output_31 = None
        hidden_states_42 = hidden_states_40 + hidden_states_41
        hidden_states_40 = hidden_states_41 = None
        hidden_states_43 = torch.nn.functional.layer_norm(
            hidden_states_42,
            (512,),
            l_self_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_42 = l_self_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_38 = torch._C._nn.linear(
            hidden_states_43,
            l_self_modules_layers_modules_3_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_3_modules_fc1_parameters_bias_,
        )
        l_self_modules_layers_modules_3_modules_fc1_parameters_weight_ = (
            l_self_modules_layers_modules_3_modules_fc1_parameters_bias_
        ) = None
        hidden_states_44 = torch.nn.functional.silu(linear_38, inplace=False)
        linear_38 = None
        hidden_states_45 = torch.nn.functional.dropout(
            hidden_states_44, p=0.0, training=False
        )
        hidden_states_44 = None
        hidden_states_46 = torch._C._nn.linear(
            hidden_states_45,
            l_self_modules_layers_modules_3_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_3_modules_fc2_parameters_bias_,
        )
        hidden_states_45 = (
            l_self_modules_layers_modules_3_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_3_modules_fc2_parameters_bias_ = None
        hidden_states_47 = torch.nn.functional.dropout(
            hidden_states_46, p=0.1, training=False
        )
        hidden_states_46 = None
        hidden_states_48 = hidden_states_43 + hidden_states_47
        hidden_states_43 = hidden_states_47 = None
        hidden_states_49 = torch.nn.functional.layer_norm(
            hidden_states_48,
            (512,),
            l_self_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_48 = (
            l_self_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_
        ) = (
            l_self_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_
        ) = None
        linear_40 = torch._C._nn.linear(
            hidden_states_49,
            l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_24 = linear_40.view(1, 16, -1, 64)
        linear_40 = None
        query_states_8 = view_24.transpose(1, 2)
        view_24 = None
        key_states_16 = torch._C._nn.linear(
            hidden_states_49,
            l_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_16 = torch._C._nn.linear(
            hidden_states_49,
            l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_25 = key_states_16.view(1, 16, -1, 64)
        key_states_16 = None
        key_states_17 = view_25.transpose(1, 2)
        view_25 = None
        view_26 = value_states_16.view(1, 16, -1, 64)
        value_states_16 = None
        value_states_17 = view_26.transpose(1, 2)
        view_26 = None
        attention_mask_8 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 16, None),
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
        reshape_9 = attn_output_33.reshape(1, 16, -1)
        attn_output_33 = None
        attn_output_34 = reshape_9.contiguous()
        reshape_9 = None
        attn_output_35 = torch._C._nn.linear(
            attn_output_34,
            l_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_34 = l_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_50 = torch.nn.functional.dropout(
            attn_output_35, p=0.1, training=False
        )
        attn_output_35 = None
        hidden_states_51 = hidden_states_49 + hidden_states_50
        hidden_states_49 = hidden_states_50 = None
        hidden_states_52 = torch.nn.functional.layer_norm(
            hidden_states_51,
            (512,),
            l_self_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_51 = l_self_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_44 = torch._C._nn.linear(
            hidden_states_52,
            l_self_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        view_27 = linear_44.view(1, 16, -1, 64)
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
        view_28 = key_states_18.view(1, 16, -1, 64)
        key_states_18 = None
        key_states_19 = view_28.transpose(1, 2)
        view_28 = None
        view_29 = value_states_18.view(1, 16, -1, 64)
        value_states_18 = None
        value_states_19 = view_29.transpose(1, 2)
        view_29 = None
        attention_mask_9 = encoder_attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 16, None),
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
        reshape_10 = attn_output_37.reshape(1, 16, -1)
        attn_output_37 = None
        attn_output_38 = reshape_10.contiguous()
        reshape_10 = None
        attn_output_39 = torch._C._nn.linear(
            attn_output_38,
            l_self_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_38 = l_self_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_53 = torch.nn.functional.dropout(
            attn_output_39, p=0.1, training=False
        )
        attn_output_39 = None
        hidden_states_54 = hidden_states_52 + hidden_states_53
        hidden_states_52 = hidden_states_53 = None
        hidden_states_55 = torch.nn.functional.layer_norm(
            hidden_states_54,
            (512,),
            l_self_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_54 = l_self_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_48 = torch._C._nn.linear(
            hidden_states_55,
            l_self_modules_layers_modules_4_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_4_modules_fc1_parameters_bias_,
        )
        l_self_modules_layers_modules_4_modules_fc1_parameters_weight_ = (
            l_self_modules_layers_modules_4_modules_fc1_parameters_bias_
        ) = None
        hidden_states_56 = torch.nn.functional.silu(linear_48, inplace=False)
        linear_48 = None
        hidden_states_57 = torch.nn.functional.dropout(
            hidden_states_56, p=0.0, training=False
        )
        hidden_states_56 = None
        hidden_states_58 = torch._C._nn.linear(
            hidden_states_57,
            l_self_modules_layers_modules_4_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_4_modules_fc2_parameters_bias_,
        )
        hidden_states_57 = (
            l_self_modules_layers_modules_4_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_4_modules_fc2_parameters_bias_ = None
        hidden_states_59 = torch.nn.functional.dropout(
            hidden_states_58, p=0.1, training=False
        )
        hidden_states_58 = None
        hidden_states_60 = hidden_states_55 + hidden_states_59
        hidden_states_55 = hidden_states_59 = None
        hidden_states_61 = torch.nn.functional.layer_norm(
            hidden_states_60,
            (512,),
            l_self_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_60 = (
            l_self_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_
        ) = (
            l_self_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_
        ) = None
        linear_50 = torch._C._nn.linear(
            hidden_states_61,
            l_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_30 = linear_50.view(1, 16, -1, 64)
        linear_50 = None
        query_states_10 = view_30.transpose(1, 2)
        view_30 = None
        key_states_20 = torch._C._nn.linear(
            hidden_states_61,
            l_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_20 = torch._C._nn.linear(
            hidden_states_61,
            l_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_31 = key_states_20.view(1, 16, -1, 64)
        key_states_20 = None
        key_states_21 = view_31.transpose(1, 2)
        view_31 = None
        view_32 = value_states_20.view(1, 16, -1, 64)
        value_states_20 = None
        value_states_21 = view_32.transpose(1, 2)
        view_32 = None
        attention_mask_10 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 16, None),
            )
        ]
        causal_mask_3 = None
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
        reshape_11 = attn_output_41.reshape(1, 16, -1)
        attn_output_41 = None
        attn_output_42 = reshape_11.contiguous()
        reshape_11 = None
        attn_output_43 = torch._C._nn.linear(
            attn_output_42,
            l_self_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_42 = l_self_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_62 = torch.nn.functional.dropout(
            attn_output_43, p=0.1, training=False
        )
        attn_output_43 = None
        hidden_states_63 = hidden_states_61 + hidden_states_62
        hidden_states_61 = hidden_states_62 = None
        hidden_states_64 = torch.nn.functional.layer_norm(
            hidden_states_63,
            (512,),
            l_self_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_63 = l_self_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_54 = torch._C._nn.linear(
            hidden_states_64,
            l_self_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        view_33 = linear_54.view(1, 16, -1, 64)
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
        l_encoder_hidden_states_ = l_self_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_34 = key_states_22.view(1, 16, -1, 64)
        key_states_22 = None
        key_states_23 = view_34.transpose(1, 2)
        view_34 = None
        view_35 = value_states_22.view(1, 16, -1, 64)
        value_states_22 = None
        value_states_23 = view_35.transpose(1, 2)
        view_35 = None
        attention_mask_11 = encoder_attention_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 16, None),
            )
        ]
        encoder_attention_mask = None
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
        reshape_12 = attn_output_45.reshape(1, 16, -1)
        attn_output_45 = None
        attn_output_46 = reshape_12.contiguous()
        reshape_12 = None
        attn_output_47 = torch._C._nn.linear(
            attn_output_46,
            l_self_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_46 = l_self_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_65 = torch.nn.functional.dropout(
            attn_output_47, p=0.1, training=False
        )
        attn_output_47 = None
        hidden_states_66 = hidden_states_64 + hidden_states_65
        hidden_states_64 = hidden_states_65 = None
        hidden_states_67 = torch.nn.functional.layer_norm(
            hidden_states_66,
            (512,),
            l_self_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_66 = l_self_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_58 = torch._C._nn.linear(
            hidden_states_67,
            l_self_modules_layers_modules_5_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_5_modules_fc1_parameters_bias_,
        )
        l_self_modules_layers_modules_5_modules_fc1_parameters_weight_ = (
            l_self_modules_layers_modules_5_modules_fc1_parameters_bias_
        ) = None
        hidden_states_68 = torch.nn.functional.silu(linear_58, inplace=False)
        linear_58 = None
        hidden_states_69 = torch.nn.functional.dropout(
            hidden_states_68, p=0.0, training=False
        )
        hidden_states_68 = None
        hidden_states_70 = torch._C._nn.linear(
            hidden_states_69,
            l_self_modules_layers_modules_5_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_5_modules_fc2_parameters_bias_,
        )
        hidden_states_69 = (
            l_self_modules_layers_modules_5_modules_fc2_parameters_weight_
        ) = l_self_modules_layers_modules_5_modules_fc2_parameters_bias_ = None
        hidden_states_71 = torch.nn.functional.dropout(
            hidden_states_70, p=0.1, training=False
        )
        hidden_states_70 = None
        hidden_states_72 = hidden_states_67 + hidden_states_71
        hidden_states_67 = hidden_states_71 = None
        hidden_states_73 = torch.nn.functional.layer_norm(
            hidden_states_72,
            (512,),
            l_self_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_72 = (
            l_self_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_
        ) = (
            l_self_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_
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
            hidden_states_73,
        )
