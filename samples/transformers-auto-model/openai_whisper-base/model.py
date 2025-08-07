import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_inputs_embeds_: torch.Tensor,
        L_self_modules_embed_positions_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_encoder_hidden_states_: torch.Tensor,
        L_self_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_inputs_embeds_ = L_inputs_embeds_
        l_self_modules_embed_positions_parameters_weight_ = (
            L_self_modules_embed_positions_parameters_weight_
        )
        l_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_
        l_encoder_hidden_states_ = L_encoder_hidden_states_
        l_self_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_bias_
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
        l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_bias_
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
        l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_bias_
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
        l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_bias_
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
        l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_bias_
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
        l_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_bias_
        l_self_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_bias_
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
        l_self_modules_layer_norm_parameters_weight_ = (
            L_self_modules_layer_norm_parameters_weight_
        )
        l_self_modules_layer_norm_parameters_bias_ = (
            L_self_modules_layer_norm_parameters_bias_
        )
        cache_position = torch.arange(0, 1, device=device(type="cuda", index=0))
        unsqueeze = cache_position.unsqueeze(0)
        position_ids = unsqueeze.repeat(1, 1)
        unsqueeze = None
        positions = l_self_modules_embed_positions_parameters_weight_[position_ids]
        l_self_modules_embed_positions_parameters_weight_ = position_ids = None
        to = positions.to(device(type="cuda", index=0))
        positions = None
        hidden_states = l_inputs_embeds_ + to
        l_inputs_embeds_ = to = None
        hidden_states_1 = torch.nn.functional.dropout(
            hidden_states, p=0.0, training=False
        )
        hidden_states = None
        kv_arange = torch.arange(1, device=device(type="cuda", index=0))
        kv_arange += 0
        kv_arange_1 = kv_arange
        kv_arange = None
        batch_arange = torch.arange(1, device=device(type="cuda", index=0))
        head_arange = torch.arange(1, device=device(type="cuda", index=0))
        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions()
        lazy_load_decompositions = None
        _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(
            1, "error"
        )
        _vmap_increment_nesting = None
        child = torch._C._functorch._add_batch_dim(batch_arange, 0, 1)
        batch_arange = child = None
        lazy_load_decompositions_1 = torch._functorch.vmap.lazy_load_decompositions()
        lazy_load_decompositions_1 = None
        _vmap_increment_nesting_1 = torch._C._functorch._vmap_increment_nesting(
            1, "error"
        )
        _vmap_increment_nesting_1 = None
        child_1 = torch._C._functorch._add_batch_dim(head_arange, 0, 2)
        head_arange = child_1 = None
        lazy_load_decompositions_2 = torch._functorch.vmap.lazy_load_decompositions()
        lazy_load_decompositions_2 = None
        _vmap_increment_nesting_2 = torch._C._functorch._vmap_increment_nesting(
            1, "error"
        )
        _vmap_increment_nesting_2 = None
        child_2 = torch._C._functorch._add_batch_dim(cache_position, 0, 3)
        cache_position = None
        lazy_load_decompositions_3 = torch._functorch.vmap.lazy_load_decompositions()
        lazy_load_decompositions_3 = None
        _vmap_increment_nesting_3 = torch._C._functorch._vmap_increment_nesting(
            1, "error"
        )
        _vmap_increment_nesting_3 = None
        _add_batch_dim_3 = torch._C._functorch._add_batch_dim(kv_arange_1, 0, 4)
        kv_arange_1 = None
        batched_outputs = _add_batch_dim_3.le(child_2)
        _add_batch_dim_3 = child_2 = None
        batched_outputs_1 = torch._C._functorch._remove_batch_dim(
            batched_outputs, 4, 1, 0
        )
        batched_outputs = None
        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting()
        _vmap_decrement_nesting = None
        batched_outputs_2 = torch._C._functorch._remove_batch_dim(
            batched_outputs_1, 3, 1, 0
        )
        batched_outputs_1 = None
        _vmap_decrement_nesting_1 = torch._C._functorch._vmap_decrement_nesting()
        _vmap_decrement_nesting_1 = None
        batched_outputs_3 = torch._C._functorch._remove_batch_dim(
            batched_outputs_2, 2, 1, 0
        )
        batched_outputs_2 = None
        _vmap_decrement_nesting_2 = torch._C._functorch._vmap_decrement_nesting()
        _vmap_decrement_nesting_2 = None
        causal_mask = torch._C._functorch._remove_batch_dim(batched_outputs_3, 1, 1, 0)
        batched_outputs_3 = None
        _vmap_decrement_nesting_3 = torch._C._functorch._vmap_decrement_nesting()
        _vmap_decrement_nesting_3 = None
        hidden_states_2 = torch.nn.functional.layer_norm(
            hidden_states_1,
            (512,),
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
        query_states_1 = query_states.view(1, 1, -1, 64)
        query_states = None
        transpose = query_states_1.transpose(1, 2)
        query_states_1 = None
        query_states_2 = transpose.contiguous()
        transpose = None
        linear_1 = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        key_states = linear_1.view(1, -1, 8, 64)
        linear_1 = None
        linear_2 = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_2 = l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        value_states = linear_2.view(1, -1, 8, 64)
        linear_2 = None
        transpose_1 = key_states.transpose(1, 2)
        key_states = None
        key_states_1 = transpose_1.contiguous()
        transpose_1 = None
        transpose_2 = value_states.transpose(1, 2)
        value_states = None
        value_states_1 = transpose_2.contiguous()
        transpose_2 = None
        attention_mask = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
            )
        ]
        query = query_states_2.contiguous()
        query_states_2 = None
        key = key_states_1.contiguous()
        value = value_states_1.contiguous()
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
        reshape = attn_output_1.reshape(1, 1, -1)
        attn_output_1 = None
        attn_output_2 = reshape.contiguous()
        reshape = None
        attn_output_3 = torch._C._nn.linear(
            attn_output_2,
            l_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_2 = l_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_3 = torch.nn.functional.dropout(
            attn_output_3, p=0.0, training=False
        )
        attn_output_3 = None
        hidden_states_4 = hidden_states_1 + hidden_states_3
        hidden_states_1 = hidden_states_3 = None
        hidden_states_5 = torch.nn.functional.layer_norm(
            hidden_states_4,
            (512,),
            l_self_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_4 = torch._C._nn.linear(
            hidden_states_5,
            l_self_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        hidden_states_5 = l_self_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_3 = linear_4 * 0.125
        linear_4 = None
        query_states_4 = query_states_3.view(1, 1, -1, 64)
        query_states_3 = None
        transpose_4 = query_states_4.transpose(1, 2)
        query_states_4 = None
        query_states_5 = transpose_4.contiguous()
        transpose_4 = None
        linear_5 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        key_states_2 = linear_5.view(1, -1, 8, 64)
        linear_5 = None
        linear_6 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        value_states_2 = linear_6.view(1, -1, 8, 64)
        linear_6 = None
        transpose_5 = key_states_2.transpose(1, 2)
        key_states_2 = None
        key_states_3 = transpose_5.contiguous()
        transpose_5 = None
        transpose_6 = value_states_2.transpose(1, 2)
        value_states_2 = None
        value_states_3 = transpose_6.contiguous()
        transpose_6 = None
        query_1 = query_states_5.contiguous()
        query_states_5 = None
        key_1 = key_states_3.contiguous()
        value_1 = value_states_3.contiguous()
        attn_output_4 = torch._C._nn.scaled_dot_product_attention(
            query_1,
            key_1,
            value_1,
            attn_mask=None,
            dropout_p=0.0,
            scale=1.0,
            is_causal=False,
        )
        query_1 = key_1 = value_1 = None
        transpose_7 = attn_output_4.transpose(1, 2)
        attn_output_4 = None
        attn_output_5 = transpose_7.contiguous()
        transpose_7 = None
        reshape_1 = attn_output_5.reshape(1, 1, -1)
        attn_output_5 = None
        attn_output_6 = reshape_1.contiguous()
        reshape_1 = None
        attn_output_7 = torch._C._nn.linear(
            attn_output_6,
            l_self_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_6 = l_self_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_6 = torch.nn.functional.dropout(
            attn_output_7, p=0.0, training=False
        )
        attn_output_7 = None
        hidden_states_7 = hidden_states_4 + hidden_states_6
        hidden_states_4 = hidden_states_6 = None
        hidden_states_8 = torch.nn.functional.layer_norm(
            hidden_states_7,
            (512,),
            l_self_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_ = (
            l_self_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_
        ) = None
        linear_8 = torch._C._nn.linear(
            hidden_states_8,
            l_self_modules_layers_modules_0_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_0_modules_fc1_parameters_bias_,
        )
        hidden_states_8 = (
            l_self_modules_layers_modules_0_modules_fc1_parameters_weight_
        ) = l_self_modules_layers_modules_0_modules_fc1_parameters_bias_ = None
        hidden_states_9 = torch._C._nn.gelu(linear_8)
        linear_8 = None
        hidden_states_10 = torch.nn.functional.dropout(
            hidden_states_9, p=0.0, training=False
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
            hidden_states_11, p=0.0, training=False
        )
        hidden_states_11 = None
        hidden_states_13 = hidden_states_7 + hidden_states_12
        hidden_states_7 = hidden_states_12 = None
        hidden_states_14 = torch.nn.functional.layer_norm(
            hidden_states_13,
            (512,),
            l_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_10 = torch._C._nn.linear(
            hidden_states_14,
            l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_6 = linear_10 * 0.125
        linear_10 = None
        query_states_7 = query_states_6.view(1, 1, -1, 64)
        query_states_6 = None
        transpose_8 = query_states_7.transpose(1, 2)
        query_states_7 = None
        query_states_8 = transpose_8.contiguous()
        transpose_8 = None
        linear_11 = torch._C._nn.linear(
            hidden_states_14,
            l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        key_states_4 = linear_11.view(1, -1, 8, 64)
        linear_11 = None
        linear_12 = torch._C._nn.linear(
            hidden_states_14,
            l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_14 = l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        value_states_4 = linear_12.view(1, -1, 8, 64)
        linear_12 = None
        transpose_9 = key_states_4.transpose(1, 2)
        key_states_4 = None
        key_states_5 = transpose_9.contiguous()
        transpose_9 = None
        transpose_10 = value_states_4.transpose(1, 2)
        value_states_4 = None
        value_states_5 = transpose_10.contiguous()
        transpose_10 = None
        attention_mask_1 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
            )
        ]
        query_2 = query_states_8.contiguous()
        query_states_8 = None
        key_2 = key_states_5.contiguous()
        value_2 = value_states_5.contiguous()
        attn_output_8 = torch._C._nn.scaled_dot_product_attention(
            query_2,
            key_2,
            value_2,
            attn_mask=attention_mask_1,
            dropout_p=0.0,
            scale=1.0,
            is_causal=False,
        )
        query_2 = key_2 = value_2 = attention_mask_1 = None
        transpose_11 = attn_output_8.transpose(1, 2)
        attn_output_8 = None
        attn_output_9 = transpose_11.contiguous()
        transpose_11 = None
        reshape_2 = attn_output_9.reshape(1, 1, -1)
        attn_output_9 = None
        attn_output_10 = reshape_2.contiguous()
        reshape_2 = None
        attn_output_11 = torch._C._nn.linear(
            attn_output_10,
            l_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_10 = l_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_15 = torch.nn.functional.dropout(
            attn_output_11, p=0.0, training=False
        )
        attn_output_11 = None
        hidden_states_16 = hidden_states_13 + hidden_states_15
        hidden_states_13 = hidden_states_15 = None
        hidden_states_17 = torch.nn.functional.layer_norm(
            hidden_states_16,
            (512,),
            l_self_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_14 = torch._C._nn.linear(
            hidden_states_17,
            l_self_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        hidden_states_17 = l_self_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_9 = linear_14 * 0.125
        linear_14 = None
        query_states_10 = query_states_9.view(1, 1, -1, 64)
        query_states_9 = None
        transpose_12 = query_states_10.transpose(1, 2)
        query_states_10 = None
        query_states_11 = transpose_12.contiguous()
        transpose_12 = None
        linear_15 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        key_states_6 = linear_15.view(1, -1, 8, 64)
        linear_15 = None
        linear_16 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        value_states_6 = linear_16.view(1, -1, 8, 64)
        linear_16 = None
        transpose_13 = key_states_6.transpose(1, 2)
        key_states_6 = None
        key_states_7 = transpose_13.contiguous()
        transpose_13 = None
        transpose_14 = value_states_6.transpose(1, 2)
        value_states_6 = None
        value_states_7 = transpose_14.contiguous()
        transpose_14 = None
        query_3 = query_states_11.contiguous()
        query_states_11 = None
        key_3 = key_states_7.contiguous()
        value_3 = value_states_7.contiguous()
        attn_output_12 = torch._C._nn.scaled_dot_product_attention(
            query_3,
            key_3,
            value_3,
            attn_mask=None,
            dropout_p=0.0,
            scale=1.0,
            is_causal=False,
        )
        query_3 = key_3 = value_3 = None
        transpose_15 = attn_output_12.transpose(1, 2)
        attn_output_12 = None
        attn_output_13 = transpose_15.contiguous()
        transpose_15 = None
        reshape_3 = attn_output_13.reshape(1, 1, -1)
        attn_output_13 = None
        attn_output_14 = reshape_3.contiguous()
        reshape_3 = None
        attn_output_15 = torch._C._nn.linear(
            attn_output_14,
            l_self_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_14 = l_self_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_18 = torch.nn.functional.dropout(
            attn_output_15, p=0.0, training=False
        )
        attn_output_15 = None
        hidden_states_19 = hidden_states_16 + hidden_states_18
        hidden_states_16 = hidden_states_18 = None
        hidden_states_20 = torch.nn.functional.layer_norm(
            hidden_states_19,
            (512,),
            l_self_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_ = (
            l_self_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_
        ) = None
        linear_18 = torch._C._nn.linear(
            hidden_states_20,
            l_self_modules_layers_modules_1_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_1_modules_fc1_parameters_bias_,
        )
        hidden_states_20 = (
            l_self_modules_layers_modules_1_modules_fc1_parameters_weight_
        ) = l_self_modules_layers_modules_1_modules_fc1_parameters_bias_ = None
        hidden_states_21 = torch._C._nn.gelu(linear_18)
        linear_18 = None
        hidden_states_22 = torch.nn.functional.dropout(
            hidden_states_21, p=0.0, training=False
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
            hidden_states_23, p=0.0, training=False
        )
        hidden_states_23 = None
        hidden_states_25 = hidden_states_19 + hidden_states_24
        hidden_states_19 = hidden_states_24 = None
        hidden_states_26 = torch.nn.functional.layer_norm(
            hidden_states_25,
            (512,),
            l_self_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_20 = torch._C._nn.linear(
            hidden_states_26,
            l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_12 = linear_20 * 0.125
        linear_20 = None
        query_states_13 = query_states_12.view(1, 1, -1, 64)
        query_states_12 = None
        transpose_16 = query_states_13.transpose(1, 2)
        query_states_13 = None
        query_states_14 = transpose_16.contiguous()
        transpose_16 = None
        linear_21 = torch._C._nn.linear(
            hidden_states_26,
            l_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        key_states_8 = linear_21.view(1, -1, 8, 64)
        linear_21 = None
        linear_22 = torch._C._nn.linear(
            hidden_states_26,
            l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_26 = l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        value_states_8 = linear_22.view(1, -1, 8, 64)
        linear_22 = None
        transpose_17 = key_states_8.transpose(1, 2)
        key_states_8 = None
        key_states_9 = transpose_17.contiguous()
        transpose_17 = None
        transpose_18 = value_states_8.transpose(1, 2)
        value_states_8 = None
        value_states_9 = transpose_18.contiguous()
        transpose_18 = None
        attention_mask_2 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
            )
        ]
        query_4 = query_states_14.contiguous()
        query_states_14 = None
        key_4 = key_states_9.contiguous()
        value_4 = value_states_9.contiguous()
        attn_output_16 = torch._C._nn.scaled_dot_product_attention(
            query_4,
            key_4,
            value_4,
            attn_mask=attention_mask_2,
            dropout_p=0.0,
            scale=1.0,
            is_causal=False,
        )
        query_4 = key_4 = value_4 = attention_mask_2 = None
        transpose_19 = attn_output_16.transpose(1, 2)
        attn_output_16 = None
        attn_output_17 = transpose_19.contiguous()
        transpose_19 = None
        reshape_4 = attn_output_17.reshape(1, 1, -1)
        attn_output_17 = None
        attn_output_18 = reshape_4.contiguous()
        reshape_4 = None
        attn_output_19 = torch._C._nn.linear(
            attn_output_18,
            l_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_18 = l_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_27 = torch.nn.functional.dropout(
            attn_output_19, p=0.0, training=False
        )
        attn_output_19 = None
        hidden_states_28 = hidden_states_25 + hidden_states_27
        hidden_states_25 = hidden_states_27 = None
        hidden_states_29 = torch.nn.functional.layer_norm(
            hidden_states_28,
            (512,),
            l_self_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_24 = torch._C._nn.linear(
            hidden_states_29,
            l_self_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        hidden_states_29 = l_self_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_15 = linear_24 * 0.125
        linear_24 = None
        query_states_16 = query_states_15.view(1, 1, -1, 64)
        query_states_15 = None
        transpose_20 = query_states_16.transpose(1, 2)
        query_states_16 = None
        query_states_17 = transpose_20.contiguous()
        transpose_20 = None
        linear_25 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        key_states_10 = linear_25.view(1, -1, 8, 64)
        linear_25 = None
        linear_26 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        value_states_10 = linear_26.view(1, -1, 8, 64)
        linear_26 = None
        transpose_21 = key_states_10.transpose(1, 2)
        key_states_10 = None
        key_states_11 = transpose_21.contiguous()
        transpose_21 = None
        transpose_22 = value_states_10.transpose(1, 2)
        value_states_10 = None
        value_states_11 = transpose_22.contiguous()
        transpose_22 = None
        query_5 = query_states_17.contiguous()
        query_states_17 = None
        key_5 = key_states_11.contiguous()
        value_5 = value_states_11.contiguous()
        attn_output_20 = torch._C._nn.scaled_dot_product_attention(
            query_5,
            key_5,
            value_5,
            attn_mask=None,
            dropout_p=0.0,
            scale=1.0,
            is_causal=False,
        )
        query_5 = key_5 = value_5 = None
        transpose_23 = attn_output_20.transpose(1, 2)
        attn_output_20 = None
        attn_output_21 = transpose_23.contiguous()
        transpose_23 = None
        reshape_5 = attn_output_21.reshape(1, 1, -1)
        attn_output_21 = None
        attn_output_22 = reshape_5.contiguous()
        reshape_5 = None
        attn_output_23 = torch._C._nn.linear(
            attn_output_22,
            l_self_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_22 = l_self_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_30 = torch.nn.functional.dropout(
            attn_output_23, p=0.0, training=False
        )
        attn_output_23 = None
        hidden_states_31 = hidden_states_28 + hidden_states_30
        hidden_states_28 = hidden_states_30 = None
        hidden_states_32 = torch.nn.functional.layer_norm(
            hidden_states_31,
            (512,),
            l_self_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_ = (
            l_self_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_
        ) = None
        linear_28 = torch._C._nn.linear(
            hidden_states_32,
            l_self_modules_layers_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_fc1_parameters_bias_,
        )
        hidden_states_32 = (
            l_self_modules_layers_modules_2_modules_fc1_parameters_weight_
        ) = l_self_modules_layers_modules_2_modules_fc1_parameters_bias_ = None
        hidden_states_33 = torch._C._nn.gelu(linear_28)
        linear_28 = None
        hidden_states_34 = torch.nn.functional.dropout(
            hidden_states_33, p=0.0, training=False
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
            hidden_states_35, p=0.0, training=False
        )
        hidden_states_35 = None
        hidden_states_37 = hidden_states_31 + hidden_states_36
        hidden_states_31 = hidden_states_36 = None
        hidden_states_38 = torch.nn.functional.layer_norm(
            hidden_states_37,
            (512,),
            l_self_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_30 = torch._C._nn.linear(
            hidden_states_38,
            l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_18 = linear_30 * 0.125
        linear_30 = None
        query_states_19 = query_states_18.view(1, 1, -1, 64)
        query_states_18 = None
        transpose_24 = query_states_19.transpose(1, 2)
        query_states_19 = None
        query_states_20 = transpose_24.contiguous()
        transpose_24 = None
        linear_31 = torch._C._nn.linear(
            hidden_states_38,
            l_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        key_states_12 = linear_31.view(1, -1, 8, 64)
        linear_31 = None
        linear_32 = torch._C._nn.linear(
            hidden_states_38,
            l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_38 = l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        value_states_12 = linear_32.view(1, -1, 8, 64)
        linear_32 = None
        transpose_25 = key_states_12.transpose(1, 2)
        key_states_12 = None
        key_states_13 = transpose_25.contiguous()
        transpose_25 = None
        transpose_26 = value_states_12.transpose(1, 2)
        value_states_12 = None
        value_states_13 = transpose_26.contiguous()
        transpose_26 = None
        attention_mask_3 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
            )
        ]
        query_6 = query_states_20.contiguous()
        query_states_20 = None
        key_6 = key_states_13.contiguous()
        value_6 = value_states_13.contiguous()
        attn_output_24 = torch._C._nn.scaled_dot_product_attention(
            query_6,
            key_6,
            value_6,
            attn_mask=attention_mask_3,
            dropout_p=0.0,
            scale=1.0,
            is_causal=False,
        )
        query_6 = key_6 = value_6 = attention_mask_3 = None
        transpose_27 = attn_output_24.transpose(1, 2)
        attn_output_24 = None
        attn_output_25 = transpose_27.contiguous()
        transpose_27 = None
        reshape_6 = attn_output_25.reshape(1, 1, -1)
        attn_output_25 = None
        attn_output_26 = reshape_6.contiguous()
        reshape_6 = None
        attn_output_27 = torch._C._nn.linear(
            attn_output_26,
            l_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_26 = l_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_39 = torch.nn.functional.dropout(
            attn_output_27, p=0.0, training=False
        )
        attn_output_27 = None
        hidden_states_40 = hidden_states_37 + hidden_states_39
        hidden_states_37 = hidden_states_39 = None
        hidden_states_41 = torch.nn.functional.layer_norm(
            hidden_states_40,
            (512,),
            l_self_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_34 = torch._C._nn.linear(
            hidden_states_41,
            l_self_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        hidden_states_41 = l_self_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_21 = linear_34 * 0.125
        linear_34 = None
        query_states_22 = query_states_21.view(1, 1, -1, 64)
        query_states_21 = None
        transpose_28 = query_states_22.transpose(1, 2)
        query_states_22 = None
        query_states_23 = transpose_28.contiguous()
        transpose_28 = None
        linear_35 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        key_states_14 = linear_35.view(1, -1, 8, 64)
        linear_35 = None
        linear_36 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        value_states_14 = linear_36.view(1, -1, 8, 64)
        linear_36 = None
        transpose_29 = key_states_14.transpose(1, 2)
        key_states_14 = None
        key_states_15 = transpose_29.contiguous()
        transpose_29 = None
        transpose_30 = value_states_14.transpose(1, 2)
        value_states_14 = None
        value_states_15 = transpose_30.contiguous()
        transpose_30 = None
        query_7 = query_states_23.contiguous()
        query_states_23 = None
        key_7 = key_states_15.contiguous()
        value_7 = value_states_15.contiguous()
        attn_output_28 = torch._C._nn.scaled_dot_product_attention(
            query_7,
            key_7,
            value_7,
            attn_mask=None,
            dropout_p=0.0,
            scale=1.0,
            is_causal=False,
        )
        query_7 = key_7 = value_7 = None
        transpose_31 = attn_output_28.transpose(1, 2)
        attn_output_28 = None
        attn_output_29 = transpose_31.contiguous()
        transpose_31 = None
        reshape_7 = attn_output_29.reshape(1, 1, -1)
        attn_output_29 = None
        attn_output_30 = reshape_7.contiguous()
        reshape_7 = None
        attn_output_31 = torch._C._nn.linear(
            attn_output_30,
            l_self_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_30 = l_self_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_42 = torch.nn.functional.dropout(
            attn_output_31, p=0.0, training=False
        )
        attn_output_31 = None
        hidden_states_43 = hidden_states_40 + hidden_states_42
        hidden_states_40 = hidden_states_42 = None
        hidden_states_44 = torch.nn.functional.layer_norm(
            hidden_states_43,
            (512,),
            l_self_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_ = (
            l_self_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_
        ) = None
        linear_38 = torch._C._nn.linear(
            hidden_states_44,
            l_self_modules_layers_modules_3_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_3_modules_fc1_parameters_bias_,
        )
        hidden_states_44 = (
            l_self_modules_layers_modules_3_modules_fc1_parameters_weight_
        ) = l_self_modules_layers_modules_3_modules_fc1_parameters_bias_ = None
        hidden_states_45 = torch._C._nn.gelu(linear_38)
        linear_38 = None
        hidden_states_46 = torch.nn.functional.dropout(
            hidden_states_45, p=0.0, training=False
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
            hidden_states_47, p=0.0, training=False
        )
        hidden_states_47 = None
        hidden_states_49 = hidden_states_43 + hidden_states_48
        hidden_states_43 = hidden_states_48 = None
        hidden_states_50 = torch.nn.functional.layer_norm(
            hidden_states_49,
            (512,),
            l_self_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_40 = torch._C._nn.linear(
            hidden_states_50,
            l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_24 = linear_40 * 0.125
        linear_40 = None
        query_states_25 = query_states_24.view(1, 1, -1, 64)
        query_states_24 = None
        transpose_32 = query_states_25.transpose(1, 2)
        query_states_25 = None
        query_states_26 = transpose_32.contiguous()
        transpose_32 = None
        linear_41 = torch._C._nn.linear(
            hidden_states_50,
            l_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        key_states_16 = linear_41.view(1, -1, 8, 64)
        linear_41 = None
        linear_42 = torch._C._nn.linear(
            hidden_states_50,
            l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_50 = l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        value_states_16 = linear_42.view(1, -1, 8, 64)
        linear_42 = None
        transpose_33 = key_states_16.transpose(1, 2)
        key_states_16 = None
        key_states_17 = transpose_33.contiguous()
        transpose_33 = None
        transpose_34 = value_states_16.transpose(1, 2)
        value_states_16 = None
        value_states_17 = transpose_34.contiguous()
        transpose_34 = None
        attention_mask_4 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
            )
        ]
        query_8 = query_states_26.contiguous()
        query_states_26 = None
        key_8 = key_states_17.contiguous()
        value_8 = value_states_17.contiguous()
        attn_output_32 = torch._C._nn.scaled_dot_product_attention(
            query_8,
            key_8,
            value_8,
            attn_mask=attention_mask_4,
            dropout_p=0.0,
            scale=1.0,
            is_causal=False,
        )
        query_8 = key_8 = value_8 = attention_mask_4 = None
        transpose_35 = attn_output_32.transpose(1, 2)
        attn_output_32 = None
        attn_output_33 = transpose_35.contiguous()
        transpose_35 = None
        reshape_8 = attn_output_33.reshape(1, 1, -1)
        attn_output_33 = None
        attn_output_34 = reshape_8.contiguous()
        reshape_8 = None
        attn_output_35 = torch._C._nn.linear(
            attn_output_34,
            l_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_34 = l_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_51 = torch.nn.functional.dropout(
            attn_output_35, p=0.0, training=False
        )
        attn_output_35 = None
        hidden_states_52 = hidden_states_49 + hidden_states_51
        hidden_states_49 = hidden_states_51 = None
        hidden_states_53 = torch.nn.functional.layer_norm(
            hidden_states_52,
            (512,),
            l_self_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_44 = torch._C._nn.linear(
            hidden_states_53,
            l_self_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        hidden_states_53 = l_self_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_27 = linear_44 * 0.125
        linear_44 = None
        query_states_28 = query_states_27.view(1, 1, -1, 64)
        query_states_27 = None
        transpose_36 = query_states_28.transpose(1, 2)
        query_states_28 = None
        query_states_29 = transpose_36.contiguous()
        transpose_36 = None
        linear_45 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        key_states_18 = linear_45.view(1, -1, 8, 64)
        linear_45 = None
        linear_46 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        value_states_18 = linear_46.view(1, -1, 8, 64)
        linear_46 = None
        transpose_37 = key_states_18.transpose(1, 2)
        key_states_18 = None
        key_states_19 = transpose_37.contiguous()
        transpose_37 = None
        transpose_38 = value_states_18.transpose(1, 2)
        value_states_18 = None
        value_states_19 = transpose_38.contiguous()
        transpose_38 = None
        query_9 = query_states_29.contiguous()
        query_states_29 = None
        key_9 = key_states_19.contiguous()
        value_9 = value_states_19.contiguous()
        attn_output_36 = torch._C._nn.scaled_dot_product_attention(
            query_9,
            key_9,
            value_9,
            attn_mask=None,
            dropout_p=0.0,
            scale=1.0,
            is_causal=False,
        )
        query_9 = key_9 = value_9 = None
        transpose_39 = attn_output_36.transpose(1, 2)
        attn_output_36 = None
        attn_output_37 = transpose_39.contiguous()
        transpose_39 = None
        reshape_9 = attn_output_37.reshape(1, 1, -1)
        attn_output_37 = None
        attn_output_38 = reshape_9.contiguous()
        reshape_9 = None
        attn_output_39 = torch._C._nn.linear(
            attn_output_38,
            l_self_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_38 = l_self_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_54 = torch.nn.functional.dropout(
            attn_output_39, p=0.0, training=False
        )
        attn_output_39 = None
        hidden_states_55 = hidden_states_52 + hidden_states_54
        hidden_states_52 = hidden_states_54 = None
        hidden_states_56 = torch.nn.functional.layer_norm(
            hidden_states_55,
            (512,),
            l_self_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_ = (
            l_self_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_
        ) = None
        linear_48 = torch._C._nn.linear(
            hidden_states_56,
            l_self_modules_layers_modules_4_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_4_modules_fc1_parameters_bias_,
        )
        hidden_states_56 = (
            l_self_modules_layers_modules_4_modules_fc1_parameters_weight_
        ) = l_self_modules_layers_modules_4_modules_fc1_parameters_bias_ = None
        hidden_states_57 = torch._C._nn.gelu(linear_48)
        linear_48 = None
        hidden_states_58 = torch.nn.functional.dropout(
            hidden_states_57, p=0.0, training=False
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
            hidden_states_59, p=0.0, training=False
        )
        hidden_states_59 = None
        hidden_states_61 = hidden_states_55 + hidden_states_60
        hidden_states_55 = hidden_states_60 = None
        hidden_states_62 = torch.nn.functional.layer_norm(
            hidden_states_61,
            (512,),
            l_self_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_50 = torch._C._nn.linear(
            hidden_states_62,
            l_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_30 = linear_50 * 0.125
        linear_50 = None
        query_states_31 = query_states_30.view(1, 1, -1, 64)
        query_states_30 = None
        transpose_40 = query_states_31.transpose(1, 2)
        query_states_31 = None
        query_states_32 = transpose_40.contiguous()
        transpose_40 = None
        linear_51 = torch._C._nn.linear(
            hidden_states_62,
            l_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        key_states_20 = linear_51.view(1, -1, 8, 64)
        linear_51 = None
        linear_52 = torch._C._nn.linear(
            hidden_states_62,
            l_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_62 = l_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        value_states_20 = linear_52.view(1, -1, 8, 64)
        linear_52 = None
        transpose_41 = key_states_20.transpose(1, 2)
        key_states_20 = None
        key_states_21 = transpose_41.contiguous()
        transpose_41 = None
        transpose_42 = value_states_20.transpose(1, 2)
        value_states_20 = None
        value_states_21 = transpose_42.contiguous()
        transpose_42 = None
        attention_mask_5 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
            )
        ]
        causal_mask = None
        query_10 = query_states_32.contiguous()
        query_states_32 = None
        key_10 = key_states_21.contiguous()
        value_10 = value_states_21.contiguous()
        attn_output_40 = torch._C._nn.scaled_dot_product_attention(
            query_10,
            key_10,
            value_10,
            attn_mask=attention_mask_5,
            dropout_p=0.0,
            scale=1.0,
            is_causal=False,
        )
        query_10 = key_10 = value_10 = attention_mask_5 = None
        transpose_43 = attn_output_40.transpose(1, 2)
        attn_output_40 = None
        attn_output_41 = transpose_43.contiguous()
        transpose_43 = None
        reshape_10 = attn_output_41.reshape(1, 1, -1)
        attn_output_41 = None
        attn_output_42 = reshape_10.contiguous()
        reshape_10 = None
        attn_output_43 = torch._C._nn.linear(
            attn_output_42,
            l_self_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_42 = l_self_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_63 = torch.nn.functional.dropout(
            attn_output_43, p=0.0, training=False
        )
        attn_output_43 = None
        hidden_states_64 = hidden_states_61 + hidden_states_63
        hidden_states_61 = hidden_states_63 = None
        hidden_states_65 = torch.nn.functional.layer_norm(
            hidden_states_64,
            (512,),
            l_self_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_54 = torch._C._nn.linear(
            hidden_states_65,
            l_self_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        hidden_states_65 = l_self_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_33 = linear_54 * 0.125
        linear_54 = None
        query_states_34 = query_states_33.view(1, 1, -1, 64)
        query_states_33 = None
        transpose_44 = query_states_34.transpose(1, 2)
        query_states_34 = None
        query_states_35 = transpose_44.contiguous()
        transpose_44 = None
        linear_55 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        key_states_22 = linear_55.view(1, -1, 8, 64)
        linear_55 = None
        linear_56 = torch._C._nn.linear(
            l_encoder_hidden_states_,
            l_self_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_encoder_hidden_states_ = l_self_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        value_states_22 = linear_56.view(1, -1, 8, 64)
        linear_56 = None
        transpose_45 = key_states_22.transpose(1, 2)
        key_states_22 = None
        key_states_23 = transpose_45.contiguous()
        transpose_45 = None
        transpose_46 = value_states_22.transpose(1, 2)
        value_states_22 = None
        value_states_23 = transpose_46.contiguous()
        transpose_46 = None
        query_11 = query_states_35.contiguous()
        query_states_35 = None
        key_11 = key_states_23.contiguous()
        value_11 = value_states_23.contiguous()
        attn_output_44 = torch._C._nn.scaled_dot_product_attention(
            query_11,
            key_11,
            value_11,
            attn_mask=None,
            dropout_p=0.0,
            scale=1.0,
            is_causal=False,
        )
        query_11 = key_11 = value_11 = None
        transpose_47 = attn_output_44.transpose(1, 2)
        attn_output_44 = None
        attn_output_45 = transpose_47.contiguous()
        transpose_47 = None
        reshape_11 = attn_output_45.reshape(1, 1, -1)
        attn_output_45 = None
        attn_output_46 = reshape_11.contiguous()
        reshape_11 = None
        attn_output_47 = torch._C._nn.linear(
            attn_output_46,
            l_self_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_46 = l_self_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_66 = torch.nn.functional.dropout(
            attn_output_47, p=0.0, training=False
        )
        attn_output_47 = None
        hidden_states_67 = hidden_states_64 + hidden_states_66
        hidden_states_64 = hidden_states_66 = None
        hidden_states_68 = torch.nn.functional.layer_norm(
            hidden_states_67,
            (512,),
            l_self_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_,
            l_self_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_ = (
            l_self_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_
        ) = None
        linear_58 = torch._C._nn.linear(
            hidden_states_68,
            l_self_modules_layers_modules_5_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_5_modules_fc1_parameters_bias_,
        )
        hidden_states_68 = (
            l_self_modules_layers_modules_5_modules_fc1_parameters_weight_
        ) = l_self_modules_layers_modules_5_modules_fc1_parameters_bias_ = None
        hidden_states_69 = torch._C._nn.gelu(linear_58)
        linear_58 = None
        hidden_states_70 = torch.nn.functional.dropout(
            hidden_states_69, p=0.0, training=False
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
            hidden_states_71, p=0.0, training=False
        )
        hidden_states_71 = None
        hidden_states_73 = hidden_states_67 + hidden_states_72
        hidden_states_67 = hidden_states_72 = None
        hidden_states_74 = torch.nn.functional.layer_norm(
            hidden_states_73,
            (512,),
            l_self_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_73 = (
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
            hidden_states_74,
        )
