import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_input_values_: torch.Tensor,
        L_self_modules_unispeech_sat_modules_feature_extractor_modules_conv_layers_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_feature_extractor_modules_conv_layers_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_feature_extractor_modules_conv_layers_modules_0_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_feature_extractor_modules_conv_layers_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_feature_extractor_modules_conv_layers_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_feature_projection_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_feature_projection_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_feature_projection_modules_projection_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_feature_projection_modules_projection_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_pos_conv_embed_modules_conv_modules_parametrizations_modules_weight_parameters_original0_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_pos_conv_embed_modules_conv_modules_parametrizations_modules_weight_parameters_original1_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_pos_conv_embed_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_attention_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_attention_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_attention_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_attention_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_attention_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_attention_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_attention_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_attention_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_feed_forward_modules_intermediate_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_feed_forward_modules_intermediate_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_feed_forward_modules_output_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_feed_forward_modules_output_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_attention_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_attention_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_attention_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_attention_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_attention_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_attention_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_attention_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_attention_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_feed_forward_modules_intermediate_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_feed_forward_modules_intermediate_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_feed_forward_modules_output_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_feed_forward_modules_output_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_attention_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_attention_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_attention_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_attention_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_attention_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_attention_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_attention_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_attention_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_feed_forward_modules_intermediate_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_feed_forward_modules_intermediate_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_feed_forward_modules_output_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_feed_forward_modules_output_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_attention_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_attention_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_attention_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_attention_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_attention_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_attention_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_attention_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_attention_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_feed_forward_modules_intermediate_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_feed_forward_modules_intermediate_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_feed_forward_modules_output_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_feed_forward_modules_output_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_projector_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_projector_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_input_values_ = L_input_values_
        l_self_modules_unispeech_sat_modules_feature_extractor_modules_conv_layers_modules_0_modules_conv_parameters_weight_ = L_self_modules_unispeech_sat_modules_feature_extractor_modules_conv_layers_modules_0_modules_conv_parameters_weight_
        l_self_modules_unispeech_sat_modules_feature_extractor_modules_conv_layers_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_unispeech_sat_modules_feature_extractor_modules_conv_layers_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_unispeech_sat_modules_feature_extractor_modules_conv_layers_modules_0_modules_layer_norm_parameters_bias_ = L_self_modules_unispeech_sat_modules_feature_extractor_modules_conv_layers_modules_0_modules_layer_norm_parameters_bias_
        l_self_modules_unispeech_sat_modules_feature_extractor_modules_conv_layers_modules_1_modules_conv_parameters_weight_ = L_self_modules_unispeech_sat_modules_feature_extractor_modules_conv_layers_modules_1_modules_conv_parameters_weight_
        l_self_modules_unispeech_sat_modules_feature_extractor_modules_conv_layers_modules_2_modules_conv_parameters_weight_ = L_self_modules_unispeech_sat_modules_feature_extractor_modules_conv_layers_modules_2_modules_conv_parameters_weight_
        l_self_modules_unispeech_sat_modules_feature_projection_modules_layer_norm_parameters_weight_ = L_self_modules_unispeech_sat_modules_feature_projection_modules_layer_norm_parameters_weight_
        l_self_modules_unispeech_sat_modules_feature_projection_modules_layer_norm_parameters_bias_ = L_self_modules_unispeech_sat_modules_feature_projection_modules_layer_norm_parameters_bias_
        l_self_modules_unispeech_sat_modules_feature_projection_modules_projection_parameters_weight_ = L_self_modules_unispeech_sat_modules_feature_projection_modules_projection_parameters_weight_
        l_self_modules_unispeech_sat_modules_feature_projection_modules_projection_parameters_bias_ = L_self_modules_unispeech_sat_modules_feature_projection_modules_projection_parameters_bias_
        l_self_modules_unispeech_sat_modules_encoder_modules_pos_conv_embed_modules_conv_modules_parametrizations_modules_weight_parameters_original0_ = L_self_modules_unispeech_sat_modules_encoder_modules_pos_conv_embed_modules_conv_modules_parametrizations_modules_weight_parameters_original0_
        l_self_modules_unispeech_sat_modules_encoder_modules_pos_conv_embed_modules_conv_modules_parametrizations_modules_weight_parameters_original1_ = L_self_modules_unispeech_sat_modules_encoder_modules_pos_conv_embed_modules_conv_modules_parametrizations_modules_weight_parameters_original1_
        l_self_modules_unispeech_sat_modules_encoder_modules_pos_conv_embed_modules_conv_parameters_bias_ = L_self_modules_unispeech_sat_modules_encoder_modules_pos_conv_embed_modules_conv_parameters_bias_
        l_self_modules_unispeech_sat_modules_encoder_modules_layer_norm_parameters_weight_ = L_self_modules_unispeech_sat_modules_encoder_modules_layer_norm_parameters_weight_
        l_self_modules_unispeech_sat_modules_encoder_modules_layer_norm_parameters_bias_ = L_self_modules_unispeech_sat_modules_encoder_modules_layer_norm_parameters_bias_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_attention_modules_q_proj_parameters_weight_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_attention_modules_q_proj_parameters_weight_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_attention_modules_q_proj_parameters_bias_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_attention_modules_q_proj_parameters_bias_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_attention_modules_k_proj_parameters_weight_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_attention_modules_k_proj_parameters_weight_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_attention_modules_k_proj_parameters_bias_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_attention_modules_k_proj_parameters_bias_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_attention_modules_v_proj_parameters_weight_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_attention_modules_v_proj_parameters_weight_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_attention_modules_v_proj_parameters_bias_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_attention_modules_v_proj_parameters_bias_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_attention_modules_out_proj_parameters_weight_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_attention_modules_out_proj_parameters_weight_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_attention_modules_out_proj_parameters_bias_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_attention_modules_out_proj_parameters_bias_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_layer_norm_parameters_bias_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_layer_norm_parameters_bias_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_feed_forward_modules_intermediate_dense_parameters_weight_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_feed_forward_modules_intermediate_dense_parameters_weight_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_feed_forward_modules_intermediate_dense_parameters_bias_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_feed_forward_modules_intermediate_dense_parameters_bias_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_feed_forward_modules_output_dense_parameters_weight_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_feed_forward_modules_output_dense_parameters_weight_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_feed_forward_modules_output_dense_parameters_bias_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_feed_forward_modules_output_dense_parameters_bias_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_attention_modules_q_proj_parameters_weight_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_attention_modules_q_proj_parameters_weight_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_attention_modules_q_proj_parameters_bias_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_attention_modules_q_proj_parameters_bias_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_attention_modules_k_proj_parameters_weight_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_attention_modules_k_proj_parameters_weight_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_attention_modules_k_proj_parameters_bias_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_attention_modules_k_proj_parameters_bias_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_attention_modules_v_proj_parameters_weight_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_attention_modules_v_proj_parameters_weight_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_attention_modules_v_proj_parameters_bias_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_attention_modules_v_proj_parameters_bias_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_attention_modules_out_proj_parameters_weight_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_attention_modules_out_proj_parameters_weight_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_attention_modules_out_proj_parameters_bias_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_attention_modules_out_proj_parameters_bias_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_layer_norm_parameters_bias_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_layer_norm_parameters_bias_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_feed_forward_modules_intermediate_dense_parameters_weight_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_feed_forward_modules_intermediate_dense_parameters_weight_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_feed_forward_modules_intermediate_dense_parameters_bias_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_feed_forward_modules_intermediate_dense_parameters_bias_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_feed_forward_modules_output_dense_parameters_weight_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_feed_forward_modules_output_dense_parameters_weight_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_feed_forward_modules_output_dense_parameters_bias_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_feed_forward_modules_output_dense_parameters_bias_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_attention_modules_q_proj_parameters_weight_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_attention_modules_q_proj_parameters_weight_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_attention_modules_q_proj_parameters_bias_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_attention_modules_q_proj_parameters_bias_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_attention_modules_k_proj_parameters_weight_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_attention_modules_k_proj_parameters_weight_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_attention_modules_k_proj_parameters_bias_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_attention_modules_k_proj_parameters_bias_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_attention_modules_v_proj_parameters_weight_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_attention_modules_v_proj_parameters_weight_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_attention_modules_v_proj_parameters_bias_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_attention_modules_v_proj_parameters_bias_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_attention_modules_out_proj_parameters_weight_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_attention_modules_out_proj_parameters_weight_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_attention_modules_out_proj_parameters_bias_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_attention_modules_out_proj_parameters_bias_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_layer_norm_parameters_weight_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_layer_norm_parameters_weight_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_layer_norm_parameters_bias_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_layer_norm_parameters_bias_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_feed_forward_modules_intermediate_dense_parameters_weight_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_feed_forward_modules_intermediate_dense_parameters_weight_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_feed_forward_modules_intermediate_dense_parameters_bias_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_feed_forward_modules_intermediate_dense_parameters_bias_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_feed_forward_modules_output_dense_parameters_weight_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_feed_forward_modules_output_dense_parameters_weight_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_feed_forward_modules_output_dense_parameters_bias_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_feed_forward_modules_output_dense_parameters_bias_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_attention_modules_q_proj_parameters_weight_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_attention_modules_q_proj_parameters_weight_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_attention_modules_q_proj_parameters_bias_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_attention_modules_q_proj_parameters_bias_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_attention_modules_k_proj_parameters_weight_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_attention_modules_k_proj_parameters_weight_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_attention_modules_k_proj_parameters_bias_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_attention_modules_k_proj_parameters_bias_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_attention_modules_v_proj_parameters_weight_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_attention_modules_v_proj_parameters_weight_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_attention_modules_v_proj_parameters_bias_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_attention_modules_v_proj_parameters_bias_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_attention_modules_out_proj_parameters_weight_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_attention_modules_out_proj_parameters_weight_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_attention_modules_out_proj_parameters_bias_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_attention_modules_out_proj_parameters_bias_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_layer_norm_parameters_weight_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_layer_norm_parameters_weight_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_layer_norm_parameters_bias_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_layer_norm_parameters_bias_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_feed_forward_modules_intermediate_dense_parameters_weight_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_feed_forward_modules_intermediate_dense_parameters_weight_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_feed_forward_modules_intermediate_dense_parameters_bias_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_feed_forward_modules_intermediate_dense_parameters_bias_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_feed_forward_modules_output_dense_parameters_weight_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_feed_forward_modules_output_dense_parameters_weight_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_feed_forward_modules_output_dense_parameters_bias_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_feed_forward_modules_output_dense_parameters_bias_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_ = L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_
        l_self_modules_projector_parameters_weight_ = (
            L_self_modules_projector_parameters_weight_
        )
        l_self_modules_projector_parameters_bias_ = (
            L_self_modules_projector_parameters_bias_
        )
        l_self_modules_classifier_parameters_weight_ = (
            L_self_modules_classifier_parameters_weight_
        )
        l_self_modules_classifier_parameters_bias_ = (
            L_self_modules_classifier_parameters_bias_
        )
        hidden_states = l_input_values_[(slice(None, None, None), None)]
        l_input_values_ = None
        hidden_states_1 = torch.conv1d(
            hidden_states,
            l_self_modules_unispeech_sat_modules_feature_extractor_modules_conv_layers_modules_0_modules_conv_parameters_weight_,
            None,
            (4,),
            (0,),
            (1,),
            1,
        )
        hidden_states = l_self_modules_unispeech_sat_modules_feature_extractor_modules_conv_layers_modules_0_modules_conv_parameters_weight_ = (None)
        hidden_states_2 = torch.nn.functional.group_norm(
            hidden_states_1,
            32,
            l_self_modules_unispeech_sat_modules_feature_extractor_modules_conv_layers_modules_0_modules_layer_norm_parameters_weight_,
            l_self_modules_unispeech_sat_modules_feature_extractor_modules_conv_layers_modules_0_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_1 = l_self_modules_unispeech_sat_modules_feature_extractor_modules_conv_layers_modules_0_modules_layer_norm_parameters_weight_ = l_self_modules_unispeech_sat_modules_feature_extractor_modules_conv_layers_modules_0_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_3 = torch._C._nn.gelu(hidden_states_2)
        hidden_states_2 = None
        hidden_states_4 = torch.conv1d(
            hidden_states_3,
            l_self_modules_unispeech_sat_modules_feature_extractor_modules_conv_layers_modules_1_modules_conv_parameters_weight_,
            None,
            (4,),
            (0,),
            (1,),
            1,
        )
        hidden_states_3 = l_self_modules_unispeech_sat_modules_feature_extractor_modules_conv_layers_modules_1_modules_conv_parameters_weight_ = (None)
        hidden_states_5 = torch._C._nn.gelu(hidden_states_4)
        hidden_states_4 = None
        hidden_states_6 = torch.conv1d(
            hidden_states_5,
            l_self_modules_unispeech_sat_modules_feature_extractor_modules_conv_layers_modules_2_modules_conv_parameters_weight_,
            None,
            (4,),
            (0,),
            (1,),
            1,
        )
        hidden_states_5 = l_self_modules_unispeech_sat_modules_feature_extractor_modules_conv_layers_modules_2_modules_conv_parameters_weight_ = (None)
        hidden_states_7 = torch._C._nn.gelu(hidden_states_6)
        hidden_states_6 = None
        extract_features = hidden_states_7.transpose(1, 2)
        hidden_states_7 = None
        norm_hidden_states = torch.nn.functional.layer_norm(
            extract_features,
            (32,),
            l_self_modules_unispeech_sat_modules_feature_projection_modules_layer_norm_parameters_weight_,
            l_self_modules_unispeech_sat_modules_feature_projection_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        extract_features = l_self_modules_unispeech_sat_modules_feature_projection_modules_layer_norm_parameters_weight_ = l_self_modules_unispeech_sat_modules_feature_projection_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_8 = torch._C._nn.linear(
            norm_hidden_states,
            l_self_modules_unispeech_sat_modules_feature_projection_modules_projection_parameters_weight_,
            l_self_modules_unispeech_sat_modules_feature_projection_modules_projection_parameters_bias_,
        )
        norm_hidden_states = l_self_modules_unispeech_sat_modules_feature_projection_modules_projection_parameters_weight_ = l_self_modules_unispeech_sat_modules_feature_projection_modules_projection_parameters_bias_ = (None)
        hidden_states_9 = torch.nn.functional.dropout(
            hidden_states_8, 0.0, False, False
        )
        hidden_states_8 = None
        hidden_states_10 = hidden_states_9.transpose(1, 2)
        x = torch._weight_norm(
            l_self_modules_unispeech_sat_modules_encoder_modules_pos_conv_embed_modules_conv_modules_parametrizations_modules_weight_parameters_original1_,
            l_self_modules_unispeech_sat_modules_encoder_modules_pos_conv_embed_modules_conv_modules_parametrizations_modules_weight_parameters_original0_,
            2,
        )
        l_self_modules_unispeech_sat_modules_encoder_modules_pos_conv_embed_modules_conv_modules_parametrizations_modules_weight_parameters_original1_ = l_self_modules_unispeech_sat_modules_encoder_modules_pos_conv_embed_modules_conv_modules_parametrizations_modules_weight_parameters_original0_ = (None)
        hidden_states_11 = torch.conv1d(
            hidden_states_10,
            x,
            l_self_modules_unispeech_sat_modules_encoder_modules_pos_conv_embed_modules_conv_parameters_bias_,
            (1,),
            (8,),
            (1,),
            2,
        )
        hidden_states_10 = (
            x
        ) = l_self_modules_unispeech_sat_modules_encoder_modules_pos_conv_embed_modules_conv_parameters_bias_ = (None)
        hidden_states_12 = hidden_states_11[
            (slice(None, None, None), slice(None, None, None), slice(None, -1, None))
        ]
        hidden_states_11 = None
        hidden_states_13 = torch._C._nn.gelu(hidden_states_12)
        hidden_states_12 = None
        hidden_states_14 = hidden_states_13.transpose(1, 2)
        hidden_states_13 = None
        hidden_states_15 = hidden_states_9 + hidden_states_14
        hidden_states_9 = hidden_states_14 = None
        hidden_states_16 = torch.nn.functional.layer_norm(
            hidden_states_15,
            (16,),
            l_self_modules_unispeech_sat_modules_encoder_modules_layer_norm_parameters_weight_,
            l_self_modules_unispeech_sat_modules_encoder_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_15 = l_self_modules_unispeech_sat_modules_encoder_modules_layer_norm_parameters_weight_ = l_self_modules_unispeech_sat_modules_encoder_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_17 = torch.nn.functional.dropout(
            hidden_states_16, 0.1, False, False
        )
        hidden_states_16 = None
        dropout_probability = torch.rand([])
        dropout_probability = None
        linear_1 = torch._C._nn.linear(
            hidden_states_17,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_attention_modules_q_proj_parameters_weight_,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_attention_modules_q_proj_parameters_bias_,
        )
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_attention_modules_q_proj_parameters_weight_ = l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_attention_modules_q_proj_parameters_bias_ = (None)
        view = linear_1.view(1, 1248, -1, 8)
        linear_1 = None
        query_states = view.transpose(1, 2)
        view = None
        linear_2 = torch._C._nn.linear(
            hidden_states_17,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_attention_modules_k_proj_parameters_weight_,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_attention_modules_k_proj_parameters_bias_,
        )
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_attention_modules_k_proj_parameters_weight_ = l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_attention_modules_k_proj_parameters_bias_ = (None)
        view_1 = linear_2.view(1, 1248, -1, 8)
        linear_2 = None
        key_states = view_1.transpose(1, 2)
        view_1 = None
        linear_3 = torch._C._nn.linear(
            hidden_states_17,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_attention_modules_v_proj_parameters_weight_,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_attention_modules_v_proj_parameters_bias_,
        )
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_attention_modules_v_proj_parameters_weight_ = l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_attention_modules_v_proj_parameters_bias_ = (None)
        view_2 = linear_3.view(1, 1248, -1, 8)
        linear_3 = None
        value_states = view_2.transpose(1, 2)
        view_2 = None
        query = query_states.contiguous()
        query_states = None
        key = key_states.contiguous()
        key_states = None
        value = value_states.contiguous()
        value_states = None
        attn_output = torch._C._nn.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            scale=0.3535533905932738,
            is_causal=False,
        )
        query = key = value = None
        transpose_6 = attn_output.transpose(1, 2)
        attn_output = None
        attn_output_1 = transpose_6.contiguous()
        transpose_6 = None
        reshape = attn_output_1.reshape(1, 1248, -1)
        attn_output_1 = None
        attn_output_2 = reshape.contiguous()
        reshape = None
        attn_output_3 = torch._C._nn.linear(
            attn_output_2,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_attention_modules_out_proj_parameters_weight_,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_attention_modules_out_proj_parameters_bias_,
        )
        attn_output_2 = l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_attention_modules_out_proj_parameters_weight_ = l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_attention_modules_out_proj_parameters_bias_ = (None)
        hidden_states_18 = torch.nn.functional.dropout(attn_output_3, 0.1, False, False)
        attn_output_3 = None
        hidden_states_19 = hidden_states_17 + hidden_states_18
        hidden_states_17 = hidden_states_18 = None
        hidden_states_20 = torch.nn.functional.layer_norm(
            hidden_states_19,
            (16,),
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_layer_norm_parameters_weight_,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_19 = l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_layer_norm_parameters_weight_ = l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_21 = torch._C._nn.linear(
            hidden_states_20,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_feed_forward_modules_intermediate_dense_parameters_weight_,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_feed_forward_modules_intermediate_dense_parameters_bias_,
        )
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_feed_forward_modules_intermediate_dense_parameters_weight_ = l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_feed_forward_modules_intermediate_dense_parameters_bias_ = (None)
        hidden_states_22 = torch._C._nn.gelu(hidden_states_21)
        hidden_states_21 = None
        hidden_states_23 = torch.nn.functional.dropout(
            hidden_states_22, 0.1, False, False
        )
        hidden_states_22 = None
        hidden_states_24 = torch._C._nn.linear(
            hidden_states_23,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_feed_forward_modules_output_dense_parameters_weight_,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_feed_forward_modules_output_dense_parameters_bias_,
        )
        hidden_states_23 = l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_feed_forward_modules_output_dense_parameters_weight_ = l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_feed_forward_modules_output_dense_parameters_bias_ = (None)
        hidden_states_25 = torch.nn.functional.dropout(
            hidden_states_24, 0.1, False, False
        )
        hidden_states_24 = None
        hidden_states_26 = hidden_states_20 + hidden_states_25
        hidden_states_20 = hidden_states_25 = None
        hidden_states_27 = torch.nn.functional.layer_norm(
            hidden_states_26,
            (16,),
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_26 = l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_ = l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_ = (None)
        dropout_probability_1 = torch.rand([])
        dropout_probability_1 = None
        linear_7 = torch._C._nn.linear(
            hidden_states_27,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_attention_modules_q_proj_parameters_weight_,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_attention_modules_q_proj_parameters_bias_,
        )
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_attention_modules_q_proj_parameters_weight_ = l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_attention_modules_q_proj_parameters_bias_ = (None)
        view_3 = linear_7.view(1, 1248, -1, 8)
        linear_7 = None
        query_states_1 = view_3.transpose(1, 2)
        view_3 = None
        linear_8 = torch._C._nn.linear(
            hidden_states_27,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_attention_modules_k_proj_parameters_weight_,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_attention_modules_k_proj_parameters_bias_,
        )
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_attention_modules_k_proj_parameters_weight_ = l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_attention_modules_k_proj_parameters_bias_ = (None)
        view_4 = linear_8.view(1, 1248, -1, 8)
        linear_8 = None
        key_states_1 = view_4.transpose(1, 2)
        view_4 = None
        linear_9 = torch._C._nn.linear(
            hidden_states_27,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_attention_modules_v_proj_parameters_weight_,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_attention_modules_v_proj_parameters_bias_,
        )
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_attention_modules_v_proj_parameters_weight_ = l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_attention_modules_v_proj_parameters_bias_ = (None)
        view_5 = linear_9.view(1, 1248, -1, 8)
        linear_9 = None
        value_states_1 = view_5.transpose(1, 2)
        view_5 = None
        query_1 = query_states_1.contiguous()
        query_states_1 = None
        key_1 = key_states_1.contiguous()
        key_states_1 = None
        value_1 = value_states_1.contiguous()
        value_states_1 = None
        attn_output_4 = torch._C._nn.scaled_dot_product_attention(
            query_1,
            key_1,
            value_1,
            attn_mask=None,
            dropout_p=0.0,
            scale=0.3535533905932738,
            is_causal=False,
        )
        query_1 = key_1 = value_1 = None
        transpose_10 = attn_output_4.transpose(1, 2)
        attn_output_4 = None
        attn_output_5 = transpose_10.contiguous()
        transpose_10 = None
        reshape_1 = attn_output_5.reshape(1, 1248, -1)
        attn_output_5 = None
        attn_output_6 = reshape_1.contiguous()
        reshape_1 = None
        attn_output_7 = torch._C._nn.linear(
            attn_output_6,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_attention_modules_out_proj_parameters_weight_,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_attention_modules_out_proj_parameters_bias_,
        )
        attn_output_6 = l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_attention_modules_out_proj_parameters_weight_ = l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_attention_modules_out_proj_parameters_bias_ = (None)
        hidden_states_28 = torch.nn.functional.dropout(attn_output_7, 0.1, False, False)
        attn_output_7 = None
        hidden_states_29 = hidden_states_27 + hidden_states_28
        hidden_states_27 = hidden_states_28 = None
        hidden_states_30 = torch.nn.functional.layer_norm(
            hidden_states_29,
            (16,),
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_layer_norm_parameters_weight_,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_29 = l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_layer_norm_parameters_weight_ = l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_31 = torch._C._nn.linear(
            hidden_states_30,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_feed_forward_modules_intermediate_dense_parameters_weight_,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_feed_forward_modules_intermediate_dense_parameters_bias_,
        )
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_feed_forward_modules_intermediate_dense_parameters_weight_ = l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_feed_forward_modules_intermediate_dense_parameters_bias_ = (None)
        hidden_states_32 = torch._C._nn.gelu(hidden_states_31)
        hidden_states_31 = None
        hidden_states_33 = torch.nn.functional.dropout(
            hidden_states_32, 0.1, False, False
        )
        hidden_states_32 = None
        hidden_states_34 = torch._C._nn.linear(
            hidden_states_33,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_feed_forward_modules_output_dense_parameters_weight_,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_feed_forward_modules_output_dense_parameters_bias_,
        )
        hidden_states_33 = l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_feed_forward_modules_output_dense_parameters_weight_ = l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_feed_forward_modules_output_dense_parameters_bias_ = (None)
        hidden_states_35 = torch.nn.functional.dropout(
            hidden_states_34, 0.1, False, False
        )
        hidden_states_34 = None
        hidden_states_36 = hidden_states_30 + hidden_states_35
        hidden_states_30 = hidden_states_35 = None
        hidden_states_37 = torch.nn.functional.layer_norm(
            hidden_states_36,
            (16,),
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_36 = l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_ = l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_ = (None)
        dropout_probability_2 = torch.rand([])
        dropout_probability_2 = None
        linear_13 = torch._C._nn.linear(
            hidden_states_37,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_attention_modules_q_proj_parameters_weight_,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_attention_modules_q_proj_parameters_bias_,
        )
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_attention_modules_q_proj_parameters_weight_ = l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_attention_modules_q_proj_parameters_bias_ = (None)
        view_6 = linear_13.view(1, 1248, -1, 8)
        linear_13 = None
        query_states_2 = view_6.transpose(1, 2)
        view_6 = None
        linear_14 = torch._C._nn.linear(
            hidden_states_37,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_attention_modules_k_proj_parameters_weight_,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_attention_modules_k_proj_parameters_bias_,
        )
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_attention_modules_k_proj_parameters_weight_ = l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_attention_modules_k_proj_parameters_bias_ = (None)
        view_7 = linear_14.view(1, 1248, -1, 8)
        linear_14 = None
        key_states_2 = view_7.transpose(1, 2)
        view_7 = None
        linear_15 = torch._C._nn.linear(
            hidden_states_37,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_attention_modules_v_proj_parameters_weight_,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_attention_modules_v_proj_parameters_bias_,
        )
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_attention_modules_v_proj_parameters_weight_ = l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_attention_modules_v_proj_parameters_bias_ = (None)
        view_8 = linear_15.view(1, 1248, -1, 8)
        linear_15 = None
        value_states_2 = view_8.transpose(1, 2)
        view_8 = None
        query_2 = query_states_2.contiguous()
        query_states_2 = None
        key_2 = key_states_2.contiguous()
        key_states_2 = None
        value_2 = value_states_2.contiguous()
        value_states_2 = None
        attn_output_8 = torch._C._nn.scaled_dot_product_attention(
            query_2,
            key_2,
            value_2,
            attn_mask=None,
            dropout_p=0.0,
            scale=0.3535533905932738,
            is_causal=False,
        )
        query_2 = key_2 = value_2 = None
        transpose_14 = attn_output_8.transpose(1, 2)
        attn_output_8 = None
        attn_output_9 = transpose_14.contiguous()
        transpose_14 = None
        reshape_2 = attn_output_9.reshape(1, 1248, -1)
        attn_output_9 = None
        attn_output_10 = reshape_2.contiguous()
        reshape_2 = None
        attn_output_11 = torch._C._nn.linear(
            attn_output_10,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_attention_modules_out_proj_parameters_weight_,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_attention_modules_out_proj_parameters_bias_,
        )
        attn_output_10 = l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_attention_modules_out_proj_parameters_weight_ = l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_attention_modules_out_proj_parameters_bias_ = (None)
        hidden_states_38 = torch.nn.functional.dropout(
            attn_output_11, 0.1, False, False
        )
        attn_output_11 = None
        hidden_states_39 = hidden_states_37 + hidden_states_38
        hidden_states_37 = hidden_states_38 = None
        hidden_states_40 = torch.nn.functional.layer_norm(
            hidden_states_39,
            (16,),
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_layer_norm_parameters_weight_,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_39 = l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_layer_norm_parameters_weight_ = l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_41 = torch._C._nn.linear(
            hidden_states_40,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_feed_forward_modules_intermediate_dense_parameters_weight_,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_feed_forward_modules_intermediate_dense_parameters_bias_,
        )
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_feed_forward_modules_intermediate_dense_parameters_weight_ = l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_feed_forward_modules_intermediate_dense_parameters_bias_ = (None)
        hidden_states_42 = torch._C._nn.gelu(hidden_states_41)
        hidden_states_41 = None
        hidden_states_43 = torch.nn.functional.dropout(
            hidden_states_42, 0.1, False, False
        )
        hidden_states_42 = None
        hidden_states_44 = torch._C._nn.linear(
            hidden_states_43,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_feed_forward_modules_output_dense_parameters_weight_,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_feed_forward_modules_output_dense_parameters_bias_,
        )
        hidden_states_43 = l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_feed_forward_modules_output_dense_parameters_weight_ = l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_feed_forward_modules_output_dense_parameters_bias_ = (None)
        hidden_states_45 = torch.nn.functional.dropout(
            hidden_states_44, 0.1, False, False
        )
        hidden_states_44 = None
        hidden_states_46 = hidden_states_40 + hidden_states_45
        hidden_states_40 = hidden_states_45 = None
        hidden_states_47 = torch.nn.functional.layer_norm(
            hidden_states_46,
            (16,),
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_46 = l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_ = l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_ = (None)
        dropout_probability_3 = torch.rand([])
        dropout_probability_3 = None
        linear_19 = torch._C._nn.linear(
            hidden_states_47,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_attention_modules_q_proj_parameters_weight_,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_attention_modules_q_proj_parameters_bias_,
        )
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_attention_modules_q_proj_parameters_weight_ = l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_attention_modules_q_proj_parameters_bias_ = (None)
        view_9 = linear_19.view(1, 1248, -1, 8)
        linear_19 = None
        query_states_3 = view_9.transpose(1, 2)
        view_9 = None
        linear_20 = torch._C._nn.linear(
            hidden_states_47,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_attention_modules_k_proj_parameters_weight_,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_attention_modules_k_proj_parameters_bias_,
        )
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_attention_modules_k_proj_parameters_weight_ = l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_attention_modules_k_proj_parameters_bias_ = (None)
        view_10 = linear_20.view(1, 1248, -1, 8)
        linear_20 = None
        key_states_3 = view_10.transpose(1, 2)
        view_10 = None
        linear_21 = torch._C._nn.linear(
            hidden_states_47,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_attention_modules_v_proj_parameters_weight_,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_attention_modules_v_proj_parameters_bias_,
        )
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_attention_modules_v_proj_parameters_weight_ = l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_attention_modules_v_proj_parameters_bias_ = (None)
        view_11 = linear_21.view(1, 1248, -1, 8)
        linear_21 = None
        value_states_3 = view_11.transpose(1, 2)
        view_11 = None
        query_3 = query_states_3.contiguous()
        query_states_3 = None
        key_3 = key_states_3.contiguous()
        key_states_3 = None
        value_3 = value_states_3.contiguous()
        value_states_3 = None
        attn_output_12 = torch._C._nn.scaled_dot_product_attention(
            query_3,
            key_3,
            value_3,
            attn_mask=None,
            dropout_p=0.0,
            scale=0.3535533905932738,
            is_causal=False,
        )
        query_3 = key_3 = value_3 = None
        transpose_18 = attn_output_12.transpose(1, 2)
        attn_output_12 = None
        attn_output_13 = transpose_18.contiguous()
        transpose_18 = None
        reshape_3 = attn_output_13.reshape(1, 1248, -1)
        attn_output_13 = None
        attn_output_14 = reshape_3.contiguous()
        reshape_3 = None
        attn_output_15 = torch._C._nn.linear(
            attn_output_14,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_attention_modules_out_proj_parameters_weight_,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_attention_modules_out_proj_parameters_bias_,
        )
        attn_output_14 = l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_attention_modules_out_proj_parameters_weight_ = l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_attention_modules_out_proj_parameters_bias_ = (None)
        hidden_states_48 = torch.nn.functional.dropout(
            attn_output_15, 0.1, False, False
        )
        attn_output_15 = None
        hidden_states_49 = hidden_states_47 + hidden_states_48
        hidden_states_47 = hidden_states_48 = None
        hidden_states_50 = torch.nn.functional.layer_norm(
            hidden_states_49,
            (16,),
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_layer_norm_parameters_weight_,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_49 = l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_layer_norm_parameters_weight_ = l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_51 = torch._C._nn.linear(
            hidden_states_50,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_feed_forward_modules_intermediate_dense_parameters_weight_,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_feed_forward_modules_intermediate_dense_parameters_bias_,
        )
        l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_feed_forward_modules_intermediate_dense_parameters_weight_ = l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_feed_forward_modules_intermediate_dense_parameters_bias_ = (None)
        hidden_states_52 = torch._C._nn.gelu(hidden_states_51)
        hidden_states_51 = None
        hidden_states_53 = torch.nn.functional.dropout(
            hidden_states_52, 0.1, False, False
        )
        hidden_states_52 = None
        hidden_states_54 = torch._C._nn.linear(
            hidden_states_53,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_feed_forward_modules_output_dense_parameters_weight_,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_feed_forward_modules_output_dense_parameters_bias_,
        )
        hidden_states_53 = l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_feed_forward_modules_output_dense_parameters_weight_ = l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_feed_forward_modules_output_dense_parameters_bias_ = (None)
        hidden_states_55 = torch.nn.functional.dropout(
            hidden_states_54, 0.1, False, False
        )
        hidden_states_54 = None
        hidden_states_56 = hidden_states_50 + hidden_states_55
        hidden_states_50 = hidden_states_55 = None
        hidden_states_57 = torch.nn.functional.layer_norm(
            hidden_states_56,
            (16,),
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_,
            l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_56 = l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_ = l_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_ = (None)
        hidden_states_58 = torch._C._nn.linear(
            hidden_states_57,
            l_self_modules_projector_parameters_weight_,
            l_self_modules_projector_parameters_bias_,
        )
        hidden_states_57 = (
            l_self_modules_projector_parameters_weight_
        ) = l_self_modules_projector_parameters_bias_ = None
        pooled_output = hidden_states_58.mean(dim=1)
        hidden_states_58 = None
        logits = torch._C._nn.linear(
            pooled_output,
            l_self_modules_classifier_parameters_weight_,
            l_self_modules_classifier_parameters_bias_,
        )
        pooled_output = (
            l_self_modules_classifier_parameters_weight_
        ) = l_self_modules_classifier_parameters_bias_ = None
        return (logits,)
