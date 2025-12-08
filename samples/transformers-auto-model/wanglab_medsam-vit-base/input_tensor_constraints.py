from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 1024}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, 3, S0, S0], "L_kwargs_pixel_values_"),
    (
        [256],
        "L_self_modules_mask_decoder_modules_iou_prediction_head_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_mask_decoder_modules_iou_prediction_head_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_mask_decoder_modules_iou_prediction_head_modules_proj_in_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_mask_decoder_modules_iou_prediction_head_modules_proj_in_parameters_weight_",
    ),
    (
        [4],
        "L_self_modules_mask_decoder_modules_iou_prediction_head_modules_proj_out_parameters_bias_",
    ),
    (
        [4, 256],
        "L_self_modules_mask_decoder_modules_iou_prediction_head_modules_proj_out_parameters_weight_",
    ),
    ([1, 256], "L_self_modules_mask_decoder_modules_iou_token_parameters_weight_"),
    ([4, 256], "L_self_modules_mask_decoder_modules_mask_tokens_parameters_weight_"),
    (
        [256],
        "L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_0_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_0_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_0_modules_proj_in_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_0_modules_proj_in_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_0_modules_proj_out_parameters_bias_",
    ),
    (
        [32, 256],
        "L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_0_modules_proj_out_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_1_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_1_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_1_modules_proj_in_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_1_modules_proj_in_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_1_modules_proj_out_parameters_bias_",
    ),
    (
        [32, 256],
        "L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_1_modules_proj_out_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_2_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_2_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_2_modules_proj_in_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_2_modules_proj_in_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_2_modules_proj_out_parameters_bias_",
    ),
    (
        [32, 256],
        "L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_2_modules_proj_out_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_3_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_3_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_3_modules_proj_in_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_3_modules_proj_in_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_3_modules_proj_out_parameters_bias_",
    ),
    (
        [32, 256],
        "L_self_modules_mask_decoder_modules_output_hypernetworks_mlps_modules_3_modules_proj_out_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_k_proj_parameters_bias_",
    ),
    (
        [128, 256],
        "L_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_k_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 128],
        "L_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_out_proj_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_q_proj_parameters_bias_",
    ),
    (
        [128, 256],
        "L_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_q_proj_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_v_proj_parameters_bias_",
    ),
    (
        [128, 256],
        "L_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_modules_v_proj_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_mask_decoder_modules_transformer_modules_final_attn_token_to_image_scaling",
    ),
    (
        [256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layer_norm_final_attn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layer_norm_final_attn_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_k_proj_parameters_bias_",
    ),
    (
        [128, 256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_k_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 128],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_out_proj_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_q_proj_parameters_bias_",
    ),
    (
        [128, 256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_q_proj_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_v_proj_parameters_bias_",
    ),
    (
        [128, 256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_modules_v_proj_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_image_to_token_scaling",
    ),
    (
        [128],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_k_proj_parameters_bias_",
    ),
    (
        [128, 256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_k_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 128],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_out_proj_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_q_proj_parameters_bias_",
    ),
    (
        [128, 256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_q_proj_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_v_proj_parameters_bias_",
    ),
    (
        [128, 256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_modules_v_proj_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_cross_attn_token_to_image_scaling",
    ),
    (
        [256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm2_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm3_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm3_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm4_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_layer_norm4_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_mlp_modules_lin1_parameters_bias_",
    ),
    (
        [2048, 256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_mlp_modules_lin1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_mlp_modules_lin2_parameters_bias_",
    ),
    (
        [256, 2048],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_mlp_modules_lin2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_0_modules_self_attn_scaling",
    ),
    (
        [128],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_k_proj_parameters_bias_",
    ),
    (
        [128, 256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_k_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 128],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_out_proj_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_q_proj_parameters_bias_",
    ),
    (
        [128, 256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_q_proj_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_v_proj_parameters_bias_",
    ),
    (
        [128, 256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_modules_v_proj_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_image_to_token_scaling",
    ),
    (
        [128],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_k_proj_parameters_bias_",
    ),
    (
        [128, 256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_k_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 128],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_out_proj_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_q_proj_parameters_bias_",
    ),
    (
        [128, 256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_q_proj_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_v_proj_parameters_bias_",
    ),
    (
        [128, 256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_modules_v_proj_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_cross_attn_token_to_image_scaling",
    ),
    (
        [256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm2_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm3_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm3_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm4_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_layer_norm4_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_mlp_modules_lin1_parameters_bias_",
    ),
    (
        [2048, 256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_mlp_modules_lin1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_mlp_modules_lin2_parameters_bias_",
    ),
    (
        [256, 2048],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_mlp_modules_lin2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_mask_decoder_modules_transformer_modules_layers_modules_1_modules_self_attn_scaling",
    ),
    ([64], "L_self_modules_mask_decoder_modules_upscale_conv1_parameters_bias_"),
    (
        [256, 64, 2, 2],
        "L_self_modules_mask_decoder_modules_upscale_conv1_parameters_weight_",
    ),
    ([32], "L_self_modules_mask_decoder_modules_upscale_conv2_parameters_bias_"),
    (
        [64, 32, 2, 2],
        "L_self_modules_mask_decoder_modules_upscale_conv2_parameters_weight_",
    ),
    ([64], "L_self_modules_mask_decoder_modules_upscale_layer_norm_parameters_bias_"),
    ([64], "L_self_modules_mask_decoder_modules_upscale_layer_norm_parameters_weight_"),
    (
        [1, 256],
        "L_self_modules_prompt_encoder_modules_no_mask_embed_parameters_weight_",
    ),
    ([2, 128], "L_self_modules_shared_image_embedding_buffers_positional_embedding_"),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_encoder_modules_layers_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_vision_encoder_modules_layers_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_vision_encoder_modules_layers_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [27, 64],
        "L_self_modules_vision_encoder_modules_layers_modules_0_modules_attn_parameters_rel_pos_h_",
    ),
    (
        [27, 64],
        "L_self_modules_vision_encoder_modules_layers_modules_0_modules_attn_parameters_rel_pos_w_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_vision_encoder_modules_layers_modules_0_modules_mlp_modules_lin1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_vision_encoder_modules_layers_modules_0_modules_mlp_modules_lin1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_0_modules_mlp_modules_lin2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_vision_encoder_modules_layers_modules_0_modules_mlp_modules_lin2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_10_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_encoder_modules_layers_modules_10_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_vision_encoder_modules_layers_modules_10_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_vision_encoder_modules_layers_modules_10_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [27, 64],
        "L_self_modules_vision_encoder_modules_layers_modules_10_modules_attn_parameters_rel_pos_h_",
    ),
    (
        [27, 64],
        "L_self_modules_vision_encoder_modules_layers_modules_10_modules_attn_parameters_rel_pos_w_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_vision_encoder_modules_layers_modules_10_modules_mlp_modules_lin1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_vision_encoder_modules_layers_modules_10_modules_mlp_modules_lin1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_10_modules_mlp_modules_lin2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_vision_encoder_modules_layers_modules_10_modules_mlp_modules_lin2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_11_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_encoder_modules_layers_modules_11_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_vision_encoder_modules_layers_modules_11_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_vision_encoder_modules_layers_modules_11_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [127, 64],
        "L_self_modules_vision_encoder_modules_layers_modules_11_modules_attn_parameters_rel_pos_h_",
    ),
    (
        [127, 64],
        "L_self_modules_vision_encoder_modules_layers_modules_11_modules_attn_parameters_rel_pos_w_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_vision_encoder_modules_layers_modules_11_modules_mlp_modules_lin1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_vision_encoder_modules_layers_modules_11_modules_mlp_modules_lin1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_11_modules_mlp_modules_lin2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_vision_encoder_modules_layers_modules_11_modules_mlp_modules_lin2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_encoder_modules_layers_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_vision_encoder_modules_layers_modules_1_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_vision_encoder_modules_layers_modules_1_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [27, 64],
        "L_self_modules_vision_encoder_modules_layers_modules_1_modules_attn_parameters_rel_pos_h_",
    ),
    (
        [27, 64],
        "L_self_modules_vision_encoder_modules_layers_modules_1_modules_attn_parameters_rel_pos_w_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_vision_encoder_modules_layers_modules_1_modules_mlp_modules_lin1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_vision_encoder_modules_layers_modules_1_modules_mlp_modules_lin1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_1_modules_mlp_modules_lin2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_vision_encoder_modules_layers_modules_1_modules_mlp_modules_lin2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_2_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_encoder_modules_layers_modules_2_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_vision_encoder_modules_layers_modules_2_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_vision_encoder_modules_layers_modules_2_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [127, 64],
        "L_self_modules_vision_encoder_modules_layers_modules_2_modules_attn_parameters_rel_pos_h_",
    ),
    (
        [127, 64],
        "L_self_modules_vision_encoder_modules_layers_modules_2_modules_attn_parameters_rel_pos_w_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_vision_encoder_modules_layers_modules_2_modules_mlp_modules_lin1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_vision_encoder_modules_layers_modules_2_modules_mlp_modules_lin1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_2_modules_mlp_modules_lin2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_vision_encoder_modules_layers_modules_2_modules_mlp_modules_lin2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_3_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_encoder_modules_layers_modules_3_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_vision_encoder_modules_layers_modules_3_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_vision_encoder_modules_layers_modules_3_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [27, 64],
        "L_self_modules_vision_encoder_modules_layers_modules_3_modules_attn_parameters_rel_pos_h_",
    ),
    (
        [27, 64],
        "L_self_modules_vision_encoder_modules_layers_modules_3_modules_attn_parameters_rel_pos_w_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_vision_encoder_modules_layers_modules_3_modules_mlp_modules_lin1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_vision_encoder_modules_layers_modules_3_modules_mlp_modules_lin1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_3_modules_mlp_modules_lin2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_vision_encoder_modules_layers_modules_3_modules_mlp_modules_lin2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_4_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_encoder_modules_layers_modules_4_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_vision_encoder_modules_layers_modules_4_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_vision_encoder_modules_layers_modules_4_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [27, 64],
        "L_self_modules_vision_encoder_modules_layers_modules_4_modules_attn_parameters_rel_pos_h_",
    ),
    (
        [27, 64],
        "L_self_modules_vision_encoder_modules_layers_modules_4_modules_attn_parameters_rel_pos_w_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_vision_encoder_modules_layers_modules_4_modules_mlp_modules_lin1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_vision_encoder_modules_layers_modules_4_modules_mlp_modules_lin1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_4_modules_mlp_modules_lin2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_vision_encoder_modules_layers_modules_4_modules_mlp_modules_lin2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_5_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_encoder_modules_layers_modules_5_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_vision_encoder_modules_layers_modules_5_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_vision_encoder_modules_layers_modules_5_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [127, 64],
        "L_self_modules_vision_encoder_modules_layers_modules_5_modules_attn_parameters_rel_pos_h_",
    ),
    (
        [127, 64],
        "L_self_modules_vision_encoder_modules_layers_modules_5_modules_attn_parameters_rel_pos_w_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_vision_encoder_modules_layers_modules_5_modules_mlp_modules_lin1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_vision_encoder_modules_layers_modules_5_modules_mlp_modules_lin1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_5_modules_mlp_modules_lin2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_vision_encoder_modules_layers_modules_5_modules_mlp_modules_lin2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_6_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_encoder_modules_layers_modules_6_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_vision_encoder_modules_layers_modules_6_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_vision_encoder_modules_layers_modules_6_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [27, 64],
        "L_self_modules_vision_encoder_modules_layers_modules_6_modules_attn_parameters_rel_pos_h_",
    ),
    (
        [27, 64],
        "L_self_modules_vision_encoder_modules_layers_modules_6_modules_attn_parameters_rel_pos_w_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_vision_encoder_modules_layers_modules_6_modules_mlp_modules_lin1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_vision_encoder_modules_layers_modules_6_modules_mlp_modules_lin1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_6_modules_mlp_modules_lin2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_vision_encoder_modules_layers_modules_6_modules_mlp_modules_lin2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_7_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_encoder_modules_layers_modules_7_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_vision_encoder_modules_layers_modules_7_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_vision_encoder_modules_layers_modules_7_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [27, 64],
        "L_self_modules_vision_encoder_modules_layers_modules_7_modules_attn_parameters_rel_pos_h_",
    ),
    (
        [27, 64],
        "L_self_modules_vision_encoder_modules_layers_modules_7_modules_attn_parameters_rel_pos_w_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_vision_encoder_modules_layers_modules_7_modules_mlp_modules_lin1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_vision_encoder_modules_layers_modules_7_modules_mlp_modules_lin1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_7_modules_mlp_modules_lin2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_vision_encoder_modules_layers_modules_7_modules_mlp_modules_lin2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_8_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_encoder_modules_layers_modules_8_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_vision_encoder_modules_layers_modules_8_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_vision_encoder_modules_layers_modules_8_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [127, 64],
        "L_self_modules_vision_encoder_modules_layers_modules_8_modules_attn_parameters_rel_pos_h_",
    ),
    (
        [127, 64],
        "L_self_modules_vision_encoder_modules_layers_modules_8_modules_attn_parameters_rel_pos_w_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_vision_encoder_modules_layers_modules_8_modules_mlp_modules_lin1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_vision_encoder_modules_layers_modules_8_modules_mlp_modules_lin1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_8_modules_mlp_modules_lin2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_vision_encoder_modules_layers_modules_8_modules_mlp_modules_lin2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_9_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_encoder_modules_layers_modules_9_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_vision_encoder_modules_layers_modules_9_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_vision_encoder_modules_layers_modules_9_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [27, 64],
        "L_self_modules_vision_encoder_modules_layers_modules_9_modules_attn_parameters_rel_pos_h_",
    ),
    (
        [27, 64],
        "L_self_modules_vision_encoder_modules_layers_modules_9_modules_attn_parameters_rel_pos_w_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_vision_encoder_modules_layers_modules_9_modules_mlp_modules_lin1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_vision_encoder_modules_layers_modules_9_modules_mlp_modules_lin1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_layers_modules_9_modules_mlp_modules_lin2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_vision_encoder_modules_layers_modules_9_modules_mlp_modules_lin2_parameters_weight_",
    ),
    (
        [256, 768, 1, 1],
        "L_self_modules_vision_encoder_modules_neck_modules_conv1_parameters_weight_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_vision_encoder_modules_neck_modules_conv2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_vision_encoder_modules_neck_modules_layer_norm1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_vision_encoder_modules_neck_modules_layer_norm1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_vision_encoder_modules_neck_modules_layer_norm2_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_vision_encoder_modules_neck_modules_layer_norm2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_encoder_modules_patch_embed_modules_projection_parameters_bias_",
    ),
    (
        [768, 3, 16, 16],
        "L_self_modules_vision_encoder_modules_patch_embed_modules_projection_parameters_weight_",
    ),
    ([1, 64, 64, 768], "L_self_modules_vision_encoder_parameters_pos_embed_"),
]
