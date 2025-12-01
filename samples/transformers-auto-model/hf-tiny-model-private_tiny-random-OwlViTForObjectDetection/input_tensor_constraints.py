dynamic_dim_constraint_symbols = []

dynamic_dim_constraint_symbol2example_value = {}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, 10], "L_attention_mask_"),
    ([1, 10], "L_input_ids_"),
    ([1, 3, 32, 32], "L_pixel_values_"),
    ([256, 4], "L_self_box_bias"),
    ([32], "L_self_modules_box_head_modules_dense0_parameters_bias_"),
    ([32, 32], "L_self_modules_box_head_modules_dense0_parameters_weight_"),
    ([32], "L_self_modules_box_head_modules_dense1_parameters_bias_"),
    ([32, 32], "L_self_modules_box_head_modules_dense1_parameters_weight_"),
    ([4], "L_self_modules_box_head_modules_dense2_parameters_bias_"),
    ([4, 32], "L_self_modules_box_head_modules_dense2_parameters_weight_"),
    ([64], "L_self_modules_class_head_modules_dense0_parameters_bias_"),
    ([64, 32], "L_self_modules_class_head_modules_dense0_parameters_weight_"),
    ([1], "L_self_modules_class_head_modules_logit_scale_parameters_bias_"),
    ([1, 32], "L_self_modules_class_head_modules_logit_scale_parameters_weight_"),
    ([1], "L_self_modules_class_head_modules_logit_shift_parameters_bias_"),
    ([1, 32], "L_self_modules_class_head_modules_logit_shift_parameters_weight_"),
    ([32], "L_self_modules_layer_norm_parameters_bias_"),
    ([32], "L_self_modules_layer_norm_parameters_weight_"),
    (
        [1, 16],
        "L_self_modules_owlvit_modules_text_model_modules_embeddings_buffers_position_ids_",
    ),
    (
        [16, 64],
        "L_self_modules_owlvit_modules_text_model_modules_embeddings_modules_position_embedding_parameters_weight_",
    ),
    (
        [1024, 64],
        "L_self_modules_owlvit_modules_text_model_modules_embeddings_modules_token_embedding_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_weight_",
    ),
    (
        [37],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [37, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [64, 37],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_weight_",
    ),
    (
        [37],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [37, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [64, 37],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_weight_",
    ),
    (
        [37],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [37, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [64, 37],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_weight_",
    ),
    (
        [37],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [37, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [64, 37],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_weight_",
    ),
    (
        [37],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [37, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [64, 37],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_weight_",
    ),
    (
        [37],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [37, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [64, 37],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_weight_",
    ),
    (
        [37],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [37, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [64, 37],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_weight_",
    ),
    (
        [37],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [37, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [64, 37],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_weight_",
    ),
    (
        [37],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [37, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [64, 37],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_weight_",
    ),
    (
        [37],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [37, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [64, 37],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_weight_",
    ),
    (
        [37],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [37, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [64, 37],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_weight_",
    ),
    (
        [37],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [37, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [64, 37],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_owlvit_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_owlvit_modules_text_model_modules_final_layer_norm_parameters_weight_",
    ),
    ([64, 64], "L_self_modules_owlvit_modules_text_projection_parameters_weight_"),
    (
        [1, 257],
        "L_self_modules_owlvit_modules_vision_model_modules_embeddings_buffers_position_ids_",
    ),
    (
        [32, 3, 2, 2],
        "L_self_modules_owlvit_modules_vision_model_modules_embeddings_modules_patch_embedding_parameters_weight_",
    ),
    (
        [257, 32],
        "L_self_modules_owlvit_modules_vision_model_modules_embeddings_modules_position_embedding_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_embeddings_parameters_class_embedding_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_weight_",
    ),
    (
        [37],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [37, 32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [32, 37],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_weight_",
    ),
    (
        [37],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [37, 32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [32, 37],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_weight_",
    ),
    (
        [37],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [37, 32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [32, 37],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_weight_",
    ),
    (
        [37],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [37, 32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [32, 37],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_weight_",
    ),
    (
        [37],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [37, 32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [32, 37],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_owlvit_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_post_layernorm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_post_layernorm_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_pre_layernorm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_owlvit_modules_vision_model_modules_pre_layernorm_parameters_weight_",
    ),
    ([64, 32], "L_self_modules_owlvit_modules_visual_projection_parameters_weight_"),
    ([], "L_self_modules_owlvit_parameters_logit_scale_"),
]
