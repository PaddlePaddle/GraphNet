from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 512}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, 3, S0, S0], "L_inputs_"),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_attn_parameters_in_proj_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_attn_parameters_in_proj_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_0_modules_ln1_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_0_modules_ln1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_0_modules_ln2_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_0_modules_ln2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_attn_parameters_in_proj_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_attn_parameters_in_proj_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_10_modules_ln1_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_10_modules_ln1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_10_modules_ln2_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_10_modules_ln2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_attn_parameters_in_proj_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_attn_parameters_in_proj_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_11_modules_ln1_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_11_modules_ln1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_11_modules_ln2_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_11_modules_ln2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_attn_parameters_in_proj_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_attn_parameters_in_proj_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_1_modules_ln1_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_1_modules_ln1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_1_modules_ln2_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_1_modules_ln2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_attn_parameters_in_proj_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_attn_parameters_in_proj_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_2_modules_ln1_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_2_modules_ln1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_2_modules_ln2_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_2_modules_ln2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_attn_parameters_in_proj_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_attn_parameters_in_proj_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_3_modules_ln1_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_3_modules_ln1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_3_modules_ln2_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_3_modules_ln2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_attn_parameters_in_proj_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_attn_parameters_in_proj_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_4_modules_ln1_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_4_modules_ln1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_4_modules_ln2_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_4_modules_ln2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_attn_parameters_in_proj_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_attn_parameters_in_proj_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_5_modules_ln1_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_5_modules_ln1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_5_modules_ln2_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_5_modules_ln2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_attn_parameters_in_proj_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_attn_parameters_in_proj_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_6_modules_ln1_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_6_modules_ln1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_6_modules_ln2_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_6_modules_ln2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_attn_parameters_in_proj_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_attn_parameters_in_proj_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_7_modules_ln1_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_7_modules_ln1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_7_modules_ln2_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_7_modules_ln2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_attn_parameters_in_proj_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_attn_parameters_in_proj_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_8_modules_ln1_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_8_modules_ln1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_8_modules_ln2_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_8_modules_ln2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_attn_parameters_in_proj_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_attn_parameters_in_proj_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_9_modules_ln1_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_9_modules_ln1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_9_modules_ln2_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_backbone_modules_layers_modules_9_modules_ln2_parameters_weight_",
    ),
    ([192], "L_self_modules_backbone_modules_ln1_parameters_bias_"),
    ([192], "L_self_modules_backbone_modules_ln1_parameters_weight_"),
    (
        [192, 3, 16, 16],
        "L_self_modules_backbone_modules_patch_embed_modules_projection_parameters_weight_",
    ),
    ([1, 1, 192], "L_self_modules_backbone_parameters_cls_token_"),
    ([1, 1025, 192], "L_self_modules_backbone_parameters_pos_embed_"),
    ([192, 192], "L_self_modules_decode_head_modules_classes_proj_parameters_weight_"),
    ([192], "L_self_modules_decode_head_modules_dec_proj_parameters_bias_"),
    ([192, 192], "L_self_modules_decode_head_modules_dec_proj_parameters_weight_"),
    ([192], "L_self_modules_decode_head_modules_decoder_norm_parameters_bias_"),
    ([192], "L_self_modules_decode_head_modules_decoder_norm_parameters_weight_"),
    (
        [192],
        "L_self_modules_decode_head_modules_layers_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_decode_head_modules_layers_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_decode_head_modules_layers_modules_0_modules_attn_modules_attn_parameters_in_proj_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_decode_head_modules_layers_modules_0_modules_attn_modules_attn_parameters_in_proj_weight_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_decode_head_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_decode_head_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_decode_head_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_decode_head_modules_layers_modules_0_modules_ln1_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_decode_head_modules_layers_modules_0_modules_ln1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_decode_head_modules_layers_modules_0_modules_ln2_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_decode_head_modules_layers_modules_0_modules_ln2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_decode_head_modules_layers_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_decode_head_modules_layers_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_decode_head_modules_layers_modules_1_modules_attn_modules_attn_parameters_in_proj_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_decode_head_modules_layers_modules_1_modules_attn_modules_attn_parameters_in_proj_weight_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_layers_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_decode_head_modules_layers_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_decode_head_modules_layers_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_decode_head_modules_layers_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_decode_head_modules_layers_modules_1_modules_ln1_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_decode_head_modules_layers_modules_1_modules_ln1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_decode_head_modules_layers_modules_1_modules_ln2_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_decode_head_modules_layers_modules_1_modules_ln2_parameters_weight_",
    ),
    ([150], "L_self_modules_decode_head_modules_mask_norm_parameters_bias_"),
    ([150], "L_self_modules_decode_head_modules_mask_norm_parameters_weight_"),
    ([192, 192], "L_self_modules_decode_head_modules_patch_proj_parameters_weight_"),
    ([1, 150, 192], "L_self_modules_decode_head_parameters_cls_emb_"),
]
