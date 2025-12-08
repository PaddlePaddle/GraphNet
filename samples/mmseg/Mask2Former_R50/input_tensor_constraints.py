from sympy import Symbol, Expr, Rel, Eq


dynamic_dim_constraint_symbols = []

dynamic_dim_constraint_symbol2example_value = {}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([151], "L_self_modules_cls_embed_parameters_bias_"),
    ([151, 256], "L_self_modules_cls_embed_parameters_weight_"),
    ([3, 256], "L_self_modules_level_embed_parameters_weight_"),
    ([256], "L_self_modules_mask_embed_modules_0_parameters_bias_"),
    ([256, 256], "L_self_modules_mask_embed_modules_0_parameters_weight_"),
    ([256], "L_self_modules_mask_embed_modules_2_parameters_bias_"),
    ([256, 256], "L_self_modules_mask_embed_modules_2_parameters_weight_"),
    ([256], "L_self_modules_mask_embed_modules_4_parameters_bias_"),
    ([256, 256], "L_self_modules_mask_embed_modules_4_parameters_weight_"),
    ([100, 256], "L_self_modules_query_embed_parameters_weight_"),
    ([100, 256], "L_self_modules_query_feat_parameters_weight_"),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_0_modules_cross_attn_modules_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_decoder_modules_layers_modules_0_modules_cross_attn_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_transformer_decoder_modules_layers_modules_0_modules_cross_attn_modules_attn_parameters_in_proj_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_transformer_decoder_modules_layers_modules_0_modules_cross_attn_modules_attn_parameters_in_proj_weight_",
    ),
    (
        [2048],
        "L_self_modules_transformer_decoder_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [2048, 256],
        "L_self_modules_transformer_decoder_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [256, 2048],
        "L_self_modules_transformer_decoder_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_0_modules_norms_modules_0_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_0_modules_norms_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_0_modules_norms_modules_1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_0_modules_norms_modules_1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_0_modules_norms_modules_2_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_0_modules_norms_modules_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_0_modules_self_attn_modules_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_decoder_modules_layers_modules_0_modules_self_attn_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_transformer_decoder_modules_layers_modules_0_modules_self_attn_modules_attn_parameters_in_proj_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_transformer_decoder_modules_layers_modules_0_modules_self_attn_modules_attn_parameters_in_proj_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_1_modules_cross_attn_modules_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_decoder_modules_layers_modules_1_modules_cross_attn_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_transformer_decoder_modules_layers_modules_1_modules_cross_attn_modules_attn_parameters_in_proj_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_transformer_decoder_modules_layers_modules_1_modules_cross_attn_modules_attn_parameters_in_proj_weight_",
    ),
    (
        [2048],
        "L_self_modules_transformer_decoder_modules_layers_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [2048, 256],
        "L_self_modules_transformer_decoder_modules_layers_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [256, 2048],
        "L_self_modules_transformer_decoder_modules_layers_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_1_modules_norms_modules_0_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_1_modules_norms_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_1_modules_norms_modules_1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_1_modules_norms_modules_1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_1_modules_norms_modules_2_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_1_modules_norms_modules_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_1_modules_self_attn_modules_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_decoder_modules_layers_modules_1_modules_self_attn_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_transformer_decoder_modules_layers_modules_1_modules_self_attn_modules_attn_parameters_in_proj_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_transformer_decoder_modules_layers_modules_1_modules_self_attn_modules_attn_parameters_in_proj_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_2_modules_cross_attn_modules_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_decoder_modules_layers_modules_2_modules_cross_attn_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_transformer_decoder_modules_layers_modules_2_modules_cross_attn_modules_attn_parameters_in_proj_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_transformer_decoder_modules_layers_modules_2_modules_cross_attn_modules_attn_parameters_in_proj_weight_",
    ),
    (
        [2048],
        "L_self_modules_transformer_decoder_modules_layers_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [2048, 256],
        "L_self_modules_transformer_decoder_modules_layers_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [256, 2048],
        "L_self_modules_transformer_decoder_modules_layers_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_2_modules_norms_modules_0_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_2_modules_norms_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_2_modules_norms_modules_1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_2_modules_norms_modules_1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_2_modules_norms_modules_2_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_2_modules_norms_modules_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_2_modules_self_attn_modules_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_decoder_modules_layers_modules_2_modules_self_attn_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_transformer_decoder_modules_layers_modules_2_modules_self_attn_modules_attn_parameters_in_proj_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_transformer_decoder_modules_layers_modules_2_modules_self_attn_modules_attn_parameters_in_proj_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_3_modules_cross_attn_modules_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_decoder_modules_layers_modules_3_modules_cross_attn_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_transformer_decoder_modules_layers_modules_3_modules_cross_attn_modules_attn_parameters_in_proj_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_transformer_decoder_modules_layers_modules_3_modules_cross_attn_modules_attn_parameters_in_proj_weight_",
    ),
    (
        [2048],
        "L_self_modules_transformer_decoder_modules_layers_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [2048, 256],
        "L_self_modules_transformer_decoder_modules_layers_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_3_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [256, 2048],
        "L_self_modules_transformer_decoder_modules_layers_modules_3_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_3_modules_norms_modules_0_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_3_modules_norms_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_3_modules_norms_modules_1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_3_modules_norms_modules_1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_3_modules_norms_modules_2_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_3_modules_norms_modules_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_3_modules_self_attn_modules_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_decoder_modules_layers_modules_3_modules_self_attn_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_transformer_decoder_modules_layers_modules_3_modules_self_attn_modules_attn_parameters_in_proj_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_transformer_decoder_modules_layers_modules_3_modules_self_attn_modules_attn_parameters_in_proj_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_4_modules_cross_attn_modules_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_decoder_modules_layers_modules_4_modules_cross_attn_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_transformer_decoder_modules_layers_modules_4_modules_cross_attn_modules_attn_parameters_in_proj_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_transformer_decoder_modules_layers_modules_4_modules_cross_attn_modules_attn_parameters_in_proj_weight_",
    ),
    (
        [2048],
        "L_self_modules_transformer_decoder_modules_layers_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [2048, 256],
        "L_self_modules_transformer_decoder_modules_layers_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_4_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [256, 2048],
        "L_self_modules_transformer_decoder_modules_layers_modules_4_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_4_modules_norms_modules_0_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_4_modules_norms_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_4_modules_norms_modules_1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_4_modules_norms_modules_1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_4_modules_norms_modules_2_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_4_modules_norms_modules_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_4_modules_self_attn_modules_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_decoder_modules_layers_modules_4_modules_self_attn_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_transformer_decoder_modules_layers_modules_4_modules_self_attn_modules_attn_parameters_in_proj_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_transformer_decoder_modules_layers_modules_4_modules_self_attn_modules_attn_parameters_in_proj_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_5_modules_cross_attn_modules_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_decoder_modules_layers_modules_5_modules_cross_attn_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_transformer_decoder_modules_layers_modules_5_modules_cross_attn_modules_attn_parameters_in_proj_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_transformer_decoder_modules_layers_modules_5_modules_cross_attn_modules_attn_parameters_in_proj_weight_",
    ),
    (
        [2048],
        "L_self_modules_transformer_decoder_modules_layers_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [2048, 256],
        "L_self_modules_transformer_decoder_modules_layers_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_5_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [256, 2048],
        "L_self_modules_transformer_decoder_modules_layers_modules_5_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_5_modules_norms_modules_0_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_5_modules_norms_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_5_modules_norms_modules_1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_5_modules_norms_modules_1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_5_modules_norms_modules_2_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_5_modules_norms_modules_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_5_modules_self_attn_modules_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_decoder_modules_layers_modules_5_modules_self_attn_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_transformer_decoder_modules_layers_modules_5_modules_self_attn_modules_attn_parameters_in_proj_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_transformer_decoder_modules_layers_modules_5_modules_self_attn_modules_attn_parameters_in_proj_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_6_modules_cross_attn_modules_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_decoder_modules_layers_modules_6_modules_cross_attn_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_transformer_decoder_modules_layers_modules_6_modules_cross_attn_modules_attn_parameters_in_proj_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_transformer_decoder_modules_layers_modules_6_modules_cross_attn_modules_attn_parameters_in_proj_weight_",
    ),
    (
        [2048],
        "L_self_modules_transformer_decoder_modules_layers_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [2048, 256],
        "L_self_modules_transformer_decoder_modules_layers_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_6_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [256, 2048],
        "L_self_modules_transformer_decoder_modules_layers_modules_6_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_6_modules_norms_modules_0_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_6_modules_norms_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_6_modules_norms_modules_1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_6_modules_norms_modules_1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_6_modules_norms_modules_2_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_6_modules_norms_modules_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_6_modules_self_attn_modules_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_decoder_modules_layers_modules_6_modules_self_attn_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_transformer_decoder_modules_layers_modules_6_modules_self_attn_modules_attn_parameters_in_proj_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_transformer_decoder_modules_layers_modules_6_modules_self_attn_modules_attn_parameters_in_proj_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_7_modules_cross_attn_modules_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_decoder_modules_layers_modules_7_modules_cross_attn_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_transformer_decoder_modules_layers_modules_7_modules_cross_attn_modules_attn_parameters_in_proj_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_transformer_decoder_modules_layers_modules_7_modules_cross_attn_modules_attn_parameters_in_proj_weight_",
    ),
    (
        [2048],
        "L_self_modules_transformer_decoder_modules_layers_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [2048, 256],
        "L_self_modules_transformer_decoder_modules_layers_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_7_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [256, 2048],
        "L_self_modules_transformer_decoder_modules_layers_modules_7_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_7_modules_norms_modules_0_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_7_modules_norms_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_7_modules_norms_modules_1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_7_modules_norms_modules_1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_7_modules_norms_modules_2_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_7_modules_norms_modules_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_7_modules_self_attn_modules_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_decoder_modules_layers_modules_7_modules_self_attn_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_transformer_decoder_modules_layers_modules_7_modules_self_attn_modules_attn_parameters_in_proj_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_transformer_decoder_modules_layers_modules_7_modules_self_attn_modules_attn_parameters_in_proj_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_8_modules_cross_attn_modules_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_decoder_modules_layers_modules_8_modules_cross_attn_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_transformer_decoder_modules_layers_modules_8_modules_cross_attn_modules_attn_parameters_in_proj_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_transformer_decoder_modules_layers_modules_8_modules_cross_attn_modules_attn_parameters_in_proj_weight_",
    ),
    (
        [2048],
        "L_self_modules_transformer_decoder_modules_layers_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [2048, 256],
        "L_self_modules_transformer_decoder_modules_layers_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_8_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [256, 2048],
        "L_self_modules_transformer_decoder_modules_layers_modules_8_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_8_modules_norms_modules_0_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_8_modules_norms_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_8_modules_norms_modules_1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_8_modules_norms_modules_1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_8_modules_norms_modules_2_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_8_modules_norms_modules_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_decoder_modules_layers_modules_8_modules_self_attn_modules_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_decoder_modules_layers_modules_8_modules_self_attn_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_transformer_decoder_modules_layers_modules_8_modules_self_attn_modules_attn_parameters_in_proj_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_transformer_decoder_modules_layers_modules_8_modules_self_attn_modules_attn_parameters_in_proj_weight_",
    ),
    ([256], "L_self_modules_transformer_decoder_modules_post_norm_parameters_bias_"),
    ([256], "L_self_modules_transformer_decoder_modules_post_norm_parameters_weight_"),
    ([1, 256, 128, 128], "L_stack0_0_"),
    ([1, 256, 16, 16], "L_stack0_1_0_"),
    ([1, 256, 32, 32], "L_stack0_1_1_"),
    ([1, 256, 64, 64], "L_stack0_1_2_"),
]
