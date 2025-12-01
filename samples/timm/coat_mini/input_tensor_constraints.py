from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1], "L_self_modules_aggregate_parameters_bias_"),
    ([1, 3, 1], "L_self_modules_aggregate_parameters_weight_"),
    ([1000], "L_self_modules_head_parameters_bias_"),
    ([1000, 216], "L_self_modules_head_parameters_weight_"),
    ([216], "L_self_modules_norm2_parameters_bias_"),
    ([216], "L_self_modules_norm2_parameters_weight_"),
    ([216], "L_self_modules_norm3_parameters_bias_"),
    ([216], "L_self_modules_norm3_parameters_weight_"),
    ([216], "L_self_modules_norm4_parameters_bias_"),
    ([216], "L_self_modules_norm4_parameters_weight_"),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe2_modules_proj_parameters_bias_",
    ),
    (
        [216, 216],
        "L_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe2_modules_proj_parameters_weight_",
    ),
    (
        [648],
        "L_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe2_modules_qkv_parameters_bias_",
    ),
    (
        [648, 216],
        "L_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe2_modules_qkv_parameters_weight_",
    ),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe3_modules_proj_parameters_bias_",
    ),
    (
        [216, 216],
        "L_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe3_modules_proj_parameters_weight_",
    ),
    (
        [648],
        "L_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe3_modules_qkv_parameters_bias_",
    ),
    (
        [648, 216],
        "L_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe3_modules_qkv_parameters_weight_",
    ),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe4_modules_proj_parameters_bias_",
    ),
    (
        [216, 216],
        "L_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe4_modules_proj_parameters_weight_",
    ),
    (
        [648],
        "L_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe4_modules_qkv_parameters_bias_",
    ),
    (
        [648, 216],
        "L_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe4_modules_qkv_parameters_weight_",
    ),
    (
        [864],
        "L_self_modules_parallel_blocks_modules_0_modules_mlp2_modules_fc1_parameters_bias_",
    ),
    (
        [864, 216],
        "L_self_modules_parallel_blocks_modules_0_modules_mlp2_modules_fc1_parameters_weight_",
    ),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_0_modules_mlp2_modules_fc2_parameters_bias_",
    ),
    (
        [216, 864],
        "L_self_modules_parallel_blocks_modules_0_modules_mlp2_modules_fc2_parameters_weight_",
    ),
    ([216], "L_self_modules_parallel_blocks_modules_0_modules_norm12_parameters_bias_"),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_0_modules_norm12_parameters_weight_",
    ),
    ([216], "L_self_modules_parallel_blocks_modules_0_modules_norm13_parameters_bias_"),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_0_modules_norm13_parameters_weight_",
    ),
    ([216], "L_self_modules_parallel_blocks_modules_0_modules_norm14_parameters_bias_"),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_0_modules_norm14_parameters_weight_",
    ),
    ([216], "L_self_modules_parallel_blocks_modules_0_modules_norm22_parameters_bias_"),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_0_modules_norm22_parameters_weight_",
    ),
    ([216], "L_self_modules_parallel_blocks_modules_0_modules_norm23_parameters_bias_"),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_0_modules_norm23_parameters_weight_",
    ),
    ([216], "L_self_modules_parallel_blocks_modules_0_modules_norm24_parameters_bias_"),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_0_modules_norm24_parameters_weight_",
    ),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe2_modules_proj_parameters_bias_",
    ),
    (
        [216, 216],
        "L_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe2_modules_proj_parameters_weight_",
    ),
    (
        [648],
        "L_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe2_modules_qkv_parameters_bias_",
    ),
    (
        [648, 216],
        "L_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe2_modules_qkv_parameters_weight_",
    ),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe3_modules_proj_parameters_bias_",
    ),
    (
        [216, 216],
        "L_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe3_modules_proj_parameters_weight_",
    ),
    (
        [648],
        "L_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe3_modules_qkv_parameters_bias_",
    ),
    (
        [648, 216],
        "L_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe3_modules_qkv_parameters_weight_",
    ),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe4_modules_proj_parameters_bias_",
    ),
    (
        [216, 216],
        "L_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe4_modules_proj_parameters_weight_",
    ),
    (
        [648],
        "L_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe4_modules_qkv_parameters_bias_",
    ),
    (
        [648, 216],
        "L_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe4_modules_qkv_parameters_weight_",
    ),
    (
        [864],
        "L_self_modules_parallel_blocks_modules_1_modules_mlp2_modules_fc1_parameters_bias_",
    ),
    (
        [864, 216],
        "L_self_modules_parallel_blocks_modules_1_modules_mlp2_modules_fc1_parameters_weight_",
    ),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_1_modules_mlp2_modules_fc2_parameters_bias_",
    ),
    (
        [216, 864],
        "L_self_modules_parallel_blocks_modules_1_modules_mlp2_modules_fc2_parameters_weight_",
    ),
    ([216], "L_self_modules_parallel_blocks_modules_1_modules_norm12_parameters_bias_"),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_1_modules_norm12_parameters_weight_",
    ),
    ([216], "L_self_modules_parallel_blocks_modules_1_modules_norm13_parameters_bias_"),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_1_modules_norm13_parameters_weight_",
    ),
    ([216], "L_self_modules_parallel_blocks_modules_1_modules_norm14_parameters_bias_"),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_1_modules_norm14_parameters_weight_",
    ),
    ([216], "L_self_modules_parallel_blocks_modules_1_modules_norm22_parameters_bias_"),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_1_modules_norm22_parameters_weight_",
    ),
    ([216], "L_self_modules_parallel_blocks_modules_1_modules_norm23_parameters_bias_"),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_1_modules_norm23_parameters_weight_",
    ),
    ([216], "L_self_modules_parallel_blocks_modules_1_modules_norm24_parameters_bias_"),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_1_modules_norm24_parameters_weight_",
    ),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe2_modules_proj_parameters_bias_",
    ),
    (
        [216, 216],
        "L_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe2_modules_proj_parameters_weight_",
    ),
    (
        [648],
        "L_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe2_modules_qkv_parameters_bias_",
    ),
    (
        [648, 216],
        "L_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe2_modules_qkv_parameters_weight_",
    ),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe3_modules_proj_parameters_bias_",
    ),
    (
        [216, 216],
        "L_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe3_modules_proj_parameters_weight_",
    ),
    (
        [648],
        "L_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe3_modules_qkv_parameters_bias_",
    ),
    (
        [648, 216],
        "L_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe3_modules_qkv_parameters_weight_",
    ),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe4_modules_proj_parameters_bias_",
    ),
    (
        [216, 216],
        "L_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe4_modules_proj_parameters_weight_",
    ),
    (
        [648],
        "L_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe4_modules_qkv_parameters_bias_",
    ),
    (
        [648, 216],
        "L_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe4_modules_qkv_parameters_weight_",
    ),
    (
        [864],
        "L_self_modules_parallel_blocks_modules_2_modules_mlp2_modules_fc1_parameters_bias_",
    ),
    (
        [864, 216],
        "L_self_modules_parallel_blocks_modules_2_modules_mlp2_modules_fc1_parameters_weight_",
    ),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_2_modules_mlp2_modules_fc2_parameters_bias_",
    ),
    (
        [216, 864],
        "L_self_modules_parallel_blocks_modules_2_modules_mlp2_modules_fc2_parameters_weight_",
    ),
    ([216], "L_self_modules_parallel_blocks_modules_2_modules_norm12_parameters_bias_"),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_2_modules_norm12_parameters_weight_",
    ),
    ([216], "L_self_modules_parallel_blocks_modules_2_modules_norm13_parameters_bias_"),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_2_modules_norm13_parameters_weight_",
    ),
    ([216], "L_self_modules_parallel_blocks_modules_2_modules_norm14_parameters_bias_"),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_2_modules_norm14_parameters_weight_",
    ),
    ([216], "L_self_modules_parallel_blocks_modules_2_modules_norm22_parameters_bias_"),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_2_modules_norm22_parameters_weight_",
    ),
    ([216], "L_self_modules_parallel_blocks_modules_2_modules_norm23_parameters_bias_"),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_2_modules_norm23_parameters_weight_",
    ),
    ([216], "L_self_modules_parallel_blocks_modules_2_modules_norm24_parameters_bias_"),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_2_modules_norm24_parameters_weight_",
    ),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe2_modules_proj_parameters_bias_",
    ),
    (
        [216, 216],
        "L_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe2_modules_proj_parameters_weight_",
    ),
    (
        [648],
        "L_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe2_modules_qkv_parameters_bias_",
    ),
    (
        [648, 216],
        "L_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe2_modules_qkv_parameters_weight_",
    ),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe3_modules_proj_parameters_bias_",
    ),
    (
        [216, 216],
        "L_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe3_modules_proj_parameters_weight_",
    ),
    (
        [648],
        "L_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe3_modules_qkv_parameters_bias_",
    ),
    (
        [648, 216],
        "L_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe3_modules_qkv_parameters_weight_",
    ),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe4_modules_proj_parameters_bias_",
    ),
    (
        [216, 216],
        "L_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe4_modules_proj_parameters_weight_",
    ),
    (
        [648],
        "L_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe4_modules_qkv_parameters_bias_",
    ),
    (
        [648, 216],
        "L_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe4_modules_qkv_parameters_weight_",
    ),
    (
        [864],
        "L_self_modules_parallel_blocks_modules_3_modules_mlp2_modules_fc1_parameters_bias_",
    ),
    (
        [864, 216],
        "L_self_modules_parallel_blocks_modules_3_modules_mlp2_modules_fc1_parameters_weight_",
    ),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_3_modules_mlp2_modules_fc2_parameters_bias_",
    ),
    (
        [216, 864],
        "L_self_modules_parallel_blocks_modules_3_modules_mlp2_modules_fc2_parameters_weight_",
    ),
    ([216], "L_self_modules_parallel_blocks_modules_3_modules_norm12_parameters_bias_"),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_3_modules_norm12_parameters_weight_",
    ),
    ([216], "L_self_modules_parallel_blocks_modules_3_modules_norm13_parameters_bias_"),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_3_modules_norm13_parameters_weight_",
    ),
    ([216], "L_self_modules_parallel_blocks_modules_3_modules_norm14_parameters_bias_"),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_3_modules_norm14_parameters_weight_",
    ),
    ([216], "L_self_modules_parallel_blocks_modules_3_modules_norm22_parameters_bias_"),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_3_modules_norm22_parameters_weight_",
    ),
    ([216], "L_self_modules_parallel_blocks_modules_3_modules_norm23_parameters_bias_"),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_3_modules_norm23_parameters_weight_",
    ),
    ([216], "L_self_modules_parallel_blocks_modules_3_modules_norm24_parameters_bias_"),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_3_modules_norm24_parameters_weight_",
    ),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe2_modules_proj_parameters_bias_",
    ),
    (
        [216, 216],
        "L_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe2_modules_proj_parameters_weight_",
    ),
    (
        [648],
        "L_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe2_modules_qkv_parameters_bias_",
    ),
    (
        [648, 216],
        "L_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe2_modules_qkv_parameters_weight_",
    ),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe3_modules_proj_parameters_bias_",
    ),
    (
        [216, 216],
        "L_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe3_modules_proj_parameters_weight_",
    ),
    (
        [648],
        "L_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe3_modules_qkv_parameters_bias_",
    ),
    (
        [648, 216],
        "L_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe3_modules_qkv_parameters_weight_",
    ),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe4_modules_proj_parameters_bias_",
    ),
    (
        [216, 216],
        "L_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe4_modules_proj_parameters_weight_",
    ),
    (
        [648],
        "L_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe4_modules_qkv_parameters_bias_",
    ),
    (
        [648, 216],
        "L_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe4_modules_qkv_parameters_weight_",
    ),
    (
        [864],
        "L_self_modules_parallel_blocks_modules_4_modules_mlp2_modules_fc1_parameters_bias_",
    ),
    (
        [864, 216],
        "L_self_modules_parallel_blocks_modules_4_modules_mlp2_modules_fc1_parameters_weight_",
    ),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_4_modules_mlp2_modules_fc2_parameters_bias_",
    ),
    (
        [216, 864],
        "L_self_modules_parallel_blocks_modules_4_modules_mlp2_modules_fc2_parameters_weight_",
    ),
    ([216], "L_self_modules_parallel_blocks_modules_4_modules_norm12_parameters_bias_"),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_4_modules_norm12_parameters_weight_",
    ),
    ([216], "L_self_modules_parallel_blocks_modules_4_modules_norm13_parameters_bias_"),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_4_modules_norm13_parameters_weight_",
    ),
    ([216], "L_self_modules_parallel_blocks_modules_4_modules_norm14_parameters_bias_"),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_4_modules_norm14_parameters_weight_",
    ),
    ([216], "L_self_modules_parallel_blocks_modules_4_modules_norm22_parameters_bias_"),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_4_modules_norm22_parameters_weight_",
    ),
    ([216], "L_self_modules_parallel_blocks_modules_4_modules_norm23_parameters_bias_"),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_4_modules_norm23_parameters_weight_",
    ),
    ([216], "L_self_modules_parallel_blocks_modules_4_modules_norm24_parameters_bias_"),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_4_modules_norm24_parameters_weight_",
    ),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe2_modules_proj_parameters_bias_",
    ),
    (
        [216, 216],
        "L_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe2_modules_proj_parameters_weight_",
    ),
    (
        [648],
        "L_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe2_modules_qkv_parameters_bias_",
    ),
    (
        [648, 216],
        "L_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe2_modules_qkv_parameters_weight_",
    ),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe3_modules_proj_parameters_bias_",
    ),
    (
        [216, 216],
        "L_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe3_modules_proj_parameters_weight_",
    ),
    (
        [648],
        "L_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe3_modules_qkv_parameters_bias_",
    ),
    (
        [648, 216],
        "L_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe3_modules_qkv_parameters_weight_",
    ),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe4_modules_proj_parameters_bias_",
    ),
    (
        [216, 216],
        "L_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe4_modules_proj_parameters_weight_",
    ),
    (
        [648],
        "L_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe4_modules_qkv_parameters_bias_",
    ),
    (
        [648, 216],
        "L_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe4_modules_qkv_parameters_weight_",
    ),
    (
        [864],
        "L_self_modules_parallel_blocks_modules_5_modules_mlp2_modules_fc1_parameters_bias_",
    ),
    (
        [864, 216],
        "L_self_modules_parallel_blocks_modules_5_modules_mlp2_modules_fc1_parameters_weight_",
    ),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_5_modules_mlp2_modules_fc2_parameters_bias_",
    ),
    (
        [216, 864],
        "L_self_modules_parallel_blocks_modules_5_modules_mlp2_modules_fc2_parameters_weight_",
    ),
    ([216], "L_self_modules_parallel_blocks_modules_5_modules_norm12_parameters_bias_"),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_5_modules_norm12_parameters_weight_",
    ),
    ([216], "L_self_modules_parallel_blocks_modules_5_modules_norm13_parameters_bias_"),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_5_modules_norm13_parameters_weight_",
    ),
    ([216], "L_self_modules_parallel_blocks_modules_5_modules_norm14_parameters_bias_"),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_5_modules_norm14_parameters_weight_",
    ),
    ([216], "L_self_modules_parallel_blocks_modules_5_modules_norm22_parameters_bias_"),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_5_modules_norm22_parameters_weight_",
    ),
    ([216], "L_self_modules_parallel_blocks_modules_5_modules_norm23_parameters_bias_"),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_5_modules_norm23_parameters_weight_",
    ),
    ([216], "L_self_modules_parallel_blocks_modules_5_modules_norm24_parameters_bias_"),
    (
        [216],
        "L_self_modules_parallel_blocks_modules_5_modules_norm24_parameters_weight_",
    ),
    ([152], "L_self_modules_patch_embed1_modules_norm_parameters_bias_"),
    ([152], "L_self_modules_patch_embed1_modules_norm_parameters_weight_"),
    ([152], "L_self_modules_patch_embed1_modules_proj_parameters_bias_"),
    ([152, 3, 4, 4], "L_self_modules_patch_embed1_modules_proj_parameters_weight_"),
    ([216], "L_self_modules_patch_embed2_modules_norm_parameters_bias_"),
    ([216], "L_self_modules_patch_embed2_modules_norm_parameters_weight_"),
    ([216], "L_self_modules_patch_embed2_modules_proj_parameters_bias_"),
    ([216, 152, 2, 2], "L_self_modules_patch_embed2_modules_proj_parameters_weight_"),
    ([216], "L_self_modules_patch_embed3_modules_norm_parameters_bias_"),
    ([216], "L_self_modules_patch_embed3_modules_norm_parameters_weight_"),
    ([216], "L_self_modules_patch_embed3_modules_proj_parameters_bias_"),
    ([216, 216, 2, 2], "L_self_modules_patch_embed3_modules_proj_parameters_weight_"),
    ([216], "L_self_modules_patch_embed4_modules_norm_parameters_bias_"),
    ([216], "L_self_modules_patch_embed4_modules_norm_parameters_weight_"),
    ([216], "L_self_modules_patch_embed4_modules_proj_parameters_bias_"),
    ([216, 216, 2, 2], "L_self_modules_patch_embed4_modules_proj_parameters_weight_"),
    (
        [152],
        "L_self_modules_serial_blocks1_modules_0_modules_cpe_modules_proj_parameters_bias_",
    ),
    (
        [152, 1, 3, 3],
        "L_self_modules_serial_blocks1_modules_0_modules_cpe_modules_proj_parameters_weight_",
    ),
    (
        [38],
        "L_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_",
    ),
    (
        [38, 1, 3, 3],
        "L_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_",
    ),
    (
        [57],
        "L_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_",
    ),
    (
        [57, 1, 5, 5],
        "L_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_",
    ),
    (
        [57],
        "L_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_",
    ),
    (
        [57, 1, 7, 7],
        "L_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_",
    ),
    (
        [152],
        "L_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_proj_parameters_bias_",
    ),
    (
        [152, 152],
        "L_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_proj_parameters_weight_",
    ),
    (
        [456],
        "L_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_qkv_parameters_bias_",
    ),
    (
        [456, 152],
        "L_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_qkv_parameters_weight_",
    ),
    (
        [608],
        "L_self_modules_serial_blocks1_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [608, 152],
        "L_self_modules_serial_blocks1_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [152],
        "L_self_modules_serial_blocks1_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [152, 608],
        "L_self_modules_serial_blocks1_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([152], "L_self_modules_serial_blocks1_modules_0_modules_norm1_parameters_bias_"),
    ([152], "L_self_modules_serial_blocks1_modules_0_modules_norm1_parameters_weight_"),
    ([152], "L_self_modules_serial_blocks1_modules_0_modules_norm2_parameters_bias_"),
    ([152], "L_self_modules_serial_blocks1_modules_0_modules_norm2_parameters_weight_"),
    (
        [152],
        "L_self_modules_serial_blocks1_modules_1_modules_factoratt_crpe_modules_proj_parameters_bias_",
    ),
    (
        [152, 152],
        "L_self_modules_serial_blocks1_modules_1_modules_factoratt_crpe_modules_proj_parameters_weight_",
    ),
    (
        [456],
        "L_self_modules_serial_blocks1_modules_1_modules_factoratt_crpe_modules_qkv_parameters_bias_",
    ),
    (
        [456, 152],
        "L_self_modules_serial_blocks1_modules_1_modules_factoratt_crpe_modules_qkv_parameters_weight_",
    ),
    (
        [608],
        "L_self_modules_serial_blocks1_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [608, 152],
        "L_self_modules_serial_blocks1_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [152],
        "L_self_modules_serial_blocks1_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [152, 608],
        "L_self_modules_serial_blocks1_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([152], "L_self_modules_serial_blocks1_modules_1_modules_norm1_parameters_bias_"),
    ([152], "L_self_modules_serial_blocks1_modules_1_modules_norm1_parameters_weight_"),
    ([152], "L_self_modules_serial_blocks1_modules_1_modules_norm2_parameters_bias_"),
    ([152], "L_self_modules_serial_blocks1_modules_1_modules_norm2_parameters_weight_"),
    (
        [216],
        "L_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_bias_",
    ),
    (
        [216, 1, 3, 3],
        "L_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_weight_",
    ),
    (
        [54],
        "L_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_",
    ),
    (
        [54, 1, 3, 3],
        "L_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_",
    ),
    (
        [81],
        "L_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_",
    ),
    (
        [81, 1, 5, 5],
        "L_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_",
    ),
    (
        [81],
        "L_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_",
    ),
    (
        [81, 1, 7, 7],
        "L_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_",
    ),
    (
        [216],
        "L_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_proj_parameters_bias_",
    ),
    (
        [216, 216],
        "L_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_proj_parameters_weight_",
    ),
    (
        [648],
        "L_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_qkv_parameters_bias_",
    ),
    (
        [648, 216],
        "L_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_qkv_parameters_weight_",
    ),
    (
        [864],
        "L_self_modules_serial_blocks2_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [864, 216],
        "L_self_modules_serial_blocks2_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [216],
        "L_self_modules_serial_blocks2_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [216, 864],
        "L_self_modules_serial_blocks2_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([216], "L_self_modules_serial_blocks2_modules_0_modules_norm1_parameters_bias_"),
    ([216], "L_self_modules_serial_blocks2_modules_0_modules_norm1_parameters_weight_"),
    ([216], "L_self_modules_serial_blocks2_modules_0_modules_norm2_parameters_bias_"),
    ([216], "L_self_modules_serial_blocks2_modules_0_modules_norm2_parameters_weight_"),
    (
        [216],
        "L_self_modules_serial_blocks2_modules_1_modules_factoratt_crpe_modules_proj_parameters_bias_",
    ),
    (
        [216, 216],
        "L_self_modules_serial_blocks2_modules_1_modules_factoratt_crpe_modules_proj_parameters_weight_",
    ),
    (
        [648],
        "L_self_modules_serial_blocks2_modules_1_modules_factoratt_crpe_modules_qkv_parameters_bias_",
    ),
    (
        [648, 216],
        "L_self_modules_serial_blocks2_modules_1_modules_factoratt_crpe_modules_qkv_parameters_weight_",
    ),
    (
        [864],
        "L_self_modules_serial_blocks2_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [864, 216],
        "L_self_modules_serial_blocks2_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [216],
        "L_self_modules_serial_blocks2_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [216, 864],
        "L_self_modules_serial_blocks2_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([216], "L_self_modules_serial_blocks2_modules_1_modules_norm1_parameters_bias_"),
    ([216], "L_self_modules_serial_blocks2_modules_1_modules_norm1_parameters_weight_"),
    ([216], "L_self_modules_serial_blocks2_modules_1_modules_norm2_parameters_bias_"),
    ([216], "L_self_modules_serial_blocks2_modules_1_modules_norm2_parameters_weight_"),
    (
        [216],
        "L_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_bias_",
    ),
    (
        [216, 1, 3, 3],
        "L_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_weight_",
    ),
    (
        [54],
        "L_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_",
    ),
    (
        [54, 1, 3, 3],
        "L_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_",
    ),
    (
        [81],
        "L_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_",
    ),
    (
        [81, 1, 5, 5],
        "L_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_",
    ),
    (
        [81],
        "L_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_",
    ),
    (
        [81, 1, 7, 7],
        "L_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_",
    ),
    (
        [216],
        "L_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_proj_parameters_bias_",
    ),
    (
        [216, 216],
        "L_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_proj_parameters_weight_",
    ),
    (
        [648],
        "L_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_qkv_parameters_bias_",
    ),
    (
        [648, 216],
        "L_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_qkv_parameters_weight_",
    ),
    (
        [864],
        "L_self_modules_serial_blocks3_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [864, 216],
        "L_self_modules_serial_blocks3_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [216],
        "L_self_modules_serial_blocks3_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [216, 864],
        "L_self_modules_serial_blocks3_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([216], "L_self_modules_serial_blocks3_modules_0_modules_norm1_parameters_bias_"),
    ([216], "L_self_modules_serial_blocks3_modules_0_modules_norm1_parameters_weight_"),
    ([216], "L_self_modules_serial_blocks3_modules_0_modules_norm2_parameters_bias_"),
    ([216], "L_self_modules_serial_blocks3_modules_0_modules_norm2_parameters_weight_"),
    (
        [216],
        "L_self_modules_serial_blocks3_modules_1_modules_factoratt_crpe_modules_proj_parameters_bias_",
    ),
    (
        [216, 216],
        "L_self_modules_serial_blocks3_modules_1_modules_factoratt_crpe_modules_proj_parameters_weight_",
    ),
    (
        [648],
        "L_self_modules_serial_blocks3_modules_1_modules_factoratt_crpe_modules_qkv_parameters_bias_",
    ),
    (
        [648, 216],
        "L_self_modules_serial_blocks3_modules_1_modules_factoratt_crpe_modules_qkv_parameters_weight_",
    ),
    (
        [864],
        "L_self_modules_serial_blocks3_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [864, 216],
        "L_self_modules_serial_blocks3_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [216],
        "L_self_modules_serial_blocks3_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [216, 864],
        "L_self_modules_serial_blocks3_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([216], "L_self_modules_serial_blocks3_modules_1_modules_norm1_parameters_bias_"),
    ([216], "L_self_modules_serial_blocks3_modules_1_modules_norm1_parameters_weight_"),
    ([216], "L_self_modules_serial_blocks3_modules_1_modules_norm2_parameters_bias_"),
    ([216], "L_self_modules_serial_blocks3_modules_1_modules_norm2_parameters_weight_"),
    (
        [216],
        "L_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_bias_",
    ),
    (
        [216, 1, 3, 3],
        "L_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_weight_",
    ),
    (
        [54],
        "L_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_",
    ),
    (
        [54, 1, 3, 3],
        "L_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_",
    ),
    (
        [81],
        "L_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_",
    ),
    (
        [81, 1, 5, 5],
        "L_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_",
    ),
    (
        [81],
        "L_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_",
    ),
    (
        [81, 1, 7, 7],
        "L_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_",
    ),
    (
        [216],
        "L_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_proj_parameters_bias_",
    ),
    (
        [216, 216],
        "L_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_proj_parameters_weight_",
    ),
    (
        [648],
        "L_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_qkv_parameters_bias_",
    ),
    (
        [648, 216],
        "L_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_qkv_parameters_weight_",
    ),
    (
        [864],
        "L_self_modules_serial_blocks4_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [864, 216],
        "L_self_modules_serial_blocks4_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [216],
        "L_self_modules_serial_blocks4_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [216, 864],
        "L_self_modules_serial_blocks4_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([216], "L_self_modules_serial_blocks4_modules_0_modules_norm1_parameters_bias_"),
    ([216], "L_self_modules_serial_blocks4_modules_0_modules_norm1_parameters_weight_"),
    ([216], "L_self_modules_serial_blocks4_modules_0_modules_norm2_parameters_bias_"),
    ([216], "L_self_modules_serial_blocks4_modules_0_modules_norm2_parameters_weight_"),
    (
        [216],
        "L_self_modules_serial_blocks4_modules_1_modules_factoratt_crpe_modules_proj_parameters_bias_",
    ),
    (
        [216, 216],
        "L_self_modules_serial_blocks4_modules_1_modules_factoratt_crpe_modules_proj_parameters_weight_",
    ),
    (
        [648],
        "L_self_modules_serial_blocks4_modules_1_modules_factoratt_crpe_modules_qkv_parameters_bias_",
    ),
    (
        [648, 216],
        "L_self_modules_serial_blocks4_modules_1_modules_factoratt_crpe_modules_qkv_parameters_weight_",
    ),
    (
        [864],
        "L_self_modules_serial_blocks4_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [864, 216],
        "L_self_modules_serial_blocks4_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [216],
        "L_self_modules_serial_blocks4_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [216, 864],
        "L_self_modules_serial_blocks4_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([216], "L_self_modules_serial_blocks4_modules_1_modules_norm1_parameters_bias_"),
    ([216], "L_self_modules_serial_blocks4_modules_1_modules_norm1_parameters_weight_"),
    ([216], "L_self_modules_serial_blocks4_modules_1_modules_norm2_parameters_bias_"),
    ([216], "L_self_modules_serial_blocks4_modules_1_modules_norm2_parameters_weight_"),
    ([1, 1, 152], "L_self_parameters_cls_token1_"),
    ([1, 1, 216], "L_self_parameters_cls_token2_"),
    ([1, 1, 216], "L_self_parameters_cls_token3_"),
    ([1, 1, 216], "L_self_parameters_cls_token4_"),
    ([1, 3, S0, S0], "L_x_"),
]
