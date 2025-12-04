from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 384}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1000], "L_self_modules_head_parameters_bias_"),
    ([1000, 512], "L_self_modules_head_parameters_weight_"),
    ([512], "L_self_modules_norm4_parameters_bias_"),
    ([512], "L_self_modules_norm4_parameters_weight_"),
    ([128], "L_self_modules_patch_embed1_modules_norm_parameters_bias_"),
    ([128], "L_self_modules_patch_embed1_modules_norm_parameters_weight_"),
    ([128], "L_self_modules_patch_embed1_modules_proj_parameters_bias_"),
    ([128, 3, 4, 4], "L_self_modules_patch_embed1_modules_proj_parameters_weight_"),
    ([256], "L_self_modules_patch_embed2_modules_norm_parameters_bias_"),
    ([256], "L_self_modules_patch_embed2_modules_norm_parameters_weight_"),
    ([256], "L_self_modules_patch_embed2_modules_proj_parameters_bias_"),
    ([256, 128, 2, 2], "L_self_modules_patch_embed2_modules_proj_parameters_weight_"),
    ([320], "L_self_modules_patch_embed3_modules_norm_parameters_bias_"),
    ([320], "L_self_modules_patch_embed3_modules_norm_parameters_weight_"),
    ([320], "L_self_modules_patch_embed3_modules_proj_parameters_bias_"),
    ([320, 256, 2, 2], "L_self_modules_patch_embed3_modules_proj_parameters_weight_"),
    ([512], "L_self_modules_patch_embed4_modules_norm_parameters_bias_"),
    ([512], "L_self_modules_patch_embed4_modules_norm_parameters_weight_"),
    ([512], "L_self_modules_patch_embed4_modules_proj_parameters_bias_"),
    ([512, 320, 2, 2], "L_self_modules_patch_embed4_modules_proj_parameters_weight_"),
    (
        [128],
        "L_self_modules_serial_blocks1_modules_0_modules_cpe_modules_proj_parameters_bias_",
    ),
    (
        [128, 1, 3, 3],
        "L_self_modules_serial_blocks1_modules_0_modules_cpe_modules_proj_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_",
    ),
    (
        [32, 1, 3, 3],
        "L_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_",
    ),
    (
        [48, 1, 5, 5],
        "L_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_",
    ),
    (
        [48, 1, 7, 7],
        "L_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_proj_parameters_bias_",
    ),
    (
        [128, 128],
        "L_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_proj_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_qkv_parameters_bias_",
    ),
    (
        [384, 128],
        "L_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_qkv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_serial_blocks1_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [512, 128],
        "L_self_modules_serial_blocks1_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_serial_blocks1_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [128, 512],
        "L_self_modules_serial_blocks1_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([128], "L_self_modules_serial_blocks1_modules_0_modules_norm1_parameters_bias_"),
    ([128], "L_self_modules_serial_blocks1_modules_0_modules_norm1_parameters_weight_"),
    ([128], "L_self_modules_serial_blocks1_modules_0_modules_norm2_parameters_bias_"),
    ([128], "L_self_modules_serial_blocks1_modules_0_modules_norm2_parameters_weight_"),
    (
        [128],
        "L_self_modules_serial_blocks1_modules_1_modules_factoratt_crpe_modules_proj_parameters_bias_",
    ),
    (
        [128, 128],
        "L_self_modules_serial_blocks1_modules_1_modules_factoratt_crpe_modules_proj_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_serial_blocks1_modules_1_modules_factoratt_crpe_modules_qkv_parameters_bias_",
    ),
    (
        [384, 128],
        "L_self_modules_serial_blocks1_modules_1_modules_factoratt_crpe_modules_qkv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_serial_blocks1_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [512, 128],
        "L_self_modules_serial_blocks1_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_serial_blocks1_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [128, 512],
        "L_self_modules_serial_blocks1_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([128], "L_self_modules_serial_blocks1_modules_1_modules_norm1_parameters_bias_"),
    ([128], "L_self_modules_serial_blocks1_modules_1_modules_norm1_parameters_weight_"),
    ([128], "L_self_modules_serial_blocks1_modules_1_modules_norm2_parameters_bias_"),
    ([128], "L_self_modules_serial_blocks1_modules_1_modules_norm2_parameters_weight_"),
    (
        [128],
        "L_self_modules_serial_blocks1_modules_2_modules_factoratt_crpe_modules_proj_parameters_bias_",
    ),
    (
        [128, 128],
        "L_self_modules_serial_blocks1_modules_2_modules_factoratt_crpe_modules_proj_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_serial_blocks1_modules_2_modules_factoratt_crpe_modules_qkv_parameters_bias_",
    ),
    (
        [384, 128],
        "L_self_modules_serial_blocks1_modules_2_modules_factoratt_crpe_modules_qkv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_serial_blocks1_modules_2_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [512, 128],
        "L_self_modules_serial_blocks1_modules_2_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_serial_blocks1_modules_2_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [128, 512],
        "L_self_modules_serial_blocks1_modules_2_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([128], "L_self_modules_serial_blocks1_modules_2_modules_norm1_parameters_bias_"),
    ([128], "L_self_modules_serial_blocks1_modules_2_modules_norm1_parameters_weight_"),
    ([128], "L_self_modules_serial_blocks1_modules_2_modules_norm2_parameters_bias_"),
    ([128], "L_self_modules_serial_blocks1_modules_2_modules_norm2_parameters_weight_"),
    (
        [256],
        "L_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_bias_",
    ),
    (
        [256, 1, 3, 3],
        "L_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_",
    ),
    (
        [64, 1, 3, 3],
        "L_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_",
    ),
    (
        [96, 1, 5, 5],
        "L_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_",
    ),
    (
        [96, 1, 7, 7],
        "L_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_qkv_parameters_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_serial_blocks2_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1024, 256],
        "L_self_modules_serial_blocks2_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_serial_blocks2_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [256, 1024],
        "L_self_modules_serial_blocks2_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([256], "L_self_modules_serial_blocks2_modules_0_modules_norm1_parameters_bias_"),
    ([256], "L_self_modules_serial_blocks2_modules_0_modules_norm1_parameters_weight_"),
    ([256], "L_self_modules_serial_blocks2_modules_0_modules_norm2_parameters_bias_"),
    ([256], "L_self_modules_serial_blocks2_modules_0_modules_norm2_parameters_weight_"),
    (
        [256],
        "L_self_modules_serial_blocks2_modules_1_modules_factoratt_crpe_modules_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_serial_blocks2_modules_1_modules_factoratt_crpe_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_serial_blocks2_modules_1_modules_factoratt_crpe_modules_qkv_parameters_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_serial_blocks2_modules_1_modules_factoratt_crpe_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_serial_blocks2_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1024, 256],
        "L_self_modules_serial_blocks2_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_serial_blocks2_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [256, 1024],
        "L_self_modules_serial_blocks2_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([256], "L_self_modules_serial_blocks2_modules_1_modules_norm1_parameters_bias_"),
    ([256], "L_self_modules_serial_blocks2_modules_1_modules_norm1_parameters_weight_"),
    ([256], "L_self_modules_serial_blocks2_modules_1_modules_norm2_parameters_bias_"),
    ([256], "L_self_modules_serial_blocks2_modules_1_modules_norm2_parameters_weight_"),
    (
        [256],
        "L_self_modules_serial_blocks2_modules_2_modules_factoratt_crpe_modules_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_serial_blocks2_modules_2_modules_factoratt_crpe_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_serial_blocks2_modules_2_modules_factoratt_crpe_modules_qkv_parameters_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_serial_blocks2_modules_2_modules_factoratt_crpe_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_serial_blocks2_modules_2_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1024, 256],
        "L_self_modules_serial_blocks2_modules_2_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_serial_blocks2_modules_2_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [256, 1024],
        "L_self_modules_serial_blocks2_modules_2_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([256], "L_self_modules_serial_blocks2_modules_2_modules_norm1_parameters_bias_"),
    ([256], "L_self_modules_serial_blocks2_modules_2_modules_norm1_parameters_weight_"),
    ([256], "L_self_modules_serial_blocks2_modules_2_modules_norm2_parameters_bias_"),
    ([256], "L_self_modules_serial_blocks2_modules_2_modules_norm2_parameters_weight_"),
    (
        [256],
        "L_self_modules_serial_blocks2_modules_3_modules_factoratt_crpe_modules_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_serial_blocks2_modules_3_modules_factoratt_crpe_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_serial_blocks2_modules_3_modules_factoratt_crpe_modules_qkv_parameters_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_serial_blocks2_modules_3_modules_factoratt_crpe_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_serial_blocks2_modules_3_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1024, 256],
        "L_self_modules_serial_blocks2_modules_3_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_serial_blocks2_modules_3_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [256, 1024],
        "L_self_modules_serial_blocks2_modules_3_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([256], "L_self_modules_serial_blocks2_modules_3_modules_norm1_parameters_bias_"),
    ([256], "L_self_modules_serial_blocks2_modules_3_modules_norm1_parameters_weight_"),
    ([256], "L_self_modules_serial_blocks2_modules_3_modules_norm2_parameters_bias_"),
    ([256], "L_self_modules_serial_blocks2_modules_3_modules_norm2_parameters_weight_"),
    (
        [256],
        "L_self_modules_serial_blocks2_modules_4_modules_factoratt_crpe_modules_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_serial_blocks2_modules_4_modules_factoratt_crpe_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_serial_blocks2_modules_4_modules_factoratt_crpe_modules_qkv_parameters_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_serial_blocks2_modules_4_modules_factoratt_crpe_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_serial_blocks2_modules_4_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1024, 256],
        "L_self_modules_serial_blocks2_modules_4_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_serial_blocks2_modules_4_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [256, 1024],
        "L_self_modules_serial_blocks2_modules_4_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([256], "L_self_modules_serial_blocks2_modules_4_modules_norm1_parameters_bias_"),
    ([256], "L_self_modules_serial_blocks2_modules_4_modules_norm1_parameters_weight_"),
    ([256], "L_self_modules_serial_blocks2_modules_4_modules_norm2_parameters_bias_"),
    ([256], "L_self_modules_serial_blocks2_modules_4_modules_norm2_parameters_weight_"),
    (
        [256],
        "L_self_modules_serial_blocks2_modules_5_modules_factoratt_crpe_modules_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_serial_blocks2_modules_5_modules_factoratt_crpe_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_serial_blocks2_modules_5_modules_factoratt_crpe_modules_qkv_parameters_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_serial_blocks2_modules_5_modules_factoratt_crpe_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_serial_blocks2_modules_5_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1024, 256],
        "L_self_modules_serial_blocks2_modules_5_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_serial_blocks2_modules_5_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [256, 1024],
        "L_self_modules_serial_blocks2_modules_5_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([256], "L_self_modules_serial_blocks2_modules_5_modules_norm1_parameters_bias_"),
    ([256], "L_self_modules_serial_blocks2_modules_5_modules_norm1_parameters_weight_"),
    ([256], "L_self_modules_serial_blocks2_modules_5_modules_norm2_parameters_bias_"),
    ([256], "L_self_modules_serial_blocks2_modules_5_modules_norm2_parameters_weight_"),
    (
        [320],
        "L_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_bias_",
    ),
    (
        [320, 1, 3, 3],
        "L_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_",
    ),
    (
        [80, 1, 3, 3],
        "L_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_",
    ),
    (
        [120],
        "L_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_",
    ),
    (
        [120, 1, 5, 5],
        "L_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_",
    ),
    (
        [120],
        "L_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_",
    ),
    (
        [120, 1, 7, 7],
        "L_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_proj_parameters_bias_",
    ),
    (
        [320, 320],
        "L_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_proj_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_qkv_parameters_bias_",
    ),
    (
        [960, 320],
        "L_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_qkv_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_serial_blocks3_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1280, 320],
        "L_self_modules_serial_blocks3_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_serial_blocks3_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [320, 1280],
        "L_self_modules_serial_blocks3_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([320], "L_self_modules_serial_blocks3_modules_0_modules_norm1_parameters_bias_"),
    ([320], "L_self_modules_serial_blocks3_modules_0_modules_norm1_parameters_weight_"),
    ([320], "L_self_modules_serial_blocks3_modules_0_modules_norm2_parameters_bias_"),
    ([320], "L_self_modules_serial_blocks3_modules_0_modules_norm2_parameters_weight_"),
    (
        [320],
        "L_self_modules_serial_blocks3_modules_1_modules_factoratt_crpe_modules_proj_parameters_bias_",
    ),
    (
        [320, 320],
        "L_self_modules_serial_blocks3_modules_1_modules_factoratt_crpe_modules_proj_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_serial_blocks3_modules_1_modules_factoratt_crpe_modules_qkv_parameters_bias_",
    ),
    (
        [960, 320],
        "L_self_modules_serial_blocks3_modules_1_modules_factoratt_crpe_modules_qkv_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_serial_blocks3_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1280, 320],
        "L_self_modules_serial_blocks3_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_serial_blocks3_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [320, 1280],
        "L_self_modules_serial_blocks3_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([320], "L_self_modules_serial_blocks3_modules_1_modules_norm1_parameters_bias_"),
    ([320], "L_self_modules_serial_blocks3_modules_1_modules_norm1_parameters_weight_"),
    ([320], "L_self_modules_serial_blocks3_modules_1_modules_norm2_parameters_bias_"),
    ([320], "L_self_modules_serial_blocks3_modules_1_modules_norm2_parameters_weight_"),
    (
        [320],
        "L_self_modules_serial_blocks3_modules_2_modules_factoratt_crpe_modules_proj_parameters_bias_",
    ),
    (
        [320, 320],
        "L_self_modules_serial_blocks3_modules_2_modules_factoratt_crpe_modules_proj_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_serial_blocks3_modules_2_modules_factoratt_crpe_modules_qkv_parameters_bias_",
    ),
    (
        [960, 320],
        "L_self_modules_serial_blocks3_modules_2_modules_factoratt_crpe_modules_qkv_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_serial_blocks3_modules_2_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1280, 320],
        "L_self_modules_serial_blocks3_modules_2_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_serial_blocks3_modules_2_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [320, 1280],
        "L_self_modules_serial_blocks3_modules_2_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([320], "L_self_modules_serial_blocks3_modules_2_modules_norm1_parameters_bias_"),
    ([320], "L_self_modules_serial_blocks3_modules_2_modules_norm1_parameters_weight_"),
    ([320], "L_self_modules_serial_blocks3_modules_2_modules_norm2_parameters_bias_"),
    ([320], "L_self_modules_serial_blocks3_modules_2_modules_norm2_parameters_weight_"),
    (
        [320],
        "L_self_modules_serial_blocks3_modules_3_modules_factoratt_crpe_modules_proj_parameters_bias_",
    ),
    (
        [320, 320],
        "L_self_modules_serial_blocks3_modules_3_modules_factoratt_crpe_modules_proj_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_serial_blocks3_modules_3_modules_factoratt_crpe_modules_qkv_parameters_bias_",
    ),
    (
        [960, 320],
        "L_self_modules_serial_blocks3_modules_3_modules_factoratt_crpe_modules_qkv_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_serial_blocks3_modules_3_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1280, 320],
        "L_self_modules_serial_blocks3_modules_3_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_serial_blocks3_modules_3_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [320, 1280],
        "L_self_modules_serial_blocks3_modules_3_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([320], "L_self_modules_serial_blocks3_modules_3_modules_norm1_parameters_bias_"),
    ([320], "L_self_modules_serial_blocks3_modules_3_modules_norm1_parameters_weight_"),
    ([320], "L_self_modules_serial_blocks3_modules_3_modules_norm2_parameters_bias_"),
    ([320], "L_self_modules_serial_blocks3_modules_3_modules_norm2_parameters_weight_"),
    (
        [320],
        "L_self_modules_serial_blocks3_modules_4_modules_factoratt_crpe_modules_proj_parameters_bias_",
    ),
    (
        [320, 320],
        "L_self_modules_serial_blocks3_modules_4_modules_factoratt_crpe_modules_proj_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_serial_blocks3_modules_4_modules_factoratt_crpe_modules_qkv_parameters_bias_",
    ),
    (
        [960, 320],
        "L_self_modules_serial_blocks3_modules_4_modules_factoratt_crpe_modules_qkv_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_serial_blocks3_modules_4_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1280, 320],
        "L_self_modules_serial_blocks3_modules_4_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_serial_blocks3_modules_4_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [320, 1280],
        "L_self_modules_serial_blocks3_modules_4_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([320], "L_self_modules_serial_blocks3_modules_4_modules_norm1_parameters_bias_"),
    ([320], "L_self_modules_serial_blocks3_modules_4_modules_norm1_parameters_weight_"),
    ([320], "L_self_modules_serial_blocks3_modules_4_modules_norm2_parameters_bias_"),
    ([320], "L_self_modules_serial_blocks3_modules_4_modules_norm2_parameters_weight_"),
    (
        [320],
        "L_self_modules_serial_blocks3_modules_5_modules_factoratt_crpe_modules_proj_parameters_bias_",
    ),
    (
        [320, 320],
        "L_self_modules_serial_blocks3_modules_5_modules_factoratt_crpe_modules_proj_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_serial_blocks3_modules_5_modules_factoratt_crpe_modules_qkv_parameters_bias_",
    ),
    (
        [960, 320],
        "L_self_modules_serial_blocks3_modules_5_modules_factoratt_crpe_modules_qkv_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_serial_blocks3_modules_5_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1280, 320],
        "L_self_modules_serial_blocks3_modules_5_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_serial_blocks3_modules_5_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [320, 1280],
        "L_self_modules_serial_blocks3_modules_5_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([320], "L_self_modules_serial_blocks3_modules_5_modules_norm1_parameters_bias_"),
    ([320], "L_self_modules_serial_blocks3_modules_5_modules_norm1_parameters_weight_"),
    ([320], "L_self_modules_serial_blocks3_modules_5_modules_norm2_parameters_bias_"),
    ([320], "L_self_modules_serial_blocks3_modules_5_modules_norm2_parameters_weight_"),
    (
        [320],
        "L_self_modules_serial_blocks3_modules_6_modules_factoratt_crpe_modules_proj_parameters_bias_",
    ),
    (
        [320, 320],
        "L_self_modules_serial_blocks3_modules_6_modules_factoratt_crpe_modules_proj_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_serial_blocks3_modules_6_modules_factoratt_crpe_modules_qkv_parameters_bias_",
    ),
    (
        [960, 320],
        "L_self_modules_serial_blocks3_modules_6_modules_factoratt_crpe_modules_qkv_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_serial_blocks3_modules_6_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1280, 320],
        "L_self_modules_serial_blocks3_modules_6_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_serial_blocks3_modules_6_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [320, 1280],
        "L_self_modules_serial_blocks3_modules_6_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([320], "L_self_modules_serial_blocks3_modules_6_modules_norm1_parameters_bias_"),
    ([320], "L_self_modules_serial_blocks3_modules_6_modules_norm1_parameters_weight_"),
    ([320], "L_self_modules_serial_blocks3_modules_6_modules_norm2_parameters_bias_"),
    ([320], "L_self_modules_serial_blocks3_modules_6_modules_norm2_parameters_weight_"),
    (
        [320],
        "L_self_modules_serial_blocks3_modules_7_modules_factoratt_crpe_modules_proj_parameters_bias_",
    ),
    (
        [320, 320],
        "L_self_modules_serial_blocks3_modules_7_modules_factoratt_crpe_modules_proj_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_serial_blocks3_modules_7_modules_factoratt_crpe_modules_qkv_parameters_bias_",
    ),
    (
        [960, 320],
        "L_self_modules_serial_blocks3_modules_7_modules_factoratt_crpe_modules_qkv_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_serial_blocks3_modules_7_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1280, 320],
        "L_self_modules_serial_blocks3_modules_7_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_serial_blocks3_modules_7_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [320, 1280],
        "L_self_modules_serial_blocks3_modules_7_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([320], "L_self_modules_serial_blocks3_modules_7_modules_norm1_parameters_bias_"),
    ([320], "L_self_modules_serial_blocks3_modules_7_modules_norm1_parameters_weight_"),
    ([320], "L_self_modules_serial_blocks3_modules_7_modules_norm2_parameters_bias_"),
    ([320], "L_self_modules_serial_blocks3_modules_7_modules_norm2_parameters_weight_"),
    (
        [320],
        "L_self_modules_serial_blocks3_modules_8_modules_factoratt_crpe_modules_proj_parameters_bias_",
    ),
    (
        [320, 320],
        "L_self_modules_serial_blocks3_modules_8_modules_factoratt_crpe_modules_proj_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_serial_blocks3_modules_8_modules_factoratt_crpe_modules_qkv_parameters_bias_",
    ),
    (
        [960, 320],
        "L_self_modules_serial_blocks3_modules_8_modules_factoratt_crpe_modules_qkv_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_serial_blocks3_modules_8_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1280, 320],
        "L_self_modules_serial_blocks3_modules_8_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_serial_blocks3_modules_8_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [320, 1280],
        "L_self_modules_serial_blocks3_modules_8_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([320], "L_self_modules_serial_blocks3_modules_8_modules_norm1_parameters_bias_"),
    ([320], "L_self_modules_serial_blocks3_modules_8_modules_norm1_parameters_weight_"),
    ([320], "L_self_modules_serial_blocks3_modules_8_modules_norm2_parameters_bias_"),
    ([320], "L_self_modules_serial_blocks3_modules_8_modules_norm2_parameters_weight_"),
    (
        [320],
        "L_self_modules_serial_blocks3_modules_9_modules_factoratt_crpe_modules_proj_parameters_bias_",
    ),
    (
        [320, 320],
        "L_self_modules_serial_blocks3_modules_9_modules_factoratt_crpe_modules_proj_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_serial_blocks3_modules_9_modules_factoratt_crpe_modules_qkv_parameters_bias_",
    ),
    (
        [960, 320],
        "L_self_modules_serial_blocks3_modules_9_modules_factoratt_crpe_modules_qkv_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_serial_blocks3_modules_9_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1280, 320],
        "L_self_modules_serial_blocks3_modules_9_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_serial_blocks3_modules_9_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [320, 1280],
        "L_self_modules_serial_blocks3_modules_9_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([320], "L_self_modules_serial_blocks3_modules_9_modules_norm1_parameters_bias_"),
    ([320], "L_self_modules_serial_blocks3_modules_9_modules_norm1_parameters_weight_"),
    ([320], "L_self_modules_serial_blocks3_modules_9_modules_norm2_parameters_bias_"),
    ([320], "L_self_modules_serial_blocks3_modules_9_modules_norm2_parameters_weight_"),
    (
        [512],
        "L_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_bias_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_",
    ),
    (
        [128, 1, 3, 3],
        "L_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_",
    ),
    (
        [192, 1, 5, 5],
        "L_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_",
    ),
    (
        [192, 1, 7, 7],
        "L_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_qkv_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_qkv_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_serial_blocks4_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_serial_blocks4_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_serial_blocks4_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_serial_blocks4_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([512], "L_self_modules_serial_blocks4_modules_0_modules_norm1_parameters_bias_"),
    ([512], "L_self_modules_serial_blocks4_modules_0_modules_norm1_parameters_weight_"),
    ([512], "L_self_modules_serial_blocks4_modules_0_modules_norm2_parameters_bias_"),
    ([512], "L_self_modules_serial_blocks4_modules_0_modules_norm2_parameters_weight_"),
    (
        [512],
        "L_self_modules_serial_blocks4_modules_1_modules_factoratt_crpe_modules_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_serial_blocks4_modules_1_modules_factoratt_crpe_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_serial_blocks4_modules_1_modules_factoratt_crpe_modules_qkv_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_serial_blocks4_modules_1_modules_factoratt_crpe_modules_qkv_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_serial_blocks4_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_serial_blocks4_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_serial_blocks4_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_serial_blocks4_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([512], "L_self_modules_serial_blocks4_modules_1_modules_norm1_parameters_bias_"),
    ([512], "L_self_modules_serial_blocks4_modules_1_modules_norm1_parameters_weight_"),
    ([512], "L_self_modules_serial_blocks4_modules_1_modules_norm2_parameters_bias_"),
    ([512], "L_self_modules_serial_blocks4_modules_1_modules_norm2_parameters_weight_"),
    (
        [512],
        "L_self_modules_serial_blocks4_modules_2_modules_factoratt_crpe_modules_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_serial_blocks4_modules_2_modules_factoratt_crpe_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_serial_blocks4_modules_2_modules_factoratt_crpe_modules_qkv_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_serial_blocks4_modules_2_modules_factoratt_crpe_modules_qkv_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_serial_blocks4_modules_2_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_serial_blocks4_modules_2_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_serial_blocks4_modules_2_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_serial_blocks4_modules_2_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([512], "L_self_modules_serial_blocks4_modules_2_modules_norm1_parameters_bias_"),
    ([512], "L_self_modules_serial_blocks4_modules_2_modules_norm1_parameters_weight_"),
    ([512], "L_self_modules_serial_blocks4_modules_2_modules_norm2_parameters_bias_"),
    ([512], "L_self_modules_serial_blocks4_modules_2_modules_norm2_parameters_weight_"),
    (
        [512],
        "L_self_modules_serial_blocks4_modules_3_modules_factoratt_crpe_modules_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_serial_blocks4_modules_3_modules_factoratt_crpe_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_serial_blocks4_modules_3_modules_factoratt_crpe_modules_qkv_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_serial_blocks4_modules_3_modules_factoratt_crpe_modules_qkv_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_serial_blocks4_modules_3_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_serial_blocks4_modules_3_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_serial_blocks4_modules_3_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_serial_blocks4_modules_3_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([512], "L_self_modules_serial_blocks4_modules_3_modules_norm1_parameters_bias_"),
    ([512], "L_self_modules_serial_blocks4_modules_3_modules_norm1_parameters_weight_"),
    ([512], "L_self_modules_serial_blocks4_modules_3_modules_norm2_parameters_bias_"),
    ([512], "L_self_modules_serial_blocks4_modules_3_modules_norm2_parameters_weight_"),
    (
        [512],
        "L_self_modules_serial_blocks4_modules_4_modules_factoratt_crpe_modules_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_serial_blocks4_modules_4_modules_factoratt_crpe_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_serial_blocks4_modules_4_modules_factoratt_crpe_modules_qkv_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_serial_blocks4_modules_4_modules_factoratt_crpe_modules_qkv_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_serial_blocks4_modules_4_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_serial_blocks4_modules_4_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_serial_blocks4_modules_4_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_serial_blocks4_modules_4_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([512], "L_self_modules_serial_blocks4_modules_4_modules_norm1_parameters_bias_"),
    ([512], "L_self_modules_serial_blocks4_modules_4_modules_norm1_parameters_weight_"),
    ([512], "L_self_modules_serial_blocks4_modules_4_modules_norm2_parameters_bias_"),
    ([512], "L_self_modules_serial_blocks4_modules_4_modules_norm2_parameters_weight_"),
    (
        [512],
        "L_self_modules_serial_blocks4_modules_5_modules_factoratt_crpe_modules_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_serial_blocks4_modules_5_modules_factoratt_crpe_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_serial_blocks4_modules_5_modules_factoratt_crpe_modules_qkv_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_serial_blocks4_modules_5_modules_factoratt_crpe_modules_qkv_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_serial_blocks4_modules_5_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_serial_blocks4_modules_5_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_serial_blocks4_modules_5_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_serial_blocks4_modules_5_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([512], "L_self_modules_serial_blocks4_modules_5_modules_norm1_parameters_bias_"),
    ([512], "L_self_modules_serial_blocks4_modules_5_modules_norm1_parameters_weight_"),
    ([512], "L_self_modules_serial_blocks4_modules_5_modules_norm2_parameters_bias_"),
    ([512], "L_self_modules_serial_blocks4_modules_5_modules_norm2_parameters_weight_"),
    (
        [512],
        "L_self_modules_serial_blocks4_modules_6_modules_factoratt_crpe_modules_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_serial_blocks4_modules_6_modules_factoratt_crpe_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_serial_blocks4_modules_6_modules_factoratt_crpe_modules_qkv_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_serial_blocks4_modules_6_modules_factoratt_crpe_modules_qkv_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_serial_blocks4_modules_6_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_serial_blocks4_modules_6_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_serial_blocks4_modules_6_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_serial_blocks4_modules_6_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([512], "L_self_modules_serial_blocks4_modules_6_modules_norm1_parameters_bias_"),
    ([512], "L_self_modules_serial_blocks4_modules_6_modules_norm1_parameters_weight_"),
    ([512], "L_self_modules_serial_blocks4_modules_6_modules_norm2_parameters_bias_"),
    ([512], "L_self_modules_serial_blocks4_modules_6_modules_norm2_parameters_weight_"),
    (
        [512],
        "L_self_modules_serial_blocks4_modules_7_modules_factoratt_crpe_modules_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_serial_blocks4_modules_7_modules_factoratt_crpe_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_serial_blocks4_modules_7_modules_factoratt_crpe_modules_qkv_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_serial_blocks4_modules_7_modules_factoratt_crpe_modules_qkv_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_serial_blocks4_modules_7_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_serial_blocks4_modules_7_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_serial_blocks4_modules_7_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_serial_blocks4_modules_7_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([512], "L_self_modules_serial_blocks4_modules_7_modules_norm1_parameters_bias_"),
    ([512], "L_self_modules_serial_blocks4_modules_7_modules_norm1_parameters_weight_"),
    ([512], "L_self_modules_serial_blocks4_modules_7_modules_norm2_parameters_bias_"),
    ([512], "L_self_modules_serial_blocks4_modules_7_modules_norm2_parameters_weight_"),
    ([1, 1, 128], "L_self_parameters_cls_token1_"),
    ([1, 1, 256], "L_self_parameters_cls_token2_"),
    ([1, 1, 320], "L_self_parameters_cls_token3_"),
    ([1, 1, 512], "L_self_parameters_cls_token4_"),
    ([1, 3, S0, S0], "L_x_"),
]
