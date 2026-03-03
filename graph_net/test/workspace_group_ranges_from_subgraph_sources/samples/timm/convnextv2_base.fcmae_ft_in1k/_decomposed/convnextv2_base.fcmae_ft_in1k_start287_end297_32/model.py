import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_grn_parameters_bias_,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_grn_parameters_weight_,
        add_42,
        x_162,
        x_168,
        x_g_14,
    ):
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_grn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_grn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_grn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_grn_parameters_weight_
        x_n_14 = x_g_14 / add_42
        x_g_14 = add_42 = None
        view_28 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_grn_parameters_bias_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_grn_parameters_bias_ = (
            None
        )
        view_29 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_grn_parameters_weight_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_grn_parameters_weight_ = (
            None
        )
        mul_14 = x_168 * x_n_14
        x_n_14 = None
        addcmul_14 = torch.addcmul(view_28, view_29, mul_14)
        view_28 = view_29 = mul_14 = None
        x_169 = x_168 + addcmul_14
        x_168 = addcmul_14 = None
        x_170 = torch.nn.functional.linear(
            x_169,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_169 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_171 = torch.nn.functional.dropout(x_170, 0.0, False, False)
        x_170 = None
        x_172 = x_171.permute(0, 3, 1, 2)
        x_171 = None
        x_173 = x_172 + x_162
        x_172 = x_162 = None
        return (x_173,)
