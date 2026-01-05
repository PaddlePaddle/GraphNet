import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_grn_parameters_bias_,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_grn_parameters_weight_,
        add_60,
        x_228,
        x_234,
        x_g_20,
    ):
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_grn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_grn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_grn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_grn_parameters_weight_
        x_n_20 = x_g_20 / add_60
        x_g_20 = add_60 = None
        view_40 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_grn_parameters_bias_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_grn_parameters_bias_ = (
            None
        )
        view_41 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_grn_parameters_weight_.view(
            (1, 1, 1, -1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_grn_parameters_weight_ = (
            None
        )
        mul_20 = x_234 * x_n_20
        x_n_20 = None
        addcmul_20 = torch.addcmul(view_40, view_41, mul_20)
        view_40 = view_41 = mul_20 = None
        x_235 = x_234 + addcmul_20
        x_234 = addcmul_20 = None
        x_236 = torch.nn.functional.linear(
            x_235,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_235 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_237 = torch.nn.functional.dropout(x_236, 0.0, False, False)
        x_236 = None
        x_238 = x_237.permute(0, 3, 1, 2)
        x_237 = None
        x_239 = x_238 + x_228
        x_238 = x_228 = None
        return (x_239,)
