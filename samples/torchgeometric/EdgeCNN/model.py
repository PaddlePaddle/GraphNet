import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_x_: torch.Tensor,
        L_edge_index_: torch.Tensor,
        L_self_modules_convs_modules_0_modules_nn_modules_lins_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_0_modules_nn_modules_lins_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_0_modules_nn_modules_lins_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_0_modules_nn_modules_lins_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_1_modules_nn_modules_lins_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_1_modules_nn_modules_lins_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_1_modules_nn_modules_lins_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_1_modules_nn_modules_lins_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_2_modules_nn_modules_lins_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_2_modules_nn_modules_lins_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_2_modules_nn_modules_lins_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_2_modules_nn_modules_lins_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_3_modules_nn_modules_lins_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_3_modules_nn_modules_lins_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_3_modules_nn_modules_lins_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_3_modules_nn_modules_lins_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_4_modules_nn_modules_lins_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_4_modules_nn_modules_lins_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_4_modules_nn_modules_lins_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_4_modules_nn_modules_lins_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_x_ = L_x_
        l_edge_index_ = L_edge_index_
        l_self_modules_convs_modules_0_modules_nn_modules_lins_modules_0_parameters_weight_ = L_self_modules_convs_modules_0_modules_nn_modules_lins_modules_0_parameters_weight_
        l_self_modules_convs_modules_0_modules_nn_modules_lins_modules_0_parameters_bias_ = L_self_modules_convs_modules_0_modules_nn_modules_lins_modules_0_parameters_bias_
        l_self_modules_convs_modules_0_modules_nn_modules_lins_modules_1_parameters_weight_ = L_self_modules_convs_modules_0_modules_nn_modules_lins_modules_1_parameters_weight_
        l_self_modules_convs_modules_0_modules_nn_modules_lins_modules_1_parameters_bias_ = L_self_modules_convs_modules_0_modules_nn_modules_lins_modules_1_parameters_bias_
        l_self_modules_convs_modules_1_modules_nn_modules_lins_modules_0_parameters_weight_ = L_self_modules_convs_modules_1_modules_nn_modules_lins_modules_0_parameters_weight_
        l_self_modules_convs_modules_1_modules_nn_modules_lins_modules_0_parameters_bias_ = L_self_modules_convs_modules_1_modules_nn_modules_lins_modules_0_parameters_bias_
        l_self_modules_convs_modules_1_modules_nn_modules_lins_modules_1_parameters_weight_ = L_self_modules_convs_modules_1_modules_nn_modules_lins_modules_1_parameters_weight_
        l_self_modules_convs_modules_1_modules_nn_modules_lins_modules_1_parameters_bias_ = L_self_modules_convs_modules_1_modules_nn_modules_lins_modules_1_parameters_bias_
        l_self_modules_convs_modules_2_modules_nn_modules_lins_modules_0_parameters_weight_ = L_self_modules_convs_modules_2_modules_nn_modules_lins_modules_0_parameters_weight_
        l_self_modules_convs_modules_2_modules_nn_modules_lins_modules_0_parameters_bias_ = L_self_modules_convs_modules_2_modules_nn_modules_lins_modules_0_parameters_bias_
        l_self_modules_convs_modules_2_modules_nn_modules_lins_modules_1_parameters_weight_ = L_self_modules_convs_modules_2_modules_nn_modules_lins_modules_1_parameters_weight_
        l_self_modules_convs_modules_2_modules_nn_modules_lins_modules_1_parameters_bias_ = L_self_modules_convs_modules_2_modules_nn_modules_lins_modules_1_parameters_bias_
        l_self_modules_convs_modules_3_modules_nn_modules_lins_modules_0_parameters_weight_ = L_self_modules_convs_modules_3_modules_nn_modules_lins_modules_0_parameters_weight_
        l_self_modules_convs_modules_3_modules_nn_modules_lins_modules_0_parameters_bias_ = L_self_modules_convs_modules_3_modules_nn_modules_lins_modules_0_parameters_bias_
        l_self_modules_convs_modules_3_modules_nn_modules_lins_modules_1_parameters_weight_ = L_self_modules_convs_modules_3_modules_nn_modules_lins_modules_1_parameters_weight_
        l_self_modules_convs_modules_3_modules_nn_modules_lins_modules_1_parameters_bias_ = L_self_modules_convs_modules_3_modules_nn_modules_lins_modules_1_parameters_bias_
        l_self_modules_convs_modules_4_modules_nn_modules_lins_modules_0_parameters_weight_ = L_self_modules_convs_modules_4_modules_nn_modules_lins_modules_0_parameters_weight_
        l_self_modules_convs_modules_4_modules_nn_modules_lins_modules_0_parameters_bias_ = L_self_modules_convs_modules_4_modules_nn_modules_lins_modules_0_parameters_bias_
        l_self_modules_convs_modules_4_modules_nn_modules_lins_modules_1_parameters_weight_ = L_self_modules_convs_modules_4_modules_nn_modules_lins_modules_1_parameters_weight_
        l_self_modules_convs_modules_4_modules_nn_modules_lins_modules_1_parameters_bias_ = L_self_modules_convs_modules_4_modules_nn_modules_lins_modules_1_parameters_bias_
        edge_index_i = l_edge_index_[1]
        edge_index_j = l_edge_index_[0]
        x_i = l_x_.index_select(-2, edge_index_i)
        x_j = l_x_.index_select(-2, edge_index_j)
        l_x_ = edge_index_j = None
        sub = x_j - x_i
        x_j = None
        cat = torch.cat([x_i, sub], dim=-1)
        x_i = sub = None
        x = torch._C._nn.linear(
            cat,
            l_self_modules_convs_modules_0_modules_nn_modules_lins_modules_0_parameters_weight_,
            l_self_modules_convs_modules_0_modules_nn_modules_lins_modules_0_parameters_bias_,
        )
        cat = l_self_modules_convs_modules_0_modules_nn_modules_lins_modules_0_parameters_weight_ = l_self_modules_convs_modules_0_modules_nn_modules_lins_modules_0_parameters_bias_ = (None)
        x_1 = torch.nn.functional.relu(x, inplace=False)
        x = None
        x_2 = torch.nn.functional.dropout(x_1, p=0.0, training=False)
        x_1 = None
        x_3 = torch._C._nn.linear(
            x_2,
            l_self_modules_convs_modules_0_modules_nn_modules_lins_modules_1_parameters_weight_,
            l_self_modules_convs_modules_0_modules_nn_modules_lins_modules_1_parameters_bias_,
        )
        x_2 = l_self_modules_convs_modules_0_modules_nn_modules_lins_modules_1_parameters_weight_ = l_self_modules_convs_modules_0_modules_nn_modules_lins_modules_1_parameters_bias_ = (None)
        x_4 = torch.nn.functional.dropout(x_3, p=0.0, training=False)
        x_3 = None
        view = edge_index_i.view((-1, 1))
        edge_index_i = None
        index = view.expand_as(x_4)
        view = None
        new_zeros = x_4.new_zeros((1000, 256))
        out = new_zeros.scatter_reduce_(
            0, index, x_4, reduce="amax", include_self=False
        )
        new_zeros = index = x_4 = None
        x_5 = torch.nn.functional.relu(out, inplace=False)
        out = None
        x_6 = torch.nn.functional.dropout(x_5, 0.0, False, False)
        x_5 = None
        edge_index_i_1 = l_edge_index_[1]
        edge_index_j_1 = l_edge_index_[0]
        x_i_1 = x_6.index_select(-2, edge_index_i_1)
        x_j_1 = x_6.index_select(-2, edge_index_j_1)
        x_6 = edge_index_j_1 = None
        sub_1 = x_j_1 - x_i_1
        x_j_1 = None
        cat_1 = torch.cat([x_i_1, sub_1], dim=-1)
        x_i_1 = sub_1 = None
        x_7 = torch._C._nn.linear(
            cat_1,
            l_self_modules_convs_modules_1_modules_nn_modules_lins_modules_0_parameters_weight_,
            l_self_modules_convs_modules_1_modules_nn_modules_lins_modules_0_parameters_bias_,
        )
        cat_1 = l_self_modules_convs_modules_1_modules_nn_modules_lins_modules_0_parameters_weight_ = l_self_modules_convs_modules_1_modules_nn_modules_lins_modules_0_parameters_bias_ = (None)
        x_8 = torch.nn.functional.relu(x_7, inplace=False)
        x_7 = None
        x_9 = torch.nn.functional.dropout(x_8, p=0.0, training=False)
        x_8 = None
        x_10 = torch._C._nn.linear(
            x_9,
            l_self_modules_convs_modules_1_modules_nn_modules_lins_modules_1_parameters_weight_,
            l_self_modules_convs_modules_1_modules_nn_modules_lins_modules_1_parameters_bias_,
        )
        x_9 = l_self_modules_convs_modules_1_modules_nn_modules_lins_modules_1_parameters_weight_ = l_self_modules_convs_modules_1_modules_nn_modules_lins_modules_1_parameters_bias_ = (None)
        x_11 = torch.nn.functional.dropout(x_10, p=0.0, training=False)
        x_10 = None
        view_1 = edge_index_i_1.view((-1, 1))
        edge_index_i_1 = None
        index_1 = view_1.expand_as(x_11)
        view_1 = None
        new_zeros_1 = x_11.new_zeros((1000, 256))
        out_1 = new_zeros_1.scatter_reduce_(
            0, index_1, x_11, reduce="amax", include_self=False
        )
        new_zeros_1 = index_1 = x_11 = None
        x_12 = torch.nn.functional.relu(out_1, inplace=False)
        out_1 = None
        x_13 = torch.nn.functional.dropout(x_12, 0.0, False, False)
        x_12 = None
        edge_index_i_2 = l_edge_index_[1]
        edge_index_j_2 = l_edge_index_[0]
        x_i_2 = x_13.index_select(-2, edge_index_i_2)
        x_j_2 = x_13.index_select(-2, edge_index_j_2)
        x_13 = edge_index_j_2 = None
        sub_2 = x_j_2 - x_i_2
        x_j_2 = None
        cat_2 = torch.cat([x_i_2, sub_2], dim=-1)
        x_i_2 = sub_2 = None
        x_14 = torch._C._nn.linear(
            cat_2,
            l_self_modules_convs_modules_2_modules_nn_modules_lins_modules_0_parameters_weight_,
            l_self_modules_convs_modules_2_modules_nn_modules_lins_modules_0_parameters_bias_,
        )
        cat_2 = l_self_modules_convs_modules_2_modules_nn_modules_lins_modules_0_parameters_weight_ = l_self_modules_convs_modules_2_modules_nn_modules_lins_modules_0_parameters_bias_ = (None)
        x_15 = torch.nn.functional.relu(x_14, inplace=False)
        x_14 = None
        x_16 = torch.nn.functional.dropout(x_15, p=0.0, training=False)
        x_15 = None
        x_17 = torch._C._nn.linear(
            x_16,
            l_self_modules_convs_modules_2_modules_nn_modules_lins_modules_1_parameters_weight_,
            l_self_modules_convs_modules_2_modules_nn_modules_lins_modules_1_parameters_bias_,
        )
        x_16 = l_self_modules_convs_modules_2_modules_nn_modules_lins_modules_1_parameters_weight_ = l_self_modules_convs_modules_2_modules_nn_modules_lins_modules_1_parameters_bias_ = (None)
        x_18 = torch.nn.functional.dropout(x_17, p=0.0, training=False)
        x_17 = None
        view_2 = edge_index_i_2.view((-1, 1))
        edge_index_i_2 = None
        index_2 = view_2.expand_as(x_18)
        view_2 = None
        new_zeros_2 = x_18.new_zeros((1000, 256))
        out_2 = new_zeros_2.scatter_reduce_(
            0, index_2, x_18, reduce="amax", include_self=False
        )
        new_zeros_2 = index_2 = x_18 = None
        x_19 = torch.nn.functional.relu(out_2, inplace=False)
        out_2 = None
        x_20 = torch.nn.functional.dropout(x_19, 0.0, False, False)
        x_19 = None
        edge_index_i_3 = l_edge_index_[1]
        edge_index_j_3 = l_edge_index_[0]
        x_i_3 = x_20.index_select(-2, edge_index_i_3)
        x_j_3 = x_20.index_select(-2, edge_index_j_3)
        x_20 = edge_index_j_3 = None
        sub_3 = x_j_3 - x_i_3
        x_j_3 = None
        cat_3 = torch.cat([x_i_3, sub_3], dim=-1)
        x_i_3 = sub_3 = None
        x_21 = torch._C._nn.linear(
            cat_3,
            l_self_modules_convs_modules_3_modules_nn_modules_lins_modules_0_parameters_weight_,
            l_self_modules_convs_modules_3_modules_nn_modules_lins_modules_0_parameters_bias_,
        )
        cat_3 = l_self_modules_convs_modules_3_modules_nn_modules_lins_modules_0_parameters_weight_ = l_self_modules_convs_modules_3_modules_nn_modules_lins_modules_0_parameters_bias_ = (None)
        x_22 = torch.nn.functional.relu(x_21, inplace=False)
        x_21 = None
        x_23 = torch.nn.functional.dropout(x_22, p=0.0, training=False)
        x_22 = None
        x_24 = torch._C._nn.linear(
            x_23,
            l_self_modules_convs_modules_3_modules_nn_modules_lins_modules_1_parameters_weight_,
            l_self_modules_convs_modules_3_modules_nn_modules_lins_modules_1_parameters_bias_,
        )
        x_23 = l_self_modules_convs_modules_3_modules_nn_modules_lins_modules_1_parameters_weight_ = l_self_modules_convs_modules_3_modules_nn_modules_lins_modules_1_parameters_bias_ = (None)
        x_25 = torch.nn.functional.dropout(x_24, p=0.0, training=False)
        x_24 = None
        view_3 = edge_index_i_3.view((-1, 1))
        edge_index_i_3 = None
        index_3 = view_3.expand_as(x_25)
        view_3 = None
        new_zeros_3 = x_25.new_zeros((1000, 256))
        out_3 = new_zeros_3.scatter_reduce_(
            0, index_3, x_25, reduce="amax", include_self=False
        )
        new_zeros_3 = index_3 = x_25 = None
        x_26 = torch.nn.functional.relu(out_3, inplace=False)
        out_3 = None
        x_27 = torch.nn.functional.dropout(x_26, 0.0, False, False)
        x_26 = None
        edge_index_i_4 = l_edge_index_[1]
        edge_index_j_4 = l_edge_index_[0]
        l_edge_index_ = None
        x_i_4 = x_27.index_select(-2, edge_index_i_4)
        x_j_4 = x_27.index_select(-2, edge_index_j_4)
        x_27 = edge_index_j_4 = None
        sub_4 = x_j_4 - x_i_4
        x_j_4 = None
        cat_4 = torch.cat([x_i_4, sub_4], dim=-1)
        x_i_4 = sub_4 = None
        x_28 = torch._C._nn.linear(
            cat_4,
            l_self_modules_convs_modules_4_modules_nn_modules_lins_modules_0_parameters_weight_,
            l_self_modules_convs_modules_4_modules_nn_modules_lins_modules_0_parameters_bias_,
        )
        cat_4 = l_self_modules_convs_modules_4_modules_nn_modules_lins_modules_0_parameters_weight_ = l_self_modules_convs_modules_4_modules_nn_modules_lins_modules_0_parameters_bias_ = (None)
        x_29 = torch.nn.functional.relu(x_28, inplace=False)
        x_28 = None
        x_30 = torch.nn.functional.dropout(x_29, p=0.0, training=False)
        x_29 = None
        x_31 = torch._C._nn.linear(
            x_30,
            l_self_modules_convs_modules_4_modules_nn_modules_lins_modules_1_parameters_weight_,
            l_self_modules_convs_modules_4_modules_nn_modules_lins_modules_1_parameters_bias_,
        )
        x_30 = l_self_modules_convs_modules_4_modules_nn_modules_lins_modules_1_parameters_weight_ = l_self_modules_convs_modules_4_modules_nn_modules_lins_modules_1_parameters_bias_ = (None)
        x_32 = torch.nn.functional.dropout(x_31, p=0.0, training=False)
        x_31 = None
        view_4 = edge_index_i_4.view((-1, 1))
        edge_index_i_4 = None
        index_4 = view_4.expand_as(x_32)
        view_4 = None
        new_zeros_4 = x_32.new_zeros((1000, 10))
        out_4 = new_zeros_4.scatter_reduce_(
            0, index_4, x_32, reduce="amax", include_self=False
        )
        new_zeros_4 = index_4 = x_32 = None
        return (out_4,)
