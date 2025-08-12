import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_x_: torch.Tensor,
        L_pos_edge_index_: torch.Tensor,
        L_self_modules_conv1_modules_lin_pos_l_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv1_modules_lin_pos_r_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv1_modules_lin_pos_r_parameters_bias_: torch.nn.parameter.Parameter,
        L_neg_edge_index_: torch.Tensor,
        L_self_modules_conv1_modules_lin_neg_l_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv1_modules_lin_neg_r_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv1_modules_lin_neg_r_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_0_modules_lin_pos_l_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_0_modules_lin_pos_r_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_0_modules_lin_pos_r_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_0_modules_lin_neg_l_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_0_modules_lin_neg_r_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_0_modules_lin_neg_r_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_x_ = L_x_
        l_pos_edge_index_ = L_pos_edge_index_
        l_self_modules_conv1_modules_lin_pos_l_parameters_weight_ = (
            L_self_modules_conv1_modules_lin_pos_l_parameters_weight_
        )
        l_self_modules_conv1_modules_lin_pos_r_parameters_weight_ = (
            L_self_modules_conv1_modules_lin_pos_r_parameters_weight_
        )
        l_self_modules_conv1_modules_lin_pos_r_parameters_bias_ = (
            L_self_modules_conv1_modules_lin_pos_r_parameters_bias_
        )
        l_neg_edge_index_ = L_neg_edge_index_
        l_self_modules_conv1_modules_lin_neg_l_parameters_weight_ = (
            L_self_modules_conv1_modules_lin_neg_l_parameters_weight_
        )
        l_self_modules_conv1_modules_lin_neg_r_parameters_weight_ = (
            L_self_modules_conv1_modules_lin_neg_r_parameters_weight_
        )
        l_self_modules_conv1_modules_lin_neg_r_parameters_bias_ = (
            L_self_modules_conv1_modules_lin_neg_r_parameters_bias_
        )
        l_self_modules_convs_modules_0_modules_lin_pos_l_parameters_weight_ = (
            L_self_modules_convs_modules_0_modules_lin_pos_l_parameters_weight_
        )
        l_self_modules_convs_modules_0_modules_lin_pos_r_parameters_weight_ = (
            L_self_modules_convs_modules_0_modules_lin_pos_r_parameters_weight_
        )
        l_self_modules_convs_modules_0_modules_lin_pos_r_parameters_bias_ = (
            L_self_modules_convs_modules_0_modules_lin_pos_r_parameters_bias_
        )
        l_self_modules_convs_modules_0_modules_lin_neg_l_parameters_weight_ = (
            L_self_modules_convs_modules_0_modules_lin_neg_l_parameters_weight_
        )
        l_self_modules_convs_modules_0_modules_lin_neg_r_parameters_weight_ = (
            L_self_modules_convs_modules_0_modules_lin_neg_r_parameters_weight_
        )
        l_self_modules_convs_modules_0_modules_lin_neg_r_parameters_bias_ = (
            L_self_modules_convs_modules_0_modules_lin_neg_r_parameters_bias_
        )
        edge_index_i = l_pos_edge_index_[1]
        edge_index_j = l_pos_edge_index_[0]
        x_j = l_x_.index_select(-2, edge_index_j)
        edge_index_j = None
        count = x_j.new_zeros(1000)
        new_ones = x_j.new_ones(100)
        scatter_add_ = count.scatter_add_(0, edge_index_i, new_ones)
        new_ones = scatter_add_ = None
        count_1 = count.clamp(min=1)
        count = None
        view = edge_index_i.view((-1, 1))
        edge_index_i = None
        index = view.expand_as(x_j)
        view = None
        new_zeros_1 = x_j.new_zeros((1000, 32))
        out = new_zeros_1.scatter_add_(0, index, x_j)
        new_zeros_1 = index = x_j = None
        view_1 = count_1.view((-1, 1))
        count_1 = None
        expand_as_1 = view_1.expand_as(out)
        view_1 = None
        out_1 = out / expand_as_1
        out = expand_as_1 = None
        out_pos = torch._C._nn.linear(
            out_1, l_self_modules_conv1_modules_lin_pos_l_parameters_weight_, None
        )
        out_1 = l_self_modules_conv1_modules_lin_pos_l_parameters_weight_ = None
        linear_1 = torch._C._nn.linear(
            l_x_,
            l_self_modules_conv1_modules_lin_pos_r_parameters_weight_,
            l_self_modules_conv1_modules_lin_pos_r_parameters_bias_,
        )
        l_self_modules_conv1_modules_lin_pos_r_parameters_weight_ = (
            l_self_modules_conv1_modules_lin_pos_r_parameters_bias_
        ) = None
        out_pos_1 = out_pos + linear_1
        out_pos = linear_1 = None
        edge_index_i_1 = l_neg_edge_index_[1]
        edge_index_j_1 = l_neg_edge_index_[0]
        x_j_1 = l_x_.index_select(-2, edge_index_j_1)
        edge_index_j_1 = None
        count_2 = x_j_1.new_zeros(1000)
        new_ones_1 = x_j_1.new_ones(100)
        scatter_add__2 = count_2.scatter_add_(0, edge_index_i_1, new_ones_1)
        new_ones_1 = scatter_add__2 = None
        count_3 = count_2.clamp(min=1)
        count_2 = None
        view_2 = edge_index_i_1.view((-1, 1))
        edge_index_i_1 = None
        index_1 = view_2.expand_as(x_j_1)
        view_2 = None
        new_zeros_3 = x_j_1.new_zeros((1000, 32))
        out_2 = new_zeros_3.scatter_add_(0, index_1, x_j_1)
        new_zeros_3 = index_1 = x_j_1 = None
        view_3 = count_3.view((-1, 1))
        count_3 = None
        expand_as_3 = view_3.expand_as(out_2)
        view_3 = None
        out_3 = out_2 / expand_as_3
        out_2 = expand_as_3 = None
        out_neg = torch._C._nn.linear(
            out_3, l_self_modules_conv1_modules_lin_neg_l_parameters_weight_, None
        )
        out_3 = l_self_modules_conv1_modules_lin_neg_l_parameters_weight_ = None
        linear_3 = torch._C._nn.linear(
            l_x_,
            l_self_modules_conv1_modules_lin_neg_r_parameters_weight_,
            l_self_modules_conv1_modules_lin_neg_r_parameters_bias_,
        )
        l_x_ = (
            l_self_modules_conv1_modules_lin_neg_r_parameters_weight_
        ) = l_self_modules_conv1_modules_lin_neg_r_parameters_bias_ = None
        out_neg_1 = out_neg + linear_3
        out_neg = linear_3 = None
        cat = torch.cat([out_pos_1, out_neg_1], dim=-1)
        out_pos_1 = out_neg_1 = None
        z = torch.nn.functional.relu(cat)
        cat = None
        _x_0 = z[(Ellipsis, slice(None, 32, None))]
        edge_index_i_2 = l_pos_edge_index_[1]
        edge_index_j_2 = l_pos_edge_index_[0]
        x_j_2 = _x_0.index_select(-2, edge_index_j_2)
        _x_0 = edge_index_j_2 = None
        count_4 = x_j_2.new_zeros(1000)
        new_ones_2 = x_j_2.new_ones(100)
        scatter_add__4 = count_4.scatter_add_(0, edge_index_i_2, new_ones_2)
        new_ones_2 = scatter_add__4 = None
        count_5 = count_4.clamp(min=1)
        count_4 = None
        view_4 = edge_index_i_2.view((-1, 1))
        edge_index_i_2 = None
        index_2 = view_4.expand_as(x_j_2)
        view_4 = None
        new_zeros_5 = x_j_2.new_zeros((1000, 32))
        out_4 = new_zeros_5.scatter_add_(0, index_2, x_j_2)
        new_zeros_5 = index_2 = x_j_2 = None
        view_5 = count_5.view((-1, 1))
        count_5 = None
        expand_as_5 = view_5.expand_as(out_4)
        view_5 = None
        out_5 = out_4 / expand_as_5
        out_4 = expand_as_5 = None
        _x_3 = z[(Ellipsis, slice(32, None, None))]
        edge_index_i_3 = l_neg_edge_index_[1]
        edge_index_j_3 = l_neg_edge_index_[0]
        x_j_3 = _x_3.index_select(-2, edge_index_j_3)
        _x_3 = edge_index_j_3 = None
        count_6 = x_j_3.new_zeros(1000)
        new_ones_3 = x_j_3.new_ones(100)
        scatter_add__6 = count_6.scatter_add_(0, edge_index_i_3, new_ones_3)
        new_ones_3 = scatter_add__6 = None
        count_7 = count_6.clamp(min=1)
        count_6 = None
        view_6 = edge_index_i_3.view((-1, 1))
        edge_index_i_3 = None
        index_3 = view_6.expand_as(x_j_3)
        view_6 = None
        new_zeros_7 = x_j_3.new_zeros((1000, 32))
        out_6 = new_zeros_7.scatter_add_(0, index_3, x_j_3)
        new_zeros_7 = index_3 = x_j_3 = None
        view_7 = count_7.view((-1, 1))
        count_7 = None
        expand_as_7 = view_7.expand_as(out_6)
        view_7 = None
        out_7 = out_6 / expand_as_7
        out_6 = expand_as_7 = None
        out_pos_2 = torch.cat([out_5, out_7], dim=-1)
        out_5 = out_7 = None
        out_pos_3 = torch._C._nn.linear(
            out_pos_2,
            l_self_modules_convs_modules_0_modules_lin_pos_l_parameters_weight_,
            None,
        )
        out_pos_2 = (
            l_self_modules_convs_modules_0_modules_lin_pos_l_parameters_weight_
        ) = None
        getitem_12 = z[(Ellipsis, slice(None, 32, None))]
        linear_5 = torch._C._nn.linear(
            getitem_12,
            l_self_modules_convs_modules_0_modules_lin_pos_r_parameters_weight_,
            l_self_modules_convs_modules_0_modules_lin_pos_r_parameters_bias_,
        )
        getitem_12 = (
            l_self_modules_convs_modules_0_modules_lin_pos_r_parameters_weight_
        ) = l_self_modules_convs_modules_0_modules_lin_pos_r_parameters_bias_ = None
        out_pos_4 = out_pos_3 + linear_5
        out_pos_3 = linear_5 = None
        _x_5 = z[(Ellipsis, slice(32, None, None))]
        edge_index_i_4 = l_pos_edge_index_[1]
        edge_index_j_4 = l_pos_edge_index_[0]
        l_pos_edge_index_ = None
        x_j_4 = _x_5.index_select(-2, edge_index_j_4)
        _x_5 = edge_index_j_4 = None
        count_8 = x_j_4.new_zeros(1000)
        new_ones_4 = x_j_4.new_ones(100)
        scatter_add__8 = count_8.scatter_add_(0, edge_index_i_4, new_ones_4)
        new_ones_4 = scatter_add__8 = None
        count_9 = count_8.clamp(min=1)
        count_8 = None
        view_8 = edge_index_i_4.view((-1, 1))
        edge_index_i_4 = None
        index_4 = view_8.expand_as(x_j_4)
        view_8 = None
        new_zeros_9 = x_j_4.new_zeros((1000, 32))
        out_8 = new_zeros_9.scatter_add_(0, index_4, x_j_4)
        new_zeros_9 = index_4 = x_j_4 = None
        view_9 = count_9.view((-1, 1))
        count_9 = None
        expand_as_9 = view_9.expand_as(out_8)
        view_9 = None
        out_9 = out_8 / expand_as_9
        out_8 = expand_as_9 = None
        _x_7 = z[(Ellipsis, slice(None, 32, None))]
        edge_index_i_5 = l_neg_edge_index_[1]
        edge_index_j_5 = l_neg_edge_index_[0]
        l_neg_edge_index_ = None
        x_j_5 = _x_7.index_select(-2, edge_index_j_5)
        _x_7 = edge_index_j_5 = None
        count_10 = x_j_5.new_zeros(1000)
        new_ones_5 = x_j_5.new_ones(100)
        scatter_add__10 = count_10.scatter_add_(0, edge_index_i_5, new_ones_5)
        new_ones_5 = scatter_add__10 = None
        count_11 = count_10.clamp(min=1)
        count_10 = None
        view_10 = edge_index_i_5.view((-1, 1))
        edge_index_i_5 = None
        index_5 = view_10.expand_as(x_j_5)
        view_10 = None
        new_zeros_11 = x_j_5.new_zeros((1000, 32))
        out_10 = new_zeros_11.scatter_add_(0, index_5, x_j_5)
        new_zeros_11 = index_5 = x_j_5 = None
        view_11 = count_11.view((-1, 1))
        count_11 = None
        expand_as_11 = view_11.expand_as(out_10)
        view_11 = None
        out_11 = out_10 / expand_as_11
        out_10 = expand_as_11 = None
        out_neg_2 = torch.cat([out_9, out_11], dim=-1)
        out_9 = out_11 = None
        out_neg_3 = torch._C._nn.linear(
            out_neg_2,
            l_self_modules_convs_modules_0_modules_lin_neg_l_parameters_weight_,
            None,
        )
        out_neg_2 = (
            l_self_modules_convs_modules_0_modules_lin_neg_l_parameters_weight_
        ) = None
        getitem_21 = z[(Ellipsis, slice(32, None, None))]
        z = None
        linear_7 = torch._C._nn.linear(
            getitem_21,
            l_self_modules_convs_modules_0_modules_lin_neg_r_parameters_weight_,
            l_self_modules_convs_modules_0_modules_lin_neg_r_parameters_bias_,
        )
        getitem_21 = (
            l_self_modules_convs_modules_0_modules_lin_neg_r_parameters_weight_
        ) = l_self_modules_convs_modules_0_modules_lin_neg_r_parameters_bias_ = None
        out_neg_4 = out_neg_3 + linear_7
        out_neg_3 = linear_7 = None
        cat_3 = torch.cat([out_pos_4, out_neg_4], dim=-1)
        out_pos_4 = out_neg_4 = None
        z_1 = torch.nn.functional.relu(cat_3)
        cat_3 = None
        return (z_1,)
