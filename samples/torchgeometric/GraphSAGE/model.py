import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_x_: torch.Tensor,
        L_edge_index_: torch.Tensor,
        L_self_modules_convs_modules_0_modules_lin_l_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_0_modules_lin_l_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_0_modules_lin_r_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_1_modules_lin_l_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_1_modules_lin_l_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_1_modules_lin_r_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_2_modules_lin_l_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_2_modules_lin_l_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_2_modules_lin_r_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_3_modules_lin_l_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_3_modules_lin_l_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_3_modules_lin_r_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_4_modules_lin_l_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_4_modules_lin_l_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_4_modules_lin_r_parameters_weight_: torch.nn.parameter.Parameter,
    ):
        l_x_ = L_x_
        l_edge_index_ = L_edge_index_
        l_self_modules_convs_modules_0_modules_lin_l_parameters_weight_ = (
            L_self_modules_convs_modules_0_modules_lin_l_parameters_weight_
        )
        l_self_modules_convs_modules_0_modules_lin_l_parameters_bias_ = (
            L_self_modules_convs_modules_0_modules_lin_l_parameters_bias_
        )
        l_self_modules_convs_modules_0_modules_lin_r_parameters_weight_ = (
            L_self_modules_convs_modules_0_modules_lin_r_parameters_weight_
        )
        l_self_modules_convs_modules_1_modules_lin_l_parameters_weight_ = (
            L_self_modules_convs_modules_1_modules_lin_l_parameters_weight_
        )
        l_self_modules_convs_modules_1_modules_lin_l_parameters_bias_ = (
            L_self_modules_convs_modules_1_modules_lin_l_parameters_bias_
        )
        l_self_modules_convs_modules_1_modules_lin_r_parameters_weight_ = (
            L_self_modules_convs_modules_1_modules_lin_r_parameters_weight_
        )
        l_self_modules_convs_modules_2_modules_lin_l_parameters_weight_ = (
            L_self_modules_convs_modules_2_modules_lin_l_parameters_weight_
        )
        l_self_modules_convs_modules_2_modules_lin_l_parameters_bias_ = (
            L_self_modules_convs_modules_2_modules_lin_l_parameters_bias_
        )
        l_self_modules_convs_modules_2_modules_lin_r_parameters_weight_ = (
            L_self_modules_convs_modules_2_modules_lin_r_parameters_weight_
        )
        l_self_modules_convs_modules_3_modules_lin_l_parameters_weight_ = (
            L_self_modules_convs_modules_3_modules_lin_l_parameters_weight_
        )
        l_self_modules_convs_modules_3_modules_lin_l_parameters_bias_ = (
            L_self_modules_convs_modules_3_modules_lin_l_parameters_bias_
        )
        l_self_modules_convs_modules_3_modules_lin_r_parameters_weight_ = (
            L_self_modules_convs_modules_3_modules_lin_r_parameters_weight_
        )
        l_self_modules_convs_modules_4_modules_lin_l_parameters_weight_ = (
            L_self_modules_convs_modules_4_modules_lin_l_parameters_weight_
        )
        l_self_modules_convs_modules_4_modules_lin_l_parameters_bias_ = (
            L_self_modules_convs_modules_4_modules_lin_l_parameters_bias_
        )
        l_self_modules_convs_modules_4_modules_lin_r_parameters_weight_ = (
            L_self_modules_convs_modules_4_modules_lin_r_parameters_weight_
        )
        edge_index_i = l_edge_index_[1]
        edge_index_j = l_edge_index_[0]
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
        new_zeros_1 = x_j.new_zeros((1000, 128))
        out = new_zeros_1.scatter_add_(0, index, x_j)
        new_zeros_1 = index = x_j = None
        view_1 = count_1.view((-1, 1))
        count_1 = None
        expand_as_1 = view_1.expand_as(out)
        view_1 = None
        out_1 = out / expand_as_1
        out = expand_as_1 = None
        out_2 = torch._C._nn.linear(
            out_1,
            l_self_modules_convs_modules_0_modules_lin_l_parameters_weight_,
            l_self_modules_convs_modules_0_modules_lin_l_parameters_bias_,
        )
        out_1 = (
            l_self_modules_convs_modules_0_modules_lin_l_parameters_weight_
        ) = l_self_modules_convs_modules_0_modules_lin_l_parameters_bias_ = None
        linear_1 = torch._C._nn.linear(
            l_x_, l_self_modules_convs_modules_0_modules_lin_r_parameters_weight_, None
        )
        l_x_ = l_self_modules_convs_modules_0_modules_lin_r_parameters_weight_ = None
        out_3 = out_2 + linear_1
        out_2 = linear_1 = None
        x = torch.nn.functional.relu(out_3, inplace=False)
        out_3 = None
        x_1 = torch.nn.functional.dropout(x, 0.0, False, False)
        x = None
        edge_index_i_1 = l_edge_index_[1]
        edge_index_j_1 = l_edge_index_[0]
        x_j_1 = x_1.index_select(-2, edge_index_j_1)
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
        new_zeros_3 = x_j_1.new_zeros((1000, 256))
        out_4 = new_zeros_3.scatter_add_(0, index_1, x_j_1)
        new_zeros_3 = index_1 = x_j_1 = None
        view_3 = count_3.view((-1, 1))
        count_3 = None
        expand_as_3 = view_3.expand_as(out_4)
        view_3 = None
        out_5 = out_4 / expand_as_3
        out_4 = expand_as_3 = None
        out_6 = torch._C._nn.linear(
            out_5,
            l_self_modules_convs_modules_1_modules_lin_l_parameters_weight_,
            l_self_modules_convs_modules_1_modules_lin_l_parameters_bias_,
        )
        out_5 = (
            l_self_modules_convs_modules_1_modules_lin_l_parameters_weight_
        ) = l_self_modules_convs_modules_1_modules_lin_l_parameters_bias_ = None
        linear_3 = torch._C._nn.linear(
            x_1, l_self_modules_convs_modules_1_modules_lin_r_parameters_weight_, None
        )
        x_1 = l_self_modules_convs_modules_1_modules_lin_r_parameters_weight_ = None
        out_7 = out_6 + linear_3
        out_6 = linear_3 = None
        x_2 = torch.nn.functional.relu(out_7, inplace=False)
        out_7 = None
        x_3 = torch.nn.functional.dropout(x_2, 0.0, False, False)
        x_2 = None
        edge_index_i_2 = l_edge_index_[1]
        edge_index_j_2 = l_edge_index_[0]
        x_j_2 = x_3.index_select(-2, edge_index_j_2)
        edge_index_j_2 = None
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
        new_zeros_5 = x_j_2.new_zeros((1000, 256))
        out_8 = new_zeros_5.scatter_add_(0, index_2, x_j_2)
        new_zeros_5 = index_2 = x_j_2 = None
        view_5 = count_5.view((-1, 1))
        count_5 = None
        expand_as_5 = view_5.expand_as(out_8)
        view_5 = None
        out_9 = out_8 / expand_as_5
        out_8 = expand_as_5 = None
        out_10 = torch._C._nn.linear(
            out_9,
            l_self_modules_convs_modules_2_modules_lin_l_parameters_weight_,
            l_self_modules_convs_modules_2_modules_lin_l_parameters_bias_,
        )
        out_9 = (
            l_self_modules_convs_modules_2_modules_lin_l_parameters_weight_
        ) = l_self_modules_convs_modules_2_modules_lin_l_parameters_bias_ = None
        linear_5 = torch._C._nn.linear(
            x_3, l_self_modules_convs_modules_2_modules_lin_r_parameters_weight_, None
        )
        x_3 = l_self_modules_convs_modules_2_modules_lin_r_parameters_weight_ = None
        out_11 = out_10 + linear_5
        out_10 = linear_5 = None
        x_4 = torch.nn.functional.relu(out_11, inplace=False)
        out_11 = None
        x_5 = torch.nn.functional.dropout(x_4, 0.0, False, False)
        x_4 = None
        edge_index_i_3 = l_edge_index_[1]
        edge_index_j_3 = l_edge_index_[0]
        x_j_3 = x_5.index_select(-2, edge_index_j_3)
        edge_index_j_3 = None
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
        new_zeros_7 = x_j_3.new_zeros((1000, 256))
        out_12 = new_zeros_7.scatter_add_(0, index_3, x_j_3)
        new_zeros_7 = index_3 = x_j_3 = None
        view_7 = count_7.view((-1, 1))
        count_7 = None
        expand_as_7 = view_7.expand_as(out_12)
        view_7 = None
        out_13 = out_12 / expand_as_7
        out_12 = expand_as_7 = None
        out_14 = torch._C._nn.linear(
            out_13,
            l_self_modules_convs_modules_3_modules_lin_l_parameters_weight_,
            l_self_modules_convs_modules_3_modules_lin_l_parameters_bias_,
        )
        out_13 = (
            l_self_modules_convs_modules_3_modules_lin_l_parameters_weight_
        ) = l_self_modules_convs_modules_3_modules_lin_l_parameters_bias_ = None
        linear_7 = torch._C._nn.linear(
            x_5, l_self_modules_convs_modules_3_modules_lin_r_parameters_weight_, None
        )
        x_5 = l_self_modules_convs_modules_3_modules_lin_r_parameters_weight_ = None
        out_15 = out_14 + linear_7
        out_14 = linear_7 = None
        x_6 = torch.nn.functional.relu(out_15, inplace=False)
        out_15 = None
        x_7 = torch.nn.functional.dropout(x_6, 0.0, False, False)
        x_6 = None
        edge_index_i_4 = l_edge_index_[1]
        edge_index_j_4 = l_edge_index_[0]
        l_edge_index_ = None
        x_j_4 = x_7.index_select(-2, edge_index_j_4)
        edge_index_j_4 = None
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
        new_zeros_9 = x_j_4.new_zeros((1000, 256))
        out_16 = new_zeros_9.scatter_add_(0, index_4, x_j_4)
        new_zeros_9 = index_4 = x_j_4 = None
        view_9 = count_9.view((-1, 1))
        count_9 = None
        expand_as_9 = view_9.expand_as(out_16)
        view_9 = None
        out_17 = out_16 / expand_as_9
        out_16 = expand_as_9 = None
        out_18 = torch._C._nn.linear(
            out_17,
            l_self_modules_convs_modules_4_modules_lin_l_parameters_weight_,
            l_self_modules_convs_modules_4_modules_lin_l_parameters_bias_,
        )
        out_17 = (
            l_self_modules_convs_modules_4_modules_lin_l_parameters_weight_
        ) = l_self_modules_convs_modules_4_modules_lin_l_parameters_bias_ = None
        linear_9 = torch._C._nn.linear(
            x_7, l_self_modules_convs_modules_4_modules_lin_r_parameters_weight_, None
        )
        x_7 = l_self_modules_convs_modules_4_modules_lin_r_parameters_weight_ = None
        out_19 = out_18 + linear_9
        out_18 = linear_9 = None
        return (out_19,)
