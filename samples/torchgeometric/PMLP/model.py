import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_edge_index_: torch.Tensor,
        L_self_modules_lins_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_x_: torch.Tensor,
        L_self_modules_lins_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_lins_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_lins_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_lins_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_lins_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_lins_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_lins_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_lins_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_lins_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_edge_index_ = L_edge_index_
        l_self_modules_lins_modules_0_parameters_weight_ = (
            L_self_modules_lins_modules_0_parameters_weight_
        )
        l_x_ = L_x_
        l_self_modules_lins_modules_0_parameters_bias_ = (
            L_self_modules_lins_modules_0_parameters_bias_
        )
        l_self_modules_lins_modules_1_parameters_weight_ = (
            L_self_modules_lins_modules_1_parameters_weight_
        )
        l_self_modules_lins_modules_1_parameters_bias_ = (
            L_self_modules_lins_modules_1_parameters_bias_
        )
        l_self_modules_lins_modules_2_parameters_weight_ = (
            L_self_modules_lins_modules_2_parameters_weight_
        )
        l_self_modules_lins_modules_2_parameters_bias_ = (
            L_self_modules_lins_modules_2_parameters_bias_
        )
        l_self_modules_lins_modules_3_parameters_weight_ = (
            L_self_modules_lins_modules_3_parameters_weight_
        )
        l_self_modules_lins_modules_3_parameters_bias_ = (
            L_self_modules_lins_modules_3_parameters_bias_
        )
        l_self_modules_lins_modules_4_parameters_weight_ = (
            L_self_modules_lins_modules_4_parameters_weight_
        )
        l_self_modules_lins_modules_4_parameters_bias_ = (
            L_self_modules_lins_modules_4_parameters_bias_
        )
        t = l_self_modules_lins_modules_0_parameters_weight_.t()
        l_self_modules_lins_modules_0_parameters_weight_ = None
        x = l_x_ @ t
        l_x_ = t = None
        arange = torch.arange(0, 1000, device=device(type="cpu"))
        view = arange.view(1, -1)
        arange = None
        loop_index = view.repeat(2, 1)
        view = None
        full_edge_index = torch.cat([l_edge_index_, loop_index], dim=1)
        loop_index = None
        edge_index_i = full_edge_index[1]
        edge_index_j = full_edge_index[0]
        full_edge_index = None
        x_j = x.index_select(-2, edge_index_j)
        x = edge_index_j = None
        count = x_j.new_zeros(1000)
        new_ones = x_j.new_ones(2000)
        scatter_add_ = count.scatter_add_(0, edge_index_i, new_ones)
        new_ones = scatter_add_ = None
        count_1 = count.clamp(min=1)
        count = None
        view_1 = edge_index_i.view((-1, 1))
        edge_index_i = None
        index = view_1.expand_as(x_j)
        view_1 = None
        new_zeros_1 = x_j.new_zeros((1000, 256))
        out = new_zeros_1.scatter_add_(0, index, x_j)
        new_zeros_1 = index = x_j = None
        view_2 = count_1.view((-1, 1))
        count_1 = None
        expand_as_1 = view_2.expand_as(out)
        view_2 = None
        out_1 = out / expand_as_1
        out = expand_as_1 = None
        x_1 = out_1 + l_self_modules_lins_modules_0_parameters_bias_
        out_1 = l_self_modules_lins_modules_0_parameters_bias_ = None
        x_2 = torch.nn.functional.batch_norm(
            x_1, None, None, None, None, True, 0.1, 1e-05
        )
        x_1 = None
        x_3 = x_2.relu()
        x_2 = None
        x_4 = torch.nn.functional.dropout(x_3, p=0.0, training=False)
        x_3 = None
        t_1 = l_self_modules_lins_modules_1_parameters_weight_.t()
        l_self_modules_lins_modules_1_parameters_weight_ = None
        x_5 = x_4 @ t_1
        x_4 = t_1 = None
        arange_1 = torch.arange(0, 1000, device=device(type="cpu"))
        view_3 = arange_1.view(1, -1)
        arange_1 = None
        loop_index_1 = view_3.repeat(2, 1)
        view_3 = None
        full_edge_index_1 = torch.cat([l_edge_index_, loop_index_1], dim=1)
        loop_index_1 = None
        edge_index_i_1 = full_edge_index_1[1]
        edge_index_j_1 = full_edge_index_1[0]
        full_edge_index_1 = None
        x_j_1 = x_5.index_select(-2, edge_index_j_1)
        x_5 = edge_index_j_1 = None
        count_2 = x_j_1.new_zeros(1000)
        new_ones_1 = x_j_1.new_ones(2000)
        scatter_add__2 = count_2.scatter_add_(0, edge_index_i_1, new_ones_1)
        new_ones_1 = scatter_add__2 = None
        count_3 = count_2.clamp(min=1)
        count_2 = None
        view_4 = edge_index_i_1.view((-1, 1))
        edge_index_i_1 = None
        index_1 = view_4.expand_as(x_j_1)
        view_4 = None
        new_zeros_3 = x_j_1.new_zeros((1000, 256))
        out_2 = new_zeros_3.scatter_add_(0, index_1, x_j_1)
        new_zeros_3 = index_1 = x_j_1 = None
        view_5 = count_3.view((-1, 1))
        count_3 = None
        expand_as_3 = view_5.expand_as(out_2)
        view_5 = None
        out_3 = out_2 / expand_as_3
        out_2 = expand_as_3 = None
        x_6 = out_3 + l_self_modules_lins_modules_1_parameters_bias_
        out_3 = l_self_modules_lins_modules_1_parameters_bias_ = None
        x_7 = torch.nn.functional.batch_norm(
            x_6, None, None, None, None, True, 0.1, 1e-05
        )
        x_6 = None
        x_8 = x_7.relu()
        x_7 = None
        x_9 = torch.nn.functional.dropout(x_8, p=0.0, training=False)
        x_8 = None
        t_2 = l_self_modules_lins_modules_2_parameters_weight_.t()
        l_self_modules_lins_modules_2_parameters_weight_ = None
        x_10 = x_9 @ t_2
        x_9 = t_2 = None
        arange_2 = torch.arange(0, 1000, device=device(type="cpu"))
        view_6 = arange_2.view(1, -1)
        arange_2 = None
        loop_index_2 = view_6.repeat(2, 1)
        view_6 = None
        full_edge_index_2 = torch.cat([l_edge_index_, loop_index_2], dim=1)
        loop_index_2 = None
        edge_index_i_2 = full_edge_index_2[1]
        edge_index_j_2 = full_edge_index_2[0]
        full_edge_index_2 = None
        x_j_2 = x_10.index_select(-2, edge_index_j_2)
        x_10 = edge_index_j_2 = None
        count_4 = x_j_2.new_zeros(1000)
        new_ones_2 = x_j_2.new_ones(2000)
        scatter_add__4 = count_4.scatter_add_(0, edge_index_i_2, new_ones_2)
        new_ones_2 = scatter_add__4 = None
        count_5 = count_4.clamp(min=1)
        count_4 = None
        view_7 = edge_index_i_2.view((-1, 1))
        edge_index_i_2 = None
        index_2 = view_7.expand_as(x_j_2)
        view_7 = None
        new_zeros_5 = x_j_2.new_zeros((1000, 256))
        out_4 = new_zeros_5.scatter_add_(0, index_2, x_j_2)
        new_zeros_5 = index_2 = x_j_2 = None
        view_8 = count_5.view((-1, 1))
        count_5 = None
        expand_as_5 = view_8.expand_as(out_4)
        view_8 = None
        out_5 = out_4 / expand_as_5
        out_4 = expand_as_5 = None
        x_11 = out_5 + l_self_modules_lins_modules_2_parameters_bias_
        out_5 = l_self_modules_lins_modules_2_parameters_bias_ = None
        x_12 = torch.nn.functional.batch_norm(
            x_11, None, None, None, None, True, 0.1, 1e-05
        )
        x_11 = None
        x_13 = x_12.relu()
        x_12 = None
        x_14 = torch.nn.functional.dropout(x_13, p=0.0, training=False)
        x_13 = None
        t_3 = l_self_modules_lins_modules_3_parameters_weight_.t()
        l_self_modules_lins_modules_3_parameters_weight_ = None
        x_15 = x_14 @ t_3
        x_14 = t_3 = None
        arange_3 = torch.arange(0, 1000, device=device(type="cpu"))
        view_9 = arange_3.view(1, -1)
        arange_3 = None
        loop_index_3 = view_9.repeat(2, 1)
        view_9 = None
        full_edge_index_3 = torch.cat([l_edge_index_, loop_index_3], dim=1)
        loop_index_3 = None
        edge_index_i_3 = full_edge_index_3[1]
        edge_index_j_3 = full_edge_index_3[0]
        full_edge_index_3 = None
        x_j_3 = x_15.index_select(-2, edge_index_j_3)
        x_15 = edge_index_j_3 = None
        count_6 = x_j_3.new_zeros(1000)
        new_ones_3 = x_j_3.new_ones(2000)
        scatter_add__6 = count_6.scatter_add_(0, edge_index_i_3, new_ones_3)
        new_ones_3 = scatter_add__6 = None
        count_7 = count_6.clamp(min=1)
        count_6 = None
        view_10 = edge_index_i_3.view((-1, 1))
        edge_index_i_3 = None
        index_3 = view_10.expand_as(x_j_3)
        view_10 = None
        new_zeros_7 = x_j_3.new_zeros((1000, 256))
        out_6 = new_zeros_7.scatter_add_(0, index_3, x_j_3)
        new_zeros_7 = index_3 = x_j_3 = None
        view_11 = count_7.view((-1, 1))
        count_7 = None
        expand_as_7 = view_11.expand_as(out_6)
        view_11 = None
        out_7 = out_6 / expand_as_7
        out_6 = expand_as_7 = None
        x_16 = out_7 + l_self_modules_lins_modules_3_parameters_bias_
        out_7 = l_self_modules_lins_modules_3_parameters_bias_ = None
        x_17 = torch.nn.functional.batch_norm(
            x_16, None, None, None, None, True, 0.1, 1e-05
        )
        x_16 = None
        x_18 = x_17.relu()
        x_17 = None
        x_19 = torch.nn.functional.dropout(x_18, p=0.0, training=False)
        x_18 = None
        t_4 = l_self_modules_lins_modules_4_parameters_weight_.t()
        l_self_modules_lins_modules_4_parameters_weight_ = None
        x_20 = x_19 @ t_4
        x_19 = t_4 = None
        arange_4 = torch.arange(0, 1000, device=device(type="cpu"))
        view_12 = arange_4.view(1, -1)
        arange_4 = None
        loop_index_4 = view_12.repeat(2, 1)
        view_12 = None
        full_edge_index_4 = torch.cat([l_edge_index_, loop_index_4], dim=1)
        l_edge_index_ = loop_index_4 = None
        edge_index_i_4 = full_edge_index_4[1]
        edge_index_j_4 = full_edge_index_4[0]
        full_edge_index_4 = None
        x_j_4 = x_20.index_select(-2, edge_index_j_4)
        x_20 = edge_index_j_4 = None
        count_8 = x_j_4.new_zeros(1000)
        new_ones_4 = x_j_4.new_ones(2000)
        scatter_add__8 = count_8.scatter_add_(0, edge_index_i_4, new_ones_4)
        new_ones_4 = scatter_add__8 = None
        count_9 = count_8.clamp(min=1)
        count_8 = None
        view_13 = edge_index_i_4.view((-1, 1))
        edge_index_i_4 = None
        index_4 = view_13.expand_as(x_j_4)
        view_13 = None
        new_zeros_9 = x_j_4.new_zeros((1000, 10))
        out_8 = new_zeros_9.scatter_add_(0, index_4, x_j_4)
        new_zeros_9 = index_4 = x_j_4 = None
        view_14 = count_9.view((-1, 1))
        count_9 = None
        expand_as_9 = view_14.expand_as(out_8)
        view_14 = None
        out_9 = out_8 / expand_as_9
        out_8 = expand_as_9 = None
        x_21 = out_9 + l_self_modules_lins_modules_4_parameters_bias_
        out_9 = l_self_modules_lins_modules_4_parameters_bias_ = None
        return (x_21,)
