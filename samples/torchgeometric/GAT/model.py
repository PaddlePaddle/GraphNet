import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_x_: torch.Tensor,
        L_self_modules_convs_modules_0_modules_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_0_parameters_att_src_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_0_parameters_att_dst_: torch.nn.parameter.Parameter,
        L_edge_index_: torch.Tensor,
        L_self_modules_convs_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_1_modules_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_1_parameters_att_src_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_1_parameters_att_dst_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_2_modules_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_2_parameters_att_src_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_2_parameters_att_dst_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_3_modules_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_3_parameters_att_src_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_3_parameters_att_dst_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_4_modules_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_4_parameters_att_src_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_4_parameters_att_dst_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_x_ = L_x_
        l_self_modules_convs_modules_0_modules_lin_parameters_weight_ = (
            L_self_modules_convs_modules_0_modules_lin_parameters_weight_
        )
        l_self_modules_convs_modules_0_parameters_att_src_ = (
            L_self_modules_convs_modules_0_parameters_att_src_
        )
        l_self_modules_convs_modules_0_parameters_att_dst_ = (
            L_self_modules_convs_modules_0_parameters_att_dst_
        )
        l_edge_index_ = L_edge_index_
        l_self_modules_convs_modules_0_parameters_bias_ = (
            L_self_modules_convs_modules_0_parameters_bias_
        )
        l_self_modules_convs_modules_1_modules_lin_parameters_weight_ = (
            L_self_modules_convs_modules_1_modules_lin_parameters_weight_
        )
        l_self_modules_convs_modules_1_parameters_att_src_ = (
            L_self_modules_convs_modules_1_parameters_att_src_
        )
        l_self_modules_convs_modules_1_parameters_att_dst_ = (
            L_self_modules_convs_modules_1_parameters_att_dst_
        )
        l_self_modules_convs_modules_1_parameters_bias_ = (
            L_self_modules_convs_modules_1_parameters_bias_
        )
        l_self_modules_convs_modules_2_modules_lin_parameters_weight_ = (
            L_self_modules_convs_modules_2_modules_lin_parameters_weight_
        )
        l_self_modules_convs_modules_2_parameters_att_src_ = (
            L_self_modules_convs_modules_2_parameters_att_src_
        )
        l_self_modules_convs_modules_2_parameters_att_dst_ = (
            L_self_modules_convs_modules_2_parameters_att_dst_
        )
        l_self_modules_convs_modules_2_parameters_bias_ = (
            L_self_modules_convs_modules_2_parameters_bias_
        )
        l_self_modules_convs_modules_3_modules_lin_parameters_weight_ = (
            L_self_modules_convs_modules_3_modules_lin_parameters_weight_
        )
        l_self_modules_convs_modules_3_parameters_att_src_ = (
            L_self_modules_convs_modules_3_parameters_att_src_
        )
        l_self_modules_convs_modules_3_parameters_att_dst_ = (
            L_self_modules_convs_modules_3_parameters_att_dst_
        )
        l_self_modules_convs_modules_3_parameters_bias_ = (
            L_self_modules_convs_modules_3_parameters_bias_
        )
        l_self_modules_convs_modules_4_modules_lin_parameters_weight_ = (
            L_self_modules_convs_modules_4_modules_lin_parameters_weight_
        )
        l_self_modules_convs_modules_4_parameters_att_src_ = (
            L_self_modules_convs_modules_4_parameters_att_src_
        )
        l_self_modules_convs_modules_4_parameters_att_dst_ = (
            L_self_modules_convs_modules_4_parameters_att_dst_
        )
        l_self_modules_convs_modules_4_parameters_bias_ = (
            L_self_modules_convs_modules_4_parameters_bias_
        )
        linear = torch._C._nn.linear(
            l_x_, l_self_modules_convs_modules_0_modules_lin_parameters_weight_, None
        )
        l_x_ = l_self_modules_convs_modules_0_modules_lin_parameters_weight_ = None
        x_src = linear.view(-1, 4, 64)
        linear = None
        mul = x_src * l_self_modules_convs_modules_0_parameters_att_src_
        l_self_modules_convs_modules_0_parameters_att_src_ = None
        alpha_src = mul.sum(dim=-1)
        mul = None
        mul_1 = x_src * l_self_modules_convs_modules_0_parameters_att_dst_
        l_self_modules_convs_modules_0_parameters_att_dst_ = None
        alpha_dst = mul_1.sum(-1)
        mul_1 = None
        getitem = l_edge_index_[0]
        getitem_1 = l_edge_index_[1]
        mask = getitem != getitem_1
        getitem = getitem_1 = None
        edge_index = l_edge_index_[(slice(None, None, None), mask)]
        mask = None
        sym_size_int = torch.ops.aten.sym_size.int(edge_index, 1)
        _check_is_size = torch._check_is_size(sym_size_int)
        _check_is_size = None
        ge = sym_size_int >= 0
        _assert_scalar_default = torch.ops.aten._assert_scalar.default(
            ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'"
        )
        ge = _assert_scalar_default = None
        le = sym_size_int <= 100
        _assert_scalar_default_1 = torch.ops.aten._assert_scalar.default(
            le, "Runtime assertion failed for expression u0 <= 100 on node 'le'"
        )
        le = _assert_scalar_default_1 = None
        arange = torch.arange(0, 1000, device=device(type="cpu"))
        view_1 = arange.view(1, -1)
        arange = None
        loop_index = view_1.repeat(2, 1)
        view_1 = None
        full_edge_index = torch.cat([edge_index, loop_index], dim=1)
        edge_index = loop_index = None
        edge_index_i = full_edge_index[1]
        edge_index_j = full_edge_index[0]
        alpha_j = alpha_src.index_select(0, edge_index_j)
        alpha_src = edge_index_j = None
        alpha_i = alpha_dst.index_select(0, edge_index_i)
        alpha_dst = None
        alpha = alpha_j + alpha_i
        alpha_j = alpha_i = None
        sym_sum = torch.sym_sum([1000, sym_size_int])
        sym_size_int = sym_sum = None
        alpha_1 = torch.nn.functional.leaky_relu(alpha, 0.2)
        alpha = None
        detach = alpha_1.detach()
        view_2 = edge_index_i.view((-1, 1))
        index = view_2.expand_as(detach)
        view_2 = None
        new_zeros = detach.new_zeros((1000, 4))
        src_max = new_zeros.scatter_reduce_(
            0, index, detach, reduce="amax", include_self=False
        )
        new_zeros = index = detach = None
        index_select_2 = src_max.index_select(0, edge_index_i)
        src_max = None
        out = alpha_1 - index_select_2
        alpha_1 = index_select_2 = None
        out_1 = out.exp()
        out = None
        view_3 = edge_index_i.view((-1, 1))
        index_1 = view_3.expand_as(out_1)
        view_3 = None
        new_zeros_1 = out_1.new_zeros((1000, 4))
        scatter_add_ = new_zeros_1.scatter_add_(0, index_1, out_1)
        new_zeros_1 = index_1 = None
        out_sum = scatter_add_ + 1e-16
        scatter_add_ = None
        out_sum_1 = out_sum.index_select(0, edge_index_i)
        out_sum = edge_index_i = None
        alpha_2 = out_1 / out_sum_1
        out_1 = out_sum_1 = None
        alpha_3 = torch.nn.functional.dropout(alpha_2, p=0.0, training=False)
        alpha_2 = None
        edge_index_i_1 = full_edge_index[1]
        edge_index_j_1 = full_edge_index[0]
        full_edge_index = None
        x_j = x_src.index_select(0, edge_index_j_1)
        x_src = edge_index_j_1 = None
        unsqueeze = alpha_3.unsqueeze(-1)
        alpha_3 = None
        out_2 = unsqueeze * x_j
        unsqueeze = x_j = None
        view_4 = edge_index_i_1.view((-1, 1, 1))
        edge_index_i_1 = None
        index_2 = view_4.expand_as(out_2)
        view_4 = None
        new_zeros_2 = out_2.new_zeros((1000, 4, 64))
        out_3 = new_zeros_2.scatter_add_(0, index_2, out_2)
        new_zeros_2 = index_2 = out_2 = None
        out_4 = out_3.view(-1, 256)
        out_3 = None
        out_5 = out_4 + l_self_modules_convs_modules_0_parameters_bias_
        out_4 = l_self_modules_convs_modules_0_parameters_bias_ = None
        x = torch.nn.functional.relu(out_5, inplace=False)
        out_5 = None
        x_1 = torch.nn.functional.dropout(x, 0.0, False, False)
        x = None
        linear_1 = torch._C._nn.linear(
            x_1, l_self_modules_convs_modules_1_modules_lin_parameters_weight_, None
        )
        x_1 = l_self_modules_convs_modules_1_modules_lin_parameters_weight_ = None
        x_src_1 = linear_1.view(-1, 4, 64)
        linear_1 = None
        mul_3 = x_src_1 * l_self_modules_convs_modules_1_parameters_att_src_
        l_self_modules_convs_modules_1_parameters_att_src_ = None
        alpha_src_1 = mul_3.sum(dim=-1)
        mul_3 = None
        mul_4 = x_src_1 * l_self_modules_convs_modules_1_parameters_att_dst_
        l_self_modules_convs_modules_1_parameters_att_dst_ = None
        alpha_dst_1 = mul_4.sum(-1)
        mul_4 = None
        getitem_21 = l_edge_index_[0]
        getitem_22 = l_edge_index_[1]
        mask_1 = getitem_21 != getitem_22
        getitem_21 = getitem_22 = None
        edge_index_1 = l_edge_index_[(slice(None, None, None), mask_1)]
        mask_1 = None
        sym_size_int_1 = torch.ops.aten.sym_size.int(edge_index_1, 1)
        _check_is_size_1 = torch._check_is_size(sym_size_int_1)
        _check_is_size_1 = None
        ge_1 = sym_size_int_1 >= 0
        _assert_scalar_default_2 = torch.ops.aten._assert_scalar.default(
            ge_1, "Runtime assertion failed for expression u1 >= 0 on node 'ge_1'"
        )
        ge_1 = _assert_scalar_default_2 = None
        le_1 = sym_size_int_1 <= 100
        _assert_scalar_default_3 = torch.ops.aten._assert_scalar.default(
            le_1, "Runtime assertion failed for expression u1 <= 100 on node 'le_1'"
        )
        le_1 = _assert_scalar_default_3 = None
        arange_1 = torch.arange(0, 1000, device=device(type="cpu"))
        view_7 = arange_1.view(1, -1)
        arange_1 = None
        loop_index_1 = view_7.repeat(2, 1)
        view_7 = None
        full_edge_index_1 = torch.cat([edge_index_1, loop_index_1], dim=1)
        edge_index_1 = loop_index_1 = None
        edge_index_i_2 = full_edge_index_1[1]
        edge_index_j_2 = full_edge_index_1[0]
        alpha_j_1 = alpha_src_1.index_select(0, edge_index_j_2)
        alpha_src_1 = edge_index_j_2 = None
        alpha_i_1 = alpha_dst_1.index_select(0, edge_index_i_2)
        alpha_dst_1 = None
        alpha_4 = alpha_j_1 + alpha_i_1
        alpha_j_1 = alpha_i_1 = None
        sym_sum_1 = torch.sym_sum([1000, sym_size_int_1])
        sym_size_int_1 = sym_sum_1 = None
        alpha_5 = torch.nn.functional.leaky_relu(alpha_4, 0.2)
        alpha_4 = None
        detach_1 = alpha_5.detach()
        view_8 = edge_index_i_2.view((-1, 1))
        index_3 = view_8.expand_as(detach_1)
        view_8 = None
        new_zeros_3 = detach_1.new_zeros((1000, 4))
        src_max_1 = new_zeros_3.scatter_reduce_(
            0, index_3, detach_1, reduce="amax", include_self=False
        )
        new_zeros_3 = index_3 = detach_1 = None
        index_select_7 = src_max_1.index_select(0, edge_index_i_2)
        src_max_1 = None
        out_6 = alpha_5 - index_select_7
        alpha_5 = index_select_7 = None
        out_7 = out_6.exp()
        out_6 = None
        view_9 = edge_index_i_2.view((-1, 1))
        index_4 = view_9.expand_as(out_7)
        view_9 = None
        new_zeros_4 = out_7.new_zeros((1000, 4))
        scatter_add__2 = new_zeros_4.scatter_add_(0, index_4, out_7)
        new_zeros_4 = index_4 = None
        out_sum_2 = scatter_add__2 + 1e-16
        scatter_add__2 = None
        out_sum_3 = out_sum_2.index_select(0, edge_index_i_2)
        out_sum_2 = edge_index_i_2 = None
        alpha_6 = out_7 / out_sum_3
        out_7 = out_sum_3 = None
        alpha_7 = torch.nn.functional.dropout(alpha_6, p=0.0, training=False)
        alpha_6 = None
        edge_index_i_3 = full_edge_index_1[1]
        edge_index_j_3 = full_edge_index_1[0]
        full_edge_index_1 = None
        x_j_1 = x_src_1.index_select(0, edge_index_j_3)
        x_src_1 = edge_index_j_3 = None
        unsqueeze_1 = alpha_7.unsqueeze(-1)
        alpha_7 = None
        out_8 = unsqueeze_1 * x_j_1
        unsqueeze_1 = x_j_1 = None
        view_10 = edge_index_i_3.view((-1, 1, 1))
        edge_index_i_3 = None
        index_5 = view_10.expand_as(out_8)
        view_10 = None
        new_zeros_5 = out_8.new_zeros((1000, 4, 64))
        out_9 = new_zeros_5.scatter_add_(0, index_5, out_8)
        new_zeros_5 = index_5 = out_8 = None
        out_10 = out_9.view(-1, 256)
        out_9 = None
        out_11 = out_10 + l_self_modules_convs_modules_1_parameters_bias_
        out_10 = l_self_modules_convs_modules_1_parameters_bias_ = None
        x_2 = torch.nn.functional.relu(out_11, inplace=False)
        out_11 = None
        x_3 = torch.nn.functional.dropout(x_2, 0.0, False, False)
        x_2 = None
        linear_2 = torch._C._nn.linear(
            x_3, l_self_modules_convs_modules_2_modules_lin_parameters_weight_, None
        )
        x_3 = l_self_modules_convs_modules_2_modules_lin_parameters_weight_ = None
        x_src_2 = linear_2.view(-1, 4, 64)
        linear_2 = None
        mul_6 = x_src_2 * l_self_modules_convs_modules_2_parameters_att_src_
        l_self_modules_convs_modules_2_parameters_att_src_ = None
        alpha_src_2 = mul_6.sum(dim=-1)
        mul_6 = None
        mul_7 = x_src_2 * l_self_modules_convs_modules_2_parameters_att_dst_
        l_self_modules_convs_modules_2_parameters_att_dst_ = None
        alpha_dst_2 = mul_7.sum(-1)
        mul_7 = None
        getitem_42 = l_edge_index_[0]
        getitem_43 = l_edge_index_[1]
        mask_2 = getitem_42 != getitem_43
        getitem_42 = getitem_43 = None
        edge_index_2 = l_edge_index_[(slice(None, None, None), mask_2)]
        mask_2 = None
        sym_size_int_2 = torch.ops.aten.sym_size.int(edge_index_2, 1)
        _check_is_size_2 = torch._check_is_size(sym_size_int_2)
        _check_is_size_2 = None
        ge_2 = sym_size_int_2 >= 0
        _assert_scalar_default_4 = torch.ops.aten._assert_scalar.default(
            ge_2, "Runtime assertion failed for expression u2 >= 0 on node 'ge_2'"
        )
        ge_2 = _assert_scalar_default_4 = None
        le_2 = sym_size_int_2 <= 100
        _assert_scalar_default_5 = torch.ops.aten._assert_scalar.default(
            le_2, "Runtime assertion failed for expression u2 <= 100 on node 'le_2'"
        )
        le_2 = _assert_scalar_default_5 = None
        arange_2 = torch.arange(0, 1000, device=device(type="cpu"))
        view_13 = arange_2.view(1, -1)
        arange_2 = None
        loop_index_2 = view_13.repeat(2, 1)
        view_13 = None
        full_edge_index_2 = torch.cat([edge_index_2, loop_index_2], dim=1)
        edge_index_2 = loop_index_2 = None
        edge_index_i_4 = full_edge_index_2[1]
        edge_index_j_4 = full_edge_index_2[0]
        alpha_j_2 = alpha_src_2.index_select(0, edge_index_j_4)
        alpha_src_2 = edge_index_j_4 = None
        alpha_i_2 = alpha_dst_2.index_select(0, edge_index_i_4)
        alpha_dst_2 = None
        alpha_8 = alpha_j_2 + alpha_i_2
        alpha_j_2 = alpha_i_2 = None
        sym_sum_2 = torch.sym_sum([1000, sym_size_int_2])
        sym_size_int_2 = sym_sum_2 = None
        alpha_9 = torch.nn.functional.leaky_relu(alpha_8, 0.2)
        alpha_8 = None
        detach_2 = alpha_9.detach()
        view_14 = edge_index_i_4.view((-1, 1))
        index_6 = view_14.expand_as(detach_2)
        view_14 = None
        new_zeros_6 = detach_2.new_zeros((1000, 4))
        src_max_2 = new_zeros_6.scatter_reduce_(
            0, index_6, detach_2, reduce="amax", include_self=False
        )
        new_zeros_6 = index_6 = detach_2 = None
        index_select_12 = src_max_2.index_select(0, edge_index_i_4)
        src_max_2 = None
        out_12 = alpha_9 - index_select_12
        alpha_9 = index_select_12 = None
        out_13 = out_12.exp()
        out_12 = None
        view_15 = edge_index_i_4.view((-1, 1))
        index_7 = view_15.expand_as(out_13)
        view_15 = None
        new_zeros_7 = out_13.new_zeros((1000, 4))
        scatter_add__4 = new_zeros_7.scatter_add_(0, index_7, out_13)
        new_zeros_7 = index_7 = None
        out_sum_4 = scatter_add__4 + 1e-16
        scatter_add__4 = None
        out_sum_5 = out_sum_4.index_select(0, edge_index_i_4)
        out_sum_4 = edge_index_i_4 = None
        alpha_10 = out_13 / out_sum_5
        out_13 = out_sum_5 = None
        alpha_11 = torch.nn.functional.dropout(alpha_10, p=0.0, training=False)
        alpha_10 = None
        edge_index_i_5 = full_edge_index_2[1]
        edge_index_j_5 = full_edge_index_2[0]
        full_edge_index_2 = None
        x_j_2 = x_src_2.index_select(0, edge_index_j_5)
        x_src_2 = edge_index_j_5 = None
        unsqueeze_2 = alpha_11.unsqueeze(-1)
        alpha_11 = None
        out_14 = unsqueeze_2 * x_j_2
        unsqueeze_2 = x_j_2 = None
        view_16 = edge_index_i_5.view((-1, 1, 1))
        edge_index_i_5 = None
        index_8 = view_16.expand_as(out_14)
        view_16 = None
        new_zeros_8 = out_14.new_zeros((1000, 4, 64))
        out_15 = new_zeros_8.scatter_add_(0, index_8, out_14)
        new_zeros_8 = index_8 = out_14 = None
        out_16 = out_15.view(-1, 256)
        out_15 = None
        out_17 = out_16 + l_self_modules_convs_modules_2_parameters_bias_
        out_16 = l_self_modules_convs_modules_2_parameters_bias_ = None
        x_4 = torch.nn.functional.relu(out_17, inplace=False)
        out_17 = None
        x_5 = torch.nn.functional.dropout(x_4, 0.0, False, False)
        x_4 = None
        linear_3 = torch._C._nn.linear(
            x_5, l_self_modules_convs_modules_3_modules_lin_parameters_weight_, None
        )
        x_5 = l_self_modules_convs_modules_3_modules_lin_parameters_weight_ = None
        x_src_3 = linear_3.view(-1, 4, 64)
        linear_3 = None
        mul_9 = x_src_3 * l_self_modules_convs_modules_3_parameters_att_src_
        l_self_modules_convs_modules_3_parameters_att_src_ = None
        alpha_src_3 = mul_9.sum(dim=-1)
        mul_9 = None
        mul_10 = x_src_3 * l_self_modules_convs_modules_3_parameters_att_dst_
        l_self_modules_convs_modules_3_parameters_att_dst_ = None
        alpha_dst_3 = mul_10.sum(-1)
        mul_10 = None
        getitem_63 = l_edge_index_[0]
        getitem_64 = l_edge_index_[1]
        mask_3 = getitem_63 != getitem_64
        getitem_63 = getitem_64 = None
        edge_index_3 = l_edge_index_[(slice(None, None, None), mask_3)]
        mask_3 = None
        sym_size_int_3 = torch.ops.aten.sym_size.int(edge_index_3, 1)
        _check_is_size_3 = torch._check_is_size(sym_size_int_3)
        _check_is_size_3 = None
        ge_3 = sym_size_int_3 >= 0
        _assert_scalar_default_6 = torch.ops.aten._assert_scalar.default(
            ge_3, "Runtime assertion failed for expression u3 >= 0 on node 'ge_3'"
        )
        ge_3 = _assert_scalar_default_6 = None
        le_3 = sym_size_int_3 <= 100
        _assert_scalar_default_7 = torch.ops.aten._assert_scalar.default(
            le_3, "Runtime assertion failed for expression u3 <= 100 on node 'le_3'"
        )
        le_3 = _assert_scalar_default_7 = None
        arange_3 = torch.arange(0, 1000, device=device(type="cpu"))
        view_19 = arange_3.view(1, -1)
        arange_3 = None
        loop_index_3 = view_19.repeat(2, 1)
        view_19 = None
        full_edge_index_3 = torch.cat([edge_index_3, loop_index_3], dim=1)
        edge_index_3 = loop_index_3 = None
        edge_index_i_6 = full_edge_index_3[1]
        edge_index_j_6 = full_edge_index_3[0]
        alpha_j_3 = alpha_src_3.index_select(0, edge_index_j_6)
        alpha_src_3 = edge_index_j_6 = None
        alpha_i_3 = alpha_dst_3.index_select(0, edge_index_i_6)
        alpha_dst_3 = None
        alpha_12 = alpha_j_3 + alpha_i_3
        alpha_j_3 = alpha_i_3 = None
        sym_sum_3 = torch.sym_sum([1000, sym_size_int_3])
        sym_size_int_3 = sym_sum_3 = None
        alpha_13 = torch.nn.functional.leaky_relu(alpha_12, 0.2)
        alpha_12 = None
        detach_3 = alpha_13.detach()
        view_20 = edge_index_i_6.view((-1, 1))
        index_9 = view_20.expand_as(detach_3)
        view_20 = None
        new_zeros_9 = detach_3.new_zeros((1000, 4))
        src_max_3 = new_zeros_9.scatter_reduce_(
            0, index_9, detach_3, reduce="amax", include_self=False
        )
        new_zeros_9 = index_9 = detach_3 = None
        index_select_17 = src_max_3.index_select(0, edge_index_i_6)
        src_max_3 = None
        out_18 = alpha_13 - index_select_17
        alpha_13 = index_select_17 = None
        out_19 = out_18.exp()
        out_18 = None
        view_21 = edge_index_i_6.view((-1, 1))
        index_10 = view_21.expand_as(out_19)
        view_21 = None
        new_zeros_10 = out_19.new_zeros((1000, 4))
        scatter_add__6 = new_zeros_10.scatter_add_(0, index_10, out_19)
        new_zeros_10 = index_10 = None
        out_sum_6 = scatter_add__6 + 1e-16
        scatter_add__6 = None
        out_sum_7 = out_sum_6.index_select(0, edge_index_i_6)
        out_sum_6 = edge_index_i_6 = None
        alpha_14 = out_19 / out_sum_7
        out_19 = out_sum_7 = None
        alpha_15 = torch.nn.functional.dropout(alpha_14, p=0.0, training=False)
        alpha_14 = None
        edge_index_i_7 = full_edge_index_3[1]
        edge_index_j_7 = full_edge_index_3[0]
        full_edge_index_3 = None
        x_j_3 = x_src_3.index_select(0, edge_index_j_7)
        x_src_3 = edge_index_j_7 = None
        unsqueeze_3 = alpha_15.unsqueeze(-1)
        alpha_15 = None
        out_20 = unsqueeze_3 * x_j_3
        unsqueeze_3 = x_j_3 = None
        view_22 = edge_index_i_7.view((-1, 1, 1))
        edge_index_i_7 = None
        index_11 = view_22.expand_as(out_20)
        view_22 = None
        new_zeros_11 = out_20.new_zeros((1000, 4, 64))
        out_21 = new_zeros_11.scatter_add_(0, index_11, out_20)
        new_zeros_11 = index_11 = out_20 = None
        out_22 = out_21.view(-1, 256)
        out_21 = None
        out_23 = out_22 + l_self_modules_convs_modules_3_parameters_bias_
        out_22 = l_self_modules_convs_modules_3_parameters_bias_ = None
        x_6 = torch.nn.functional.relu(out_23, inplace=False)
        out_23 = None
        x_7 = torch.nn.functional.dropout(x_6, 0.0, False, False)
        x_6 = None
        linear_4 = torch._C._nn.linear(
            x_7, l_self_modules_convs_modules_4_modules_lin_parameters_weight_, None
        )
        x_7 = l_self_modules_convs_modules_4_modules_lin_parameters_weight_ = None
        x_src_4 = linear_4.view(-1, 4, 10)
        linear_4 = None
        mul_12 = x_src_4 * l_self_modules_convs_modules_4_parameters_att_src_
        l_self_modules_convs_modules_4_parameters_att_src_ = None
        alpha_src_4 = mul_12.sum(dim=-1)
        mul_12 = None
        mul_13 = x_src_4 * l_self_modules_convs_modules_4_parameters_att_dst_
        l_self_modules_convs_modules_4_parameters_att_dst_ = None
        alpha_dst_4 = mul_13.sum(-1)
        mul_13 = None
        getitem_84 = l_edge_index_[0]
        getitem_85 = l_edge_index_[1]
        mask_4 = getitem_84 != getitem_85
        getitem_84 = getitem_85 = None
        edge_index_4 = l_edge_index_[(slice(None, None, None), mask_4)]
        l_edge_index_ = mask_4 = None
        sym_size_int_4 = torch.ops.aten.sym_size.int(edge_index_4, 1)
        _check_is_size_4 = torch._check_is_size(sym_size_int_4)
        _check_is_size_4 = None
        ge_4 = sym_size_int_4 >= 0
        _assert_scalar_default_8 = torch.ops.aten._assert_scalar.default(
            ge_4, "Runtime assertion failed for expression u4 >= 0 on node 'ge_4'"
        )
        ge_4 = _assert_scalar_default_8 = None
        le_4 = sym_size_int_4 <= 100
        _assert_scalar_default_9 = torch.ops.aten._assert_scalar.default(
            le_4, "Runtime assertion failed for expression u4 <= 100 on node 'le_4'"
        )
        le_4 = _assert_scalar_default_9 = None
        arange_4 = torch.arange(0, 1000, device=device(type="cpu"))
        view_25 = arange_4.view(1, -1)
        arange_4 = None
        loop_index_4 = view_25.repeat(2, 1)
        view_25 = None
        full_edge_index_4 = torch.cat([edge_index_4, loop_index_4], dim=1)
        edge_index_4 = loop_index_4 = None
        edge_index_i_8 = full_edge_index_4[1]
        edge_index_j_8 = full_edge_index_4[0]
        alpha_j_4 = alpha_src_4.index_select(0, edge_index_j_8)
        alpha_src_4 = edge_index_j_8 = None
        alpha_i_4 = alpha_dst_4.index_select(0, edge_index_i_8)
        alpha_dst_4 = None
        alpha_16 = alpha_j_4 + alpha_i_4
        alpha_j_4 = alpha_i_4 = None
        sym_sum_4 = torch.sym_sum([1000, sym_size_int_4])
        sym_size_int_4 = sym_sum_4 = None
        alpha_17 = torch.nn.functional.leaky_relu(alpha_16, 0.2)
        alpha_16 = None
        detach_4 = alpha_17.detach()
        view_26 = edge_index_i_8.view((-1, 1))
        index_12 = view_26.expand_as(detach_4)
        view_26 = None
        new_zeros_12 = detach_4.new_zeros((1000, 4))
        src_max_4 = new_zeros_12.scatter_reduce_(
            0, index_12, detach_4, reduce="amax", include_self=False
        )
        new_zeros_12 = index_12 = detach_4 = None
        index_select_22 = src_max_4.index_select(0, edge_index_i_8)
        src_max_4 = None
        out_24 = alpha_17 - index_select_22
        alpha_17 = index_select_22 = None
        out_25 = out_24.exp()
        out_24 = None
        view_27 = edge_index_i_8.view((-1, 1))
        index_13 = view_27.expand_as(out_25)
        view_27 = None
        new_zeros_13 = out_25.new_zeros((1000, 4))
        scatter_add__8 = new_zeros_13.scatter_add_(0, index_13, out_25)
        new_zeros_13 = index_13 = None
        out_sum_8 = scatter_add__8 + 1e-16
        scatter_add__8 = None
        out_sum_9 = out_sum_8.index_select(0, edge_index_i_8)
        out_sum_8 = edge_index_i_8 = None
        alpha_18 = out_25 / out_sum_9
        out_25 = out_sum_9 = None
        alpha_19 = torch.nn.functional.dropout(alpha_18, p=0.0, training=False)
        alpha_18 = None
        edge_index_i_9 = full_edge_index_4[1]
        edge_index_j_9 = full_edge_index_4[0]
        full_edge_index_4 = None
        x_j_4 = x_src_4.index_select(0, edge_index_j_9)
        x_src_4 = edge_index_j_9 = None
        unsqueeze_4 = alpha_19.unsqueeze(-1)
        alpha_19 = None
        out_26 = unsqueeze_4 * x_j_4
        unsqueeze_4 = x_j_4 = None
        view_28 = edge_index_i_9.view((-1, 1, 1))
        edge_index_i_9 = None
        index_14 = view_28.expand_as(out_26)
        view_28 = None
        new_zeros_14 = out_26.new_zeros((1000, 4, 10))
        out_27 = new_zeros_14.scatter_add_(0, index_14, out_26)
        new_zeros_14 = index_14 = out_26 = None
        out_28 = out_27.mean(dim=1)
        out_27 = None
        out_29 = out_28 + l_self_modules_convs_modules_4_parameters_bias_
        out_28 = l_self_modules_convs_modules_4_parameters_bias_ = None
        return (out_29,)
