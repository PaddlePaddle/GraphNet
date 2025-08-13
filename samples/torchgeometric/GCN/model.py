import torch

from torch import device

from torch import inf


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_x_: torch.Tensor,
        L_edge_index_: torch.Tensor,
        L_self_modules_convs_modules_0_modules_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_1_modules_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_2_modules_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_3_modules_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_4_modules_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_x_ = L_x_
        l_edge_index_ = L_edge_index_
        l_self_modules_convs_modules_0_modules_lin_parameters_weight_ = (
            L_self_modules_convs_modules_0_modules_lin_parameters_weight_
        )
        l_self_modules_convs_modules_0_parameters_bias_ = (
            L_self_modules_convs_modules_0_parameters_bias_
        )
        l_self_modules_convs_modules_1_modules_lin_parameters_weight_ = (
            L_self_modules_convs_modules_1_modules_lin_parameters_weight_
        )
        l_self_modules_convs_modules_1_parameters_bias_ = (
            L_self_modules_convs_modules_1_parameters_bias_
        )
        l_self_modules_convs_modules_2_modules_lin_parameters_weight_ = (
            L_self_modules_convs_modules_2_modules_lin_parameters_weight_
        )
        l_self_modules_convs_modules_2_parameters_bias_ = (
            L_self_modules_convs_modules_2_parameters_bias_
        )
        l_self_modules_convs_modules_3_modules_lin_parameters_weight_ = (
            L_self_modules_convs_modules_3_modules_lin_parameters_weight_
        )
        l_self_modules_convs_modules_3_parameters_bias_ = (
            L_self_modules_convs_modules_3_parameters_bias_
        )
        l_self_modules_convs_modules_4_modules_lin_parameters_weight_ = (
            L_self_modules_convs_modules_4_modules_lin_parameters_weight_
        )
        l_self_modules_convs_modules_4_parameters_bias_ = (
            L_self_modules_convs_modules_4_parameters_bias_
        )
        getitem = l_edge_index_[0]
        getitem_1 = l_edge_index_[1]
        mask = getitem != getitem_1
        getitem = getitem_1 = None
        arange = torch.arange(0, 1000, device=device(type="cpu"))
        view = arange.view(1, -1)
        arange = None
        loop_index = view.repeat(2, 1)
        view = None
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
        edge_index_1 = torch.cat([edge_index, loop_index], dim=1)
        edge_index = loop_index = None
        sym_sum = torch.sym_sum([1000, sym_size_int])
        sym_size_int = None
        edge_weight = torch.ones(
            (sym_sum,), dtype=torch.float32, device=device(type="cpu")
        )
        sym_sum = None
        row = edge_index_1[0]
        col = edge_index_1[1]
        view_1 = col.view((-1,))
        index = view_1.expand_as(edge_weight)
        view_1 = None
        new_zeros = edge_weight.new_zeros((1000,))
        deg = new_zeros.scatter_add_(0, index, edge_weight)
        new_zeros = index = None
        deg_inv_sqrt = deg.pow_(-0.5)
        deg = None
        eq = deg_inv_sqrt.__eq__(inf)
        masked_fill_ = deg_inv_sqrt.masked_fill_(eq, 0)
        eq = masked_fill_ = None
        getitem_7 = deg_inv_sqrt[row]
        row = None
        mul = getitem_7 * edge_weight
        getitem_7 = edge_weight = None
        getitem_8 = deg_inv_sqrt[col]
        deg_inv_sqrt = col = None
        edge_weight_1 = mul * getitem_8
        mul = getitem_8 = None
        x = torch._C._nn.linear(
            l_x_, l_self_modules_convs_modules_0_modules_lin_parameters_weight_, None
        )
        l_x_ = l_self_modules_convs_modules_0_modules_lin_parameters_weight_ = None
        edge_index_i = edge_index_1[1]
        edge_index_j = edge_index_1[0]
        edge_index_1 = None
        x_j = x.index_select(-2, edge_index_j)
        x = edge_index_j = None
        view_2 = edge_weight_1.view(-1, 1)
        edge_weight_1 = None
        out = view_2 * x_j
        view_2 = x_j = None
        view_3 = edge_index_i.view((-1, 1))
        edge_index_i = None
        index_1 = view_3.expand_as(out)
        view_3 = None
        new_zeros_1 = out.new_zeros((1000, 256))
        out_1 = new_zeros_1.scatter_add_(0, index_1, out)
        new_zeros_1 = index_1 = out = None
        out_2 = out_1 + l_self_modules_convs_modules_0_parameters_bias_
        out_1 = l_self_modules_convs_modules_0_parameters_bias_ = None
        x_1 = torch.nn.functional.relu(out_2, inplace=False)
        out_2 = None
        x_2 = torch.nn.functional.dropout(x_1, 0.0, False, False)
        x_1 = None
        getitem_15 = l_edge_index_[0]
        getitem_16 = l_edge_index_[1]
        mask_1 = getitem_15 != getitem_16
        getitem_15 = getitem_16 = None
        arange_1 = torch.arange(0, 1000, device=device(type="cpu"))
        view_4 = arange_1.view(1, -1)
        arange_1 = None
        loop_index_1 = view_4.repeat(2, 1)
        view_4 = None
        edge_index_2 = l_edge_index_[(slice(None, None, None), mask_1)]
        mask_1 = None
        sym_size_int_1 = torch.ops.aten.sym_size.int(edge_index_2, 1)
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
        edge_index_3 = torch.cat([edge_index_2, loop_index_1], dim=1)
        edge_index_2 = loop_index_1 = None
        sym_sum_1 = torch.sym_sum([1000, sym_size_int_1])
        sym_size_int_1 = None
        edge_weight_2 = torch.ones(
            (sym_sum_1,), dtype=torch.float32, device=device(type="cpu")
        )
        sym_sum_1 = None
        row_1 = edge_index_3[0]
        col_1 = edge_index_3[1]
        view_5 = col_1.view((-1,))
        index_2 = view_5.expand_as(edge_weight_2)
        view_5 = None
        new_zeros_2 = edge_weight_2.new_zeros((1000,))
        deg_1 = new_zeros_2.scatter_add_(0, index_2, edge_weight_2)
        new_zeros_2 = index_2 = None
        deg_inv_sqrt_1 = deg_1.pow_(-0.5)
        deg_1 = None
        eq_1 = deg_inv_sqrt_1.__eq__(inf)
        masked_fill__1 = deg_inv_sqrt_1.masked_fill_(eq_1, 0)
        eq_1 = masked_fill__1 = None
        getitem_22 = deg_inv_sqrt_1[row_1]
        row_1 = None
        mul_3 = getitem_22 * edge_weight_2
        getitem_22 = edge_weight_2 = None
        getitem_23 = deg_inv_sqrt_1[col_1]
        deg_inv_sqrt_1 = col_1 = None
        edge_weight_3 = mul_3 * getitem_23
        mul_3 = getitem_23 = None
        x_3 = torch._C._nn.linear(
            x_2, l_self_modules_convs_modules_1_modules_lin_parameters_weight_, None
        )
        x_2 = l_self_modules_convs_modules_1_modules_lin_parameters_weight_ = None
        edge_index_i_1 = edge_index_3[1]
        edge_index_j_1 = edge_index_3[0]
        edge_index_3 = None
        x_j_1 = x_3.index_select(-2, edge_index_j_1)
        x_3 = edge_index_j_1 = None
        view_6 = edge_weight_3.view(-1, 1)
        edge_weight_3 = None
        out_3 = view_6 * x_j_1
        view_6 = x_j_1 = None
        view_7 = edge_index_i_1.view((-1, 1))
        edge_index_i_1 = None
        index_3 = view_7.expand_as(out_3)
        view_7 = None
        new_zeros_3 = out_3.new_zeros((1000, 256))
        out_4 = new_zeros_3.scatter_add_(0, index_3, out_3)
        new_zeros_3 = index_3 = out_3 = None
        out_5 = out_4 + l_self_modules_convs_modules_1_parameters_bias_
        out_4 = l_self_modules_convs_modules_1_parameters_bias_ = None
        x_4 = torch.nn.functional.relu(out_5, inplace=False)
        out_5 = None
        x_5 = torch.nn.functional.dropout(x_4, 0.0, False, False)
        x_4 = None
        getitem_30 = l_edge_index_[0]
        getitem_31 = l_edge_index_[1]
        mask_2 = getitem_30 != getitem_31
        getitem_30 = getitem_31 = None
        arange_2 = torch.arange(0, 1000, device=device(type="cpu"))
        view_8 = arange_2.view(1, -1)
        arange_2 = None
        loop_index_2 = view_8.repeat(2, 1)
        view_8 = None
        edge_index_4 = l_edge_index_[(slice(None, None, None), mask_2)]
        mask_2 = None
        sym_size_int_2 = torch.ops.aten.sym_size.int(edge_index_4, 1)
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
        edge_index_5 = torch.cat([edge_index_4, loop_index_2], dim=1)
        edge_index_4 = loop_index_2 = None
        sym_sum_2 = torch.sym_sum([1000, sym_size_int_2])
        sym_size_int_2 = None
        edge_weight_4 = torch.ones(
            (sym_sum_2,), dtype=torch.float32, device=device(type="cpu")
        )
        sym_sum_2 = None
        row_2 = edge_index_5[0]
        col_2 = edge_index_5[1]
        view_9 = col_2.view((-1,))
        index_4 = view_9.expand_as(edge_weight_4)
        view_9 = None
        new_zeros_4 = edge_weight_4.new_zeros((1000,))
        deg_2 = new_zeros_4.scatter_add_(0, index_4, edge_weight_4)
        new_zeros_4 = index_4 = None
        deg_inv_sqrt_2 = deg_2.pow_(-0.5)
        deg_2 = None
        eq_2 = deg_inv_sqrt_2.__eq__(inf)
        masked_fill__2 = deg_inv_sqrt_2.masked_fill_(eq_2, 0)
        eq_2 = masked_fill__2 = None
        getitem_37 = deg_inv_sqrt_2[row_2]
        row_2 = None
        mul_6 = getitem_37 * edge_weight_4
        getitem_37 = edge_weight_4 = None
        getitem_38 = deg_inv_sqrt_2[col_2]
        deg_inv_sqrt_2 = col_2 = None
        edge_weight_5 = mul_6 * getitem_38
        mul_6 = getitem_38 = None
        x_6 = torch._C._nn.linear(
            x_5, l_self_modules_convs_modules_2_modules_lin_parameters_weight_, None
        )
        x_5 = l_self_modules_convs_modules_2_modules_lin_parameters_weight_ = None
        edge_index_i_2 = edge_index_5[1]
        edge_index_j_2 = edge_index_5[0]
        edge_index_5 = None
        x_j_2 = x_6.index_select(-2, edge_index_j_2)
        x_6 = edge_index_j_2 = None
        view_10 = edge_weight_5.view(-1, 1)
        edge_weight_5 = None
        out_6 = view_10 * x_j_2
        view_10 = x_j_2 = None
        view_11 = edge_index_i_2.view((-1, 1))
        edge_index_i_2 = None
        index_5 = view_11.expand_as(out_6)
        view_11 = None
        new_zeros_5 = out_6.new_zeros((1000, 256))
        out_7 = new_zeros_5.scatter_add_(0, index_5, out_6)
        new_zeros_5 = index_5 = out_6 = None
        out_8 = out_7 + l_self_modules_convs_modules_2_parameters_bias_
        out_7 = l_self_modules_convs_modules_2_parameters_bias_ = None
        x_7 = torch.nn.functional.relu(out_8, inplace=False)
        out_8 = None
        x_8 = torch.nn.functional.dropout(x_7, 0.0, False, False)
        x_7 = None
        getitem_45 = l_edge_index_[0]
        getitem_46 = l_edge_index_[1]
        mask_3 = getitem_45 != getitem_46
        getitem_45 = getitem_46 = None
        arange_3 = torch.arange(0, 1000, device=device(type="cpu"))
        view_12 = arange_3.view(1, -1)
        arange_3 = None
        loop_index_3 = view_12.repeat(2, 1)
        view_12 = None
        edge_index_6 = l_edge_index_[(slice(None, None, None), mask_3)]
        mask_3 = None
        sym_size_int_3 = torch.ops.aten.sym_size.int(edge_index_6, 1)
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
        edge_index_7 = torch.cat([edge_index_6, loop_index_3], dim=1)
        edge_index_6 = loop_index_3 = None
        sym_sum_3 = torch.sym_sum([1000, sym_size_int_3])
        sym_size_int_3 = None
        edge_weight_6 = torch.ones(
            (sym_sum_3,), dtype=torch.float32, device=device(type="cpu")
        )
        sym_sum_3 = None
        row_3 = edge_index_7[0]
        col_3 = edge_index_7[1]
        view_13 = col_3.view((-1,))
        index_6 = view_13.expand_as(edge_weight_6)
        view_13 = None
        new_zeros_6 = edge_weight_6.new_zeros((1000,))
        deg_3 = new_zeros_6.scatter_add_(0, index_6, edge_weight_6)
        new_zeros_6 = index_6 = None
        deg_inv_sqrt_3 = deg_3.pow_(-0.5)
        deg_3 = None
        eq_3 = deg_inv_sqrt_3.__eq__(inf)
        masked_fill__3 = deg_inv_sqrt_3.masked_fill_(eq_3, 0)
        eq_3 = masked_fill__3 = None
        getitem_52 = deg_inv_sqrt_3[row_3]
        row_3 = None
        mul_9 = getitem_52 * edge_weight_6
        getitem_52 = edge_weight_6 = None
        getitem_53 = deg_inv_sqrt_3[col_3]
        deg_inv_sqrt_3 = col_3 = None
        edge_weight_7 = mul_9 * getitem_53
        mul_9 = getitem_53 = None
        x_9 = torch._C._nn.linear(
            x_8, l_self_modules_convs_modules_3_modules_lin_parameters_weight_, None
        )
        x_8 = l_self_modules_convs_modules_3_modules_lin_parameters_weight_ = None
        edge_index_i_3 = edge_index_7[1]
        edge_index_j_3 = edge_index_7[0]
        edge_index_7 = None
        x_j_3 = x_9.index_select(-2, edge_index_j_3)
        x_9 = edge_index_j_3 = None
        view_14 = edge_weight_7.view(-1, 1)
        edge_weight_7 = None
        out_9 = view_14 * x_j_3
        view_14 = x_j_3 = None
        view_15 = edge_index_i_3.view((-1, 1))
        edge_index_i_3 = None
        index_7 = view_15.expand_as(out_9)
        view_15 = None
        new_zeros_7 = out_9.new_zeros((1000, 256))
        out_10 = new_zeros_7.scatter_add_(0, index_7, out_9)
        new_zeros_7 = index_7 = out_9 = None
        out_11 = out_10 + l_self_modules_convs_modules_3_parameters_bias_
        out_10 = l_self_modules_convs_modules_3_parameters_bias_ = None
        x_10 = torch.nn.functional.relu(out_11, inplace=False)
        out_11 = None
        x_11 = torch.nn.functional.dropout(x_10, 0.0, False, False)
        x_10 = None
        getitem_60 = l_edge_index_[0]
        getitem_61 = l_edge_index_[1]
        mask_4 = getitem_60 != getitem_61
        getitem_60 = getitem_61 = None
        arange_4 = torch.arange(0, 1000, device=device(type="cpu"))
        view_16 = arange_4.view(1, -1)
        arange_4 = None
        loop_index_4 = view_16.repeat(2, 1)
        view_16 = None
        edge_index_8 = l_edge_index_[(slice(None, None, None), mask_4)]
        l_edge_index_ = mask_4 = None
        sym_size_int_4 = torch.ops.aten.sym_size.int(edge_index_8, 1)
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
        edge_index_9 = torch.cat([edge_index_8, loop_index_4], dim=1)
        edge_index_8 = loop_index_4 = None
        sym_sum_4 = torch.sym_sum([1000, sym_size_int_4])
        sym_size_int_4 = None
        edge_weight_8 = torch.ones(
            (sym_sum_4,), dtype=torch.float32, device=device(type="cpu")
        )
        sym_sum_4 = None
        row_4 = edge_index_9[0]
        col_4 = edge_index_9[1]
        view_17 = col_4.view((-1,))
        index_8 = view_17.expand_as(edge_weight_8)
        view_17 = None
        new_zeros_8 = edge_weight_8.new_zeros((1000,))
        deg_4 = new_zeros_8.scatter_add_(0, index_8, edge_weight_8)
        new_zeros_8 = index_8 = None
        deg_inv_sqrt_4 = deg_4.pow_(-0.5)
        deg_4 = None
        eq_4 = deg_inv_sqrt_4.__eq__(inf)
        masked_fill__4 = deg_inv_sqrt_4.masked_fill_(eq_4, 0)
        eq_4 = masked_fill__4 = None
        getitem_67 = deg_inv_sqrt_4[row_4]
        row_4 = None
        mul_12 = getitem_67 * edge_weight_8
        getitem_67 = edge_weight_8 = None
        getitem_68 = deg_inv_sqrt_4[col_4]
        deg_inv_sqrt_4 = col_4 = None
        edge_weight_9 = mul_12 * getitem_68
        mul_12 = getitem_68 = None
        x_12 = torch._C._nn.linear(
            x_11, l_self_modules_convs_modules_4_modules_lin_parameters_weight_, None
        )
        x_11 = l_self_modules_convs_modules_4_modules_lin_parameters_weight_ = None
        edge_index_i_4 = edge_index_9[1]
        edge_index_j_4 = edge_index_9[0]
        edge_index_9 = None
        x_j_4 = x_12.index_select(-2, edge_index_j_4)
        x_12 = edge_index_j_4 = None
        view_18 = edge_weight_9.view(-1, 1)
        edge_weight_9 = None
        out_12 = view_18 * x_j_4
        view_18 = x_j_4 = None
        view_19 = edge_index_i_4.view((-1, 1))
        edge_index_i_4 = None
        index_9 = view_19.expand_as(out_12)
        view_19 = None
        new_zeros_9 = out_12.new_zeros((1000, 10))
        out_13 = new_zeros_9.scatter_add_(0, index_9, out_12)
        new_zeros_9 = index_9 = out_12 = None
        out_14 = out_13 + l_self_modules_convs_modules_4_parameters_bias_
        out_13 = l_self_modules_convs_modules_4_parameters_bias_ = None
        return (out_14,)
