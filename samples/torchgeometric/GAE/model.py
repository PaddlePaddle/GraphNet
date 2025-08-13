import torch

from torch import device

from torch import inf


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_args_0_: torch.Tensor,
        L_args_1_: torch.Tensor,
        L_self_modules_encoder_modules_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_args_0_ = L_args_0_
        l_args_1_ = L_args_1_
        l_self_modules_encoder_modules_lin_parameters_weight_ = (
            L_self_modules_encoder_modules_lin_parameters_weight_
        )
        l_self_modules_encoder_parameters_bias_ = (
            L_self_modules_encoder_parameters_bias_
        )
        getitem = l_args_1_[0]
        getitem_1 = l_args_1_[1]
        mask = getitem != getitem_1
        getitem = getitem_1 = None
        arange = torch.arange(0, 1000, device=device(type="cpu"))
        view = arange.view(1, -1)
        arange = None
        loop_index = view.repeat(2, 1)
        view = None
        edge_index = l_args_1_[(slice(None, None, None), mask)]
        l_args_1_ = mask = None
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
            l_args_0_, l_self_modules_encoder_modules_lin_parameters_weight_, None
        )
        l_args_0_ = l_self_modules_encoder_modules_lin_parameters_weight_ = None
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
        new_zeros_1 = out.new_zeros((1000, 16))
        out_1 = new_zeros_1.scatter_add_(0, index_1, out)
        new_zeros_1 = index_1 = out = None
        out_2 = out_1 + l_self_modules_encoder_parameters_bias_
        out_1 = l_self_modules_encoder_parameters_bias_ = None
        return (out_2,)
