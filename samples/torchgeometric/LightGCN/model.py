import torch

from torch import device

from torch import inf


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_edge_index_: torch.Tensor,
        L_self_modules_embedding_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_buffers_alpha_: torch.Tensor,
    ):
        l_edge_index_ = L_edge_index_
        l_self_modules_embedding_parameters_weight_ = (
            L_self_modules_embedding_parameters_weight_
        )
        l_self_buffers_alpha_ = L_self_buffers_alpha_
        getitem = l_self_buffers_alpha_[0]
        out = l_self_modules_embedding_parameters_weight_ * getitem
        getitem = None
        edge_weight = torch.ones((500,), dtype=torch.float32, device=device(type="cpu"))
        row = l_edge_index_[0]
        col = l_edge_index_[1]
        view = col.view((-1,))
        index = view.expand_as(edge_weight)
        view = None
        new_zeros = edge_weight.new_zeros((1000,))
        deg = new_zeros.scatter_add_(0, index, edge_weight)
        new_zeros = index = None
        deg_inv_sqrt = deg.pow_(-0.5)
        deg = None
        eq = deg_inv_sqrt.__eq__(inf)
        masked_fill_ = deg_inv_sqrt.masked_fill_(eq, 0)
        eq = masked_fill_ = None
        getitem_3 = deg_inv_sqrt[row]
        row = None
        mul_1 = getitem_3 * edge_weight
        getitem_3 = edge_weight = None
        getitem_4 = deg_inv_sqrt[col]
        deg_inv_sqrt = col = None
        edge_weight_1 = mul_1 * getitem_4
        mul_1 = getitem_4 = None
        edge_index_i = l_edge_index_[1]
        edge_index_j = l_edge_index_[0]
        x_j = l_self_modules_embedding_parameters_weight_.index_select(-2, edge_index_j)
        l_self_modules_embedding_parameters_weight_ = edge_index_j = None
        view_1 = edge_weight_1.view(-1, 1)
        edge_weight_1 = None
        out_1 = view_1 * x_j
        view_1 = x_j = None
        view_2 = edge_index_i.view((-1, 1))
        edge_index_i = None
        index_1 = view_2.expand_as(out_1)
        view_2 = None
        new_zeros_1 = out_1.new_zeros((1000, 64))
        out_2 = new_zeros_1.scatter_add_(0, index_1, out_1)
        new_zeros_1 = index_1 = out_1 = None
        getitem_7 = l_self_buffers_alpha_[1]
        mul_4 = out_2 * getitem_7
        getitem_7 = None
        out_3 = out + mul_4
        out = mul_4 = None
        edge_weight_2 = torch.ones(
            (500,), dtype=torch.float32, device=device(type="cpu")
        )
        row_1 = l_edge_index_[0]
        col_1 = l_edge_index_[1]
        view_3 = col_1.view((-1,))
        index_2 = view_3.expand_as(edge_weight_2)
        view_3 = None
        new_zeros_2 = edge_weight_2.new_zeros((1000,))
        deg_1 = new_zeros_2.scatter_add_(0, index_2, edge_weight_2)
        new_zeros_2 = index_2 = None
        deg_inv_sqrt_1 = deg_1.pow_(-0.5)
        deg_1 = None
        eq_1 = deg_inv_sqrt_1.__eq__(inf)
        masked_fill__1 = deg_inv_sqrt_1.masked_fill_(eq_1, 0)
        eq_1 = masked_fill__1 = None
        getitem_10 = deg_inv_sqrt_1[row_1]
        row_1 = None
        mul_5 = getitem_10 * edge_weight_2
        getitem_10 = edge_weight_2 = None
        getitem_11 = deg_inv_sqrt_1[col_1]
        deg_inv_sqrt_1 = col_1 = None
        edge_weight_3 = mul_5 * getitem_11
        mul_5 = getitem_11 = None
        edge_index_i_1 = l_edge_index_[1]
        edge_index_j_1 = l_edge_index_[0]
        x_j_1 = out_2.index_select(-2, edge_index_j_1)
        out_2 = edge_index_j_1 = None
        view_4 = edge_weight_3.view(-1, 1)
        edge_weight_3 = None
        out_4 = view_4 * x_j_1
        view_4 = x_j_1 = None
        view_5 = edge_index_i_1.view((-1, 1))
        edge_index_i_1 = None
        index_3 = view_5.expand_as(out_4)
        view_5 = None
        new_zeros_3 = out_4.new_zeros((1000, 64))
        out_5 = new_zeros_3.scatter_add_(0, index_3, out_4)
        new_zeros_3 = index_3 = out_4 = None
        getitem_14 = l_self_buffers_alpha_[2]
        mul_8 = out_5 * getitem_14
        getitem_14 = None
        out_6 = out_3 + mul_8
        out_3 = mul_8 = None
        edge_weight_4 = torch.ones(
            (500,), dtype=torch.float32, device=device(type="cpu")
        )
        row_2 = l_edge_index_[0]
        col_2 = l_edge_index_[1]
        view_6 = col_2.view((-1,))
        index_4 = view_6.expand_as(edge_weight_4)
        view_6 = None
        new_zeros_4 = edge_weight_4.new_zeros((1000,))
        deg_2 = new_zeros_4.scatter_add_(0, index_4, edge_weight_4)
        new_zeros_4 = index_4 = None
        deg_inv_sqrt_2 = deg_2.pow_(-0.5)
        deg_2 = None
        eq_2 = deg_inv_sqrt_2.__eq__(inf)
        masked_fill__2 = deg_inv_sqrt_2.masked_fill_(eq_2, 0)
        eq_2 = masked_fill__2 = None
        getitem_17 = deg_inv_sqrt_2[row_2]
        row_2 = None
        mul_9 = getitem_17 * edge_weight_4
        getitem_17 = edge_weight_4 = None
        getitem_18 = deg_inv_sqrt_2[col_2]
        deg_inv_sqrt_2 = col_2 = None
        edge_weight_5 = mul_9 * getitem_18
        mul_9 = getitem_18 = None
        edge_index_i_2 = l_edge_index_[1]
        edge_index_j_2 = l_edge_index_[0]
        x_j_2 = out_5.index_select(-2, edge_index_j_2)
        out_5 = edge_index_j_2 = None
        view_7 = edge_weight_5.view(-1, 1)
        edge_weight_5 = None
        out_7 = view_7 * x_j_2
        view_7 = x_j_2 = None
        view_8 = edge_index_i_2.view((-1, 1))
        edge_index_i_2 = None
        index_5 = view_8.expand_as(out_7)
        view_8 = None
        new_zeros_5 = out_7.new_zeros((1000, 64))
        out_8 = new_zeros_5.scatter_add_(0, index_5, out_7)
        new_zeros_5 = index_5 = out_7 = None
        getitem_21 = l_self_buffers_alpha_[3]
        l_self_buffers_alpha_ = None
        mul_12 = out_8 * getitem_21
        out_8 = getitem_21 = None
        out_9 = out_6 + mul_12
        out_6 = mul_12 = None
        getitem_22 = l_edge_index_[0]
        out_src = out_9[getitem_22]
        getitem_22 = None
        getitem_24 = l_edge_index_[1]
        l_edge_index_ = None
        out_dst = out_9[getitem_24]
        out_9 = getitem_24 = None
        mul_13 = out_src * out_dst
        out_src = out_dst = None
        sum_1 = mul_13.sum(dim=-1)
        mul_13 = None
        return (sum_1,)
