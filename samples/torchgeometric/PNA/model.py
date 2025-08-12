import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_x_: torch.Tensor,
        L_edge_index_: torch.Tensor,
        L_self_modules_convs_modules_0_modules_pre_nns_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_0_modules_pre_nns_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_0_modules_aggr_module_buffers_avg_deg_log_: torch.Tensor,
        L_self_modules_convs_modules_0_modules_post_nns_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_0_modules_post_nns_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_0_modules_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_0_modules_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_1_modules_pre_nns_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_1_modules_pre_nns_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_1_modules_aggr_module_buffers_avg_deg_log_: torch.Tensor,
        L_self_modules_convs_modules_1_modules_post_nns_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_1_modules_post_nns_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_1_modules_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_1_modules_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_2_modules_pre_nns_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_2_modules_pre_nns_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_2_modules_aggr_module_buffers_avg_deg_log_: torch.Tensor,
        L_self_modules_convs_modules_2_modules_post_nns_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_2_modules_post_nns_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_2_modules_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_2_modules_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_3_modules_pre_nns_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_3_modules_pre_nns_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_3_modules_aggr_module_buffers_avg_deg_log_: torch.Tensor,
        L_self_modules_convs_modules_3_modules_post_nns_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_3_modules_post_nns_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_3_modules_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_3_modules_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_4_modules_pre_nns_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_4_modules_pre_nns_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_4_modules_aggr_module_buffers_avg_deg_log_: torch.Tensor,
        L_self_modules_convs_modules_4_modules_post_nns_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_4_modules_post_nns_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_4_modules_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_4_modules_lin_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_x_ = L_x_
        l_edge_index_ = L_edge_index_
        l_self_modules_convs_modules_0_modules_pre_nns_modules_0_modules_0_parameters_weight_ = L_self_modules_convs_modules_0_modules_pre_nns_modules_0_modules_0_parameters_weight_
        l_self_modules_convs_modules_0_modules_pre_nns_modules_0_modules_0_parameters_bias_ = L_self_modules_convs_modules_0_modules_pre_nns_modules_0_modules_0_parameters_bias_
        l_self_modules_convs_modules_0_modules_aggr_module_buffers_avg_deg_log_ = (
            L_self_modules_convs_modules_0_modules_aggr_module_buffers_avg_deg_log_
        )
        l_self_modules_convs_modules_0_modules_post_nns_modules_0_modules_0_parameters_weight_ = L_self_modules_convs_modules_0_modules_post_nns_modules_0_modules_0_parameters_weight_
        l_self_modules_convs_modules_0_modules_post_nns_modules_0_modules_0_parameters_bias_ = L_self_modules_convs_modules_0_modules_post_nns_modules_0_modules_0_parameters_bias_
        l_self_modules_convs_modules_0_modules_lin_parameters_weight_ = (
            L_self_modules_convs_modules_0_modules_lin_parameters_weight_
        )
        l_self_modules_convs_modules_0_modules_lin_parameters_bias_ = (
            L_self_modules_convs_modules_0_modules_lin_parameters_bias_
        )
        l_self_modules_convs_modules_1_modules_pre_nns_modules_0_modules_0_parameters_weight_ = L_self_modules_convs_modules_1_modules_pre_nns_modules_0_modules_0_parameters_weight_
        l_self_modules_convs_modules_1_modules_pre_nns_modules_0_modules_0_parameters_bias_ = L_self_modules_convs_modules_1_modules_pre_nns_modules_0_modules_0_parameters_bias_
        l_self_modules_convs_modules_1_modules_aggr_module_buffers_avg_deg_log_ = (
            L_self_modules_convs_modules_1_modules_aggr_module_buffers_avg_deg_log_
        )
        l_self_modules_convs_modules_1_modules_post_nns_modules_0_modules_0_parameters_weight_ = L_self_modules_convs_modules_1_modules_post_nns_modules_0_modules_0_parameters_weight_
        l_self_modules_convs_modules_1_modules_post_nns_modules_0_modules_0_parameters_bias_ = L_self_modules_convs_modules_1_modules_post_nns_modules_0_modules_0_parameters_bias_
        l_self_modules_convs_modules_1_modules_lin_parameters_weight_ = (
            L_self_modules_convs_modules_1_modules_lin_parameters_weight_
        )
        l_self_modules_convs_modules_1_modules_lin_parameters_bias_ = (
            L_self_modules_convs_modules_1_modules_lin_parameters_bias_
        )
        l_self_modules_convs_modules_2_modules_pre_nns_modules_0_modules_0_parameters_weight_ = L_self_modules_convs_modules_2_modules_pre_nns_modules_0_modules_0_parameters_weight_
        l_self_modules_convs_modules_2_modules_pre_nns_modules_0_modules_0_parameters_bias_ = L_self_modules_convs_modules_2_modules_pre_nns_modules_0_modules_0_parameters_bias_
        l_self_modules_convs_modules_2_modules_aggr_module_buffers_avg_deg_log_ = (
            L_self_modules_convs_modules_2_modules_aggr_module_buffers_avg_deg_log_
        )
        l_self_modules_convs_modules_2_modules_post_nns_modules_0_modules_0_parameters_weight_ = L_self_modules_convs_modules_2_modules_post_nns_modules_0_modules_0_parameters_weight_
        l_self_modules_convs_modules_2_modules_post_nns_modules_0_modules_0_parameters_bias_ = L_self_modules_convs_modules_2_modules_post_nns_modules_0_modules_0_parameters_bias_
        l_self_modules_convs_modules_2_modules_lin_parameters_weight_ = (
            L_self_modules_convs_modules_2_modules_lin_parameters_weight_
        )
        l_self_modules_convs_modules_2_modules_lin_parameters_bias_ = (
            L_self_modules_convs_modules_2_modules_lin_parameters_bias_
        )
        l_self_modules_convs_modules_3_modules_pre_nns_modules_0_modules_0_parameters_weight_ = L_self_modules_convs_modules_3_modules_pre_nns_modules_0_modules_0_parameters_weight_
        l_self_modules_convs_modules_3_modules_pre_nns_modules_0_modules_0_parameters_bias_ = L_self_modules_convs_modules_3_modules_pre_nns_modules_0_modules_0_parameters_bias_
        l_self_modules_convs_modules_3_modules_aggr_module_buffers_avg_deg_log_ = (
            L_self_modules_convs_modules_3_modules_aggr_module_buffers_avg_deg_log_
        )
        l_self_modules_convs_modules_3_modules_post_nns_modules_0_modules_0_parameters_weight_ = L_self_modules_convs_modules_3_modules_post_nns_modules_0_modules_0_parameters_weight_
        l_self_modules_convs_modules_3_modules_post_nns_modules_0_modules_0_parameters_bias_ = L_self_modules_convs_modules_3_modules_post_nns_modules_0_modules_0_parameters_bias_
        l_self_modules_convs_modules_3_modules_lin_parameters_weight_ = (
            L_self_modules_convs_modules_3_modules_lin_parameters_weight_
        )
        l_self_modules_convs_modules_3_modules_lin_parameters_bias_ = (
            L_self_modules_convs_modules_3_modules_lin_parameters_bias_
        )
        l_self_modules_convs_modules_4_modules_pre_nns_modules_0_modules_0_parameters_weight_ = L_self_modules_convs_modules_4_modules_pre_nns_modules_0_modules_0_parameters_weight_
        l_self_modules_convs_modules_4_modules_pre_nns_modules_0_modules_0_parameters_bias_ = L_self_modules_convs_modules_4_modules_pre_nns_modules_0_modules_0_parameters_bias_
        l_self_modules_convs_modules_4_modules_aggr_module_buffers_avg_deg_log_ = (
            L_self_modules_convs_modules_4_modules_aggr_module_buffers_avg_deg_log_
        )
        l_self_modules_convs_modules_4_modules_post_nns_modules_0_modules_0_parameters_weight_ = L_self_modules_convs_modules_4_modules_post_nns_modules_0_modules_0_parameters_weight_
        l_self_modules_convs_modules_4_modules_post_nns_modules_0_modules_0_parameters_bias_ = L_self_modules_convs_modules_4_modules_post_nns_modules_0_modules_0_parameters_bias_
        l_self_modules_convs_modules_4_modules_lin_parameters_weight_ = (
            L_self_modules_convs_modules_4_modules_lin_parameters_weight_
        )
        l_self_modules_convs_modules_4_modules_lin_parameters_bias_ = (
            L_self_modules_convs_modules_4_modules_lin_parameters_bias_
        )
        view = l_x_.view(-1, 1, 128)
        l_x_ = None
        x = view.repeat(1, 1, 1)
        view = None
        edge_index_i = l_edge_index_[1]
        edge_index_j = l_edge_index_[0]
        x_i = x.index_select(0, edge_index_i)
        x_j = x.index_select(0, edge_index_j)
        edge_index_j = None
        h = torch.cat([x_i, x_j], dim=-1)
        x_i = x_j = None
        getitem_2 = h[(slice(None, None, None), 0)]
        h = None
        input_1 = torch._C._nn.linear(
            getitem_2,
            l_self_modules_convs_modules_0_modules_pre_nns_modules_0_modules_0_parameters_weight_,
            l_self_modules_convs_modules_0_modules_pre_nns_modules_0_modules_0_parameters_bias_,
        )
        getitem_2 = l_self_modules_convs_modules_0_modules_pre_nns_modules_0_modules_0_parameters_weight_ = l_self_modules_convs_modules_0_modules_pre_nns_modules_0_modules_0_parameters_bias_ = (None)
        out = torch.stack([input_1], dim=1)
        input_1 = None
        count = out.new_zeros(1000)
        new_ones = out.new_ones(100)
        scatter_add_ = count.scatter_add_(0, edge_index_i, new_ones)
        new_ones = scatter_add_ = None
        count_1 = count.clamp(min=1)
        count = None
        view_1 = edge_index_i.view((-1, 1, 1))
        index = view_1.expand_as(out)
        view_1 = None
        new_zeros_1 = out.new_zeros((1000, 1, 128))
        out_1 = new_zeros_1.scatter_add_(0, index, out)
        new_zeros_1 = index = None
        view_2 = count_1.view((-1, 1, 1))
        count_1 = None
        expand_as_1 = view_2.expand_as(out_1)
        view_2 = None
        truediv = out_1 / expand_as_1
        out_1 = expand_as_1 = None
        view_3 = edge_index_i.view((-1, 1, 1))
        index_1 = view_3.expand_as(out)
        view_3 = None
        new_zeros_2 = out.new_zeros((1000, 1, 128))
        scatter_reduce_ = new_zeros_2.scatter_reduce_(
            0, index_1, out, reduce="amin", include_self=False
        )
        new_zeros_2 = index_1 = None
        view_4 = edge_index_i.view((-1, 1, 1))
        index_2 = view_4.expand_as(out)
        view_4 = None
        new_zeros_3 = out.new_zeros((1000, 1, 128))
        scatter_reduce__1 = new_zeros_3.scatter_reduce_(
            0, index_2, out, reduce="amax", include_self=False
        )
        new_zeros_3 = index_2 = None
        count_2 = out.new_zeros(1000)
        new_ones_1 = out.new_ones(100)
        scatter_add__2 = count_2.scatter_add_(0, edge_index_i, new_ones_1)
        new_ones_1 = scatter_add__2 = None
        count_3 = count_2.clamp(min=1)
        count_2 = None
        view_5 = edge_index_i.view((-1, 1, 1))
        index_3 = view_5.expand_as(out)
        view_5 = None
        new_zeros_5 = out.new_zeros((1000, 1, 128))
        out_2 = new_zeros_5.scatter_add_(0, index_3, out)
        new_zeros_5 = index_3 = None
        view_6 = count_3.view((-1, 1, 1))
        count_3 = None
        expand_as_5 = view_6.expand_as(out_2)
        view_6 = None
        mean = out_2 / expand_as_5
        out_2 = expand_as_5 = None
        mul = out * out
        out = None
        count_4 = mul.new_zeros(1000)
        new_ones_2 = mul.new_ones(100)
        scatter_add__4 = count_4.scatter_add_(0, edge_index_i, new_ones_2)
        new_ones_2 = scatter_add__4 = None
        count_5 = count_4.clamp(min=1)
        count_4 = None
        view_7 = edge_index_i.view((-1, 1, 1))
        index_4 = view_7.expand_as(mul)
        view_7 = None
        new_zeros_7 = mul.new_zeros((1000, 1, 128))
        out_3 = new_zeros_7.scatter_add_(0, index_4, mul)
        new_zeros_7 = index_4 = mul = None
        view_8 = count_5.view((-1, 1, 1))
        count_5 = None
        expand_as_7 = view_8.expand_as(out_3)
        view_8 = None
        mean2 = out_3 / expand_as_7
        out_3 = expand_as_7 = None
        mul_1 = mean * mean
        mean = None
        var = mean2 - mul_1
        mean2 = mul_1 = None
        clamp_3 = var.clamp(min=1e-05)
        var = None
        out_4 = clamp_3.sqrt()
        clamp_3 = None
        le = out_4 <= 0.0031622776601683794
        out_5 = out_4.masked_fill(le, 0.0)
        out_4 = le = None
        out_6 = torch.cat([truediv, scatter_reduce_, scatter_reduce__1, out_5], dim=-1)
        truediv = scatter_reduce_ = scatter_reduce__1 = out_5 = None
        out_7 = torch.zeros((1000,), dtype=torch.float32, device=device(type="cpu"))
        one = torch.ones((100,), dtype=torch.float32, device=device(type="cpu"))
        deg = out_7.scatter_add_(0, edge_index_i, one)
        out_7 = edge_index_i = one = None
        deg_1 = deg.view([-1, 1, 1])
        deg = None
        add = deg_1 + 1
        log = torch.log(add)
        add = None
        truediv_3 = (
            log
            / l_self_modules_convs_modules_0_modules_aggr_module_buffers_avg_deg_log_
        )
        log = None
        out_scaler = out_6 * truediv_3
        truediv_3 = None
        clamp_4 = deg_1.clamp(min=1)
        deg_1 = None
        add_1 = clamp_4 + 1
        clamp_4 = None
        log_1 = torch.log(add_1)
        add_1 = None
        truediv_4 = (
            l_self_modules_convs_modules_0_modules_aggr_module_buffers_avg_deg_log_
            / log_1
        )
        l_self_modules_convs_modules_0_modules_aggr_module_buffers_avg_deg_log_ = (
            log_1
        ) = None
        out_scaler_1 = out_6 * truediv_4
        truediv_4 = None
        out_8 = torch.cat([out_6, out_scaler, out_scaler_1], dim=-1)
        out_6 = out_scaler = out_scaler_1 = None
        out_9 = torch.cat([x, out_8], dim=-1)
        x = out_8 = None
        getitem_3 = out_9[(slice(None, None, None), 0)]
        out_9 = None
        input_2 = torch._C._nn.linear(
            getitem_3,
            l_self_modules_convs_modules_0_modules_post_nns_modules_0_modules_0_parameters_weight_,
            l_self_modules_convs_modules_0_modules_post_nns_modules_0_modules_0_parameters_bias_,
        )
        getitem_3 = l_self_modules_convs_modules_0_modules_post_nns_modules_0_modules_0_parameters_weight_ = l_self_modules_convs_modules_0_modules_post_nns_modules_0_modules_0_parameters_bias_ = (None)
        out_10 = torch.cat([input_2], dim=1)
        input_2 = None
        x_1 = torch._C._nn.linear(
            out_10,
            l_self_modules_convs_modules_0_modules_lin_parameters_weight_,
            l_self_modules_convs_modules_0_modules_lin_parameters_bias_,
        )
        out_10 = (
            l_self_modules_convs_modules_0_modules_lin_parameters_weight_
        ) = l_self_modules_convs_modules_0_modules_lin_parameters_bias_ = None
        x_2 = torch.nn.functional.relu(x_1, inplace=False)
        x_1 = None
        x_3 = torch.nn.functional.dropout(x_2, 0.0, False, False)
        x_2 = None
        view_10 = x_3.view(-1, 1, 256)
        x_3 = None
        x_4 = view_10.repeat(1, 1, 1)
        view_10 = None
        edge_index_i_1 = l_edge_index_[1]
        edge_index_j_1 = l_edge_index_[0]
        x_i_1 = x_4.index_select(0, edge_index_i_1)
        x_j_1 = x_4.index_select(0, edge_index_j_1)
        edge_index_j_1 = None
        h_1 = torch.cat([x_i_1, x_j_1], dim=-1)
        x_i_1 = x_j_1 = None
        getitem_6 = h_1[(slice(None, None, None), 0)]
        h_1 = None
        input_3 = torch._C._nn.linear(
            getitem_6,
            l_self_modules_convs_modules_1_modules_pre_nns_modules_0_modules_0_parameters_weight_,
            l_self_modules_convs_modules_1_modules_pre_nns_modules_0_modules_0_parameters_bias_,
        )
        getitem_6 = l_self_modules_convs_modules_1_modules_pre_nns_modules_0_modules_0_parameters_weight_ = l_self_modules_convs_modules_1_modules_pre_nns_modules_0_modules_0_parameters_bias_ = (None)
        out_11 = torch.stack([input_3], dim=1)
        input_3 = None
        count_6 = out_11.new_zeros(1000)
        new_ones_3 = out_11.new_ones(100)
        scatter_add__7 = count_6.scatter_add_(0, edge_index_i_1, new_ones_3)
        new_ones_3 = scatter_add__7 = None
        count_7 = count_6.clamp(min=1)
        count_6 = None
        view_11 = edge_index_i_1.view((-1, 1, 1))
        index_5 = view_11.expand_as(out_11)
        view_11 = None
        new_zeros_9 = out_11.new_zeros((1000, 1, 256))
        out_12 = new_zeros_9.scatter_add_(0, index_5, out_11)
        new_zeros_9 = index_5 = None
        view_12 = count_7.view((-1, 1, 1))
        count_7 = None
        expand_as_9 = view_12.expand_as(out_12)
        view_12 = None
        truediv_5 = out_12 / expand_as_9
        out_12 = expand_as_9 = None
        view_13 = edge_index_i_1.view((-1, 1, 1))
        index_6 = view_13.expand_as(out_11)
        view_13 = None
        new_zeros_10 = out_11.new_zeros((1000, 1, 256))
        scatter_reduce__2 = new_zeros_10.scatter_reduce_(
            0, index_6, out_11, reduce="amin", include_self=False
        )
        new_zeros_10 = index_6 = None
        view_14 = edge_index_i_1.view((-1, 1, 1))
        index_7 = view_14.expand_as(out_11)
        view_14 = None
        new_zeros_11 = out_11.new_zeros((1000, 1, 256))
        scatter_reduce__3 = new_zeros_11.scatter_reduce_(
            0, index_7, out_11, reduce="amax", include_self=False
        )
        new_zeros_11 = index_7 = None
        count_8 = out_11.new_zeros(1000)
        new_ones_4 = out_11.new_ones(100)
        scatter_add__9 = count_8.scatter_add_(0, edge_index_i_1, new_ones_4)
        new_ones_4 = scatter_add__9 = None
        count_9 = count_8.clamp(min=1)
        count_8 = None
        view_15 = edge_index_i_1.view((-1, 1, 1))
        index_8 = view_15.expand_as(out_11)
        view_15 = None
        new_zeros_13 = out_11.new_zeros((1000, 1, 256))
        out_13 = new_zeros_13.scatter_add_(0, index_8, out_11)
        new_zeros_13 = index_8 = None
        view_16 = count_9.view((-1, 1, 1))
        count_9 = None
        expand_as_13 = view_16.expand_as(out_13)
        view_16 = None
        mean_1 = out_13 / expand_as_13
        out_13 = expand_as_13 = None
        mul_4 = out_11 * out_11
        out_11 = None
        count_10 = mul_4.new_zeros(1000)
        new_ones_5 = mul_4.new_ones(100)
        scatter_add__11 = count_10.scatter_add_(0, edge_index_i_1, new_ones_5)
        new_ones_5 = scatter_add__11 = None
        count_11 = count_10.clamp(min=1)
        count_10 = None
        view_17 = edge_index_i_1.view((-1, 1, 1))
        index_9 = view_17.expand_as(mul_4)
        view_17 = None
        new_zeros_15 = mul_4.new_zeros((1000, 1, 256))
        out_14 = new_zeros_15.scatter_add_(0, index_9, mul_4)
        new_zeros_15 = index_9 = mul_4 = None
        view_18 = count_11.view((-1, 1, 1))
        count_11 = None
        expand_as_15 = view_18.expand_as(out_14)
        view_18 = None
        mean2_1 = out_14 / expand_as_15
        out_14 = expand_as_15 = None
        mul_5 = mean_1 * mean_1
        mean_1 = None
        var_1 = mean2_1 - mul_5
        mean2_1 = mul_5 = None
        clamp_8 = var_1.clamp(min=1e-05)
        var_1 = None
        out_15 = clamp_8.sqrt()
        clamp_8 = None
        le_1 = out_15 <= 0.0031622776601683794
        out_16 = out_15.masked_fill(le_1, 0.0)
        out_15 = le_1 = None
        out_17 = torch.cat(
            [truediv_5, scatter_reduce__2, scatter_reduce__3, out_16], dim=-1
        )
        truediv_5 = scatter_reduce__2 = scatter_reduce__3 = out_16 = None
        out_18 = torch.zeros((1000,), dtype=torch.float32, device=device(type="cpu"))
        one_1 = torch.ones((100,), dtype=torch.float32, device=device(type="cpu"))
        deg_2 = out_18.scatter_add_(0, edge_index_i_1, one_1)
        out_18 = edge_index_i_1 = one_1 = None
        deg_3 = deg_2.view([-1, 1, 1])
        deg_2 = None
        add_2 = deg_3 + 1
        log_2 = torch.log(add_2)
        add_2 = None
        truediv_8 = (
            log_2
            / l_self_modules_convs_modules_1_modules_aggr_module_buffers_avg_deg_log_
        )
        log_2 = None
        out_scaler_2 = out_17 * truediv_8
        truediv_8 = None
        clamp_9 = deg_3.clamp(min=1)
        deg_3 = None
        add_3 = clamp_9 + 1
        clamp_9 = None
        log_3 = torch.log(add_3)
        add_3 = None
        truediv_9 = (
            l_self_modules_convs_modules_1_modules_aggr_module_buffers_avg_deg_log_
            / log_3
        )
        l_self_modules_convs_modules_1_modules_aggr_module_buffers_avg_deg_log_ = (
            log_3
        ) = None
        out_scaler_3 = out_17 * truediv_9
        truediv_9 = None
        out_19 = torch.cat([out_17, out_scaler_2, out_scaler_3], dim=-1)
        out_17 = out_scaler_2 = out_scaler_3 = None
        out_20 = torch.cat([x_4, out_19], dim=-1)
        x_4 = out_19 = None
        getitem_7 = out_20[(slice(None, None, None), 0)]
        out_20 = None
        input_4 = torch._C._nn.linear(
            getitem_7,
            l_self_modules_convs_modules_1_modules_post_nns_modules_0_modules_0_parameters_weight_,
            l_self_modules_convs_modules_1_modules_post_nns_modules_0_modules_0_parameters_bias_,
        )
        getitem_7 = l_self_modules_convs_modules_1_modules_post_nns_modules_0_modules_0_parameters_weight_ = l_self_modules_convs_modules_1_modules_post_nns_modules_0_modules_0_parameters_bias_ = (None)
        out_21 = torch.cat([input_4], dim=1)
        input_4 = None
        x_5 = torch._C._nn.linear(
            out_21,
            l_self_modules_convs_modules_1_modules_lin_parameters_weight_,
            l_self_modules_convs_modules_1_modules_lin_parameters_bias_,
        )
        out_21 = (
            l_self_modules_convs_modules_1_modules_lin_parameters_weight_
        ) = l_self_modules_convs_modules_1_modules_lin_parameters_bias_ = None
        x_6 = torch.nn.functional.relu(x_5, inplace=False)
        x_5 = None
        x_7 = torch.nn.functional.dropout(x_6, 0.0, False, False)
        x_6 = None
        view_20 = x_7.view(-1, 1, 256)
        x_7 = None
        x_8 = view_20.repeat(1, 1, 1)
        view_20 = None
        edge_index_i_2 = l_edge_index_[1]
        edge_index_j_2 = l_edge_index_[0]
        x_i_2 = x_8.index_select(0, edge_index_i_2)
        x_j_2 = x_8.index_select(0, edge_index_j_2)
        edge_index_j_2 = None
        h_2 = torch.cat([x_i_2, x_j_2], dim=-1)
        x_i_2 = x_j_2 = None
        getitem_10 = h_2[(slice(None, None, None), 0)]
        h_2 = None
        input_5 = torch._C._nn.linear(
            getitem_10,
            l_self_modules_convs_modules_2_modules_pre_nns_modules_0_modules_0_parameters_weight_,
            l_self_modules_convs_modules_2_modules_pre_nns_modules_0_modules_0_parameters_bias_,
        )
        getitem_10 = l_self_modules_convs_modules_2_modules_pre_nns_modules_0_modules_0_parameters_weight_ = l_self_modules_convs_modules_2_modules_pre_nns_modules_0_modules_0_parameters_bias_ = (None)
        out_22 = torch.stack([input_5], dim=1)
        input_5 = None
        count_12 = out_22.new_zeros(1000)
        new_ones_6 = out_22.new_ones(100)
        scatter_add__14 = count_12.scatter_add_(0, edge_index_i_2, new_ones_6)
        new_ones_6 = scatter_add__14 = None
        count_13 = count_12.clamp(min=1)
        count_12 = None
        view_21 = edge_index_i_2.view((-1, 1, 1))
        index_10 = view_21.expand_as(out_22)
        view_21 = None
        new_zeros_17 = out_22.new_zeros((1000, 1, 256))
        out_23 = new_zeros_17.scatter_add_(0, index_10, out_22)
        new_zeros_17 = index_10 = None
        view_22 = count_13.view((-1, 1, 1))
        count_13 = None
        expand_as_17 = view_22.expand_as(out_23)
        view_22 = None
        truediv_10 = out_23 / expand_as_17
        out_23 = expand_as_17 = None
        view_23 = edge_index_i_2.view((-1, 1, 1))
        index_11 = view_23.expand_as(out_22)
        view_23 = None
        new_zeros_18 = out_22.new_zeros((1000, 1, 256))
        scatter_reduce__4 = new_zeros_18.scatter_reduce_(
            0, index_11, out_22, reduce="amin", include_self=False
        )
        new_zeros_18 = index_11 = None
        view_24 = edge_index_i_2.view((-1, 1, 1))
        index_12 = view_24.expand_as(out_22)
        view_24 = None
        new_zeros_19 = out_22.new_zeros((1000, 1, 256))
        scatter_reduce__5 = new_zeros_19.scatter_reduce_(
            0, index_12, out_22, reduce="amax", include_self=False
        )
        new_zeros_19 = index_12 = None
        count_14 = out_22.new_zeros(1000)
        new_ones_7 = out_22.new_ones(100)
        scatter_add__16 = count_14.scatter_add_(0, edge_index_i_2, new_ones_7)
        new_ones_7 = scatter_add__16 = None
        count_15 = count_14.clamp(min=1)
        count_14 = None
        view_25 = edge_index_i_2.view((-1, 1, 1))
        index_13 = view_25.expand_as(out_22)
        view_25 = None
        new_zeros_21 = out_22.new_zeros((1000, 1, 256))
        out_24 = new_zeros_21.scatter_add_(0, index_13, out_22)
        new_zeros_21 = index_13 = None
        view_26 = count_15.view((-1, 1, 1))
        count_15 = None
        expand_as_21 = view_26.expand_as(out_24)
        view_26 = None
        mean_2 = out_24 / expand_as_21
        out_24 = expand_as_21 = None
        mul_8 = out_22 * out_22
        out_22 = None
        count_16 = mul_8.new_zeros(1000)
        new_ones_8 = mul_8.new_ones(100)
        scatter_add__18 = count_16.scatter_add_(0, edge_index_i_2, new_ones_8)
        new_ones_8 = scatter_add__18 = None
        count_17 = count_16.clamp(min=1)
        count_16 = None
        view_27 = edge_index_i_2.view((-1, 1, 1))
        index_14 = view_27.expand_as(mul_8)
        view_27 = None
        new_zeros_23 = mul_8.new_zeros((1000, 1, 256))
        out_25 = new_zeros_23.scatter_add_(0, index_14, mul_8)
        new_zeros_23 = index_14 = mul_8 = None
        view_28 = count_17.view((-1, 1, 1))
        count_17 = None
        expand_as_23 = view_28.expand_as(out_25)
        view_28 = None
        mean2_2 = out_25 / expand_as_23
        out_25 = expand_as_23 = None
        mul_9 = mean_2 * mean_2
        mean_2 = None
        var_2 = mean2_2 - mul_9
        mean2_2 = mul_9 = None
        clamp_13 = var_2.clamp(min=1e-05)
        var_2 = None
        out_26 = clamp_13.sqrt()
        clamp_13 = None
        le_2 = out_26 <= 0.0031622776601683794
        out_27 = out_26.masked_fill(le_2, 0.0)
        out_26 = le_2 = None
        out_28 = torch.cat(
            [truediv_10, scatter_reduce__4, scatter_reduce__5, out_27], dim=-1
        )
        truediv_10 = scatter_reduce__4 = scatter_reduce__5 = out_27 = None
        out_29 = torch.zeros((1000,), dtype=torch.float32, device=device(type="cpu"))
        one_2 = torch.ones((100,), dtype=torch.float32, device=device(type="cpu"))
        deg_4 = out_29.scatter_add_(0, edge_index_i_2, one_2)
        out_29 = edge_index_i_2 = one_2 = None
        deg_5 = deg_4.view([-1, 1, 1])
        deg_4 = None
        add_4 = deg_5 + 1
        log_4 = torch.log(add_4)
        add_4 = None
        truediv_13 = (
            log_4
            / l_self_modules_convs_modules_2_modules_aggr_module_buffers_avg_deg_log_
        )
        log_4 = None
        out_scaler_4 = out_28 * truediv_13
        truediv_13 = None
        clamp_14 = deg_5.clamp(min=1)
        deg_5 = None
        add_5 = clamp_14 + 1
        clamp_14 = None
        log_5 = torch.log(add_5)
        add_5 = None
        truediv_14 = (
            l_self_modules_convs_modules_2_modules_aggr_module_buffers_avg_deg_log_
            / log_5
        )
        l_self_modules_convs_modules_2_modules_aggr_module_buffers_avg_deg_log_ = (
            log_5
        ) = None
        out_scaler_5 = out_28 * truediv_14
        truediv_14 = None
        out_30 = torch.cat([out_28, out_scaler_4, out_scaler_5], dim=-1)
        out_28 = out_scaler_4 = out_scaler_5 = None
        out_31 = torch.cat([x_8, out_30], dim=-1)
        x_8 = out_30 = None
        getitem_11 = out_31[(slice(None, None, None), 0)]
        out_31 = None
        input_6 = torch._C._nn.linear(
            getitem_11,
            l_self_modules_convs_modules_2_modules_post_nns_modules_0_modules_0_parameters_weight_,
            l_self_modules_convs_modules_2_modules_post_nns_modules_0_modules_0_parameters_bias_,
        )
        getitem_11 = l_self_modules_convs_modules_2_modules_post_nns_modules_0_modules_0_parameters_weight_ = l_self_modules_convs_modules_2_modules_post_nns_modules_0_modules_0_parameters_bias_ = (None)
        out_32 = torch.cat([input_6], dim=1)
        input_6 = None
        x_9 = torch._C._nn.linear(
            out_32,
            l_self_modules_convs_modules_2_modules_lin_parameters_weight_,
            l_self_modules_convs_modules_2_modules_lin_parameters_bias_,
        )
        out_32 = (
            l_self_modules_convs_modules_2_modules_lin_parameters_weight_
        ) = l_self_modules_convs_modules_2_modules_lin_parameters_bias_ = None
        x_10 = torch.nn.functional.relu(x_9, inplace=False)
        x_9 = None
        x_11 = torch.nn.functional.dropout(x_10, 0.0, False, False)
        x_10 = None
        view_30 = x_11.view(-1, 1, 256)
        x_11 = None
        x_12 = view_30.repeat(1, 1, 1)
        view_30 = None
        edge_index_i_3 = l_edge_index_[1]
        edge_index_j_3 = l_edge_index_[0]
        x_i_3 = x_12.index_select(0, edge_index_i_3)
        x_j_3 = x_12.index_select(0, edge_index_j_3)
        edge_index_j_3 = None
        h_3 = torch.cat([x_i_3, x_j_3], dim=-1)
        x_i_3 = x_j_3 = None
        getitem_14 = h_3[(slice(None, None, None), 0)]
        h_3 = None
        input_7 = torch._C._nn.linear(
            getitem_14,
            l_self_modules_convs_modules_3_modules_pre_nns_modules_0_modules_0_parameters_weight_,
            l_self_modules_convs_modules_3_modules_pre_nns_modules_0_modules_0_parameters_bias_,
        )
        getitem_14 = l_self_modules_convs_modules_3_modules_pre_nns_modules_0_modules_0_parameters_weight_ = l_self_modules_convs_modules_3_modules_pre_nns_modules_0_modules_0_parameters_bias_ = (None)
        out_33 = torch.stack([input_7], dim=1)
        input_7 = None
        count_18 = out_33.new_zeros(1000)
        new_ones_9 = out_33.new_ones(100)
        scatter_add__21 = count_18.scatter_add_(0, edge_index_i_3, new_ones_9)
        new_ones_9 = scatter_add__21 = None
        count_19 = count_18.clamp(min=1)
        count_18 = None
        view_31 = edge_index_i_3.view((-1, 1, 1))
        index_15 = view_31.expand_as(out_33)
        view_31 = None
        new_zeros_25 = out_33.new_zeros((1000, 1, 256))
        out_34 = new_zeros_25.scatter_add_(0, index_15, out_33)
        new_zeros_25 = index_15 = None
        view_32 = count_19.view((-1, 1, 1))
        count_19 = None
        expand_as_25 = view_32.expand_as(out_34)
        view_32 = None
        truediv_15 = out_34 / expand_as_25
        out_34 = expand_as_25 = None
        view_33 = edge_index_i_3.view((-1, 1, 1))
        index_16 = view_33.expand_as(out_33)
        view_33 = None
        new_zeros_26 = out_33.new_zeros((1000, 1, 256))
        scatter_reduce__6 = new_zeros_26.scatter_reduce_(
            0, index_16, out_33, reduce="amin", include_self=False
        )
        new_zeros_26 = index_16 = None
        view_34 = edge_index_i_3.view((-1, 1, 1))
        index_17 = view_34.expand_as(out_33)
        view_34 = None
        new_zeros_27 = out_33.new_zeros((1000, 1, 256))
        scatter_reduce__7 = new_zeros_27.scatter_reduce_(
            0, index_17, out_33, reduce="amax", include_self=False
        )
        new_zeros_27 = index_17 = None
        count_20 = out_33.new_zeros(1000)
        new_ones_10 = out_33.new_ones(100)
        scatter_add__23 = count_20.scatter_add_(0, edge_index_i_3, new_ones_10)
        new_ones_10 = scatter_add__23 = None
        count_21 = count_20.clamp(min=1)
        count_20 = None
        view_35 = edge_index_i_3.view((-1, 1, 1))
        index_18 = view_35.expand_as(out_33)
        view_35 = None
        new_zeros_29 = out_33.new_zeros((1000, 1, 256))
        out_35 = new_zeros_29.scatter_add_(0, index_18, out_33)
        new_zeros_29 = index_18 = None
        view_36 = count_21.view((-1, 1, 1))
        count_21 = None
        expand_as_29 = view_36.expand_as(out_35)
        view_36 = None
        mean_3 = out_35 / expand_as_29
        out_35 = expand_as_29 = None
        mul_12 = out_33 * out_33
        out_33 = None
        count_22 = mul_12.new_zeros(1000)
        new_ones_11 = mul_12.new_ones(100)
        scatter_add__25 = count_22.scatter_add_(0, edge_index_i_3, new_ones_11)
        new_ones_11 = scatter_add__25 = None
        count_23 = count_22.clamp(min=1)
        count_22 = None
        view_37 = edge_index_i_3.view((-1, 1, 1))
        index_19 = view_37.expand_as(mul_12)
        view_37 = None
        new_zeros_31 = mul_12.new_zeros((1000, 1, 256))
        out_36 = new_zeros_31.scatter_add_(0, index_19, mul_12)
        new_zeros_31 = index_19 = mul_12 = None
        view_38 = count_23.view((-1, 1, 1))
        count_23 = None
        expand_as_31 = view_38.expand_as(out_36)
        view_38 = None
        mean2_3 = out_36 / expand_as_31
        out_36 = expand_as_31 = None
        mul_13 = mean_3 * mean_3
        mean_3 = None
        var_3 = mean2_3 - mul_13
        mean2_3 = mul_13 = None
        clamp_18 = var_3.clamp(min=1e-05)
        var_3 = None
        out_37 = clamp_18.sqrt()
        clamp_18 = None
        le_3 = out_37 <= 0.0031622776601683794
        out_38 = out_37.masked_fill(le_3, 0.0)
        out_37 = le_3 = None
        out_39 = torch.cat(
            [truediv_15, scatter_reduce__6, scatter_reduce__7, out_38], dim=-1
        )
        truediv_15 = scatter_reduce__6 = scatter_reduce__7 = out_38 = None
        out_40 = torch.zeros((1000,), dtype=torch.float32, device=device(type="cpu"))
        one_3 = torch.ones((100,), dtype=torch.float32, device=device(type="cpu"))
        deg_6 = out_40.scatter_add_(0, edge_index_i_3, one_3)
        out_40 = edge_index_i_3 = one_3 = None
        deg_7 = deg_6.view([-1, 1, 1])
        deg_6 = None
        add_6 = deg_7 + 1
        log_6 = torch.log(add_6)
        add_6 = None
        truediv_18 = (
            log_6
            / l_self_modules_convs_modules_3_modules_aggr_module_buffers_avg_deg_log_
        )
        log_6 = None
        out_scaler_6 = out_39 * truediv_18
        truediv_18 = None
        clamp_19 = deg_7.clamp(min=1)
        deg_7 = None
        add_7 = clamp_19 + 1
        clamp_19 = None
        log_7 = torch.log(add_7)
        add_7 = None
        truediv_19 = (
            l_self_modules_convs_modules_3_modules_aggr_module_buffers_avg_deg_log_
            / log_7
        )
        l_self_modules_convs_modules_3_modules_aggr_module_buffers_avg_deg_log_ = (
            log_7
        ) = None
        out_scaler_7 = out_39 * truediv_19
        truediv_19 = None
        out_41 = torch.cat([out_39, out_scaler_6, out_scaler_7], dim=-1)
        out_39 = out_scaler_6 = out_scaler_7 = None
        out_42 = torch.cat([x_12, out_41], dim=-1)
        x_12 = out_41 = None
        getitem_15 = out_42[(slice(None, None, None), 0)]
        out_42 = None
        input_8 = torch._C._nn.linear(
            getitem_15,
            l_self_modules_convs_modules_3_modules_post_nns_modules_0_modules_0_parameters_weight_,
            l_self_modules_convs_modules_3_modules_post_nns_modules_0_modules_0_parameters_bias_,
        )
        getitem_15 = l_self_modules_convs_modules_3_modules_post_nns_modules_0_modules_0_parameters_weight_ = l_self_modules_convs_modules_3_modules_post_nns_modules_0_modules_0_parameters_bias_ = (None)
        out_43 = torch.cat([input_8], dim=1)
        input_8 = None
        x_13 = torch._C._nn.linear(
            out_43,
            l_self_modules_convs_modules_3_modules_lin_parameters_weight_,
            l_self_modules_convs_modules_3_modules_lin_parameters_bias_,
        )
        out_43 = (
            l_self_modules_convs_modules_3_modules_lin_parameters_weight_
        ) = l_self_modules_convs_modules_3_modules_lin_parameters_bias_ = None
        x_14 = torch.nn.functional.relu(x_13, inplace=False)
        x_13 = None
        x_15 = torch.nn.functional.dropout(x_14, 0.0, False, False)
        x_14 = None
        view_40 = x_15.view(-1, 1, 256)
        x_15 = None
        x_16 = view_40.repeat(1, 1, 1)
        view_40 = None
        edge_index_i_4 = l_edge_index_[1]
        edge_index_j_4 = l_edge_index_[0]
        l_edge_index_ = None
        x_i_4 = x_16.index_select(0, edge_index_i_4)
        x_j_4 = x_16.index_select(0, edge_index_j_4)
        edge_index_j_4 = None
        h_4 = torch.cat([x_i_4, x_j_4], dim=-1)
        x_i_4 = x_j_4 = None
        getitem_18 = h_4[(slice(None, None, None), 0)]
        h_4 = None
        input_9 = torch._C._nn.linear(
            getitem_18,
            l_self_modules_convs_modules_4_modules_pre_nns_modules_0_modules_0_parameters_weight_,
            l_self_modules_convs_modules_4_modules_pre_nns_modules_0_modules_0_parameters_bias_,
        )
        getitem_18 = l_self_modules_convs_modules_4_modules_pre_nns_modules_0_modules_0_parameters_weight_ = l_self_modules_convs_modules_4_modules_pre_nns_modules_0_modules_0_parameters_bias_ = (None)
        out_44 = torch.stack([input_9], dim=1)
        input_9 = None
        count_24 = out_44.new_zeros(1000)
        new_ones_12 = out_44.new_ones(100)
        scatter_add__28 = count_24.scatter_add_(0, edge_index_i_4, new_ones_12)
        new_ones_12 = scatter_add__28 = None
        count_25 = count_24.clamp(min=1)
        count_24 = None
        view_41 = edge_index_i_4.view((-1, 1, 1))
        index_20 = view_41.expand_as(out_44)
        view_41 = None
        new_zeros_33 = out_44.new_zeros((1000, 1, 256))
        out_45 = new_zeros_33.scatter_add_(0, index_20, out_44)
        new_zeros_33 = index_20 = None
        view_42 = count_25.view((-1, 1, 1))
        count_25 = None
        expand_as_33 = view_42.expand_as(out_45)
        view_42 = None
        truediv_20 = out_45 / expand_as_33
        out_45 = expand_as_33 = None
        view_43 = edge_index_i_4.view((-1, 1, 1))
        index_21 = view_43.expand_as(out_44)
        view_43 = None
        new_zeros_34 = out_44.new_zeros((1000, 1, 256))
        scatter_reduce__8 = new_zeros_34.scatter_reduce_(
            0, index_21, out_44, reduce="amin", include_self=False
        )
        new_zeros_34 = index_21 = None
        view_44 = edge_index_i_4.view((-1, 1, 1))
        index_22 = view_44.expand_as(out_44)
        view_44 = None
        new_zeros_35 = out_44.new_zeros((1000, 1, 256))
        scatter_reduce__9 = new_zeros_35.scatter_reduce_(
            0, index_22, out_44, reduce="amax", include_self=False
        )
        new_zeros_35 = index_22 = None
        count_26 = out_44.new_zeros(1000)
        new_ones_13 = out_44.new_ones(100)
        scatter_add__30 = count_26.scatter_add_(0, edge_index_i_4, new_ones_13)
        new_ones_13 = scatter_add__30 = None
        count_27 = count_26.clamp(min=1)
        count_26 = None
        view_45 = edge_index_i_4.view((-1, 1, 1))
        index_23 = view_45.expand_as(out_44)
        view_45 = None
        new_zeros_37 = out_44.new_zeros((1000, 1, 256))
        out_46 = new_zeros_37.scatter_add_(0, index_23, out_44)
        new_zeros_37 = index_23 = None
        view_46 = count_27.view((-1, 1, 1))
        count_27 = None
        expand_as_37 = view_46.expand_as(out_46)
        view_46 = None
        mean_4 = out_46 / expand_as_37
        out_46 = expand_as_37 = None
        mul_16 = out_44 * out_44
        out_44 = None
        count_28 = mul_16.new_zeros(1000)
        new_ones_14 = mul_16.new_ones(100)
        scatter_add__32 = count_28.scatter_add_(0, edge_index_i_4, new_ones_14)
        new_ones_14 = scatter_add__32 = None
        count_29 = count_28.clamp(min=1)
        count_28 = None
        view_47 = edge_index_i_4.view((-1, 1, 1))
        index_24 = view_47.expand_as(mul_16)
        view_47 = None
        new_zeros_39 = mul_16.new_zeros((1000, 1, 256))
        out_47 = new_zeros_39.scatter_add_(0, index_24, mul_16)
        new_zeros_39 = index_24 = mul_16 = None
        view_48 = count_29.view((-1, 1, 1))
        count_29 = None
        expand_as_39 = view_48.expand_as(out_47)
        view_48 = None
        mean2_4 = out_47 / expand_as_39
        out_47 = expand_as_39 = None
        mul_17 = mean_4 * mean_4
        mean_4 = None
        var_4 = mean2_4 - mul_17
        mean2_4 = mul_17 = None
        clamp_23 = var_4.clamp(min=1e-05)
        var_4 = None
        out_48 = clamp_23.sqrt()
        clamp_23 = None
        le_4 = out_48 <= 0.0031622776601683794
        out_49 = out_48.masked_fill(le_4, 0.0)
        out_48 = le_4 = None
        out_50 = torch.cat(
            [truediv_20, scatter_reduce__8, scatter_reduce__9, out_49], dim=-1
        )
        truediv_20 = scatter_reduce__8 = scatter_reduce__9 = out_49 = None
        out_51 = torch.zeros((1000,), dtype=torch.float32, device=device(type="cpu"))
        one_4 = torch.ones((100,), dtype=torch.float32, device=device(type="cpu"))
        deg_8 = out_51.scatter_add_(0, edge_index_i_4, one_4)
        out_51 = edge_index_i_4 = one_4 = None
        deg_9 = deg_8.view([-1, 1, 1])
        deg_8 = None
        add_8 = deg_9 + 1
        log_8 = torch.log(add_8)
        add_8 = None
        truediv_23 = (
            log_8
            / l_self_modules_convs_modules_4_modules_aggr_module_buffers_avg_deg_log_
        )
        log_8 = None
        out_scaler_8 = out_50 * truediv_23
        truediv_23 = None
        clamp_24 = deg_9.clamp(min=1)
        deg_9 = None
        add_9 = clamp_24 + 1
        clamp_24 = None
        log_9 = torch.log(add_9)
        add_9 = None
        truediv_24 = (
            l_self_modules_convs_modules_4_modules_aggr_module_buffers_avg_deg_log_
            / log_9
        )
        l_self_modules_convs_modules_4_modules_aggr_module_buffers_avg_deg_log_ = (
            log_9
        ) = None
        out_scaler_9 = out_50 * truediv_24
        truediv_24 = None
        out_52 = torch.cat([out_50, out_scaler_8, out_scaler_9], dim=-1)
        out_50 = out_scaler_8 = out_scaler_9 = None
        out_53 = torch.cat([x_16, out_52], dim=-1)
        x_16 = out_52 = None
        getitem_19 = out_53[(slice(None, None, None), 0)]
        out_53 = None
        input_10 = torch._C._nn.linear(
            getitem_19,
            l_self_modules_convs_modules_4_modules_post_nns_modules_0_modules_0_parameters_weight_,
            l_self_modules_convs_modules_4_modules_post_nns_modules_0_modules_0_parameters_bias_,
        )
        getitem_19 = l_self_modules_convs_modules_4_modules_post_nns_modules_0_modules_0_parameters_weight_ = l_self_modules_convs_modules_4_modules_post_nns_modules_0_modules_0_parameters_bias_ = (None)
        out_54 = torch.cat([input_10], dim=1)
        input_10 = None
        x_17 = torch._C._nn.linear(
            out_54,
            l_self_modules_convs_modules_4_modules_lin_parameters_weight_,
            l_self_modules_convs_modules_4_modules_lin_parameters_bias_,
        )
        out_54 = (
            l_self_modules_convs_modules_4_modules_lin_parameters_weight_
        ) = l_self_modules_convs_modules_4_modules_lin_parameters_bias_ = None
        return (x_17,)
