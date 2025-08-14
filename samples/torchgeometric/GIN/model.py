import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_x_: torch.Tensor,
        L_edge_index_: torch.Tensor,
        L_self_modules_convs_modules_0_buffers_eps_: torch.Tensor,
        L_self_modules_convs_modules_0_modules_nn_modules_lins_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_0_modules_nn_modules_lins_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_0_modules_nn_modules_lins_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_0_modules_nn_modules_lins_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_1_buffers_eps_: torch.Tensor,
        L_self_modules_convs_modules_1_modules_nn_modules_lins_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_1_modules_nn_modules_lins_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_1_modules_nn_modules_lins_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_1_modules_nn_modules_lins_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_2_buffers_eps_: torch.Tensor,
        L_self_modules_convs_modules_2_modules_nn_modules_lins_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_2_modules_nn_modules_lins_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_2_modules_nn_modules_lins_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_2_modules_nn_modules_lins_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_3_buffers_eps_: torch.Tensor,
        L_self_modules_convs_modules_3_modules_nn_modules_lins_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_3_modules_nn_modules_lins_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_3_modules_nn_modules_lins_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_3_modules_nn_modules_lins_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_4_buffers_eps_: torch.Tensor,
        L_self_modules_convs_modules_4_modules_nn_modules_lins_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_4_modules_nn_modules_lins_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_4_modules_nn_modules_lins_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_convs_modules_4_modules_nn_modules_lins_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_x_ = L_x_
        l_edge_index_ = L_edge_index_
        l_self_modules_convs_modules_0_buffers_eps_ = (
            L_self_modules_convs_modules_0_buffers_eps_
        )
        l_self_modules_convs_modules_0_modules_nn_modules_lins_modules_0_parameters_weight_ = L_self_modules_convs_modules_0_modules_nn_modules_lins_modules_0_parameters_weight_
        l_self_modules_convs_modules_0_modules_nn_modules_lins_modules_0_parameters_bias_ = L_self_modules_convs_modules_0_modules_nn_modules_lins_modules_0_parameters_bias_
        l_self_modules_convs_modules_0_modules_nn_modules_lins_modules_1_parameters_weight_ = L_self_modules_convs_modules_0_modules_nn_modules_lins_modules_1_parameters_weight_
        l_self_modules_convs_modules_0_modules_nn_modules_lins_modules_1_parameters_bias_ = L_self_modules_convs_modules_0_modules_nn_modules_lins_modules_1_parameters_bias_
        l_self_modules_convs_modules_1_buffers_eps_ = (
            L_self_modules_convs_modules_1_buffers_eps_
        )
        l_self_modules_convs_modules_1_modules_nn_modules_lins_modules_0_parameters_weight_ = L_self_modules_convs_modules_1_modules_nn_modules_lins_modules_0_parameters_weight_
        l_self_modules_convs_modules_1_modules_nn_modules_lins_modules_0_parameters_bias_ = L_self_modules_convs_modules_1_modules_nn_modules_lins_modules_0_parameters_bias_
        l_self_modules_convs_modules_1_modules_nn_modules_lins_modules_1_parameters_weight_ = L_self_modules_convs_modules_1_modules_nn_modules_lins_modules_1_parameters_weight_
        l_self_modules_convs_modules_1_modules_nn_modules_lins_modules_1_parameters_bias_ = L_self_modules_convs_modules_1_modules_nn_modules_lins_modules_1_parameters_bias_
        l_self_modules_convs_modules_2_buffers_eps_ = (
            L_self_modules_convs_modules_2_buffers_eps_
        )
        l_self_modules_convs_modules_2_modules_nn_modules_lins_modules_0_parameters_weight_ = L_self_modules_convs_modules_2_modules_nn_modules_lins_modules_0_parameters_weight_
        l_self_modules_convs_modules_2_modules_nn_modules_lins_modules_0_parameters_bias_ = L_self_modules_convs_modules_2_modules_nn_modules_lins_modules_0_parameters_bias_
        l_self_modules_convs_modules_2_modules_nn_modules_lins_modules_1_parameters_weight_ = L_self_modules_convs_modules_2_modules_nn_modules_lins_modules_1_parameters_weight_
        l_self_modules_convs_modules_2_modules_nn_modules_lins_modules_1_parameters_bias_ = L_self_modules_convs_modules_2_modules_nn_modules_lins_modules_1_parameters_bias_
        l_self_modules_convs_modules_3_buffers_eps_ = (
            L_self_modules_convs_modules_3_buffers_eps_
        )
        l_self_modules_convs_modules_3_modules_nn_modules_lins_modules_0_parameters_weight_ = L_self_modules_convs_modules_3_modules_nn_modules_lins_modules_0_parameters_weight_
        l_self_modules_convs_modules_3_modules_nn_modules_lins_modules_0_parameters_bias_ = L_self_modules_convs_modules_3_modules_nn_modules_lins_modules_0_parameters_bias_
        l_self_modules_convs_modules_3_modules_nn_modules_lins_modules_1_parameters_weight_ = L_self_modules_convs_modules_3_modules_nn_modules_lins_modules_1_parameters_weight_
        l_self_modules_convs_modules_3_modules_nn_modules_lins_modules_1_parameters_bias_ = L_self_modules_convs_modules_3_modules_nn_modules_lins_modules_1_parameters_bias_
        l_self_modules_convs_modules_4_buffers_eps_ = (
            L_self_modules_convs_modules_4_buffers_eps_
        )
        l_self_modules_convs_modules_4_modules_nn_modules_lins_modules_0_parameters_weight_ = L_self_modules_convs_modules_4_modules_nn_modules_lins_modules_0_parameters_weight_
        l_self_modules_convs_modules_4_modules_nn_modules_lins_modules_0_parameters_bias_ = L_self_modules_convs_modules_4_modules_nn_modules_lins_modules_0_parameters_bias_
        l_self_modules_convs_modules_4_modules_nn_modules_lins_modules_1_parameters_weight_ = L_self_modules_convs_modules_4_modules_nn_modules_lins_modules_1_parameters_weight_
        l_self_modules_convs_modules_4_modules_nn_modules_lins_modules_1_parameters_bias_ = L_self_modules_convs_modules_4_modules_nn_modules_lins_modules_1_parameters_bias_
        edge_index_i = l_edge_index_[1]
        edge_index_j = l_edge_index_[0]
        x_j = l_x_.index_select(-2, edge_index_j)
        edge_index_j = None
        view = edge_index_i.view((-1, 1))
        edge_index_i = None
        index = view.expand_as(x_j)
        view = None
        new_zeros = x_j.new_zeros((1000, 128))
        out = new_zeros.scatter_add_(0, index, x_j)
        new_zeros = index = x_j = None
        add = 1 + l_self_modules_convs_modules_0_buffers_eps_
        l_self_modules_convs_modules_0_buffers_eps_ = None
        mul = add * l_x_
        add = l_x_ = None
        out_1 = out + mul
        out = mul = None
        x = torch._C._nn.linear(
            out_1,
            l_self_modules_convs_modules_0_modules_nn_modules_lins_modules_0_parameters_weight_,
            l_self_modules_convs_modules_0_modules_nn_modules_lins_modules_0_parameters_bias_,
        )
        out_1 = l_self_modules_convs_modules_0_modules_nn_modules_lins_modules_0_parameters_weight_ = l_self_modules_convs_modules_0_modules_nn_modules_lins_modules_0_parameters_bias_ = (None)
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
        x_5 = torch.nn.functional.relu(x_4, inplace=False)
        x_4 = None
        x_6 = torch.nn.functional.dropout(x_5, 0.0, False, False)
        x_5 = None
        edge_index_i_1 = l_edge_index_[1]
        edge_index_j_1 = l_edge_index_[0]
        x_j_1 = x_6.index_select(-2, edge_index_j_1)
        edge_index_j_1 = None
        view_1 = edge_index_i_1.view((-1, 1))
        edge_index_i_1 = None
        index_1 = view_1.expand_as(x_j_1)
        view_1 = None
        new_zeros_1 = x_j_1.new_zeros((1000, 256))
        out_2 = new_zeros_1.scatter_add_(0, index_1, x_j_1)
        new_zeros_1 = index_1 = x_j_1 = None
        add_2 = 1 + l_self_modules_convs_modules_1_buffers_eps_
        l_self_modules_convs_modules_1_buffers_eps_ = None
        mul_1 = add_2 * x_6
        add_2 = x_6 = None
        out_3 = out_2 + mul_1
        out_2 = mul_1 = None
        x_7 = torch._C._nn.linear(
            out_3,
            l_self_modules_convs_modules_1_modules_nn_modules_lins_modules_0_parameters_weight_,
            l_self_modules_convs_modules_1_modules_nn_modules_lins_modules_0_parameters_bias_,
        )
        out_3 = l_self_modules_convs_modules_1_modules_nn_modules_lins_modules_0_parameters_weight_ = l_self_modules_convs_modules_1_modules_nn_modules_lins_modules_0_parameters_bias_ = (None)
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
        x_12 = torch.nn.functional.relu(x_11, inplace=False)
        x_11 = None
        x_13 = torch.nn.functional.dropout(x_12, 0.0, False, False)
        x_12 = None
        edge_index_i_2 = l_edge_index_[1]
        edge_index_j_2 = l_edge_index_[0]
        x_j_2 = x_13.index_select(-2, edge_index_j_2)
        edge_index_j_2 = None
        view_2 = edge_index_i_2.view((-1, 1))
        edge_index_i_2 = None
        index_2 = view_2.expand_as(x_j_2)
        view_2 = None
        new_zeros_2 = x_j_2.new_zeros((1000, 256))
        out_4 = new_zeros_2.scatter_add_(0, index_2, x_j_2)
        new_zeros_2 = index_2 = x_j_2 = None
        add_4 = 1 + l_self_modules_convs_modules_2_buffers_eps_
        l_self_modules_convs_modules_2_buffers_eps_ = None
        mul_2 = add_4 * x_13
        add_4 = x_13 = None
        out_5 = out_4 + mul_2
        out_4 = mul_2 = None
        x_14 = torch._C._nn.linear(
            out_5,
            l_self_modules_convs_modules_2_modules_nn_modules_lins_modules_0_parameters_weight_,
            l_self_modules_convs_modules_2_modules_nn_modules_lins_modules_0_parameters_bias_,
        )
        out_5 = l_self_modules_convs_modules_2_modules_nn_modules_lins_modules_0_parameters_weight_ = l_self_modules_convs_modules_2_modules_nn_modules_lins_modules_0_parameters_bias_ = (None)
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
        x_19 = torch.nn.functional.relu(x_18, inplace=False)
        x_18 = None
        x_20 = torch.nn.functional.dropout(x_19, 0.0, False, False)
        x_19 = None
        edge_index_i_3 = l_edge_index_[1]
        edge_index_j_3 = l_edge_index_[0]
        x_j_3 = x_20.index_select(-2, edge_index_j_3)
        edge_index_j_3 = None
        view_3 = edge_index_i_3.view((-1, 1))
        edge_index_i_3 = None
        index_3 = view_3.expand_as(x_j_3)
        view_3 = None
        new_zeros_3 = x_j_3.new_zeros((1000, 256))
        out_6 = new_zeros_3.scatter_add_(0, index_3, x_j_3)
        new_zeros_3 = index_3 = x_j_3 = None
        add_6 = 1 + l_self_modules_convs_modules_3_buffers_eps_
        l_self_modules_convs_modules_3_buffers_eps_ = None
        mul_3 = add_6 * x_20
        add_6 = x_20 = None
        out_7 = out_6 + mul_3
        out_6 = mul_3 = None
        x_21 = torch._C._nn.linear(
            out_7,
            l_self_modules_convs_modules_3_modules_nn_modules_lins_modules_0_parameters_weight_,
            l_self_modules_convs_modules_3_modules_nn_modules_lins_modules_0_parameters_bias_,
        )
        out_7 = l_self_modules_convs_modules_3_modules_nn_modules_lins_modules_0_parameters_weight_ = l_self_modules_convs_modules_3_modules_nn_modules_lins_modules_0_parameters_bias_ = (None)
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
        x_26 = torch.nn.functional.relu(x_25, inplace=False)
        x_25 = None
        x_27 = torch.nn.functional.dropout(x_26, 0.0, False, False)
        x_26 = None
        edge_index_i_4 = l_edge_index_[1]
        edge_index_j_4 = l_edge_index_[0]
        l_edge_index_ = None
        x_j_4 = x_27.index_select(-2, edge_index_j_4)
        edge_index_j_4 = None
        view_4 = edge_index_i_4.view((-1, 1))
        edge_index_i_4 = None
        index_4 = view_4.expand_as(x_j_4)
        view_4 = None
        new_zeros_4 = x_j_4.new_zeros((1000, 256))
        out_8 = new_zeros_4.scatter_add_(0, index_4, x_j_4)
        new_zeros_4 = index_4 = x_j_4 = None
        add_8 = 1 + l_self_modules_convs_modules_4_buffers_eps_
        l_self_modules_convs_modules_4_buffers_eps_ = None
        mul_4 = add_8 * x_27
        add_8 = x_27 = None
        out_9 = out_8 + mul_4
        out_8 = mul_4 = None
        x_28 = torch._C._nn.linear(
            out_9,
            l_self_modules_convs_modules_4_modules_nn_modules_lins_modules_0_parameters_weight_,
            l_self_modules_convs_modules_4_modules_nn_modules_lins_modules_0_parameters_bias_,
        )
        out_9 = l_self_modules_convs_modules_4_modules_nn_modules_lins_modules_0_parameters_weight_ = l_self_modules_convs_modules_4_modules_nn_modules_lins_modules_0_parameters_bias_ = (None)
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
        return (x_32,)
