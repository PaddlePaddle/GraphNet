import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_edge_lin_parameters_weight_: torch.nn.parameter.Parameter,
        L_edge_index_: torch.Tensor,
        L_self_modules_edge_lin_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_cat_lin1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_cat_lin1_parameters_bias_: torch.nn.parameter.Parameter,
        L_x_: torch.Tensor,
        L_self_modules_node_mlp_modules_lins_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_node_mlp_modules_lins_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_cat_lin2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_cat_lin2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_final_mlp_modules_lins_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_final_mlp_modules_lins_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_final_mlp_modules_norms_modules_0_modules_module_buffers_running_mean_: torch.Tensor,
        L_self_modules_final_mlp_modules_norms_modules_0_modules_module_buffers_running_var_: torch.Tensor,
        L_self_modules_final_mlp_modules_norms_modules_0_modules_module_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_final_mlp_modules_norms_modules_0_modules_module_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_final_mlp_modules_lins_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_final_mlp_modules_lins_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_final_mlp_modules_norms_modules_1_modules_module_buffers_running_mean_: torch.Tensor,
        L_self_modules_final_mlp_modules_norms_modules_1_modules_module_buffers_running_var_: torch.Tensor,
        L_self_modules_final_mlp_modules_norms_modules_1_modules_module_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_final_mlp_modules_norms_modules_1_modules_module_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_final_mlp_modules_lins_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_final_mlp_modules_lins_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_final_mlp_modules_norms_modules_2_modules_module_buffers_running_mean_: torch.Tensor,
        L_self_modules_final_mlp_modules_norms_modules_2_modules_module_buffers_running_var_: torch.Tensor,
        L_self_modules_final_mlp_modules_norms_modules_2_modules_module_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_final_mlp_modules_norms_modules_2_modules_module_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_final_mlp_modules_lins_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_final_mlp_modules_lins_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_final_mlp_modules_norms_modules_3_modules_module_buffers_running_mean_: torch.Tensor,
        L_self_modules_final_mlp_modules_norms_modules_3_modules_module_buffers_running_var_: torch.Tensor,
        L_self_modules_final_mlp_modules_norms_modules_3_modules_module_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_final_mlp_modules_norms_modules_3_modules_module_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_final_mlp_modules_lins_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_final_mlp_modules_lins_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_edge_lin_parameters_weight_ = (
            L_self_modules_edge_lin_parameters_weight_
        )
        l_edge_index_ = L_edge_index_
        l_self_modules_edge_lin_parameters_bias_ = (
            L_self_modules_edge_lin_parameters_bias_
        )
        l_self_modules_cat_lin1_parameters_weight_ = (
            L_self_modules_cat_lin1_parameters_weight_
        )
        l_self_modules_cat_lin1_parameters_bias_ = (
            L_self_modules_cat_lin1_parameters_bias_
        )
        l_x_ = L_x_
        l_self_modules_node_mlp_modules_lins_modules_0_parameters_weight_ = (
            L_self_modules_node_mlp_modules_lins_modules_0_parameters_weight_
        )
        l_self_modules_node_mlp_modules_lins_modules_0_parameters_bias_ = (
            L_self_modules_node_mlp_modules_lins_modules_0_parameters_bias_
        )
        l_self_modules_cat_lin2_parameters_weight_ = (
            L_self_modules_cat_lin2_parameters_weight_
        )
        l_self_modules_cat_lin2_parameters_bias_ = (
            L_self_modules_cat_lin2_parameters_bias_
        )
        l_self_modules_final_mlp_modules_lins_modules_0_parameters_weight_ = (
            L_self_modules_final_mlp_modules_lins_modules_0_parameters_weight_
        )
        l_self_modules_final_mlp_modules_lins_modules_0_parameters_bias_ = (
            L_self_modules_final_mlp_modules_lins_modules_0_parameters_bias_
        )
        l_self_modules_final_mlp_modules_norms_modules_0_modules_module_buffers_running_mean_ = L_self_modules_final_mlp_modules_norms_modules_0_modules_module_buffers_running_mean_
        l_self_modules_final_mlp_modules_norms_modules_0_modules_module_buffers_running_var_ = L_self_modules_final_mlp_modules_norms_modules_0_modules_module_buffers_running_var_
        l_self_modules_final_mlp_modules_norms_modules_0_modules_module_parameters_weight_ = L_self_modules_final_mlp_modules_norms_modules_0_modules_module_parameters_weight_
        l_self_modules_final_mlp_modules_norms_modules_0_modules_module_parameters_bias_ = L_self_modules_final_mlp_modules_norms_modules_0_modules_module_parameters_bias_
        l_self_modules_final_mlp_modules_lins_modules_1_parameters_weight_ = (
            L_self_modules_final_mlp_modules_lins_modules_1_parameters_weight_
        )
        l_self_modules_final_mlp_modules_lins_modules_1_parameters_bias_ = (
            L_self_modules_final_mlp_modules_lins_modules_1_parameters_bias_
        )
        l_self_modules_final_mlp_modules_norms_modules_1_modules_module_buffers_running_mean_ = L_self_modules_final_mlp_modules_norms_modules_1_modules_module_buffers_running_mean_
        l_self_modules_final_mlp_modules_norms_modules_1_modules_module_buffers_running_var_ = L_self_modules_final_mlp_modules_norms_modules_1_modules_module_buffers_running_var_
        l_self_modules_final_mlp_modules_norms_modules_1_modules_module_parameters_weight_ = L_self_modules_final_mlp_modules_norms_modules_1_modules_module_parameters_weight_
        l_self_modules_final_mlp_modules_norms_modules_1_modules_module_parameters_bias_ = L_self_modules_final_mlp_modules_norms_modules_1_modules_module_parameters_bias_
        l_self_modules_final_mlp_modules_lins_modules_2_parameters_weight_ = (
            L_self_modules_final_mlp_modules_lins_modules_2_parameters_weight_
        )
        l_self_modules_final_mlp_modules_lins_modules_2_parameters_bias_ = (
            L_self_modules_final_mlp_modules_lins_modules_2_parameters_bias_
        )
        l_self_modules_final_mlp_modules_norms_modules_2_modules_module_buffers_running_mean_ = L_self_modules_final_mlp_modules_norms_modules_2_modules_module_buffers_running_mean_
        l_self_modules_final_mlp_modules_norms_modules_2_modules_module_buffers_running_var_ = L_self_modules_final_mlp_modules_norms_modules_2_modules_module_buffers_running_var_
        l_self_modules_final_mlp_modules_norms_modules_2_modules_module_parameters_weight_ = L_self_modules_final_mlp_modules_norms_modules_2_modules_module_parameters_weight_
        l_self_modules_final_mlp_modules_norms_modules_2_modules_module_parameters_bias_ = L_self_modules_final_mlp_modules_norms_modules_2_modules_module_parameters_bias_
        l_self_modules_final_mlp_modules_lins_modules_3_parameters_weight_ = (
            L_self_modules_final_mlp_modules_lins_modules_3_parameters_weight_
        )
        l_self_modules_final_mlp_modules_lins_modules_3_parameters_bias_ = (
            L_self_modules_final_mlp_modules_lins_modules_3_parameters_bias_
        )
        l_self_modules_final_mlp_modules_norms_modules_3_modules_module_buffers_running_mean_ = L_self_modules_final_mlp_modules_norms_modules_3_modules_module_buffers_running_mean_
        l_self_modules_final_mlp_modules_norms_modules_3_modules_module_buffers_running_var_ = L_self_modules_final_mlp_modules_norms_modules_3_modules_module_buffers_running_var_
        l_self_modules_final_mlp_modules_norms_modules_3_modules_module_parameters_weight_ = L_self_modules_final_mlp_modules_norms_modules_3_modules_module_parameters_weight_
        l_self_modules_final_mlp_modules_norms_modules_3_modules_module_parameters_bias_ = L_self_modules_final_mlp_modules_norms_modules_3_modules_module_parameters_bias_
        l_self_modules_final_mlp_modules_lins_modules_4_parameters_weight_ = (
            L_self_modules_final_mlp_modules_lins_modules_4_parameters_weight_
        )
        l_self_modules_final_mlp_modules_lins_modules_4_parameters_bias_ = (
            L_self_modules_final_mlp_modules_lins_modules_4_parameters_bias_
        )
        edge_index_i = l_edge_index_[1]
        edge_index_j = l_edge_index_[0]
        l_edge_index_ = None
        weight_j = l_self_modules_edge_lin_parameters_weight_.index_select(
            -2, edge_index_j
        )
        l_self_modules_edge_lin_parameters_weight_ = edge_index_j = None
        view = edge_index_i.view((-1, 1))
        edge_index_i = None
        index = view.expand_as(weight_j)
        view = None
        new_zeros = weight_j.new_zeros((1000, 128))
        out = new_zeros.scatter_add_(0, index, weight_j)
        new_zeros = index = weight_j = None
        out_1 = out + l_self_modules_edge_lin_parameters_bias_
        out = l_self_modules_edge_lin_parameters_bias_ = None
        linear = torch._C._nn.linear(
            out_1,
            l_self_modules_cat_lin1_parameters_weight_,
            l_self_modules_cat_lin1_parameters_bias_,
        )
        l_self_modules_cat_lin1_parameters_weight_ = (
            l_self_modules_cat_lin1_parameters_bias_
        ) = None
        out_2 = out_1 + linear
        out_1 = linear = None
        x = torch._C._nn.linear(
            l_x_,
            l_self_modules_node_mlp_modules_lins_modules_0_parameters_weight_,
            l_self_modules_node_mlp_modules_lins_modules_0_parameters_bias_,
        )
        l_x_ = (
            l_self_modules_node_mlp_modules_lins_modules_0_parameters_weight_
        ) = l_self_modules_node_mlp_modules_lins_modules_0_parameters_bias_ = None
        x_1 = torch.nn.functional.dropout(x, p=0.0, training=False)
        x = None
        out_3 = out_2 + x_1
        out_2 = None
        linear_2 = torch._C._nn.linear(
            x_1,
            l_self_modules_cat_lin2_parameters_weight_,
            l_self_modules_cat_lin2_parameters_bias_,
        )
        x_1 = (
            l_self_modules_cat_lin2_parameters_weight_
        ) = l_self_modules_cat_lin2_parameters_bias_ = None
        out_4 = out_3 + linear_2
        out_3 = linear_2 = None
        relu_ = out_4.relu_()
        out_4 = None
        x_2 = torch._C._nn.linear(
            relu_,
            l_self_modules_final_mlp_modules_lins_modules_0_parameters_weight_,
            l_self_modules_final_mlp_modules_lins_modules_0_parameters_bias_,
        )
        relu_ = (
            l_self_modules_final_mlp_modules_lins_modules_0_parameters_weight_
        ) = l_self_modules_final_mlp_modules_lins_modules_0_parameters_bias_ = None
        x_3 = torch.nn.functional.relu(x_2, inplace=False)
        x_2 = None
        x_4 = torch.nn.functional.batch_norm(
            x_3,
            l_self_modules_final_mlp_modules_norms_modules_0_modules_module_buffers_running_mean_,
            l_self_modules_final_mlp_modules_norms_modules_0_modules_module_buffers_running_var_,
            l_self_modules_final_mlp_modules_norms_modules_0_modules_module_parameters_weight_,
            l_self_modules_final_mlp_modules_norms_modules_0_modules_module_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_3 = l_self_modules_final_mlp_modules_norms_modules_0_modules_module_buffers_running_mean_ = l_self_modules_final_mlp_modules_norms_modules_0_modules_module_buffers_running_var_ = l_self_modules_final_mlp_modules_norms_modules_0_modules_module_parameters_weight_ = l_self_modules_final_mlp_modules_norms_modules_0_modules_module_parameters_bias_ = (None)
        x_5 = torch.nn.functional.dropout(x_4, p=0.0, training=False)
        x_4 = None
        x_6 = torch._C._nn.linear(
            x_5,
            l_self_modules_final_mlp_modules_lins_modules_1_parameters_weight_,
            l_self_modules_final_mlp_modules_lins_modules_1_parameters_bias_,
        )
        x_5 = (
            l_self_modules_final_mlp_modules_lins_modules_1_parameters_weight_
        ) = l_self_modules_final_mlp_modules_lins_modules_1_parameters_bias_ = None
        x_7 = torch.nn.functional.relu(x_6, inplace=False)
        x_6 = None
        x_8 = torch.nn.functional.batch_norm(
            x_7,
            l_self_modules_final_mlp_modules_norms_modules_1_modules_module_buffers_running_mean_,
            l_self_modules_final_mlp_modules_norms_modules_1_modules_module_buffers_running_var_,
            l_self_modules_final_mlp_modules_norms_modules_1_modules_module_parameters_weight_,
            l_self_modules_final_mlp_modules_norms_modules_1_modules_module_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_7 = l_self_modules_final_mlp_modules_norms_modules_1_modules_module_buffers_running_mean_ = l_self_modules_final_mlp_modules_norms_modules_1_modules_module_buffers_running_var_ = l_self_modules_final_mlp_modules_norms_modules_1_modules_module_parameters_weight_ = l_self_modules_final_mlp_modules_norms_modules_1_modules_module_parameters_bias_ = (None)
        x_9 = torch.nn.functional.dropout(x_8, p=0.0, training=False)
        x_8 = None
        x_10 = torch._C._nn.linear(
            x_9,
            l_self_modules_final_mlp_modules_lins_modules_2_parameters_weight_,
            l_self_modules_final_mlp_modules_lins_modules_2_parameters_bias_,
        )
        x_9 = (
            l_self_modules_final_mlp_modules_lins_modules_2_parameters_weight_
        ) = l_self_modules_final_mlp_modules_lins_modules_2_parameters_bias_ = None
        x_11 = torch.nn.functional.relu(x_10, inplace=False)
        x_10 = None
        x_12 = torch.nn.functional.batch_norm(
            x_11,
            l_self_modules_final_mlp_modules_norms_modules_2_modules_module_buffers_running_mean_,
            l_self_modules_final_mlp_modules_norms_modules_2_modules_module_buffers_running_var_,
            l_self_modules_final_mlp_modules_norms_modules_2_modules_module_parameters_weight_,
            l_self_modules_final_mlp_modules_norms_modules_2_modules_module_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_11 = l_self_modules_final_mlp_modules_norms_modules_2_modules_module_buffers_running_mean_ = l_self_modules_final_mlp_modules_norms_modules_2_modules_module_buffers_running_var_ = l_self_modules_final_mlp_modules_norms_modules_2_modules_module_parameters_weight_ = l_self_modules_final_mlp_modules_norms_modules_2_modules_module_parameters_bias_ = (None)
        x_13 = torch.nn.functional.dropout(x_12, p=0.0, training=False)
        x_12 = None
        x_14 = torch._C._nn.linear(
            x_13,
            l_self_modules_final_mlp_modules_lins_modules_3_parameters_weight_,
            l_self_modules_final_mlp_modules_lins_modules_3_parameters_bias_,
        )
        x_13 = (
            l_self_modules_final_mlp_modules_lins_modules_3_parameters_weight_
        ) = l_self_modules_final_mlp_modules_lins_modules_3_parameters_bias_ = None
        x_15 = torch.nn.functional.relu(x_14, inplace=False)
        x_14 = None
        x_16 = torch.nn.functional.batch_norm(
            x_15,
            l_self_modules_final_mlp_modules_norms_modules_3_modules_module_buffers_running_mean_,
            l_self_modules_final_mlp_modules_norms_modules_3_modules_module_buffers_running_var_,
            l_self_modules_final_mlp_modules_norms_modules_3_modules_module_parameters_weight_,
            l_self_modules_final_mlp_modules_norms_modules_3_modules_module_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_15 = l_self_modules_final_mlp_modules_norms_modules_3_modules_module_buffers_running_mean_ = l_self_modules_final_mlp_modules_norms_modules_3_modules_module_buffers_running_var_ = l_self_modules_final_mlp_modules_norms_modules_3_modules_module_parameters_weight_ = l_self_modules_final_mlp_modules_norms_modules_3_modules_module_parameters_bias_ = (None)
        x_17 = torch.nn.functional.dropout(x_16, p=0.0, training=False)
        x_16 = None
        x_18 = torch._C._nn.linear(
            x_17,
            l_self_modules_final_mlp_modules_lins_modules_4_parameters_weight_,
            l_self_modules_final_mlp_modules_lins_modules_4_parameters_bias_,
        )
        x_17 = (
            l_self_modules_final_mlp_modules_lins_modules_4_parameters_weight_
        ) = l_self_modules_final_mlp_modules_lins_modules_4_parameters_bias_ = None
        x_19 = torch.nn.functional.dropout(x_18, p=0.0, training=False)
        x_18 = None
        return (x_19,)
