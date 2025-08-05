import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        s0: torch.SymInt,
        L_stack0_: torch.Tensor,
        L_self_modules_branches_modules_0_modules_0_modules_self_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_0_modules_0_modules_self_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_0_modules_0_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_0_modules_0_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_0_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_0_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_0_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_0_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_0_modules_0_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_0_modules_0_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_0_modules_0_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_0_modules_0_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_0_modules_1_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_0_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_0_modules_2_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_0_modules_2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_0_modules_2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_0_modules_2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_1_modules_0_modules_self_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_1_modules_0_modules_self_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_1_modules_0_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_1_modules_0_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_1_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_1_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_1_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_1_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_1_modules_0_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_1_modules_0_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_1_modules_0_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_1_modules_0_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_1_modules_1_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_1_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_1_modules_2_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_1_modules_2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_1_modules_2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_1_modules_2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_2_modules_0_modules_self_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_2_modules_0_modules_self_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_2_modules_0_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_2_modules_0_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_2_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_2_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_2_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_2_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_2_modules_0_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_2_modules_0_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_2_modules_0_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_2_modules_0_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_2_modules_1_parameters_alpha_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_2_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_2_modules_2_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_2_modules_2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_2_modules_2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_branches_modules_2_modules_2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_stack0_ = L_stack0_
        l_self_modules_branches_modules_0_modules_0_modules_self_attn_parameters_in_proj_bias_ = L_self_modules_branches_modules_0_modules_0_modules_self_attn_parameters_in_proj_bias_
        l_self_modules_branches_modules_0_modules_0_modules_self_attn_parameters_in_proj_weight_ = L_self_modules_branches_modules_0_modules_0_modules_self_attn_parameters_in_proj_weight_
        l_self_modules_branches_modules_0_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_branches_modules_0_modules_0_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_branches_modules_0_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_branches_modules_0_modules_0_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_branches_modules_0_modules_0_modules_norm1_parameters_weight_ = (
            L_self_modules_branches_modules_0_modules_0_modules_norm1_parameters_weight_
        )
        l_self_modules_branches_modules_0_modules_0_modules_norm1_parameters_bias_ = (
            L_self_modules_branches_modules_0_modules_0_modules_norm1_parameters_bias_
        )
        l_self_modules_branches_modules_0_modules_0_modules_norm2_parameters_weight_ = (
            L_self_modules_branches_modules_0_modules_0_modules_norm2_parameters_weight_
        )
        l_self_modules_branches_modules_0_modules_0_modules_norm2_parameters_bias_ = (
            L_self_modules_branches_modules_0_modules_0_modules_norm2_parameters_bias_
        )
        l_self_modules_branches_modules_0_modules_0_modules_linear1_parameters_weight_ = L_self_modules_branches_modules_0_modules_0_modules_linear1_parameters_weight_
        l_self_modules_branches_modules_0_modules_0_modules_linear1_parameters_bias_ = (
            L_self_modules_branches_modules_0_modules_0_modules_linear1_parameters_bias_
        )
        l_self_modules_branches_modules_0_modules_0_modules_linear2_parameters_weight_ = L_self_modules_branches_modules_0_modules_0_modules_linear2_parameters_weight_
        l_self_modules_branches_modules_0_modules_0_modules_linear2_parameters_bias_ = (
            L_self_modules_branches_modules_0_modules_0_modules_linear2_parameters_bias_
        )
        l_self_modules_branches_modules_0_modules_1_parameters_alpha_ = (
            L_self_modules_branches_modules_0_modules_1_parameters_alpha_
        )
        l_self_modules_branches_modules_0_modules_2_modules_0_parameters_weight_ = (
            L_self_modules_branches_modules_0_modules_2_modules_0_parameters_weight_
        )
        l_self_modules_branches_modules_0_modules_2_modules_0_parameters_bias_ = (
            L_self_modules_branches_modules_0_modules_2_modules_0_parameters_bias_
        )
        l_self_modules_branches_modules_0_modules_2_modules_1_parameters_weight_ = (
            L_self_modules_branches_modules_0_modules_2_modules_1_parameters_weight_
        )
        l_self_modules_branches_modules_0_modules_2_modules_2_parameters_weight_ = (
            L_self_modules_branches_modules_0_modules_2_modules_2_parameters_weight_
        )
        l_self_modules_branches_modules_0_modules_2_modules_2_parameters_bias_ = (
            L_self_modules_branches_modules_0_modules_2_modules_2_parameters_bias_
        )
        l_self_modules_branches_modules_1_modules_0_modules_self_attn_parameters_in_proj_bias_ = L_self_modules_branches_modules_1_modules_0_modules_self_attn_parameters_in_proj_bias_
        l_self_modules_branches_modules_1_modules_0_modules_self_attn_parameters_in_proj_weight_ = L_self_modules_branches_modules_1_modules_0_modules_self_attn_parameters_in_proj_weight_
        l_self_modules_branches_modules_1_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_branches_modules_1_modules_0_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_branches_modules_1_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_branches_modules_1_modules_0_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_branches_modules_1_modules_0_modules_norm1_parameters_weight_ = (
            L_self_modules_branches_modules_1_modules_0_modules_norm1_parameters_weight_
        )
        l_self_modules_branches_modules_1_modules_0_modules_norm1_parameters_bias_ = (
            L_self_modules_branches_modules_1_modules_0_modules_norm1_parameters_bias_
        )
        l_self_modules_branches_modules_1_modules_0_modules_norm2_parameters_weight_ = (
            L_self_modules_branches_modules_1_modules_0_modules_norm2_parameters_weight_
        )
        l_self_modules_branches_modules_1_modules_0_modules_norm2_parameters_bias_ = (
            L_self_modules_branches_modules_1_modules_0_modules_norm2_parameters_bias_
        )
        l_self_modules_branches_modules_1_modules_0_modules_linear1_parameters_weight_ = L_self_modules_branches_modules_1_modules_0_modules_linear1_parameters_weight_
        l_self_modules_branches_modules_1_modules_0_modules_linear1_parameters_bias_ = (
            L_self_modules_branches_modules_1_modules_0_modules_linear1_parameters_bias_
        )
        l_self_modules_branches_modules_1_modules_0_modules_linear2_parameters_weight_ = L_self_modules_branches_modules_1_modules_0_modules_linear2_parameters_weight_
        l_self_modules_branches_modules_1_modules_0_modules_linear2_parameters_bias_ = (
            L_self_modules_branches_modules_1_modules_0_modules_linear2_parameters_bias_
        )
        l_self_modules_branches_modules_1_modules_1_parameters_alpha_ = (
            L_self_modules_branches_modules_1_modules_1_parameters_alpha_
        )
        l_self_modules_branches_modules_1_modules_2_modules_0_parameters_weight_ = (
            L_self_modules_branches_modules_1_modules_2_modules_0_parameters_weight_
        )
        l_self_modules_branches_modules_1_modules_2_modules_0_parameters_bias_ = (
            L_self_modules_branches_modules_1_modules_2_modules_0_parameters_bias_
        )
        l_self_modules_branches_modules_1_modules_2_modules_1_parameters_weight_ = (
            L_self_modules_branches_modules_1_modules_2_modules_1_parameters_weight_
        )
        l_self_modules_branches_modules_1_modules_2_modules_2_parameters_weight_ = (
            L_self_modules_branches_modules_1_modules_2_modules_2_parameters_weight_
        )
        l_self_modules_branches_modules_1_modules_2_modules_2_parameters_bias_ = (
            L_self_modules_branches_modules_1_modules_2_modules_2_parameters_bias_
        )
        l_self_modules_branches_modules_2_modules_0_modules_self_attn_parameters_in_proj_bias_ = L_self_modules_branches_modules_2_modules_0_modules_self_attn_parameters_in_proj_bias_
        l_self_modules_branches_modules_2_modules_0_modules_self_attn_parameters_in_proj_weight_ = L_self_modules_branches_modules_2_modules_0_modules_self_attn_parameters_in_proj_weight_
        l_self_modules_branches_modules_2_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_branches_modules_2_modules_0_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_branches_modules_2_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_branches_modules_2_modules_0_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_branches_modules_2_modules_0_modules_norm1_parameters_weight_ = (
            L_self_modules_branches_modules_2_modules_0_modules_norm1_parameters_weight_
        )
        l_self_modules_branches_modules_2_modules_0_modules_norm1_parameters_bias_ = (
            L_self_modules_branches_modules_2_modules_0_modules_norm1_parameters_bias_
        )
        l_self_modules_branches_modules_2_modules_0_modules_norm2_parameters_weight_ = (
            L_self_modules_branches_modules_2_modules_0_modules_norm2_parameters_weight_
        )
        l_self_modules_branches_modules_2_modules_0_modules_norm2_parameters_bias_ = (
            L_self_modules_branches_modules_2_modules_0_modules_norm2_parameters_bias_
        )
        l_self_modules_branches_modules_2_modules_0_modules_linear1_parameters_weight_ = L_self_modules_branches_modules_2_modules_0_modules_linear1_parameters_weight_
        l_self_modules_branches_modules_2_modules_0_modules_linear1_parameters_bias_ = (
            L_self_modules_branches_modules_2_modules_0_modules_linear1_parameters_bias_
        )
        l_self_modules_branches_modules_2_modules_0_modules_linear2_parameters_weight_ = L_self_modules_branches_modules_2_modules_0_modules_linear2_parameters_weight_
        l_self_modules_branches_modules_2_modules_0_modules_linear2_parameters_bias_ = (
            L_self_modules_branches_modules_2_modules_0_modules_linear2_parameters_bias_
        )
        l_self_modules_branches_modules_2_modules_1_parameters_alpha_ = (
            L_self_modules_branches_modules_2_modules_1_parameters_alpha_
        )
        l_self_modules_branches_modules_2_modules_2_modules_0_parameters_weight_ = (
            L_self_modules_branches_modules_2_modules_2_modules_0_parameters_weight_
        )
        l_self_modules_branches_modules_2_modules_2_modules_0_parameters_bias_ = (
            L_self_modules_branches_modules_2_modules_2_modules_0_parameters_bias_
        )
        l_self_modules_branches_modules_2_modules_2_modules_1_parameters_weight_ = (
            L_self_modules_branches_modules_2_modules_2_modules_1_parameters_weight_
        )
        l_self_modules_branches_modules_2_modules_2_modules_2_parameters_weight_ = (
            L_self_modules_branches_modules_2_modules_2_modules_2_parameters_weight_
        )
        l_self_modules_branches_modules_2_modules_2_modules_2_parameters_bias_ = (
            L_self_modules_branches_modules_2_modules_2_modules_2_parameters_bias_
        )
        query = l_stack0_.transpose(1, 0)
        multi_head_attention_forward = torch.nn.functional.multi_head_attention_forward(
            query,
            query,
            query,
            256,
            4,
            l_self_modules_branches_modules_0_modules_0_modules_self_attn_parameters_in_proj_weight_,
            l_self_modules_branches_modules_0_modules_0_modules_self_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_branches_modules_0_modules_0_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_branches_modules_0_modules_0_modules_self_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=False,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        query = l_self_modules_branches_modules_0_modules_0_modules_self_attn_parameters_in_proj_weight_ = l_self_modules_branches_modules_0_modules_0_modules_self_attn_parameters_in_proj_bias_ = l_self_modules_branches_modules_0_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_branches_modules_0_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output = multi_head_attention_forward[0]
        multi_head_attention_forward = None
        x = attn_output.transpose(1, 0)
        attn_output = None
        dropout = torch.nn.functional.dropout(x, 0.0, False, False)
        x = None
        add = l_stack0_ + dropout
        dropout = None
        x_1 = torch.nn.functional.layer_norm(
            add,
            (256,),
            l_self_modules_branches_modules_0_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_branches_modules_0_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        add = (
            l_self_modules_branches_modules_0_modules_0_modules_norm1_parameters_weight_
        ) = (
            l_self_modules_branches_modules_0_modules_0_modules_norm1_parameters_bias_
        ) = None
        linear = torch._C._nn.linear(
            x_1,
            l_self_modules_branches_modules_0_modules_0_modules_linear1_parameters_weight_,
            l_self_modules_branches_modules_0_modules_0_modules_linear1_parameters_bias_,
        )
        l_self_modules_branches_modules_0_modules_0_modules_linear1_parameters_weight_ = (
            l_self_modules_branches_modules_0_modules_0_modules_linear1_parameters_bias_
        ) = None
        relu = torch.nn.functional.relu(linear)
        linear = None
        dropout_1 = torch.nn.functional.dropout(relu, 0.0, False, False)
        relu = None
        x_2 = torch._C._nn.linear(
            dropout_1,
            l_self_modules_branches_modules_0_modules_0_modules_linear2_parameters_weight_,
            l_self_modules_branches_modules_0_modules_0_modules_linear2_parameters_bias_,
        )
        dropout_1 = l_self_modules_branches_modules_0_modules_0_modules_linear2_parameters_weight_ = (
            l_self_modules_branches_modules_0_modules_0_modules_linear2_parameters_bias_
        ) = None
        dropout_2 = torch.nn.functional.dropout(x_2, 0.0, False, False)
        x_2 = None
        add_1 = x_1 + dropout_2
        x_1 = dropout_2 = None
        x_3 = torch.nn.functional.layer_norm(
            add_1,
            (256,),
            l_self_modules_branches_modules_0_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_branches_modules_0_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        add_1 = (
            l_self_modules_branches_modules_0_modules_0_modules_norm2_parameters_weight_
        ) = (
            l_self_modules_branches_modules_0_modules_0_modules_norm2_parameters_bias_
        ) = None
        mul = torch.mul(
            x_3, l_self_modules_branches_modules_0_modules_1_parameters_alpha_
        )
        l_self_modules_branches_modules_0_modules_1_parameters_alpha_ = None
        weight = torch.nn.functional.softmax(mul, 1, _stacklevel=5)
        mul = None
        mul_1 = torch.mul(x_3, weight)
        x_3 = weight = None
        out = torch.sum(mul_1, dim=1)
        mul_1 = None
        input_1 = torch._C._nn.linear(
            out,
            l_self_modules_branches_modules_0_modules_2_modules_0_parameters_weight_,
            l_self_modules_branches_modules_0_modules_2_modules_0_parameters_bias_,
        )
        out = (
            l_self_modules_branches_modules_0_modules_2_modules_0_parameters_weight_
        ) = (
            l_self_modules_branches_modules_0_modules_2_modules_0_parameters_bias_
        ) = None
        input_2 = torch.prelu(
            input_1,
            l_self_modules_branches_modules_0_modules_2_modules_1_parameters_weight_,
        )
        input_1 = (
            l_self_modules_branches_modules_0_modules_2_modules_1_parameters_weight_
        ) = None
        input_3 = torch._C._nn.linear(
            input_2,
            l_self_modules_branches_modules_0_modules_2_modules_2_parameters_weight_,
            l_self_modules_branches_modules_0_modules_2_modules_2_parameters_bias_,
        )
        input_2 = (
            l_self_modules_branches_modules_0_modules_2_modules_2_parameters_weight_
        ) = (
            l_self_modules_branches_modules_0_modules_2_modules_2_parameters_bias_
        ) = None
        sigmoid = torch.sigmoid(input_3)
        input_3 = None
        mul_2 = sigmoid * 1.0
        sigmoid = None
        out_1 = mul_2 + 0.0
        mul_2 = None
        squeeze = out_1.squeeze(dim=1)
        out_1 = None
        query_1 = l_stack0_.transpose(1, 0)
        multi_head_attention_forward_1 = torch.nn.functional.multi_head_attention_forward(
            query_1,
            query_1,
            query_1,
            256,
            4,
            l_self_modules_branches_modules_1_modules_0_modules_self_attn_parameters_in_proj_weight_,
            l_self_modules_branches_modules_1_modules_0_modules_self_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_branches_modules_1_modules_0_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_branches_modules_1_modules_0_modules_self_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=False,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        query_1 = l_self_modules_branches_modules_1_modules_0_modules_self_attn_parameters_in_proj_weight_ = l_self_modules_branches_modules_1_modules_0_modules_self_attn_parameters_in_proj_bias_ = l_self_modules_branches_modules_1_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_branches_modules_1_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_1 = multi_head_attention_forward_1[0]
        multi_head_attention_forward_1 = None
        x_4 = attn_output_1.transpose(1, 0)
        attn_output_1 = None
        dropout_3 = torch.nn.functional.dropout(x_4, 0.0, False, False)
        x_4 = None
        add_3 = l_stack0_ + dropout_3
        dropout_3 = None
        x_5 = torch.nn.functional.layer_norm(
            add_3,
            (256,),
            l_self_modules_branches_modules_1_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_branches_modules_1_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        add_3 = (
            l_self_modules_branches_modules_1_modules_0_modules_norm1_parameters_weight_
        ) = (
            l_self_modules_branches_modules_1_modules_0_modules_norm1_parameters_bias_
        ) = None
        linear_4 = torch._C._nn.linear(
            x_5,
            l_self_modules_branches_modules_1_modules_0_modules_linear1_parameters_weight_,
            l_self_modules_branches_modules_1_modules_0_modules_linear1_parameters_bias_,
        )
        l_self_modules_branches_modules_1_modules_0_modules_linear1_parameters_weight_ = (
            l_self_modules_branches_modules_1_modules_0_modules_linear1_parameters_bias_
        ) = None
        relu_1 = torch.nn.functional.relu(linear_4)
        linear_4 = None
        dropout_4 = torch.nn.functional.dropout(relu_1, 0.0, False, False)
        relu_1 = None
        x_6 = torch._C._nn.linear(
            dropout_4,
            l_self_modules_branches_modules_1_modules_0_modules_linear2_parameters_weight_,
            l_self_modules_branches_modules_1_modules_0_modules_linear2_parameters_bias_,
        )
        dropout_4 = l_self_modules_branches_modules_1_modules_0_modules_linear2_parameters_weight_ = (
            l_self_modules_branches_modules_1_modules_0_modules_linear2_parameters_bias_
        ) = None
        dropout_5 = torch.nn.functional.dropout(x_6, 0.0, False, False)
        x_6 = None
        add_4 = x_5 + dropout_5
        x_5 = dropout_5 = None
        x_7 = torch.nn.functional.layer_norm(
            add_4,
            (256,),
            l_self_modules_branches_modules_1_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_branches_modules_1_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        add_4 = (
            l_self_modules_branches_modules_1_modules_0_modules_norm2_parameters_weight_
        ) = (
            l_self_modules_branches_modules_1_modules_0_modules_norm2_parameters_bias_
        ) = None
        mul_3 = torch.mul(
            x_7, l_self_modules_branches_modules_1_modules_1_parameters_alpha_
        )
        l_self_modules_branches_modules_1_modules_1_parameters_alpha_ = None
        weight_1 = torch.nn.functional.softmax(mul_3, 1, _stacklevel=5)
        mul_3 = None
        mul_4 = torch.mul(x_7, weight_1)
        x_7 = weight_1 = None
        out_2 = torch.sum(mul_4, dim=1)
        mul_4 = None
        input_4 = torch._C._nn.linear(
            out_2,
            l_self_modules_branches_modules_1_modules_2_modules_0_parameters_weight_,
            l_self_modules_branches_modules_1_modules_2_modules_0_parameters_bias_,
        )
        out_2 = (
            l_self_modules_branches_modules_1_modules_2_modules_0_parameters_weight_
        ) = (
            l_self_modules_branches_modules_1_modules_2_modules_0_parameters_bias_
        ) = None
        input_5 = torch.prelu(
            input_4,
            l_self_modules_branches_modules_1_modules_2_modules_1_parameters_weight_,
        )
        input_4 = (
            l_self_modules_branches_modules_1_modules_2_modules_1_parameters_weight_
        ) = None
        input_6 = torch._C._nn.linear(
            input_5,
            l_self_modules_branches_modules_1_modules_2_modules_2_parameters_weight_,
            l_self_modules_branches_modules_1_modules_2_modules_2_parameters_bias_,
        )
        input_5 = (
            l_self_modules_branches_modules_1_modules_2_modules_2_parameters_weight_
        ) = (
            l_self_modules_branches_modules_1_modules_2_modules_2_parameters_bias_
        ) = None
        sigmoid_1 = torch.sigmoid(input_6)
        input_6 = None
        mul_5 = sigmoid_1 * 3.6438887493362575
        sigmoid_1 = None
        out_3 = mul_5 + 1.0
        mul_5 = None
        squeeze_1 = out_3.squeeze(dim=1)
        out_3 = None
        query_2 = l_stack0_.transpose(1, 0)
        multi_head_attention_forward_2 = torch.nn.functional.multi_head_attention_forward(
            query_2,
            query_2,
            query_2,
            256,
            4,
            l_self_modules_branches_modules_2_modules_0_modules_self_attn_parameters_in_proj_weight_,
            l_self_modules_branches_modules_2_modules_0_modules_self_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_branches_modules_2_modules_0_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_branches_modules_2_modules_0_modules_self_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=False,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        query_2 = l_self_modules_branches_modules_2_modules_0_modules_self_attn_parameters_in_proj_weight_ = l_self_modules_branches_modules_2_modules_0_modules_self_attn_parameters_in_proj_bias_ = l_self_modules_branches_modules_2_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_branches_modules_2_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_2 = multi_head_attention_forward_2[0]
        multi_head_attention_forward_2 = None
        x_8 = attn_output_2.transpose(1, 0)
        attn_output_2 = None
        dropout_6 = torch.nn.functional.dropout(x_8, 0.0, False, False)
        x_8 = None
        add_6 = l_stack0_ + dropout_6
        l_stack0_ = dropout_6 = None
        x_9 = torch.nn.functional.layer_norm(
            add_6,
            (256,),
            l_self_modules_branches_modules_2_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_branches_modules_2_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        add_6 = (
            l_self_modules_branches_modules_2_modules_0_modules_norm1_parameters_weight_
        ) = (
            l_self_modules_branches_modules_2_modules_0_modules_norm1_parameters_bias_
        ) = None
        linear_8 = torch._C._nn.linear(
            x_9,
            l_self_modules_branches_modules_2_modules_0_modules_linear1_parameters_weight_,
            l_self_modules_branches_modules_2_modules_0_modules_linear1_parameters_bias_,
        )
        l_self_modules_branches_modules_2_modules_0_modules_linear1_parameters_weight_ = (
            l_self_modules_branches_modules_2_modules_0_modules_linear1_parameters_bias_
        ) = None
        relu_2 = torch.nn.functional.relu(linear_8)
        linear_8 = None
        dropout_7 = torch.nn.functional.dropout(relu_2, 0.0, False, False)
        relu_2 = None
        x_10 = torch._C._nn.linear(
            dropout_7,
            l_self_modules_branches_modules_2_modules_0_modules_linear2_parameters_weight_,
            l_self_modules_branches_modules_2_modules_0_modules_linear2_parameters_bias_,
        )
        dropout_7 = l_self_modules_branches_modules_2_modules_0_modules_linear2_parameters_weight_ = (
            l_self_modules_branches_modules_2_modules_0_modules_linear2_parameters_bias_
        ) = None
        dropout_8 = torch.nn.functional.dropout(x_10, 0.0, False, False)
        x_10 = None
        add_7 = x_9 + dropout_8
        x_9 = dropout_8 = None
        x_11 = torch.nn.functional.layer_norm(
            add_7,
            (256,),
            l_self_modules_branches_modules_2_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_branches_modules_2_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        add_7 = (
            l_self_modules_branches_modules_2_modules_0_modules_norm2_parameters_weight_
        ) = (
            l_self_modules_branches_modules_2_modules_0_modules_norm2_parameters_bias_
        ) = None
        mul_6 = torch.mul(
            x_11, l_self_modules_branches_modules_2_modules_1_parameters_alpha_
        )
        l_self_modules_branches_modules_2_modules_1_parameters_alpha_ = None
        weight_2 = torch.nn.functional.softmax(mul_6, 1, _stacklevel=5)
        mul_6 = None
        mul_7 = torch.mul(x_11, weight_2)
        x_11 = weight_2 = None
        out_4 = torch.sum(mul_7, dim=1)
        mul_7 = None
        input_7 = torch._C._nn.linear(
            out_4,
            l_self_modules_branches_modules_2_modules_2_modules_0_parameters_weight_,
            l_self_modules_branches_modules_2_modules_2_modules_0_parameters_bias_,
        )
        out_4 = (
            l_self_modules_branches_modules_2_modules_2_modules_0_parameters_weight_
        ) = (
            l_self_modules_branches_modules_2_modules_2_modules_0_parameters_bias_
        ) = None
        input_8 = torch.prelu(
            input_7,
            l_self_modules_branches_modules_2_modules_2_modules_1_parameters_weight_,
        )
        input_7 = (
            l_self_modules_branches_modules_2_modules_2_modules_1_parameters_weight_
        ) = None
        input_9 = torch._C._nn.linear(
            input_8,
            l_self_modules_branches_modules_2_modules_2_modules_2_parameters_weight_,
            l_self_modules_branches_modules_2_modules_2_modules_2_parameters_bias_,
        )
        input_8 = (
            l_self_modules_branches_modules_2_modules_2_modules_2_parameters_weight_
        ) = (
            l_self_modules_branches_modules_2_modules_2_modules_2_parameters_bias_
        ) = None
        squeeze_2 = input_9.squeeze(dim=1)
        input_9 = None
        return (squeeze, squeeze_1, squeeze_2)
