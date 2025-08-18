import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        s61: torch.SymInt,
        L_stack0_0_: torch.Tensor,
        L_self_dropout: torch.Tensor,
        L_third_residual_: torch.Tensor,
        L_self_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_attn_layer_norm_eps: torch.Tensor,
        L_self_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_activation_dropout: torch.Tensor,
        L_self_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_final_layer_norm_eps: torch.Tensor,
    ):
        l_stack0_0_ = L_stack0_0_
        l_self_dropout = L_self_dropout
        l_third_residual_ = L_third_residual_
        l_self_modules_encoder_attn_layer_norm_parameters_weight_ = (
            L_self_modules_encoder_attn_layer_norm_parameters_weight_
        )
        l_self_modules_encoder_attn_layer_norm_parameters_bias_ = (
            L_self_modules_encoder_attn_layer_norm_parameters_bias_
        )
        l_self_modules_encoder_attn_layer_norm_eps = (
            L_self_modules_encoder_attn_layer_norm_eps
        )
        l_self_modules_fc1_parameters_weight_ = L_self_modules_fc1_parameters_weight_
        l_self_modules_fc1_parameters_bias_ = L_self_modules_fc1_parameters_bias_
        l_self_activation_dropout = L_self_activation_dropout
        l_self_modules_fc2_parameters_weight_ = L_self_modules_fc2_parameters_weight_
        l_self_modules_fc2_parameters_bias_ = L_self_modules_fc2_parameters_bias_
        l_self_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_final_layer_norm_parameters_bias_
        )
        l_self_modules_final_layer_norm_eps = L_self_modules_final_layer_norm_eps
        item = l_self_dropout.item()
        l_self_dropout = None
        hidden_states = torch.nn.functional.dropout(l_stack0_0_, p=item, training=False)
        l_stack0_0_ = None
        hidden_states_1 = l_third_residual_ + hidden_states
        l_third_residual_ = hidden_states = None
        item_1 = l_self_modules_encoder_attn_layer_norm_eps.item()
        l_self_modules_encoder_attn_layer_norm_eps = None
        hidden_states_2 = torch.nn.functional.layer_norm(
            hidden_states_1,
            (256,),
            l_self_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_encoder_attn_layer_norm_parameters_bias_,
            item_1,
        )
        hidden_states_1 = (
            l_self_modules_encoder_attn_layer_norm_parameters_weight_
        ) = l_self_modules_encoder_attn_layer_norm_parameters_bias_ = item_1 = None
        linear = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_fc1_parameters_weight_,
            l_self_modules_fc1_parameters_bias_,
        )
        l_self_modules_fc1_parameters_weight_ = (
            l_self_modules_fc1_parameters_bias_
        ) = None
        hidden_states_3 = torch.nn.functional.relu(linear, inplace=False)
        linear = None
        item_2 = l_self_activation_dropout.item()
        l_self_activation_dropout = None
        hidden_states_4 = torch.nn.functional.dropout(
            hidden_states_3, p=item_2, training=False
        )
        hidden_states_3 = item_2 = None
        hidden_states_5 = torch._C._nn.linear(
            hidden_states_4,
            l_self_modules_fc2_parameters_weight_,
            l_self_modules_fc2_parameters_bias_,
        )
        hidden_states_4 = (
            l_self_modules_fc2_parameters_weight_
        ) = l_self_modules_fc2_parameters_bias_ = None
        hidden_states_6 = torch.nn.functional.dropout(
            hidden_states_5, p=item, training=False
        )
        hidden_states_5 = item = None
        hidden_states_7 = hidden_states_2 + hidden_states_6
        hidden_states_2 = hidden_states_6 = None
        item_3 = l_self_modules_final_layer_norm_eps.item()
        l_self_modules_final_layer_norm_eps = None
        hidden_states_8 = torch.nn.functional.layer_norm(
            hidden_states_7,
            (256,),
            l_self_modules_final_layer_norm_parameters_weight_,
            l_self_modules_final_layer_norm_parameters_bias_,
            item_3,
        )
        hidden_states_7 = (
            l_self_modules_final_layer_norm_parameters_weight_
        ) = l_self_modules_final_layer_norm_parameters_bias_ = item_3 = None
        return (hidden_states_8,)
