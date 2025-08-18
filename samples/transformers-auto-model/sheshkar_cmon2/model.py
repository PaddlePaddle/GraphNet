import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_stack0_0_: torch.Tensor,
        L_third_residual_: torch.Tensor,
        L_self_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_stack0_0_ = L_stack0_0_
        l_third_residual_ = L_third_residual_
        l_self_modules_encoder_attn_layer_norm_parameters_weight_ = (
            L_self_modules_encoder_attn_layer_norm_parameters_weight_
        )
        l_self_modules_encoder_attn_layer_norm_parameters_bias_ = (
            L_self_modules_encoder_attn_layer_norm_parameters_bias_
        )
        l_self_modules_fc1_parameters_weight_ = L_self_modules_fc1_parameters_weight_
        l_self_modules_fc1_parameters_bias_ = L_self_modules_fc1_parameters_bias_
        l_self_modules_fc2_parameters_weight_ = L_self_modules_fc2_parameters_weight_
        l_self_modules_fc2_parameters_bias_ = L_self_modules_fc2_parameters_bias_
        l_self_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_final_layer_norm_parameters_bias_
        )
        hidden_states = torch.nn.functional.dropout(l_stack0_0_, p=0.1, training=False)
        l_stack0_0_ = None
        hidden_states_1 = l_third_residual_ + hidden_states
        l_third_residual_ = hidden_states = None
        hidden_states_2 = torch.nn.functional.layer_norm(
            hidden_states_1,
            (256,),
            l_self_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_1 = (
            l_self_modules_encoder_attn_layer_norm_parameters_weight_
        ) = l_self_modules_encoder_attn_layer_norm_parameters_bias_ = None
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
        hidden_states_4 = torch.nn.functional.dropout(
            hidden_states_3, p=0.0, training=False
        )
        hidden_states_3 = None
        hidden_states_5 = torch._C._nn.linear(
            hidden_states_4,
            l_self_modules_fc2_parameters_weight_,
            l_self_modules_fc2_parameters_bias_,
        )
        hidden_states_4 = (
            l_self_modules_fc2_parameters_weight_
        ) = l_self_modules_fc2_parameters_bias_ = None
        hidden_states_6 = torch.nn.functional.dropout(
            hidden_states_5, p=0.1, training=False
        )
        hidden_states_5 = None
        hidden_states_7 = hidden_states_2 + hidden_states_6
        hidden_states_2 = hidden_states_6 = None
        hidden_states_8 = torch.nn.functional.layer_norm(
            hidden_states_7,
            (256,),
            l_self_modules_final_layer_norm_parameters_weight_,
            l_self_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_7 = (
            l_self_modules_final_layer_norm_parameters_weight_
        ) = l_self_modules_final_layer_norm_parameters_bias_ = None
        return (hidden_states_8,)
