
class GraphModule(torch.nn.Module):

    def forward(self, L_self_modules_features_modules_0_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_features_modules_0_parameters_bias_ : torch.nn.parameter.Parameter, s1 : torch.SymInt, L_x_ : torch.Tensor, L_self_modules_features_modules_3_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_features_modules_3_parameters_bias_ : torch.nn.parameter.Parameter, L_self_modules_features_modules_6_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_features_modules_6_parameters_bias_ : torch.nn.parameter.Parameter, L_self_modules_features_modules_8_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_features_modules_8_parameters_bias_ : torch.nn.parameter.Parameter, L_self_modules_features_modules_10_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_features_modules_10_parameters_bias_ : torch.nn.parameter.Parameter, L_self_modules_classifier_modules_1_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_classifier_modules_1_parameters_bias_ : torch.nn.parameter.Parameter, L_self_modules_classifier_modules_4_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_classifier_modules_4_parameters_bias_ : torch.nn.parameter.Parameter, L_self_modules_classifier_modules_6_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_classifier_modules_6_parameters_bias_ : torch.nn.parameter.Parameter):
        l_self_modules_features_modules_0_parameters_weight_ = L_self_modules_features_modules_0_parameters_weight_
        l_self_modules_features_modules_0_parameters_bias_ = L_self_modules_features_modules_0_parameters_bias_
        l_x_ = L_x_
        l_self_modules_features_modules_3_parameters_weight_ = L_self_modules_features_modules_3_parameters_weight_
        l_self_modules_features_modules_3_parameters_bias_ = L_self_modules_features_modules_3_parameters_bias_
        l_self_modules_features_modules_6_parameters_weight_ = L_self_modules_features_modules_6_parameters_weight_
        l_self_modules_features_modules_6_parameters_bias_ = L_self_modules_features_modules_6_parameters_bias_
        l_self_modules_features_modules_8_parameters_weight_ = L_self_modules_features_modules_8_parameters_weight_
        l_self_modules_features_modules_8_parameters_bias_ = L_self_modules_features_modules_8_parameters_bias_
        l_self_modules_features_modules_10_parameters_weight_ = L_self_modules_features_modules_10_parameters_weight_
        l_self_modules_features_modules_10_parameters_bias_ = L_self_modules_features_modules_10_parameters_bias_
        l_self_modules_classifier_modules_1_parameters_weight_ = L_self_modules_classifier_modules_1_parameters_weight_
        l_self_modules_classifier_modules_1_parameters_bias_ = L_self_modules_classifier_modules_1_parameters_bias_
        l_self_modules_classifier_modules_4_parameters_weight_ = L_self_modules_classifier_modules_4_parameters_weight_
        l_self_modules_classifier_modules_4_parameters_bias_ = L_self_modules_classifier_modules_4_parameters_bias_
        l_self_modules_classifier_modules_6_parameters_weight_ = L_self_modules_classifier_modules_6_parameters_weight_
        l_self_modules_classifier_modules_6_parameters_bias_ = L_self_modules_classifier_modules_6_parameters_bias_
        input_1 = torch.conv2d(l_x_, l_self_modules_features_modules_0_parameters_weight_, l_self_modules_features_modules_0_parameters_bias_, (4, 4), (2, 2), (1, 1), 1);  l_x_ = l_self_modules_features_modules_0_parameters_weight_ = l_self_modules_features_modules_0_parameters_bias_ = None
        input_2 = torch.nn.functional.relu(input_1, inplace = True);  input_1 = None
        input_3 = torch.nn.functional.max_pool2d(input_2, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  input_2 = None
        input_4 = torch.conv2d(input_3, l_self_modules_features_modules_3_parameters_weight_, l_self_modules_features_modules_3_parameters_bias_, (1, 1), (2, 2), (1, 1), 1);  input_3 = l_self_modules_features_modules_3_parameters_weight_ = l_self_modules_features_modules_3_parameters_bias_ = None
        input_5 = torch.nn.functional.relu(input_4, inplace = True);  input_4 = None
        input_6 = torch.nn.functional.max_pool2d(input_5, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  input_5 = None
        input_7 = torch.conv2d(input_6, l_self_modules_features_modules_6_parameters_weight_, l_self_modules_features_modules_6_parameters_bias_, (1, 1), (1, 1), (1, 1), 1);  input_6 = l_self_modules_features_modules_6_parameters_weight_ = l_self_modules_features_modules_6_parameters_bias_ = None
        input_8 = torch.nn.functional.relu(input_7, inplace = True);  input_7 = None
        input_9 = torch.conv2d(input_8, l_self_modules_features_modules_8_parameters_weight_, l_self_modules_features_modules_8_parameters_bias_, (1, 1), (1, 1), (1, 1), 1);  input_8 = l_self_modules_features_modules_8_parameters_weight_ = l_self_modules_features_modules_8_parameters_bias_ = None
        input_10 = torch.nn.functional.relu(input_9, inplace = True);  input_9 = None
        input_11 = torch.conv2d(input_10, l_self_modules_features_modules_10_parameters_weight_, l_self_modules_features_modules_10_parameters_bias_, (1, 1), (1, 1), (1, 1), 1);  input_10 = l_self_modules_features_modules_10_parameters_weight_ = l_self_modules_features_modules_10_parameters_bias_ = None
        input_12 = torch.nn.functional.relu(input_11, inplace = True);  input_11 = None
        input_13 = torch.nn.functional.max_pool2d(input_12, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  input_12 = None
        x = torch.nn.functional.adaptive_avg_pool2d(input_13, (6, 6));  input_13 = None
        x_1 = torch.flatten(x, 1);  x = None
        input_14 = torch.nn.functional.dropout(x_1, 0.5, False, False);  x_1 = None
        input_15 = torch._C._nn.linear(input_14, l_self_modules_classifier_modules_1_parameters_weight_, l_self_modules_classifier_modules_1_parameters_bias_);  input_14 = l_self_modules_classifier_modules_1_parameters_weight_ = l_self_modules_classifier_modules_1_parameters_bias_ = None
        input_16 = torch.nn.functional.relu(input_15, inplace = True);  input_15 = None
        input_17 = torch.nn.functional.dropout(input_16, 0.5, False, False);  input_16 = None
        input_18 = torch._C._nn.linear(input_17, l_self_modules_classifier_modules_4_parameters_weight_, l_self_modules_classifier_modules_4_parameters_bias_);  input_17 = l_self_modules_classifier_modules_4_parameters_weight_ = l_self_modules_classifier_modules_4_parameters_bias_ = None
        input_19 = torch.nn.functional.relu(input_18, inplace = True);  input_18 = None
        input_20 = torch._C._nn.linear(input_19, l_self_modules_classifier_modules_6_parameters_weight_, l_self_modules_classifier_modules_6_parameters_bias_);  input_19 = l_self_modules_classifier_modules_6_parameters_weight_ = l_self_modules_classifier_modules_6_parameters_bias_ = None
        return (input_20,)
        
    # To see more debug info, please use `graph_module.print_readable()`
