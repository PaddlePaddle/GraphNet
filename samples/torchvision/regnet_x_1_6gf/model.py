import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_stem_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_stem_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stem_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stem_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block1_modules_block1_0_modules_proj_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block1_modules_block1_0_modules_proj_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block1_modules_block1_0_modules_proj_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block1_modules_block1_0_modules_proj_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block1_modules_block1_0_modules_proj_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_a_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_a_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_a_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_a_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_a_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_b_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_b_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_b_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_b_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_b_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_c_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_c_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_c_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_c_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_c_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_a_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_a_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_a_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_a_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_a_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_b_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_b_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_b_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_b_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_b_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_c_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_c_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_c_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_c_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_c_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_a_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_a_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_a_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_a_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_a_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_b_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_b_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_b_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_b_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_b_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_a_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_a_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_a_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_a_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_a_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_b_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_b_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_b_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_b_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_b_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_a_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_a_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_a_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_a_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_a_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_b_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_b_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_b_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_b_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_b_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_c_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_c_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_c_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_c_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_c_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_a_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_a_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_a_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_a_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_a_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_b_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_b_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_b_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_b_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_b_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_c_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_c_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_c_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_c_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_c_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_a_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_a_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_a_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_a_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_a_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_b_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_b_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_b_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_b_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_b_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_c_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_c_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_c_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_c_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_c_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_a_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_a_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_a_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_a_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_a_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_b_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_b_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_b_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_b_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_b_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_a_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_a_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_a_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_a_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_a_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_b_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_b_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_b_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_b_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_b_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_c_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_c_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_c_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_c_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_c_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_fc_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_stem_modules_0_parameters_weight_ = (
            L_self_modules_stem_modules_0_parameters_weight_
        )
        l_x_ = L_x_
        l_self_modules_stem_modules_1_buffers_running_mean_ = (
            L_self_modules_stem_modules_1_buffers_running_mean_
        )
        l_self_modules_stem_modules_1_buffers_running_var_ = (
            L_self_modules_stem_modules_1_buffers_running_var_
        )
        l_self_modules_stem_modules_1_parameters_weight_ = (
            L_self_modules_stem_modules_1_parameters_weight_
        )
        l_self_modules_stem_modules_1_parameters_bias_ = (
            L_self_modules_stem_modules_1_parameters_bias_
        )
        l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_proj_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block1_modules_block1_0_modules_proj_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_proj_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block1_modules_block1_0_modules_proj_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_proj_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block1_modules_block1_0_modules_proj_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_proj_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block1_modules_block1_0_modules_proj_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_proj_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block1_modules_block1_0_modules_proj_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_a_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_a_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_a_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_a_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_a_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_a_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_a_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_a_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_a_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_a_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_b_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_b_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_b_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_b_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_b_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_b_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_b_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_b_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_b_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_b_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_c_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_c_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_c_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_c_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_c_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_c_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_c_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_c_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_c_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_c_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_a_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_a_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_a_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_a_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_a_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_a_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_a_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_a_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_a_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_a_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_b_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_b_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_b_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_b_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_b_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_b_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_b_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_b_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_b_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_b_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_c_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_c_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_c_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_c_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_c_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_c_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_c_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_c_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_c_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_c_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_a_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_a_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_a_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_a_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_a_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_a_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_a_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_a_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_a_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_a_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_b_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_b_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_b_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_b_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_b_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_b_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_b_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_b_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_b_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_b_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_a_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_a_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_a_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_a_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_a_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_a_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_a_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_a_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_a_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_a_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_b_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_b_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_b_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_b_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_b_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_b_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_b_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_b_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_b_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_b_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_a_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_a_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_a_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_a_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_a_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_a_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_a_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_a_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_a_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_a_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_b_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_b_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_b_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_b_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_b_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_b_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_b_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_b_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_b_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_b_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_c_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_c_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_c_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_c_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_c_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_c_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_c_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_c_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_c_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_c_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_a_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_a_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_a_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_a_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_a_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_a_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_a_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_a_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_a_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_a_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_b_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_b_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_b_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_b_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_b_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_b_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_b_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_b_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_b_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_b_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_c_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_c_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_c_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_c_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_c_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_c_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_c_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_c_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_c_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_c_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_a_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_a_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_a_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_a_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_a_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_a_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_a_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_a_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_a_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_a_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_b_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_b_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_b_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_b_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_b_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_b_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_b_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_b_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_b_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_b_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_c_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_c_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_c_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_c_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_c_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_c_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_c_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_c_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_c_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_c_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_a_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_a_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_a_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_a_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_a_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_a_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_a_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_a_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_a_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_a_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_b_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_b_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_b_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_b_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_b_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_b_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_b_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_b_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_b_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_b_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_a_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_a_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_a_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_a_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_a_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_a_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_a_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_a_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_a_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_a_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_b_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_b_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_b_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_b_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_b_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_b_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_b_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_b_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_b_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_b_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_c_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_c_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_c_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_c_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_c_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_c_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_c_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_c_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_c_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_c_modules_1_parameters_bias_
        l_self_modules_fc_parameters_weight_ = L_self_modules_fc_parameters_weight_
        l_self_modules_fc_parameters_bias_ = L_self_modules_fc_parameters_bias_
        input_1 = torch.conv2d(
            l_x_,
            l_self_modules_stem_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_x_ = l_self_modules_stem_modules_0_parameters_weight_ = None
        input_2 = torch.nn.functional.batch_norm(
            input_1,
            l_self_modules_stem_modules_1_buffers_running_mean_,
            l_self_modules_stem_modules_1_buffers_running_var_,
            l_self_modules_stem_modules_1_parameters_weight_,
            l_self_modules_stem_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_1 = (
            l_self_modules_stem_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_stem_modules_1_buffers_running_var_
        ) = (
            l_self_modules_stem_modules_1_parameters_weight_
        ) = l_self_modules_stem_modules_1_parameters_bias_ = None
        input_3 = torch.nn.functional.relu(input_2, inplace=True)
        input_2 = None
        input_4 = torch.conv2d(
            input_3,
            l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_proj_modules_0_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_proj_modules_0_parameters_weight_ = (
            None
        )
        input_5 = torch.nn.functional.batch_norm(
            input_4,
            l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_proj_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_proj_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_proj_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_proj_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_4 = l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_proj_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_proj_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_proj_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_proj_modules_1_parameters_bias_ = (None)
        input_6 = torch.conv2d(
            input_3,
            l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_a_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_3 = l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_a_modules_0_parameters_weight_ = (None)
        input_7 = torch.nn.functional.batch_norm(
            input_6,
            l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_6 = l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_8 = torch.nn.functional.relu(input_7, inplace=True)
        input_7 = None
        input_9 = torch.conv2d(
            input_8,
            l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            3,
        )
        input_8 = l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_10 = torch.nn.functional.batch_norm(
            input_9,
            l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_9 = l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_11 = torch.nn.functional.relu(input_10, inplace=True)
        input_10 = None
        input_12 = torch.conv2d(
            input_11,
            l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_11 = l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_13 = torch.nn.functional.batch_norm(
            input_12,
            l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_12 = l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x = input_5 + input_13
        input_5 = input_13 = None
        input_14 = torch.nn.functional.relu(x, inplace=True)
        x = None
        input_15 = torch.conv2d(
            input_14,
            l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_a_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_a_modules_0_parameters_weight_ = (
            None
        )
        input_16 = torch.nn.functional.batch_norm(
            input_15,
            l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_15 = l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_17 = torch.nn.functional.relu(input_16, inplace=True)
        input_16 = None
        input_18 = torch.conv2d(
            input_17,
            l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            3,
        )
        input_17 = l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_19 = torch.nn.functional.batch_norm(
            input_18,
            l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_18 = l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_20 = torch.nn.functional.relu(input_19, inplace=True)
        input_19 = None
        input_21 = torch.conv2d(
            input_20,
            l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_20 = l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_22 = torch.nn.functional.batch_norm(
            input_21,
            l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_21 = l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_1 = input_14 + input_22
        input_14 = input_22 = None
        input_23 = torch.nn.functional.relu(x_1, inplace=True)
        x_1 = None
        input_24 = torch.conv2d(
            input_23,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_0_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_0_parameters_weight_ = (
            None
        )
        input_25 = torch.nn.functional.batch_norm(
            input_24,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_24 = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_1_parameters_bias_ = (None)
        input_26 = torch.conv2d(
            input_23,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_23 = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_0_parameters_weight_ = (None)
        input_27 = torch.nn.functional.batch_norm(
            input_26,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_26 = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_28 = torch.nn.functional.relu(input_27, inplace=True)
        input_27 = None
        input_29 = torch.conv2d(
            input_28,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            7,
        )
        input_28 = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_30 = torch.nn.functional.batch_norm(
            input_29,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_29 = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_31 = torch.nn.functional.relu(input_30, inplace=True)
        input_30 = None
        input_32 = torch.conv2d(
            input_31,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_31 = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_33 = torch.nn.functional.batch_norm(
            input_32,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_32 = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_2 = input_25 + input_33
        input_25 = input_33 = None
        input_34 = torch.nn.functional.relu(x_2, inplace=True)
        x_2 = None
        input_35 = torch.conv2d(
            input_34,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_0_parameters_weight_ = (
            None
        )
        input_36 = torch.nn.functional.batch_norm(
            input_35,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_35 = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_37 = torch.nn.functional.relu(input_36, inplace=True)
        input_36 = None
        input_38 = torch.conv2d(
            input_37,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            7,
        )
        input_37 = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_39 = torch.nn.functional.batch_norm(
            input_38,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_38 = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_40 = torch.nn.functional.relu(input_39, inplace=True)
        input_39 = None
        input_41 = torch.conv2d(
            input_40,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_40 = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_42 = torch.nn.functional.batch_norm(
            input_41,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_41 = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_3 = input_34 + input_42
        input_34 = input_42 = None
        input_43 = torch.nn.functional.relu(x_3, inplace=True)
        x_3 = None
        input_44 = torch.conv2d(
            input_43,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_a_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_a_modules_0_parameters_weight_ = (
            None
        )
        input_45 = torch.nn.functional.batch_norm(
            input_44,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_44 = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_46 = torch.nn.functional.relu(input_45, inplace=True)
        input_45 = None
        input_47 = torch.conv2d(
            input_46,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            7,
        )
        input_46 = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_48 = torch.nn.functional.batch_norm(
            input_47,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_47 = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_49 = torch.nn.functional.relu(input_48, inplace=True)
        input_48 = None
        input_50 = torch.conv2d(
            input_49,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_49 = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_51 = torch.nn.functional.batch_norm(
            input_50,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_50 = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_4 = input_43 + input_51
        input_43 = input_51 = None
        input_52 = torch.nn.functional.relu(x_4, inplace=True)
        x_4 = None
        input_53 = torch.conv2d(
            input_52,
            l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_a_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_a_modules_0_parameters_weight_ = (
            None
        )
        input_54 = torch.nn.functional.batch_norm(
            input_53,
            l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_53 = l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_55 = torch.nn.functional.relu(input_54, inplace=True)
        input_54 = None
        input_56 = torch.conv2d(
            input_55,
            l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            7,
        )
        input_55 = l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_57 = torch.nn.functional.batch_norm(
            input_56,
            l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_56 = l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_58 = torch.nn.functional.relu(input_57, inplace=True)
        input_57 = None
        input_59 = torch.conv2d(
            input_58,
            l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_58 = l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_60 = torch.nn.functional.batch_norm(
            input_59,
            l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_59 = l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_5 = input_52 + input_60
        input_52 = input_60 = None
        input_61 = torch.nn.functional.relu(x_5, inplace=True)
        x_5 = None
        input_62 = torch.conv2d(
            input_61,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_0_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_0_parameters_weight_ = (
            None
        )
        input_63 = torch.nn.functional.batch_norm(
            input_62,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_62 = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_1_parameters_bias_ = (None)
        input_64 = torch.conv2d(
            input_61,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_61 = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_0_parameters_weight_ = (None)
        input_65 = torch.nn.functional.batch_norm(
            input_64,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_64 = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_66 = torch.nn.functional.relu(input_65, inplace=True)
        input_65 = None
        input_67 = torch.conv2d(
            input_66,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            17,
        )
        input_66 = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_68 = torch.nn.functional.batch_norm(
            input_67,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_67 = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_69 = torch.nn.functional.relu(input_68, inplace=True)
        input_68 = None
        input_70 = torch.conv2d(
            input_69,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_69 = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_71 = torch.nn.functional.batch_norm(
            input_70,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_70 = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_6 = input_63 + input_71
        input_63 = input_71 = None
        input_72 = torch.nn.functional.relu(x_6, inplace=True)
        x_6 = None
        input_73 = torch.conv2d(
            input_72,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_0_parameters_weight_ = (
            None
        )
        input_74 = torch.nn.functional.batch_norm(
            input_73,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_73 = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_75 = torch.nn.functional.relu(input_74, inplace=True)
        input_74 = None
        input_76 = torch.conv2d(
            input_75,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            17,
        )
        input_75 = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_77 = torch.nn.functional.batch_norm(
            input_76,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_76 = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_78 = torch.nn.functional.relu(input_77, inplace=True)
        input_77 = None
        input_79 = torch.conv2d(
            input_78,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_78 = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_80 = torch.nn.functional.batch_norm(
            input_79,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_79 = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_7 = input_72 + input_80
        input_72 = input_80 = None
        input_81 = torch.nn.functional.relu(x_7, inplace=True)
        x_7 = None
        input_82 = torch.conv2d(
            input_81,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_0_parameters_weight_ = (
            None
        )
        input_83 = torch.nn.functional.batch_norm(
            input_82,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_82 = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_84 = torch.nn.functional.relu(input_83, inplace=True)
        input_83 = None
        input_85 = torch.conv2d(
            input_84,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            17,
        )
        input_84 = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_86 = torch.nn.functional.batch_norm(
            input_85,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_85 = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_87 = torch.nn.functional.relu(input_86, inplace=True)
        input_86 = None
        input_88 = torch.conv2d(
            input_87,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_87 = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_89 = torch.nn.functional.batch_norm(
            input_88,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_88 = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_8 = input_81 + input_89
        input_81 = input_89 = None
        input_90 = torch.nn.functional.relu(x_8, inplace=True)
        x_8 = None
        input_91 = torch.conv2d(
            input_90,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_0_parameters_weight_ = (
            None
        )
        input_92 = torch.nn.functional.batch_norm(
            input_91,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_91 = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_93 = torch.nn.functional.relu(input_92, inplace=True)
        input_92 = None
        input_94 = torch.conv2d(
            input_93,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            17,
        )
        input_93 = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_95 = torch.nn.functional.batch_norm(
            input_94,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_94 = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_96 = torch.nn.functional.relu(input_95, inplace=True)
        input_95 = None
        input_97 = torch.conv2d(
            input_96,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_96 = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_98 = torch.nn.functional.batch_norm(
            input_97,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_97 = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_9 = input_90 + input_98
        input_90 = input_98 = None
        input_99 = torch.nn.functional.relu(x_9, inplace=True)
        x_9 = None
        input_100 = torch.conv2d(
            input_99,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_0_parameters_weight_ = (
            None
        )
        input_101 = torch.nn.functional.batch_norm(
            input_100,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_100 = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_102 = torch.nn.functional.relu(input_101, inplace=True)
        input_101 = None
        input_103 = torch.conv2d(
            input_102,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            17,
        )
        input_102 = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_104 = torch.nn.functional.batch_norm(
            input_103,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_103 = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_105 = torch.nn.functional.relu(input_104, inplace=True)
        input_104 = None
        input_106 = torch.conv2d(
            input_105,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_105 = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_107 = torch.nn.functional.batch_norm(
            input_106,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_106 = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_10 = input_99 + input_107
        input_99 = input_107 = None
        input_108 = torch.nn.functional.relu(x_10, inplace=True)
        x_10 = None
        input_109 = torch.conv2d(
            input_108,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_0_parameters_weight_ = (
            None
        )
        input_110 = torch.nn.functional.batch_norm(
            input_109,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_109 = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_111 = torch.nn.functional.relu(input_110, inplace=True)
        input_110 = None
        input_112 = torch.conv2d(
            input_111,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            17,
        )
        input_111 = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_113 = torch.nn.functional.batch_norm(
            input_112,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_112 = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_114 = torch.nn.functional.relu(input_113, inplace=True)
        input_113 = None
        input_115 = torch.conv2d(
            input_114,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_114 = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_116 = torch.nn.functional.batch_norm(
            input_115,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_115 = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_11 = input_108 + input_116
        input_108 = input_116 = None
        input_117 = torch.nn.functional.relu(x_11, inplace=True)
        x_11 = None
        input_118 = torch.conv2d(
            input_117,
            l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_a_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_a_modules_0_parameters_weight_ = (
            None
        )
        input_119 = torch.nn.functional.batch_norm(
            input_118,
            l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_118 = l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_120 = torch.nn.functional.relu(input_119, inplace=True)
        input_119 = None
        input_121 = torch.conv2d(
            input_120,
            l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            17,
        )
        input_120 = l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_122 = torch.nn.functional.batch_norm(
            input_121,
            l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_121 = l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_123 = torch.nn.functional.relu(input_122, inplace=True)
        input_122 = None
        input_124 = torch.conv2d(
            input_123,
            l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_123 = l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_125 = torch.nn.functional.batch_norm(
            input_124,
            l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_124 = l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_12 = input_117 + input_125
        input_117 = input_125 = None
        input_126 = torch.nn.functional.relu(x_12, inplace=True)
        x_12 = None
        input_127 = torch.conv2d(
            input_126,
            l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_a_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_a_modules_0_parameters_weight_ = (
            None
        )
        input_128 = torch.nn.functional.batch_norm(
            input_127,
            l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_127 = l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_129 = torch.nn.functional.relu(input_128, inplace=True)
        input_128 = None
        input_130 = torch.conv2d(
            input_129,
            l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            17,
        )
        input_129 = l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_131 = torch.nn.functional.batch_norm(
            input_130,
            l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_130 = l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_132 = torch.nn.functional.relu(input_131, inplace=True)
        input_131 = None
        input_133 = torch.conv2d(
            input_132,
            l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_132 = l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_134 = torch.nn.functional.batch_norm(
            input_133,
            l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_133 = l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_13 = input_126 + input_134
        input_126 = input_134 = None
        input_135 = torch.nn.functional.relu(x_13, inplace=True)
        x_13 = None
        input_136 = torch.conv2d(
            input_135,
            l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_a_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_a_modules_0_parameters_weight_ = (
            None
        )
        input_137 = torch.nn.functional.batch_norm(
            input_136,
            l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_136 = l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_138 = torch.nn.functional.relu(input_137, inplace=True)
        input_137 = None
        input_139 = torch.conv2d(
            input_138,
            l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            17,
        )
        input_138 = l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_140 = torch.nn.functional.batch_norm(
            input_139,
            l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_139 = l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_141 = torch.nn.functional.relu(input_140, inplace=True)
        input_140 = None
        input_142 = torch.conv2d(
            input_141,
            l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_141 = l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_143 = torch.nn.functional.batch_norm(
            input_142,
            l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_142 = l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_14 = input_135 + input_143
        input_135 = input_143 = None
        input_144 = torch.nn.functional.relu(x_14, inplace=True)
        x_14 = None
        input_145 = torch.conv2d(
            input_144,
            l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_a_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_a_modules_0_parameters_weight_ = (
            None
        )
        input_146 = torch.nn.functional.batch_norm(
            input_145,
            l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_145 = l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_147 = torch.nn.functional.relu(input_146, inplace=True)
        input_146 = None
        input_148 = torch.conv2d(
            input_147,
            l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            17,
        )
        input_147 = l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_149 = torch.nn.functional.batch_norm(
            input_148,
            l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_148 = l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_150 = torch.nn.functional.relu(input_149, inplace=True)
        input_149 = None
        input_151 = torch.conv2d(
            input_150,
            l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_150 = l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_152 = torch.nn.functional.batch_norm(
            input_151,
            l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_151 = l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_15 = input_144 + input_152
        input_144 = input_152 = None
        input_153 = torch.nn.functional.relu(x_15, inplace=True)
        x_15 = None
        input_154 = torch.conv2d(
            input_153,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_0_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_0_parameters_weight_ = (
            None
        )
        input_155 = torch.nn.functional.batch_norm(
            input_154,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_154 = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_1_parameters_bias_ = (None)
        input_156 = torch.conv2d(
            input_153,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_153 = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_0_parameters_weight_ = (None)
        input_157 = torch.nn.functional.batch_norm(
            input_156,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_156 = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_158 = torch.nn.functional.relu(input_157, inplace=True)
        input_157 = None
        input_159 = torch.conv2d(
            input_158,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            38,
        )
        input_158 = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_160 = torch.nn.functional.batch_norm(
            input_159,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_159 = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_161 = torch.nn.functional.relu(input_160, inplace=True)
        input_160 = None
        input_162 = torch.conv2d(
            input_161,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_161 = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_163 = torch.nn.functional.batch_norm(
            input_162,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_162 = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_16 = input_155 + input_163
        input_155 = input_163 = None
        input_164 = torch.nn.functional.relu(x_16, inplace=True)
        x_16 = None
        input_165 = torch.conv2d(
            input_164,
            l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_a_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_a_modules_0_parameters_weight_ = (
            None
        )
        input_166 = torch.nn.functional.batch_norm(
            input_165,
            l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_165 = l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_167 = torch.nn.functional.relu(input_166, inplace=True)
        input_166 = None
        input_168 = torch.conv2d(
            input_167,
            l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            38,
        )
        input_167 = l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_169 = torch.nn.functional.batch_norm(
            input_168,
            l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_168 = l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_170 = torch.nn.functional.relu(input_169, inplace=True)
        input_169 = None
        input_171 = torch.conv2d(
            input_170,
            l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_170 = l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_172 = torch.nn.functional.batch_norm(
            input_171,
            l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_171 = l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_17 = input_164 + input_172
        input_164 = input_172 = None
        input_173 = torch.nn.functional.relu(x_17, inplace=True)
        x_17 = None
        x_18 = torch.nn.functional.adaptive_avg_pool2d(input_173, (1, 1))
        input_173 = None
        x_19 = x_18.flatten(start_dim=1)
        x_18 = None
        x_20 = torch._C._nn.linear(
            x_19,
            l_self_modules_fc_parameters_weight_,
            l_self_modules_fc_parameters_bias_,
        )
        x_19 = (
            l_self_modules_fc_parameters_weight_
        ) = l_self_modules_fc_parameters_bias_ = None
        return (x_20,)
