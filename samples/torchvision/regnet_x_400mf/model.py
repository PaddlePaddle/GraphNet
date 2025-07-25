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
        L_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_a_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_a_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_a_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_a_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_a_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_b_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_b_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_b_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_b_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_b_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_c_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_c_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_c_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_c_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_c_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_a_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_a_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_a_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_a_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_a_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_b_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_b_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_b_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_b_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_b_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_c_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_c_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_c_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_c_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_c_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_a_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_a_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_a_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_a_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_a_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_b_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_b_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_b_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_b_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_b_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_c_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_c_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_c_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_c_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_c_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_a_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_a_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_a_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_a_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_a_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_b_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_b_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_b_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_b_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_b_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_a_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_a_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_a_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_a_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_a_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_b_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_b_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_b_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_b_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_b_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_c_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_c_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_c_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_c_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_c_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_a_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_a_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_a_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_a_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_a_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_b_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_b_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_b_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_b_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_b_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_c_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_c_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_c_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_c_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_c_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_a_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_a_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_a_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_a_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_a_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_b_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_b_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_b_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_b_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_b_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_c_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_c_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_c_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_c_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_c_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_a_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_a_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_a_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_a_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_a_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_b_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_b_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_b_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_b_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_b_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_c_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_c_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_c_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_c_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_c_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_a_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_a_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_a_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_a_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_a_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_b_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_b_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_b_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_b_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_b_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_c_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_c_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_c_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_c_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_c_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_a_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_a_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_a_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_a_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_a_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_b_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_b_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_b_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_b_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_b_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_c_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_c_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_c_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_c_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_c_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_a_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_a_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_a_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_a_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_a_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_a_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_a_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_a_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_a_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_a_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_b_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_b_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_b_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_b_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_b_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_b_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_b_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_b_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_b_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_b_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_c_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_c_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_c_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_c_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_c_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_c_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_c_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_c_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_c_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_c_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_a_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_a_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_a_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_a_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_a_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_a_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_a_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_a_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_a_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_a_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_b_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_b_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_b_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_b_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_b_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_b_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_b_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_b_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_b_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_b_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_c_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_c_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_c_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_c_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_c_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_c_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_c_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_c_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_c_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_c_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_a_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_a_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_a_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_a_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_a_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_a_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_a_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_a_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_a_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_a_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_b_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_b_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_b_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_b_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_b_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_b_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_b_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_b_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_b_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_b_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_c_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_c_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_c_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_c_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_c_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_c_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_c_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_c_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_c_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_c_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_a_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_a_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_a_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_a_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_a_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_a_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_a_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_a_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_a_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_a_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_b_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_b_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_b_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_b_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_b_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_b_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_b_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_b_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_b_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_b_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_a_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_a_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_a_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_a_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_a_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_a_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_a_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_a_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_a_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_a_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_b_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_b_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_b_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_b_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_b_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_b_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_b_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_b_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_b_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_b_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_c_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_c_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_c_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_c_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_c_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_c_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_c_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_c_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_c_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_c_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_a_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_a_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_a_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_a_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_a_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_a_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_a_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_a_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_a_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_a_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_b_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_b_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_b_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_b_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_b_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_b_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_b_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_b_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_b_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_b_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_c_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_c_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_c_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_c_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_c_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_c_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_c_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_c_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_c_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_c_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_a_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_a_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_a_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_a_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_a_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_a_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_a_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_a_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_a_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_a_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_b_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_b_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_b_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_b_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_b_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_b_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_b_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_b_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_b_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_b_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_c_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_c_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_c_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_c_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_c_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_c_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_c_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_c_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_c_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_c_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_a_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_a_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_a_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_a_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_a_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_a_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_a_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_a_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_a_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_a_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_b_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_b_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_b_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_b_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_b_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_b_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_b_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_b_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_b_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_b_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_c_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_c_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_c_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_c_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_c_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_c_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_c_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_c_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_c_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_c_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_a_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_a_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_a_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_a_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_a_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_a_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_a_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_a_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_a_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_a_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_b_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_b_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_b_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_b_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_b_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_b_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_b_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_b_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_b_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_b_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_c_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_c_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_c_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_c_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_c_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_c_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_c_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_c_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_c_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_c_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_a_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_a_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_a_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_a_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_a_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_a_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_a_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_a_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_a_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_a_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_b_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_b_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_b_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_b_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_b_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_b_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_b_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_b_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_b_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_b_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_c_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_c_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_c_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_c_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_c_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_c_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_c_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_c_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_c_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_c_modules_1_parameters_bias_
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
            2,
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
        input_16 = torch.nn.functional.batch_norm(
            input_15,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_15 = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_1_parameters_bias_ = (None)
        input_17 = torch.conv2d(
            input_14,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_14 = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_0_parameters_weight_ = (None)
        input_18 = torch.nn.functional.batch_norm(
            input_17,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_17 = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_19 = torch.nn.functional.relu(input_18, inplace=True)
        input_18 = None
        input_20 = torch.conv2d(
            input_19,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            4,
        )
        input_19 = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_21 = torch.nn.functional.batch_norm(
            input_20,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_20 = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_22 = torch.nn.functional.relu(input_21, inplace=True)
        input_21 = None
        input_23 = torch.conv2d(
            input_22,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_22 = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_24 = torch.nn.functional.batch_norm(
            input_23,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_23 = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_1 = input_16 + input_24
        input_16 = input_24 = None
        input_25 = torch.nn.functional.relu(x_1, inplace=True)
        x_1 = None
        input_26 = torch.conv2d(
            input_25,
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
        input_27 = torch.nn.functional.batch_norm(
            input_26,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_26 = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_28 = torch.nn.functional.relu(input_27, inplace=True)
        input_27 = None
        input_29 = torch.conv2d(
            input_28,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            4,
        )
        input_28 = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_30 = torch.nn.functional.batch_norm(
            input_29,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_29 = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_31 = torch.nn.functional.relu(input_30, inplace=True)
        input_30 = None
        input_32 = torch.conv2d(
            input_31,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_31 = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_33 = torch.nn.functional.batch_norm(
            input_32,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_32 = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_2 = input_25 + input_33
        input_25 = input_33 = None
        input_34 = torch.nn.functional.relu(x_2, inplace=True)
        x_2 = None
        input_35 = torch.conv2d(
            input_34,
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
        input_36 = torch.nn.functional.batch_norm(
            input_35,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_35 = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_1_parameters_bias_ = (None)
        input_37 = torch.conv2d(
            input_34,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_34 = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_0_parameters_weight_ = (None)
        input_38 = torch.nn.functional.batch_norm(
            input_37,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_37 = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_39 = torch.nn.functional.relu(input_38, inplace=True)
        input_38 = None
        input_40 = torch.conv2d(
            input_39,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            10,
        )
        input_39 = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_41 = torch.nn.functional.batch_norm(
            input_40,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_40 = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_42 = torch.nn.functional.relu(input_41, inplace=True)
        input_41 = None
        input_43 = torch.conv2d(
            input_42,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_42 = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_44 = torch.nn.functional.batch_norm(
            input_43,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_43 = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_3 = input_36 + input_44
        input_36 = input_44 = None
        input_45 = torch.nn.functional.relu(x_3, inplace=True)
        x_3 = None
        input_46 = torch.conv2d(
            input_45,
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
        input_47 = torch.nn.functional.batch_norm(
            input_46,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_46 = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_48 = torch.nn.functional.relu(input_47, inplace=True)
        input_47 = None
        input_49 = torch.conv2d(
            input_48,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            10,
        )
        input_48 = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_50 = torch.nn.functional.batch_norm(
            input_49,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_49 = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_51 = torch.nn.functional.relu(input_50, inplace=True)
        input_50 = None
        input_52 = torch.conv2d(
            input_51,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_51 = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_53 = torch.nn.functional.batch_norm(
            input_52,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_52 = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_4 = input_45 + input_53
        input_45 = input_53 = None
        input_54 = torch.nn.functional.relu(x_4, inplace=True)
        x_4 = None
        input_55 = torch.conv2d(
            input_54,
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
        input_56 = torch.nn.functional.batch_norm(
            input_55,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_55 = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_57 = torch.nn.functional.relu(input_56, inplace=True)
        input_56 = None
        input_58 = torch.conv2d(
            input_57,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            10,
        )
        input_57 = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_59 = torch.nn.functional.batch_norm(
            input_58,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_58 = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_60 = torch.nn.functional.relu(input_59, inplace=True)
        input_59 = None
        input_61 = torch.conv2d(
            input_60,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_60 = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_62 = torch.nn.functional.batch_norm(
            input_61,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_61 = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_5 = input_54 + input_62
        input_54 = input_62 = None
        input_63 = torch.nn.functional.relu(x_5, inplace=True)
        x_5 = None
        input_64 = torch.conv2d(
            input_63,
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
        input_65 = torch.nn.functional.batch_norm(
            input_64,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_64 = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_66 = torch.nn.functional.relu(input_65, inplace=True)
        input_65 = None
        input_67 = torch.conv2d(
            input_66,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            10,
        )
        input_66 = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_68 = torch.nn.functional.batch_norm(
            input_67,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_67 = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_69 = torch.nn.functional.relu(input_68, inplace=True)
        input_68 = None
        input_70 = torch.conv2d(
            input_69,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_69 = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_71 = torch.nn.functional.batch_norm(
            input_70,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_70 = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_6 = input_63 + input_71
        input_63 = input_71 = None
        input_72 = torch.nn.functional.relu(x_6, inplace=True)
        x_6 = None
        input_73 = torch.conv2d(
            input_72,
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
        input_74 = torch.nn.functional.batch_norm(
            input_73,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_73 = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_75 = torch.nn.functional.relu(input_74, inplace=True)
        input_74 = None
        input_76 = torch.conv2d(
            input_75,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            10,
        )
        input_75 = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_77 = torch.nn.functional.batch_norm(
            input_76,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_76 = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_78 = torch.nn.functional.relu(input_77, inplace=True)
        input_77 = None
        input_79 = torch.conv2d(
            input_78,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_78 = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_80 = torch.nn.functional.batch_norm(
            input_79,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_79 = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_7 = input_72 + input_80
        input_72 = input_80 = None
        input_81 = torch.nn.functional.relu(x_7, inplace=True)
        x_7 = None
        input_82 = torch.conv2d(
            input_81,
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
        input_83 = torch.nn.functional.batch_norm(
            input_82,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_82 = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_84 = torch.nn.functional.relu(input_83, inplace=True)
        input_83 = None
        input_85 = torch.conv2d(
            input_84,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            10,
        )
        input_84 = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_86 = torch.nn.functional.batch_norm(
            input_85,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_85 = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_87 = torch.nn.functional.relu(input_86, inplace=True)
        input_86 = None
        input_88 = torch.conv2d(
            input_87,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_87 = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_89 = torch.nn.functional.batch_norm(
            input_88,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_88 = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_8 = input_81 + input_89
        input_81 = input_89 = None
        input_90 = torch.nn.functional.relu(x_8, inplace=True)
        x_8 = None
        input_91 = torch.conv2d(
            input_90,
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
        input_92 = torch.nn.functional.batch_norm(
            input_91,
            l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_91 = l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_93 = torch.nn.functional.relu(input_92, inplace=True)
        input_92 = None
        input_94 = torch.conv2d(
            input_93,
            l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            10,
        )
        input_93 = l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_95 = torch.nn.functional.batch_norm(
            input_94,
            l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_94 = l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_96 = torch.nn.functional.relu(input_95, inplace=True)
        input_95 = None
        input_97 = torch.conv2d(
            input_96,
            l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_96 = l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_98 = torch.nn.functional.batch_norm(
            input_97,
            l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_97 = l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_9 = input_90 + input_98
        input_90 = input_98 = None
        input_99 = torch.nn.functional.relu(x_9, inplace=True)
        x_9 = None
        input_100 = torch.conv2d(
            input_99,
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
        input_101 = torch.nn.functional.batch_norm(
            input_100,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_100 = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_1_parameters_bias_ = (None)
        input_102 = torch.conv2d(
            input_99,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_99 = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_0_parameters_weight_ = (None)
        input_103 = torch.nn.functional.batch_norm(
            input_102,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_102 = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_104 = torch.nn.functional.relu(input_103, inplace=True)
        input_103 = None
        input_105 = torch.conv2d(
            input_104,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            25,
        )
        input_104 = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_106 = torch.nn.functional.batch_norm(
            input_105,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_105 = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_107 = torch.nn.functional.relu(input_106, inplace=True)
        input_106 = None
        input_108 = torch.conv2d(
            input_107,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_107 = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_109 = torch.nn.functional.batch_norm(
            input_108,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_108 = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_10 = input_101 + input_109
        input_101 = input_109 = None
        input_110 = torch.nn.functional.relu(x_10, inplace=True)
        x_10 = None
        input_111 = torch.conv2d(
            input_110,
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
        input_112 = torch.nn.functional.batch_norm(
            input_111,
            l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_111 = l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_113 = torch.nn.functional.relu(input_112, inplace=True)
        input_112 = None
        input_114 = torch.conv2d(
            input_113,
            l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            25,
        )
        input_113 = l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_115 = torch.nn.functional.batch_norm(
            input_114,
            l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_114 = l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_116 = torch.nn.functional.relu(input_115, inplace=True)
        input_115 = None
        input_117 = torch.conv2d(
            input_116,
            l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_116 = l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_118 = torch.nn.functional.batch_norm(
            input_117,
            l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_117 = l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_11 = input_110 + input_118
        input_110 = input_118 = None
        input_119 = torch.nn.functional.relu(x_11, inplace=True)
        x_11 = None
        input_120 = torch.conv2d(
            input_119,
            l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_a_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_a_modules_0_parameters_weight_ = (
            None
        )
        input_121 = torch.nn.functional.batch_norm(
            input_120,
            l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_120 = l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_122 = torch.nn.functional.relu(input_121, inplace=True)
        input_121 = None
        input_123 = torch.conv2d(
            input_122,
            l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            25,
        )
        input_122 = l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_124 = torch.nn.functional.batch_norm(
            input_123,
            l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_123 = l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_125 = torch.nn.functional.relu(input_124, inplace=True)
        input_124 = None
        input_126 = torch.conv2d(
            input_125,
            l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_125 = l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_127 = torch.nn.functional.batch_norm(
            input_126,
            l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_126 = l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_12 = input_119 + input_127
        input_119 = input_127 = None
        input_128 = torch.nn.functional.relu(x_12, inplace=True)
        x_12 = None
        input_129 = torch.conv2d(
            input_128,
            l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_a_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_a_modules_0_parameters_weight_ = (
            None
        )
        input_130 = torch.nn.functional.batch_norm(
            input_129,
            l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_129 = l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_131 = torch.nn.functional.relu(input_130, inplace=True)
        input_130 = None
        input_132 = torch.conv2d(
            input_131,
            l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            25,
        )
        input_131 = l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_133 = torch.nn.functional.batch_norm(
            input_132,
            l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_132 = l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_134 = torch.nn.functional.relu(input_133, inplace=True)
        input_133 = None
        input_135 = torch.conv2d(
            input_134,
            l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_134 = l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_136 = torch.nn.functional.batch_norm(
            input_135,
            l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_135 = l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_13 = input_128 + input_136
        input_128 = input_136 = None
        input_137 = torch.nn.functional.relu(x_13, inplace=True)
        x_13 = None
        input_138 = torch.conv2d(
            input_137,
            l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_a_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_a_modules_0_parameters_weight_ = (
            None
        )
        input_139 = torch.nn.functional.batch_norm(
            input_138,
            l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_138 = l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_140 = torch.nn.functional.relu(input_139, inplace=True)
        input_139 = None
        input_141 = torch.conv2d(
            input_140,
            l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            25,
        )
        input_140 = l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_142 = torch.nn.functional.batch_norm(
            input_141,
            l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_141 = l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_143 = torch.nn.functional.relu(input_142, inplace=True)
        input_142 = None
        input_144 = torch.conv2d(
            input_143,
            l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_143 = l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_145 = torch.nn.functional.batch_norm(
            input_144,
            l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_144 = l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_14 = input_137 + input_145
        input_137 = input_145 = None
        input_146 = torch.nn.functional.relu(x_14, inplace=True)
        x_14 = None
        input_147 = torch.conv2d(
            input_146,
            l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_a_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_a_modules_0_parameters_weight_ = (
            None
        )
        input_148 = torch.nn.functional.batch_norm(
            input_147,
            l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_147 = l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_149 = torch.nn.functional.relu(input_148, inplace=True)
        input_148 = None
        input_150 = torch.conv2d(
            input_149,
            l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            25,
        )
        input_149 = l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_151 = torch.nn.functional.batch_norm(
            input_150,
            l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_150 = l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_152 = torch.nn.functional.relu(input_151, inplace=True)
        input_151 = None
        input_153 = torch.conv2d(
            input_152,
            l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_152 = l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_154 = torch.nn.functional.batch_norm(
            input_153,
            l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_153 = l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_15 = input_146 + input_154
        input_146 = input_154 = None
        input_155 = torch.nn.functional.relu(x_15, inplace=True)
        x_15 = None
        input_156 = torch.conv2d(
            input_155,
            l_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_a_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_a_modules_0_parameters_weight_ = (
            None
        )
        input_157 = torch.nn.functional.batch_norm(
            input_156,
            l_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_156 = l_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_158 = torch.nn.functional.relu(input_157, inplace=True)
        input_157 = None
        input_159 = torch.conv2d(
            input_158,
            l_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            25,
        )
        input_158 = l_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_160 = torch.nn.functional.batch_norm(
            input_159,
            l_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_159 = l_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_161 = torch.nn.functional.relu(input_160, inplace=True)
        input_160 = None
        input_162 = torch.conv2d(
            input_161,
            l_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_161 = l_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_163 = torch.nn.functional.batch_norm(
            input_162,
            l_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_162 = l_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_6_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_16 = input_155 + input_163
        input_155 = input_163 = None
        input_164 = torch.nn.functional.relu(x_16, inplace=True)
        x_16 = None
        input_165 = torch.conv2d(
            input_164,
            l_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_a_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_a_modules_0_parameters_weight_ = (
            None
        )
        input_166 = torch.nn.functional.batch_norm(
            input_165,
            l_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_165 = l_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_167 = torch.nn.functional.relu(input_166, inplace=True)
        input_166 = None
        input_168 = torch.conv2d(
            input_167,
            l_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            25,
        )
        input_167 = l_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_169 = torch.nn.functional.batch_norm(
            input_168,
            l_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_168 = l_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_170 = torch.nn.functional.relu(input_169, inplace=True)
        input_169 = None
        input_171 = torch.conv2d(
            input_170,
            l_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_170 = l_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_172 = torch.nn.functional.batch_norm(
            input_171,
            l_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_171 = l_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_7_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_17 = input_164 + input_172
        input_164 = input_172 = None
        input_173 = torch.nn.functional.relu(x_17, inplace=True)
        x_17 = None
        input_174 = torch.conv2d(
            input_173,
            l_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_a_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_a_modules_0_parameters_weight_ = (
            None
        )
        input_175 = torch.nn.functional.batch_norm(
            input_174,
            l_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_174 = l_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_176 = torch.nn.functional.relu(input_175, inplace=True)
        input_175 = None
        input_177 = torch.conv2d(
            input_176,
            l_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            25,
        )
        input_176 = l_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_178 = torch.nn.functional.batch_norm(
            input_177,
            l_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_177 = l_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_179 = torch.nn.functional.relu(input_178, inplace=True)
        input_178 = None
        input_180 = torch.conv2d(
            input_179,
            l_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_179 = l_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_181 = torch.nn.functional.batch_norm(
            input_180,
            l_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_180 = l_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_8_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_18 = input_173 + input_181
        input_173 = input_181 = None
        input_182 = torch.nn.functional.relu(x_18, inplace=True)
        x_18 = None
        input_183 = torch.conv2d(
            input_182,
            l_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_a_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_a_modules_0_parameters_weight_ = (
            None
        )
        input_184 = torch.nn.functional.batch_norm(
            input_183,
            l_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_183 = l_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_185 = torch.nn.functional.relu(input_184, inplace=True)
        input_184 = None
        input_186 = torch.conv2d(
            input_185,
            l_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            25,
        )
        input_185 = l_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_187 = torch.nn.functional.batch_norm(
            input_186,
            l_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_186 = l_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_188 = torch.nn.functional.relu(input_187, inplace=True)
        input_187 = None
        input_189 = torch.conv2d(
            input_188,
            l_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_188 = l_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_190 = torch.nn.functional.batch_norm(
            input_189,
            l_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_189 = l_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_9_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_19 = input_182 + input_190
        input_182 = input_190 = None
        input_191 = torch.nn.functional.relu(x_19, inplace=True)
        x_19 = None
        input_192 = torch.conv2d(
            input_191,
            l_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_a_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_a_modules_0_parameters_weight_ = (
            None
        )
        input_193 = torch.nn.functional.batch_norm(
            input_192,
            l_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_192 = l_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_194 = torch.nn.functional.relu(input_193, inplace=True)
        input_193 = None
        input_195 = torch.conv2d(
            input_194,
            l_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            25,
        )
        input_194 = l_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_196 = torch.nn.functional.batch_norm(
            input_195,
            l_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_195 = l_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_197 = torch.nn.functional.relu(input_196, inplace=True)
        input_196 = None
        input_198 = torch.conv2d(
            input_197,
            l_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_197 = l_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_199 = torch.nn.functional.batch_norm(
            input_198,
            l_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_198 = l_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_10_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_20 = input_191 + input_199
        input_191 = input_199 = None
        input_200 = torch.nn.functional.relu(x_20, inplace=True)
        x_20 = None
        input_201 = torch.conv2d(
            input_200,
            l_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_a_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_a_modules_0_parameters_weight_ = (
            None
        )
        input_202 = torch.nn.functional.batch_norm(
            input_201,
            l_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_201 = l_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_203 = torch.nn.functional.relu(input_202, inplace=True)
        input_202 = None
        input_204 = torch.conv2d(
            input_203,
            l_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            25,
        )
        input_203 = l_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_205 = torch.nn.functional.batch_norm(
            input_204,
            l_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_204 = l_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_206 = torch.nn.functional.relu(input_205, inplace=True)
        input_205 = None
        input_207 = torch.conv2d(
            input_206,
            l_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_206 = l_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_208 = torch.nn.functional.batch_norm(
            input_207,
            l_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_207 = l_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_11_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_21 = input_200 + input_208
        input_200 = input_208 = None
        input_209 = torch.nn.functional.relu(x_21, inplace=True)
        x_21 = None
        x_22 = torch.nn.functional.adaptive_avg_pool2d(input_209, (1, 1))
        input_209 = None
        x_23 = x_22.flatten(start_dim=1)
        x_22 = None
        x_24 = torch._C._nn.linear(
            x_23,
            l_self_modules_fc_parameters_weight_,
            l_self_modules_fc_parameters_bias_,
        )
        x_23 = (
            l_self_modules_fc_parameters_weight_
        ) = l_self_modules_fc_parameters_bias_ = None
        return (x_24,)
