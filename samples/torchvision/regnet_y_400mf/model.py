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
        L_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_se_modules_fc1_parameters_weight_ = L_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_se_modules_fc1_parameters_weight_
        l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_se_modules_fc1_parameters_bias_ = L_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_se_modules_fc1_parameters_bias_
        l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_se_modules_fc2_parameters_weight_ = L_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_se_modules_fc2_parameters_weight_
        l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_se_modules_fc2_parameters_bias_ = L_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_se_modules_fc2_parameters_bias_
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
        l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_se_modules_fc1_parameters_weight_ = L_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_se_modules_fc1_parameters_weight_
        l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_se_modules_fc1_parameters_bias_ = L_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_se_modules_fc1_parameters_bias_
        l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_se_modules_fc2_parameters_weight_ = L_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_se_modules_fc2_parameters_weight_
        l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_se_modules_fc2_parameters_bias_ = L_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_se_modules_fc2_parameters_bias_
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
        l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_se_modules_fc1_parameters_weight_ = L_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_se_modules_fc1_parameters_weight_
        l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_se_modules_fc1_parameters_bias_ = L_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_se_modules_fc1_parameters_bias_
        l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_se_modules_fc2_parameters_weight_ = L_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_se_modules_fc2_parameters_weight_
        l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_se_modules_fc2_parameters_bias_ = L_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_se_modules_fc2_parameters_bias_
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
        l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_se_modules_fc1_parameters_weight_ = L_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_se_modules_fc1_parameters_weight_
        l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_se_modules_fc1_parameters_bias_ = L_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_se_modules_fc1_parameters_bias_
        l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_se_modules_fc2_parameters_weight_ = L_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_se_modules_fc2_parameters_weight_
        l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_se_modules_fc2_parameters_bias_ = L_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_se_modules_fc2_parameters_bias_
        l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_1_parameters_bias_
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
        l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_se_modules_fc1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_se_modules_fc1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_se_modules_fc1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_se_modules_fc1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_se_modules_fc2_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_se_modules_fc2_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_se_modules_fc2_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_se_modules_fc2_parameters_bias_
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
        l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_se_modules_fc1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_se_modules_fc1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_se_modules_fc1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_se_modules_fc1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_se_modules_fc2_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_se_modules_fc2_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_se_modules_fc2_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_se_modules_fc2_parameters_bias_
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
        l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_se_modules_fc1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_se_modules_fc1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_se_modules_fc1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_se_modules_fc1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_se_modules_fc2_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_se_modules_fc2_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_se_modules_fc2_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_se_modules_fc2_parameters_bias_
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
        l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_se_modules_fc1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_se_modules_fc1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_se_modules_fc1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_se_modules_fc1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_se_modules_fc2_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_se_modules_fc2_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_se_modules_fc2_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_se_modules_fc2_parameters_bias_
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
        l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_se_modules_fc1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_se_modules_fc1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_se_modules_fc1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_se_modules_fc1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_se_modules_fc2_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_se_modules_fc2_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_se_modules_fc2_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_se_modules_fc2_parameters_bias_
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
        l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_se_modules_fc1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_se_modules_fc1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_se_modules_fc1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_se_modules_fc1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_se_modules_fc2_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_se_modules_fc2_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_se_modules_fc2_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_se_modules_fc2_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_parameters_bias_
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
        l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_se_modules_fc1_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_se_modules_fc1_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_se_modules_fc1_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_se_modules_fc1_parameters_bias_
        l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_se_modules_fc2_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_se_modules_fc2_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_se_modules_fc2_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_se_modules_fc2_parameters_bias_
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
        l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_se_modules_fc1_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_se_modules_fc1_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_se_modules_fc1_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_se_modules_fc1_parameters_bias_
        l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_se_modules_fc2_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_se_modules_fc2_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_se_modules_fc2_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_se_modules_fc2_parameters_bias_
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
        l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_se_modules_fc1_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_se_modules_fc1_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_se_modules_fc1_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_se_modules_fc1_parameters_bias_
        l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_se_modules_fc2_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_se_modules_fc2_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_se_modules_fc2_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_se_modules_fc2_parameters_bias_
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
        l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_se_modules_fc1_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_se_modules_fc1_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_se_modules_fc1_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_se_modules_fc1_parameters_bias_
        l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_se_modules_fc2_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_se_modules_fc2_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_se_modules_fc2_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_se_modules_fc2_parameters_bias_
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
        l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_se_modules_fc1_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_se_modules_fc1_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_se_modules_fc1_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_se_modules_fc1_parameters_bias_
        l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_se_modules_fc2_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_se_modules_fc2_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_se_modules_fc2_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_se_modules_fc2_parameters_bias_
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
        l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_se_modules_fc1_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_se_modules_fc1_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_se_modules_fc1_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_se_modules_fc1_parameters_bias_
        l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_se_modules_fc2_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_se_modules_fc2_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_se_modules_fc2_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_se_modules_fc2_parameters_bias_
        l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_1_parameters_bias_
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
            6,
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
        scale = torch.nn.functional.adaptive_avg_pool2d(input_11, 1)
        scale_1 = torch.conv2d(
            scale,
            l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale = l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_se_modules_fc1_parameters_weight_ = l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_se_modules_fc1_parameters_bias_ = (None)
        scale_2 = torch.nn.functional.relu(scale_1, inplace=False)
        scale_1 = None
        scale_3 = torch.conv2d(
            scale_2,
            l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_2 = l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_se_modules_fc2_parameters_weight_ = l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_se_modules_fc2_parameters_bias_ = (None)
        scale_4 = torch.sigmoid(scale_3)
        scale_3 = None
        input_12 = scale_4 * input_11
        scale_4 = input_11 = None
        input_13 = torch.conv2d(
            input_12,
            l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_12 = l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_14 = torch.nn.functional.batch_norm(
            input_13,
            l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_13 = l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block1_modules_block1_0_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x = input_5 + input_14
        input_5 = input_14 = None
        input_15 = torch.nn.functional.relu(x, inplace=True)
        x = None
        input_16 = torch.conv2d(
            input_15,
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
        input_17 = torch.nn.functional.batch_norm(
            input_16,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_16 = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_1_parameters_bias_ = (None)
        input_18 = torch.conv2d(
            input_15,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_15 = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_0_parameters_weight_ = (None)
        input_19 = torch.nn.functional.batch_norm(
            input_18,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_18 = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_20 = torch.nn.functional.relu(input_19, inplace=True)
        input_19 = None
        input_21 = torch.conv2d(
            input_20,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            13,
        )
        input_20 = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_22 = torch.nn.functional.batch_norm(
            input_21,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_21 = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_23 = torch.nn.functional.relu(input_22, inplace=True)
        input_22 = None
        scale_5 = torch.nn.functional.adaptive_avg_pool2d(input_23, 1)
        scale_6 = torch.conv2d(
            scale_5,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_5 = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_se_modules_fc1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_se_modules_fc1_parameters_bias_ = (None)
        scale_7 = torch.nn.functional.relu(scale_6, inplace=False)
        scale_6 = None
        scale_8 = torch.conv2d(
            scale_7,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_7 = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_se_modules_fc2_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_se_modules_fc2_parameters_bias_ = (None)
        scale_9 = torch.sigmoid(scale_8)
        scale_8 = None
        input_24 = scale_9 * input_23
        scale_9 = input_23 = None
        input_25 = torch.conv2d(
            input_24,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_24 = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_26 = torch.nn.functional.batch_norm(
            input_25,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_25 = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_1 = input_17 + input_26
        input_17 = input_26 = None
        input_27 = torch.nn.functional.relu(x_1, inplace=True)
        x_1 = None
        input_28 = torch.conv2d(
            input_27,
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
        input_29 = torch.nn.functional.batch_norm(
            input_28,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_28 = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_30 = torch.nn.functional.relu(input_29, inplace=True)
        input_29 = None
        input_31 = torch.conv2d(
            input_30,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            13,
        )
        input_30 = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_32 = torch.nn.functional.batch_norm(
            input_31,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_31 = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_33 = torch.nn.functional.relu(input_32, inplace=True)
        input_32 = None
        scale_10 = torch.nn.functional.adaptive_avg_pool2d(input_33, 1)
        scale_11 = torch.conv2d(
            scale_10,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_10 = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_se_modules_fc1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_se_modules_fc1_parameters_bias_ = (None)
        scale_12 = torch.nn.functional.relu(scale_11, inplace=False)
        scale_11 = None
        scale_13 = torch.conv2d(
            scale_12,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_12 = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_se_modules_fc2_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_se_modules_fc2_parameters_bias_ = (None)
        scale_14 = torch.sigmoid(scale_13)
        scale_13 = None
        input_34 = scale_14 * input_33
        scale_14 = input_33 = None
        input_35 = torch.conv2d(
            input_34,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_34 = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_36 = torch.nn.functional.batch_norm(
            input_35,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_35 = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_2 = input_27 + input_36
        input_27 = input_36 = None
        input_37 = torch.nn.functional.relu(x_2, inplace=True)
        x_2 = None
        input_38 = torch.conv2d(
            input_37,
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
        input_39 = torch.nn.functional.batch_norm(
            input_38,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_38 = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_40 = torch.nn.functional.relu(input_39, inplace=True)
        input_39 = None
        input_41 = torch.conv2d(
            input_40,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            13,
        )
        input_40 = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_42 = torch.nn.functional.batch_norm(
            input_41,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_41 = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_43 = torch.nn.functional.relu(input_42, inplace=True)
        input_42 = None
        scale_15 = torch.nn.functional.adaptive_avg_pool2d(input_43, 1)
        scale_16 = torch.conv2d(
            scale_15,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_15 = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_se_modules_fc1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_se_modules_fc1_parameters_bias_ = (None)
        scale_17 = torch.nn.functional.relu(scale_16, inplace=False)
        scale_16 = None
        scale_18 = torch.conv2d(
            scale_17,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_17 = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_se_modules_fc2_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_se_modules_fc2_parameters_bias_ = (None)
        scale_19 = torch.sigmoid(scale_18)
        scale_18 = None
        input_44 = scale_19 * input_43
        scale_19 = input_43 = None
        input_45 = torch.conv2d(
            input_44,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_44 = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_46 = torch.nn.functional.batch_norm(
            input_45,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_45 = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_3 = input_37 + input_46
        input_37 = input_46 = None
        input_47 = torch.nn.functional.relu(x_3, inplace=True)
        x_3 = None
        input_48 = torch.conv2d(
            input_47,
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
        input_49 = torch.nn.functional.batch_norm(
            input_48,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_48 = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_1_parameters_bias_ = (None)
        input_50 = torch.conv2d(
            input_47,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_47 = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_0_parameters_weight_ = (None)
        input_51 = torch.nn.functional.batch_norm(
            input_50,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_50 = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_52 = torch.nn.functional.relu(input_51, inplace=True)
        input_51 = None
        input_53 = torch.conv2d(
            input_52,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            26,
        )
        input_52 = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_54 = torch.nn.functional.batch_norm(
            input_53,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_53 = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_55 = torch.nn.functional.relu(input_54, inplace=True)
        input_54 = None
        scale_20 = torch.nn.functional.adaptive_avg_pool2d(input_55, 1)
        scale_21 = torch.conv2d(
            scale_20,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_20 = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_se_modules_fc1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_se_modules_fc1_parameters_bias_ = (None)
        scale_22 = torch.nn.functional.relu(scale_21, inplace=False)
        scale_21 = None
        scale_23 = torch.conv2d(
            scale_22,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_22 = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_se_modules_fc2_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_se_modules_fc2_parameters_bias_ = (None)
        scale_24 = torch.sigmoid(scale_23)
        scale_23 = None
        input_56 = scale_24 * input_55
        scale_24 = input_55 = None
        input_57 = torch.conv2d(
            input_56,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_56 = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_58 = torch.nn.functional.batch_norm(
            input_57,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_57 = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_4 = input_49 + input_58
        input_49 = input_58 = None
        input_59 = torch.nn.functional.relu(x_4, inplace=True)
        x_4 = None
        input_60 = torch.conv2d(
            input_59,
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
        input_61 = torch.nn.functional.batch_norm(
            input_60,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_60 = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_62 = torch.nn.functional.relu(input_61, inplace=True)
        input_61 = None
        input_63 = torch.conv2d(
            input_62,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            26,
        )
        input_62 = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_64 = torch.nn.functional.batch_norm(
            input_63,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_63 = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_65 = torch.nn.functional.relu(input_64, inplace=True)
        input_64 = None
        scale_25 = torch.nn.functional.adaptive_avg_pool2d(input_65, 1)
        scale_26 = torch.conv2d(
            scale_25,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_25 = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_se_modules_fc1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_se_modules_fc1_parameters_bias_ = (None)
        scale_27 = torch.nn.functional.relu(scale_26, inplace=False)
        scale_26 = None
        scale_28 = torch.conv2d(
            scale_27,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_27 = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_se_modules_fc2_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_se_modules_fc2_parameters_bias_ = (None)
        scale_29 = torch.sigmoid(scale_28)
        scale_28 = None
        input_66 = scale_29 * input_65
        scale_29 = input_65 = None
        input_67 = torch.conv2d(
            input_66,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_66 = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_68 = torch.nn.functional.batch_norm(
            input_67,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_67 = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_5 = input_59 + input_68
        input_59 = input_68 = None
        input_69 = torch.nn.functional.relu(x_5, inplace=True)
        x_5 = None
        input_70 = torch.conv2d(
            input_69,
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
        input_71 = torch.nn.functional.batch_norm(
            input_70,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_70 = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_72 = torch.nn.functional.relu(input_71, inplace=True)
        input_71 = None
        input_73 = torch.conv2d(
            input_72,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            26,
        )
        input_72 = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_74 = torch.nn.functional.batch_norm(
            input_73,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_73 = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_75 = torch.nn.functional.relu(input_74, inplace=True)
        input_74 = None
        scale_30 = torch.nn.functional.adaptive_avg_pool2d(input_75, 1)
        scale_31 = torch.conv2d(
            scale_30,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_30 = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_se_modules_fc1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_se_modules_fc1_parameters_bias_ = (None)
        scale_32 = torch.nn.functional.relu(scale_31, inplace=False)
        scale_31 = None
        scale_33 = torch.conv2d(
            scale_32,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_32 = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_se_modules_fc2_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_se_modules_fc2_parameters_bias_ = (None)
        scale_34 = torch.sigmoid(scale_33)
        scale_33 = None
        input_76 = scale_34 * input_75
        scale_34 = input_75 = None
        input_77 = torch.conv2d(
            input_76,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_76 = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_78 = torch.nn.functional.batch_norm(
            input_77,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_77 = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_6 = input_69 + input_78
        input_69 = input_78 = None
        input_79 = torch.nn.functional.relu(x_6, inplace=True)
        x_6 = None
        input_80 = torch.conv2d(
            input_79,
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
        input_81 = torch.nn.functional.batch_norm(
            input_80,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_80 = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_82 = torch.nn.functional.relu(input_81, inplace=True)
        input_81 = None
        input_83 = torch.conv2d(
            input_82,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            26,
        )
        input_82 = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_84 = torch.nn.functional.batch_norm(
            input_83,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_83 = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_85 = torch.nn.functional.relu(input_84, inplace=True)
        input_84 = None
        scale_35 = torch.nn.functional.adaptive_avg_pool2d(input_85, 1)
        scale_36 = torch.conv2d(
            scale_35,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_35 = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_se_modules_fc1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_se_modules_fc1_parameters_bias_ = (None)
        scale_37 = torch.nn.functional.relu(scale_36, inplace=False)
        scale_36 = None
        scale_38 = torch.conv2d(
            scale_37,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_37 = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_se_modules_fc2_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_se_modules_fc2_parameters_bias_ = (None)
        scale_39 = torch.sigmoid(scale_38)
        scale_38 = None
        input_86 = scale_39 * input_85
        scale_39 = input_85 = None
        input_87 = torch.conv2d(
            input_86,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_86 = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_88 = torch.nn.functional.batch_norm(
            input_87,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_87 = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_7 = input_79 + input_88
        input_79 = input_88 = None
        input_89 = torch.nn.functional.relu(x_7, inplace=True)
        x_7 = None
        input_90 = torch.conv2d(
            input_89,
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
        input_91 = torch.nn.functional.batch_norm(
            input_90,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_90 = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_92 = torch.nn.functional.relu(input_91, inplace=True)
        input_91 = None
        input_93 = torch.conv2d(
            input_92,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            26,
        )
        input_92 = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_94 = torch.nn.functional.batch_norm(
            input_93,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_93 = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_95 = torch.nn.functional.relu(input_94, inplace=True)
        input_94 = None
        scale_40 = torch.nn.functional.adaptive_avg_pool2d(input_95, 1)
        scale_41 = torch.conv2d(
            scale_40,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_40 = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_se_modules_fc1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_se_modules_fc1_parameters_bias_ = (None)
        scale_42 = torch.nn.functional.relu(scale_41, inplace=False)
        scale_41 = None
        scale_43 = torch.conv2d(
            scale_42,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_42 = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_se_modules_fc2_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_se_modules_fc2_parameters_bias_ = (None)
        scale_44 = torch.sigmoid(scale_43)
        scale_43 = None
        input_96 = scale_44 * input_95
        scale_44 = input_95 = None
        input_97 = torch.conv2d(
            input_96,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_96 = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_98 = torch.nn.functional.batch_norm(
            input_97,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_97 = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_8 = input_89 + input_98
        input_89 = input_98 = None
        input_99 = torch.nn.functional.relu(x_8, inplace=True)
        x_8 = None
        input_100 = torch.conv2d(
            input_99,
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
        input_101 = torch.nn.functional.batch_norm(
            input_100,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_100 = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_102 = torch.nn.functional.relu(input_101, inplace=True)
        input_101 = None
        input_103 = torch.conv2d(
            input_102,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            26,
        )
        input_102 = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_104 = torch.nn.functional.batch_norm(
            input_103,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_103 = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_105 = torch.nn.functional.relu(input_104, inplace=True)
        input_104 = None
        scale_45 = torch.nn.functional.adaptive_avg_pool2d(input_105, 1)
        scale_46 = torch.conv2d(
            scale_45,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_45 = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_se_modules_fc1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_se_modules_fc1_parameters_bias_ = (None)
        scale_47 = torch.nn.functional.relu(scale_46, inplace=False)
        scale_46 = None
        scale_48 = torch.conv2d(
            scale_47,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_47 = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_se_modules_fc2_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_se_modules_fc2_parameters_bias_ = (None)
        scale_49 = torch.sigmoid(scale_48)
        scale_48 = None
        input_106 = scale_49 * input_105
        scale_49 = input_105 = None
        input_107 = torch.conv2d(
            input_106,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_106 = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_108 = torch.nn.functional.batch_norm(
            input_107,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_107 = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_9 = input_99 + input_108
        input_99 = input_108 = None
        input_109 = torch.nn.functional.relu(x_9, inplace=True)
        x_9 = None
        input_110 = torch.conv2d(
            input_109,
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
        input_111 = torch.nn.functional.batch_norm(
            input_110,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_110 = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_1_parameters_bias_ = (None)
        input_112 = torch.conv2d(
            input_109,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_109 = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_0_parameters_weight_ = (None)
        input_113 = torch.nn.functional.batch_norm(
            input_112,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_112 = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_114 = torch.nn.functional.relu(input_113, inplace=True)
        input_113 = None
        input_115 = torch.conv2d(
            input_114,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            55,
        )
        input_114 = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_116 = torch.nn.functional.batch_norm(
            input_115,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_115 = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_117 = torch.nn.functional.relu(input_116, inplace=True)
        input_116 = None
        scale_50 = torch.nn.functional.adaptive_avg_pool2d(input_117, 1)
        scale_51 = torch.conv2d(
            scale_50,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_50 = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_se_modules_fc1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_se_modules_fc1_parameters_bias_ = (None)
        scale_52 = torch.nn.functional.relu(scale_51, inplace=False)
        scale_51 = None
        scale_53 = torch.conv2d(
            scale_52,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_52 = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_se_modules_fc2_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_se_modules_fc2_parameters_bias_ = (None)
        scale_54 = torch.sigmoid(scale_53)
        scale_53 = None
        input_118 = scale_54 * input_117
        scale_54 = input_117 = None
        input_119 = torch.conv2d(
            input_118,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_118 = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_120 = torch.nn.functional.batch_norm(
            input_119,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_119 = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_10 = input_111 + input_120
        input_111 = input_120 = None
        input_121 = torch.nn.functional.relu(x_10, inplace=True)
        x_10 = None
        input_122 = torch.conv2d(
            input_121,
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
        input_123 = torch.nn.functional.batch_norm(
            input_122,
            l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_122 = l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_124 = torch.nn.functional.relu(input_123, inplace=True)
        input_123 = None
        input_125 = torch.conv2d(
            input_124,
            l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            55,
        )
        input_124 = l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_126 = torch.nn.functional.batch_norm(
            input_125,
            l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_125 = l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_127 = torch.nn.functional.relu(input_126, inplace=True)
        input_126 = None
        scale_55 = torch.nn.functional.adaptive_avg_pool2d(input_127, 1)
        scale_56 = torch.conv2d(
            scale_55,
            l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_55 = l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_se_modules_fc1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_se_modules_fc1_parameters_bias_ = (None)
        scale_57 = torch.nn.functional.relu(scale_56, inplace=False)
        scale_56 = None
        scale_58 = torch.conv2d(
            scale_57,
            l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_57 = l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_se_modules_fc2_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_se_modules_fc2_parameters_bias_ = (None)
        scale_59 = torch.sigmoid(scale_58)
        scale_58 = None
        input_128 = scale_59 * input_127
        scale_59 = input_127 = None
        input_129 = torch.conv2d(
            input_128,
            l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_128 = l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_130 = torch.nn.functional.batch_norm(
            input_129,
            l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_129 = l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_1_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_11 = input_121 + input_130
        input_121 = input_130 = None
        input_131 = torch.nn.functional.relu(x_11, inplace=True)
        x_11 = None
        input_132 = torch.conv2d(
            input_131,
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
        input_133 = torch.nn.functional.batch_norm(
            input_132,
            l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_132 = l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_134 = torch.nn.functional.relu(input_133, inplace=True)
        input_133 = None
        input_135 = torch.conv2d(
            input_134,
            l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            55,
        )
        input_134 = l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_136 = torch.nn.functional.batch_norm(
            input_135,
            l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_135 = l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_137 = torch.nn.functional.relu(input_136, inplace=True)
        input_136 = None
        scale_60 = torch.nn.functional.adaptive_avg_pool2d(input_137, 1)
        scale_61 = torch.conv2d(
            scale_60,
            l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_60 = l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_se_modules_fc1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_se_modules_fc1_parameters_bias_ = (None)
        scale_62 = torch.nn.functional.relu(scale_61, inplace=False)
        scale_61 = None
        scale_63 = torch.conv2d(
            scale_62,
            l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_62 = l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_se_modules_fc2_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_se_modules_fc2_parameters_bias_ = (None)
        scale_64 = torch.sigmoid(scale_63)
        scale_63 = None
        input_138 = scale_64 * input_137
        scale_64 = input_137 = None
        input_139 = torch.conv2d(
            input_138,
            l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_138 = l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_140 = torch.nn.functional.batch_norm(
            input_139,
            l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_139 = l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_2_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_12 = input_131 + input_140
        input_131 = input_140 = None
        input_141 = torch.nn.functional.relu(x_12, inplace=True)
        x_12 = None
        input_142 = torch.conv2d(
            input_141,
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
        input_143 = torch.nn.functional.batch_norm(
            input_142,
            l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_142 = l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_144 = torch.nn.functional.relu(input_143, inplace=True)
        input_143 = None
        input_145 = torch.conv2d(
            input_144,
            l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            55,
        )
        input_144 = l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_146 = torch.nn.functional.batch_norm(
            input_145,
            l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_145 = l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_147 = torch.nn.functional.relu(input_146, inplace=True)
        input_146 = None
        scale_65 = torch.nn.functional.adaptive_avg_pool2d(input_147, 1)
        scale_66 = torch.conv2d(
            scale_65,
            l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_65 = l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_se_modules_fc1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_se_modules_fc1_parameters_bias_ = (None)
        scale_67 = torch.nn.functional.relu(scale_66, inplace=False)
        scale_66 = None
        scale_68 = torch.conv2d(
            scale_67,
            l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_67 = l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_se_modules_fc2_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_se_modules_fc2_parameters_bias_ = (None)
        scale_69 = torch.sigmoid(scale_68)
        scale_68 = None
        input_148 = scale_69 * input_147
        scale_69 = input_147 = None
        input_149 = torch.conv2d(
            input_148,
            l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_148 = l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_150 = torch.nn.functional.batch_norm(
            input_149,
            l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_149 = l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_3_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_13 = input_141 + input_150
        input_141 = input_150 = None
        input_151 = torch.nn.functional.relu(x_13, inplace=True)
        x_13 = None
        input_152 = torch.conv2d(
            input_151,
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
        input_153 = torch.nn.functional.batch_norm(
            input_152,
            l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_152 = l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_154 = torch.nn.functional.relu(input_153, inplace=True)
        input_153 = None
        input_155 = torch.conv2d(
            input_154,
            l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            55,
        )
        input_154 = l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_156 = torch.nn.functional.batch_norm(
            input_155,
            l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_155 = l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_157 = torch.nn.functional.relu(input_156, inplace=True)
        input_156 = None
        scale_70 = torch.nn.functional.adaptive_avg_pool2d(input_157, 1)
        scale_71 = torch.conv2d(
            scale_70,
            l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_70 = l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_se_modules_fc1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_se_modules_fc1_parameters_bias_ = (None)
        scale_72 = torch.nn.functional.relu(scale_71, inplace=False)
        scale_71 = None
        scale_73 = torch.conv2d(
            scale_72,
            l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_72 = l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_se_modules_fc2_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_se_modules_fc2_parameters_bias_ = (None)
        scale_74 = torch.sigmoid(scale_73)
        scale_73 = None
        input_158 = scale_74 * input_157
        scale_74 = input_157 = None
        input_159 = torch.conv2d(
            input_158,
            l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_158 = l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_160 = torch.nn.functional.batch_norm(
            input_159,
            l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_159 = l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_4_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_14 = input_151 + input_160
        input_151 = input_160 = None
        input_161 = torch.nn.functional.relu(x_14, inplace=True)
        x_14 = None
        input_162 = torch.conv2d(
            input_161,
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
        input_163 = torch.nn.functional.batch_norm(
            input_162,
            l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_162 = l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_164 = torch.nn.functional.relu(input_163, inplace=True)
        input_163 = None
        input_165 = torch.conv2d(
            input_164,
            l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            55,
        )
        input_164 = l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_166 = torch.nn.functional.batch_norm(
            input_165,
            l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_165 = l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_167 = torch.nn.functional.relu(input_166, inplace=True)
        input_166 = None
        scale_75 = torch.nn.functional.adaptive_avg_pool2d(input_167, 1)
        scale_76 = torch.conv2d(
            scale_75,
            l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_75 = l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_se_modules_fc1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_se_modules_fc1_parameters_bias_ = (None)
        scale_77 = torch.nn.functional.relu(scale_76, inplace=False)
        scale_76 = None
        scale_78 = torch.conv2d(
            scale_77,
            l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_77 = l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_se_modules_fc2_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_se_modules_fc2_parameters_bias_ = (None)
        scale_79 = torch.sigmoid(scale_78)
        scale_78 = None
        input_168 = scale_79 * input_167
        scale_79 = input_167 = None
        input_169 = torch.conv2d(
            input_168,
            l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_168 = l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_170 = torch.nn.functional.batch_norm(
            input_169,
            l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_169 = l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_5_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_15 = input_161 + input_170
        input_161 = input_170 = None
        input_171 = torch.nn.functional.relu(x_15, inplace=True)
        x_15 = None
        x_16 = torch.nn.functional.adaptive_avg_pool2d(input_171, (1, 1))
        input_171 = None
        x_17 = x_16.flatten(start_dim=1)
        x_16 = None
        x_18 = torch._C._nn.linear(
            x_17,
            l_self_modules_fc_parameters_weight_,
            l_self_modules_fc_parameters_bias_,
        )
        x_17 = (
            l_self_modules_fc_parameters_weight_
        ) = l_self_modules_fc_parameters_bias_ = None
        return (x_18,)
