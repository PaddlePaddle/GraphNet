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
        L_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_a_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_a_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_a_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_a_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_a_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_b_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_b_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_b_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_b_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_b_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_c_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_c_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_c_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_c_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_c_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_a_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_a_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_a_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_a_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_a_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_b_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_b_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_b_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_b_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_b_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_c_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_c_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_c_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_c_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_c_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_a_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_a_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_a_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_a_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_a_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_b_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_b_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_b_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_b_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_b_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_c_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_c_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_c_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_c_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_c_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_a_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_a_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_a_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_a_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_a_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_b_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_b_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_b_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_b_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_b_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_c_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_c_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_c_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_c_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_c_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_se_modules_fc1_parameters_weight_ = L_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_se_modules_fc1_parameters_weight_
        l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_se_modules_fc1_parameters_bias_ = L_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_se_modules_fc1_parameters_bias_
        l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_se_modules_fc2_parameters_weight_ = L_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_se_modules_fc2_parameters_weight_
        l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_se_modules_fc2_parameters_bias_ = L_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_se_modules_fc2_parameters_bias_
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
        l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_se_modules_fc1_parameters_weight_ = L_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_se_modules_fc1_parameters_weight_
        l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_se_modules_fc1_parameters_bias_ = L_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_se_modules_fc1_parameters_bias_
        l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_se_modules_fc2_parameters_weight_ = L_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_se_modules_fc2_parameters_weight_
        l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_se_modules_fc2_parameters_bias_ = L_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_se_modules_fc2_parameters_bias_
        l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_a_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_a_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_a_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_a_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_a_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_a_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_a_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_a_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_a_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_a_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_b_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_b_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_b_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_b_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_b_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_b_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_b_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_b_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_b_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_b_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_se_modules_fc1_parameters_weight_ = L_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_se_modules_fc1_parameters_weight_
        l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_se_modules_fc1_parameters_bias_ = L_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_se_modules_fc1_parameters_bias_
        l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_se_modules_fc2_parameters_weight_ = L_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_se_modules_fc2_parameters_weight_
        l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_se_modules_fc2_parameters_bias_ = L_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_se_modules_fc2_parameters_bias_
        l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_c_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_c_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_c_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_c_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_c_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_c_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_c_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_c_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_c_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_c_modules_1_parameters_bias_
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
        l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_se_modules_fc1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_se_modules_fc1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_se_modules_fc1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_se_modules_fc1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_se_modules_fc2_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_se_modules_fc2_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_se_modules_fc2_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_se_modules_fc2_parameters_bias_
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
        l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_se_modules_fc1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_se_modules_fc1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_se_modules_fc1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_se_modules_fc1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_se_modules_fc2_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_se_modules_fc2_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_se_modules_fc2_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_se_modules_fc2_parameters_bias_
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
        l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_se_modules_fc1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_se_modules_fc1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_se_modules_fc1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_se_modules_fc1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_se_modules_fc2_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_se_modules_fc2_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_se_modules_fc2_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_se_modules_fc2_parameters_bias_
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
        l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_se_modules_fc1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_se_modules_fc1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_se_modules_fc1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_se_modules_fc1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_se_modules_fc2_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_se_modules_fc2_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_se_modules_fc2_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_se_modules_fc2_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_a_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_a_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_a_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_a_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_a_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_a_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_a_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_a_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_a_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_a_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_b_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_b_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_b_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_b_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_b_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_b_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_b_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_b_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_b_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_b_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_se_modules_fc1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_se_modules_fc1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_se_modules_fc1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_se_modules_fc1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_se_modules_fc2_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_se_modules_fc2_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_se_modules_fc2_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_se_modules_fc2_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_c_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_c_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_c_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_c_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_c_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_c_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_c_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_c_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_c_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_c_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_a_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_a_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_a_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_a_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_a_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_a_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_a_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_a_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_a_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_a_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_b_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_b_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_b_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_b_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_b_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_b_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_b_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_b_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_b_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_b_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_se_modules_fc1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_se_modules_fc1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_se_modules_fc1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_se_modules_fc1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_se_modules_fc2_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_se_modules_fc2_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_se_modules_fc2_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_se_modules_fc2_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_c_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_c_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_c_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_c_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_c_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_c_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_c_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_c_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_c_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_c_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_a_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_a_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_a_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_a_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_a_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_a_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_a_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_a_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_a_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_a_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_b_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_b_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_b_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_b_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_b_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_b_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_b_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_b_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_b_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_b_modules_1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_se_modules_fc1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_se_modules_fc1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_se_modules_fc1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_se_modules_fc1_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_se_modules_fc2_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_se_modules_fc2_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_se_modules_fc2_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_se_modules_fc2_parameters_bias_
        l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_c_modules_0_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_c_modules_0_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_c_modules_1_buffers_running_mean_ = L_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_c_modules_1_buffers_running_mean_
        l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_c_modules_1_buffers_running_var_ = L_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_c_modules_1_buffers_running_var_
        l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_c_modules_1_parameters_weight_ = L_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_c_modules_1_parameters_weight_
        l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_c_modules_1_parameters_bias_ = L_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_c_modules_1_parameters_bias_
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
        input_17 = torch.nn.functional.batch_norm(
            input_16,
            l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_16 = l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_18 = torch.nn.functional.relu(input_17, inplace=True)
        input_17 = None
        input_19 = torch.conv2d(
            input_18,
            l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            3,
        )
        input_18 = l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_20 = torch.nn.functional.batch_norm(
            input_19,
            l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_19 = l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_21 = torch.nn.functional.relu(input_20, inplace=True)
        input_20 = None
        scale_5 = torch.nn.functional.adaptive_avg_pool2d(input_21, 1)
        scale_6 = torch.conv2d(
            scale_5,
            l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_5 = l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_se_modules_fc1_parameters_weight_ = l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_se_modules_fc1_parameters_bias_ = (None)
        scale_7 = torch.nn.functional.relu(scale_6, inplace=False)
        scale_6 = None
        scale_8 = torch.conv2d(
            scale_7,
            l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_7 = l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_se_modules_fc2_parameters_weight_ = l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_se_modules_fc2_parameters_bias_ = (None)
        scale_9 = torch.sigmoid(scale_8)
        scale_8 = None
        input_22 = scale_9 * input_21
        scale_9 = input_21 = None
        input_23 = torch.conv2d(
            input_22,
            l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_22 = l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_24 = torch.nn.functional.batch_norm(
            input_23,
            l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_23 = l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block1_modules_block1_1_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_1 = input_15 + input_24
        input_15 = input_24 = None
        input_25 = torch.nn.functional.relu(x_1, inplace=True)
        x_1 = None
        input_26 = torch.conv2d(
            input_25,
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
        input_27 = torch.nn.functional.batch_norm(
            input_26,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_26 = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_proj_modules_1_parameters_bias_ = (None)
        input_28 = torch.conv2d(
            input_25,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_25 = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_0_parameters_weight_ = (None)
        input_29 = torch.nn.functional.batch_norm(
            input_28,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_28 = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_30 = torch.nn.functional.relu(input_29, inplace=True)
        input_29 = None
        input_31 = torch.conv2d(
            input_30,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            9,
        )
        input_30 = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_32 = torch.nn.functional.batch_norm(
            input_31,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_31 = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_33 = torch.nn.functional.relu(input_32, inplace=True)
        input_32 = None
        scale_10 = torch.nn.functional.adaptive_avg_pool2d(input_33, 1)
        scale_11 = torch.conv2d(
            scale_10,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_10 = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_se_modules_fc1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_se_modules_fc1_parameters_bias_ = (None)
        scale_12 = torch.nn.functional.relu(scale_11, inplace=False)
        scale_11 = None
        scale_13 = torch.conv2d(
            scale_12,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_12 = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_se_modules_fc2_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_se_modules_fc2_parameters_bias_ = (None)
        scale_14 = torch.sigmoid(scale_13)
        scale_13 = None
        input_34 = scale_14 * input_33
        scale_14 = input_33 = None
        input_35 = torch.conv2d(
            input_34,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_34 = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_36 = torch.nn.functional.batch_norm(
            input_35,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_35 = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_0_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_2 = input_27 + input_36
        input_27 = input_36 = None
        input_37 = torch.nn.functional.relu(x_2, inplace=True)
        x_2 = None
        input_38 = torch.conv2d(
            input_37,
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
        input_39 = torch.nn.functional.batch_norm(
            input_38,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_38 = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_40 = torch.nn.functional.relu(input_39, inplace=True)
        input_39 = None
        input_41 = torch.conv2d(
            input_40,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            9,
        )
        input_40 = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_42 = torch.nn.functional.batch_norm(
            input_41,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_41 = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_43 = torch.nn.functional.relu(input_42, inplace=True)
        input_42 = None
        scale_15 = torch.nn.functional.adaptive_avg_pool2d(input_43, 1)
        scale_16 = torch.conv2d(
            scale_15,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_15 = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_se_modules_fc1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_se_modules_fc1_parameters_bias_ = (None)
        scale_17 = torch.nn.functional.relu(scale_16, inplace=False)
        scale_16 = None
        scale_18 = torch.conv2d(
            scale_17,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_17 = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_se_modules_fc2_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_se_modules_fc2_parameters_bias_ = (None)
        scale_19 = torch.sigmoid(scale_18)
        scale_18 = None
        input_44 = scale_19 * input_43
        scale_19 = input_43 = None
        input_45 = torch.conv2d(
            input_44,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_44 = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_46 = torch.nn.functional.batch_norm(
            input_45,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_45 = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_1_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_3 = input_37 + input_46
        input_37 = input_46 = None
        input_47 = torch.nn.functional.relu(x_3, inplace=True)
        x_3 = None
        input_48 = torch.conv2d(
            input_47,
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
        input_49 = torch.nn.functional.batch_norm(
            input_48,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_48 = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_50 = torch.nn.functional.relu(input_49, inplace=True)
        input_49 = None
        input_51 = torch.conv2d(
            input_50,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            9,
        )
        input_50 = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_52 = torch.nn.functional.batch_norm(
            input_51,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_51 = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_53 = torch.nn.functional.relu(input_52, inplace=True)
        input_52 = None
        scale_20 = torch.nn.functional.adaptive_avg_pool2d(input_53, 1)
        scale_21 = torch.conv2d(
            scale_20,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_20 = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_se_modules_fc1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_se_modules_fc1_parameters_bias_ = (None)
        scale_22 = torch.nn.functional.relu(scale_21, inplace=False)
        scale_21 = None
        scale_23 = torch.conv2d(
            scale_22,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_22 = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_se_modules_fc2_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_se_modules_fc2_parameters_bias_ = (None)
        scale_24 = torch.sigmoid(scale_23)
        scale_23 = None
        input_54 = scale_24 * input_53
        scale_24 = input_53 = None
        input_55 = torch.conv2d(
            input_54,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_54 = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_56 = torch.nn.functional.batch_norm(
            input_55,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_55 = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_2_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_4 = input_47 + input_56
        input_47 = input_56 = None
        input_57 = torch.nn.functional.relu(x_4, inplace=True)
        x_4 = None
        input_58 = torch.conv2d(
            input_57,
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
        input_59 = torch.nn.functional.batch_norm(
            input_58,
            l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_58 = l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_60 = torch.nn.functional.relu(input_59, inplace=True)
        input_59 = None
        input_61 = torch.conv2d(
            input_60,
            l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            9,
        )
        input_60 = l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_62 = torch.nn.functional.batch_norm(
            input_61,
            l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_61 = l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_63 = torch.nn.functional.relu(input_62, inplace=True)
        input_62 = None
        scale_25 = torch.nn.functional.adaptive_avg_pool2d(input_63, 1)
        scale_26 = torch.conv2d(
            scale_25,
            l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_25 = l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_se_modules_fc1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_se_modules_fc1_parameters_bias_ = (None)
        scale_27 = torch.nn.functional.relu(scale_26, inplace=False)
        scale_26 = None
        scale_28 = torch.conv2d(
            scale_27,
            l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_27 = l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_se_modules_fc2_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_se_modules_fc2_parameters_bias_ = (None)
        scale_29 = torch.sigmoid(scale_28)
        scale_28 = None
        input_64 = scale_29 * input_63
        scale_29 = input_63 = None
        input_65 = torch.conv2d(
            input_64,
            l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_64 = l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_66 = torch.nn.functional.batch_norm(
            input_65,
            l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_65 = l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_3_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_5 = input_57 + input_66
        input_57 = input_66 = None
        input_67 = torch.nn.functional.relu(x_5, inplace=True)
        x_5 = None
        input_68 = torch.conv2d(
            input_67,
            l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_a_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_a_modules_0_parameters_weight_ = (
            None
        )
        input_69 = torch.nn.functional.batch_norm(
            input_68,
            l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_68 = l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_70 = torch.nn.functional.relu(input_69, inplace=True)
        input_69 = None
        input_71 = torch.conv2d(
            input_70,
            l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            9,
        )
        input_70 = l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_72 = torch.nn.functional.batch_norm(
            input_71,
            l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_71 = l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_73 = torch.nn.functional.relu(input_72, inplace=True)
        input_72 = None
        scale_30 = torch.nn.functional.adaptive_avg_pool2d(input_73, 1)
        scale_31 = torch.conv2d(
            scale_30,
            l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_30 = l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_se_modules_fc1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_se_modules_fc1_parameters_bias_ = (None)
        scale_32 = torch.nn.functional.relu(scale_31, inplace=False)
        scale_31 = None
        scale_33 = torch.conv2d(
            scale_32,
            l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_32 = l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_se_modules_fc2_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_se_modules_fc2_parameters_bias_ = (None)
        scale_34 = torch.sigmoid(scale_33)
        scale_33 = None
        input_74 = scale_34 * input_73
        scale_34 = input_73 = None
        input_75 = torch.conv2d(
            input_74,
            l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_74 = l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_76 = torch.nn.functional.batch_norm(
            input_75,
            l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_75 = l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block2_modules_block2_4_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_6 = input_67 + input_76
        input_67 = input_76 = None
        input_77 = torch.nn.functional.relu(x_6, inplace=True)
        x_6 = None
        input_78 = torch.conv2d(
            input_77,
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
        input_79 = torch.nn.functional.batch_norm(
            input_78,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_78 = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_proj_modules_1_parameters_bias_ = (None)
        input_80 = torch.conv2d(
            input_77,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_77 = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_0_parameters_weight_ = (None)
        input_81 = torch.nn.functional.batch_norm(
            input_80,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_80 = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_82 = torch.nn.functional.relu(input_81, inplace=True)
        input_81 = None
        input_83 = torch.conv2d(
            input_82,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            24,
        )
        input_82 = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_84 = torch.nn.functional.batch_norm(
            input_83,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_83 = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_85 = torch.nn.functional.relu(input_84, inplace=True)
        input_84 = None
        scale_35 = torch.nn.functional.adaptive_avg_pool2d(input_85, 1)
        scale_36 = torch.conv2d(
            scale_35,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_35 = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_se_modules_fc1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_se_modules_fc1_parameters_bias_ = (None)
        scale_37 = torch.nn.functional.relu(scale_36, inplace=False)
        scale_36 = None
        scale_38 = torch.conv2d(
            scale_37,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_37 = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_se_modules_fc2_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_se_modules_fc2_parameters_bias_ = (None)
        scale_39 = torch.sigmoid(scale_38)
        scale_38 = None
        input_86 = scale_39 * input_85
        scale_39 = input_85 = None
        input_87 = torch.conv2d(
            input_86,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_86 = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_88 = torch.nn.functional.batch_norm(
            input_87,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_87 = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_0_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_7 = input_79 + input_88
        input_79 = input_88 = None
        input_89 = torch.nn.functional.relu(x_7, inplace=True)
        x_7 = None
        input_90 = torch.conv2d(
            input_89,
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
        input_91 = torch.nn.functional.batch_norm(
            input_90,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_90 = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_92 = torch.nn.functional.relu(input_91, inplace=True)
        input_91 = None
        input_93 = torch.conv2d(
            input_92,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            24,
        )
        input_92 = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_94 = torch.nn.functional.batch_norm(
            input_93,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_93 = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_95 = torch.nn.functional.relu(input_94, inplace=True)
        input_94 = None
        scale_40 = torch.nn.functional.adaptive_avg_pool2d(input_95, 1)
        scale_41 = torch.conv2d(
            scale_40,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_40 = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_se_modules_fc1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_se_modules_fc1_parameters_bias_ = (None)
        scale_42 = torch.nn.functional.relu(scale_41, inplace=False)
        scale_41 = None
        scale_43 = torch.conv2d(
            scale_42,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_42 = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_se_modules_fc2_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_se_modules_fc2_parameters_bias_ = (None)
        scale_44 = torch.sigmoid(scale_43)
        scale_43 = None
        input_96 = scale_44 * input_95
        scale_44 = input_95 = None
        input_97 = torch.conv2d(
            input_96,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_96 = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_98 = torch.nn.functional.batch_norm(
            input_97,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_97 = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_1_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_8 = input_89 + input_98
        input_89 = input_98 = None
        input_99 = torch.nn.functional.relu(x_8, inplace=True)
        x_8 = None
        input_100 = torch.conv2d(
            input_99,
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
        input_101 = torch.nn.functional.batch_norm(
            input_100,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_100 = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_102 = torch.nn.functional.relu(input_101, inplace=True)
        input_101 = None
        input_103 = torch.conv2d(
            input_102,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            24,
        )
        input_102 = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_104 = torch.nn.functional.batch_norm(
            input_103,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_103 = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_105 = torch.nn.functional.relu(input_104, inplace=True)
        input_104 = None
        scale_45 = torch.nn.functional.adaptive_avg_pool2d(input_105, 1)
        scale_46 = torch.conv2d(
            scale_45,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_45 = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_se_modules_fc1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_se_modules_fc1_parameters_bias_ = (None)
        scale_47 = torch.nn.functional.relu(scale_46, inplace=False)
        scale_46 = None
        scale_48 = torch.conv2d(
            scale_47,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_47 = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_se_modules_fc2_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_se_modules_fc2_parameters_bias_ = (None)
        scale_49 = torch.sigmoid(scale_48)
        scale_48 = None
        input_106 = scale_49 * input_105
        scale_49 = input_105 = None
        input_107 = torch.conv2d(
            input_106,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_106 = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_108 = torch.nn.functional.batch_norm(
            input_107,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_107 = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_2_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_9 = input_99 + input_108
        input_99 = input_108 = None
        input_109 = torch.nn.functional.relu(x_9, inplace=True)
        x_9 = None
        input_110 = torch.conv2d(
            input_109,
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
        input_111 = torch.nn.functional.batch_norm(
            input_110,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_110 = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_112 = torch.nn.functional.relu(input_111, inplace=True)
        input_111 = None
        input_113 = torch.conv2d(
            input_112,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            24,
        )
        input_112 = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_114 = torch.nn.functional.batch_norm(
            input_113,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_113 = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_115 = torch.nn.functional.relu(input_114, inplace=True)
        input_114 = None
        scale_50 = torch.nn.functional.adaptive_avg_pool2d(input_115, 1)
        scale_51 = torch.conv2d(
            scale_50,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_50 = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_se_modules_fc1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_se_modules_fc1_parameters_bias_ = (None)
        scale_52 = torch.nn.functional.relu(scale_51, inplace=False)
        scale_51 = None
        scale_53 = torch.conv2d(
            scale_52,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_52 = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_se_modules_fc2_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_se_modules_fc2_parameters_bias_ = (None)
        scale_54 = torch.sigmoid(scale_53)
        scale_53 = None
        input_116 = scale_54 * input_115
        scale_54 = input_115 = None
        input_117 = torch.conv2d(
            input_116,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_116 = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_118 = torch.nn.functional.batch_norm(
            input_117,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_117 = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_3_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_10 = input_109 + input_118
        input_109 = input_118 = None
        input_119 = torch.nn.functional.relu(x_10, inplace=True)
        x_10 = None
        input_120 = torch.conv2d(
            input_119,
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
        input_121 = torch.nn.functional.batch_norm(
            input_120,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_120 = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_122 = torch.nn.functional.relu(input_121, inplace=True)
        input_121 = None
        input_123 = torch.conv2d(
            input_122,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            24,
        )
        input_122 = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_124 = torch.nn.functional.batch_norm(
            input_123,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_123 = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_125 = torch.nn.functional.relu(input_124, inplace=True)
        input_124 = None
        scale_55 = torch.nn.functional.adaptive_avg_pool2d(input_125, 1)
        scale_56 = torch.conv2d(
            scale_55,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_55 = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_se_modules_fc1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_se_modules_fc1_parameters_bias_ = (None)
        scale_57 = torch.nn.functional.relu(scale_56, inplace=False)
        scale_56 = None
        scale_58 = torch.conv2d(
            scale_57,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_57 = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_se_modules_fc2_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_se_modules_fc2_parameters_bias_ = (None)
        scale_59 = torch.sigmoid(scale_58)
        scale_58 = None
        input_126 = scale_59 * input_125
        scale_59 = input_125 = None
        input_127 = torch.conv2d(
            input_126,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_126 = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_128 = torch.nn.functional.batch_norm(
            input_127,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_127 = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_4_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_11 = input_119 + input_128
        input_119 = input_128 = None
        input_129 = torch.nn.functional.relu(x_11, inplace=True)
        x_11 = None
        input_130 = torch.conv2d(
            input_129,
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
        input_131 = torch.nn.functional.batch_norm(
            input_130,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_130 = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_132 = torch.nn.functional.relu(input_131, inplace=True)
        input_131 = None
        input_133 = torch.conv2d(
            input_132,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            24,
        )
        input_132 = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_134 = torch.nn.functional.batch_norm(
            input_133,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_133 = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_135 = torch.nn.functional.relu(input_134, inplace=True)
        input_134 = None
        scale_60 = torch.nn.functional.adaptive_avg_pool2d(input_135, 1)
        scale_61 = torch.conv2d(
            scale_60,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_60 = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_se_modules_fc1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_se_modules_fc1_parameters_bias_ = (None)
        scale_62 = torch.nn.functional.relu(scale_61, inplace=False)
        scale_61 = None
        scale_63 = torch.conv2d(
            scale_62,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_62 = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_se_modules_fc2_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_se_modules_fc2_parameters_bias_ = (None)
        scale_64 = torch.sigmoid(scale_63)
        scale_63 = None
        input_136 = scale_64 * input_135
        scale_64 = input_135 = None
        input_137 = torch.conv2d(
            input_136,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_136 = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_138 = torch.nn.functional.batch_norm(
            input_137,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_137 = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_5_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_12 = input_129 + input_138
        input_129 = input_138 = None
        input_139 = torch.nn.functional.relu(x_12, inplace=True)
        x_12 = None
        input_140 = torch.conv2d(
            input_139,
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
        input_141 = torch.nn.functional.batch_norm(
            input_140,
            l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_140 = l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_142 = torch.nn.functional.relu(input_141, inplace=True)
        input_141 = None
        input_143 = torch.conv2d(
            input_142,
            l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            24,
        )
        input_142 = l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_144 = torch.nn.functional.batch_norm(
            input_143,
            l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_143 = l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_145 = torch.nn.functional.relu(input_144, inplace=True)
        input_144 = None
        scale_65 = torch.nn.functional.adaptive_avg_pool2d(input_145, 1)
        scale_66 = torch.conv2d(
            scale_65,
            l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_65 = l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_se_modules_fc1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_se_modules_fc1_parameters_bias_ = (None)
        scale_67 = torch.nn.functional.relu(scale_66, inplace=False)
        scale_66 = None
        scale_68 = torch.conv2d(
            scale_67,
            l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_67 = l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_se_modules_fc2_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_se_modules_fc2_parameters_bias_ = (None)
        scale_69 = torch.sigmoid(scale_68)
        scale_68 = None
        input_146 = scale_69 * input_145
        scale_69 = input_145 = None
        input_147 = torch.conv2d(
            input_146,
            l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_146 = l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_148 = torch.nn.functional.batch_norm(
            input_147,
            l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_147 = l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_6_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_13 = input_139 + input_148
        input_139 = input_148 = None
        input_149 = torch.nn.functional.relu(x_13, inplace=True)
        x_13 = None
        input_150 = torch.conv2d(
            input_149,
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
        input_151 = torch.nn.functional.batch_norm(
            input_150,
            l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_150 = l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_152 = torch.nn.functional.relu(input_151, inplace=True)
        input_151 = None
        input_153 = torch.conv2d(
            input_152,
            l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            24,
        )
        input_152 = l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_154 = torch.nn.functional.batch_norm(
            input_153,
            l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_153 = l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_155 = torch.nn.functional.relu(input_154, inplace=True)
        input_154 = None
        scale_70 = torch.nn.functional.adaptive_avg_pool2d(input_155, 1)
        scale_71 = torch.conv2d(
            scale_70,
            l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_70 = l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_se_modules_fc1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_se_modules_fc1_parameters_bias_ = (None)
        scale_72 = torch.nn.functional.relu(scale_71, inplace=False)
        scale_71 = None
        scale_73 = torch.conv2d(
            scale_72,
            l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_72 = l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_se_modules_fc2_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_se_modules_fc2_parameters_bias_ = (None)
        scale_74 = torch.sigmoid(scale_73)
        scale_73 = None
        input_156 = scale_74 * input_155
        scale_74 = input_155 = None
        input_157 = torch.conv2d(
            input_156,
            l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_156 = l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_158 = torch.nn.functional.batch_norm(
            input_157,
            l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_157 = l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_7_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_14 = input_149 + input_158
        input_149 = input_158 = None
        input_159 = torch.nn.functional.relu(x_14, inplace=True)
        x_14 = None
        input_160 = torch.conv2d(
            input_159,
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
        input_161 = torch.nn.functional.batch_norm(
            input_160,
            l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_160 = l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_162 = torch.nn.functional.relu(input_161, inplace=True)
        input_161 = None
        input_163 = torch.conv2d(
            input_162,
            l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            24,
        )
        input_162 = l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_164 = torch.nn.functional.batch_norm(
            input_163,
            l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_163 = l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_165 = torch.nn.functional.relu(input_164, inplace=True)
        input_164 = None
        scale_75 = torch.nn.functional.adaptive_avg_pool2d(input_165, 1)
        scale_76 = torch.conv2d(
            scale_75,
            l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_75 = l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_se_modules_fc1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_se_modules_fc1_parameters_bias_ = (None)
        scale_77 = torch.nn.functional.relu(scale_76, inplace=False)
        scale_76 = None
        scale_78 = torch.conv2d(
            scale_77,
            l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_77 = l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_se_modules_fc2_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_se_modules_fc2_parameters_bias_ = (None)
        scale_79 = torch.sigmoid(scale_78)
        scale_78 = None
        input_166 = scale_79 * input_165
        scale_79 = input_165 = None
        input_167 = torch.conv2d(
            input_166,
            l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_166 = l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_168 = torch.nn.functional.batch_norm(
            input_167,
            l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_167 = l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_8_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_15 = input_159 + input_168
        input_159 = input_168 = None
        input_169 = torch.nn.functional.relu(x_15, inplace=True)
        x_15 = None
        input_170 = torch.conv2d(
            input_169,
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
        input_171 = torch.nn.functional.batch_norm(
            input_170,
            l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_170 = l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_172 = torch.nn.functional.relu(input_171, inplace=True)
        input_171 = None
        input_173 = torch.conv2d(
            input_172,
            l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            24,
        )
        input_172 = l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_174 = torch.nn.functional.batch_norm(
            input_173,
            l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_173 = l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_175 = torch.nn.functional.relu(input_174, inplace=True)
        input_174 = None
        scale_80 = torch.nn.functional.adaptive_avg_pool2d(input_175, 1)
        scale_81 = torch.conv2d(
            scale_80,
            l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_80 = l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_se_modules_fc1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_se_modules_fc1_parameters_bias_ = (None)
        scale_82 = torch.nn.functional.relu(scale_81, inplace=False)
        scale_81 = None
        scale_83 = torch.conv2d(
            scale_82,
            l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_82 = l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_se_modules_fc2_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_se_modules_fc2_parameters_bias_ = (None)
        scale_84 = torch.sigmoid(scale_83)
        scale_83 = None
        input_176 = scale_84 * input_175
        scale_84 = input_175 = None
        input_177 = torch.conv2d(
            input_176,
            l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_176 = l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_178 = torch.nn.functional.batch_norm(
            input_177,
            l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_177 = l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_9_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_16 = input_169 + input_178
        input_169 = input_178 = None
        input_179 = torch.nn.functional.relu(x_16, inplace=True)
        x_16 = None
        input_180 = torch.conv2d(
            input_179,
            l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_a_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_a_modules_0_parameters_weight_ = (
            None
        )
        input_181 = torch.nn.functional.batch_norm(
            input_180,
            l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_180 = l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_182 = torch.nn.functional.relu(input_181, inplace=True)
        input_181 = None
        input_183 = torch.conv2d(
            input_182,
            l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            24,
        )
        input_182 = l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_184 = torch.nn.functional.batch_norm(
            input_183,
            l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_183 = l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_185 = torch.nn.functional.relu(input_184, inplace=True)
        input_184 = None
        scale_85 = torch.nn.functional.adaptive_avg_pool2d(input_185, 1)
        scale_86 = torch.conv2d(
            scale_85,
            l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_85 = l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_se_modules_fc1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_se_modules_fc1_parameters_bias_ = (None)
        scale_87 = torch.nn.functional.relu(scale_86, inplace=False)
        scale_86 = None
        scale_88 = torch.conv2d(
            scale_87,
            l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_87 = l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_se_modules_fc2_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_se_modules_fc2_parameters_bias_ = (None)
        scale_89 = torch.sigmoid(scale_88)
        scale_88 = None
        input_186 = scale_89 * input_185
        scale_89 = input_185 = None
        input_187 = torch.conv2d(
            input_186,
            l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_186 = l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_188 = torch.nn.functional.batch_norm(
            input_187,
            l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_187 = l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_10_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_17 = input_179 + input_188
        input_179 = input_188 = None
        input_189 = torch.nn.functional.relu(x_17, inplace=True)
        x_17 = None
        input_190 = torch.conv2d(
            input_189,
            l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_a_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_a_modules_0_parameters_weight_ = (
            None
        )
        input_191 = torch.nn.functional.batch_norm(
            input_190,
            l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_190 = l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_192 = torch.nn.functional.relu(input_191, inplace=True)
        input_191 = None
        input_193 = torch.conv2d(
            input_192,
            l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            24,
        )
        input_192 = l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_194 = torch.nn.functional.batch_norm(
            input_193,
            l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_193 = l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_195 = torch.nn.functional.relu(input_194, inplace=True)
        input_194 = None
        scale_90 = torch.nn.functional.adaptive_avg_pool2d(input_195, 1)
        scale_91 = torch.conv2d(
            scale_90,
            l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_90 = l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_se_modules_fc1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_se_modules_fc1_parameters_bias_ = (None)
        scale_92 = torch.nn.functional.relu(scale_91, inplace=False)
        scale_91 = None
        scale_93 = torch.conv2d(
            scale_92,
            l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_92 = l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_se_modules_fc2_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_se_modules_fc2_parameters_bias_ = (None)
        scale_94 = torch.sigmoid(scale_93)
        scale_93 = None
        input_196 = scale_94 * input_195
        scale_94 = input_195 = None
        input_197 = torch.conv2d(
            input_196,
            l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_196 = l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_198 = torch.nn.functional.batch_norm(
            input_197,
            l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_197 = l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_11_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_18 = input_189 + input_198
        input_189 = input_198 = None
        input_199 = torch.nn.functional.relu(x_18, inplace=True)
        x_18 = None
        input_200 = torch.conv2d(
            input_199,
            l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_a_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_a_modules_0_parameters_weight_ = (
            None
        )
        input_201 = torch.nn.functional.batch_norm(
            input_200,
            l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_200 = l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_202 = torch.nn.functional.relu(input_201, inplace=True)
        input_201 = None
        input_203 = torch.conv2d(
            input_202,
            l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            24,
        )
        input_202 = l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_204 = torch.nn.functional.batch_norm(
            input_203,
            l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_203 = l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_205 = torch.nn.functional.relu(input_204, inplace=True)
        input_204 = None
        scale_95 = torch.nn.functional.adaptive_avg_pool2d(input_205, 1)
        scale_96 = torch.conv2d(
            scale_95,
            l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_95 = l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_se_modules_fc1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_se_modules_fc1_parameters_bias_ = (None)
        scale_97 = torch.nn.functional.relu(scale_96, inplace=False)
        scale_96 = None
        scale_98 = torch.conv2d(
            scale_97,
            l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_97 = l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_se_modules_fc2_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_se_modules_fc2_parameters_bias_ = (None)
        scale_99 = torch.sigmoid(scale_98)
        scale_98 = None
        input_206 = scale_99 * input_205
        scale_99 = input_205 = None
        input_207 = torch.conv2d(
            input_206,
            l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_206 = l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_208 = torch.nn.functional.batch_norm(
            input_207,
            l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_207 = l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block3_modules_block3_12_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_19 = input_199 + input_208
        input_199 = input_208 = None
        input_209 = torch.nn.functional.relu(x_19, inplace=True)
        x_19 = None
        input_210 = torch.conv2d(
            input_209,
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
        input_211 = torch.nn.functional.batch_norm(
            input_210,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_210 = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_proj_modules_1_parameters_bias_ = (None)
        input_212 = torch.conv2d(
            input_209,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_209 = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_0_parameters_weight_ = (None)
        input_213 = torch.nn.functional.batch_norm(
            input_212,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_212 = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_a_modules_1_parameters_bias_ = (None)
        input_214 = torch.nn.functional.relu(input_213, inplace=True)
        input_213 = None
        input_215 = torch.conv2d(
            input_214,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            63,
        )
        input_214 = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_0_parameters_weight_ = (None)
        input_216 = torch.nn.functional.batch_norm(
            input_215,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_215 = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_b_modules_1_parameters_bias_ = (None)
        input_217 = torch.nn.functional.relu(input_216, inplace=True)
        input_216 = None
        scale_100 = torch.nn.functional.adaptive_avg_pool2d(input_217, 1)
        scale_101 = torch.conv2d(
            scale_100,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_100 = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_se_modules_fc1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_se_modules_fc1_parameters_bias_ = (None)
        scale_102 = torch.nn.functional.relu(scale_101, inplace=False)
        scale_101 = None
        scale_103 = torch.conv2d(
            scale_102,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_102 = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_se_modules_fc2_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_se_modules_fc2_parameters_bias_ = (None)
        scale_104 = torch.sigmoid(scale_103)
        scale_103 = None
        input_218 = scale_104 * input_217
        scale_104 = input_217 = None
        input_219 = torch.conv2d(
            input_218,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_218 = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_0_parameters_weight_ = (None)
        input_220 = torch.nn.functional.batch_norm(
            input_219,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_1_buffers_running_mean_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_1_buffers_running_var_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_1_parameters_weight_,
            l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_219 = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_1_buffers_running_mean_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_1_buffers_running_var_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_1_parameters_weight_ = l_self_modules_trunk_output_modules_block4_modules_block4_0_modules_f_modules_c_modules_1_parameters_bias_ = (None)
        x_20 = input_211 + input_220
        input_211 = input_220 = None
        input_221 = torch.nn.functional.relu(x_20, inplace=True)
        x_20 = None
        x_21 = torch.nn.functional.adaptive_avg_pool2d(input_221, (1, 1))
        input_221 = None
        x_22 = x_21.flatten(start_dim=1)
        x_21 = None
        x_23 = torch._C._nn.linear(
            x_22,
            l_self_modules_fc_parameters_weight_,
            l_self_modules_fc_parameters_bias_,
        )
        x_22 = (
            l_self_modules_fc_parameters_weight_
        ) = l_self_modules_fc_parameters_bias_ = None
        return (x_23,)
