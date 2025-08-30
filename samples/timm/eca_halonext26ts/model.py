import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_stem_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_x_: torch.Tensor,
        L_self_modules_stem_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stem_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stem_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stem_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stem_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stem_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stem_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_self_attn_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_self_attn_modules_kv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_self_attn_modules_pos_embed_parameters_width_rel_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_self_attn_modules_pos_embed_parameters_height_rel_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_post_attn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_post_attn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_post_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_post_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_self_attn_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_self_attn_modules_kv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_self_attn_modules_pos_embed_parameters_width_rel_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_self_attn_modules_pos_embed_parameters_height_rel_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_post_attn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_post_attn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_post_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_post_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_self_attn_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_self_attn_modules_kv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_self_attn_modules_pos_embed_parameters_width_rel_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_self_attn_modules_pos_embed_parameters_height_rel_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_post_attn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_1_modules_post_attn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_1_modules_post_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_post_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_stem_modules_conv1_modules_conv_parameters_weight_ = (
            L_self_modules_stem_modules_conv1_modules_conv_parameters_weight_
        )
        l_x_ = L_x_
        l_self_modules_stem_modules_conv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_stem_modules_conv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_stem_modules_conv1_modules_bn_buffers_running_var_ = (
            L_self_modules_stem_modules_conv1_modules_bn_buffers_running_var_
        )
        l_self_modules_stem_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_stem_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_stem_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_stem_modules_conv1_modules_bn_parameters_bias_
        )
        l_self_modules_stem_modules_conv2_modules_conv_parameters_weight_ = (
            L_self_modules_stem_modules_conv2_modules_conv_parameters_weight_
        )
        l_self_modules_stem_modules_conv2_modules_bn_buffers_running_mean_ = (
            L_self_modules_stem_modules_conv2_modules_bn_buffers_running_mean_
        )
        l_self_modules_stem_modules_conv2_modules_bn_buffers_running_var_ = (
            L_self_modules_stem_modules_conv2_modules_bn_buffers_running_var_
        )
        l_self_modules_stem_modules_conv2_modules_bn_parameters_weight_ = (
            L_self_modules_stem_modules_conv2_modules_bn_parameters_weight_
        )
        l_self_modules_stem_modules_conv2_modules_bn_parameters_bias_ = (
            L_self_modules_stem_modules_conv2_modules_bn_parameters_bias_
        )
        l_self_modules_stem_modules_conv3_modules_conv_parameters_weight_ = (
            L_self_modules_stem_modules_conv3_modules_conv_parameters_weight_
        )
        l_self_modules_stem_modules_conv3_modules_bn_buffers_running_mean_ = (
            L_self_modules_stem_modules_conv3_modules_bn_buffers_running_mean_
        )
        l_self_modules_stem_modules_conv3_modules_bn_buffers_running_var_ = (
            L_self_modules_stem_modules_conv3_modules_bn_buffers_running_var_
        )
        l_self_modules_stem_modules_conv3_modules_bn_parameters_weight_ = (
            L_self_modules_stem_modules_conv3_modules_bn_parameters_weight_
        )
        l_self_modules_stem_modules_conv3_modules_bn_parameters_bias_ = (
            L_self_modules_stem_modules_conv3_modules_bn_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_1_modules_self_attn_modules_q_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_self_attn_modules_q_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_self_attn_modules_kv_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_self_attn_modules_kv_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_self_attn_modules_pos_embed_parameters_width_rel_ = L_self_modules_stages_modules_2_modules_1_modules_self_attn_modules_pos_embed_parameters_width_rel_
        l_self_modules_stages_modules_2_modules_1_modules_self_attn_modules_pos_embed_parameters_height_rel_ = L_self_modules_stages_modules_2_modules_1_modules_self_attn_modules_pos_embed_parameters_height_rel_
        l_self_modules_stages_modules_2_modules_1_modules_post_attn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_1_modules_post_attn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_1_modules_post_attn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_1_modules_post_attn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_1_modules_post_attn_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_post_attn_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_post_attn_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_1_modules_post_attn_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_0_modules_self_attn_modules_q_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_self_attn_modules_q_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_self_attn_modules_kv_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_self_attn_modules_kv_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_self_attn_modules_pos_embed_parameters_width_rel_ = L_self_modules_stages_modules_3_modules_0_modules_self_attn_modules_pos_embed_parameters_width_rel_
        l_self_modules_stages_modules_3_modules_0_modules_self_attn_modules_pos_embed_parameters_height_rel_ = L_self_modules_stages_modules_3_modules_0_modules_self_attn_modules_pos_embed_parameters_height_rel_
        l_self_modules_stages_modules_3_modules_0_modules_post_attn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_0_modules_post_attn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_0_modules_post_attn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_0_modules_post_attn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_0_modules_post_attn_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_post_attn_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_post_attn_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_0_modules_post_attn_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_1_modules_self_attn_modules_q_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_self_attn_modules_q_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_self_attn_modules_kv_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_self_attn_modules_kv_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_self_attn_modules_pos_embed_parameters_width_rel_ = L_self_modules_stages_modules_3_modules_1_modules_self_attn_modules_pos_embed_parameters_width_rel_
        l_self_modules_stages_modules_3_modules_1_modules_self_attn_modules_pos_embed_parameters_height_rel_ = L_self_modules_stages_modules_3_modules_1_modules_self_attn_modules_pos_embed_parameters_height_rel_
        l_self_modules_stages_modules_3_modules_1_modules_post_attn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_1_modules_post_attn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_1_modules_post_attn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_1_modules_post_attn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_1_modules_post_attn_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_post_attn_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_post_attn_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_1_modules_post_attn_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_head_modules_fc_parameters_weight_ = (
            L_self_modules_head_modules_fc_parameters_weight_
        )
        l_self_modules_head_modules_fc_parameters_bias_ = (
            L_self_modules_head_modules_fc_parameters_bias_
        )
        x = torch.conv2d(
            l_x_,
            l_self_modules_stem_modules_conv1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_x_ = l_self_modules_stem_modules_conv1_modules_conv_parameters_weight_ = None
        x_1 = torch.nn.functional.batch_norm(
            x,
            l_self_modules_stem_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stem_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stem_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stem_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x = (
            l_self_modules_stem_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_stem_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_stem_modules_conv1_modules_bn_parameters_weight_
        ) = l_self_modules_stem_modules_conv1_modules_bn_parameters_bias_ = None
        x_2 = torch.nn.functional.silu(x_1, inplace=True)
        x_1 = None
        x_3 = torch.conv2d(
            x_2,
            l_self_modules_stem_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_2 = l_self_modules_stem_modules_conv2_modules_conv_parameters_weight_ = None
        x_4 = torch.nn.functional.batch_norm(
            x_3,
            l_self_modules_stem_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stem_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stem_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stem_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_3 = (
            l_self_modules_stem_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_stem_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_stem_modules_conv2_modules_bn_parameters_weight_
        ) = l_self_modules_stem_modules_conv2_modules_bn_parameters_bias_ = None
        x_5 = torch.nn.functional.silu(x_4, inplace=True)
        x_4 = None
        x_6 = torch.conv2d(
            x_5,
            l_self_modules_stem_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_5 = l_self_modules_stem_modules_conv3_modules_conv_parameters_weight_ = None
        x_7 = torch.nn.functional.batch_norm(
            x_6,
            l_self_modules_stem_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_stem_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_stem_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_stem_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_6 = (
            l_self_modules_stem_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_stem_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_stem_modules_conv3_modules_bn_parameters_weight_
        ) = l_self_modules_stem_modules_conv3_modules_bn_parameters_bias_ = None
        x_8 = torch.nn.functional.silu(x_7, inplace=True)
        x_7 = None
        input_1 = torch.nn.functional.max_pool2d(
            x_8, 3, 2, 1, 1, ceil_mode=False, return_indices=False
        )
        x_8 = None
        x_9 = torch.conv2d(
            input_1,
            l_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_10 = torch.nn.functional.batch_norm(
            x_9,
            l_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_9 = l_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_11 = torch.nn.functional.silu(x_10, inplace=True)
        x_10 = None
        x_12 = torch.conv2d(
            x_11,
            l_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            4,
        )
        x_11 = l_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_13 = torch.nn.functional.batch_norm(
            x_12,
            l_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_12 = l_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_14 = torch.nn.functional.silu(x_13, inplace=True)
        x_13 = None
        mean = x_14.mean((2, 3))
        y = mean.view(1, 1, -1)
        mean = None
        y_1 = torch.conv1d(
            y,
            l_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv_parameters_weight_,
            None,
            (1,),
            (1,),
            (1,),
            1,
        )
        y = l_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv_parameters_weight_ = (None)
        sigmoid = y_1.sigmoid()
        y_1 = None
        y_2 = sigmoid.view(1, -1, 1, 1)
        sigmoid = None
        expand_as = y_2.expand_as(x_14)
        y_2 = None
        x_15 = x_14 * expand_as
        x_14 = expand_as = None
        x_16 = torch.conv2d(
            x_15,
            l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_15 = l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_17 = torch.nn.functional.batch_norm(
            x_16,
            l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_16 = l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_18 = torch.conv2d(
            input_1,
            l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_1 = l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_conv_parameters_weight_ = (None)
        x_19 = torch.nn.functional.batch_norm(
            x_18,
            l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_18 = l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_parameters_bias_ = (None)
        x_20 = x_17 + x_19
        x_17 = x_19 = None
        input_2 = torch.nn.functional.silu(x_20, inplace=True)
        x_20 = None
        x_21 = torch.conv2d(
            input_2,
            l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_22 = torch.nn.functional.batch_norm(
            x_21,
            l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_21 = l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_23 = torch.nn.functional.silu(x_22, inplace=True)
        x_22 = None
        x_24 = torch.conv2d(
            x_23,
            l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            4,
        )
        x_23 = l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_25 = torch.nn.functional.batch_norm(
            x_24,
            l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_24 = l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_26 = torch.nn.functional.silu(x_25, inplace=True)
        x_25 = None
        mean_1 = x_26.mean((2, 3))
        y_3 = mean_1.view(1, 1, -1)
        mean_1 = None
        y_4 = torch.conv1d(
            y_3,
            l_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv_parameters_weight_,
            None,
            (1,),
            (1,),
            (1,),
            1,
        )
        y_3 = l_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv_parameters_weight_ = (None)
        sigmoid_1 = y_4.sigmoid()
        y_4 = None
        y_5 = sigmoid_1.view(1, -1, 1, 1)
        sigmoid_1 = None
        expand_as_1 = y_5.expand_as(x_26)
        y_5 = None
        x_27 = x_26 * expand_as_1
        x_26 = expand_as_1 = None
        x_28 = torch.conv2d(
            x_27,
            l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_27 = l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_29 = torch.nn.functional.batch_norm(
            x_28,
            l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_28 = l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_30 = x_29 + input_2
        x_29 = input_2 = None
        input_3 = torch.nn.functional.silu(x_30, inplace=True)
        x_30 = None
        x_31 = torch.conv2d(
            input_3,
            l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_32 = torch.nn.functional.batch_norm(
            x_31,
            l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_31 = l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_33 = torch.nn.functional.silu(x_32, inplace=True)
        x_32 = None
        x_34 = torch.conv2d(
            x_33,
            l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            8,
        )
        x_33 = l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_35 = torch.nn.functional.batch_norm(
            x_34,
            l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_34 = l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_36 = torch.nn.functional.silu(x_35, inplace=True)
        x_35 = None
        mean_2 = x_36.mean((2, 3))
        y_6 = mean_2.view(1, 1, -1)
        mean_2 = None
        y_7 = torch.conv1d(
            y_6,
            l_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_6 = l_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv_parameters_weight_ = (None)
        sigmoid_2 = y_7.sigmoid()
        y_7 = None
        y_8 = sigmoid_2.view(1, -1, 1, 1)
        sigmoid_2 = None
        expand_as_2 = y_8.expand_as(x_36)
        y_8 = None
        x_37 = x_36 * expand_as_2
        x_36 = expand_as_2 = None
        x_38 = torch.conv2d(
            x_37,
            l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_37 = l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_39 = torch.nn.functional.batch_norm(
            x_38,
            l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_38 = l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_40 = torch.conv2d(
            input_3,
            l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        input_3 = l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_conv_parameters_weight_ = (None)
        x_41 = torch.nn.functional.batch_norm(
            x_40,
            l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_40 = l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_parameters_bias_ = (None)
        x_42 = x_39 + x_41
        x_39 = x_41 = None
        input_4 = torch.nn.functional.silu(x_42, inplace=True)
        x_42 = None
        x_43 = torch.conv2d(
            input_4,
            l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_44 = torch.nn.functional.batch_norm(
            x_43,
            l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_43 = l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_45 = torch.nn.functional.silu(x_44, inplace=True)
        x_44 = None
        x_46 = torch.conv2d(
            x_45,
            l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        x_45 = l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_47 = torch.nn.functional.batch_norm(
            x_46,
            l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_46 = l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_48 = torch.nn.functional.silu(x_47, inplace=True)
        x_47 = None
        mean_3 = x_48.mean((2, 3))
        y_9 = mean_3.view(1, 1, -1)
        mean_3 = None
        y_10 = torch.conv1d(
            y_9,
            l_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_9 = l_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv_parameters_weight_ = (None)
        sigmoid_3 = y_10.sigmoid()
        y_10 = None
        y_11 = sigmoid_3.view(1, -1, 1, 1)
        sigmoid_3 = None
        expand_as_3 = y_11.expand_as(x_48)
        y_11 = None
        x_49 = x_48 * expand_as_3
        x_48 = expand_as_3 = None
        x_50 = torch.conv2d(
            x_49,
            l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_49 = l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_51 = torch.nn.functional.batch_norm(
            x_50,
            l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_50 = l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_52 = x_51 + input_4
        x_51 = input_4 = None
        input_5 = torch.nn.functional.silu(x_52, inplace=True)
        x_52 = None
        x_53 = torch.conv2d(
            input_5,
            l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_54 = torch.nn.functional.batch_norm(
            x_53,
            l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_53 = l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_55 = torch.nn.functional.silu(x_54, inplace=True)
        x_54 = None
        x_56 = torch.conv2d(
            x_55,
            l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            16,
        )
        x_55 = l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_57 = torch.nn.functional.batch_norm(
            x_56,
            l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_56 = l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_58 = torch.nn.functional.silu(x_57, inplace=True)
        x_57 = None
        mean_4 = x_58.mean((2, 3))
        y_12 = mean_4.view(1, 1, -1)
        mean_4 = None
        y_13 = torch.conv1d(
            y_12,
            l_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_12 = l_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv_parameters_weight_ = (None)
        sigmoid_4 = y_13.sigmoid()
        y_13 = None
        y_14 = sigmoid_4.view(1, -1, 1, 1)
        sigmoid_4 = None
        expand_as_4 = y_14.expand_as(x_58)
        y_14 = None
        x_59 = x_58 * expand_as_4
        x_58 = expand_as_4 = None
        x_60 = torch.conv2d(
            x_59,
            l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_59 = l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_61 = torch.nn.functional.batch_norm(
            x_60,
            l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_60 = l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_62 = torch.conv2d(
            input_5,
            l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        input_5 = l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_parameters_weight_ = (None)
        x_63 = torch.nn.functional.batch_norm(
            x_62,
            l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_62 = l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_parameters_bias_ = (None)
        x_64 = x_61 + x_63
        x_61 = x_63 = None
        input_6 = torch.nn.functional.silu(x_64, inplace=True)
        x_64 = None
        x_65 = torch.conv2d(
            input_6,
            l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_66 = torch.nn.functional.batch_norm(
            x_65,
            l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_65 = l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_67 = torch.nn.functional.silu(x_66, inplace=True)
        x_66 = None
        q = torch.conv2d(
            x_67,
            l_self_modules_stages_modules_2_modules_1_modules_self_attn_modules_q_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_1_modules_self_attn_modules_q_parameters_weight_ = (
            None
        )
        reshape = q.reshape(-1, 16, 2, 8, 2, 8)
        q = None
        q_1 = reshape.permute(0, 1, 3, 5, 2, 4)
        reshape = None
        reshape_1 = q_1.reshape(8, 16, -1, 4)
        q_1 = None
        q_2 = reshape_1.transpose(1, 3)
        reshape_1 = None
        kv = torch.conv2d(
            x_67,
            l_self_modules_stages_modules_2_modules_1_modules_self_attn_modules_kv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_67 = l_self_modules_stages_modules_2_modules_1_modules_self_attn_modules_kv_parameters_weight_ = (None)
        kv_1 = torch._C._nn.pad(kv, [2, 2, 2, 2], "constant", None)
        kv = None
        unfold = kv_1.unfold(2, 12, 8)
        kv_1 = None
        unfold_1 = unfold.unfold(3, 12, 8)
        unfold = None
        reshape_2 = unfold_1.reshape(8, 48, 4, -1)
        unfold_1 = None
        kv_2 = reshape_2.permute(0, 2, 3, 1)
        reshape_2 = None
        split = torch.functional.split(kv_2, [16, 32], dim=-1)
        kv_2 = None
        k = split[0]
        v = split[1]
        split = None
        transpose_1 = k.transpose(-1, -2)
        k = None
        matmul = q_2 @ transpose_1
        transpose_1 = None
        mul_5 = matmul * 0.25
        matmul = None
        q_3 = q_2.reshape(-1, 8, 8, 16)
        q_2 = None
        transpose_2 = l_self_modules_stages_modules_2_modules_1_modules_self_attn_modules_pos_embed_parameters_width_rel_.transpose(
            -1, -2
        )
        l_self_modules_stages_modules_2_modules_1_modules_self_attn_modules_pos_embed_parameters_width_rel_ = (
            None
        )
        x_68 = q_3 @ transpose_2
        transpose_2 = None
        x_69 = x_68.reshape(-1, 8, 23)
        x_68 = None
        pad_1 = torch._C._nn.pad(x_69, [0, 1], "constant", None)
        x_69 = None
        x_pad = pad_1.flatten(1)
        pad_1 = None
        x_pad_1 = torch._C._nn.pad(x_pad, [0, 15], "constant", None)
        x_pad = None
        x_pad_2 = x_pad_1.reshape(-1, 9, 23)
        x_pad_1 = None
        x_70 = x_pad_2[
            (slice(None, None, None), slice(None, 8, None), slice(11, None, None))
        ]
        x_pad_2 = None
        reshape_6 = x_70.reshape(32, 8, 1, 8, 12)
        x_70 = None
        x_71 = reshape_6.expand(-1, -1, 12, -1, -1)
        reshape_6 = None
        rel_logits_w = x_71.permute((0, 1, 3, 2, 4))
        x_71 = None
        q_4 = q_3.transpose(1, 2)
        q_3 = None
        transpose_4 = l_self_modules_stages_modules_2_modules_1_modules_self_attn_modules_pos_embed_parameters_height_rel_.transpose(
            -1, -2
        )
        l_self_modules_stages_modules_2_modules_1_modules_self_attn_modules_pos_embed_parameters_height_rel_ = (
            None
        )
        x_72 = q_4 @ transpose_4
        q_4 = transpose_4 = None
        x_73 = x_72.reshape(-1, 8, 23)
        x_72 = None
        pad_3 = torch._C._nn.pad(x_73, [0, 1], "constant", None)
        x_73 = None
        x_pad_3 = pad_3.flatten(1)
        pad_3 = None
        x_pad_4 = torch._C._nn.pad(x_pad_3, [0, 15], "constant", None)
        x_pad_3 = None
        x_pad_5 = x_pad_4.reshape(-1, 9, 23)
        x_pad_4 = None
        x_74 = x_pad_5[
            (slice(None, None, None), slice(None, 8, None), slice(11, None, None))
        ]
        x_pad_5 = None
        reshape_9 = x_74.reshape(32, 8, 1, 8, 12)
        x_74 = None
        x_75 = reshape_9.expand(-1, -1, 12, -1, -1)
        reshape_9 = None
        rel_logits_h = x_75.permute((0, 3, 1, 4, 2))
        x_75 = None
        rel_logits = rel_logits_h + rel_logits_w
        rel_logits_h = rel_logits_w = None
        rel_logits_1 = rel_logits.reshape(8, 4, 64, -1)
        rel_logits = None
        attn = mul_5 + rel_logits_1
        mul_5 = rel_logits_1 = None
        attn_1 = attn.softmax(dim=-1)
        attn = None
        matmul_3 = attn_1 @ v
        attn_1 = v = None
        out = matmul_3.transpose(1, 3)
        matmul_3 = None
        out_1 = out.reshape(-1, 8, 8, 2, 2)
        out = None
        permute_4 = out_1.permute(0, 3, 1, 4, 2)
        out_1 = None
        contiguous = permute_4.contiguous()
        permute_4 = None
        out_2 = contiguous.view(1, 256, 16, 16)
        contiguous = None
        x_76 = torch.nn.functional.batch_norm(
            out_2,
            l_self_modules_stages_modules_2_modules_1_modules_post_attn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_1_modules_post_attn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_1_modules_post_attn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_1_modules_post_attn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_2 = l_self_modules_stages_modules_2_modules_1_modules_post_attn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_1_modules_post_attn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_1_modules_post_attn_parameters_weight_ = (
            l_self_modules_stages_modules_2_modules_1_modules_post_attn_parameters_bias_
        ) = None
        x_77 = torch.nn.functional.silu(x_76, inplace=True)
        x_76 = None
        x_78 = torch.conv2d(
            x_77,
            l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_77 = l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_79 = torch.nn.functional.batch_norm(
            x_78,
            l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_78 = l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_80 = x_79 + input_6
        x_79 = input_6 = None
        input_7 = torch.nn.functional.silu(x_80, inplace=True)
        x_80 = None
        x_81 = torch.conv2d(
            input_7,
            l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_82 = torch.nn.functional.batch_norm(
            x_81,
            l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_81 = l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_83 = torch.nn.functional.silu(x_82, inplace=True)
        x_82 = None
        q_5 = torch.conv2d(
            x_83,
            l_self_modules_stages_modules_3_modules_0_modules_self_attn_modules_q_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_0_modules_self_attn_modules_q_parameters_weight_ = (
            None
        )
        reshape_12 = q_5.reshape(-1, 16, 2, 4, 2, 4)
        q_5 = None
        q_6 = reshape_12.permute(0, 1, 3, 5, 2, 4)
        reshape_12 = None
        reshape_13 = q_6.reshape(8, 16, -1, 4)
        q_6 = None
        q_7 = reshape_13.transpose(1, 3)
        reshape_13 = None
        kv_3 = torch.conv2d(
            x_83,
            l_self_modules_stages_modules_3_modules_0_modules_self_attn_modules_kv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_83 = l_self_modules_stages_modules_3_modules_0_modules_self_attn_modules_kv_parameters_weight_ = (None)
        kv_4 = torch._C._nn.pad(kv_3, [2, 2, 2, 2], "constant", None)
        kv_3 = None
        unfold_2 = kv_4.unfold(2, 12, 8)
        kv_4 = None
        unfold_3 = unfold_2.unfold(3, 12, 8)
        unfold_2 = None
        reshape_14 = unfold_3.reshape(8, 80, 4, -1)
        unfold_3 = None
        kv_5 = reshape_14.permute(0, 2, 3, 1)
        reshape_14 = None
        split_1 = torch.functional.split(kv_5, [16, 64], dim=-1)
        kv_5 = None
        k_1 = split_1[0]
        v_1 = split_1[1]
        split_1 = None
        transpose_7 = k_1.transpose(-1, -2)
        k_1 = None
        matmul_4 = q_7 @ transpose_7
        transpose_7 = None
        mul_6 = matmul_4 * 0.25
        matmul_4 = None
        q_8 = q_7.reshape(-1, 4, 4, 16)
        q_7 = None
        transpose_8 = l_self_modules_stages_modules_3_modules_0_modules_self_attn_modules_pos_embed_parameters_width_rel_.transpose(
            -1, -2
        )
        l_self_modules_stages_modules_3_modules_0_modules_self_attn_modules_pos_embed_parameters_width_rel_ = (
            None
        )
        x_84 = q_8 @ transpose_8
        transpose_8 = None
        x_85 = x_84.reshape(-1, 4, 23)
        x_84 = None
        pad_6 = torch._C._nn.pad(x_85, [0, 1], "constant", None)
        x_85 = None
        x_pad_6 = pad_6.flatten(1)
        pad_6 = None
        x_pad_7 = torch._C._nn.pad(x_pad_6, [0, 19], "constant", None)
        x_pad_6 = None
        x_pad_8 = x_pad_7.reshape(-1, 5, 23)
        x_pad_7 = None
        x_86 = x_pad_8[
            (slice(None, None, None), slice(None, 4, None), slice(11, None, None))
        ]
        x_pad_8 = None
        reshape_18 = x_86.reshape(32, 4, 1, 4, 12)
        x_86 = None
        x_87 = reshape_18.expand(-1, -1, 12, -1, -1)
        reshape_18 = None
        rel_logits_w_1 = x_87.permute((0, 1, 3, 2, 4))
        x_87 = None
        q_9 = q_8.transpose(1, 2)
        q_8 = None
        transpose_10 = l_self_modules_stages_modules_3_modules_0_modules_self_attn_modules_pos_embed_parameters_height_rel_.transpose(
            -1, -2
        )
        l_self_modules_stages_modules_3_modules_0_modules_self_attn_modules_pos_embed_parameters_height_rel_ = (
            None
        )
        x_88 = q_9 @ transpose_10
        q_9 = transpose_10 = None
        x_89 = x_88.reshape(-1, 4, 23)
        x_88 = None
        pad_8 = torch._C._nn.pad(x_89, [0, 1], "constant", None)
        x_89 = None
        x_pad_9 = pad_8.flatten(1)
        pad_8 = None
        x_pad_10 = torch._C._nn.pad(x_pad_9, [0, 19], "constant", None)
        x_pad_9 = None
        x_pad_11 = x_pad_10.reshape(-1, 5, 23)
        x_pad_10 = None
        x_90 = x_pad_11[
            (slice(None, None, None), slice(None, 4, None), slice(11, None, None))
        ]
        x_pad_11 = None
        reshape_21 = x_90.reshape(32, 4, 1, 4, 12)
        x_90 = None
        x_91 = reshape_21.expand(-1, -1, 12, -1, -1)
        reshape_21 = None
        rel_logits_h_1 = x_91.permute((0, 3, 1, 4, 2))
        x_91 = None
        rel_logits_2 = rel_logits_h_1 + rel_logits_w_1
        rel_logits_h_1 = rel_logits_w_1 = None
        rel_logits_3 = rel_logits_2.reshape(8, 4, 16, -1)
        rel_logits_2 = None
        attn_2 = mul_6 + rel_logits_3
        mul_6 = rel_logits_3 = None
        attn_3 = attn_2.softmax(dim=-1)
        attn_2 = None
        matmul_7 = attn_3 @ v_1
        attn_3 = v_1 = None
        out_3 = matmul_7.transpose(1, 3)
        matmul_7 = None
        out_4 = out_3.reshape(-1, 4, 4, 2, 2)
        out_3 = None
        permute_9 = out_4.permute(0, 3, 1, 4, 2)
        out_4 = None
        contiguous_1 = permute_9.contiguous()
        permute_9 = None
        out_5 = contiguous_1.view(1, 512, 8, 8)
        contiguous_1 = None
        x_92 = torch.nn.functional.batch_norm(
            out_5,
            l_self_modules_stages_modules_3_modules_0_modules_post_attn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_0_modules_post_attn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_0_modules_post_attn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_0_modules_post_attn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_5 = l_self_modules_stages_modules_3_modules_0_modules_post_attn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_0_modules_post_attn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_0_modules_post_attn_parameters_weight_ = (
            l_self_modules_stages_modules_3_modules_0_modules_post_attn_parameters_bias_
        ) = None
        x_93 = torch.nn.functional.silu(x_92, inplace=True)
        x_92 = None
        x_94 = torch.conv2d(
            x_93,
            l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_93 = l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_95 = torch.nn.functional.batch_norm(
            x_94,
            l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_94 = l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_96 = torch.conv2d(
            input_7,
            l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        input_7 = l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_parameters_weight_ = (None)
        x_97 = torch.nn.functional.batch_norm(
            x_96,
            l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_96 = l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_parameters_bias_ = (None)
        x_98 = x_95 + x_97
        x_95 = x_97 = None
        input_8 = torch.nn.functional.silu(x_98, inplace=True)
        x_98 = None
        x_99 = torch.conv2d(
            input_8,
            l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_100 = torch.nn.functional.batch_norm(
            x_99,
            l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_99 = l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_101 = torch.nn.functional.silu(x_100, inplace=True)
        x_100 = None
        q_10 = torch.conv2d(
            x_101,
            l_self_modules_stages_modules_3_modules_1_modules_self_attn_modules_q_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_1_modules_self_attn_modules_q_parameters_weight_ = (
            None
        )
        reshape_24 = q_10.reshape(-1, 16, 1, 8, 1, 8)
        q_10 = None
        q_11 = reshape_24.permute(0, 1, 3, 5, 2, 4)
        reshape_24 = None
        reshape_25 = q_11.reshape(8, 16, -1, 1)
        q_11 = None
        q_12 = reshape_25.transpose(1, 3)
        reshape_25 = None
        kv_6 = torch.conv2d(
            x_101,
            l_self_modules_stages_modules_3_modules_1_modules_self_attn_modules_kv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_101 = l_self_modules_stages_modules_3_modules_1_modules_self_attn_modules_kv_parameters_weight_ = (None)
        kv_7 = torch._C._nn.pad(kv_6, [2, 2, 2, 2], "constant", None)
        kv_6 = None
        unfold_4 = kv_7.unfold(2, 12, 8)
        kv_7 = None
        unfold_5 = unfold_4.unfold(3, 12, 8)
        unfold_4 = None
        reshape_26 = unfold_5.reshape(8, 80, 1, -1)
        unfold_5 = None
        kv_8 = reshape_26.permute(0, 2, 3, 1)
        reshape_26 = None
        split_2 = torch.functional.split(kv_8, [16, 64], dim=-1)
        kv_8 = None
        k_2 = split_2[0]
        v_2 = split_2[1]
        split_2 = None
        transpose_13 = k_2.transpose(-1, -2)
        k_2 = None
        matmul_8 = q_12 @ transpose_13
        transpose_13 = None
        mul_7 = matmul_8 * 0.25
        matmul_8 = None
        q_13 = q_12.reshape(-1, 8, 8, 16)
        q_12 = None
        transpose_14 = l_self_modules_stages_modules_3_modules_1_modules_self_attn_modules_pos_embed_parameters_width_rel_.transpose(
            -1, -2
        )
        l_self_modules_stages_modules_3_modules_1_modules_self_attn_modules_pos_embed_parameters_width_rel_ = (
            None
        )
        x_102 = q_13 @ transpose_14
        transpose_14 = None
        x_103 = x_102.reshape(-1, 8, 23)
        x_102 = None
        pad_11 = torch._C._nn.pad(x_103, [0, 1], "constant", None)
        x_103 = None
        x_pad_12 = pad_11.flatten(1)
        pad_11 = None
        x_pad_13 = torch._C._nn.pad(x_pad_12, [0, 15], "constant", None)
        x_pad_12 = None
        x_pad_14 = x_pad_13.reshape(-1, 9, 23)
        x_pad_13 = None
        x_104 = x_pad_14[
            (slice(None, None, None), slice(None, 8, None), slice(11, None, None))
        ]
        x_pad_14 = None
        reshape_30 = x_104.reshape(8, 8, 1, 8, 12)
        x_104 = None
        x_105 = reshape_30.expand(-1, -1, 12, -1, -1)
        reshape_30 = None
        rel_logits_w_2 = x_105.permute((0, 1, 3, 2, 4))
        x_105 = None
        q_14 = q_13.transpose(1, 2)
        q_13 = None
        transpose_16 = l_self_modules_stages_modules_3_modules_1_modules_self_attn_modules_pos_embed_parameters_height_rel_.transpose(
            -1, -2
        )
        l_self_modules_stages_modules_3_modules_1_modules_self_attn_modules_pos_embed_parameters_height_rel_ = (
            None
        )
        x_106 = q_14 @ transpose_16
        q_14 = transpose_16 = None
        x_107 = x_106.reshape(-1, 8, 23)
        x_106 = None
        pad_13 = torch._C._nn.pad(x_107, [0, 1], "constant", None)
        x_107 = None
        x_pad_15 = pad_13.flatten(1)
        pad_13 = None
        x_pad_16 = torch._C._nn.pad(x_pad_15, [0, 15], "constant", None)
        x_pad_15 = None
        x_pad_17 = x_pad_16.reshape(-1, 9, 23)
        x_pad_16 = None
        x_108 = x_pad_17[
            (slice(None, None, None), slice(None, 8, None), slice(11, None, None))
        ]
        x_pad_17 = None
        reshape_33 = x_108.reshape(8, 8, 1, 8, 12)
        x_108 = None
        x_109 = reshape_33.expand(-1, -1, 12, -1, -1)
        reshape_33 = None
        rel_logits_h_2 = x_109.permute((0, 3, 1, 4, 2))
        x_109 = None
        rel_logits_4 = rel_logits_h_2 + rel_logits_w_2
        rel_logits_h_2 = rel_logits_w_2 = None
        rel_logits_5 = rel_logits_4.reshape(8, 1, 64, -1)
        rel_logits_4 = None
        attn_4 = mul_7 + rel_logits_5
        mul_7 = rel_logits_5 = None
        attn_5 = attn_4.softmax(dim=-1)
        attn_4 = None
        matmul_11 = attn_5 @ v_2
        attn_5 = v_2 = None
        out_6 = matmul_11.transpose(1, 3)
        matmul_11 = None
        out_7 = out_6.reshape(-1, 8, 8, 1, 1)
        out_6 = None
        permute_14 = out_7.permute(0, 3, 1, 4, 2)
        out_7 = None
        contiguous_2 = permute_14.contiguous()
        permute_14 = None
        out_8 = contiguous_2.view(1, 512, 8, 8)
        contiguous_2 = None
        x_110 = torch.nn.functional.batch_norm(
            out_8,
            l_self_modules_stages_modules_3_modules_1_modules_post_attn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_1_modules_post_attn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_1_modules_post_attn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_1_modules_post_attn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_8 = l_self_modules_stages_modules_3_modules_1_modules_post_attn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_1_modules_post_attn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_1_modules_post_attn_parameters_weight_ = (
            l_self_modules_stages_modules_3_modules_1_modules_post_attn_parameters_bias_
        ) = None
        x_111 = torch.nn.functional.silu(x_110, inplace=True)
        x_110 = None
        x_112 = torch.conv2d(
            x_111,
            l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_111 = l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_113 = torch.nn.functional.batch_norm(
            x_112,
            l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_112 = l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_114 = x_113 + input_8
        x_113 = input_8 = None
        input_9 = torch.nn.functional.silu(x_114, inplace=True)
        x_114 = None
        x_115 = torch.nn.functional.adaptive_avg_pool2d(input_9, 1)
        input_9 = None
        x_116 = x_115.flatten(1, -1)
        x_115 = None
        x_117 = torch.nn.functional.dropout(x_116, 0.0, False, False)
        x_116 = None
        x_118 = torch._C._nn.linear(
            x_117,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_117 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_118,)
