import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_kwargs_input_ids_: torch.Tensor,
        L_self_modules_gpt_neox_modules_embed_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_kwargs_attention_mask_: torch.Tensor,
        L_self_modules_gpt_neox_modules_rotary_emb_buffers_inv_freq_: torch.Tensor,
        L_self_modules_gpt_neox_modules_layers_modules_0_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_0_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_0_modules_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_0_modules_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_0_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_0_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_0_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_1_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_1_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_1_modules_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_1_modules_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_1_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_1_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_1_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_2_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_2_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_2_modules_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_2_modules_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_2_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_2_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_2_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_3_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_3_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_3_modules_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_3_modules_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_3_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_3_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_3_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_4_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_4_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_4_modules_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_4_modules_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_4_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_4_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_4_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_5_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_5_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_5_modules_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_5_modules_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_5_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_5_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_5_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_5_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_5_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_5_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_5_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_embed_out_parameters_weight_: torch.nn.parameter.Parameter,
    ):
        l_kwargs_input_ids_ = L_kwargs_input_ids_
        l_self_modules_gpt_neox_modules_embed_in_parameters_weight_ = (
            L_self_modules_gpt_neox_modules_embed_in_parameters_weight_
        )
        l_kwargs_attention_mask_ = L_kwargs_attention_mask_
        l_self_modules_gpt_neox_modules_rotary_emb_buffers_inv_freq_ = (
            L_self_modules_gpt_neox_modules_rotary_emb_buffers_inv_freq_
        )
        l_self_modules_gpt_neox_modules_layers_modules_0_modules_input_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_0_modules_input_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_0_modules_input_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_0_modules_input_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_0_modules_attention_modules_query_key_value_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_0_modules_attention_modules_query_key_value_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_0_modules_attention_modules_query_key_value_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_0_modules_attention_modules_query_key_value_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_0_modules_attention_modules_dense_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_0_modules_attention_modules_dense_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_0_modules_attention_modules_dense_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_0_modules_attention_modules_dense_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_0_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_0_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_1_modules_input_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_1_modules_input_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_1_modules_input_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_1_modules_input_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_1_modules_attention_modules_query_key_value_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_1_modules_attention_modules_query_key_value_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_1_modules_attention_modules_query_key_value_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_1_modules_attention_modules_query_key_value_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_1_modules_attention_modules_dense_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_1_modules_attention_modules_dense_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_1_modules_attention_modules_dense_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_1_modules_attention_modules_dense_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_1_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_1_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_2_modules_input_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_2_modules_input_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_2_modules_input_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_2_modules_input_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_2_modules_attention_modules_query_key_value_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_2_modules_attention_modules_query_key_value_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_2_modules_attention_modules_query_key_value_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_2_modules_attention_modules_query_key_value_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_2_modules_attention_modules_dense_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_2_modules_attention_modules_dense_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_2_modules_attention_modules_dense_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_2_modules_attention_modules_dense_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_2_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_2_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_3_modules_input_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_3_modules_input_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_3_modules_input_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_3_modules_input_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_3_modules_attention_modules_query_key_value_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_3_modules_attention_modules_query_key_value_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_3_modules_attention_modules_query_key_value_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_3_modules_attention_modules_query_key_value_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_3_modules_attention_modules_dense_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_3_modules_attention_modules_dense_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_3_modules_attention_modules_dense_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_3_modules_attention_modules_dense_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_3_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_3_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_4_modules_input_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_4_modules_input_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_4_modules_input_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_4_modules_input_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_4_modules_attention_modules_query_key_value_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_4_modules_attention_modules_query_key_value_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_4_modules_attention_modules_query_key_value_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_4_modules_attention_modules_query_key_value_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_4_modules_attention_modules_dense_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_4_modules_attention_modules_dense_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_4_modules_attention_modules_dense_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_4_modules_attention_modules_dense_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_4_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_4_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_5_modules_input_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_5_modules_input_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_5_modules_input_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_5_modules_input_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_5_modules_attention_modules_query_key_value_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_5_modules_attention_modules_query_key_value_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_5_modules_attention_modules_query_key_value_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_5_modules_attention_modules_query_key_value_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_5_modules_attention_modules_dense_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_5_modules_attention_modules_dense_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_5_modules_attention_modules_dense_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_5_modules_attention_modules_dense_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_5_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_5_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_5_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_5_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_5_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_5_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_5_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_5_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_5_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_5_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_gpt_neox_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_gpt_neox_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_gpt_neox_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_gpt_neox_modules_final_layer_norm_parameters_bias_
        )
        l_self_modules_embed_out_parameters_weight_ = (
            L_self_modules_embed_out_parameters_weight_
        )
        inputs_embeds = torch.nn.functional.embedding(
            l_kwargs_input_ids_,
            l_self_modules_gpt_neox_modules_embed_in_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        l_kwargs_input_ids_ = (
            l_self_modules_gpt_neox_modules_embed_in_parameters_weight_
        ) = None
        cache_position = torch.arange(0, 19, device=device(type="cpu"))
        position_ids = cache_position.unsqueeze(0)
        attention_mask = l_kwargs_attention_mask_.to(
            device=device(type="cpu"), dtype=torch.bool
        )
        l_kwargs_attention_mask_ = None
        kv_arange = torch.arange(19, device=device(type="cpu"))
        kv_arange += 0
        kv_arange_1 = kv_arange
        kv_arange = None
        batch_arange = torch.arange(1, device=device(type="cpu"))
        head_arange = torch.arange(1, device=device(type="cpu"))
        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions()
        lazy_load_decompositions = None
        _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(
            1, "error"
        )
        _vmap_increment_nesting = None
        child = torch._C._functorch._add_batch_dim(batch_arange, 0, 1)
        batch_arange = None
        lazy_load_decompositions_1 = torch._functorch.vmap.lazy_load_decompositions()
        lazy_load_decompositions_1 = None
        _vmap_increment_nesting_1 = torch._C._functorch._vmap_increment_nesting(
            1, "error"
        )
        _vmap_increment_nesting_1 = None
        child_1 = torch._C._functorch._add_batch_dim(head_arange, 0, 2)
        head_arange = child_1 = None
        lazy_load_decompositions_2 = torch._functorch.vmap.lazy_load_decompositions()
        lazy_load_decompositions_2 = None
        _vmap_increment_nesting_2 = torch._C._functorch._vmap_increment_nesting(
            19, "error"
        )
        _vmap_increment_nesting_2 = None
        child_2 = torch._C._functorch._add_batch_dim(cache_position, 0, 3)
        cache_position = None
        lazy_load_decompositions_3 = torch._functorch.vmap.lazy_load_decompositions()
        lazy_load_decompositions_3 = None
        _vmap_increment_nesting_3 = torch._C._functorch._vmap_increment_nesting(
            19, "error"
        )
        _vmap_increment_nesting_3 = None
        child_3 = torch._C._functorch._add_batch_dim(kv_arange_1, 0, 4)
        kv_arange_1 = None
        result = child_2.new_ones((), dtype=torch.bool)
        le = child_3.le(child_2)
        child_2 = None
        result_1 = result.__and__(le)
        result = le = None
        function_ctx = torch.autograd.function.FunctionCtx()
        function_ctx = None
        index = torch.ops.aten.index(attention_mask, [child, child_3])
        attention_mask = child = child_3 = None
        result_2 = result_1.__and__(index)
        result_1 = index = None
        batched_outputs = torch._C._functorch._remove_batch_dim(result_2, 4, 19, 0)
        result_2 = None
        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting()
        _vmap_decrement_nesting = None
        batched_outputs_1 = torch._C._functorch._remove_batch_dim(
            batched_outputs, 3, 19, 0
        )
        batched_outputs = None
        _vmap_decrement_nesting_1 = torch._C._functorch._vmap_decrement_nesting()
        _vmap_decrement_nesting_1 = None
        batched_outputs_2 = torch._C._functorch._remove_batch_dim(
            batched_outputs_1, 2, 1, 0
        )
        batched_outputs_1 = None
        _vmap_decrement_nesting_2 = torch._C._functorch._vmap_decrement_nesting()
        _vmap_decrement_nesting_2 = None
        causal_mask = torch._C._functorch._remove_batch_dim(batched_outputs_2, 1, 1, 0)
        batched_outputs_2 = None
        _vmap_decrement_nesting_3 = torch._C._functorch._vmap_decrement_nesting()
        _vmap_decrement_nesting_3 = None
        hidden_states = torch.nn.functional.dropout(inputs_embeds, 0.0, False, False)
        inputs_embeds = None
        getitem = l_self_modules_gpt_neox_modules_rotary_emb_buffers_inv_freq_[
            (None, slice(None, None, None), None)
        ]
        l_self_modules_gpt_neox_modules_rotary_emb_buffers_inv_freq_ = None
        float_1 = getitem.float()
        getitem = None
        expand = float_1.expand(1, -1, 1)
        float_1 = None
        inv_freq_expanded = expand.to(device(type="cpu"))
        expand = None
        getitem_1 = position_ids[
            (slice(None, None, None), None, slice(None, None, None))
        ]
        position_ids = None
        position_ids_expanded = getitem_1.float()
        getitem_1 = None
        _enter_autocast = torch.amp.autocast_mode._enter_autocast(
            "cpu", None, False, None
        )
        float_3 = inv_freq_expanded.float()
        inv_freq_expanded = None
        float_4 = position_ids_expanded.float()
        position_ids_expanded = None
        matmul = float_3 @ float_4
        float_3 = float_4 = None
        freqs = matmul.transpose(1, 2)
        matmul = None
        emb = torch.cat((freqs, freqs), dim=-1)
        freqs = None
        cos = emb.cos()
        cos_1 = cos * 1.0
        cos = None
        sin = emb.sin()
        emb = None
        sin_1 = sin * 1.0
        sin = None
        _exit_autocast = torch.amp.autocast_mode._exit_autocast(_enter_autocast)
        _enter_autocast = _exit_autocast = None
        cos_2 = cos_1.to(dtype=torch.float16)
        cos_1 = None
        sin_2 = sin_1.to(dtype=torch.float16)
        sin_1 = None
        layer_norm = torch.nn.functional.layer_norm(
            hidden_states,
            (512,),
            l_self_modules_gpt_neox_modules_layers_modules_0_modules_input_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_0_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_0_modules_input_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_0_modules_input_layernorm_parameters_bias_ = (None)
        linear = torch._C._nn.linear(
            layer_norm,
            l_self_modules_gpt_neox_modules_layers_modules_0_modules_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_0_modules_attention_modules_query_key_value_parameters_bias_,
        )
        layer_norm = l_self_modules_gpt_neox_modules_layers_modules_0_modules_attention_modules_query_key_value_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_0_modules_attention_modules_query_key_value_parameters_bias_ = (None)
        view = linear.view((1, 19, -1, 192))
        linear = None
        qkv = view.transpose(1, 2)
        view = None
        chunk = qkv.chunk(3, dim=-1)
        qkv = None
        query_states = chunk[0]
        key_states = chunk[1]
        value_states = chunk[2]
        chunk = None
        cos_3 = cos_2.unsqueeze(1)
        sin_3 = sin_2.unsqueeze(1)
        q_rot = query_states[(Ellipsis, slice(None, 16, None))]
        q_pass = query_states[(Ellipsis, slice(16, None, None))]
        query_states = None
        k_rot = key_states[(Ellipsis, slice(None, 16, None))]
        k_pass = key_states[(Ellipsis, slice(16, None, None))]
        key_states = None
        mul_2 = q_rot * cos_3
        x1 = q_rot[(Ellipsis, slice(None, 8, None))]
        x2 = q_rot[(Ellipsis, slice(8, None, None))]
        q_rot = None
        neg = -x2
        x2 = None
        cat_1 = torch.cat((neg, x1), dim=-1)
        neg = x1 = None
        mul_3 = cat_1 * sin_3
        cat_1 = None
        q_embed = mul_2 + mul_3
        mul_2 = mul_3 = None
        mul_4 = k_rot * cos_3
        cos_3 = None
        x1_1 = k_rot[(Ellipsis, slice(None, 8, None))]
        x2_1 = k_rot[(Ellipsis, slice(8, None, None))]
        k_rot = None
        neg_1 = -x2_1
        x2_1 = None
        cat_2 = torch.cat((neg_1, x1_1), dim=-1)
        neg_1 = x1_1 = None
        mul_5 = cat_2 * sin_3
        cat_2 = sin_3 = None
        k_embed = mul_4 + mul_5
        mul_4 = mul_5 = None
        q_embed_1 = torch.cat([q_embed, q_pass], dim=-1)
        q_embed = q_pass = None
        k_embed_1 = torch.cat([k_embed, k_pass], dim=-1)
        k_embed = k_pass = None
        attention_mask_1 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query = q_embed_1.contiguous()
        q_embed_1 = None
        key = k_embed_1.contiguous()
        k_embed_1 = None
        value = value_states.contiguous()
        value_states = None
        attn_output = torch._C._nn.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask_1,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query = key = value = attention_mask_1 = None
        transpose_2 = attn_output.transpose(1, 2)
        attn_output = None
        attn_output_1 = transpose_2.contiguous()
        transpose_2 = None
        reshape = attn_output_1.reshape(1, 19, -1)
        attn_output_1 = None
        attn_output_2 = reshape.contiguous()
        reshape = None
        attn_output_3 = torch._C._nn.linear(
            attn_output_2,
            l_self_modules_gpt_neox_modules_layers_modules_0_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_0_modules_attention_modules_dense_parameters_bias_,
        )
        attn_output_2 = l_self_modules_gpt_neox_modules_layers_modules_0_modules_attention_modules_dense_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_0_modules_attention_modules_dense_parameters_bias_ = (None)
        attn_output_4 = torch.nn.functional.dropout(attn_output_3, 0.0, False, False)
        attn_output_3 = None
        layer_norm_1 = torch.nn.functional.layer_norm(
            hidden_states,
            (512,),
            l_self_modules_gpt_neox_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_0_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_0_modules_post_attention_layernorm_parameters_bias_ = (None)
        hidden_states_1 = torch._C._nn.linear(
            layer_norm_1,
            l_self_modules_gpt_neox_modules_layers_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layer_norm_1 = l_self_modules_gpt_neox_modules_layers_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        hidden_states_2 = torch._C._nn.gelu(hidden_states_1)
        hidden_states_1 = None
        hidden_states_3 = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_gpt_neox_modules_layers_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_2 = l_self_modules_gpt_neox_modules_layers_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        mlp_output = torch.nn.functional.dropout(hidden_states_3, 0.0, False, False)
        hidden_states_3 = None
        add_2 = mlp_output + attn_output_4
        mlp_output = attn_output_4 = None
        hidden_states_4 = add_2 + hidden_states
        add_2 = hidden_states = None
        layer_norm_2 = torch.nn.functional.layer_norm(
            hidden_states_4,
            (512,),
            l_self_modules_gpt_neox_modules_layers_modules_1_modules_input_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_1_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_1_modules_input_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_1_modules_input_layernorm_parameters_bias_ = (None)
        linear_4 = torch._C._nn.linear(
            layer_norm_2,
            l_self_modules_gpt_neox_modules_layers_modules_1_modules_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_1_modules_attention_modules_query_key_value_parameters_bias_,
        )
        layer_norm_2 = l_self_modules_gpt_neox_modules_layers_modules_1_modules_attention_modules_query_key_value_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_1_modules_attention_modules_query_key_value_parameters_bias_ = (None)
        view_1 = linear_4.view((1, 19, -1, 192))
        linear_4 = None
        qkv_1 = view_1.transpose(1, 2)
        view_1 = None
        chunk_1 = qkv_1.chunk(3, dim=-1)
        qkv_1 = None
        query_states_1 = chunk_1[0]
        key_states_1 = chunk_1[1]
        value_states_1 = chunk_1[2]
        chunk_1 = None
        cos_4 = cos_2.unsqueeze(1)
        sin_4 = sin_2.unsqueeze(1)
        q_rot_1 = query_states_1[(Ellipsis, slice(None, 16, None))]
        q_pass_1 = query_states_1[(Ellipsis, slice(16, None, None))]
        query_states_1 = None
        k_rot_1 = key_states_1[(Ellipsis, slice(None, 16, None))]
        k_pass_1 = key_states_1[(Ellipsis, slice(16, None, None))]
        key_states_1 = None
        mul_6 = q_rot_1 * cos_4
        x1_2 = q_rot_1[(Ellipsis, slice(None, 8, None))]
        x2_2 = q_rot_1[(Ellipsis, slice(8, None, None))]
        q_rot_1 = None
        neg_2 = -x2_2
        x2_2 = None
        cat_5 = torch.cat((neg_2, x1_2), dim=-1)
        neg_2 = x1_2 = None
        mul_7 = cat_5 * sin_4
        cat_5 = None
        q_embed_2 = mul_6 + mul_7
        mul_6 = mul_7 = None
        mul_8 = k_rot_1 * cos_4
        cos_4 = None
        x1_3 = k_rot_1[(Ellipsis, slice(None, 8, None))]
        x2_3 = k_rot_1[(Ellipsis, slice(8, None, None))]
        k_rot_1 = None
        neg_3 = -x2_3
        x2_3 = None
        cat_6 = torch.cat((neg_3, x1_3), dim=-1)
        neg_3 = x1_3 = None
        mul_9 = cat_6 * sin_4
        cat_6 = sin_4 = None
        k_embed_2 = mul_8 + mul_9
        mul_8 = mul_9 = None
        q_embed_3 = torch.cat([q_embed_2, q_pass_1], dim=-1)
        q_embed_2 = q_pass_1 = None
        k_embed_3 = torch.cat([k_embed_2, k_pass_1], dim=-1)
        k_embed_2 = k_pass_1 = None
        attention_mask_2 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_1 = q_embed_3.contiguous()
        q_embed_3 = None
        key_1 = k_embed_3.contiguous()
        k_embed_3 = None
        value_1 = value_states_1.contiguous()
        value_states_1 = None
        attn_output_5 = torch._C._nn.scaled_dot_product_attention(
            query_1,
            key_1,
            value_1,
            attn_mask=attention_mask_2,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_1 = key_1 = value_1 = attention_mask_2 = None
        transpose_4 = attn_output_5.transpose(1, 2)
        attn_output_5 = None
        attn_output_6 = transpose_4.contiguous()
        transpose_4 = None
        reshape_1 = attn_output_6.reshape(1, 19, -1)
        attn_output_6 = None
        attn_output_7 = reshape_1.contiguous()
        reshape_1 = None
        attn_output_8 = torch._C._nn.linear(
            attn_output_7,
            l_self_modules_gpt_neox_modules_layers_modules_1_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_1_modules_attention_modules_dense_parameters_bias_,
        )
        attn_output_7 = l_self_modules_gpt_neox_modules_layers_modules_1_modules_attention_modules_dense_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_1_modules_attention_modules_dense_parameters_bias_ = (None)
        attn_output_9 = torch.nn.functional.dropout(attn_output_8, 0.0, False, False)
        attn_output_8 = None
        layer_norm_3 = torch.nn.functional.layer_norm(
            hidden_states_4,
            (512,),
            l_self_modules_gpt_neox_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_1_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_1_modules_post_attention_layernorm_parameters_bias_ = (None)
        hidden_states_5 = torch._C._nn.linear(
            layer_norm_3,
            l_self_modules_gpt_neox_modules_layers_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layer_norm_3 = l_self_modules_gpt_neox_modules_layers_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        hidden_states_6 = torch._C._nn.gelu(hidden_states_5)
        hidden_states_5 = None
        hidden_states_7 = torch._C._nn.linear(
            hidden_states_6,
            l_self_modules_gpt_neox_modules_layers_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_6 = l_self_modules_gpt_neox_modules_layers_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        mlp_output_1 = torch.nn.functional.dropout(hidden_states_7, 0.0, False, False)
        hidden_states_7 = None
        add_6 = mlp_output_1 + attn_output_9
        mlp_output_1 = attn_output_9 = None
        hidden_states_8 = add_6 + hidden_states_4
        add_6 = hidden_states_4 = None
        layer_norm_4 = torch.nn.functional.layer_norm(
            hidden_states_8,
            (512,),
            l_self_modules_gpt_neox_modules_layers_modules_2_modules_input_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_2_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_2_modules_input_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_2_modules_input_layernorm_parameters_bias_ = (None)
        linear_8 = torch._C._nn.linear(
            layer_norm_4,
            l_self_modules_gpt_neox_modules_layers_modules_2_modules_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_2_modules_attention_modules_query_key_value_parameters_bias_,
        )
        layer_norm_4 = l_self_modules_gpt_neox_modules_layers_modules_2_modules_attention_modules_query_key_value_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_2_modules_attention_modules_query_key_value_parameters_bias_ = (None)
        view_2 = linear_8.view((1, 19, -1, 192))
        linear_8 = None
        qkv_2 = view_2.transpose(1, 2)
        view_2 = None
        chunk_2 = qkv_2.chunk(3, dim=-1)
        qkv_2 = None
        query_states_2 = chunk_2[0]
        key_states_2 = chunk_2[1]
        value_states_2 = chunk_2[2]
        chunk_2 = None
        cos_5 = cos_2.unsqueeze(1)
        sin_5 = sin_2.unsqueeze(1)
        q_rot_2 = query_states_2[(Ellipsis, slice(None, 16, None))]
        q_pass_2 = query_states_2[(Ellipsis, slice(16, None, None))]
        query_states_2 = None
        k_rot_2 = key_states_2[(Ellipsis, slice(None, 16, None))]
        k_pass_2 = key_states_2[(Ellipsis, slice(16, None, None))]
        key_states_2 = None
        mul_10 = q_rot_2 * cos_5
        x1_4 = q_rot_2[(Ellipsis, slice(None, 8, None))]
        x2_4 = q_rot_2[(Ellipsis, slice(8, None, None))]
        q_rot_2 = None
        neg_4 = -x2_4
        x2_4 = None
        cat_9 = torch.cat((neg_4, x1_4), dim=-1)
        neg_4 = x1_4 = None
        mul_11 = cat_9 * sin_5
        cat_9 = None
        q_embed_4 = mul_10 + mul_11
        mul_10 = mul_11 = None
        mul_12 = k_rot_2 * cos_5
        cos_5 = None
        x1_5 = k_rot_2[(Ellipsis, slice(None, 8, None))]
        x2_5 = k_rot_2[(Ellipsis, slice(8, None, None))]
        k_rot_2 = None
        neg_5 = -x2_5
        x2_5 = None
        cat_10 = torch.cat((neg_5, x1_5), dim=-1)
        neg_5 = x1_5 = None
        mul_13 = cat_10 * sin_5
        cat_10 = sin_5 = None
        k_embed_4 = mul_12 + mul_13
        mul_12 = mul_13 = None
        q_embed_5 = torch.cat([q_embed_4, q_pass_2], dim=-1)
        q_embed_4 = q_pass_2 = None
        k_embed_5 = torch.cat([k_embed_4, k_pass_2], dim=-1)
        k_embed_4 = k_pass_2 = None
        attention_mask_3 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_2 = q_embed_5.contiguous()
        q_embed_5 = None
        key_2 = k_embed_5.contiguous()
        k_embed_5 = None
        value_2 = value_states_2.contiguous()
        value_states_2 = None
        attn_output_10 = torch._C._nn.scaled_dot_product_attention(
            query_2,
            key_2,
            value_2,
            attn_mask=attention_mask_3,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_2 = key_2 = value_2 = attention_mask_3 = None
        transpose_6 = attn_output_10.transpose(1, 2)
        attn_output_10 = None
        attn_output_11 = transpose_6.contiguous()
        transpose_6 = None
        reshape_2 = attn_output_11.reshape(1, 19, -1)
        attn_output_11 = None
        attn_output_12 = reshape_2.contiguous()
        reshape_2 = None
        attn_output_13 = torch._C._nn.linear(
            attn_output_12,
            l_self_modules_gpt_neox_modules_layers_modules_2_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_2_modules_attention_modules_dense_parameters_bias_,
        )
        attn_output_12 = l_self_modules_gpt_neox_modules_layers_modules_2_modules_attention_modules_dense_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_2_modules_attention_modules_dense_parameters_bias_ = (None)
        attn_output_14 = torch.nn.functional.dropout(attn_output_13, 0.0, False, False)
        attn_output_13 = None
        layer_norm_5 = torch.nn.functional.layer_norm(
            hidden_states_8,
            (512,),
            l_self_modules_gpt_neox_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_2_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_2_modules_post_attention_layernorm_parameters_bias_ = (None)
        hidden_states_9 = torch._C._nn.linear(
            layer_norm_5,
            l_self_modules_gpt_neox_modules_layers_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layer_norm_5 = l_self_modules_gpt_neox_modules_layers_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        hidden_states_10 = torch._C._nn.gelu(hidden_states_9)
        hidden_states_9 = None
        hidden_states_11 = torch._C._nn.linear(
            hidden_states_10,
            l_self_modules_gpt_neox_modules_layers_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_10 = l_self_modules_gpt_neox_modules_layers_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        mlp_output_2 = torch.nn.functional.dropout(hidden_states_11, 0.0, False, False)
        hidden_states_11 = None
        add_10 = mlp_output_2 + attn_output_14
        mlp_output_2 = attn_output_14 = None
        hidden_states_12 = add_10 + hidden_states_8
        add_10 = hidden_states_8 = None
        layer_norm_6 = torch.nn.functional.layer_norm(
            hidden_states_12,
            (512,),
            l_self_modules_gpt_neox_modules_layers_modules_3_modules_input_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_3_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_3_modules_input_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_3_modules_input_layernorm_parameters_bias_ = (None)
        linear_12 = torch._C._nn.linear(
            layer_norm_6,
            l_self_modules_gpt_neox_modules_layers_modules_3_modules_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_3_modules_attention_modules_query_key_value_parameters_bias_,
        )
        layer_norm_6 = l_self_modules_gpt_neox_modules_layers_modules_3_modules_attention_modules_query_key_value_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_3_modules_attention_modules_query_key_value_parameters_bias_ = (None)
        view_3 = linear_12.view((1, 19, -1, 192))
        linear_12 = None
        qkv_3 = view_3.transpose(1, 2)
        view_3 = None
        chunk_3 = qkv_3.chunk(3, dim=-1)
        qkv_3 = None
        query_states_3 = chunk_3[0]
        key_states_3 = chunk_3[1]
        value_states_3 = chunk_3[2]
        chunk_3 = None
        cos_6 = cos_2.unsqueeze(1)
        sin_6 = sin_2.unsqueeze(1)
        q_rot_3 = query_states_3[(Ellipsis, slice(None, 16, None))]
        q_pass_3 = query_states_3[(Ellipsis, slice(16, None, None))]
        query_states_3 = None
        k_rot_3 = key_states_3[(Ellipsis, slice(None, 16, None))]
        k_pass_3 = key_states_3[(Ellipsis, slice(16, None, None))]
        key_states_3 = None
        mul_14 = q_rot_3 * cos_6
        x1_6 = q_rot_3[(Ellipsis, slice(None, 8, None))]
        x2_6 = q_rot_3[(Ellipsis, slice(8, None, None))]
        q_rot_3 = None
        neg_6 = -x2_6
        x2_6 = None
        cat_13 = torch.cat((neg_6, x1_6), dim=-1)
        neg_6 = x1_6 = None
        mul_15 = cat_13 * sin_6
        cat_13 = None
        q_embed_6 = mul_14 + mul_15
        mul_14 = mul_15 = None
        mul_16 = k_rot_3 * cos_6
        cos_6 = None
        x1_7 = k_rot_3[(Ellipsis, slice(None, 8, None))]
        x2_7 = k_rot_3[(Ellipsis, slice(8, None, None))]
        k_rot_3 = None
        neg_7 = -x2_7
        x2_7 = None
        cat_14 = torch.cat((neg_7, x1_7), dim=-1)
        neg_7 = x1_7 = None
        mul_17 = cat_14 * sin_6
        cat_14 = sin_6 = None
        k_embed_6 = mul_16 + mul_17
        mul_16 = mul_17 = None
        q_embed_7 = torch.cat([q_embed_6, q_pass_3], dim=-1)
        q_embed_6 = q_pass_3 = None
        k_embed_7 = torch.cat([k_embed_6, k_pass_3], dim=-1)
        k_embed_6 = k_pass_3 = None
        attention_mask_4 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_3 = q_embed_7.contiguous()
        q_embed_7 = None
        key_3 = k_embed_7.contiguous()
        k_embed_7 = None
        value_3 = value_states_3.contiguous()
        value_states_3 = None
        attn_output_15 = torch._C._nn.scaled_dot_product_attention(
            query_3,
            key_3,
            value_3,
            attn_mask=attention_mask_4,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_3 = key_3 = value_3 = attention_mask_4 = None
        transpose_8 = attn_output_15.transpose(1, 2)
        attn_output_15 = None
        attn_output_16 = transpose_8.contiguous()
        transpose_8 = None
        reshape_3 = attn_output_16.reshape(1, 19, -1)
        attn_output_16 = None
        attn_output_17 = reshape_3.contiguous()
        reshape_3 = None
        attn_output_18 = torch._C._nn.linear(
            attn_output_17,
            l_self_modules_gpt_neox_modules_layers_modules_3_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_3_modules_attention_modules_dense_parameters_bias_,
        )
        attn_output_17 = l_self_modules_gpt_neox_modules_layers_modules_3_modules_attention_modules_dense_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_3_modules_attention_modules_dense_parameters_bias_ = (None)
        attn_output_19 = torch.nn.functional.dropout(attn_output_18, 0.0, False, False)
        attn_output_18 = None
        layer_norm_7 = torch.nn.functional.layer_norm(
            hidden_states_12,
            (512,),
            l_self_modules_gpt_neox_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_3_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_3_modules_post_attention_layernorm_parameters_bias_ = (None)
        hidden_states_13 = torch._C._nn.linear(
            layer_norm_7,
            l_self_modules_gpt_neox_modules_layers_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layer_norm_7 = l_self_modules_gpt_neox_modules_layers_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        hidden_states_14 = torch._C._nn.gelu(hidden_states_13)
        hidden_states_13 = None
        hidden_states_15 = torch._C._nn.linear(
            hidden_states_14,
            l_self_modules_gpt_neox_modules_layers_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_14 = l_self_modules_gpt_neox_modules_layers_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        mlp_output_3 = torch.nn.functional.dropout(hidden_states_15, 0.0, False, False)
        hidden_states_15 = None
        add_14 = mlp_output_3 + attn_output_19
        mlp_output_3 = attn_output_19 = None
        hidden_states_16 = add_14 + hidden_states_12
        add_14 = hidden_states_12 = None
        layer_norm_8 = torch.nn.functional.layer_norm(
            hidden_states_16,
            (512,),
            l_self_modules_gpt_neox_modules_layers_modules_4_modules_input_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_4_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_4_modules_input_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_4_modules_input_layernorm_parameters_bias_ = (None)
        linear_16 = torch._C._nn.linear(
            layer_norm_8,
            l_self_modules_gpt_neox_modules_layers_modules_4_modules_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_4_modules_attention_modules_query_key_value_parameters_bias_,
        )
        layer_norm_8 = l_self_modules_gpt_neox_modules_layers_modules_4_modules_attention_modules_query_key_value_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_4_modules_attention_modules_query_key_value_parameters_bias_ = (None)
        view_4 = linear_16.view((1, 19, -1, 192))
        linear_16 = None
        qkv_4 = view_4.transpose(1, 2)
        view_4 = None
        chunk_4 = qkv_4.chunk(3, dim=-1)
        qkv_4 = None
        query_states_4 = chunk_4[0]
        key_states_4 = chunk_4[1]
        value_states_4 = chunk_4[2]
        chunk_4 = None
        cos_7 = cos_2.unsqueeze(1)
        sin_7 = sin_2.unsqueeze(1)
        q_rot_4 = query_states_4[(Ellipsis, slice(None, 16, None))]
        q_pass_4 = query_states_4[(Ellipsis, slice(16, None, None))]
        query_states_4 = None
        k_rot_4 = key_states_4[(Ellipsis, slice(None, 16, None))]
        k_pass_4 = key_states_4[(Ellipsis, slice(16, None, None))]
        key_states_4 = None
        mul_18 = q_rot_4 * cos_7
        x1_8 = q_rot_4[(Ellipsis, slice(None, 8, None))]
        x2_8 = q_rot_4[(Ellipsis, slice(8, None, None))]
        q_rot_4 = None
        neg_8 = -x2_8
        x2_8 = None
        cat_17 = torch.cat((neg_8, x1_8), dim=-1)
        neg_8 = x1_8 = None
        mul_19 = cat_17 * sin_7
        cat_17 = None
        q_embed_8 = mul_18 + mul_19
        mul_18 = mul_19 = None
        mul_20 = k_rot_4 * cos_7
        cos_7 = None
        x1_9 = k_rot_4[(Ellipsis, slice(None, 8, None))]
        x2_9 = k_rot_4[(Ellipsis, slice(8, None, None))]
        k_rot_4 = None
        neg_9 = -x2_9
        x2_9 = None
        cat_18 = torch.cat((neg_9, x1_9), dim=-1)
        neg_9 = x1_9 = None
        mul_21 = cat_18 * sin_7
        cat_18 = sin_7 = None
        k_embed_8 = mul_20 + mul_21
        mul_20 = mul_21 = None
        q_embed_9 = torch.cat([q_embed_8, q_pass_4], dim=-1)
        q_embed_8 = q_pass_4 = None
        k_embed_9 = torch.cat([k_embed_8, k_pass_4], dim=-1)
        k_embed_8 = k_pass_4 = None
        attention_mask_5 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_4 = q_embed_9.contiguous()
        q_embed_9 = None
        key_4 = k_embed_9.contiguous()
        k_embed_9 = None
        value_4 = value_states_4.contiguous()
        value_states_4 = None
        attn_output_20 = torch._C._nn.scaled_dot_product_attention(
            query_4,
            key_4,
            value_4,
            attn_mask=attention_mask_5,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_4 = key_4 = value_4 = attention_mask_5 = None
        transpose_10 = attn_output_20.transpose(1, 2)
        attn_output_20 = None
        attn_output_21 = transpose_10.contiguous()
        transpose_10 = None
        reshape_4 = attn_output_21.reshape(1, 19, -1)
        attn_output_21 = None
        attn_output_22 = reshape_4.contiguous()
        reshape_4 = None
        attn_output_23 = torch._C._nn.linear(
            attn_output_22,
            l_self_modules_gpt_neox_modules_layers_modules_4_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_4_modules_attention_modules_dense_parameters_bias_,
        )
        attn_output_22 = l_self_modules_gpt_neox_modules_layers_modules_4_modules_attention_modules_dense_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_4_modules_attention_modules_dense_parameters_bias_ = (None)
        attn_output_24 = torch.nn.functional.dropout(attn_output_23, 0.0, False, False)
        attn_output_23 = None
        layer_norm_9 = torch.nn.functional.layer_norm(
            hidden_states_16,
            (512,),
            l_self_modules_gpt_neox_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_4_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_4_modules_post_attention_layernorm_parameters_bias_ = (None)
        hidden_states_17 = torch._C._nn.linear(
            layer_norm_9,
            l_self_modules_gpt_neox_modules_layers_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layer_norm_9 = l_self_modules_gpt_neox_modules_layers_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        hidden_states_18 = torch._C._nn.gelu(hidden_states_17)
        hidden_states_17 = None
        hidden_states_19 = torch._C._nn.linear(
            hidden_states_18,
            l_self_modules_gpt_neox_modules_layers_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_18 = l_self_modules_gpt_neox_modules_layers_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        mlp_output_4 = torch.nn.functional.dropout(hidden_states_19, 0.0, False, False)
        hidden_states_19 = None
        add_18 = mlp_output_4 + attn_output_24
        mlp_output_4 = attn_output_24 = None
        hidden_states_20 = add_18 + hidden_states_16
        add_18 = hidden_states_16 = None
        layer_norm_10 = torch.nn.functional.layer_norm(
            hidden_states_20,
            (512,),
            l_self_modules_gpt_neox_modules_layers_modules_5_modules_input_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_5_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_5_modules_input_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_5_modules_input_layernorm_parameters_bias_ = (None)
        linear_20 = torch._C._nn.linear(
            layer_norm_10,
            l_self_modules_gpt_neox_modules_layers_modules_5_modules_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_5_modules_attention_modules_query_key_value_parameters_bias_,
        )
        layer_norm_10 = l_self_modules_gpt_neox_modules_layers_modules_5_modules_attention_modules_query_key_value_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_5_modules_attention_modules_query_key_value_parameters_bias_ = (None)
        view_5 = linear_20.view((1, 19, -1, 192))
        linear_20 = None
        qkv_5 = view_5.transpose(1, 2)
        view_5 = None
        chunk_5 = qkv_5.chunk(3, dim=-1)
        qkv_5 = None
        query_states_5 = chunk_5[0]
        key_states_5 = chunk_5[1]
        value_states_5 = chunk_5[2]
        chunk_5 = None
        cos_8 = cos_2.unsqueeze(1)
        cos_2 = None
        sin_8 = sin_2.unsqueeze(1)
        sin_2 = None
        q_rot_5 = query_states_5[(Ellipsis, slice(None, 16, None))]
        q_pass_5 = query_states_5[(Ellipsis, slice(16, None, None))]
        query_states_5 = None
        k_rot_5 = key_states_5[(Ellipsis, slice(None, 16, None))]
        k_pass_5 = key_states_5[(Ellipsis, slice(16, None, None))]
        key_states_5 = None
        mul_22 = q_rot_5 * cos_8
        x1_10 = q_rot_5[(Ellipsis, slice(None, 8, None))]
        x2_10 = q_rot_5[(Ellipsis, slice(8, None, None))]
        q_rot_5 = None
        neg_10 = -x2_10
        x2_10 = None
        cat_21 = torch.cat((neg_10, x1_10), dim=-1)
        neg_10 = x1_10 = None
        mul_23 = cat_21 * sin_8
        cat_21 = None
        q_embed_10 = mul_22 + mul_23
        mul_22 = mul_23 = None
        mul_24 = k_rot_5 * cos_8
        cos_8 = None
        x1_11 = k_rot_5[(Ellipsis, slice(None, 8, None))]
        x2_11 = k_rot_5[(Ellipsis, slice(8, None, None))]
        k_rot_5 = None
        neg_11 = -x2_11
        x2_11 = None
        cat_22 = torch.cat((neg_11, x1_11), dim=-1)
        neg_11 = x1_11 = None
        mul_25 = cat_22 * sin_8
        cat_22 = sin_8 = None
        k_embed_10 = mul_24 + mul_25
        mul_24 = mul_25 = None
        q_embed_11 = torch.cat([q_embed_10, q_pass_5], dim=-1)
        q_embed_10 = q_pass_5 = None
        k_embed_11 = torch.cat([k_embed_10, k_pass_5], dim=-1)
        k_embed_10 = k_pass_5 = None
        attention_mask_6 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        causal_mask = None
        query_5 = q_embed_11.contiguous()
        q_embed_11 = None
        key_5 = k_embed_11.contiguous()
        k_embed_11 = None
        value_5 = value_states_5.contiguous()
        value_states_5 = None
        attn_output_25 = torch._C._nn.scaled_dot_product_attention(
            query_5,
            key_5,
            value_5,
            attn_mask=attention_mask_6,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_5 = key_5 = value_5 = attention_mask_6 = None
        transpose_12 = attn_output_25.transpose(1, 2)
        attn_output_25 = None
        attn_output_26 = transpose_12.contiguous()
        transpose_12 = None
        reshape_5 = attn_output_26.reshape(1, 19, -1)
        attn_output_26 = None
        attn_output_27 = reshape_5.contiguous()
        reshape_5 = None
        attn_output_28 = torch._C._nn.linear(
            attn_output_27,
            l_self_modules_gpt_neox_modules_layers_modules_5_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_5_modules_attention_modules_dense_parameters_bias_,
        )
        attn_output_27 = l_self_modules_gpt_neox_modules_layers_modules_5_modules_attention_modules_dense_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_5_modules_attention_modules_dense_parameters_bias_ = (None)
        attn_output_29 = torch.nn.functional.dropout(attn_output_28, 0.0, False, False)
        attn_output_28 = None
        layer_norm_11 = torch.nn.functional.layer_norm(
            hidden_states_20,
            (512,),
            l_self_modules_gpt_neox_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_5_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_5_modules_post_attention_layernorm_parameters_bias_ = (None)
        hidden_states_21 = torch._C._nn.linear(
            layer_norm_11,
            l_self_modules_gpt_neox_modules_layers_modules_5_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_5_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layer_norm_11 = l_self_modules_gpt_neox_modules_layers_modules_5_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_5_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        hidden_states_22 = torch._C._nn.gelu(hidden_states_21)
        hidden_states_21 = None
        hidden_states_23 = torch._C._nn.linear(
            hidden_states_22,
            l_self_modules_gpt_neox_modules_layers_modules_5_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_5_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_22 = l_self_modules_gpt_neox_modules_layers_modules_5_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_5_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        mlp_output_5 = torch.nn.functional.dropout(hidden_states_23, 0.0, False, False)
        hidden_states_23 = None
        add_22 = mlp_output_5 + attn_output_29
        mlp_output_5 = attn_output_29 = None
        hidden_states_24 = add_22 + hidden_states_20
        add_22 = hidden_states_20 = None
        hidden_states_25 = torch.nn.functional.layer_norm(
            hidden_states_24,
            (512,),
            l_self_modules_gpt_neox_modules_final_layer_norm_parameters_weight_,
            l_self_modules_gpt_neox_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_24 = (
            l_self_modules_gpt_neox_modules_final_layer_norm_parameters_weight_
        ) = l_self_modules_gpt_neox_modules_final_layer_norm_parameters_bias_ = None
        getitem_74 = hidden_states_25[
            (slice(None, None, None), slice(0, None, None), slice(None, None, None))
        ]
        hidden_states_25 = None
        logits = torch._C._nn.linear(
            getitem_74, l_self_modules_embed_out_parameters_weight_, None
        )
        getitem_74 = l_self_modules_embed_out_parameters_weight_ = None
        return (logits,)
