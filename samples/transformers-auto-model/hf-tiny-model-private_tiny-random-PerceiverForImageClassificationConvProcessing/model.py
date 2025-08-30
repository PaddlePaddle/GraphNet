import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_stack0_0_: torch.Tensor,
        L_self_modules_embeddings_parameters_latents_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_layernorm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_layernorm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_layernorm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_layernorm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_cross_attention_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_cross_attention_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_cross_attention_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_cross_attention_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_cross_attention_modules_mlp_modules_dense1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_cross_attention_modules_mlp_modules_dense1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_cross_attention_modules_mlp_modules_dense2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_cross_attention_modules_mlp_modules_dense2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_self_modules_layernorm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_self_modules_layernorm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_self_attends_modules_0_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_self_attends_modules_0_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_self_attends_modules_0_modules_mlp_modules_dense1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_self_attends_modules_0_modules_mlp_modules_dense1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_self_attends_modules_0_modules_mlp_modules_dense2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_self_attends_modules_0_modules_mlp_modules_dense2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_self_modules_layernorm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_self_modules_layernorm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_self_attends_modules_1_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_self_attends_modules_1_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_self_attends_modules_1_modules_mlp_modules_dense1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_self_attends_modules_1_modules_mlp_modules_dense1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_self_attends_modules_1_modules_mlp_modules_dense2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_self_attends_modules_1_modules_mlp_modules_dense2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_decoder_modules_output_position_encodings_parameters_position_embeddings_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_layernorm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_layernorm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_layernorm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_layernorm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_mlp_modules_dense1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_mlp_modules_dense1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_mlp_modules_dense2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_mlp_modules_dense2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_decoder_modules_final_layer_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_decoder_modules_final_layer_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_stack0_0_ = L_stack0_0_
        l_self_modules_embeddings_parameters_latents_ = (
            L_self_modules_embeddings_parameters_latents_
        )
        l_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_layernorm1_parameters_weight_ = L_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_layernorm1_parameters_weight_
        l_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_layernorm1_parameters_bias_ = L_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_layernorm1_parameters_bias_
        l_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_layernorm2_parameters_weight_ = L_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_layernorm2_parameters_weight_
        l_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_layernorm2_parameters_bias_ = L_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_layernorm2_parameters_bias_
        l_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_cross_attention_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_cross_attention_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_cross_attention_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_cross_attention_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_cross_attention_modules_layernorm_parameters_weight_ = L_self_modules_encoder_modules_cross_attention_modules_layernorm_parameters_weight_
        l_self_modules_encoder_modules_cross_attention_modules_layernorm_parameters_bias_ = L_self_modules_encoder_modules_cross_attention_modules_layernorm_parameters_bias_
        l_self_modules_encoder_modules_cross_attention_modules_mlp_modules_dense1_parameters_weight_ = L_self_modules_encoder_modules_cross_attention_modules_mlp_modules_dense1_parameters_weight_
        l_self_modules_encoder_modules_cross_attention_modules_mlp_modules_dense1_parameters_bias_ = L_self_modules_encoder_modules_cross_attention_modules_mlp_modules_dense1_parameters_bias_
        l_self_modules_encoder_modules_cross_attention_modules_mlp_modules_dense2_parameters_weight_ = L_self_modules_encoder_modules_cross_attention_modules_mlp_modules_dense2_parameters_weight_
        l_self_modules_encoder_modules_cross_attention_modules_mlp_modules_dense2_parameters_bias_ = L_self_modules_encoder_modules_cross_attention_modules_mlp_modules_dense2_parameters_bias_
        l_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_self_modules_layernorm1_parameters_weight_ = L_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_self_modules_layernorm1_parameters_weight_
        l_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_self_modules_layernorm1_parameters_bias_ = L_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_self_modules_layernorm1_parameters_bias_
        l_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_self_attends_modules_0_modules_layernorm_parameters_weight_ = L_self_modules_encoder_modules_self_attends_modules_0_modules_layernorm_parameters_weight_
        l_self_modules_encoder_modules_self_attends_modules_0_modules_layernorm_parameters_bias_ = L_self_modules_encoder_modules_self_attends_modules_0_modules_layernorm_parameters_bias_
        l_self_modules_encoder_modules_self_attends_modules_0_modules_mlp_modules_dense1_parameters_weight_ = L_self_modules_encoder_modules_self_attends_modules_0_modules_mlp_modules_dense1_parameters_weight_
        l_self_modules_encoder_modules_self_attends_modules_0_modules_mlp_modules_dense1_parameters_bias_ = L_self_modules_encoder_modules_self_attends_modules_0_modules_mlp_modules_dense1_parameters_bias_
        l_self_modules_encoder_modules_self_attends_modules_0_modules_mlp_modules_dense2_parameters_weight_ = L_self_modules_encoder_modules_self_attends_modules_0_modules_mlp_modules_dense2_parameters_weight_
        l_self_modules_encoder_modules_self_attends_modules_0_modules_mlp_modules_dense2_parameters_bias_ = L_self_modules_encoder_modules_self_attends_modules_0_modules_mlp_modules_dense2_parameters_bias_
        l_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_self_modules_layernorm1_parameters_weight_ = L_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_self_modules_layernorm1_parameters_weight_
        l_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_self_modules_layernorm1_parameters_bias_ = L_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_self_modules_layernorm1_parameters_bias_
        l_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_self_attends_modules_1_modules_layernorm_parameters_weight_ = L_self_modules_encoder_modules_self_attends_modules_1_modules_layernorm_parameters_weight_
        l_self_modules_encoder_modules_self_attends_modules_1_modules_layernorm_parameters_bias_ = L_self_modules_encoder_modules_self_attends_modules_1_modules_layernorm_parameters_bias_
        l_self_modules_encoder_modules_self_attends_modules_1_modules_mlp_modules_dense1_parameters_weight_ = L_self_modules_encoder_modules_self_attends_modules_1_modules_mlp_modules_dense1_parameters_weight_
        l_self_modules_encoder_modules_self_attends_modules_1_modules_mlp_modules_dense1_parameters_bias_ = L_self_modules_encoder_modules_self_attends_modules_1_modules_mlp_modules_dense1_parameters_bias_
        l_self_modules_encoder_modules_self_attends_modules_1_modules_mlp_modules_dense2_parameters_weight_ = L_self_modules_encoder_modules_self_attends_modules_1_modules_mlp_modules_dense2_parameters_weight_
        l_self_modules_encoder_modules_self_attends_modules_1_modules_mlp_modules_dense2_parameters_bias_ = L_self_modules_encoder_modules_self_attends_modules_1_modules_mlp_modules_dense2_parameters_bias_
        l_self_modules_decoder_modules_decoder_modules_output_position_encodings_parameters_position_embeddings_ = L_self_modules_decoder_modules_decoder_modules_output_position_encodings_parameters_position_embeddings_
        l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_layernorm1_parameters_weight_ = L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_layernorm1_parameters_weight_
        l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_layernorm1_parameters_bias_ = L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_layernorm1_parameters_bias_
        l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_layernorm2_parameters_weight_ = L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_layernorm2_parameters_weight_
        l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_layernorm2_parameters_bias_ = L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_layernorm2_parameters_bias_
        l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_layernorm_parameters_weight_ = L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_layernorm_parameters_weight_
        l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_layernorm_parameters_bias_ = L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_layernorm_parameters_bias_
        l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_mlp_modules_dense1_parameters_weight_ = L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_mlp_modules_dense1_parameters_weight_
        l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_mlp_modules_dense1_parameters_bias_ = L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_mlp_modules_dense1_parameters_bias_
        l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_mlp_modules_dense2_parameters_weight_ = L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_mlp_modules_dense2_parameters_weight_
        l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_mlp_modules_dense2_parameters_bias_ = L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_mlp_modules_dense2_parameters_bias_
        l_self_modules_decoder_modules_decoder_modules_final_layer_parameters_weight_ = L_self_modules_decoder_modules_decoder_modules_final_layer_parameters_weight_
        l_self_modules_decoder_modules_decoder_modules_final_layer_parameters_bias_ = (
            L_self_modules_decoder_modules_decoder_modules_final_layer_parameters_bias_
        )
        attention_mask = torch.ones((1, 49), device=device(type="cuda", index=0))
        encoder_extended_attention_mask = attention_mask[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        attention_mask = None
        encoder_extended_attention_mask_1 = encoder_extended_attention_mask.to(
            dtype=torch.float32
        )
        encoder_extended_attention_mask = None
        sub = 1.0 - encoder_extended_attention_mask_1
        encoder_extended_attention_mask_1 = None
        encoder_extended_attention_mask_2 = sub * -3.4028234663852886e38
        sub = None
        embedding_output = l_self_modules_embeddings_parameters_latents_.expand(
            1, -1, -1
        )
        l_self_modules_embeddings_parameters_latents_ = None
        hidden_states = torch.nn.functional.layer_norm(
            embedding_output,
            (20,),
            l_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_layernorm1_parameters_weight_,
            l_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_layernorm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_layernorm1_parameters_weight_ = l_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_layernorm1_parameters_bias_ = (None)
        inputs = torch.nn.functional.layer_norm(
            l_stack0_0_,
            (322,),
            l_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_layernorm2_parameters_weight_,
            l_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_layernorm2_parameters_bias_,
            1e-05,
        )
        l_stack0_0_ = l_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_layernorm2_parameters_weight_ = l_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_layernorm2_parameters_bias_ = (None)
        queries = torch._C._nn.linear(
            hidden_states,
            l_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        hidden_states = l_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        keys = torch._C._nn.linear(
            inputs,
            l_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        values = torch._C._nn.linear(
            inputs,
            l_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        inputs = l_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        x = queries.view(1, 10, 1, 20)
        queries = None
        queries_1 = x.permute(0, 2, 1, 3)
        x = None
        x_1 = keys.view(1, 49, 1, 20)
        keys = None
        keys_1 = x_1.permute(0, 2, 1, 3)
        x_1 = None
        x_2 = values.view(1, 49, 1, 20)
        values = None
        values_1 = x_2.permute(0, 2, 1, 3)
        x_2 = None
        transpose = keys_1.transpose(-1, -2)
        keys_1 = None
        attention_scores = torch.matmul(queries_1, transpose)
        queries_1 = transpose = None
        attention_scores_1 = attention_scores / 4.47213595499958
        attention_scores = None
        attention_scores_2 = attention_scores_1 + encoder_extended_attention_mask_2
        attention_scores_1 = encoder_extended_attention_mask_2 = None
        _log_api_usage_once = torch._C._log_api_usage_once("python.nn_module")
        _log_api_usage_once = None
        attention_probs = torch.nn.functional.softmax(
            attention_scores_2, -1, _stacklevel=5
        )
        attention_scores_2 = None
        attention_probs_1 = torch.nn.functional.dropout(
            attention_probs, 0.1, False, False
        )
        attention_probs = None
        context_layer = torch.matmul(attention_probs_1, values_1)
        attention_probs_1 = values_1 = None
        permute_3 = context_layer.permute(0, 2, 1, 3)
        context_layer = None
        context_layer_1 = permute_3.contiguous()
        permute_3 = None
        context_layer_2 = context_layer_1.view(1, 10, 20)
        context_layer_1 = None
        hidden_states_1 = torch._C._nn.linear(
            context_layer_2,
            l_self_modules_encoder_modules_cross_attention_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_cross_attention_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_2 = l_self_modules_encoder_modules_cross_attention_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_cross_attention_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        attention_output = hidden_states_1 + embedding_output
        hidden_states_1 = embedding_output = None
        layer_output = torch.nn.functional.layer_norm(
            attention_output,
            (20,),
            l_self_modules_encoder_modules_cross_attention_modules_layernorm_parameters_weight_,
            l_self_modules_encoder_modules_cross_attention_modules_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_cross_attention_modules_layernorm_parameters_weight_ = l_self_modules_encoder_modules_cross_attention_modules_layernorm_parameters_bias_ = (None)
        hidden_states_2 = torch._C._nn.linear(
            layer_output,
            l_self_modules_encoder_modules_cross_attention_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_cross_attention_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_output = l_self_modules_encoder_modules_cross_attention_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_cross_attention_modules_mlp_modules_dense1_parameters_bias_ = (None)
        hidden_states_3 = torch._C._nn.gelu(hidden_states_2)
        hidden_states_2 = None
        hidden_states_4 = torch._C._nn.linear(
            hidden_states_3,
            l_self_modules_encoder_modules_cross_attention_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_cross_attention_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_3 = l_self_modules_encoder_modules_cross_attention_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_cross_attention_modules_mlp_modules_dense2_parameters_bias_ = (None)
        layer_output_1 = hidden_states_4 + attention_output
        hidden_states_4 = attention_output = None
        hidden_states_5 = torch.nn.functional.layer_norm(
            layer_output_1,
            (20,),
            l_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_self_modules_layernorm1_parameters_weight_,
            l_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_self_modules_layernorm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_self_modules_layernorm1_parameters_weight_ = l_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_self_modules_layernorm1_parameters_bias_ = (None)
        queries_2 = torch._C._nn.linear(
            hidden_states_5,
            l_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        keys_2 = torch._C._nn.linear(
            hidden_states_5,
            l_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        values_2 = torch._C._nn.linear(
            hidden_states_5,
            l_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_5 = l_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        x_3 = queries_2.view(1, 10, 1, 20)
        queries_2 = None
        queries_3 = x_3.permute(0, 2, 1, 3)
        x_3 = None
        x_4 = keys_2.view(1, 10, 1, 20)
        keys_2 = None
        keys_3 = x_4.permute(0, 2, 1, 3)
        x_4 = None
        x_5 = values_2.view(1, 10, 1, 20)
        values_2 = None
        values_3 = x_5.permute(0, 2, 1, 3)
        x_5 = None
        transpose_1 = keys_3.transpose(-1, -2)
        keys_3 = None
        attention_scores_3 = torch.matmul(queries_3, transpose_1)
        queries_3 = transpose_1 = None
        attention_scores_4 = attention_scores_3 / 4.47213595499958
        attention_scores_3 = None
        _log_api_usage_once_1 = torch._C._log_api_usage_once("python.nn_module")
        _log_api_usage_once_1 = None
        attention_probs_2 = torch.nn.functional.softmax(
            attention_scores_4, -1, _stacklevel=5
        )
        attention_scores_4 = None
        attention_probs_3 = torch.nn.functional.dropout(
            attention_probs_2, 0.1, False, False
        )
        attention_probs_2 = None
        context_layer_3 = torch.matmul(attention_probs_3, values_3)
        attention_probs_3 = values_3 = None
        permute_7 = context_layer_3.permute(0, 2, 1, 3)
        context_layer_3 = None
        context_layer_4 = permute_7.contiguous()
        permute_7 = None
        context_layer_5 = context_layer_4.view(1, 10, 20)
        context_layer_4 = None
        hidden_states_6 = torch._C._nn.linear(
            context_layer_5,
            l_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_5 = l_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        attention_output_1 = hidden_states_6 + layer_output_1
        hidden_states_6 = layer_output_1 = None
        layer_output_2 = torch.nn.functional.layer_norm(
            attention_output_1,
            (20,),
            l_self_modules_encoder_modules_self_attends_modules_0_modules_layernorm_parameters_weight_,
            l_self_modules_encoder_modules_self_attends_modules_0_modules_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_self_attends_modules_0_modules_layernorm_parameters_weight_ = l_self_modules_encoder_modules_self_attends_modules_0_modules_layernorm_parameters_bias_ = (None)
        hidden_states_7 = torch._C._nn.linear(
            layer_output_2,
            l_self_modules_encoder_modules_self_attends_modules_0_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_self_attends_modules_0_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_output_2 = l_self_modules_encoder_modules_self_attends_modules_0_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_self_attends_modules_0_modules_mlp_modules_dense1_parameters_bias_ = (None)
        hidden_states_8 = torch._C._nn.gelu(hidden_states_7)
        hidden_states_7 = None
        hidden_states_9 = torch._C._nn.linear(
            hidden_states_8,
            l_self_modules_encoder_modules_self_attends_modules_0_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_self_attends_modules_0_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_8 = l_self_modules_encoder_modules_self_attends_modules_0_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_self_attends_modules_0_modules_mlp_modules_dense2_parameters_bias_ = (None)
        layer_output_3 = hidden_states_9 + attention_output_1
        hidden_states_9 = attention_output_1 = None
        hidden_states_10 = torch.nn.functional.layer_norm(
            layer_output_3,
            (20,),
            l_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_self_modules_layernorm1_parameters_weight_,
            l_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_self_modules_layernorm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_self_modules_layernorm1_parameters_weight_ = l_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_self_modules_layernorm1_parameters_bias_ = (None)
        queries_4 = torch._C._nn.linear(
            hidden_states_10,
            l_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        keys_4 = torch._C._nn.linear(
            hidden_states_10,
            l_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        values_4 = torch._C._nn.linear(
            hidden_states_10,
            l_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_10 = l_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        x_6 = queries_4.view(1, 10, 1, 20)
        queries_4 = None
        queries_5 = x_6.permute(0, 2, 1, 3)
        x_6 = None
        x_7 = keys_4.view(1, 10, 1, 20)
        keys_4 = None
        keys_5 = x_7.permute(0, 2, 1, 3)
        x_7 = None
        x_8 = values_4.view(1, 10, 1, 20)
        values_4 = None
        values_5 = x_8.permute(0, 2, 1, 3)
        x_8 = None
        transpose_2 = keys_5.transpose(-1, -2)
        keys_5 = None
        attention_scores_5 = torch.matmul(queries_5, transpose_2)
        queries_5 = transpose_2 = None
        attention_scores_6 = attention_scores_5 / 4.47213595499958
        attention_scores_5 = None
        _log_api_usage_once_2 = torch._C._log_api_usage_once("python.nn_module")
        _log_api_usage_once_2 = None
        attention_probs_4 = torch.nn.functional.softmax(
            attention_scores_6, -1, _stacklevel=5
        )
        attention_scores_6 = None
        attention_probs_5 = torch.nn.functional.dropout(
            attention_probs_4, 0.1, False, False
        )
        attention_probs_4 = None
        context_layer_6 = torch.matmul(attention_probs_5, values_5)
        attention_probs_5 = values_5 = None
        permute_11 = context_layer_6.permute(0, 2, 1, 3)
        context_layer_6 = None
        context_layer_7 = permute_11.contiguous()
        permute_11 = None
        context_layer_8 = context_layer_7.view(1, 10, 20)
        context_layer_7 = None
        hidden_states_11 = torch._C._nn.linear(
            context_layer_8,
            l_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_8 = l_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        attention_output_2 = hidden_states_11 + layer_output_3
        hidden_states_11 = layer_output_3 = None
        layer_output_4 = torch.nn.functional.layer_norm(
            attention_output_2,
            (20,),
            l_self_modules_encoder_modules_self_attends_modules_1_modules_layernorm_parameters_weight_,
            l_self_modules_encoder_modules_self_attends_modules_1_modules_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_self_attends_modules_1_modules_layernorm_parameters_weight_ = l_self_modules_encoder_modules_self_attends_modules_1_modules_layernorm_parameters_bias_ = (None)
        hidden_states_12 = torch._C._nn.linear(
            layer_output_4,
            l_self_modules_encoder_modules_self_attends_modules_1_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_self_attends_modules_1_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_output_4 = l_self_modules_encoder_modules_self_attends_modules_1_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_self_attends_modules_1_modules_mlp_modules_dense1_parameters_bias_ = (None)
        hidden_states_13 = torch._C._nn.gelu(hidden_states_12)
        hidden_states_12 = None
        hidden_states_14 = torch._C._nn.linear(
            hidden_states_13,
            l_self_modules_encoder_modules_self_attends_modules_1_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_self_attends_modules_1_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_13 = l_self_modules_encoder_modules_self_attends_modules_1_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_self_attends_modules_1_modules_mlp_modules_dense2_parameters_bias_ = (None)
        layer_output_5 = hidden_states_14 + attention_output_2
        hidden_states_14 = attention_output_2 = None
        position_embeddings = l_self_modules_decoder_modules_decoder_modules_output_position_encodings_parameters_position_embeddings_.expand(
            1, -1, -1
        )
        l_self_modules_decoder_modules_decoder_modules_output_position_encodings_parameters_position_embeddings_ = (
            None
        )
        hidden_states_15 = torch.nn.functional.layer_norm(
            position_embeddings,
            (20,),
            l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_layernorm1_parameters_weight_,
            l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_layernorm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_layernorm1_parameters_weight_ = l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_layernorm1_parameters_bias_ = (None)
        inputs_1 = torch.nn.functional.layer_norm(
            layer_output_5,
            (20,),
            l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_layernorm2_parameters_weight_,
            l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_layernorm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_layernorm2_parameters_weight_ = l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_layernorm2_parameters_bias_ = (None)
        queries_6 = torch._C._nn.linear(
            hidden_states_15,
            l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        hidden_states_15 = l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        keys_6 = torch._C._nn.linear(
            inputs_1,
            l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        values_6 = torch._C._nn.linear(
            inputs_1,
            l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        inputs_1 = l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        x_9 = queries_6.view(1, 1, 1, 20)
        queries_6 = None
        queries_7 = x_9.permute(0, 2, 1, 3)
        x_9 = None
        x_10 = keys_6.view(1, 10, 1, 20)
        keys_6 = None
        keys_7 = x_10.permute(0, 2, 1, 3)
        x_10 = None
        x_11 = values_6.view(1, 10, 1, 20)
        values_6 = None
        values_7 = x_11.permute(0, 2, 1, 3)
        x_11 = None
        transpose_3 = keys_7.transpose(-1, -2)
        keys_7 = None
        attention_scores_7 = torch.matmul(queries_7, transpose_3)
        queries_7 = transpose_3 = None
        attention_scores_8 = attention_scores_7 / 4.47213595499958
        attention_scores_7 = None
        _log_api_usage_once_3 = torch._C._log_api_usage_once("python.nn_module")
        _log_api_usage_once_3 = None
        attention_probs_6 = torch.nn.functional.softmax(
            attention_scores_8, -1, _stacklevel=5
        )
        attention_scores_8 = None
        attention_probs_7 = torch.nn.functional.dropout(
            attention_probs_6, 0.1, False, False
        )
        attention_probs_6 = None
        context_layer_9 = torch.matmul(attention_probs_7, values_7)
        attention_probs_7 = values_7 = None
        permute_15 = context_layer_9.permute(0, 2, 1, 3)
        context_layer_9 = None
        context_layer_10 = permute_15.contiguous()
        permute_15 = None
        context_layer_11 = context_layer_10.view(1, 1, 20)
        context_layer_10 = None
        hidden_states_16 = torch._C._nn.linear(
            context_layer_11,
            l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_11 = l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        attention_output_3 = hidden_states_16 + position_embeddings
        hidden_states_16 = position_embeddings = None
        layer_output_6 = torch.nn.functional.layer_norm(
            attention_output_3,
            (20,),
            l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_layernorm_parameters_weight_,
            l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_layernorm_parameters_weight_ = l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_layernorm_parameters_bias_ = (None)
        hidden_states_17 = torch._C._nn.linear(
            layer_output_6,
            l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_output_6 = l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_mlp_modules_dense1_parameters_bias_ = (None)
        hidden_states_18 = torch._C._nn.gelu(hidden_states_17)
        hidden_states_17 = None
        hidden_states_19 = torch._C._nn.linear(
            hidden_states_18,
            l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_18 = l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_mlp_modules_dense2_parameters_bias_ = (None)
        layer_output_7 = hidden_states_19 + attention_output_3
        hidden_states_19 = attention_output_3 = None
        logits = torch._C._nn.linear(
            layer_output_7,
            l_self_modules_decoder_modules_decoder_modules_final_layer_parameters_weight_,
            l_self_modules_decoder_modules_decoder_modules_final_layer_parameters_bias_,
        )
        layer_output_7 = l_self_modules_decoder_modules_decoder_modules_final_layer_parameters_weight_ = (
            l_self_modules_decoder_modules_decoder_modules_final_layer_parameters_bias_
        ) = None
        logits_1 = logits[(slice(None, None, None), 0, slice(None, None, None))]
        logits = None
        return (logits_1, layer_output_5)
