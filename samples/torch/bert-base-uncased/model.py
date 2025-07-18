
class GraphModule(torch.nn.Module):

    def forward(self, p_bert_embeddings_word_embeddings_weight, p_bert_embeddings_token_type_embeddings_weight, p_bert_embeddings_position_embeddings_weight, p_bert_embeddings_layernorm_weight, p_bert_embeddings_layernorm_bias, p_bert_encoder_layer_0_attention_self_query_weight, p_bert_encoder_layer_0_attention_self_query_bias, p_bert_encoder_layer_0_attention_self_key_weight, p_bert_encoder_layer_0_attention_self_key_bias, p_bert_encoder_layer_0_attention_self_value_weight, p_bert_encoder_layer_0_attention_self_value_bias, p_bert_encoder_layer_0_attention_output_dense_weight, p_bert_encoder_layer_0_attention_output_dense_bias, p_bert_encoder_layer_0_attention_output_layernorm_weight, p_bert_encoder_layer_0_attention_output_layernorm_bias, p_bert_encoder_layer_0_intermediate_dense_weight, p_bert_encoder_layer_0_intermediate_dense_bias, p_bert_encoder_layer_0_output_dense_weight, p_bert_encoder_layer_0_output_dense_bias, p_bert_encoder_layer_0_output_layernorm_weight, p_bert_encoder_layer_0_output_layernorm_bias, p_bert_encoder_layer_1_attention_self_query_weight, p_bert_encoder_layer_1_attention_self_query_bias, p_bert_encoder_layer_1_attention_self_key_weight, p_bert_encoder_layer_1_attention_self_key_bias, p_bert_encoder_layer_1_attention_self_value_weight, p_bert_encoder_layer_1_attention_self_value_bias, p_bert_encoder_layer_1_attention_output_dense_weight, p_bert_encoder_layer_1_attention_output_dense_bias, p_bert_encoder_layer_1_attention_output_layernorm_weight, p_bert_encoder_layer_1_attention_output_layernorm_bias, p_bert_encoder_layer_1_intermediate_dense_weight, p_bert_encoder_layer_1_intermediate_dense_bias, p_bert_encoder_layer_1_output_dense_weight, p_bert_encoder_layer_1_output_dense_bias, p_bert_encoder_layer_1_output_layernorm_weight, p_bert_encoder_layer_1_output_layernorm_bias, p_bert_encoder_layer_2_attention_self_query_weight, p_bert_encoder_layer_2_attention_self_query_bias, p_bert_encoder_layer_2_attention_self_key_weight, p_bert_encoder_layer_2_attention_self_key_bias, p_bert_encoder_layer_2_attention_self_value_weight, p_bert_encoder_layer_2_attention_self_value_bias, p_bert_encoder_layer_2_attention_output_dense_weight, p_bert_encoder_layer_2_attention_output_dense_bias, p_bert_encoder_layer_2_attention_output_layernorm_weight, p_bert_encoder_layer_2_attention_output_layernorm_bias, p_bert_encoder_layer_2_intermediate_dense_weight, p_bert_encoder_layer_2_intermediate_dense_bias, p_bert_encoder_layer_2_output_dense_weight, p_bert_encoder_layer_2_output_dense_bias, p_bert_encoder_layer_2_output_layernorm_weight, p_bert_encoder_layer_2_output_layernorm_bias, p_bert_encoder_layer_3_attention_self_query_weight, p_bert_encoder_layer_3_attention_self_query_bias, p_bert_encoder_layer_3_attention_self_key_weight, p_bert_encoder_layer_3_attention_self_key_bias, p_bert_encoder_layer_3_attention_self_value_weight, p_bert_encoder_layer_3_attention_self_value_bias, p_bert_encoder_layer_3_attention_output_dense_weight, p_bert_encoder_layer_3_attention_output_dense_bias, p_bert_encoder_layer_3_attention_output_layernorm_weight, p_bert_encoder_layer_3_attention_output_layernorm_bias, p_bert_encoder_layer_3_intermediate_dense_weight, p_bert_encoder_layer_3_intermediate_dense_bias, p_bert_encoder_layer_3_output_dense_weight, p_bert_encoder_layer_3_output_dense_bias, p_bert_encoder_layer_3_output_layernorm_weight, p_bert_encoder_layer_3_output_layernorm_bias, p_bert_encoder_layer_4_attention_self_query_weight, p_bert_encoder_layer_4_attention_self_query_bias, p_bert_encoder_layer_4_attention_self_key_weight, p_bert_encoder_layer_4_attention_self_key_bias, p_bert_encoder_layer_4_attention_self_value_weight, p_bert_encoder_layer_4_attention_self_value_bias, p_bert_encoder_layer_4_attention_output_dense_weight, p_bert_encoder_layer_4_attention_output_dense_bias, p_bert_encoder_layer_4_attention_output_layernorm_weight, p_bert_encoder_layer_4_attention_output_layernorm_bias, p_bert_encoder_layer_4_intermediate_dense_weight, p_bert_encoder_layer_4_intermediate_dense_bias, p_bert_encoder_layer_4_output_dense_weight, p_bert_encoder_layer_4_output_dense_bias, p_bert_encoder_layer_4_output_layernorm_weight, p_bert_encoder_layer_4_output_layernorm_bias, p_bert_encoder_layer_5_attention_self_query_weight, p_bert_encoder_layer_5_attention_self_query_bias, p_bert_encoder_layer_5_attention_self_key_weight, p_bert_encoder_layer_5_attention_self_key_bias, p_bert_encoder_layer_5_attention_self_value_weight, p_bert_encoder_layer_5_attention_self_value_bias, p_bert_encoder_layer_5_attention_output_dense_weight, p_bert_encoder_layer_5_attention_output_dense_bias, p_bert_encoder_layer_5_attention_output_layernorm_weight, p_bert_encoder_layer_5_attention_output_layernorm_bias, p_bert_encoder_layer_5_intermediate_dense_weight, p_bert_encoder_layer_5_intermediate_dense_bias, p_bert_encoder_layer_5_output_dense_weight, p_bert_encoder_layer_5_output_dense_bias, p_bert_encoder_layer_5_output_layernorm_weight, p_bert_encoder_layer_5_output_layernorm_bias, p_bert_encoder_layer_6_attention_self_query_weight, p_bert_encoder_layer_6_attention_self_query_bias, p_bert_encoder_layer_6_attention_self_key_weight, p_bert_encoder_layer_6_attention_self_key_bias, p_bert_encoder_layer_6_attention_self_value_weight, p_bert_encoder_layer_6_attention_self_value_bias, p_bert_encoder_layer_6_attention_output_dense_weight, p_bert_encoder_layer_6_attention_output_dense_bias, p_bert_encoder_layer_6_attention_output_layernorm_weight, p_bert_encoder_layer_6_attention_output_layernorm_bias, p_bert_encoder_layer_6_intermediate_dense_weight, p_bert_encoder_layer_6_intermediate_dense_bias, p_bert_encoder_layer_6_output_dense_weight, p_bert_encoder_layer_6_output_dense_bias, p_bert_encoder_layer_6_output_layernorm_weight, p_bert_encoder_layer_6_output_layernorm_bias, p_bert_encoder_layer_7_attention_self_query_weight, p_bert_encoder_layer_7_attention_self_query_bias, p_bert_encoder_layer_7_attention_self_key_weight, p_bert_encoder_layer_7_attention_self_key_bias, p_bert_encoder_layer_7_attention_self_value_weight, p_bert_encoder_layer_7_attention_self_value_bias, p_bert_encoder_layer_7_attention_output_dense_weight, p_bert_encoder_layer_7_attention_output_dense_bias, p_bert_encoder_layer_7_attention_output_layernorm_weight, p_bert_encoder_layer_7_attention_output_layernorm_bias, p_bert_encoder_layer_7_intermediate_dense_weight, p_bert_encoder_layer_7_intermediate_dense_bias, p_bert_encoder_layer_7_output_dense_weight, p_bert_encoder_layer_7_output_dense_bias, p_bert_encoder_layer_7_output_layernorm_weight, p_bert_encoder_layer_7_output_layernorm_bias, p_bert_encoder_layer_8_attention_self_query_weight, p_bert_encoder_layer_8_attention_self_query_bias, p_bert_encoder_layer_8_attention_self_key_weight, p_bert_encoder_layer_8_attention_self_key_bias, p_bert_encoder_layer_8_attention_self_value_weight, p_bert_encoder_layer_8_attention_self_value_bias, p_bert_encoder_layer_8_attention_output_dense_weight, p_bert_encoder_layer_8_attention_output_dense_bias, p_bert_encoder_layer_8_attention_output_layernorm_weight, p_bert_encoder_layer_8_attention_output_layernorm_bias, p_bert_encoder_layer_8_intermediate_dense_weight, p_bert_encoder_layer_8_intermediate_dense_bias, p_bert_encoder_layer_8_output_dense_weight, p_bert_encoder_layer_8_output_dense_bias, p_bert_encoder_layer_8_output_layernorm_weight, p_bert_encoder_layer_8_output_layernorm_bias, p_bert_encoder_layer_9_attention_self_query_weight, p_bert_encoder_layer_9_attention_self_query_bias, p_bert_encoder_layer_9_attention_self_key_weight, p_bert_encoder_layer_9_attention_self_key_bias, p_bert_encoder_layer_9_attention_self_value_weight, p_bert_encoder_layer_9_attention_self_value_bias, p_bert_encoder_layer_9_attention_output_dense_weight, p_bert_encoder_layer_9_attention_output_dense_bias, p_bert_encoder_layer_9_attention_output_layernorm_weight, p_bert_encoder_layer_9_attention_output_layernorm_bias, p_bert_encoder_layer_9_intermediate_dense_weight, p_bert_encoder_layer_9_intermediate_dense_bias, p_bert_encoder_layer_9_output_dense_weight, p_bert_encoder_layer_9_output_dense_bias, p_bert_encoder_layer_9_output_layernorm_weight, p_bert_encoder_layer_9_output_layernorm_bias, p_bert_encoder_layer_10_attention_self_query_weight, p_bert_encoder_layer_10_attention_self_query_bias, p_bert_encoder_layer_10_attention_self_key_weight, p_bert_encoder_layer_10_attention_self_key_bias, p_bert_encoder_layer_10_attention_self_value_weight, p_bert_encoder_layer_10_attention_self_value_bias, p_bert_encoder_layer_10_attention_output_dense_weight, p_bert_encoder_layer_10_attention_output_dense_bias, p_bert_encoder_layer_10_attention_output_layernorm_weight, p_bert_encoder_layer_10_attention_output_layernorm_bias, p_bert_encoder_layer_10_intermediate_dense_weight, p_bert_encoder_layer_10_intermediate_dense_bias, p_bert_encoder_layer_10_output_dense_weight, p_bert_encoder_layer_10_output_dense_bias, p_bert_encoder_layer_10_output_layernorm_weight, p_bert_encoder_layer_10_output_layernorm_bias, p_bert_encoder_layer_11_attention_self_query_weight, p_bert_encoder_layer_11_attention_self_query_bias, p_bert_encoder_layer_11_attention_self_key_weight, p_bert_encoder_layer_11_attention_self_key_bias, p_bert_encoder_layer_11_attention_self_value_weight, p_bert_encoder_layer_11_attention_self_value_bias, p_bert_encoder_layer_11_attention_output_dense_weight, p_bert_encoder_layer_11_attention_output_dense_bias, p_bert_encoder_layer_11_attention_output_layernorm_weight, p_bert_encoder_layer_11_attention_output_layernorm_bias, p_bert_encoder_layer_11_intermediate_dense_weight, p_bert_encoder_layer_11_intermediate_dense_bias, p_bert_encoder_layer_11_output_dense_weight, p_bert_encoder_layer_11_output_dense_bias, p_bert_encoder_layer_11_output_layernorm_weight, p_bert_encoder_layer_11_output_layernorm_bias, p_bert_pooler_dense_weight, p_bert_pooler_dense_bias, p_classifier_weight, p_classifier_bias, b_bert_embeddings_position_ids, input_ids, token_type_ids, attention_mask):
        slice_1 = torch.ops.aten.slice.Tensor(b_bert_embeddings_position_ids, 0, 0, 9223372036854775807);  b_bert_embeddings_position_ids = None
        slice_2 = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 36);  slice_1 = None
        embedding = torch.ops.aten.embedding.default(p_bert_embeddings_word_embeddings_weight, input_ids, 0);  p_bert_embeddings_word_embeddings_weight = input_ids = None
        embedding_1 = torch.ops.aten.embedding.default(p_bert_embeddings_token_type_embeddings_weight, token_type_ids);  p_bert_embeddings_token_type_embeddings_weight = token_type_ids = None
        add = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        embedding_2 = torch.ops.aten.embedding.default(p_bert_embeddings_position_embeddings_weight, slice_2);  p_bert_embeddings_position_embeddings_weight = slice_2 = None
        add_ = torch.ops.aten.add_.Tensor(add, embedding_2);  add = embedding_2 = None
        layer_norm = torch.ops.aten.layer_norm.default(add_, [768], p_bert_embeddings_layernorm_weight, p_bert_embeddings_layernorm_bias, 1e-12);  add_ = p_bert_embeddings_layernorm_weight = p_bert_embeddings_layernorm_bias = None
        dropout = torch.ops.aten.dropout.default(layer_norm, 0.1, False);  layer_norm = None
        slice_3 = torch.ops.aten.slice.Tensor(attention_mask, 0, 0, 9223372036854775807);  attention_mask = None
        unsqueeze = torch.ops.aten.unsqueeze.default(slice_3, 1);  slice_3 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        slice_4 = torch.ops.aten.slice.Tensor(unsqueeze_1, 3, 0, 9223372036854775807);  unsqueeze_1 = None
        expand = torch.ops.aten.expand.default(slice_4, [1, 1, 36, 36]);  slice_4 = None
        to = torch.ops.aten.to.dtype(expand, torch.float32);  expand = None
        rsub = torch.ops.aten.rsub.Scalar(to, 1.0);  to = None
        to_1 = torch.ops.aten.to.dtype(rsub, torch.bool)
        masked_fill = torch.ops.aten.masked_fill.Scalar(rsub, to_1, -3.4028234663852886e+38);  rsub = to_1 = None
        linear = torch.ops.aten.linear.default(dropout, p_bert_encoder_layer_0_attention_self_query_weight, p_bert_encoder_layer_0_attention_self_query_bias);  p_bert_encoder_layer_0_attention_self_query_weight = p_bert_encoder_layer_0_attention_self_query_bias = None
        view = torch.ops.aten.view.default(linear, [1, 36, 12, 64]);  linear = None
        permute = torch.ops.aten.permute.default(view, [0, 2, 1, 3]);  view = None
        linear_1 = torch.ops.aten.linear.default(dropout, p_bert_encoder_layer_0_attention_self_key_weight, p_bert_encoder_layer_0_attention_self_key_bias);  p_bert_encoder_layer_0_attention_self_key_weight = p_bert_encoder_layer_0_attention_self_key_bias = None
        view_1 = torch.ops.aten.view.default(linear_1, [1, 36, 12, 64]);  linear_1 = None
        permute_1 = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        linear_2 = torch.ops.aten.linear.default(dropout, p_bert_encoder_layer_0_attention_self_value_weight, p_bert_encoder_layer_0_attention_self_value_bias);  p_bert_encoder_layer_0_attention_self_value_weight = p_bert_encoder_layer_0_attention_self_value_bias = None
        view_2 = torch.ops.aten.view.default(linear_2, [1, 36, 12, 64]);  linear_2 = None
        permute_2 = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        scaled_dot_product_attention = torch.ops.aten.scaled_dot_product_attention.default(permute, permute_1, permute_2, masked_fill);  permute = permute_1 = permute_2 = None
        transpose = torch.ops.aten.transpose.int(scaled_dot_product_attention, 1, 2);  scaled_dot_product_attention = None
        reshape = torch.ops.aten.reshape.default(transpose, [1, 36, 768]);  transpose = None
        linear_3 = torch.ops.aten.linear.default(reshape, p_bert_encoder_layer_0_attention_output_dense_weight, p_bert_encoder_layer_0_attention_output_dense_bias);  reshape = p_bert_encoder_layer_0_attention_output_dense_weight = p_bert_encoder_layer_0_attention_output_dense_bias = None
        dropout_1 = torch.ops.aten.dropout.default(linear_3, 0.1, False);  linear_3 = None
        add_1 = torch.ops.aten.add.Tensor(dropout_1, dropout);  dropout_1 = dropout = None
        layer_norm_1 = torch.ops.aten.layer_norm.default(add_1, [768], p_bert_encoder_layer_0_attention_output_layernorm_weight, p_bert_encoder_layer_0_attention_output_layernorm_bias, 1e-12);  add_1 = p_bert_encoder_layer_0_attention_output_layernorm_weight = p_bert_encoder_layer_0_attention_output_layernorm_bias = None
        linear_4 = torch.ops.aten.linear.default(layer_norm_1, p_bert_encoder_layer_0_intermediate_dense_weight, p_bert_encoder_layer_0_intermediate_dense_bias);  p_bert_encoder_layer_0_intermediate_dense_weight = p_bert_encoder_layer_0_intermediate_dense_bias = None
        gelu = torch.ops.aten.gelu.default(linear_4);  linear_4 = None
        linear_5 = torch.ops.aten.linear.default(gelu, p_bert_encoder_layer_0_output_dense_weight, p_bert_encoder_layer_0_output_dense_bias);  gelu = p_bert_encoder_layer_0_output_dense_weight = p_bert_encoder_layer_0_output_dense_bias = None
        dropout_2 = torch.ops.aten.dropout.default(linear_5, 0.1, False);  linear_5 = None
        add_2 = torch.ops.aten.add.Tensor(dropout_2, layer_norm_1);  dropout_2 = layer_norm_1 = None
        layer_norm_2 = torch.ops.aten.layer_norm.default(add_2, [768], p_bert_encoder_layer_0_output_layernorm_weight, p_bert_encoder_layer_0_output_layernorm_bias, 1e-12);  add_2 = p_bert_encoder_layer_0_output_layernorm_weight = p_bert_encoder_layer_0_output_layernorm_bias = None
        linear_6 = torch.ops.aten.linear.default(layer_norm_2, p_bert_encoder_layer_1_attention_self_query_weight, p_bert_encoder_layer_1_attention_self_query_bias);  p_bert_encoder_layer_1_attention_self_query_weight = p_bert_encoder_layer_1_attention_self_query_bias = None
        view_3 = torch.ops.aten.view.default(linear_6, [1, 36, 12, 64]);  linear_6 = None
        permute_3 = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        linear_7 = torch.ops.aten.linear.default(layer_norm_2, p_bert_encoder_layer_1_attention_self_key_weight, p_bert_encoder_layer_1_attention_self_key_bias);  p_bert_encoder_layer_1_attention_self_key_weight = p_bert_encoder_layer_1_attention_self_key_bias = None
        view_4 = torch.ops.aten.view.default(linear_7, [1, 36, 12, 64]);  linear_7 = None
        permute_4 = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        linear_8 = torch.ops.aten.linear.default(layer_norm_2, p_bert_encoder_layer_1_attention_self_value_weight, p_bert_encoder_layer_1_attention_self_value_bias);  p_bert_encoder_layer_1_attention_self_value_weight = p_bert_encoder_layer_1_attention_self_value_bias = None
        view_5 = torch.ops.aten.view.default(linear_8, [1, 36, 12, 64]);  linear_8 = None
        permute_5 = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        scaled_dot_product_attention_1 = torch.ops.aten.scaled_dot_product_attention.default(permute_3, permute_4, permute_5, masked_fill);  permute_3 = permute_4 = permute_5 = None
        transpose_1 = torch.ops.aten.transpose.int(scaled_dot_product_attention_1, 1, 2);  scaled_dot_product_attention_1 = None
        reshape_1 = torch.ops.aten.reshape.default(transpose_1, [1, 36, 768]);  transpose_1 = None
        linear_9 = torch.ops.aten.linear.default(reshape_1, p_bert_encoder_layer_1_attention_output_dense_weight, p_bert_encoder_layer_1_attention_output_dense_bias);  reshape_1 = p_bert_encoder_layer_1_attention_output_dense_weight = p_bert_encoder_layer_1_attention_output_dense_bias = None
        dropout_3 = torch.ops.aten.dropout.default(linear_9, 0.1, False);  linear_9 = None
        add_3 = torch.ops.aten.add.Tensor(dropout_3, layer_norm_2);  dropout_3 = layer_norm_2 = None
        layer_norm_3 = torch.ops.aten.layer_norm.default(add_3, [768], p_bert_encoder_layer_1_attention_output_layernorm_weight, p_bert_encoder_layer_1_attention_output_layernorm_bias, 1e-12);  add_3 = p_bert_encoder_layer_1_attention_output_layernorm_weight = p_bert_encoder_layer_1_attention_output_layernorm_bias = None
        linear_10 = torch.ops.aten.linear.default(layer_norm_3, p_bert_encoder_layer_1_intermediate_dense_weight, p_bert_encoder_layer_1_intermediate_dense_bias);  p_bert_encoder_layer_1_intermediate_dense_weight = p_bert_encoder_layer_1_intermediate_dense_bias = None
        gelu_1 = torch.ops.aten.gelu.default(linear_10);  linear_10 = None
        linear_11 = torch.ops.aten.linear.default(gelu_1, p_bert_encoder_layer_1_output_dense_weight, p_bert_encoder_layer_1_output_dense_bias);  gelu_1 = p_bert_encoder_layer_1_output_dense_weight = p_bert_encoder_layer_1_output_dense_bias = None
        dropout_4 = torch.ops.aten.dropout.default(linear_11, 0.1, False);  linear_11 = None
        add_4 = torch.ops.aten.add.Tensor(dropout_4, layer_norm_3);  dropout_4 = layer_norm_3 = None
        layer_norm_4 = torch.ops.aten.layer_norm.default(add_4, [768], p_bert_encoder_layer_1_output_layernorm_weight, p_bert_encoder_layer_1_output_layernorm_bias, 1e-12);  add_4 = p_bert_encoder_layer_1_output_layernorm_weight = p_bert_encoder_layer_1_output_layernorm_bias = None
        linear_12 = torch.ops.aten.linear.default(layer_norm_4, p_bert_encoder_layer_2_attention_self_query_weight, p_bert_encoder_layer_2_attention_self_query_bias);  p_bert_encoder_layer_2_attention_self_query_weight = p_bert_encoder_layer_2_attention_self_query_bias = None
        view_6 = torch.ops.aten.view.default(linear_12, [1, 36, 12, 64]);  linear_12 = None
        permute_6 = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        linear_13 = torch.ops.aten.linear.default(layer_norm_4, p_bert_encoder_layer_2_attention_self_key_weight, p_bert_encoder_layer_2_attention_self_key_bias);  p_bert_encoder_layer_2_attention_self_key_weight = p_bert_encoder_layer_2_attention_self_key_bias = None
        view_7 = torch.ops.aten.view.default(linear_13, [1, 36, 12, 64]);  linear_13 = None
        permute_7 = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
        linear_14 = torch.ops.aten.linear.default(layer_norm_4, p_bert_encoder_layer_2_attention_self_value_weight, p_bert_encoder_layer_2_attention_self_value_bias);  p_bert_encoder_layer_2_attention_self_value_weight = p_bert_encoder_layer_2_attention_self_value_bias = None
        view_8 = torch.ops.aten.view.default(linear_14, [1, 36, 12, 64]);  linear_14 = None
        permute_8 = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        scaled_dot_product_attention_2 = torch.ops.aten.scaled_dot_product_attention.default(permute_6, permute_7, permute_8, masked_fill);  permute_6 = permute_7 = permute_8 = None
        transpose_2 = torch.ops.aten.transpose.int(scaled_dot_product_attention_2, 1, 2);  scaled_dot_product_attention_2 = None
        reshape_2 = torch.ops.aten.reshape.default(transpose_2, [1, 36, 768]);  transpose_2 = None
        linear_15 = torch.ops.aten.linear.default(reshape_2, p_bert_encoder_layer_2_attention_output_dense_weight, p_bert_encoder_layer_2_attention_output_dense_bias);  reshape_2 = p_bert_encoder_layer_2_attention_output_dense_weight = p_bert_encoder_layer_2_attention_output_dense_bias = None
        dropout_5 = torch.ops.aten.dropout.default(linear_15, 0.1, False);  linear_15 = None
        add_5 = torch.ops.aten.add.Tensor(dropout_5, layer_norm_4);  dropout_5 = layer_norm_4 = None
        layer_norm_5 = torch.ops.aten.layer_norm.default(add_5, [768], p_bert_encoder_layer_2_attention_output_layernorm_weight, p_bert_encoder_layer_2_attention_output_layernorm_bias, 1e-12);  add_5 = p_bert_encoder_layer_2_attention_output_layernorm_weight = p_bert_encoder_layer_2_attention_output_layernorm_bias = None
        linear_16 = torch.ops.aten.linear.default(layer_norm_5, p_bert_encoder_layer_2_intermediate_dense_weight, p_bert_encoder_layer_2_intermediate_dense_bias);  p_bert_encoder_layer_2_intermediate_dense_weight = p_bert_encoder_layer_2_intermediate_dense_bias = None
        gelu_2 = torch.ops.aten.gelu.default(linear_16);  linear_16 = None
        linear_17 = torch.ops.aten.linear.default(gelu_2, p_bert_encoder_layer_2_output_dense_weight, p_bert_encoder_layer_2_output_dense_bias);  gelu_2 = p_bert_encoder_layer_2_output_dense_weight = p_bert_encoder_layer_2_output_dense_bias = None
        dropout_6 = torch.ops.aten.dropout.default(linear_17, 0.1, False);  linear_17 = None
        add_6 = torch.ops.aten.add.Tensor(dropout_6, layer_norm_5);  dropout_6 = layer_norm_5 = None
        layer_norm_6 = torch.ops.aten.layer_norm.default(add_6, [768], p_bert_encoder_layer_2_output_layernorm_weight, p_bert_encoder_layer_2_output_layernorm_bias, 1e-12);  add_6 = p_bert_encoder_layer_2_output_layernorm_weight = p_bert_encoder_layer_2_output_layernorm_bias = None
        linear_18 = torch.ops.aten.linear.default(layer_norm_6, p_bert_encoder_layer_3_attention_self_query_weight, p_bert_encoder_layer_3_attention_self_query_bias);  p_bert_encoder_layer_3_attention_self_query_weight = p_bert_encoder_layer_3_attention_self_query_bias = None
        view_9 = torch.ops.aten.view.default(linear_18, [1, 36, 12, 64]);  linear_18 = None
        permute_9 = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
        linear_19 = torch.ops.aten.linear.default(layer_norm_6, p_bert_encoder_layer_3_attention_self_key_weight, p_bert_encoder_layer_3_attention_self_key_bias);  p_bert_encoder_layer_3_attention_self_key_weight = p_bert_encoder_layer_3_attention_self_key_bias = None
        view_10 = torch.ops.aten.view.default(linear_19, [1, 36, 12, 64]);  linear_19 = None
        permute_10 = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
        linear_20 = torch.ops.aten.linear.default(layer_norm_6, p_bert_encoder_layer_3_attention_self_value_weight, p_bert_encoder_layer_3_attention_self_value_bias);  p_bert_encoder_layer_3_attention_self_value_weight = p_bert_encoder_layer_3_attention_self_value_bias = None
        view_11 = torch.ops.aten.view.default(linear_20, [1, 36, 12, 64]);  linear_20 = None
        permute_11 = torch.ops.aten.permute.default(view_11, [0, 2, 1, 3]);  view_11 = None
        scaled_dot_product_attention_3 = torch.ops.aten.scaled_dot_product_attention.default(permute_9, permute_10, permute_11, masked_fill);  permute_9 = permute_10 = permute_11 = None
        transpose_3 = torch.ops.aten.transpose.int(scaled_dot_product_attention_3, 1, 2);  scaled_dot_product_attention_3 = None
        reshape_3 = torch.ops.aten.reshape.default(transpose_3, [1, 36, 768]);  transpose_3 = None
        linear_21 = torch.ops.aten.linear.default(reshape_3, p_bert_encoder_layer_3_attention_output_dense_weight, p_bert_encoder_layer_3_attention_output_dense_bias);  reshape_3 = p_bert_encoder_layer_3_attention_output_dense_weight = p_bert_encoder_layer_3_attention_output_dense_bias = None
        dropout_7 = torch.ops.aten.dropout.default(linear_21, 0.1, False);  linear_21 = None
        add_7 = torch.ops.aten.add.Tensor(dropout_7, layer_norm_6);  dropout_7 = layer_norm_6 = None
        layer_norm_7 = torch.ops.aten.layer_norm.default(add_7, [768], p_bert_encoder_layer_3_attention_output_layernorm_weight, p_bert_encoder_layer_3_attention_output_layernorm_bias, 1e-12);  add_7 = p_bert_encoder_layer_3_attention_output_layernorm_weight = p_bert_encoder_layer_3_attention_output_layernorm_bias = None
        linear_22 = torch.ops.aten.linear.default(layer_norm_7, p_bert_encoder_layer_3_intermediate_dense_weight, p_bert_encoder_layer_3_intermediate_dense_bias);  p_bert_encoder_layer_3_intermediate_dense_weight = p_bert_encoder_layer_3_intermediate_dense_bias = None
        gelu_3 = torch.ops.aten.gelu.default(linear_22);  linear_22 = None
        linear_23 = torch.ops.aten.linear.default(gelu_3, p_bert_encoder_layer_3_output_dense_weight, p_bert_encoder_layer_3_output_dense_bias);  gelu_3 = p_bert_encoder_layer_3_output_dense_weight = p_bert_encoder_layer_3_output_dense_bias = None
        dropout_8 = torch.ops.aten.dropout.default(linear_23, 0.1, False);  linear_23 = None
        add_8 = torch.ops.aten.add.Tensor(dropout_8, layer_norm_7);  dropout_8 = layer_norm_7 = None
        layer_norm_8 = torch.ops.aten.layer_norm.default(add_8, [768], p_bert_encoder_layer_3_output_layernorm_weight, p_bert_encoder_layer_3_output_layernorm_bias, 1e-12);  add_8 = p_bert_encoder_layer_3_output_layernorm_weight = p_bert_encoder_layer_3_output_layernorm_bias = None
        linear_24 = torch.ops.aten.linear.default(layer_norm_8, p_bert_encoder_layer_4_attention_self_query_weight, p_bert_encoder_layer_4_attention_self_query_bias);  p_bert_encoder_layer_4_attention_self_query_weight = p_bert_encoder_layer_4_attention_self_query_bias = None
        view_12 = torch.ops.aten.view.default(linear_24, [1, 36, 12, 64]);  linear_24 = None
        permute_12 = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None
        linear_25 = torch.ops.aten.linear.default(layer_norm_8, p_bert_encoder_layer_4_attention_self_key_weight, p_bert_encoder_layer_4_attention_self_key_bias);  p_bert_encoder_layer_4_attention_self_key_weight = p_bert_encoder_layer_4_attention_self_key_bias = None
        view_13 = torch.ops.aten.view.default(linear_25, [1, 36, 12, 64]);  linear_25 = None
        permute_13 = torch.ops.aten.permute.default(view_13, [0, 2, 1, 3]);  view_13 = None
        linear_26 = torch.ops.aten.linear.default(layer_norm_8, p_bert_encoder_layer_4_attention_self_value_weight, p_bert_encoder_layer_4_attention_self_value_bias);  p_bert_encoder_layer_4_attention_self_value_weight = p_bert_encoder_layer_4_attention_self_value_bias = None
        view_14 = torch.ops.aten.view.default(linear_26, [1, 36, 12, 64]);  linear_26 = None
        permute_14 = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
        scaled_dot_product_attention_4 = torch.ops.aten.scaled_dot_product_attention.default(permute_12, permute_13, permute_14, masked_fill);  permute_12 = permute_13 = permute_14 = None
        transpose_4 = torch.ops.aten.transpose.int(scaled_dot_product_attention_4, 1, 2);  scaled_dot_product_attention_4 = None
        reshape_4 = torch.ops.aten.reshape.default(transpose_4, [1, 36, 768]);  transpose_4 = None
        linear_27 = torch.ops.aten.linear.default(reshape_4, p_bert_encoder_layer_4_attention_output_dense_weight, p_bert_encoder_layer_4_attention_output_dense_bias);  reshape_4 = p_bert_encoder_layer_4_attention_output_dense_weight = p_bert_encoder_layer_4_attention_output_dense_bias = None
        dropout_9 = torch.ops.aten.dropout.default(linear_27, 0.1, False);  linear_27 = None
        add_9 = torch.ops.aten.add.Tensor(dropout_9, layer_norm_8);  dropout_9 = layer_norm_8 = None
        layer_norm_9 = torch.ops.aten.layer_norm.default(add_9, [768], p_bert_encoder_layer_4_attention_output_layernorm_weight, p_bert_encoder_layer_4_attention_output_layernorm_bias, 1e-12);  add_9 = p_bert_encoder_layer_4_attention_output_layernorm_weight = p_bert_encoder_layer_4_attention_output_layernorm_bias = None
        linear_28 = torch.ops.aten.linear.default(layer_norm_9, p_bert_encoder_layer_4_intermediate_dense_weight, p_bert_encoder_layer_4_intermediate_dense_bias);  p_bert_encoder_layer_4_intermediate_dense_weight = p_bert_encoder_layer_4_intermediate_dense_bias = None
        gelu_4 = torch.ops.aten.gelu.default(linear_28);  linear_28 = None
        linear_29 = torch.ops.aten.linear.default(gelu_4, p_bert_encoder_layer_4_output_dense_weight, p_bert_encoder_layer_4_output_dense_bias);  gelu_4 = p_bert_encoder_layer_4_output_dense_weight = p_bert_encoder_layer_4_output_dense_bias = None
        dropout_10 = torch.ops.aten.dropout.default(linear_29, 0.1, False);  linear_29 = None
        add_10 = torch.ops.aten.add.Tensor(dropout_10, layer_norm_9);  dropout_10 = layer_norm_9 = None
        layer_norm_10 = torch.ops.aten.layer_norm.default(add_10, [768], p_bert_encoder_layer_4_output_layernorm_weight, p_bert_encoder_layer_4_output_layernorm_bias, 1e-12);  add_10 = p_bert_encoder_layer_4_output_layernorm_weight = p_bert_encoder_layer_4_output_layernorm_bias = None
        linear_30 = torch.ops.aten.linear.default(layer_norm_10, p_bert_encoder_layer_5_attention_self_query_weight, p_bert_encoder_layer_5_attention_self_query_bias);  p_bert_encoder_layer_5_attention_self_query_weight = p_bert_encoder_layer_5_attention_self_query_bias = None
        view_15 = torch.ops.aten.view.default(linear_30, [1, 36, 12, 64]);  linear_30 = None
        permute_15 = torch.ops.aten.permute.default(view_15, [0, 2, 1, 3]);  view_15 = None
        linear_31 = torch.ops.aten.linear.default(layer_norm_10, p_bert_encoder_layer_5_attention_self_key_weight, p_bert_encoder_layer_5_attention_self_key_bias);  p_bert_encoder_layer_5_attention_self_key_weight = p_bert_encoder_layer_5_attention_self_key_bias = None
        view_16 = torch.ops.aten.view.default(linear_31, [1, 36, 12, 64]);  linear_31 = None
        permute_16 = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
        linear_32 = torch.ops.aten.linear.default(layer_norm_10, p_bert_encoder_layer_5_attention_self_value_weight, p_bert_encoder_layer_5_attention_self_value_bias);  p_bert_encoder_layer_5_attention_self_value_weight = p_bert_encoder_layer_5_attention_self_value_bias = None
        view_17 = torch.ops.aten.view.default(linear_32, [1, 36, 12, 64]);  linear_32 = None
        permute_17 = torch.ops.aten.permute.default(view_17, [0, 2, 1, 3]);  view_17 = None
        scaled_dot_product_attention_5 = torch.ops.aten.scaled_dot_product_attention.default(permute_15, permute_16, permute_17, masked_fill);  permute_15 = permute_16 = permute_17 = None
        transpose_5 = torch.ops.aten.transpose.int(scaled_dot_product_attention_5, 1, 2);  scaled_dot_product_attention_5 = None
        reshape_5 = torch.ops.aten.reshape.default(transpose_5, [1, 36, 768]);  transpose_5 = None
        linear_33 = torch.ops.aten.linear.default(reshape_5, p_bert_encoder_layer_5_attention_output_dense_weight, p_bert_encoder_layer_5_attention_output_dense_bias);  reshape_5 = p_bert_encoder_layer_5_attention_output_dense_weight = p_bert_encoder_layer_5_attention_output_dense_bias = None
        dropout_11 = torch.ops.aten.dropout.default(linear_33, 0.1, False);  linear_33 = None
        add_11 = torch.ops.aten.add.Tensor(dropout_11, layer_norm_10);  dropout_11 = layer_norm_10 = None
        layer_norm_11 = torch.ops.aten.layer_norm.default(add_11, [768], p_bert_encoder_layer_5_attention_output_layernorm_weight, p_bert_encoder_layer_5_attention_output_layernorm_bias, 1e-12);  add_11 = p_bert_encoder_layer_5_attention_output_layernorm_weight = p_bert_encoder_layer_5_attention_output_layernorm_bias = None
        linear_34 = torch.ops.aten.linear.default(layer_norm_11, p_bert_encoder_layer_5_intermediate_dense_weight, p_bert_encoder_layer_5_intermediate_dense_bias);  p_bert_encoder_layer_5_intermediate_dense_weight = p_bert_encoder_layer_5_intermediate_dense_bias = None
        gelu_5 = torch.ops.aten.gelu.default(linear_34);  linear_34 = None
        linear_35 = torch.ops.aten.linear.default(gelu_5, p_bert_encoder_layer_5_output_dense_weight, p_bert_encoder_layer_5_output_dense_bias);  gelu_5 = p_bert_encoder_layer_5_output_dense_weight = p_bert_encoder_layer_5_output_dense_bias = None
        dropout_12 = torch.ops.aten.dropout.default(linear_35, 0.1, False);  linear_35 = None
        add_12 = torch.ops.aten.add.Tensor(dropout_12, layer_norm_11);  dropout_12 = layer_norm_11 = None
        layer_norm_12 = torch.ops.aten.layer_norm.default(add_12, [768], p_bert_encoder_layer_5_output_layernorm_weight, p_bert_encoder_layer_5_output_layernorm_bias, 1e-12);  add_12 = p_bert_encoder_layer_5_output_layernorm_weight = p_bert_encoder_layer_5_output_layernorm_bias = None
        linear_36 = torch.ops.aten.linear.default(layer_norm_12, p_bert_encoder_layer_6_attention_self_query_weight, p_bert_encoder_layer_6_attention_self_query_bias);  p_bert_encoder_layer_6_attention_self_query_weight = p_bert_encoder_layer_6_attention_self_query_bias = None
        view_18 = torch.ops.aten.view.default(linear_36, [1, 36, 12, 64]);  linear_36 = None
        permute_18 = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3]);  view_18 = None
        linear_37 = torch.ops.aten.linear.default(layer_norm_12, p_bert_encoder_layer_6_attention_self_key_weight, p_bert_encoder_layer_6_attention_self_key_bias);  p_bert_encoder_layer_6_attention_self_key_weight = p_bert_encoder_layer_6_attention_self_key_bias = None
        view_19 = torch.ops.aten.view.default(linear_37, [1, 36, 12, 64]);  linear_37 = None
        permute_19 = torch.ops.aten.permute.default(view_19, [0, 2, 1, 3]);  view_19 = None
        linear_38 = torch.ops.aten.linear.default(layer_norm_12, p_bert_encoder_layer_6_attention_self_value_weight, p_bert_encoder_layer_6_attention_self_value_bias);  p_bert_encoder_layer_6_attention_self_value_weight = p_bert_encoder_layer_6_attention_self_value_bias = None
        view_20 = torch.ops.aten.view.default(linear_38, [1, 36, 12, 64]);  linear_38 = None
        permute_20 = torch.ops.aten.permute.default(view_20, [0, 2, 1, 3]);  view_20 = None
        scaled_dot_product_attention_6 = torch.ops.aten.scaled_dot_product_attention.default(permute_18, permute_19, permute_20, masked_fill);  permute_18 = permute_19 = permute_20 = None
        transpose_6 = torch.ops.aten.transpose.int(scaled_dot_product_attention_6, 1, 2);  scaled_dot_product_attention_6 = None
        reshape_6 = torch.ops.aten.reshape.default(transpose_6, [1, 36, 768]);  transpose_6 = None
        linear_39 = torch.ops.aten.linear.default(reshape_6, p_bert_encoder_layer_6_attention_output_dense_weight, p_bert_encoder_layer_6_attention_output_dense_bias);  reshape_6 = p_bert_encoder_layer_6_attention_output_dense_weight = p_bert_encoder_layer_6_attention_output_dense_bias = None
        dropout_13 = torch.ops.aten.dropout.default(linear_39, 0.1, False);  linear_39 = None
        add_13 = torch.ops.aten.add.Tensor(dropout_13, layer_norm_12);  dropout_13 = layer_norm_12 = None
        layer_norm_13 = torch.ops.aten.layer_norm.default(add_13, [768], p_bert_encoder_layer_6_attention_output_layernorm_weight, p_bert_encoder_layer_6_attention_output_layernorm_bias, 1e-12);  add_13 = p_bert_encoder_layer_6_attention_output_layernorm_weight = p_bert_encoder_layer_6_attention_output_layernorm_bias = None
        linear_40 = torch.ops.aten.linear.default(layer_norm_13, p_bert_encoder_layer_6_intermediate_dense_weight, p_bert_encoder_layer_6_intermediate_dense_bias);  p_bert_encoder_layer_6_intermediate_dense_weight = p_bert_encoder_layer_6_intermediate_dense_bias = None
        gelu_6 = torch.ops.aten.gelu.default(linear_40);  linear_40 = None
        linear_41 = torch.ops.aten.linear.default(gelu_6, p_bert_encoder_layer_6_output_dense_weight, p_bert_encoder_layer_6_output_dense_bias);  gelu_6 = p_bert_encoder_layer_6_output_dense_weight = p_bert_encoder_layer_6_output_dense_bias = None
        dropout_14 = torch.ops.aten.dropout.default(linear_41, 0.1, False);  linear_41 = None
        add_14 = torch.ops.aten.add.Tensor(dropout_14, layer_norm_13);  dropout_14 = layer_norm_13 = None
        layer_norm_14 = torch.ops.aten.layer_norm.default(add_14, [768], p_bert_encoder_layer_6_output_layernorm_weight, p_bert_encoder_layer_6_output_layernorm_bias, 1e-12);  add_14 = p_bert_encoder_layer_6_output_layernorm_weight = p_bert_encoder_layer_6_output_layernorm_bias = None
        linear_42 = torch.ops.aten.linear.default(layer_norm_14, p_bert_encoder_layer_7_attention_self_query_weight, p_bert_encoder_layer_7_attention_self_query_bias);  p_bert_encoder_layer_7_attention_self_query_weight = p_bert_encoder_layer_7_attention_self_query_bias = None
        view_21 = torch.ops.aten.view.default(linear_42, [1, 36, 12, 64]);  linear_42 = None
        permute_21 = torch.ops.aten.permute.default(view_21, [0, 2, 1, 3]);  view_21 = None
        linear_43 = torch.ops.aten.linear.default(layer_norm_14, p_bert_encoder_layer_7_attention_self_key_weight, p_bert_encoder_layer_7_attention_self_key_bias);  p_bert_encoder_layer_7_attention_self_key_weight = p_bert_encoder_layer_7_attention_self_key_bias = None
        view_22 = torch.ops.aten.view.default(linear_43, [1, 36, 12, 64]);  linear_43 = None
        permute_22 = torch.ops.aten.permute.default(view_22, [0, 2, 1, 3]);  view_22 = None
        linear_44 = torch.ops.aten.linear.default(layer_norm_14, p_bert_encoder_layer_7_attention_self_value_weight, p_bert_encoder_layer_7_attention_self_value_bias);  p_bert_encoder_layer_7_attention_self_value_weight = p_bert_encoder_layer_7_attention_self_value_bias = None
        view_23 = torch.ops.aten.view.default(linear_44, [1, 36, 12, 64]);  linear_44 = None
        permute_23 = torch.ops.aten.permute.default(view_23, [0, 2, 1, 3]);  view_23 = None
        scaled_dot_product_attention_7 = torch.ops.aten.scaled_dot_product_attention.default(permute_21, permute_22, permute_23, masked_fill);  permute_21 = permute_22 = permute_23 = None
        transpose_7 = torch.ops.aten.transpose.int(scaled_dot_product_attention_7, 1, 2);  scaled_dot_product_attention_7 = None
        reshape_7 = torch.ops.aten.reshape.default(transpose_7, [1, 36, 768]);  transpose_7 = None
        linear_45 = torch.ops.aten.linear.default(reshape_7, p_bert_encoder_layer_7_attention_output_dense_weight, p_bert_encoder_layer_7_attention_output_dense_bias);  reshape_7 = p_bert_encoder_layer_7_attention_output_dense_weight = p_bert_encoder_layer_7_attention_output_dense_bias = None
        dropout_15 = torch.ops.aten.dropout.default(linear_45, 0.1, False);  linear_45 = None
        add_15 = torch.ops.aten.add.Tensor(dropout_15, layer_norm_14);  dropout_15 = layer_norm_14 = None
        layer_norm_15 = torch.ops.aten.layer_norm.default(add_15, [768], p_bert_encoder_layer_7_attention_output_layernorm_weight, p_bert_encoder_layer_7_attention_output_layernorm_bias, 1e-12);  add_15 = p_bert_encoder_layer_7_attention_output_layernorm_weight = p_bert_encoder_layer_7_attention_output_layernorm_bias = None
        linear_46 = torch.ops.aten.linear.default(layer_norm_15, p_bert_encoder_layer_7_intermediate_dense_weight, p_bert_encoder_layer_7_intermediate_dense_bias);  p_bert_encoder_layer_7_intermediate_dense_weight = p_bert_encoder_layer_7_intermediate_dense_bias = None
        gelu_7 = torch.ops.aten.gelu.default(linear_46);  linear_46 = None
        linear_47 = torch.ops.aten.linear.default(gelu_7, p_bert_encoder_layer_7_output_dense_weight, p_bert_encoder_layer_7_output_dense_bias);  gelu_7 = p_bert_encoder_layer_7_output_dense_weight = p_bert_encoder_layer_7_output_dense_bias = None
        dropout_16 = torch.ops.aten.dropout.default(linear_47, 0.1, False);  linear_47 = None
        add_16 = torch.ops.aten.add.Tensor(dropout_16, layer_norm_15);  dropout_16 = layer_norm_15 = None
        layer_norm_16 = torch.ops.aten.layer_norm.default(add_16, [768], p_bert_encoder_layer_7_output_layernorm_weight, p_bert_encoder_layer_7_output_layernorm_bias, 1e-12);  add_16 = p_bert_encoder_layer_7_output_layernorm_weight = p_bert_encoder_layer_7_output_layernorm_bias = None
        linear_48 = torch.ops.aten.linear.default(layer_norm_16, p_bert_encoder_layer_8_attention_self_query_weight, p_bert_encoder_layer_8_attention_self_query_bias);  p_bert_encoder_layer_8_attention_self_query_weight = p_bert_encoder_layer_8_attention_self_query_bias = None
        view_24 = torch.ops.aten.view.default(linear_48, [1, 36, 12, 64]);  linear_48 = None
        permute_24 = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
        linear_49 = torch.ops.aten.linear.default(layer_norm_16, p_bert_encoder_layer_8_attention_self_key_weight, p_bert_encoder_layer_8_attention_self_key_bias);  p_bert_encoder_layer_8_attention_self_key_weight = p_bert_encoder_layer_8_attention_self_key_bias = None
        view_25 = torch.ops.aten.view.default(linear_49, [1, 36, 12, 64]);  linear_49 = None
        permute_25 = torch.ops.aten.permute.default(view_25, [0, 2, 1, 3]);  view_25 = None
        linear_50 = torch.ops.aten.linear.default(layer_norm_16, p_bert_encoder_layer_8_attention_self_value_weight, p_bert_encoder_layer_8_attention_self_value_bias);  p_bert_encoder_layer_8_attention_self_value_weight = p_bert_encoder_layer_8_attention_self_value_bias = None
        view_26 = torch.ops.aten.view.default(linear_50, [1, 36, 12, 64]);  linear_50 = None
        permute_26 = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None
        scaled_dot_product_attention_8 = torch.ops.aten.scaled_dot_product_attention.default(permute_24, permute_25, permute_26, masked_fill);  permute_24 = permute_25 = permute_26 = None
        transpose_8 = torch.ops.aten.transpose.int(scaled_dot_product_attention_8, 1, 2);  scaled_dot_product_attention_8 = None
        reshape_8 = torch.ops.aten.reshape.default(transpose_8, [1, 36, 768]);  transpose_8 = None
        linear_51 = torch.ops.aten.linear.default(reshape_8, p_bert_encoder_layer_8_attention_output_dense_weight, p_bert_encoder_layer_8_attention_output_dense_bias);  reshape_8 = p_bert_encoder_layer_8_attention_output_dense_weight = p_bert_encoder_layer_8_attention_output_dense_bias = None
        dropout_17 = torch.ops.aten.dropout.default(linear_51, 0.1, False);  linear_51 = None
        add_17 = torch.ops.aten.add.Tensor(dropout_17, layer_norm_16);  dropout_17 = layer_norm_16 = None
        layer_norm_17 = torch.ops.aten.layer_norm.default(add_17, [768], p_bert_encoder_layer_8_attention_output_layernorm_weight, p_bert_encoder_layer_8_attention_output_layernorm_bias, 1e-12);  add_17 = p_bert_encoder_layer_8_attention_output_layernorm_weight = p_bert_encoder_layer_8_attention_output_layernorm_bias = None
        linear_52 = torch.ops.aten.linear.default(layer_norm_17, p_bert_encoder_layer_8_intermediate_dense_weight, p_bert_encoder_layer_8_intermediate_dense_bias);  p_bert_encoder_layer_8_intermediate_dense_weight = p_bert_encoder_layer_8_intermediate_dense_bias = None
        gelu_8 = torch.ops.aten.gelu.default(linear_52);  linear_52 = None
        linear_53 = torch.ops.aten.linear.default(gelu_8, p_bert_encoder_layer_8_output_dense_weight, p_bert_encoder_layer_8_output_dense_bias);  gelu_8 = p_bert_encoder_layer_8_output_dense_weight = p_bert_encoder_layer_8_output_dense_bias = None
        dropout_18 = torch.ops.aten.dropout.default(linear_53, 0.1, False);  linear_53 = None
        add_18 = torch.ops.aten.add.Tensor(dropout_18, layer_norm_17);  dropout_18 = layer_norm_17 = None
        layer_norm_18 = torch.ops.aten.layer_norm.default(add_18, [768], p_bert_encoder_layer_8_output_layernorm_weight, p_bert_encoder_layer_8_output_layernorm_bias, 1e-12);  add_18 = p_bert_encoder_layer_8_output_layernorm_weight = p_bert_encoder_layer_8_output_layernorm_bias = None
        linear_54 = torch.ops.aten.linear.default(layer_norm_18, p_bert_encoder_layer_9_attention_self_query_weight, p_bert_encoder_layer_9_attention_self_query_bias);  p_bert_encoder_layer_9_attention_self_query_weight = p_bert_encoder_layer_9_attention_self_query_bias = None
        view_27 = torch.ops.aten.view.default(linear_54, [1, 36, 12, 64]);  linear_54 = None
        permute_27 = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None
        linear_55 = torch.ops.aten.linear.default(layer_norm_18, p_bert_encoder_layer_9_attention_self_key_weight, p_bert_encoder_layer_9_attention_self_key_bias);  p_bert_encoder_layer_9_attention_self_key_weight = p_bert_encoder_layer_9_attention_self_key_bias = None
        view_28 = torch.ops.aten.view.default(linear_55, [1, 36, 12, 64]);  linear_55 = None
        permute_28 = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
        linear_56 = torch.ops.aten.linear.default(layer_norm_18, p_bert_encoder_layer_9_attention_self_value_weight, p_bert_encoder_layer_9_attention_self_value_bias);  p_bert_encoder_layer_9_attention_self_value_weight = p_bert_encoder_layer_9_attention_self_value_bias = None
        view_29 = torch.ops.aten.view.default(linear_56, [1, 36, 12, 64]);  linear_56 = None
        permute_29 = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
        scaled_dot_product_attention_9 = torch.ops.aten.scaled_dot_product_attention.default(permute_27, permute_28, permute_29, masked_fill);  permute_27 = permute_28 = permute_29 = None
        transpose_9 = torch.ops.aten.transpose.int(scaled_dot_product_attention_9, 1, 2);  scaled_dot_product_attention_9 = None
        reshape_9 = torch.ops.aten.reshape.default(transpose_9, [1, 36, 768]);  transpose_9 = None
        linear_57 = torch.ops.aten.linear.default(reshape_9, p_bert_encoder_layer_9_attention_output_dense_weight, p_bert_encoder_layer_9_attention_output_dense_bias);  reshape_9 = p_bert_encoder_layer_9_attention_output_dense_weight = p_bert_encoder_layer_9_attention_output_dense_bias = None
        dropout_19 = torch.ops.aten.dropout.default(linear_57, 0.1, False);  linear_57 = None
        add_19 = torch.ops.aten.add.Tensor(dropout_19, layer_norm_18);  dropout_19 = layer_norm_18 = None
        layer_norm_19 = torch.ops.aten.layer_norm.default(add_19, [768], p_bert_encoder_layer_9_attention_output_layernorm_weight, p_bert_encoder_layer_9_attention_output_layernorm_bias, 1e-12);  add_19 = p_bert_encoder_layer_9_attention_output_layernorm_weight = p_bert_encoder_layer_9_attention_output_layernorm_bias = None
        linear_58 = torch.ops.aten.linear.default(layer_norm_19, p_bert_encoder_layer_9_intermediate_dense_weight, p_bert_encoder_layer_9_intermediate_dense_bias);  p_bert_encoder_layer_9_intermediate_dense_weight = p_bert_encoder_layer_9_intermediate_dense_bias = None
        gelu_9 = torch.ops.aten.gelu.default(linear_58);  linear_58 = None
        linear_59 = torch.ops.aten.linear.default(gelu_9, p_bert_encoder_layer_9_output_dense_weight, p_bert_encoder_layer_9_output_dense_bias);  gelu_9 = p_bert_encoder_layer_9_output_dense_weight = p_bert_encoder_layer_9_output_dense_bias = None
        dropout_20 = torch.ops.aten.dropout.default(linear_59, 0.1, False);  linear_59 = None
        add_20 = torch.ops.aten.add.Tensor(dropout_20, layer_norm_19);  dropout_20 = layer_norm_19 = None
        layer_norm_20 = torch.ops.aten.layer_norm.default(add_20, [768], p_bert_encoder_layer_9_output_layernorm_weight, p_bert_encoder_layer_9_output_layernorm_bias, 1e-12);  add_20 = p_bert_encoder_layer_9_output_layernorm_weight = p_bert_encoder_layer_9_output_layernorm_bias = None
        linear_60 = torch.ops.aten.linear.default(layer_norm_20, p_bert_encoder_layer_10_attention_self_query_weight, p_bert_encoder_layer_10_attention_self_query_bias);  p_bert_encoder_layer_10_attention_self_query_weight = p_bert_encoder_layer_10_attention_self_query_bias = None
        view_30 = torch.ops.aten.view.default(linear_60, [1, 36, 12, 64]);  linear_60 = None
        permute_30 = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
        linear_61 = torch.ops.aten.linear.default(layer_norm_20, p_bert_encoder_layer_10_attention_self_key_weight, p_bert_encoder_layer_10_attention_self_key_bias);  p_bert_encoder_layer_10_attention_self_key_weight = p_bert_encoder_layer_10_attention_self_key_bias = None
        view_31 = torch.ops.aten.view.default(linear_61, [1, 36, 12, 64]);  linear_61 = None
        permute_31 = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None
        linear_62 = torch.ops.aten.linear.default(layer_norm_20, p_bert_encoder_layer_10_attention_self_value_weight, p_bert_encoder_layer_10_attention_self_value_bias);  p_bert_encoder_layer_10_attention_self_value_weight = p_bert_encoder_layer_10_attention_self_value_bias = None
        view_32 = torch.ops.aten.view.default(linear_62, [1, 36, 12, 64]);  linear_62 = None
        permute_32 = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None
        scaled_dot_product_attention_10 = torch.ops.aten.scaled_dot_product_attention.default(permute_30, permute_31, permute_32, masked_fill);  permute_30 = permute_31 = permute_32 = None
        transpose_10 = torch.ops.aten.transpose.int(scaled_dot_product_attention_10, 1, 2);  scaled_dot_product_attention_10 = None
        reshape_10 = torch.ops.aten.reshape.default(transpose_10, [1, 36, 768]);  transpose_10 = None
        linear_63 = torch.ops.aten.linear.default(reshape_10, p_bert_encoder_layer_10_attention_output_dense_weight, p_bert_encoder_layer_10_attention_output_dense_bias);  reshape_10 = p_bert_encoder_layer_10_attention_output_dense_weight = p_bert_encoder_layer_10_attention_output_dense_bias = None
        dropout_21 = torch.ops.aten.dropout.default(linear_63, 0.1, False);  linear_63 = None
        add_21 = torch.ops.aten.add.Tensor(dropout_21, layer_norm_20);  dropout_21 = layer_norm_20 = None
        layer_norm_21 = torch.ops.aten.layer_norm.default(add_21, [768], p_bert_encoder_layer_10_attention_output_layernorm_weight, p_bert_encoder_layer_10_attention_output_layernorm_bias, 1e-12);  add_21 = p_bert_encoder_layer_10_attention_output_layernorm_weight = p_bert_encoder_layer_10_attention_output_layernorm_bias = None
        linear_64 = torch.ops.aten.linear.default(layer_norm_21, p_bert_encoder_layer_10_intermediate_dense_weight, p_bert_encoder_layer_10_intermediate_dense_bias);  p_bert_encoder_layer_10_intermediate_dense_weight = p_bert_encoder_layer_10_intermediate_dense_bias = None
        gelu_10 = torch.ops.aten.gelu.default(linear_64);  linear_64 = None
        linear_65 = torch.ops.aten.linear.default(gelu_10, p_bert_encoder_layer_10_output_dense_weight, p_bert_encoder_layer_10_output_dense_bias);  gelu_10 = p_bert_encoder_layer_10_output_dense_weight = p_bert_encoder_layer_10_output_dense_bias = None
        dropout_22 = torch.ops.aten.dropout.default(linear_65, 0.1, False);  linear_65 = None
        add_22 = torch.ops.aten.add.Tensor(dropout_22, layer_norm_21);  dropout_22 = layer_norm_21 = None
        layer_norm_22 = torch.ops.aten.layer_norm.default(add_22, [768], p_bert_encoder_layer_10_output_layernorm_weight, p_bert_encoder_layer_10_output_layernorm_bias, 1e-12);  add_22 = p_bert_encoder_layer_10_output_layernorm_weight = p_bert_encoder_layer_10_output_layernorm_bias = None
        linear_66 = torch.ops.aten.linear.default(layer_norm_22, p_bert_encoder_layer_11_attention_self_query_weight, p_bert_encoder_layer_11_attention_self_query_bias);  p_bert_encoder_layer_11_attention_self_query_weight = p_bert_encoder_layer_11_attention_self_query_bias = None
        view_33 = torch.ops.aten.view.default(linear_66, [1, 36, 12, 64]);  linear_66 = None
        permute_33 = torch.ops.aten.permute.default(view_33, [0, 2, 1, 3]);  view_33 = None
        linear_67 = torch.ops.aten.linear.default(layer_norm_22, p_bert_encoder_layer_11_attention_self_key_weight, p_bert_encoder_layer_11_attention_self_key_bias);  p_bert_encoder_layer_11_attention_self_key_weight = p_bert_encoder_layer_11_attention_self_key_bias = None
        view_34 = torch.ops.aten.view.default(linear_67, [1, 36, 12, 64]);  linear_67 = None
        permute_34 = torch.ops.aten.permute.default(view_34, [0, 2, 1, 3]);  view_34 = None
        linear_68 = torch.ops.aten.linear.default(layer_norm_22, p_bert_encoder_layer_11_attention_self_value_weight, p_bert_encoder_layer_11_attention_self_value_bias);  p_bert_encoder_layer_11_attention_self_value_weight = p_bert_encoder_layer_11_attention_self_value_bias = None
        view_35 = torch.ops.aten.view.default(linear_68, [1, 36, 12, 64]);  linear_68 = None
        permute_35 = torch.ops.aten.permute.default(view_35, [0, 2, 1, 3]);  view_35 = None
        scaled_dot_product_attention_11 = torch.ops.aten.scaled_dot_product_attention.default(permute_33, permute_34, permute_35, masked_fill);  permute_33 = permute_34 = permute_35 = masked_fill = None
        transpose_11 = torch.ops.aten.transpose.int(scaled_dot_product_attention_11, 1, 2);  scaled_dot_product_attention_11 = None
        reshape_11 = torch.ops.aten.reshape.default(transpose_11, [1, 36, 768]);  transpose_11 = None
        linear_69 = torch.ops.aten.linear.default(reshape_11, p_bert_encoder_layer_11_attention_output_dense_weight, p_bert_encoder_layer_11_attention_output_dense_bias);  reshape_11 = p_bert_encoder_layer_11_attention_output_dense_weight = p_bert_encoder_layer_11_attention_output_dense_bias = None
        dropout_23 = torch.ops.aten.dropout.default(linear_69, 0.1, False);  linear_69 = None
        add_23 = torch.ops.aten.add.Tensor(dropout_23, layer_norm_22);  dropout_23 = layer_norm_22 = None
        layer_norm_23 = torch.ops.aten.layer_norm.default(add_23, [768], p_bert_encoder_layer_11_attention_output_layernorm_weight, p_bert_encoder_layer_11_attention_output_layernorm_bias, 1e-12);  add_23 = p_bert_encoder_layer_11_attention_output_layernorm_weight = p_bert_encoder_layer_11_attention_output_layernorm_bias = None
        linear_70 = torch.ops.aten.linear.default(layer_norm_23, p_bert_encoder_layer_11_intermediate_dense_weight, p_bert_encoder_layer_11_intermediate_dense_bias);  p_bert_encoder_layer_11_intermediate_dense_weight = p_bert_encoder_layer_11_intermediate_dense_bias = None
        gelu_11 = torch.ops.aten.gelu.default(linear_70);  linear_70 = None
        linear_71 = torch.ops.aten.linear.default(gelu_11, p_bert_encoder_layer_11_output_dense_weight, p_bert_encoder_layer_11_output_dense_bias);  gelu_11 = p_bert_encoder_layer_11_output_dense_weight = p_bert_encoder_layer_11_output_dense_bias = None
        dropout_24 = torch.ops.aten.dropout.default(linear_71, 0.1, False);  linear_71 = None
        add_24 = torch.ops.aten.add.Tensor(dropout_24, layer_norm_23);  dropout_24 = layer_norm_23 = None
        layer_norm_24 = torch.ops.aten.layer_norm.default(add_24, [768], p_bert_encoder_layer_11_output_layernorm_weight, p_bert_encoder_layer_11_output_layernorm_bias, 1e-12);  add_24 = p_bert_encoder_layer_11_output_layernorm_weight = p_bert_encoder_layer_11_output_layernorm_bias = None
        slice_5 = torch.ops.aten.slice.Tensor(layer_norm_24, 0, 0, 9223372036854775807);  layer_norm_24 = None
        select = torch.ops.aten.select.int(slice_5, 1, 0);  slice_5 = None
        linear_72 = torch.ops.aten.linear.default(select, p_bert_pooler_dense_weight, p_bert_pooler_dense_bias);  select = p_bert_pooler_dense_weight = p_bert_pooler_dense_bias = None
        tanh = torch.ops.aten.tanh.default(linear_72);  linear_72 = None
        dropout_25 = torch.ops.aten.dropout.default(tanh, 0.1, False);  tanh = None
        linear_73 = torch.ops.aten.linear.default(dropout_25, p_classifier_weight, p_classifier_bias);  dropout_25 = p_classifier_weight = p_classifier_bias = None
        return (linear_73,)
        
    # To see more debug info, please use `graph_module.print_readable()`
