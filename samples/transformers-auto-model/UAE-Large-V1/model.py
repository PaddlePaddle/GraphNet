import torch


class GraphModule(torch.nn.Module):
    def forward(self, L_features_token_embeddings_: torch.Tensor):
        l_features_token_embeddings_ = L_features_token_embeddings_
        cls_token = l_features_token_embeddings_[(slice(None, None, None), 0)]
        l_features_token_embeddings_ = None
        output_vector = torch.cat([cls_token], 1)
        cls_token = None
        return (output_vector,)
