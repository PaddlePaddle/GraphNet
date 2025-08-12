import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_embedding_parameters_weight_: torch.nn.parameter.Parameter,
        L_batch_: torch.Tensor,
    ):
        l_self_modules_embedding_parameters_weight_ = (
            L_self_modules_embedding_parameters_weight_
        )
        l_batch_ = L_batch_
        emb = l_self_modules_embedding_parameters_weight_[slice(500, 1000, None)]
        l_self_modules_embedding_parameters_weight_ = None
        index_select = emb.index_select(0, l_batch_)
        emb = l_batch_ = None
        return (index_select,)
