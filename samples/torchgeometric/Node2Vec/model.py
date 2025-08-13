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
        getitem = l_self_modules_embedding_parameters_weight_[l_batch_]
        l_self_modules_embedding_parameters_weight_ = l_batch_ = None
        return (getitem,)
