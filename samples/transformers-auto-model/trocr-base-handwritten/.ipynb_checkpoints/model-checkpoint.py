import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(self, L_logits_: torch.Tensor, L_labels_: torch.Tensor):
        l_logits_ = L_logits_
        l_labels_ = L_labels_
        logits = l_logits_.float()
        l_logits_ = None
        labels = torch._C._nn.pad(l_labels_, (0, 1), "constant", -100)
        l_labels_ = None
        getitem = labels[(Ellipsis, slice(1, None, None))]
        labels = None
        shift_labels = getitem.contiguous()
        getitem = None
        logits_1 = logits.view(-1, 50265)
        logits = None
        shift_labels_1 = shift_labels.view(-1)
        shift_labels = None
        shift_labels_2 = shift_labels_1.to(device(type="cpu"))
        shift_labels_1 = None
        loss = torch.nn.functional.cross_entropy(
            logits_1, shift_labels_2, ignore_index=-100, reduction="mean"
        )
        logits_1 = shift_labels_2 = None
        return (loss,)
