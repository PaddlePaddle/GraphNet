import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        s0: torch.SymInt,
        L_index_indices: torch.Tensor,
        L_index_num_segments: torch.Tensor,
    ):
        l_index_indices = L_index_indices
        l_index_num_segments = L_index_num_segments
        tensor = torch.tensor([1])
        batch_size = torch.prod(tensor)
        tensor = None
        arange = torch.arange(start=0, end=batch_size, device=device(type="cpu"))
        offset = arange * l_index_num_segments
        arange = None
        offset_1 = offset.view((1,))
        offset = None
        offset_2 = offset_1.unsqueeze(-1)
        offset_1 = None
        indices = offset_2 + l_index_indices
        offset_2 = l_index_indices = None
        view_1 = indices.view(-1)
        indices = None
        mul_1 = l_index_num_segments * batch_size
        l_index_num_segments = batch_size = None
        as_tensor = torch.as_tensor(view_1, device=device(type="cpu"))
        view_1 = None
        as_tensor_1 = torch.as_tensor(mul_1, device=device(type="cpu"))
        mul_1 = None
        as_tensor_2 = torch.as_tensor([-1], dtype=torch.int64)
        as_tensor_3 = torch.as_tensor((), dtype=torch.int64)
        flattened_shape = torch.cat([as_tensor_2, as_tensor_3], dim=0)
        as_tensor_2 = as_tensor_3 = None
        return (flattened_shape, as_tensor_1, as_tensor)
