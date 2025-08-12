import torch


class GraphModule(torch.nn.Module):
    def forward(self, L_x_: torch.Tensor, L_batch_x_: torch.Tensor):
        l_x_ = L_x_
        l_batch_x_ = L_batch_x_
        x = l_x_.contiguous()
        y = l_x_.contiguous()
        l_x_ = None
        max_1 = l_batch_x_.max()
        item = max_1.item()
        max_1 = None
        add = item + 1
        item = None
        max_2 = l_batch_x_.max()
        l_batch_x_ = None
        item_1 = max_2.item()
        max_2 = None
        sym_sum = torch.sym_sum([1, item_1])
        item_1 = None
        sym_max_1 = torch.sym_max(add, sym_sum)
        add = sym_sum = None
        gt_2 = sym_max_1 > 0
        _assert_scalar_default = torch.ops.aten._assert_scalar.default(
            gt_2,
            "Runtime assertion failed for expression 0 < Max(u0 + 1, u1 + 1) on node 'gt_2'",
        )
        gt_2 = _assert_scalar_default = None
        gt_1 = sym_max_1 > 1
        return (gt_1, x, y, sym_max_1)
