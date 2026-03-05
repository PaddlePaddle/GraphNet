import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, s38 : torch.SymInt, L_labels_ : torch.Tensor):
        l_labels_ = L_labels_
        size = l_labels_.size()
        shifted_input_ids = l_labels_.new_zeros(size);  size = None
        getitem_2 = l_labels_[(Ellipsis, slice(None, -1, None))];  l_labels_ = None
        clone = getitem_2.clone();  getitem_2 = None
        shifted_input_ids[(Ellipsis, slice(1, None, None))] = clone;  setitem = shifted_input_ids;  clone = setitem = None
        shifted_input_ids[(Ellipsis, 0)] = 2;  setitem_1 = shifted_input_ids;  setitem_1 = None
        eq = shifted_input_ids.__eq__(-100)
        masked_fill_ = shifted_input_ids.masked_fill_(eq, 0);  eq = masked_fill_ = None
        return (shifted_input_ids,)
        