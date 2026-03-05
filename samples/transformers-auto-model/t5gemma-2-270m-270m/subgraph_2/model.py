import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, s70 : torch.SymInt, L_input_ids_ : torch.Tensor, s53 : torch.SymInt, L_inputs_embeds_ : torch.Tensor, L_image_features_ : torch.Tensor):
        l_input_ids_ = L_input_ids_
        l_inputs_embeds_ = L_inputs_embeds_
        l_image_features_ = L_image_features_
        special_image_mask = l_input_ids_.__eq__(256001);  l_input_ids_ = None
        n_image_tokens = special_image_mask.sum()
        unsqueeze = special_image_mask.unsqueeze(-1);  special_image_mask = None
        expand_as = unsqueeze.expand_as(l_inputs_embeds_);  unsqueeze = None
        special_image_mask_1 = expand_as.to(device(type='cuda', index=0));  expand_as = None
        getitem_6 = l_inputs_embeds_[special_image_mask_1];  l_inputs_embeds_ = None
        sym_size_int = torch.ops.aten.sym_size.int(getitem_6, 0);  getitem_6 = None
        _check_is_size = torch._check_is_size(sym_size_int);  _check_is_size = None
        ge = sym_size_int >= 0
        _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None
        numel_1 = l_image_features_.numel();  l_image_features_ = None
        eq_1 = sym_size_int == numel_1;  sym_size_int = numel_1 = None
        return (eq_1, n_image_tokens, special_image_mask_1)
        