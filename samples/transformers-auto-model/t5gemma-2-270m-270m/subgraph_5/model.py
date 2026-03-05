import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, L_logits_to_keep_ : torch.SymInt, s11 : torch.SymInt, L_stack0_last_hidden_state : torch.Tensor, L_self_modules_lm_head_modules_out_proj_parameters_weight_ : torch.nn.parameter.Parameter, L_labels_ : torch.Tensor):
        l_logits_to_keep_ = L_logits_to_keep_
        l_stack0_last_hidden_state = L_stack0_last_hidden_state
        l_self_modules_lm_head_modules_out_proj_parameters_weight_ = L_self_modules_lm_head_modules_out_proj_parameters_weight_
        l_labels_ = L_labels_
        neg = -l_logits_to_keep_;  l_logits_to_keep_ = None
        getitem = l_stack0_last_hidden_state[(slice(None, None, None), slice(neg, None, None), slice(None, None, None))];  l_stack0_last_hidden_state = neg = None
        logits = torch.nn.functional.linear(getitem, l_self_modules_lm_head_modules_out_proj_parameters_weight_, None);  getitem = l_self_modules_lm_head_modules_out_proj_parameters_weight_ = None
        logits_1 = logits.float()
        logits_2 = logits_1.view(-1, 262144);  logits_1 = None
        labels = l_labels_.view(-1);  l_labels_ = None
        labels_1 = labels.to(device(type='cuda', index=0));  labels = None
        loss = torch.nn.functional.cross_entropy(logits_2, labels_1, ignore_index = -100, reduction = 'mean');  logits_2 = labels_1 = None
        return (loss, logits)
        