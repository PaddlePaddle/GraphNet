import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        s0: torch.SymInt,
        L_stack0_last_hidden_state: torch.Tensor,
        L_logits_to_keep_: torch.SymInt,
        L_self_modules_lm_head_parameters_weight_: torch.nn.parameter.Parameter,
    ):
        l_stack0_last_hidden_state = L_stack0_last_hidden_state
        l_logits_to_keep_ = L_logits_to_keep_
        l_self_modules_lm_head_parameters_weight_ = (
            L_self_modules_lm_head_parameters_weight_
        )
        neg = -l_logits_to_keep_
        l_logits_to_keep_ = None
        getitem = l_stack0_last_hidden_state[
            (slice(None, None, None), slice(neg, None, None), slice(None, None, None))
        ]
        l_stack0_last_hidden_state = neg = None
        logits = torch._C._nn.linear(
            getitem, l_self_modules_lm_head_parameters_weight_, None
        )
        getitem = l_self_modules_lm_head_parameters_weight_ = None
        return (logits,)
