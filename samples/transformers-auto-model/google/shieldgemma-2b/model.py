import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_stack0_last_hidden_state: torch.Tensor,
        L_self_modules_lm_head_parameters_weight_: torch.nn.parameter.Parameter,
    ):
        l_stack0_last_hidden_state = L_stack0_last_hidden_state
        l_self_modules_lm_head_parameters_weight_ = (
            L_self_modules_lm_head_parameters_weight_
        )
        getitem = l_stack0_last_hidden_state[
            (slice(None, None, None), slice(0, None, None), slice(None, None, None))
        ]
        l_stack0_last_hidden_state = None
        logits = torch._C._nn.linear(
            getitem, l_self_modules_lm_head_parameters_weight_, None
        )
        getitem = l_self_modules_lm_head_parameters_weight_ = None
        logits_1 = logits / 30.0
        logits = None
        logits_2 = torch.tanh(logits_1)
        logits_1 = None
        logits_3 = logits_2 * 30.0
        logits_2 = None
        return (logits_3,)
