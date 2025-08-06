import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_input_ids_: torch.Tensor,
        L_self_modules_transformer_modules_wte_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_wpe_parameters_weight_: torch.nn.parameter.Parameter,
        L_attention_mask_: torch.Tensor,
        L_self_modules_transformer_modules_h_modules_0_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_2_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_2_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_2_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_2_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_3_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_3_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_3_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_3_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_4_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_4_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_4_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_4_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_5_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_5_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_attn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_attn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_5_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_5_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_ln_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transformer_modules_ln_f_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_input_ids_ = L_input_ids_
        l_self_modules_transformer_modules_wte_parameters_weight_ = (
            L_self_modules_transformer_modules_wte_parameters_weight_
        )
        l_self_modules_transformer_modules_wpe_parameters_weight_ = (
            L_self_modules_transformer_modules_wpe_parameters_weight_
        )
        l_attention_mask_ = L_attention_mask_
        l_self_modules_transformer_modules_h_modules_0_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_0_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_0_modules_ln_1_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_0_modules_ln_1_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_0_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_0_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_0_modules_ln_2_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_0_modules_ln_2_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_1_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_1_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_1_modules_ln_1_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_1_modules_ln_1_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_1_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_1_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_1_modules_ln_2_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_1_modules_ln_2_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_2_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_2_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_2_modules_ln_1_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_2_modules_ln_1_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_2_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_2_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_2_modules_ln_2_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_2_modules_ln_2_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_3_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_3_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_3_modules_ln_1_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_3_modules_ln_1_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_3_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_3_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_3_modules_ln_2_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_3_modules_ln_2_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_4_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_4_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_4_modules_ln_1_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_4_modules_ln_1_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_4_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_4_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_4_modules_ln_2_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_4_modules_ln_2_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_5_modules_ln_1_parameters_weight_ = L_self_modules_transformer_modules_h_modules_5_modules_ln_1_parameters_weight_
        l_self_modules_transformer_modules_h_modules_5_modules_ln_1_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_5_modules_ln_1_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_attn_parameters_bias_ = L_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_attn_parameters_bias_
        l_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_attn_parameters_weight_ = L_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_attn_parameters_weight_
        l_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_h_modules_5_modules_ln_2_parameters_weight_ = L_self_modules_transformer_modules_h_modules_5_modules_ln_2_parameters_weight_
        l_self_modules_transformer_modules_h_modules_5_modules_ln_2_parameters_bias_ = (
            L_self_modules_transformer_modules_h_modules_5_modules_ln_2_parameters_bias_
        )
        l_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_bias_ = L_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_bias_
        l_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_weight_ = L_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_weight_
        l_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_bias_ = L_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_bias_
        l_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_weight_ = L_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_weight_
        l_self_modules_transformer_modules_ln_f_parameters_weight_ = (
            L_self_modules_transformer_modules_ln_f_parameters_weight_
        )
        l_self_modules_transformer_modules_ln_f_parameters_bias_ = (
            L_self_modules_transformer_modules_ln_f_parameters_bias_
        )
        input_ids = l_input_ids_.view(-1, 19)
        l_input_ids_ = None
        inputs_embeds = torch.nn.functional.embedding(
            input_ids,
            l_self_modules_transformer_modules_wte_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        input_ids = None
        cache_position = torch.arange(0, 19, device=device(type="cpu"))
        position_ids = cache_position.unsqueeze(0)
        position_embeds = torch.nn.functional.embedding(
            position_ids,
            l_self_modules_transformer_modules_wpe_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        position_ids = l_self_modules_transformer_modules_wpe_parameters_weight_ = None
        to = position_embeds.to(device(type="cpu"))
        position_embeds = None
        hidden_states = inputs_embeds + to
        inputs_embeds = to = None
        attention_mask = l_attention_mask_.view(1, -1)
        l_attention_mask_ = None
        attention_mask_1 = attention_mask.to(
            device=device(type="cpu"), dtype=torch.bool
        )
        attention_mask = None
        kv_arange = torch.arange(19, device=device(type="cpu"))
        kv_arange += 0
        kv_arange_1 = kv_arange
        kv_arange = None
        batch_arange = torch.arange(1, device=device(type="cpu"))
        head_arange = torch.arange(1, device=device(type="cpu"))
        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions()
        lazy_load_decompositions = None
        _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(
            1, "error"
        )
        _vmap_increment_nesting = None
        child = torch._C._functorch._add_batch_dim(batch_arange, 0, 1)
        batch_arange = None
        lazy_load_decompositions_1 = torch._functorch.vmap.lazy_load_decompositions()
        lazy_load_decompositions_1 = None
        _vmap_increment_nesting_1 = torch._C._functorch._vmap_increment_nesting(
            1, "error"
        )
        _vmap_increment_nesting_1 = None
        child_1 = torch._C._functorch._add_batch_dim(head_arange, 0, 2)
        head_arange = child_1 = None
        lazy_load_decompositions_2 = torch._functorch.vmap.lazy_load_decompositions()
        lazy_load_decompositions_2 = None
        _vmap_increment_nesting_2 = torch._C._functorch._vmap_increment_nesting(
            19, "error"
        )
        _vmap_increment_nesting_2 = None
        child_2 = torch._C._functorch._add_batch_dim(cache_position, 0, 3)
        cache_position = None
        lazy_load_decompositions_3 = torch._functorch.vmap.lazy_load_decompositions()
        lazy_load_decompositions_3 = None
        _vmap_increment_nesting_3 = torch._C._functorch._vmap_increment_nesting(
            19, "error"
        )
        _vmap_increment_nesting_3 = None
        child_3 = torch._C._functorch._add_batch_dim(kv_arange_1, 0, 4)
        kv_arange_1 = None
        result = child_2.new_ones((), dtype=torch.bool)
        le = child_3.le(child_2)
        child_2 = None
        result_1 = result.__and__(le)
        result = le = None
        function_ctx = torch.autograd.function.FunctionCtx()
        function_ctx = None
        index = torch.ops.aten.index(attention_mask_1, [child, child_3])
        attention_mask_1 = child = child_3 = None
        result_2 = result_1.__and__(index)
        result_1 = index = None
        batched_outputs = torch._C._functorch._remove_batch_dim(result_2, 4, 19, 0)
        result_2 = None
        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting()
        _vmap_decrement_nesting = None
        batched_outputs_1 = torch._C._functorch._remove_batch_dim(
            batched_outputs, 3, 19, 0
        )
        batched_outputs = None
        _vmap_decrement_nesting_1 = torch._C._functorch._vmap_decrement_nesting()
        _vmap_decrement_nesting_1 = None
        batched_outputs_2 = torch._C._functorch._remove_batch_dim(
            batched_outputs_1, 2, 1, 0
        )
        batched_outputs_1 = None
        _vmap_decrement_nesting_2 = torch._C._functorch._vmap_decrement_nesting()
        _vmap_decrement_nesting_2 = None
        causal_mask = torch._C._functorch._remove_batch_dim(batched_outputs_2, 1, 1, 0)
        batched_outputs_2 = None
        _vmap_decrement_nesting_3 = torch._C._functorch._vmap_decrement_nesting()
        _vmap_decrement_nesting_3 = None
        hidden_states_1 = torch.nn.functional.dropout(hidden_states, 0.1, False, False)
        hidden_states = None
        hidden_states_2 = torch.nn.functional.layer_norm(
            hidden_states_1,
            (768,),
            l_self_modules_transformer_modules_h_modules_0_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_0_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_0_modules_ln_1_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_0_modules_ln_1_parameters_bias_
        ) = None
        view_2 = hidden_states_2.view(-1, 768)
        hidden_states_2 = None
        x = torch.addmm(
            l_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_attn_parameters_bias_,
            view_2,
            l_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_attn_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_attn_parameters_bias_ = (
            view_2
        ) = l_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_attn_parameters_weight_ = (None)
        x_1 = x.view((1, 19, 2304))
        x = None
        split = x_1.split(768, dim=2)
        x_1 = None
        query_states = split[0]
        key_states = split[1]
        value_states = split[2]
        split = None
        view_4 = key_states.view((1, 19, -1, 64))
        key_states = None
        key_states_1 = view_4.transpose(1, 2)
        view_4 = None
        view_5 = value_states.view((1, 19, -1, 64))
        value_states = None
        value_states_1 = view_5.transpose(1, 2)
        view_5 = None
        view_6 = query_states.view((1, 19, -1, 64))
        query_states = None
        query_states_1 = view_6.transpose(1, 2)
        view_6 = None
        attention_mask_2 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query = query_states_1.contiguous()
        query_states_1 = None
        key = key_states_1.contiguous()
        key_states_1 = None
        value = value_states_1.contiguous()
        value_states_1 = None
        attn_output = torch._C._nn.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask_2,
            dropout_p=0.0,
            scale=None,
            is_causal=False,
        )
        query = key = value = attention_mask_2 = None
        transpose_3 = attn_output.transpose(1, 2)
        attn_output = None
        attn_output_1 = transpose_3.contiguous()
        transpose_3 = None
        reshape = attn_output_1.reshape(1, 19, -1)
        attn_output_1 = None
        attn_output_2 = reshape.contiguous()
        reshape = None
        view_7 = attn_output_2.view(-1, 768)
        attn_output_2 = None
        x_2 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_proj_parameters_bias_,
            view_7,
            l_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_proj_parameters_bias_ = (
            view_7
        ) = l_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_proj_parameters_weight_ = (None)
        x_3 = x_2.view((1, 19, 768))
        x_2 = None
        attn_output_3 = torch.nn.functional.dropout(x_3, 0.1, False, False)
        x_3 = None
        hidden_states_3 = attn_output_3 + hidden_states_1
        attn_output_3 = hidden_states_1 = None
        hidden_states_4 = torch.nn.functional.layer_norm(
            hidden_states_3,
            (768,),
            l_self_modules_transformer_modules_h_modules_0_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_0_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_0_modules_ln_2_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_0_modules_ln_2_parameters_bias_
        ) = None
        view_9 = hidden_states_4.view(-1, 768)
        hidden_states_4 = None
        x_4 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_bias_,
            view_9,
            l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_bias_ = (
            view_9
        ) = l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_weight_ = (None)
        x_5 = x_4.view((1, 19, 3072))
        x_4 = None
        mul = 0.5 * x_5
        pow_1 = torch.pow(x_5, 3.0)
        mul_1 = 0.044715 * pow_1
        pow_1 = None
        add_2 = x_5 + mul_1
        x_5 = mul_1 = None
        mul_2 = 0.7978845608028654 * add_2
        add_2 = None
        tanh = torch.tanh(mul_2)
        mul_2 = None
        add_3 = 1.0 + tanh
        tanh = None
        hidden_states_5 = mul * add_3
        mul = add_3 = None
        view_11 = hidden_states_5.view(-1, 3072)
        hidden_states_5 = None
        x_6 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_bias_,
            view_11,
            l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_bias_ = (
            view_11
        ) = l_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_weight_ = (None)
        x_7 = x_6.view((1, 19, 768))
        x_6 = None
        hidden_states_6 = torch.nn.functional.dropout(x_7, 0.1, False, False)
        x_7 = None
        hidden_states_7 = hidden_states_3 + hidden_states_6
        hidden_states_3 = hidden_states_6 = None
        hidden_states_8 = torch.nn.functional.layer_norm(
            hidden_states_7,
            (768,),
            l_self_modules_transformer_modules_h_modules_1_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_1_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_1_modules_ln_1_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_1_modules_ln_1_parameters_bias_
        ) = None
        view_13 = hidden_states_8.view(-1, 768)
        hidden_states_8 = None
        x_8 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_attn_parameters_bias_,
            view_13,
            l_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_attn_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_attn_parameters_bias_ = (
            view_13
        ) = l_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_attn_parameters_weight_ = (None)
        x_9 = x_8.view((1, 19, 2304))
        x_8 = None
        split_1 = x_9.split(768, dim=2)
        x_9 = None
        query_states_2 = split_1[0]
        key_states_2 = split_1[1]
        value_states_2 = split_1[2]
        split_1 = None
        view_15 = key_states_2.view((1, 19, -1, 64))
        key_states_2 = None
        key_states_3 = view_15.transpose(1, 2)
        view_15 = None
        view_16 = value_states_2.view((1, 19, -1, 64))
        value_states_2 = None
        value_states_3 = view_16.transpose(1, 2)
        view_16 = None
        view_17 = query_states_2.view((1, 19, -1, 64))
        query_states_2 = None
        query_states_3 = view_17.transpose(1, 2)
        view_17 = None
        attention_mask_3 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_1 = query_states_3.contiguous()
        query_states_3 = None
        key_1 = key_states_3.contiguous()
        key_states_3 = None
        value_1 = value_states_3.contiguous()
        value_states_3 = None
        attn_output_4 = torch._C._nn.scaled_dot_product_attention(
            query_1,
            key_1,
            value_1,
            attn_mask=attention_mask_3,
            dropout_p=0.0,
            scale=None,
            is_causal=False,
        )
        query_1 = key_1 = value_1 = attention_mask_3 = None
        transpose_7 = attn_output_4.transpose(1, 2)
        attn_output_4 = None
        attn_output_5 = transpose_7.contiguous()
        transpose_7 = None
        reshape_1 = attn_output_5.reshape(1, 19, -1)
        attn_output_5 = None
        attn_output_6 = reshape_1.contiguous()
        reshape_1 = None
        view_18 = attn_output_6.view(-1, 768)
        attn_output_6 = None
        x_10 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_proj_parameters_bias_,
            view_18,
            l_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_proj_parameters_bias_ = (
            view_18
        ) = l_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_proj_parameters_weight_ = (None)
        x_11 = x_10.view((1, 19, 768))
        x_10 = None
        attn_output_7 = torch.nn.functional.dropout(x_11, 0.1, False, False)
        x_11 = None
        hidden_states_9 = attn_output_7 + hidden_states_7
        attn_output_7 = hidden_states_7 = None
        hidden_states_10 = torch.nn.functional.layer_norm(
            hidden_states_9,
            (768,),
            l_self_modules_transformer_modules_h_modules_1_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_1_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_1_modules_ln_2_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_1_modules_ln_2_parameters_bias_
        ) = None
        view_20 = hidden_states_10.view(-1, 768)
        hidden_states_10 = None
        x_12 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_bias_,
            view_20,
            l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_bias_ = (
            view_20
        ) = l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_weight_ = (None)
        x_13 = x_12.view((1, 19, 3072))
        x_12 = None
        mul_4 = 0.5 * x_13
        pow_2 = torch.pow(x_13, 3.0)
        mul_5 = 0.044715 * pow_2
        pow_2 = None
        add_6 = x_13 + mul_5
        x_13 = mul_5 = None
        mul_6 = 0.7978845608028654 * add_6
        add_6 = None
        tanh_1 = torch.tanh(mul_6)
        mul_6 = None
        add_7 = 1.0 + tanh_1
        tanh_1 = None
        hidden_states_11 = mul_4 * add_7
        mul_4 = add_7 = None
        view_22 = hidden_states_11.view(-1, 3072)
        hidden_states_11 = None
        x_14 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_bias_,
            view_22,
            l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_bias_ = (
            view_22
        ) = l_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_weight_ = (None)
        x_15 = x_14.view((1, 19, 768))
        x_14 = None
        hidden_states_12 = torch.nn.functional.dropout(x_15, 0.1, False, False)
        x_15 = None
        hidden_states_13 = hidden_states_9 + hidden_states_12
        hidden_states_9 = hidden_states_12 = None
        hidden_states_14 = torch.nn.functional.layer_norm(
            hidden_states_13,
            (768,),
            l_self_modules_transformer_modules_h_modules_2_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_2_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_2_modules_ln_1_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_2_modules_ln_1_parameters_bias_
        ) = None
        view_24 = hidden_states_14.view(-1, 768)
        hidden_states_14 = None
        x_16 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_attn_parameters_bias_,
            view_24,
            l_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_attn_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_attn_parameters_bias_ = (
            view_24
        ) = l_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_attn_parameters_weight_ = (None)
        x_17 = x_16.view((1, 19, 2304))
        x_16 = None
        split_2 = x_17.split(768, dim=2)
        x_17 = None
        query_states_4 = split_2[0]
        key_states_4 = split_2[1]
        value_states_4 = split_2[2]
        split_2 = None
        view_26 = key_states_4.view((1, 19, -1, 64))
        key_states_4 = None
        key_states_5 = view_26.transpose(1, 2)
        view_26 = None
        view_27 = value_states_4.view((1, 19, -1, 64))
        value_states_4 = None
        value_states_5 = view_27.transpose(1, 2)
        view_27 = None
        view_28 = query_states_4.view((1, 19, -1, 64))
        query_states_4 = None
        query_states_5 = view_28.transpose(1, 2)
        view_28 = None
        attention_mask_4 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_2 = query_states_5.contiguous()
        query_states_5 = None
        key_2 = key_states_5.contiguous()
        key_states_5 = None
        value_2 = value_states_5.contiguous()
        value_states_5 = None
        attn_output_8 = torch._C._nn.scaled_dot_product_attention(
            query_2,
            key_2,
            value_2,
            attn_mask=attention_mask_4,
            dropout_p=0.0,
            scale=None,
            is_causal=False,
        )
        query_2 = key_2 = value_2 = attention_mask_4 = None
        transpose_11 = attn_output_8.transpose(1, 2)
        attn_output_8 = None
        attn_output_9 = transpose_11.contiguous()
        transpose_11 = None
        reshape_2 = attn_output_9.reshape(1, 19, -1)
        attn_output_9 = None
        attn_output_10 = reshape_2.contiguous()
        reshape_2 = None
        view_29 = attn_output_10.view(-1, 768)
        attn_output_10 = None
        x_18 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_proj_parameters_bias_,
            view_29,
            l_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_proj_parameters_bias_ = (
            view_29
        ) = l_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_proj_parameters_weight_ = (None)
        x_19 = x_18.view((1, 19, 768))
        x_18 = None
        attn_output_11 = torch.nn.functional.dropout(x_19, 0.1, False, False)
        x_19 = None
        hidden_states_15 = attn_output_11 + hidden_states_13
        attn_output_11 = hidden_states_13 = None
        hidden_states_16 = torch.nn.functional.layer_norm(
            hidden_states_15,
            (768,),
            l_self_modules_transformer_modules_h_modules_2_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_2_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_2_modules_ln_2_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_2_modules_ln_2_parameters_bias_
        ) = None
        view_31 = hidden_states_16.view(-1, 768)
        hidden_states_16 = None
        x_20 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_bias_,
            view_31,
            l_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_bias_ = (
            view_31
        ) = l_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_weight_ = (None)
        x_21 = x_20.view((1, 19, 3072))
        x_20 = None
        mul_8 = 0.5 * x_21
        pow_3 = torch.pow(x_21, 3.0)
        mul_9 = 0.044715 * pow_3
        pow_3 = None
        add_10 = x_21 + mul_9
        x_21 = mul_9 = None
        mul_10 = 0.7978845608028654 * add_10
        add_10 = None
        tanh_2 = torch.tanh(mul_10)
        mul_10 = None
        add_11 = 1.0 + tanh_2
        tanh_2 = None
        hidden_states_17 = mul_8 * add_11
        mul_8 = add_11 = None
        view_33 = hidden_states_17.view(-1, 3072)
        hidden_states_17 = None
        x_22 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_bias_,
            view_33,
            l_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_bias_ = (
            view_33
        ) = l_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_weight_ = (None)
        x_23 = x_22.view((1, 19, 768))
        x_22 = None
        hidden_states_18 = torch.nn.functional.dropout(x_23, 0.1, False, False)
        x_23 = None
        hidden_states_19 = hidden_states_15 + hidden_states_18
        hidden_states_15 = hidden_states_18 = None
        hidden_states_20 = torch.nn.functional.layer_norm(
            hidden_states_19,
            (768,),
            l_self_modules_transformer_modules_h_modules_3_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_3_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_3_modules_ln_1_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_3_modules_ln_1_parameters_bias_
        ) = None
        view_35 = hidden_states_20.view(-1, 768)
        hidden_states_20 = None
        x_24 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_attn_parameters_bias_,
            view_35,
            l_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_attn_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_attn_parameters_bias_ = (
            view_35
        ) = l_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_attn_parameters_weight_ = (None)
        x_25 = x_24.view((1, 19, 2304))
        x_24 = None
        split_3 = x_25.split(768, dim=2)
        x_25 = None
        query_states_6 = split_3[0]
        key_states_6 = split_3[1]
        value_states_6 = split_3[2]
        split_3 = None
        view_37 = key_states_6.view((1, 19, -1, 64))
        key_states_6 = None
        key_states_7 = view_37.transpose(1, 2)
        view_37 = None
        view_38 = value_states_6.view((1, 19, -1, 64))
        value_states_6 = None
        value_states_7 = view_38.transpose(1, 2)
        view_38 = None
        view_39 = query_states_6.view((1, 19, -1, 64))
        query_states_6 = None
        query_states_7 = view_39.transpose(1, 2)
        view_39 = None
        attention_mask_5 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_3 = query_states_7.contiguous()
        query_states_7 = None
        key_3 = key_states_7.contiguous()
        key_states_7 = None
        value_3 = value_states_7.contiguous()
        value_states_7 = None
        attn_output_12 = torch._C._nn.scaled_dot_product_attention(
            query_3,
            key_3,
            value_3,
            attn_mask=attention_mask_5,
            dropout_p=0.0,
            scale=None,
            is_causal=False,
        )
        query_3 = key_3 = value_3 = attention_mask_5 = None
        transpose_15 = attn_output_12.transpose(1, 2)
        attn_output_12 = None
        attn_output_13 = transpose_15.contiguous()
        transpose_15 = None
        reshape_3 = attn_output_13.reshape(1, 19, -1)
        attn_output_13 = None
        attn_output_14 = reshape_3.contiguous()
        reshape_3 = None
        view_40 = attn_output_14.view(-1, 768)
        attn_output_14 = None
        x_26 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_proj_parameters_bias_,
            view_40,
            l_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_proj_parameters_bias_ = (
            view_40
        ) = l_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_proj_parameters_weight_ = (None)
        x_27 = x_26.view((1, 19, 768))
        x_26 = None
        attn_output_15 = torch.nn.functional.dropout(x_27, 0.1, False, False)
        x_27 = None
        hidden_states_21 = attn_output_15 + hidden_states_19
        attn_output_15 = hidden_states_19 = None
        hidden_states_22 = torch.nn.functional.layer_norm(
            hidden_states_21,
            (768,),
            l_self_modules_transformer_modules_h_modules_3_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_3_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_3_modules_ln_2_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_3_modules_ln_2_parameters_bias_
        ) = None
        view_42 = hidden_states_22.view(-1, 768)
        hidden_states_22 = None
        x_28 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_bias_,
            view_42,
            l_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_bias_ = (
            view_42
        ) = l_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_weight_ = (None)
        x_29 = x_28.view((1, 19, 3072))
        x_28 = None
        mul_12 = 0.5 * x_29
        pow_4 = torch.pow(x_29, 3.0)
        mul_13 = 0.044715 * pow_4
        pow_4 = None
        add_14 = x_29 + mul_13
        x_29 = mul_13 = None
        mul_14 = 0.7978845608028654 * add_14
        add_14 = None
        tanh_3 = torch.tanh(mul_14)
        mul_14 = None
        add_15 = 1.0 + tanh_3
        tanh_3 = None
        hidden_states_23 = mul_12 * add_15
        mul_12 = add_15 = None
        view_44 = hidden_states_23.view(-1, 3072)
        hidden_states_23 = None
        x_30 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_bias_,
            view_44,
            l_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_bias_ = (
            view_44
        ) = l_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_weight_ = (None)
        x_31 = x_30.view((1, 19, 768))
        x_30 = None
        hidden_states_24 = torch.nn.functional.dropout(x_31, 0.1, False, False)
        x_31 = None
        hidden_states_25 = hidden_states_21 + hidden_states_24
        hidden_states_21 = hidden_states_24 = None
        hidden_states_26 = torch.nn.functional.layer_norm(
            hidden_states_25,
            (768,),
            l_self_modules_transformer_modules_h_modules_4_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_4_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_4_modules_ln_1_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_4_modules_ln_1_parameters_bias_
        ) = None
        view_46 = hidden_states_26.view(-1, 768)
        hidden_states_26 = None
        x_32 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_attn_parameters_bias_,
            view_46,
            l_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_attn_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_attn_parameters_bias_ = (
            view_46
        ) = l_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_attn_parameters_weight_ = (None)
        x_33 = x_32.view((1, 19, 2304))
        x_32 = None
        split_4 = x_33.split(768, dim=2)
        x_33 = None
        query_states_8 = split_4[0]
        key_states_8 = split_4[1]
        value_states_8 = split_4[2]
        split_4 = None
        view_48 = key_states_8.view((1, 19, -1, 64))
        key_states_8 = None
        key_states_9 = view_48.transpose(1, 2)
        view_48 = None
        view_49 = value_states_8.view((1, 19, -1, 64))
        value_states_8 = None
        value_states_9 = view_49.transpose(1, 2)
        view_49 = None
        view_50 = query_states_8.view((1, 19, -1, 64))
        query_states_8 = None
        query_states_9 = view_50.transpose(1, 2)
        view_50 = None
        attention_mask_6 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_4 = query_states_9.contiguous()
        query_states_9 = None
        key_4 = key_states_9.contiguous()
        key_states_9 = None
        value_4 = value_states_9.contiguous()
        value_states_9 = None
        attn_output_16 = torch._C._nn.scaled_dot_product_attention(
            query_4,
            key_4,
            value_4,
            attn_mask=attention_mask_6,
            dropout_p=0.0,
            scale=None,
            is_causal=False,
        )
        query_4 = key_4 = value_4 = attention_mask_6 = None
        transpose_19 = attn_output_16.transpose(1, 2)
        attn_output_16 = None
        attn_output_17 = transpose_19.contiguous()
        transpose_19 = None
        reshape_4 = attn_output_17.reshape(1, 19, -1)
        attn_output_17 = None
        attn_output_18 = reshape_4.contiguous()
        reshape_4 = None
        view_51 = attn_output_18.view(-1, 768)
        attn_output_18 = None
        x_34 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_proj_parameters_bias_,
            view_51,
            l_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_proj_parameters_bias_ = (
            view_51
        ) = l_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_proj_parameters_weight_ = (None)
        x_35 = x_34.view((1, 19, 768))
        x_34 = None
        attn_output_19 = torch.nn.functional.dropout(x_35, 0.1, False, False)
        x_35 = None
        hidden_states_27 = attn_output_19 + hidden_states_25
        attn_output_19 = hidden_states_25 = None
        hidden_states_28 = torch.nn.functional.layer_norm(
            hidden_states_27,
            (768,),
            l_self_modules_transformer_modules_h_modules_4_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_4_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_4_modules_ln_2_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_4_modules_ln_2_parameters_bias_
        ) = None
        view_53 = hidden_states_28.view(-1, 768)
        hidden_states_28 = None
        x_36 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_bias_,
            view_53,
            l_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_bias_ = (
            view_53
        ) = l_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_weight_ = (None)
        x_37 = x_36.view((1, 19, 3072))
        x_36 = None
        mul_16 = 0.5 * x_37
        pow_5 = torch.pow(x_37, 3.0)
        mul_17 = 0.044715 * pow_5
        pow_5 = None
        add_18 = x_37 + mul_17
        x_37 = mul_17 = None
        mul_18 = 0.7978845608028654 * add_18
        add_18 = None
        tanh_4 = torch.tanh(mul_18)
        mul_18 = None
        add_19 = 1.0 + tanh_4
        tanh_4 = None
        hidden_states_29 = mul_16 * add_19
        mul_16 = add_19 = None
        view_55 = hidden_states_29.view(-1, 3072)
        hidden_states_29 = None
        x_38 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_bias_,
            view_55,
            l_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_bias_ = (
            view_55
        ) = l_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_weight_ = (None)
        x_39 = x_38.view((1, 19, 768))
        x_38 = None
        hidden_states_30 = torch.nn.functional.dropout(x_39, 0.1, False, False)
        x_39 = None
        hidden_states_31 = hidden_states_27 + hidden_states_30
        hidden_states_27 = hidden_states_30 = None
        hidden_states_32 = torch.nn.functional.layer_norm(
            hidden_states_31,
            (768,),
            l_self_modules_transformer_modules_h_modules_5_modules_ln_1_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_5_modules_ln_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_5_modules_ln_1_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_5_modules_ln_1_parameters_bias_
        ) = None
        view_57 = hidden_states_32.view(-1, 768)
        hidden_states_32 = None
        x_40 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_attn_parameters_bias_,
            view_57,
            l_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_attn_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_attn_parameters_bias_ = (
            view_57
        ) = l_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_attn_parameters_weight_ = (None)
        x_41 = x_40.view((1, 19, 2304))
        x_40 = None
        split_5 = x_41.split(768, dim=2)
        x_41 = None
        query_states_10 = split_5[0]
        key_states_10 = split_5[1]
        value_states_10 = split_5[2]
        split_5 = None
        view_59 = key_states_10.view((1, 19, -1, 64))
        key_states_10 = None
        key_states_11 = view_59.transpose(1, 2)
        view_59 = None
        view_60 = value_states_10.view((1, 19, -1, 64))
        value_states_10 = None
        value_states_11 = view_60.transpose(1, 2)
        view_60 = None
        view_61 = query_states_10.view((1, 19, -1, 64))
        query_states_10 = None
        query_states_11 = view_61.transpose(1, 2)
        view_61 = None
        attention_mask_7 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        causal_mask = None
        query_5 = query_states_11.contiguous()
        query_states_11 = None
        key_5 = key_states_11.contiguous()
        key_states_11 = None
        value_5 = value_states_11.contiguous()
        value_states_11 = None
        attn_output_20 = torch._C._nn.scaled_dot_product_attention(
            query_5,
            key_5,
            value_5,
            attn_mask=attention_mask_7,
            dropout_p=0.0,
            scale=None,
            is_causal=False,
        )
        query_5 = key_5 = value_5 = attention_mask_7 = None
        transpose_23 = attn_output_20.transpose(1, 2)
        attn_output_20 = None
        attn_output_21 = transpose_23.contiguous()
        transpose_23 = None
        reshape_5 = attn_output_21.reshape(1, 19, -1)
        attn_output_21 = None
        attn_output_22 = reshape_5.contiguous()
        reshape_5 = None
        view_62 = attn_output_22.view(-1, 768)
        attn_output_22 = None
        x_42 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_proj_parameters_bias_,
            view_62,
            l_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_proj_parameters_bias_ = (
            view_62
        ) = l_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_proj_parameters_weight_ = (None)
        x_43 = x_42.view((1, 19, 768))
        x_42 = None
        attn_output_23 = torch.nn.functional.dropout(x_43, 0.1, False, False)
        x_43 = None
        hidden_states_33 = attn_output_23 + hidden_states_31
        attn_output_23 = hidden_states_31 = None
        hidden_states_34 = torch.nn.functional.layer_norm(
            hidden_states_33,
            (768,),
            l_self_modules_transformer_modules_h_modules_5_modules_ln_2_parameters_weight_,
            l_self_modules_transformer_modules_h_modules_5_modules_ln_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_transformer_modules_h_modules_5_modules_ln_2_parameters_weight_ = (
            l_self_modules_transformer_modules_h_modules_5_modules_ln_2_parameters_bias_
        ) = None
        view_64 = hidden_states_34.view(-1, 768)
        hidden_states_34 = None
        x_44 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_bias_,
            view_64,
            l_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_bias_ = (
            view_64
        ) = l_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_weight_ = (None)
        x_45 = x_44.view((1, 19, 3072))
        x_44 = None
        mul_20 = 0.5 * x_45
        pow_6 = torch.pow(x_45, 3.0)
        mul_21 = 0.044715 * pow_6
        pow_6 = None
        add_22 = x_45 + mul_21
        x_45 = mul_21 = None
        mul_22 = 0.7978845608028654 * add_22
        add_22 = None
        tanh_5 = torch.tanh(mul_22)
        mul_22 = None
        add_23 = 1.0 + tanh_5
        tanh_5 = None
        hidden_states_35 = mul_20 * add_23
        mul_20 = add_23 = None
        view_66 = hidden_states_35.view(-1, 3072)
        hidden_states_35 = None
        x_46 = torch.addmm(
            l_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_bias_,
            view_66,
            l_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_weight_,
        )
        l_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_bias_ = (
            view_66
        ) = l_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_weight_ = (None)
        x_47 = x_46.view((1, 19, 768))
        x_46 = None
        hidden_states_36 = torch.nn.functional.dropout(x_47, 0.1, False, False)
        x_47 = None
        hidden_states_37 = hidden_states_33 + hidden_states_36
        hidden_states_33 = hidden_states_36 = None
        hidden_states_38 = torch.nn.functional.layer_norm(
            hidden_states_37,
            (768,),
            l_self_modules_transformer_modules_ln_f_parameters_weight_,
            l_self_modules_transformer_modules_ln_f_parameters_bias_,
            1e-05,
        )
        hidden_states_37 = (
            l_self_modules_transformer_modules_ln_f_parameters_weight_
        ) = l_self_modules_transformer_modules_ln_f_parameters_bias_ = None
        hidden_states_39 = hidden_states_38.view((-1, 19, 768))
        hidden_states_38 = None
        getitem_24 = hidden_states_39[
            (slice(None, None, None), slice(0, None, None), slice(None, None, None))
        ]
        hidden_states_39 = None
        logits = torch._C._nn.linear(
            getitem_24, l_self_modules_transformer_modules_wte_parameters_weight_, None
        )
        getitem_24 = l_self_modules_transformer_modules_wte_parameters_weight_ = None
        return (logits,)
