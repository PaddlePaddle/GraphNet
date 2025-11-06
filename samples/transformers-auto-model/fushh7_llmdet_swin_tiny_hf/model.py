import torch

from torch import device

from torch import inf


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_stack0_encoder_last_hidden_state_text: torch.Tensor,
        L_stack0_intermediate_hidden_states: torch.Tensor,
        L_stack0_init_reference_points: torch.Tensor,
        L_stack0_intermediate_reference_points: torch.Tensor,
        L_attention_mask_: torch.Tensor,
        L_self_modules_bbox_embed_modules_0_modules_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bbox_embed_modules_0_modules_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bbox_embed_modules_0_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bbox_embed_modules_0_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bbox_embed_modules_0_modules_layers_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bbox_embed_modules_0_modules_layers_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bbox_embed_modules_1_modules_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bbox_embed_modules_1_modules_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bbox_embed_modules_1_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bbox_embed_modules_1_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bbox_embed_modules_1_modules_layers_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bbox_embed_modules_1_modules_layers_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bbox_embed_modules_2_modules_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bbox_embed_modules_2_modules_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bbox_embed_modules_2_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bbox_embed_modules_2_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bbox_embed_modules_2_modules_layers_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bbox_embed_modules_2_modules_layers_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bbox_embed_modules_3_modules_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bbox_embed_modules_3_modules_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bbox_embed_modules_3_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bbox_embed_modules_3_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bbox_embed_modules_3_modules_layers_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bbox_embed_modules_3_modules_layers_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bbox_embed_modules_4_modules_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bbox_embed_modules_4_modules_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bbox_embed_modules_4_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bbox_embed_modules_4_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bbox_embed_modules_4_modules_layers_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bbox_embed_modules_4_modules_layers_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bbox_embed_modules_5_modules_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bbox_embed_modules_5_modules_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bbox_embed_modules_5_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bbox_embed_modules_5_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bbox_embed_modules_5_modules_layers_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bbox_embed_modules_5_modules_layers_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_stack0_encoder_last_hidden_state_text = (
            L_stack0_encoder_last_hidden_state_text
        )
        l_stack0_intermediate_hidden_states = L_stack0_intermediate_hidden_states
        l_stack0_init_reference_points = L_stack0_init_reference_points
        l_stack0_intermediate_reference_points = L_stack0_intermediate_reference_points
        l_attention_mask_ = L_attention_mask_
        l_self_modules_bbox_embed_modules_0_modules_layers_modules_0_parameters_weight_ = L_self_modules_bbox_embed_modules_0_modules_layers_modules_0_parameters_weight_
        l_self_modules_bbox_embed_modules_0_modules_layers_modules_0_parameters_bias_ = L_self_modules_bbox_embed_modules_0_modules_layers_modules_0_parameters_bias_
        l_self_modules_bbox_embed_modules_0_modules_layers_modules_1_parameters_weight_ = L_self_modules_bbox_embed_modules_0_modules_layers_modules_1_parameters_weight_
        l_self_modules_bbox_embed_modules_0_modules_layers_modules_1_parameters_bias_ = L_self_modules_bbox_embed_modules_0_modules_layers_modules_1_parameters_bias_
        l_self_modules_bbox_embed_modules_0_modules_layers_modules_2_parameters_weight_ = L_self_modules_bbox_embed_modules_0_modules_layers_modules_2_parameters_weight_
        l_self_modules_bbox_embed_modules_0_modules_layers_modules_2_parameters_bias_ = L_self_modules_bbox_embed_modules_0_modules_layers_modules_2_parameters_bias_
        l_self_modules_bbox_embed_modules_1_modules_layers_modules_0_parameters_weight_ = L_self_modules_bbox_embed_modules_1_modules_layers_modules_0_parameters_weight_
        l_self_modules_bbox_embed_modules_1_modules_layers_modules_0_parameters_bias_ = L_self_modules_bbox_embed_modules_1_modules_layers_modules_0_parameters_bias_
        l_self_modules_bbox_embed_modules_1_modules_layers_modules_1_parameters_weight_ = L_self_modules_bbox_embed_modules_1_modules_layers_modules_1_parameters_weight_
        l_self_modules_bbox_embed_modules_1_modules_layers_modules_1_parameters_bias_ = L_self_modules_bbox_embed_modules_1_modules_layers_modules_1_parameters_bias_
        l_self_modules_bbox_embed_modules_1_modules_layers_modules_2_parameters_weight_ = L_self_modules_bbox_embed_modules_1_modules_layers_modules_2_parameters_weight_
        l_self_modules_bbox_embed_modules_1_modules_layers_modules_2_parameters_bias_ = L_self_modules_bbox_embed_modules_1_modules_layers_modules_2_parameters_bias_
        l_self_modules_bbox_embed_modules_2_modules_layers_modules_0_parameters_weight_ = L_self_modules_bbox_embed_modules_2_modules_layers_modules_0_parameters_weight_
        l_self_modules_bbox_embed_modules_2_modules_layers_modules_0_parameters_bias_ = L_self_modules_bbox_embed_modules_2_modules_layers_modules_0_parameters_bias_
        l_self_modules_bbox_embed_modules_2_modules_layers_modules_1_parameters_weight_ = L_self_modules_bbox_embed_modules_2_modules_layers_modules_1_parameters_weight_
        l_self_modules_bbox_embed_modules_2_modules_layers_modules_1_parameters_bias_ = L_self_modules_bbox_embed_modules_2_modules_layers_modules_1_parameters_bias_
        l_self_modules_bbox_embed_modules_2_modules_layers_modules_2_parameters_weight_ = L_self_modules_bbox_embed_modules_2_modules_layers_modules_2_parameters_weight_
        l_self_modules_bbox_embed_modules_2_modules_layers_modules_2_parameters_bias_ = L_self_modules_bbox_embed_modules_2_modules_layers_modules_2_parameters_bias_
        l_self_modules_bbox_embed_modules_3_modules_layers_modules_0_parameters_weight_ = L_self_modules_bbox_embed_modules_3_modules_layers_modules_0_parameters_weight_
        l_self_modules_bbox_embed_modules_3_modules_layers_modules_0_parameters_bias_ = L_self_modules_bbox_embed_modules_3_modules_layers_modules_0_parameters_bias_
        l_self_modules_bbox_embed_modules_3_modules_layers_modules_1_parameters_weight_ = L_self_modules_bbox_embed_modules_3_modules_layers_modules_1_parameters_weight_
        l_self_modules_bbox_embed_modules_3_modules_layers_modules_1_parameters_bias_ = L_self_modules_bbox_embed_modules_3_modules_layers_modules_1_parameters_bias_
        l_self_modules_bbox_embed_modules_3_modules_layers_modules_2_parameters_weight_ = L_self_modules_bbox_embed_modules_3_modules_layers_modules_2_parameters_weight_
        l_self_modules_bbox_embed_modules_3_modules_layers_modules_2_parameters_bias_ = L_self_modules_bbox_embed_modules_3_modules_layers_modules_2_parameters_bias_
        l_self_modules_bbox_embed_modules_4_modules_layers_modules_0_parameters_weight_ = L_self_modules_bbox_embed_modules_4_modules_layers_modules_0_parameters_weight_
        l_self_modules_bbox_embed_modules_4_modules_layers_modules_0_parameters_bias_ = L_self_modules_bbox_embed_modules_4_modules_layers_modules_0_parameters_bias_
        l_self_modules_bbox_embed_modules_4_modules_layers_modules_1_parameters_weight_ = L_self_modules_bbox_embed_modules_4_modules_layers_modules_1_parameters_weight_
        l_self_modules_bbox_embed_modules_4_modules_layers_modules_1_parameters_bias_ = L_self_modules_bbox_embed_modules_4_modules_layers_modules_1_parameters_bias_
        l_self_modules_bbox_embed_modules_4_modules_layers_modules_2_parameters_weight_ = L_self_modules_bbox_embed_modules_4_modules_layers_modules_2_parameters_weight_
        l_self_modules_bbox_embed_modules_4_modules_layers_modules_2_parameters_bias_ = L_self_modules_bbox_embed_modules_4_modules_layers_modules_2_parameters_bias_
        l_self_modules_bbox_embed_modules_5_modules_layers_modules_0_parameters_weight_ = L_self_modules_bbox_embed_modules_5_modules_layers_modules_0_parameters_weight_
        l_self_modules_bbox_embed_modules_5_modules_layers_modules_0_parameters_bias_ = L_self_modules_bbox_embed_modules_5_modules_layers_modules_0_parameters_bias_
        l_self_modules_bbox_embed_modules_5_modules_layers_modules_1_parameters_weight_ = L_self_modules_bbox_embed_modules_5_modules_layers_modules_1_parameters_weight_
        l_self_modules_bbox_embed_modules_5_modules_layers_modules_1_parameters_bias_ = L_self_modules_bbox_embed_modules_5_modules_layers_modules_1_parameters_bias_
        l_self_modules_bbox_embed_modules_5_modules_layers_modules_2_parameters_weight_ = L_self_modules_bbox_embed_modules_5_modules_layers_modules_2_parameters_weight_
        l_self_modules_bbox_embed_modules_5_modules_layers_modules_2_parameters_bias_ = L_self_modules_bbox_embed_modules_5_modules_layers_modules_2_parameters_bias_
        reference = torch._C._special.special_logit(
            l_stack0_init_reference_points, eps=1e-05
        )
        l_stack0_init_reference_points = None
        getitem = l_stack0_intermediate_hidden_states[(slice(None, None, None), 0)]
        bool_1 = l_attention_mask_.bool()
        transpose = l_stack0_encoder_last_hidden_state_text.transpose(-1, -2)
        output = getitem @ transpose
        getitem = transpose = None
        getitem_1 = bool_1[(slice(None, None, None), None, slice(None, None, None))]
        bool_1 = None
        invert = ~getitem_1
        getitem_1 = None
        output_1 = output.masked_fill(invert, -1e6)
        output = invert = None
        new_output = torch.full(
            (1, 900, 256), -1e6, device=device(type="cuda", index=0)
        )
        new_output[(Ellipsis, slice(None, 7, None))] = output_1
        setitem = new_output
        output_1 = setitem = None
        getitem_2 = l_stack0_intermediate_hidden_states[(slice(None, None, None), 0)]
        linear = torch._C._nn.linear(
            getitem_2,
            l_self_modules_bbox_embed_modules_0_modules_layers_modules_0_parameters_weight_,
            l_self_modules_bbox_embed_modules_0_modules_layers_modules_0_parameters_bias_,
        )
        getitem_2 = l_self_modules_bbox_embed_modules_0_modules_layers_modules_0_parameters_weight_ = l_self_modules_bbox_embed_modules_0_modules_layers_modules_0_parameters_bias_ = (None)
        x = torch.nn.functional.relu(linear)
        linear = None
        linear_1 = torch._C._nn.linear(
            x,
            l_self_modules_bbox_embed_modules_0_modules_layers_modules_1_parameters_weight_,
            l_self_modules_bbox_embed_modules_0_modules_layers_modules_1_parameters_bias_,
        )
        x = l_self_modules_bbox_embed_modules_0_modules_layers_modules_1_parameters_weight_ = l_self_modules_bbox_embed_modules_0_modules_layers_modules_1_parameters_bias_ = (None)
        x_1 = torch.nn.functional.relu(linear_1)
        linear_1 = None
        x_2 = torch._C._nn.linear(
            x_1,
            l_self_modules_bbox_embed_modules_0_modules_layers_modules_2_parameters_weight_,
            l_self_modules_bbox_embed_modules_0_modules_layers_modules_2_parameters_bias_,
        )
        x_1 = l_self_modules_bbox_embed_modules_0_modules_layers_modules_2_parameters_weight_ = l_self_modules_bbox_embed_modules_0_modules_layers_modules_2_parameters_bias_ = (None)
        outputs_coord_logits = x_2 + reference
        x_2 = reference = None
        outputs_coord = outputs_coord_logits.sigmoid()
        outputs_coord_logits = None
        reference_1 = l_stack0_intermediate_reference_points[
            (slice(None, None, None), 0)
        ]
        reference_2 = torch._C._special.special_logit(reference_1, eps=1e-05)
        reference_1 = None
        getitem_4 = l_stack0_intermediate_hidden_states[(slice(None, None, None), 1)]
        bool_2 = l_attention_mask_.bool()
        transpose_1 = l_stack0_encoder_last_hidden_state_text.transpose(-1, -2)
        output_2 = getitem_4 @ transpose_1
        getitem_4 = transpose_1 = None
        getitem_5 = bool_2[(slice(None, None, None), None, slice(None, None, None))]
        bool_2 = None
        invert_1 = ~getitem_5
        getitem_5 = None
        output_3 = output_2.masked_fill(invert_1, -1e6)
        output_2 = invert_1 = None
        new_output_1 = torch.full(
            (1, 900, 256), -1e6, device=device(type="cuda", index=0)
        )
        new_output_1[(Ellipsis, slice(None, 7, None))] = output_3
        setitem_1 = new_output_1
        output_3 = setitem_1 = None
        getitem_6 = l_stack0_intermediate_hidden_states[(slice(None, None, None), 1)]
        linear_3 = torch._C._nn.linear(
            getitem_6,
            l_self_modules_bbox_embed_modules_1_modules_layers_modules_0_parameters_weight_,
            l_self_modules_bbox_embed_modules_1_modules_layers_modules_0_parameters_bias_,
        )
        getitem_6 = l_self_modules_bbox_embed_modules_1_modules_layers_modules_0_parameters_weight_ = l_self_modules_bbox_embed_modules_1_modules_layers_modules_0_parameters_bias_ = (None)
        x_3 = torch.nn.functional.relu(linear_3)
        linear_3 = None
        linear_4 = torch._C._nn.linear(
            x_3,
            l_self_modules_bbox_embed_modules_1_modules_layers_modules_1_parameters_weight_,
            l_self_modules_bbox_embed_modules_1_modules_layers_modules_1_parameters_bias_,
        )
        x_3 = l_self_modules_bbox_embed_modules_1_modules_layers_modules_1_parameters_weight_ = l_self_modules_bbox_embed_modules_1_modules_layers_modules_1_parameters_bias_ = (None)
        x_4 = torch.nn.functional.relu(linear_4)
        linear_4 = None
        x_5 = torch._C._nn.linear(
            x_4,
            l_self_modules_bbox_embed_modules_1_modules_layers_modules_2_parameters_weight_,
            l_self_modules_bbox_embed_modules_1_modules_layers_modules_2_parameters_bias_,
        )
        x_4 = l_self_modules_bbox_embed_modules_1_modules_layers_modules_2_parameters_weight_ = l_self_modules_bbox_embed_modules_1_modules_layers_modules_2_parameters_bias_ = (None)
        outputs_coord_logits_1 = x_5 + reference_2
        x_5 = reference_2 = None
        outputs_coord_1 = outputs_coord_logits_1.sigmoid()
        outputs_coord_logits_1 = None
        reference_3 = l_stack0_intermediate_reference_points[
            (slice(None, None, None), 1)
        ]
        reference_4 = torch._C._special.special_logit(reference_3, eps=1e-05)
        reference_3 = None
        getitem_8 = l_stack0_intermediate_hidden_states[(slice(None, None, None), 2)]
        bool_3 = l_attention_mask_.bool()
        transpose_2 = l_stack0_encoder_last_hidden_state_text.transpose(-1, -2)
        output_4 = getitem_8 @ transpose_2
        getitem_8 = transpose_2 = None
        getitem_9 = bool_3[(slice(None, None, None), None, slice(None, None, None))]
        bool_3 = None
        invert_2 = ~getitem_9
        getitem_9 = None
        output_5 = output_4.masked_fill(invert_2, -1e6)
        output_4 = invert_2 = None
        new_output_2 = torch.full(
            (1, 900, 256), -1e6, device=device(type="cuda", index=0)
        )
        new_output_2[(Ellipsis, slice(None, 7, None))] = output_5
        setitem_2 = new_output_2
        output_5 = setitem_2 = None
        getitem_10 = l_stack0_intermediate_hidden_states[(slice(None, None, None), 2)]
        linear_6 = torch._C._nn.linear(
            getitem_10,
            l_self_modules_bbox_embed_modules_2_modules_layers_modules_0_parameters_weight_,
            l_self_modules_bbox_embed_modules_2_modules_layers_modules_0_parameters_bias_,
        )
        getitem_10 = l_self_modules_bbox_embed_modules_2_modules_layers_modules_0_parameters_weight_ = l_self_modules_bbox_embed_modules_2_modules_layers_modules_0_parameters_bias_ = (None)
        x_6 = torch.nn.functional.relu(linear_6)
        linear_6 = None
        linear_7 = torch._C._nn.linear(
            x_6,
            l_self_modules_bbox_embed_modules_2_modules_layers_modules_1_parameters_weight_,
            l_self_modules_bbox_embed_modules_2_modules_layers_modules_1_parameters_bias_,
        )
        x_6 = l_self_modules_bbox_embed_modules_2_modules_layers_modules_1_parameters_weight_ = l_self_modules_bbox_embed_modules_2_modules_layers_modules_1_parameters_bias_ = (None)
        x_7 = torch.nn.functional.relu(linear_7)
        linear_7 = None
        x_8 = torch._C._nn.linear(
            x_7,
            l_self_modules_bbox_embed_modules_2_modules_layers_modules_2_parameters_weight_,
            l_self_modules_bbox_embed_modules_2_modules_layers_modules_2_parameters_bias_,
        )
        x_7 = l_self_modules_bbox_embed_modules_2_modules_layers_modules_2_parameters_weight_ = l_self_modules_bbox_embed_modules_2_modules_layers_modules_2_parameters_bias_ = (None)
        outputs_coord_logits_2 = x_8 + reference_4
        x_8 = reference_4 = None
        outputs_coord_2 = outputs_coord_logits_2.sigmoid()
        outputs_coord_logits_2 = None
        reference_5 = l_stack0_intermediate_reference_points[
            (slice(None, None, None), 2)
        ]
        reference_6 = torch._C._special.special_logit(reference_5, eps=1e-05)
        reference_5 = None
        getitem_12 = l_stack0_intermediate_hidden_states[(slice(None, None, None), 3)]
        bool_4 = l_attention_mask_.bool()
        transpose_3 = l_stack0_encoder_last_hidden_state_text.transpose(-1, -2)
        output_6 = getitem_12 @ transpose_3
        getitem_12 = transpose_3 = None
        getitem_13 = bool_4[(slice(None, None, None), None, slice(None, None, None))]
        bool_4 = None
        invert_3 = ~getitem_13
        getitem_13 = None
        output_7 = output_6.masked_fill(invert_3, -1e6)
        output_6 = invert_3 = None
        new_output_3 = torch.full(
            (1, 900, 256), -1e6, device=device(type="cuda", index=0)
        )
        new_output_3[(Ellipsis, slice(None, 7, None))] = output_7
        setitem_3 = new_output_3
        output_7 = setitem_3 = None
        getitem_14 = l_stack0_intermediate_hidden_states[(slice(None, None, None), 3)]
        linear_9 = torch._C._nn.linear(
            getitem_14,
            l_self_modules_bbox_embed_modules_3_modules_layers_modules_0_parameters_weight_,
            l_self_modules_bbox_embed_modules_3_modules_layers_modules_0_parameters_bias_,
        )
        getitem_14 = l_self_modules_bbox_embed_modules_3_modules_layers_modules_0_parameters_weight_ = l_self_modules_bbox_embed_modules_3_modules_layers_modules_0_parameters_bias_ = (None)
        x_9 = torch.nn.functional.relu(linear_9)
        linear_9 = None
        linear_10 = torch._C._nn.linear(
            x_9,
            l_self_modules_bbox_embed_modules_3_modules_layers_modules_1_parameters_weight_,
            l_self_modules_bbox_embed_modules_3_modules_layers_modules_1_parameters_bias_,
        )
        x_9 = l_self_modules_bbox_embed_modules_3_modules_layers_modules_1_parameters_weight_ = l_self_modules_bbox_embed_modules_3_modules_layers_modules_1_parameters_bias_ = (None)
        x_10 = torch.nn.functional.relu(linear_10)
        linear_10 = None
        x_11 = torch._C._nn.linear(
            x_10,
            l_self_modules_bbox_embed_modules_3_modules_layers_modules_2_parameters_weight_,
            l_self_modules_bbox_embed_modules_3_modules_layers_modules_2_parameters_bias_,
        )
        x_10 = l_self_modules_bbox_embed_modules_3_modules_layers_modules_2_parameters_weight_ = l_self_modules_bbox_embed_modules_3_modules_layers_modules_2_parameters_bias_ = (None)
        outputs_coord_logits_3 = x_11 + reference_6
        x_11 = reference_6 = None
        outputs_coord_3 = outputs_coord_logits_3.sigmoid()
        outputs_coord_logits_3 = None
        reference_7 = l_stack0_intermediate_reference_points[
            (slice(None, None, None), 3)
        ]
        reference_8 = torch._C._special.special_logit(reference_7, eps=1e-05)
        reference_7 = None
        getitem_16 = l_stack0_intermediate_hidden_states[(slice(None, None, None), 4)]
        bool_5 = l_attention_mask_.bool()
        transpose_4 = l_stack0_encoder_last_hidden_state_text.transpose(-1, -2)
        output_8 = getitem_16 @ transpose_4
        getitem_16 = transpose_4 = None
        getitem_17 = bool_5[(slice(None, None, None), None, slice(None, None, None))]
        bool_5 = None
        invert_4 = ~getitem_17
        getitem_17 = None
        output_9 = output_8.masked_fill(invert_4, -1e6)
        output_8 = invert_4 = None
        new_output_4 = torch.full(
            (1, 900, 256), -1e6, device=device(type="cuda", index=0)
        )
        new_output_4[(Ellipsis, slice(None, 7, None))] = output_9
        setitem_4 = new_output_4
        output_9 = setitem_4 = None
        getitem_18 = l_stack0_intermediate_hidden_states[(slice(None, None, None), 4)]
        linear_12 = torch._C._nn.linear(
            getitem_18,
            l_self_modules_bbox_embed_modules_4_modules_layers_modules_0_parameters_weight_,
            l_self_modules_bbox_embed_modules_4_modules_layers_modules_0_parameters_bias_,
        )
        getitem_18 = l_self_modules_bbox_embed_modules_4_modules_layers_modules_0_parameters_weight_ = l_self_modules_bbox_embed_modules_4_modules_layers_modules_0_parameters_bias_ = (None)
        x_12 = torch.nn.functional.relu(linear_12)
        linear_12 = None
        linear_13 = torch._C._nn.linear(
            x_12,
            l_self_modules_bbox_embed_modules_4_modules_layers_modules_1_parameters_weight_,
            l_self_modules_bbox_embed_modules_4_modules_layers_modules_1_parameters_bias_,
        )
        x_12 = l_self_modules_bbox_embed_modules_4_modules_layers_modules_1_parameters_weight_ = l_self_modules_bbox_embed_modules_4_modules_layers_modules_1_parameters_bias_ = (None)
        x_13 = torch.nn.functional.relu(linear_13)
        linear_13 = None
        x_14 = torch._C._nn.linear(
            x_13,
            l_self_modules_bbox_embed_modules_4_modules_layers_modules_2_parameters_weight_,
            l_self_modules_bbox_embed_modules_4_modules_layers_modules_2_parameters_bias_,
        )
        x_13 = l_self_modules_bbox_embed_modules_4_modules_layers_modules_2_parameters_weight_ = l_self_modules_bbox_embed_modules_4_modules_layers_modules_2_parameters_bias_ = (None)
        outputs_coord_logits_4 = x_14 + reference_8
        x_14 = reference_8 = None
        outputs_coord_4 = outputs_coord_logits_4.sigmoid()
        outputs_coord_logits_4 = None
        reference_9 = l_stack0_intermediate_reference_points[
            (slice(None, None, None), 4)
        ]
        l_stack0_intermediate_reference_points = None
        reference_10 = torch._C._special.special_logit(reference_9, eps=1e-05)
        reference_9 = None
        getitem_20 = l_stack0_intermediate_hidden_states[(slice(None, None, None), 5)]
        bool_6 = l_attention_mask_.bool()
        l_attention_mask_ = None
        transpose_5 = l_stack0_encoder_last_hidden_state_text.transpose(-1, -2)
        l_stack0_encoder_last_hidden_state_text = None
        output_10 = getitem_20 @ transpose_5
        getitem_20 = transpose_5 = None
        getitem_21 = bool_6[(slice(None, None, None), None, slice(None, None, None))]
        bool_6 = None
        invert_5 = ~getitem_21
        getitem_21 = None
        output_11 = output_10.masked_fill(invert_5, -1e6)
        output_10 = invert_5 = None
        new_output_5 = torch.full(
            (1, 900, 256), -1e6, device=device(type="cuda", index=0)
        )
        new_output_5[(Ellipsis, slice(None, 7, None))] = output_11
        setitem_5 = new_output_5
        output_11 = setitem_5 = None
        getitem_22 = l_stack0_intermediate_hidden_states[(slice(None, None, None), 5)]
        l_stack0_intermediate_hidden_states = None
        linear_15 = torch._C._nn.linear(
            getitem_22,
            l_self_modules_bbox_embed_modules_5_modules_layers_modules_0_parameters_weight_,
            l_self_modules_bbox_embed_modules_5_modules_layers_modules_0_parameters_bias_,
        )
        getitem_22 = l_self_modules_bbox_embed_modules_5_modules_layers_modules_0_parameters_weight_ = l_self_modules_bbox_embed_modules_5_modules_layers_modules_0_parameters_bias_ = (None)
        x_15 = torch.nn.functional.relu(linear_15)
        linear_15 = None
        linear_16 = torch._C._nn.linear(
            x_15,
            l_self_modules_bbox_embed_modules_5_modules_layers_modules_1_parameters_weight_,
            l_self_modules_bbox_embed_modules_5_modules_layers_modules_1_parameters_bias_,
        )
        x_15 = l_self_modules_bbox_embed_modules_5_modules_layers_modules_1_parameters_weight_ = l_self_modules_bbox_embed_modules_5_modules_layers_modules_1_parameters_bias_ = (None)
        x_16 = torch.nn.functional.relu(linear_16)
        linear_16 = None
        x_17 = torch._C._nn.linear(
            x_16,
            l_self_modules_bbox_embed_modules_5_modules_layers_modules_2_parameters_weight_,
            l_self_modules_bbox_embed_modules_5_modules_layers_modules_2_parameters_bias_,
        )
        x_16 = l_self_modules_bbox_embed_modules_5_modules_layers_modules_2_parameters_weight_ = l_self_modules_bbox_embed_modules_5_modules_layers_modules_2_parameters_bias_ = (None)
        outputs_coord_logits_5 = x_17 + reference_10
        x_17 = reference_10 = None
        outputs_coord_5 = outputs_coord_logits_5.sigmoid()
        outputs_coord_logits_5 = None
        outputs_class = torch.stack(
            [
                new_output,
                new_output_1,
                new_output_2,
                new_output_3,
                new_output_4,
                new_output_5,
            ]
        )
        new_output = (
            new_output_1
        ) = new_output_2 = new_output_3 = new_output_4 = new_output_5 = None
        outputs_coord_6 = torch.stack(
            [
                outputs_coord,
                outputs_coord_1,
                outputs_coord_2,
                outputs_coord_3,
                outputs_coord_4,
                outputs_coord_5,
            ]
        )
        outputs_coord = (
            outputs_coord_1
        ) = outputs_coord_2 = outputs_coord_3 = outputs_coord_4 = outputs_coord_5 = None
        logits = outputs_class[-1]
        outputs_class = None
        pred_boxes = outputs_coord_6[-1]
        outputs_coord_6 = None
        return (logits, pred_boxes)
