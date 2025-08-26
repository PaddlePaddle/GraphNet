import typing as t
import numpy as np
from graph_net.torch.rp_expr.rp_expr import Tokenize, PrimitiveId, LetsListTokenRpExpr
from graph_net.torch.rp_expr.rp_expr_passes import (
    FlattenTokenListPass,
    FoldTokensPass,
    RecursiveFoldTokensPass,
    FoldIfTokenIdGreatEqualPass,
    UnflattenAndSubThresholdPass,
)


class RpExprParser:
    def __init__(self, window_size=8, fold_policy="default", fold_times=None):
        self.window_size = window_size
        self.fold_policy = fold_policy
        self.fold_times = fold_times

    def __call__(self, primitive_id_lists: t.List[t.List[PrimitiveId]]):
        token_list, id_allocator, token_id2primitive_id = Tokenize(primitive_id_lists)
        flatten_pass = FlattenTokenListPass(id_allocator)
        success, flattened_rp_expr = flatten_pass(token_list)
        assert success
        fold_pass = RecursiveFoldTokensPass(
            id_allocator,
            self.window_size,
            fold_policy=self.fold_policy,
            fold_times=self.fold_times,
        )
        success, fold_rp_expr = fold_pass(flattened_rp_expr.flattened_tensor)
        if not success:
            primitive_id2token_id = {
                primitive_id: token_id
                for token_id, primitive_id in enumerate(token_id2primitive_id)
            }
            lets_list_token_rp_expr = LetsListTokenRpExpr(
                symbol_token_ids=[],
                symbol_token_tensors=[],
                body_rp_expr=[
                    np.array(
                        [
                            primitive_id2token_id[primitive_id]
                            for primitive_id in primitive_ids
                        ],
                        dtype=np.int64,
                    )
                    for primitive_ids in primitive_id_lists
                ],
            )
            return lets_list_token_rp_expr, token_id2primitive_id
        assert success, f"{self.window_size=}, {self.fold_policy=}, {self.fold_times=}"
        threshold = len(primitive_id_lists)
        unflatten_pass = UnflattenAndSubThresholdPass(
            id_allocator=id_allocator,
            threshold_start_token_id=threshold,
        )
        success, threshold_fold_rp_expr = unflatten_pass(fold_rp_expr)
        assert success
        threshold_fold_rp_expr.inplace_group_consecutive_primitives(
            token_id2primitive_id
        )
        return threshold_fold_rp_expr, token_id2primitive_id
