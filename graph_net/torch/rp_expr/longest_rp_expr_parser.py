import typing as t
from graph_net.torch.rp_expr.rp_expr_parser import RpExprParser
from graph_net.torch.rp_expr.rp_expr import PrimitiveId, LetsListTokenRpExpr
import numpy as np
import sys


class LongestRpExprParser:
    def __init__(self, max_window_size=1024, min_window_size=4):
        self.max_window_size = max_window_size
        self.min_window_size = min_window_size

    def __call__(self, primitive_id_lists: t.List[t.List[PrimitiveId]]):
        fold_policy = "default"
        rp_expr_parser = RpExprParser(
            self.max_window_size,
            fold_policy=fold_policy,
            fold_times=1,
        )
        lets_list_rp_expr, token_id2primitive_id = rp_expr_parser(primitive_id_lists)
        for window_size in self._get_sub_window_sizes():
            rp_expr_parser = RpExprParser(
                window_size,
                fold_policy=fold_policy,
                fold_times=1,
            )
            cur_primitive_id_lists = [
                [token_id2primitive_id[token_id] for token_id in tensor.tolist()]
                for tensor in lets_list_rp_expr.get_pure_primitive_binding_tensors(
                    token_id2primitive_id
                )
            ]
            if not cur_primitive_id_lists:
                continue
            cur_lets_list_rp_expr, cur_token_id2primitive_id = rp_expr_parser(
                cur_primitive_id_lists
            )
            # cur_lets_list_rp_expr.try_unwrap_body_of_sole_symbol_token()
            lets_list_rp_expr = self._merge_lets_list_rp_expr(
                inner=cur_lets_list_rp_expr,
                outer=lets_list_rp_expr,
                inner_token_id2primitive_id=cur_token_id2primitive_id,
                outer_token_id2primitive_id=token_id2primitive_id,
            )
        lets_list_rp_expr.try_recursive_inline_symbol_sole_used(
            token_id2primitive_id=token_id2primitive_id
        )
        # lets_list_rp_expr.try_unwrap_body_of_sole_symbol_token()
        return lets_list_rp_expr, token_id2primitive_id

    def _merge_lets_list_rp_expr(
        self,
        inner,
        outer,
        inner_token_id2primitive_id,
        outer_token_id2primitive_id,
    ):
        def get_inner_token_id2outer_token_id():
            primitive_id2outer_token_id = {}
            for token_id, primitive_id in enumerate(outer_token_id2primitive_id):
                assert primitive_id not in primitive_id2outer_token_id
                primitive_id2outer_token_id[primitive_id] = token_id
            return [
                primitive_id2outer_token_id[primitive_id]
                for primitive_id in inner_token_id2primitive_id
            ]

        kInner = "inner"
        kOuter = "outer"
        uid2new_symbol_token = self._make_uid2new_symbol_token_id(
            inner=inner,
            outer=outer,
            inner_uid_prefix=kInner,
            outer_uid_prefix=kOuter,
            outer_primitive_table_size=len(outer_token_id2primitive_id),
        )
        inner_symbol_token_ids = self._convert_symbol_token_ids(
            symbol_token_ids=inner.symbol_token_ids,
            new_token4old_token=(
                lambda old_token: uid2new_symbol_token[f"{kInner}{old_token}"]
            ),
        )
        inner_token_id2outer_token_id = get_inner_token_id2outer_token_id()
        inner_symbol_token_tensors = self._convert_token_tensors(
            inner.symbol_token_tensors,
            new_token4old_primitive_token=(
                lambda old_token: inner_token_id2outer_token_id[old_token]
            ),
            new_token4old_symbol_token=(
                lambda old_token: uid2new_symbol_token[f"{kInner}{old_token}"]
            ),
            primitive_ids_table_size=len(inner_token_id2primitive_id),
        )

        inner_body_rp_expr = self._convert_token_tensors(
            inner.body_rp_expr,
            new_token4old_primitive_token=(
                lambda old_token: inner_token_id2outer_token_id[old_token]
            ),
            new_token4old_symbol_token=(
                lambda old_token: uid2new_symbol_token[f"{kInner}{old_token}"]
            ),
            primitive_ids_table_size=len(inner_token_id2primitive_id),
        )

        inner_symbol_token2token_tensor = {
            symbol_token: token_tensor
            for symbol_token, token_tensor in zip(
                inner_symbol_token_ids, inner_symbol_token_tensors
            )
        }

        outer_symbol_token_tensors = self._convert_outer_symbol_binding_token_tensors(
            inner_body_rp_expr=inner_body_rp_expr,
            inner_symbol_token2token_tensor=inner_symbol_token2token_tensor,
            outer_lets_list_rp_expr=outer,
            new_token4old_primitive_token=lambda x: x,
            new_token4old_symbol_token=(
                lambda old_token: uid2new_symbol_token[f"{kOuter}{old_token}"]
            ),
            outer_token_id2primitive_id=outer_token_id2primitive_id,
        )

        symbol_token_ids = inner_symbol_token_ids + self._convert_symbol_token_ids(
            symbol_token_ids=outer.symbol_token_ids,
            new_token4old_token=(
                lambda old_token: uid2new_symbol_token[f"{kOuter}{old_token}"]
            ),
        )

        symbol_token_tensors = inner_symbol_token_tensors + outer_symbol_token_tensors

        body_rp_expr = self._convert_token_tensors(
            outer.body_rp_expr,
            new_token4old_primitive_token=lambda x: x,
            new_token4old_symbol_token=(
                lambda old_token: uid2new_symbol_token[f"{kOuter}{old_token}"]
            ),
            primitive_ids_table_size=len(outer_token_id2primitive_id),
        )
        ret_lets_list_token_rp_expr = LetsListTokenRpExpr(
            symbol_token_ids=symbol_token_ids,
            symbol_token_tensors=symbol_token_tensors,
            body_rp_expr=body_rp_expr,
        )
        ret_lets_list_token_rp_expr.move_pure_primitive_bindings_front(
            outer_token_id2primitive_id
        )
        return ret_lets_list_token_rp_expr

    def _convert_outer_symbol_binding_token_tensors(
        self,
        inner_body_rp_expr,
        inner_symbol_token2token_tensor,
        outer_lets_list_rp_expr,
        new_token4old_primitive_token,
        new_token4old_symbol_token,
        outer_token_id2primitive_id,
    ):
        indexes = outer_lets_list_rp_expr.get_pure_primitive_binding_indexes(
            outer_token_id2primitive_id
        )
        assert len(inner_body_rp_expr) == len(indexes)
        index2inner_body_rp_expr_idx = {
            index: inner_body_rp_expr_idx
            for inner_body_rp_expr_idx, index in enumerate(indexes)
        }
        old_tensors = outer_lets_list_rp_expr.symbol_token_tensors
        return [
            (
                inner_body_rp_expr[index2inner_body_rp_expr_idx[index]]
                if index in index2inner_body_rp_expr_idx
                else self._convert_token_tensor(
                    tensor=old_tensors[index],
                    new_token4old_primitive_token=new_token4old_primitive_token,
                    new_token4old_symbol_token=new_token4old_symbol_token,
                    primitive_ids_table_size=len(outer_token_id2primitive_id),
                )
            )
            for index in range(len(old_tensors))
        ]

    def _convert_token_tensors(
        self,
        tensors,
        new_token4old_primitive_token,
        new_token4old_symbol_token,
        primitive_ids_table_size,
    ):
        return [
            self._convert_token_tensor(
                tensor,
                new_token4old_primitive_token,
                new_token4old_symbol_token,
                primitive_ids_table_size,
            )
            for tensor in tensors
        ]

    def _convert_token_tensor(
        self,
        tensor,
        new_token4old_primitive_token,
        new_token4old_symbol_token,
        primitive_ids_table_size,
    ):
        return np.array(
            [
                (
                    new_token4old_primitive_token(token_id)
                    if token_id < primitive_ids_table_size
                    else new_token4old_symbol_token(token_id)
                )
                for token_id in tensor.tolist()
            ],
            dtype=np.int64,
        )

    def _make_uid2new_symbol_token_id(
        self,
        inner,
        outer,
        inner_uid_prefix,
        outer_uid_prefix,
        outer_primitive_table_size,
    ):
        new_symbol_token_id = outer_primitive_table_size

        def get_new_symbol_token_id():
            nonlocal new_symbol_token_id
            ret = new_symbol_token_id
            new_symbol_token_id += 1
            return ret

        uid2new_symbol_token_id = {}
        for inner_symbol_token_id in inner.symbol_token_ids:
            uid = f"{inner_uid_prefix}{inner_symbol_token_id}"
            uid2new_symbol_token_id[uid] = get_new_symbol_token_id()
        for outer_symbol_token_id in outer.symbol_token_ids:
            uid = f"{outer_uid_prefix}{outer_symbol_token_id}"
            uid2new_symbol_token_id[uid] = get_new_symbol_token_id()
        return uid2new_symbol_token_id

    def _convert_symbol_token_ids(self, symbol_token_ids, new_token4old_token):
        return [
            new_token4old_token(symbol_token_id) for symbol_token_id in symbol_token_ids
        ]

    def _get_sub_window_sizes(self):
        min_window_size = max(1, self.min_window_size)
        window_size = self.max_window_size // 2
        while window_size > min_window_size:
            yield window_size
            window_size = window_size // 2
