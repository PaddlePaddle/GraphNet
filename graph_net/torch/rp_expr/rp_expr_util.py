import typing as t
from graph_net.torch.rp_expr.rp_expr import LetsListTokenRpExpr
from graph_net.torch.rp_expr.nested_range import Range, Tree
from collections import defaultdict
from dataclasses import dataclass


def MakeNestedIndexRangeFromLetsListTokenRpExpr(
    rp_expr: LetsListTokenRpExpr, uid_prefix: str = ""
) -> t.List[Tree]:
    symbol_token_tensors = [
        tuple(token_ids.tolist()) for token_ids in rp_expr.symbol_token_tensors
    ]
    token2len = defaultdict(lambda: 1)
    token2children = defaultdict(lambda: ())
    for token, token_ids in zip(rp_expr.symbol_token_ids, symbol_token_tensors):
        token2len[token] = sum(map(lambda x: token2len[x], token_ids))
        token2children[token] = token_ids

    for tensor in rp_expr.body_rp_expr:
        assert len(tensor.shape) == 1
        assert tensor.shape[0] == 1

    return [
        MakeNestedIndexRangeFromRangeTokenCtx(
            offset=0,
            root_token_id=root_token_id,
            token2len=token2len,
            token2children=token2children,
            uid_prefix=uid_prefix,
        )
        for tensor in rp_expr.body_rp_expr
        for root_token_id in tensor.tolist()
    ]


def MakeNestedIndexRangeFromRangeTokenCtx(
    offset: int,
    root_token_id: int,
    token2len: t.Dict[int, int],
    token2children: t.Dict[int, t.List[int]],
    uid_prefix: str,
):
    if len(token2children[root_token_id]) == 0:
        assert token2len[root_token_id] == 1
        return Range(offset, offset + token2len[root_token_id])

    def GetThenIncreaseChildOffset(length):
        nonlocal offset
        child_offset = offset
        offset += length
        return child_offset

    return Tree(
        uid=f"{uid_prefix}{root_token_id}",
        node=Range(offset, offset + token2len[root_token_id]),
        children=[
            MakeNestedIndexRangeFromRangeTokenCtx(
                offset=child_offfset,
                root_token_id=child_token_id,
                token2len=token2len,
                token2children=token2children,
                uid_prefix=uid_prefix,
            )
            for child_token_id in token2children[root_token_id]
            for child_offfset in [GetThenIncreaseChildOffset(token2len[child_token_id])]
        ],
    )
