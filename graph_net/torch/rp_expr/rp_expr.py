from dataclasses import dataclass
import typing as t
import numpy as np
import torch
from collections import defaultdict
import functools

PrimitiveId = t.TypeVar("PrimitiveId")

TokenId = int


# Repeat Pattern Expression
@dataclass
class RpExpr:
    pass


@dataclass
class ListRpExpr(RpExpr):
    pass


@dataclass
class NaiveTokenListRpExpr(ListRpExpr):
    tensors: t.List[np.ndarray["N", np.int64]]


@dataclass
class TokenizedRpExpr(RpExpr):
    token_id2primitive_id: t.List[PrimitiveId]
    token_tensors: ListRpExpr


@dataclass
class TokenRpExpr(RpExpr):
    pass


@dataclass
class FlattenedTokenListRpExpr(ListRpExpr):
    tensor_list_size: int
    flattened_tensor: TokenRpExpr


@dataclass
class NaiveTokenRpExpr(TokenRpExpr):
    tensor: np.ndarray["N", np.int64]


@dataclass
class LetsTokenRpExpr(TokenRpExpr):
    symbol_token_ids: t.List[TokenId]
    symbol_token_tensors: t.List[np.ndarray["N", np.int64]]
    body_rp_expr: NaiveTokenRpExpr


@dataclass
class LetsListTokenRpExpr(TokenRpExpr):
    symbol_token_ids: t.List[TokenId]
    symbol_token_tensors: t.List[np.ndarray["N", np.int64]]
    body_rp_expr: t.List[np.ndarray["N", np.int64]]

    def DebugStrings(
        self,
        token_id2primitive_id: t.List[PrimitiveId],
        prefix="sequence",
        end_of_line="",
    ):
        return self._DebugStrings(
            token_id2primitive_id,
            prefix=prefix,
            end_of_line=end_of_line,
        )

    def get_pure_primitive_binding_indexes(self, token_id2primitive_id):
        return self._get_pure_primitive_binding_indexes(token_id2primitive_id)

    def get_pure_primitive_binding_tensors(self, token_id2primitive_id):
        return [
            self.symbol_token_tensors[index]
            for index in self._get_pure_primitive_binding_indexes(token_id2primitive_id)
        ]

    def inplace_group_consecutive_primitives(self, token_id2primitive_id):
        return self._inplace_group_consecutive_primitives(token_id2primitive_id)

    def try_unwrap_body_of_sole_symbol_token(self):
        return self._try_unwrap_body_of_sole_symbol_token()

    def try_recursive_inline_symbol_sole_used(self, token_id2primitive_id):
        return self._try_recursive_inline_symbol_sole_used(token_id2primitive_id)

    def try_recursive_inline_symbol(self, token_id2primitive_id):
        return self._try_recursive_inline_symbol(token_id2primitive_id)

    def _try_recursive_inline_symbol(self, token_id2primitive_id):
        while self._try_inline_symbol(token_id2primitive_id):
            pass

    def _try_inline_symbol(self, token_id2primitive_id):
        pure_primitive_indexes = self.get_pure_primitive_binding_indexes(
            token_id2primitive_id
        )
        symbol_token2index = {
            symbol_token: index
            for index, symbol_token in enumerate(self.symbol_token_ids)
            if index not in pure_primitive_indexes
        }
        symbol_token2used_count = {
            symbol_token: 0 for symbol_token, _ in symbol_token2index.items()
        }
        for tensor in self.symbol_token_tensors:
            for token in tensor.tolist():
                if token not in symbol_token2used_count:
                    continue
                symbol_token2used_count[token] += 1
        found = False
        symbol_token = None
        symbol_index = None
        symbol_tensor = None
        for cur_symbol_token, used_count in symbol_token2used_count.items():
            if used_count >= 1:
                found = True
                symbol_token = cur_symbol_token
                symbol_index = symbol_token2index[symbol_token]
                symbol_tensor = self.symbol_token_tensors[symbol_index]
                break
        if not found:
            return False

        def get_self_or_inlined(x):
            if x == symbol_token:
                yield from symbol_tensor.tolist()
            else:
                yield x

        def inline_list(lst):
            return [x for token in lst for x in get_self_or_inlined(token)]

        def inline_tensor(tensor):
            return np.array(inline_list(tensor.tolist()), dtype=np.int64)

        def inline_tensor_list(tensor_list):
            return [inline_tensor(tensor) for tensor in tensor_list]

        self.symbol_token_ids.pop(symbol_index)
        self.symbol_token_tensors.pop(symbol_index)
        self.symbol_token_tensors = inline_tensor_list(self.symbol_token_tensors)
        self.body_rp_expr = inline_tensor_list(self.body_rp_expr)
        return True

    def _try_recursive_inline_symbol_sole_used(self, token_id2primitive_id):
        while self._try_inline_symbol_sole_used(token_id2primitive_id):
            pass

    def _try_inline_symbol_sole_used(self, token_id2primitive_id):
        pure_primitive_indexes = self.get_pure_primitive_binding_indexes(
            token_id2primitive_id
        )
        symbol_token2index = {
            symbol_token: index
            for index, symbol_token in enumerate(self.symbol_token_ids)
            if index not in pure_primitive_indexes
        }
        symbol_token2used_count = {
            symbol_token: 0 for symbol_token, _ in symbol_token2index.items()
        }
        for tensor in self.symbol_token_tensors:
            for token in tensor.tolist():
                if token not in symbol_token2used_count:
                    continue
                symbol_token2used_count[token] += 1
        found = False
        symbol_token = None
        symbol_index = None
        symbol_tensor = None
        for cur_symbol_token, used_count in symbol_token2used_count.items():
            if used_count == 1:
                found = True
                symbol_token = cur_symbol_token
                symbol_index = symbol_token2index[symbol_token]
                symbol_tensor = self.symbol_token_tensors[symbol_index]
                break
        if not found:
            return False

        def get_self_or_inlined(x):
            if x == symbol_token:
                yield from symbol_tensor.tolist()
            else:
                yield x

        def inline_list(lst):
            return [x for token in lst for x in get_self_or_inlined(token)]

        def inline_tensor(tensor):
            return np.array(inline_list(tensor.tolist()), dtype=np.int64)

        def inline_tensor_list(tensor_list):
            return [inline_tensor(tensor) for tensor in tensor_list]

        self.symbol_token_ids.pop(symbol_index)
        self.symbol_token_tensors.pop(symbol_index)
        self.symbol_token_tensors = inline_tensor_list(self.symbol_token_tensors)
        self.body_rp_expr = inline_tensor_list(self.body_rp_expr)
        return True

    def _try_unwrap_body_of_sole_symbol_token(self):
        symbol_token2symbol_tensor = {
            symbol_token: symbol_tensor
            for symbol_token, symbol_tensor in zip(
                self.symbol_token_ids, self.symbol_token_tensors
            )
        }
        token2used_count = {}
        for tensor in self.symbol_token_tensors + self.body_rp_expr:
            for token in tensor.tolist():
                if token not in token2used_count:
                    token2used_count[token] = 1
                else:
                    token2used_count[token] += 1
        sole_symbol_token_body_indexes = [
            i
            for i in range(len(self.body_rp_expr))
            for body_item in [self.body_rp_expr[i]]
            if body_item.size == 1
            if body_item[0] in symbol_token2symbol_tensor
            if token2used_count[body_item[0]] == 1
        ]
        symbol_tokens_in_sole_symbol_body = [
            self.body_rp_expr[i][0] for i in sole_symbol_token_body_indexes
        ]
        symbol_token_and_symbol_tensors = [
            (symbol_token, symbol_tensor)
            for symbol_token, symbol_tensor in zip(
                self.symbol_token_ids, self.symbol_token_tensors
            )
            if symbol_token not in symbol_tokens_in_sole_symbol_body
        ]
        self.symbol_token_ids = [x[0] for x in symbol_token_and_symbol_tensors]
        self.symbol_token_tensors = [x[1] for x in symbol_token_and_symbol_tensors]
        self.body_rp_expr = [
            (
                symbol_token2symbol_tensor[self.body_rp_expr[i][0]]
                if i in sole_symbol_token_body_indexes
                else self.body_rp_expr[i]
            )
            for i in range(len(self.body_rp_expr))
        ]

    def move_pure_primitive_bindings_front(self, token_id2primitive_id):
        return self._move_pure_primitive_bindings_front(token_id2primitive_id)

    def _move_pure_primitive_bindings_front(self, token_id2primitive_id):
        indexes = self.get_pure_primitive_binding_indexes(token_id2primitive_id)

        def reorder(lst):
            return [lst[i] for i in range(len(lst)) if i in indexes] + [
                lst[i] for i in range(len(lst)) if i not in indexes
            ]

        self.symbol_token_ids = reorder(self.symbol_token_ids)
        self.symbol_token_tensors = reorder(self.symbol_token_tensors)

    def _get_pure_primitive_binding_indexes(self, token_id2primitive_id):
        primitive_table_size = len(token_id2primitive_id)
        ret = []
        for i, tensor in enumerate(self.symbol_token_tensors):
            primitive_splited_tensors = self._split_consecutive_primitive(
                tensor, primitive_table_size
            )
            if (
                len(primitive_splited_tensors) == 1
                and primitive_splited_tensors[0][0] < primitive_table_size
            ):
                ret.append(i)
        return ret

    def _inplace_group_consecutive_primitives(self, token_id2primitive_id):
        get_auto_symbol_token_id = self._getter_auto_symbol_token_id(
            token_id2primitive_id
        )
        primitive_table_size = len(token_id2primitive_id)
        (
            primitives_new_token_ids_in_binding,
            primitives_new_token_tensors_in_binding,
            replaced_tensors_of_bindings,
        ) = self._group_token_tensors_consecutive_primitives(
            self.symbol_token_tensors, get_auto_symbol_token_id, primitive_table_size
        )
        (
            primitives_new_token_ids_in_body,
            primitives_new_token_tensors_in_body,
            replaced_tensors_of_body,
        ) = self._group_token_tensors_consecutive_primitives(
            self.body_rp_expr, get_auto_symbol_token_id, primitive_table_size
        )
        primitives_new_token_ids = (
            primitives_new_token_ids_in_binding + primitives_new_token_ids_in_body
        )
        primitives_new_token_tensors = (
            primitives_new_token_tensors_in_binding
            + primitives_new_token_tensors_in_body
        )
        self.symbol_token_ids = primitives_new_token_ids + self.symbol_token_ids
        self.symbol_token_tensors = (
            primitives_new_token_tensors + replaced_tensors_of_bindings
        )
        self.body_rp_expr = replaced_tensors_of_body

    def _group_token_tensors_consecutive_primitives(
        self, token_tensors, get_auto_symbol_token_id, primitive_table_size
    ):
        primitives_new_token_ids = []
        primitives_new_token_tensors = []
        ret_token_tensors = []
        for token_tensor in token_tensors:
            (
                cur_primitives_new_token_ids,
                cur_primitives_new_token_tensors,
                cur_ret_token_tensor,
            ) = self._group_consecutive_primitives(
                token_tensor, get_auto_symbol_token_id, primitive_table_size
            )
            primitives_new_token_ids += cur_primitives_new_token_ids
            primitives_new_token_tensors += cur_primitives_new_token_tensors
            ret_token_tensors.append(cur_ret_token_tensor)
        return primitives_new_token_ids, primitives_new_token_tensors, ret_token_tensors

    def _group_consecutive_primitives(
        self, token_tensor, get_auto_symbol_token_id, primitive_table_size
    ):
        primitive_splited_tensors = self._split_consecutive_primitive(
            token_tensor, primitive_table_size
        )
        if (
            len(primitive_splited_tensors) == 1
            and primitive_splited_tensors[0][0] < primitive_table_size
        ):
            return [], [], token_tensor
        primitives_new_token_ids = []
        primitives_new_token_tensors = []
        ret_token_tensors = []
        for tensor in primitive_splited_tensors:
            assert int(tensor.size(0)) > 0
            if tensor[0] < primitive_table_size:
                new_token_id = get_auto_symbol_token_id()
                primitives_new_token_ids.append(new_token_id)
                primitives_new_token_tensors.append(tensor)
                ret_token_tensors.append(np.array([new_token_id], dtype=np.int64))
            else:
                ret_token_tensors.append(tensor)
        ret_token_tensor = np.concatenate(ret_token_tensors)
        return primitives_new_token_ids, primitives_new_token_tensors, ret_token_tensor

    def _split_consecutive_primitive(self, token_tensor, primitive_table_size):
        is_primitive_tensor = token_tensor < primitive_table_size
        consecutive_tensors = consecutive(is_primitive_tensor, stepsize=0)
        global_start = 0

        def get_range(size):
            nonlocal global_start
            start = global_start
            end = start + size
            global_start = end
            return (start, end)

        segments = [
            token_tensor[start:end]
            for consecutive_tensor in consecutive_tensors
            for start, end in [get_range(int(consecutive_tensor.size(0)))]
        ]

        return segments

    def _getter_auto_symbol_token_id(self, token_id2primitive_id):
        start_token_id = len(token_id2primitive_id)
        for token_id in self.symbol_token_ids:
            start_token_id = max(token_id + 1, start_token_id)

        def get_new_symbol_token_id():
            nonlocal start_token_id
            ret = start_token_id
            start_token_id += 1
            return ret

        return get_new_symbol_token_id

    def _DebugStrings(
        self,
        token_id2primitive_id: t.List[PrimitiveId],
        prefix="sequence",
        end_of_line="",
    ):
        indexes = self.get_pure_primitive_binding_indexes(token_id2primitive_id)
        pure_primitive_symbol_token_set = set(
            symbol_token
            for index in indexes
            for symbol_token in [self.symbol_token_ids[index]]
        )

        def IsPrimitive(token_id):
            return token_id < len(token_id2primitive_id)

        def SymbolToString(symbol_id):
            return (
                f"{prefix}{symbol_id}"
                if symbol_id in pure_primitive_symbol_token_set
                else f"fold_{prefix}{symbol_id}"
            )

        def ValueToString(token_id):
            if IsPrimitive(token_id):
                return token_id2primitive_id[token_id]
            return f"{SymbolToString(token_id)}()"

        yield from (
            pycode
            for symbol_id, tensor in zip(
                self.symbol_token_ids, self.symbol_token_tensors
            )
            for token_ids in [tensor.tolist()]
            for pycode in [
                f"def {SymbolToString(symbol_id)}():",
                *[f"  {ValueToString(x)}{end_of_line}" for x in token_ids],
                "",
            ]
        )
        yield from [
            f"def main():",
            *[
                f"  {SymbolToString(int(t[0]))}(){end_of_line}"
                for t in self.body_rp_expr
            ],
        ]


def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


class TokenIdAllocator:
    def __init__(self, next_token_id: int = 0):
        self.next_token_id = next_token_id

    def NewTokenId(self):
        value = self.next_token_id
        self.next_token_id += 1
        return value

    def NextTokenId(self):
        return self.next_token_id

    def Skip(self, size):
        self.next_token_id += size


def Tokenize(
    primitive_id_lists: t.List[t.List[PrimitiveId]],
) -> t.Tuple[TokenizedRpExpr, TokenIdAllocator]:
    token_id_allocator = TokenIdAllocator()
    primitive_id2token_id = defaultdict(token_id_allocator.NewTokenId)
    token_tensors = [
        torch.tensor(
            [primitive_id2token_id[primitive_id] for primitive_id in primitive_id_list],
            dtype=torch.int64,
        )
        for primitive_id_list in primitive_id_lists
    ]
    token_id2primitive_id = [None] * len(primitive_id2token_id)
    for primitive_id, token_id in primitive_id2token_id.items():
        token_id2primitive_id[token_id] = primitive_id
    return (
        NaiveTokenListRpExpr(token_tensors),
        token_id_allocator,
        token_id2primitive_id,
    )
