from dataclasses import dataclass
import typing as t
import numpy as np
import re
import itertools
import torch
import torch.nn.functional as F
import math
from graph_net.torch.rp_expr.rp_expr import (
    TokenIdAllocator,
    NaiveTokenListRpExpr,
    FlattenedTokenListRpExpr,
    NaiveTokenRpExpr,
    LetsTokenRpExpr,
    LetsListTokenRpExpr,
)
import itertools
import sys


class Pass:
    pass


class FlattenTokenListPass(Pass):
    def __init__(self, id_allocator: TokenIdAllocator):
        self.id_allocator = id_allocator

    def __call__(self, token_tensors_rp_expr: NaiveTokenListRpExpr):
        tensor_list_size = len(token_tensors_rp_expr.tensors)
        self.id_allocator.Skip(tensor_list_size)

        def GetSepTensor(i):
            if i == 0:
                return []
            return [torch.tensor([i], dtype=torch.int64)]

        token_tensors = [
            tensor
            for i, token_tensor in enumerate(token_tensors_rp_expr.tensors)
            for tensor in GetSepTensor(i) + [token_tensor + tensor_list_size]
        ]
        return True, FlattenedTokenListRpExpr(
            tensor_list_size=tensor_list_size,
            flattened_tensor=NaiveTokenRpExpr(
                tensor=torch.cat(token_tensors, dim=0),
            ),
        )


class FoldTokensPass(Pass):
    def __init__(self, id_allocator: TokenIdAllocator, window_size=8, policy="default"):
        self.window_size = window_size
        self.random_feature_size = 2
        self.id_allocator = id_allocator
        size = id_allocator.NextTokenId()
        torch.manual_seed(2024)
        self.embedding = torch.empty(
            size, self.random_feature_size, dtype=torch.float64
        ).uniform_(-1, 1)
        self.embedding.requires_grad_(True)
        self.policy = policy

    def __call__(self, token_tensor: NaiveTokenRpExpr):
        input_tensor = token_tensor.tensor
        raw_most_frequent_length, indexes = self.GetMostFrequentPatternLengthAndIndexes(
            input_tensor
        )
        most_frequent_length = self.GetAdaptivePatternLength(
            raw_most_frequent_length,
            indexes,
        )
        # print(f"{self.window_size=}, {raw_most_frequent_length=}, {most_frequent_length=}")
        new_token_id, replacement = self.Replace(
            pattern_length=most_frequent_length,
            indexes=indexes,
            input_tensor=input_tensor,
        )
        if new_token_id is None:
            return False, token_tensor
        start = indexes[0]
        return True, LetsTokenRpExpr(
            symbol_token_ids=[new_token_id],
            symbol_token_tensors=[input_tensor[start : (start + most_frequent_length)]],
            body_rp_expr=NaiveTokenRpExpr(tensor=replacement),
        )

    def GetAdaptivePatternLength(self, pattern_length, indexes):
        indexes = indexes.numpy().tolist()
        kLimit = self.window_size * 2
        while pattern_length > 1:
            disjoint_range_starts = [
                start for start in self.GetDisjoint(pattern_length, indexes)
            ]
            if len(disjoint_range_starts) > 1:
                break
            if pattern_length > kLimit:
                pattern_length = pattern_length // 2
            else:
                pattern_length -= 1
        return pattern_length

    def Replace(
        self,
        pattern_length,
        indexes,
        input_tensor: np.ndarray["N", np.int64],
    ) -> t.Tuple[bool, int, np.ndarray["N", np.int64]]:
        num_tokens = input_tensor.shape[0]
        if pattern_length == 1:
            return None, input_tensor
        assert indexes.shape[0] > 0
        disjoint_range_starts = [
            start
            for start in self.GetDisjoint(pattern_length, indexes.numpy().tolist())
        ]
        if len(disjoint_range_starts) <= 1:
            return None, input_tensor
        assert disjoint_range_starts[-1] + pattern_length <= num_tokens
        first_start = disjoint_range_starts[0]
        pattern_tensor = input_tensor[first_start : (first_start + pattern_length)]
        segment_starts = (
            [0]
            + [
                index
                for start in disjoint_range_starts
                for index in [start, start + pattern_length]
            ]
            + [num_tokens]
        )
        uniqued_segment_starts = torch.unique(torch.tensor(segment_starts))
        segment_lengths = torch.diff(uniqued_segment_starts).numpy().tolist()

        new_token_id = self.id_allocator.NewTokenId()
        new_token_tensor = torch.tensor([new_token_id], dtype=torch.int64)

        def ReplaceTensor(tensor):
            if tensor.shape != pattern_tensor.shape:
                return tensor
            if bool(torch.all(tensor == pattern_tensor)):
                return new_token_tensor
            return tensor

        replaced_segment_tensors = [
            ReplaceTensor(tensor)
            for tensor in torch.split(input_tensor, segment_lengths)
        ]
        output_tensor = torch.cat(replaced_segment_tensors)
        return new_token_id, output_tensor

    def GetConv(self, num_tokens):
        windows_size = min(num_tokens, self.window_size)

        def GetWeight():
            torch.manual_seed(2024)
            weight = torch.empty(
                windows_size, windows_size, dtype=torch.float64
            ).uniform_(-1, 1)
            weight.requires_grad_(True)
            weight = (
                torch.triu(weight)
                .transpose(0, 1)
                .reshape(windows_size, 1, windows_size)
            )
            return weight

        conv_weight = torch.cat(
            [GetWeight() for _ in range(self.random_feature_size)], dim=1
        )
        conv = lambda input: F.conv1d(input, conv_weight, padding=0)
        return conv, windows_size

    def GetDisjoint(self, gap, indexes):
        if len(indexes) == 0:
            return
        last = indexes[0]
        yield last
        for current in indexes:
            if current >= (last + gap):
                yield current
                last = current

    def GetMostFrequentPatternLengthAndIndexes(
        self,
        token_tensor: np.ndarray["N", np.int64],
    ):
        conv, windows_size = self.GetConv(num_tokens=token_tensor.shape[0])
        input = torch.index_select(self.embedding, 0, token_tensor)
        input.requires_grad_(True)
        zeros = torch.zeros(
            windows_size - 1, self.random_feature_size, dtype=torch.float64
        )
        input = torch.cat([input, zeros])
        input = input.reshape(1, -1, self.random_feature_size).transpose(1, 2)
        y = conv(input)
        y = y.reshape(windows_size, -1)
        y_hash = y.view(torch.int64)
        # `pattern_len_sub_1` means `pattern_length - 1`
        pattern_len_sub_1 = (
            torch.arange(windows_size).reshape(-1, 1).expand(y_hash.shape)
        )
        pattern_len_sub_1_and_hash = torch.cat(
            [pattern_len_sub_1.reshape(-1, 1), y_hash.reshape(-1, 1)], dim=1
        )
        unique_pattern_len_sub_1_and_hash, counts = torch.unique(
            pattern_len_sub_1_and_hash, dim=0, return_counts=True
        )
        if self.policy == "default":
            most_frequent_hash_idx = torch.argmax(
                unique_pattern_len_sub_1_and_hash[:, 0] * (counts - 1)
            )
        elif self.policy == "longest":
            most_frequent_hash_idx = torch.argmax(
                unique_pattern_len_sub_1_and_hash[:, 0] * (counts > 1).to(torch.int64)
            )
        else:
            assert False, f"policy {self.policy} not implemneted."

        most_frequent_hash = int(
            unique_pattern_len_sub_1_and_hash[most_frequent_hash_idx, 1]
        )
        most_frequent_pattern_len_sub_1 = int(
            unique_pattern_len_sub_1_and_hash[most_frequent_hash_idx, 0]
        )
        indexes = torch.nonzero(
            most_frequent_hash == y_hash[most_frequent_pattern_len_sub_1, :]
        ).flatten()
        return most_frequent_pattern_len_sub_1 + 1, indexes


class RecursiveFoldTokensPass(Pass):
    def __init__(
        self,
        id_allocator: TokenIdAllocator,
        window_size=8,
        fold_policy="default",
        fold_times=None,
    ):
        self.id_allocator = id_allocator
        self.window_size = window_size
        self.fold_policy = fold_policy
        self.fold_times = fold_times if fold_times is not None else sys.maxsize

    def __call__(self, token_tensor: NaiveTokenRpExpr):
        fold_pass = FoldTokensPass(
            self.id_allocator, self.window_size, policy=self.fold_policy
        )
        success, ret = fold_pass(token_tensor)
        if not success:
            return False, token_tensor
        symbol_token_ids = ret.symbol_token_ids
        symbol_token_tensors = ret.symbol_token_tensors
        token_tensor = ret.body_rp_expr
        counter = itertools.count()
        kLimit = 9999999
        for _ in range(self.fold_times):
            fold_pass = FoldTokensPass(
                self.id_allocator, self.window_size, policy=self.fold_policy
            )
            success, ret = fold_pass(token_tensor)
            if not success:
                token_tensor = ret
                break
            assert ret.body_rp_expr.tensor.shape[0] < token_tensor.tensor.shape[0]
            if next(counter) > kLimit:
                raise RuntimeError("dead loop detected.")
            symbol_token_ids += ret.symbol_token_ids
            symbol_token_tensors += ret.symbol_token_tensors
            token_tensor = ret.body_rp_expr
        return True, LetsTokenRpExpr(
            symbol_token_ids=symbol_token_ids,
            symbol_token_tensors=symbol_token_tensors,
            body_rp_expr=token_tensor,
        )


class UnflattenAndSubThresholdPass(Pass):
    def __init__(
        self,
        id_allocator: TokenIdAllocator,
        threshold_start_token_id: int,
    ):
        self.id_allocator = id_allocator
        self.threshold_start_token_id = threshold_start_token_id

    def __call__(self, lists_token_rp_expr: LetsTokenRpExpr):
        threshold_fold_pass = FoldIfTokenIdGreatEqualPass(
            id_allocator=self.id_allocator,
            threshold_start_token_id=self.threshold_start_token_id,
        )
        success, threshold_fold_rp_expr = threshold_fold_pass(
            lists_token_rp_expr.body_rp_expr
        )
        assert success
        return True, self.MergeAndUnflatten(
            lists_token_rp_expr, threshold_fold_rp_expr, self.threshold_start_token_id
        )

    def MergeAndUnflatten(self, fold_rp_expr, threshold_fold_rp_expr, threshold):
        assert len(threshold_fold_rp_expr.body_rp_expr) == threshold
        return LetsListTokenRpExpr(
            symbol_token_ids=[
                x - threshold
                for x in (
                    fold_rp_expr.symbol_token_ids
                    + threshold_fold_rp_expr.symbol_token_ids
                )
            ],
            symbol_token_tensors=[
                x - threshold
                for x in (
                    fold_rp_expr.symbol_token_tensors
                    + threshold_fold_rp_expr.symbol_token_tensors
                )
            ],
            body_rp_expr=[x - threshold for x in threshold_fold_rp_expr.body_rp_expr],
        )


class FoldIfTokenIdGreatEqualPass(Pass):
    def __init__(
        self,
        id_allocator: TokenIdAllocator,
        threshold_start_token_id: int,
    ):
        self.id_allocator = id_allocator
        self.threshold_start_token_id = threshold_start_token_id

    def __call__(self, token_rp_expr: NaiveTokenRpExpr):
        indexes_ge_threshold = self.GetIndexesGeThreshold(token_rp_expr.tensor)
        token_ids_ge_threshold = torch.index_select(
            token_rp_expr.tensor, 0, indexes_ge_threshold
        )
        consecutive_index_range_lengths = self.GetConsecutiveIndexRangeLengths(
            indexes_ge_threshold=indexes_ge_threshold,
        )
        tensors = torch.split(token_ids_ge_threshold, consecutive_index_range_lengths)

        def GetSymbolsValuesBodyTriple(tensor):
            if tensor.shape[0] == 1:
                return [], [], tensor
            new_token_id = self.id_allocator.NewTokenId()
            return (
                [new_token_id],
                [tensor],
                torch.tensor([new_token_id], dtype=torch.int64),
            )

        symbols_values_body_triples = [
            GetSymbolsValuesBodyTriple(tensor) for tensor in tensors
        ]
        return True, LetsListTokenRpExpr(
            symbol_token_ids=[
                token_id
                for new_token_ids, _, _ in symbols_values_body_triples
                for token_id in new_token_ids
            ],
            symbol_token_tensors=[
                token_tensor
                for _, token_tensors, _ in symbols_values_body_triples
                for token_tensor in token_tensors
            ],
            body_rp_expr=[
                body_tensor for _, _, body_tensor in symbols_values_body_triples
            ],
        )

    def GetIndexesGeThreshold(self, token_tensor: np.ndarray["N", np.int64]):
        indexes = torch.nonzero(token_tensor >= self.threshold_start_token_id).flatten()
        return indexes

    def GetConsecutiveIndexRangeLengths(self, indexes_ge_threshold):
        groups = self.GetNumpyConsecutiveGroups(indexes_ge_threshold.numpy())
        return [group.shape[0] for group in groups]

    # reference: https://stackoverflow.com/questions/7352684/how-to-find-the-groups-of-consecutive-elements-in-a-numpy-array
    def GetNumpyConsecutiveGroups(self, data, stepsize=1):
        return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)
