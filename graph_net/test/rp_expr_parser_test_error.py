import unittest
from graph_net.torch.rp_expr import Tokenize
from graph_net.torch.rp_expr.rp_expr_passes import (
    FlattenTokenListPass,
    FoldTokensPass,
    RecursiveFoldTokensPass,
    FoldIfTokenIdGreatEqualPass,
)
from graph_net.torch.rp_expr.nested_range import Range, Tree
from graph_net.torch.rp_expr.rp_expr_parser import RpExprParser
from graph_net.torch.rp_expr.rp_expr_util import (
    MakeNestedIndexRangeFromLetsListTokenRpExpr,
)


class TestRpExprParser(unittest.TestCase):
    def test_simple(self):
        base = 3
        size = 3
        print("\n", "-" * 50, "\nTestRpExprParser.test_simple info:")
        primitive_id_lists = [list(range(base + i)) for i in range(size)]
        print(">>> primitive_id_lists: ", primitive_id_lists)

        parser = RpExprParser()
        lets_list_rp_expr, token_id2primitive_id = parser(primitive_id_lists)
        print(">>> lets_list_rp_expr: ", lets_list_rp_expr)
        print(">>> token_id2primitive_id: ", token_id2primitive_id)

        pattern = [x.tolist() for x in lets_list_rp_expr.symbol_token_tensors]
        print(">>> pattern: ", pattern)

        replacement = lets_list_rp_expr.symbol_token_ids
        print(">>> replacement: ", replacement)

        output = [x.tolist() for x in lets_list_rp_expr.body_rp_expr]
        print(">>> output: ", output)

        self.assertEqual(pattern, [[0, 1, 2], [5, 3], [6, 4]])
        self.assertEqual(replacement, [5, 6, 7])
        self.assertEqual(output, [[5], [6], [7]])
        self.assertEqual(token_id2primitive_id, [0, 1, 2, 3, 4])

    def test_two_elem_pattern(self):
        primitive_id_lists = [[1600, 411, 441, 411, 1600, 411, 32]]
        # embeeding = [[0, 1, 2, 1, 0, 1, 3]]
        parser = RpExprParser()
        print("\n", "-" * 50, "\nTestRpExprParser.test_two_elem_pattern info:")
        lets_list_rp_expr, token_id2primitive_id = parser(primitive_id_lists)
        print(">>> lets_list_rp_expr: ", lets_list_rp_expr)
        print(">>> token_id2primitive_id: ", token_id2primitive_id)

        pattern = [x.tolist() for x in lets_list_rp_expr.symbol_token_tensors]
        print(">>> pattern: ", pattern)

        replacement = lets_list_rp_expr.symbol_token_ids
        print(">>> replacement: ", replacement)

        output = [x.tolist() for x in lets_list_rp_expr.body_rp_expr]
        print(">>> output: ", output)

        self.assertEqual(pattern, [[0, 1], [4, 2, 1, 4, 3]])
        self.assertEqual(replacement, [4, 5])
        self.assertEqual(output, [[5]])
        self.assertEqual(token_id2primitive_id, [1600, 411, 441, 32])


class TestMakeNestedIndexRangeFromLetsListTokenRpExpr(unittest.TestCase):
    def test_simple(self):
        base = 3
        size = 3
        print("\n", "-" * 50, "\nTestMakeNestedIndexRangeFromLetsListTokenRpExpr info:")
        primitive_id_lists = [list(range(base + i)) for i in range(size)]
        parser = RpExprParser()
        lets_list_rp_expr, token_id2primitive_id = parser(primitive_id_lists)
        print(">>> lets_list_rp_expr: ", lets_list_rp_expr)
        print(">>> token_id2primitive_id: ", token_id2primitive_id)

        trees = MakeNestedIndexRangeFromLetsListTokenRpExpr(lets_list_rp_expr)
        tree0 = Tree(
            uid="0",
            node=Range(0, 3),
            children=[
                Range(0, 1),
                Range(1, 2),
                Range(2, 3),
            ],
        )
        tree1 = Tree(
            uid="1",
            node=Range(0, 4),
            children=[
                tree0,
                Range(3, 4),
            ],
        )
        tree2 = Tree(
            uid="2",
            node=Range(0, 5),
            children=[
                tree1,
                Range(4, 5),
            ],
        )
        self.assertEqual(trees, [tree0, tree1, tree2])


if __name__ == "__main__":
    unittest.main()
