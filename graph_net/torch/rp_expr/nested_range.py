from dataclasses import dataclass
import typing as t


@dataclass
class NestedRange:
    pass


@dataclass
class Range(NestedRange):
    start: int
    end: int

    def FilterSubTreeRangeBySize(self, min_len: int, max_len: int):
        length = self.end - self.start
        if (length >= min_len) and (length < max_len):
            yield (self.start, self.end)


@dataclass
class Tree(NestedRange):
    uid: str
    node: Range
    children: t.List[NestedRange]

    def FilterSubTreeRangeBySize(self, min_len: int, max_len: int):
        length = self.node.end - self.node.start
        if length < min_len:
            yield from ()
        elif length < max_len:
            yield from self.node.FilterSubTreeRangeBySize(min_len, max_len)
        else:
            yield from (
                node_range
                for child in self.children
                for node_range in child.FilterSubTreeRangeBySize(min_len, max_len)
            )
