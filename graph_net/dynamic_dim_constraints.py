import sys
import sympy
import importlib.util as imp
from dataclasses import dataclass
import copy
from typing import Callable
from collections import namedtuple


@dataclass
class DynamicDimConstraints:
    kSymbolVarNamePrefix = "S"

    symbols: list[sympy.Symbol]
    kSymbols = "dynamic_dim_constraint_symbols"

    # len(symbol2example_value) == len(symbols)
    symbol2example_value: dict[sympy.Symbol, int]
    kSymbol2ExampleValue = "dynamic_dim_constraint_symbol2example_value"

    relations: list[sympy.Rel]
    kRelations = "dynamic_dim_constraint_relations"

    # len(input_shapes) equals number of Model.forward arguments
    input_shapes: list[(tuple[sympy.Expr | int], "var-name")]
    kInputShapes = "dynamic_dim_constraint_input_shapes"

    @classmethod
    def make_by_named_inputs(cls, named_shapes):
        return cls(
            symbols=[],
            symbol2example_value={},
            relations=[],
            input_shapes=named_shapes,
        )

    def symbolize(
        self,
        filter_fn: Callable[[str, int, int, int], bool],
    ) -> sympy.Symbol | None:
        """
        filter_fn: Callable[
            ["input_name:str", "input_idx:int", "axis:int", "dim:int"], bool
        ]
        Returns created symbol.
        """
        InputDim = namedtuple("InputDim", ["input_idx", "axis", "dim"])
        input_dims = [
            InputDim(input_idx, axis, dim)
            for input_idx, namedshape in enumerate(self.input_shapes)
            for shape, input_name in [namedshape]
            for axis, dim in enumerate(shape)
            if isinstance(dim, int)
            if filter_fn(input_name, input_idx, axis, dim)
        ]
        if len(input_dims) == 0:
            return None
        unique_dims = set(dim for input_dix, axis, dim in input_dims)
        assert len(unique_dims) == 1
        dim = list(unique_dims)[0]
        new_sym = self._new_symbol(example_value=dim)
        for input_dix, axis, _ in input_dims:
            self.input_shapes[input_dix][0][axis] = new_sym
        return new_sym

    def update_symbol2example_value(self, symbol2example_value: dict):
        self.symbol2example_value = self._merge_symbol2example_value(
            symbol2example_value
        )
        return self

    def get_reified_input_shapes(self):
        return [
            [self._try_reify(dim) for dim in shape] for shape, name in self.input_shapes
        ]

    def _try_reify(self, dim):
        if isinstance(dim, sympy.Expr):
            dim = int(dim.subs(self.symbol2example_value))
        assert isinstance(dim, (int, type(None))), f"{type(dim)=} {dim=}"
        return dim

    def check_delta_symbol2example_value(self, symbol2example_value: dict):
        if len(symbol2example_value) == 0:
            return True

        symbol2example_value = self._merge_symbol2example_value(symbol2example_value)

        sym_exprs = [
            *self._get_sym_exprs_from_input_shapes(),
        ]

        relations = [*self.relations, *(sym_expr > 0 for sym_expr in sym_exprs)]

        return all(
            relation.subs(symbol2example_value) == sympy.true for relation in relations
        )

    def serialize_to_py_str(self):
        symbols_definition = "\n".join(
            f"{name} = Symbol('{name}')"
            for symbol in self.symbols
            for name in [symbol.name]
        )
        return f"""
from sympy import Symbol, Expr, Rel, Eq

{symbols_definition}

{self.kSymbols} = {self.symbols}

{self.kSymbol2ExampleValue} = {self.symbol2example_value}

{self.kRelations} = {self.relations}

{self.kInputShapes} = {self.input_shapes}
"""

    @classmethod
    def unserialize_from_py_file(cls, filepath):
        module = cls.load_module(filepath)
        return cls(
            symbols=cls.module_symbols(module),
            symbol2example_value=cls.module_symbol2example_value(module),
            relations=cls.module_relations(module),
            input_shapes=cls.module_input_shapes(module),
        )

    @classmethod
    def module_symbols(cls, module):
        return cls.get_module_list_attr(module, cls.kSymbols)

    @classmethod
    def module_symbol2example_value(cls, module):
        return cls.get_module_dict_attr(module, cls.kSymbol2ExampleValue)

    @classmethod
    def module_relations(cls, module):
        return cls.get_module_list_attr(module, cls.kRelations)

    @classmethod
    def module_input_shapes(cls, module):
        return cls.get_module_list_attr(module, cls.kInputShapes)

    @classmethod
    def get_module_list_attr(cls, module, attr):
        return cls.get_module_attr(module, attr, default=[])

    @classmethod
    def get_module_dict_attr(cls, module, attr):
        return cls.get_module_attr(module, attr, default={})

    @classmethod
    def get_module_attr(cls, module, attr, default):
        return getattr(module, attr) if hasattr(module, attr) else default

    @classmethod
    def load_module(cls, path, name="unamed"):
        spec = imp.spec_from_file_location(name, path)
        module = imp.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _new_symbol(self, example_value):
        max_existed_seq_no = max(
            [
                -1,
                *(
                    seq_no
                    for symbol in self.symbols
                    for seq_no in [int(symbol.name[len(self.kSymbolVarNamePrefix) :])]
                ),
            ]
        )
        seq_no = max_existed_seq_no + 1
        symbol = sympy.Symbol(f"{self.kSymbolVarNamePrefix}{seq_no}")
        self.symbol2example_value[symbol] = example_value
        self.symbols.append(symbol)
        return symbol

    def _merge_symbol2example_value(self, symbol2example_value: dict):
        return {
            k: v
            for k, v in [
                *self.symbol2example_value.items(),
                *symbol2example_value.items(),
            ]
        }

    def _get_sym_exprs_from_input_shapes(self):
        yield from (
            sym_dim
            for shape, name in self.input_shapes
            for sym_dim in shape
            if isinstance(sym_dim, sympy.Expr)
        )


if __name__ == "__main__":
    cstr_code = """
import sympy

x = sympy.Symbol('x')
y = sympy.Symbol('y')

dynamic_dim_constraint_symbol2example_value = [(x, 2)]

dynamic_dim_constraint_symbols = [x, y]

dynamic_dim_constraint_relations = [x > 0]
    """

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w+", encoding="utf-8") as tmp:
        tmp.write(cstr_code)
        tmp.flush()
        cstr = DynamicDimConstraints.unserialize_from_py_file(tmp.name)
        print(cstr)
        print(cstr.serialize_to_py_str())

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w+", encoding="utf-8") as tmp:
        tmp.write(cstr.serialize_to_py_str())
        tmp.flush()
        cstr = DynamicDimConstraints.unserialize_from_py_file(tmp.name)
        print(cstr)
        print(cstr.serialize_to_py_str())
