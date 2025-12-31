from graph_net.torch.sym_dim_reifiers.reify_util import get_dynamic_dim_constraints
from graph_net.torch.sym_dim_reifiers.reifier_base import ReifierBase
import os
import sympy


class ConcreteReifier(ReifierBase):
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path)
        self.dyn_dim_cstrs = get_dynamic_dim_constraints(model_path)

    def get_reifier_name(self) -> bool:
        return os.path.basename(__file__)[:-3]

    def match(self) -> bool:
        if self.dyn_dim_cstrs is None:
            return False
        sym_shapes_str = self.dyn_dim_cstrs.serialize_symbolic_input_shapes_to_str()
        if sym_shapes_str == "[]":
            return False
        elif sym_shapes_str not in self._get_map_nlp_sym_shapes_str2reifier():
            print(
                f"[NLP SymDim Reify] No reifier matched symbolic shapes:{sym_shapes_str} \nPlease add a reify rule to _get_map_nlp_sym_shapes_str2reifier()"
            )
            return False
        return True

    def reify(self):
        assert self.match()
        sym_shapes_str = self.dyn_dim_cstrs.serialize_symbolic_input_shapes_to_str()
        reifier = self._get_map_nlp_sym_shapes_str2reifier()[sym_shapes_str]
        return reifier(self)

    @classmethod
    def _get_map_nlp_sym_shapes_str2reifier(cls):
        if not hasattr(cls, "g_nlp_sym_shapes_str2reifier"):
            cls.g_nlp_sym_shapes_str2reifier = {
                "[(S0,1),(S0,S1),(S0,S1)]": cls.reify_batch_s0_seq_s1,
                "[(S0,S1),(S0,S1),(S0,S1)]": cls.reify_batch_s0_seq_s1,
                "[(S0,S1),(S0,S1)]": cls.reify_batch_s0_seq_s1,
                "[(S0,S1,768)]": cls.reify_batch_s0_seq_s1,
                "[(S0,S1)]": cls.reify_nlp_or_gnn_batch_s0_seq_s1,
            }
        return cls.g_nlp_sym_shapes_str2reifier

    def reify_batch_s0_seq_s1(self):
        S0S1 = (sympy.Symbol("S0"), sympy.Symbol("S1"))
        return {
            S0S1: [
                [1, 64],
                [1, 512],
                [16, 128],
                [32, 64],
                [8, 256],
                [4, 512],
                [2, 1024],
                [64, 128],
                [128, 64],
            ]
        }

    def reify_nlp_or_gnn_batch_s0_seq_s1(self):
        S0S1 = (sympy.Symbol("S0"), sympy.Symbol("S1"))
        return {
            S0S1: [
                [1, 128],
                [1, 1024],
                [32, 64],
                [16, 128],
                [8, 256],
                [4, 512],
                [2, 1024],
                [64, 128],
                [128, 64],
            ],
        }
