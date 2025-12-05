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
        return sym_shapes_str in self._get_map_cv_sym_shapes_str2reifier()

    def reify(self):
        assert self.match()
        sym_shapes_str = self.dyn_dim_cstrs.serialize_symbolic_input_shapes_to_str()
        reifier = self._get_map_cv_sym_shapes_str2reifier()[sym_shapes_str]
        return reifier(self)

    @classmethod
    def _get_map_cv_sym_shapes_str2reifier(cls):
        if not hasattr(cls, "g_cv_sym_shapes_str2reifier"):
            cls.g_cv_sym_shapes_str2reifier = {
                "[(S0,3,S1,S1)]": cls.reify_s0_s1,
                "[(1,3,S0,S0)]": cls.reify_vit_related_hw_s0,
                "[(S0,3,512,512)]": cls.reify_mmseg_related_batch_s0,
                "[(S0,3,224,224)]": cls.reify_timm_related_big_batch_s0,
                "[(S0,3,256,192)]": cls.reify_mmpose_related_big_batch_s0,
                "[(S0,3,256,256)]": cls.reify_mmpose_related_big_batch_s0,
                "[(S0,3,S1,S2)]": cls.reify_mmpose_related_s0_s1_s2,
                "[(1,S0,3,S1,S1)]": cls.reify_vivit_related_s0_s1,
            }
        return cls.g_cv_sym_shapes_str2reifier

    def reify_s0_s1(self):
        S0S1 = (sympy.Symbol("S0"), sympy.Symbol("S1"))
        return {
            S0S1: [
                [1, 224],
                [1, 256],
                [1, 384],
                [32, 224],
                [32, 256],
                [32, 384],
                [128, 224],
                [128, 256],
                [128, 384],
            ],
        }

    def reify_vit_related_hw_s0(self):
        return {
            (sympy.Symbol("S0"),): [
                [128],
                [192],
                [224],
                [256],
                [336],
                [384],
                [448],
                [512],
                [640],
            ],
        }

    def reify_mmseg_related_batch_s0(self):
        return {
            (sympy.Symbol("S0"),): [[1], [2], [4], [8], [12], [16], [24], [32], [64]],
        }

    def reify_timm_related_big_batch_s0(self):
        return {
            (sympy.Symbol("S0"),): [
                [1],
                [4],
                [8],
                [16],
                [32],
                [64],
                [128],
                [256],
                [512],
            ],
        }

    def reify_mmpose_related_big_batch_s0(self):
        return {
            (sympy.Symbol("S0"),): [
                [1],
                [4],
                [8],
                [16],
                [32],
                [64],
                [128],
                [256],
                [512],
            ],
        }

    def reify_mmpose_related_s0_s1_s2(self):
        S0S1S2 = (sympy.Symbol("S0"), sympy.Symbol("S1"), sympy.Symbol("S2"))
        return {
            S0S1S2: [
                [1, 256, 192],
                [4, 128, 128],
                [1, 384, 288],
                [8, 256, 256],
                [2, 512, 512],
                [64, 96, 96],
                [16, 256, 192],
                [32, 192, 256],
                [8, 480, 640],
            ],
        }

    def reify_vivit_related_s0_s1(self):
        S0S1 = (sympy.Symbol("S0"), sympy.Symbol("S1"))
        return {
            S0S1: [
                [8, 112],
                [16, 112],
                [32, 112],
                [8, 224],
                [16, 224],
                [64, 112],
                [4, 448],
                [8, 384],
                [32, 224],
            ],
        }
