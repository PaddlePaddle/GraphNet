from enum import Enum


class GraphFusibility(Enum):
    kFullyFusible = "fully_fusible"
    kNotFullyFusible = "not_fully_fusible"


class GraphFusibilityStatus(Exception):
    def __init__(self, graph_fusibility: GraphFusibility):
        message = f"{graph_fusibility=}"
        super().__init__(message)
        self.graph_fusibility = graph_fusibility
