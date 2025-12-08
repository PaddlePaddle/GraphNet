import graph_net.torch.sym_dim_reifiers.reifier_mgr as reifier_mgr


class ReifierFactory:
    def __init__(self, config, model_path):
        self.config = config
        self.model_path = model_path

    def get_matched_reifier_name(self):
        reifier_names = self.get_reifier_names()
        for reifier_name in reifier_names:
            reifier_class = reifier_mgr.get_reifier(reifier_name)
            reifier_instance = reifier_class(self.model_path)
            if reifier_instance.match():
                return reifier_name
        return None

    def get_reifier_names(self):
        return [
            "naive_cv_sym_dim_reifier",
            "naive_nlp_sym_dim_reifier",
        ]
