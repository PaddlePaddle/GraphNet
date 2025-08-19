import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
import graph_net.torch
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.integrations.executorch import sdpa_mask_without_vmap


def get_model_name():
    return "Qwen/Qwen2.5-3B"


def create_model():
    config = AutoConfig.from_pretrained(get_model_name())
    model = AutoModel.from_config(config)
    # https://github.com/huggingface/transformers/blob/6b5bd117231f969713ed79fd4870903ab3c93edf/docs/source/en/attention_interface.md?plain=1#L194-L195
    ALL_MASK_ATTENTION_FUNCTIONS.register("sdpa_without_vmap", sdpa_mask_without_vmap)
    ALL_ATTENTION_FUNCTIONS.register(
        "sdpa_without_vmap", ALL_ATTENTION_FUNCTIONS["sdpa"]
    )
    model.config._attn_implementation = "sdpa_without_vmap"
    model.eval()
    return model.to(device)


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(get_model_name())

    text = "Hello world"
    inputs = tokenizer(text, return_tensors="pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    class DummyAutocast:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass

    class PatchedModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, *args, **kwargs):
            # TODO
            original_autocast1 = torch.cuda.amp.autocast
            original_autocast2 = torch.amp.autocast
            original_autocast3 = torch.autocast
            try:
                torch.cuda.amp.autocast = DummyAutocast
                torch.amp.autocast = DummyAutocast
                torch.autocast = DummyAutocast
                return self.model.forward(*args, **kwargs)
            finally:
                torch.cuda.amp.autocast = original_autocast1
                torch.amp.autocast = original_autocast2
                torch.autocast = original_autocast3

    model = create_model()
    model = PatchedModel(model)
    model = graph_net.torch.extract(name=get_model_name(), dynamic=False)(model)

    print("Running inference...")
    output = model(**inputs)
    print("Inference finished. Output shape:", output.last_hidden_state.shape)
