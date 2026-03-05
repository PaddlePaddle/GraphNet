import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
import graph_net.torch


def get_model_name():
    return "uer/roberta-base-finetuned-cluener2020-chinese"


def get_graph_name():
    return "uer_roberta-base-finetuned-cluener2020-chinese"


def create_model(device: torch.device):
    model = AutoModelForTokenClassification.from_pretrained(get_model_name())
    model.eval()
    return model.to(device)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(get_model_name())
    text = "你好"
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=4,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model = create_model(device)
    model = graph_net.torch.extract(
        name=get_graph_name(),
        dynamic=True,
    )(model)

    print("Running inference...")
    with torch.no_grad():
        outputs = model(**inputs)
    if hasattr(outputs, "logits"):
        print("Inference finished. Logits shape:", outputs.logits.shape)
    else:
        print("Inference finished.")
