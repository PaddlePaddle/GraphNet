import torch
from transformers import AutoModel, AutoTokenizer
from graph_net.torch._extract.extractor import extract
import os

os.environ["GRAPH_NET_EXTRACT_WORKSPACE"] = "../../samples/torch/extracted_models"

if __name__ == '__main__':
    model_name = "distilbert-base-uncased"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    text = "Hello world"
    inputs = tokenizer(text, return_tensors="pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model = extract(name="distilbert-base-uncased")(model)

    print("Running inference...")
    output = model(**inputs) 
    print("Inference finished. Output shape:", output.last_hidden_state.shape)