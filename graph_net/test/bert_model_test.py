import torch
from transformers import AutoModel, AutoTokenizer
import graph_net.torch 
import os


@graph_net.torch.extract(name="distilbert-function")
def create_model():
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    model.eval()
    return model.to(device)

if __name__ == '__main__':
    model_name = "distilbert-base-uncased"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    text = "Hello world"
    inputs = tokenizer(text, return_tensors="pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model = create_model()

    print("Running inference...")
    output = model(**inputs) 
    print("Inference finished. Output shape:", output.last_hidden_state.shape)