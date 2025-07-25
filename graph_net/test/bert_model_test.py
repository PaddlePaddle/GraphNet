import torch
from transformers import AutoModel, AutoTokenizer
import graph_net.torch 
import os

def get_model_name():
    return "distilbert-base-uncased"

def create_model():
    model = AutoModel.from_pretrained(get_model_name())
    model.eval()
    return model.to(device)

if __name__ == '__main__':
    
    tokenizer = AutoTokenizer.from_pretrained(get_model_name())

    text = "Hello world"
    inputs = tokenizer(text, return_tensors="pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model = create_model()
    model = graph_net.torch.extract(name=get_model_name())(model)

    print("Running inference...")
    output = model(**inputs) 
    print("Inference finished. Output shape:", output.last_hidden_state.shape)
