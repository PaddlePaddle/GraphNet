import os
import logging
from typing import Dict, Any

class TemplateCoder:
    def __init__(self):
        self.logger = logging.getLogger("TemplateCoder")

    def generate(self, model_dir: str, meta_info: Dict[str, Any]) -> str:
        """
        Generate a python script to load the model and run extraction.
        """
        script_content = self._create_script_content(model_dir, meta_info)
        
        output_path = os.path.join(model_dir, "run_extraction.py")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(script_content)
            
        return output_path

    def _create_script_content(self, model_dir: str, meta_info: Dict[str, Any]) -> str:
        # Basic template for HF models
        input_names = meta_info.get("input_names", ["input_ids"])
        input_shape = meta_info.get("input_shape", [1, 128])
        input_dtype = meta_info.get("input_dtype", "int64")
        
        # Construct input generation code
        input_gen_code = ""
        if meta_info["task_type"] == "nlp":
            input_gen_code += f"""
    # NLP Inputs
    input_ids = torch.randint(0, 100, {tuple(input_shape)}, dtype=torch.int64)
    attention_mask = torch.ones({tuple(input_shape)}, dtype=torch.int64)
    inputs = (input_ids, attention_mask)
            """
        elif meta_info["task_type"] == "cv":
            input_gen_code += f"""
    # CV Inputs
    inputs = (torch.randn({tuple(input_shape)}, dtype=torch.float32),)
            """
        
        template = f"""
import sys
import os
import torch
from transformers import AutoModel, AutoConfig

# Ensure graph_net is in path
sys.path.append(os.getcwd())

def main():
    model_path = r"{model_dir}"
    output_dir = r"{model_dir}/extracted_sample"
    
    print(f"Loading model from {{model_path}}...")
    try:
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {{e}}")
        sys.exit(1)

    print("Generating inputs...")
    {input_gen_code}
    
    # Move to CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    inputs = tuple(t.to(device) for t in inputs)

    print("Starting extraction...")
    # Setup environment variable for GraphNet workspace
    os.environ['GRAPH_NET_EXTRACT_WORKSPACE'] = output_dir
    
    # Use the extract API from graph_net
    # extract(name, dynamic=True)(model) returns a compiled model
    # We need to run it once to trigger compilation and extraction
    from graph_net.torch.extractor import extract
    
    compiled_model = extract(name="subgraph", dynamic=True)(model)
    
    print("Running forward pass to trigger extraction...")
    with torch.no_grad():
        if isinstance(inputs, tuple):
            compiled_model(*inputs)
        elif isinstance(inputs, dict):
            compiled_model(**inputs)
        else:
            compiled_model(inputs)
    
    print(f"Extraction complete. Results in {{output_dir}}")

if __name__ == "__main__":
    main()
"""
        return template
