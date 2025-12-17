import os
import logging
import json

class BasicVerifier:
    def __init__(self):
        self.logger = logging.getLogger("BasicVerifier")

    def verify(self, sample_dir: str) -> bool:
        """
        Verify the integrity of the extracted sample.
        """
        # GraphNet might generate subdirectories like 'subgraph' or 'subgraph_1'
        # We need to find the actual model directory.
        target_dir = sample_dir
        if os.path.exists(os.path.join(sample_dir, "subgraph")):
            target_dir = os.path.join(sample_dir, "subgraph")
            
        required_files = ["graph_net.json", "model.py", "input_meta.py"]
        
        missing = []
        for f in required_files:
            if not os.path.exists(os.path.join(target_dir, f)):
                missing.append(f)
        
        if missing:
            self.logger.error(f"Verification Failed: Missing files {missing} in {target_dir}")
            return False
            
        # Optional: Check graph_net.json validity
        try:
            with open(os.path.join(target_dir, "graph_net.json"), "r") as f:
                config = json.load(f)
                if "hash" not in config and "model_hash" not in config:
                     # Just a warning or strict check
                     pass
        except Exception as e:
            self.logger.error(f"Verification Failed: Invalid json - {e}")
            return False
            
        self.logger.info("Verification Passed.")
        return True
