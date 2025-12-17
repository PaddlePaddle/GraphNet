import os
import logging
from typing import Optional

from .fetcher import HFFetcher
from .analyzer import ConfigAnalyzer
from .coder.template import TemplateCoder
from .extractor import SubprocessExtractor
from .verifier import BasicVerifier

class GraphNetAgent:
    def __init__(self, workspace: str, hf_token: Optional[str] = None):
        """
        Initialize the GraphNet Agent.
        
        Args:
            workspace (str): Directory where models and samples will be stored.
            hf_token (str, optional): Hugging Face API token.
        """
        self.workspace = os.path.abspath(workspace)
        os.makedirs(self.workspace, exist_ok=True)
        
        self.logger = logging.getLogger("GraphNetAgent")
        self.logger.setLevel(logging.INFO)
        
        # Initialize components
        self.fetcher = HFFetcher(self.workspace, token=hf_token)
        self.analyzer = ConfigAnalyzer()
        self.coder = TemplateCoder()
        self.extractor = SubprocessExtractor(self.workspace)
        self.verifier = BasicVerifier()

    def process_model(self, model_id: str) -> bool:
        """
        Process a single model: Download -> Analyze -> Generate Code -> Extract -> Verify.
        
        Args:
            model_id (str): Hugging Face model ID (e.g. 'bert-base-uncased')
            
        Returns:
            bool: True if successful, False otherwise.
        """
        self.logger.info(f"Starting process for model: {model_id}")
        
        try:
            # 1. Download Model
            self.logger.info("Step 1: Downloading model...")
            model_dir = self.fetcher.download(model_id)
            self.logger.info(f"Model downloaded to: {model_dir}")
            
            # 2. Analyze Model Config
            self.logger.info("Step 2: Analyzing model config...")
            meta_info = self.analyzer.analyze(model_dir)
            self.logger.info(f"Analysis result: {meta_info}")
            
            # 3. Generate Running Script
            self.logger.info("Step 3: Generating run_model.py...")
            script_path = self.coder.generate(model_dir, meta_info)
            self.logger.info(f"Script generated at: {script_path}")
            
            # 4. Extract Subgraph
            self.logger.info("Step 4: Extracting subgraph...")
            output_dir = self.extractor.extract(script_path, model_id)
            self.logger.info(f"Extraction output dir: {output_dir}")
            
            # 5. Verify Result
            self.logger.info("Step 5: Verifying result...")
            is_valid = self.verifier.verify(output_dir)
            
            if is_valid:
                self.logger.info(f"SUCCESS: Model {model_id} processed successfully.")
                return True
            else:
                self.logger.error(f"FAILURE: Verification failed for {model_id}.")
                return False
                
        except Exception as e:
            self.logger.error(f"Error processing {model_id}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
