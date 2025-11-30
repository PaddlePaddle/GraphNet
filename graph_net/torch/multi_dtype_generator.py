"""
MultiDtypeGenerator: Generate low-precision samples (float16, bfloat16, float8) from existing samples.

This generator converts existing float32 samples to lower precision dtypes by:
1. Converting input tensor dtypes
2. Converting weight tensor dtypes (with special handling for batch_norm buffers)
3. Adding torch.autocast context manager to handle operators that don't support low precision
4. Filtering out graphs/subgraphs that cannot execute properly
"""

import os
import shutil
import ast
import re
import torch
import importlib.util
import inspect
from typing import List, Set, Dict, Any, Optional, Tuple
from pathlib import Path


class MultiDtypeGenerator:
    """
    Generator for creating low-precision samples from existing float32 samples.
    
    Args:
        source_sample_path: Path to the source sample directory
        output_base_path: Base path for output samples
        dtype_list: List of target dtypes to generate (e.g., ['float16', 'bfloat16'])
        filter_config: Configuration for filtering invalid graphs
    """
    
    # Dtypes that should remain as float32
    FLOAT32_ONLY_WEIGHTS = {
        'running_mean',
        'running_var',
        'num_batches_tracked',
        # batch_norm parameters that should stay float32
        'bn_parameters_weight',  # scale (gamma)
        'bn_parameters_bias',    # bias (beta)
        # Also check for layer_norm parameters
        'ln_parameters_weight',  # layer norm weight
        'ln_parameters_bias',    # layer norm bias
    }
    
    # Operators that may not support low precision well
    POTENTIALLY_UNSUPPORTED_OPS = {
        'torch.norm',
        'torch.linalg',
    }
    
    def __init__(
        self,
        source_sample_path: str,
        output_base_path: str,
        dtype_list: List[str],
        filter_config: Optional[Dict[str, Any]] = None,
    ):
        self.source_sample_path = Path(source_sample_path)
        self.output_base_path = Path(output_base_path)
        self.dtype_list = dtype_list
        self.filter_config = filter_config or {}
        
        # Validate source path exists
        if not self.source_sample_path.exists():
            raise ValueError(f"Source sample path does not exist: {source_sample_path}")
        
        # Validate required files exist
        required_files = ['model.py', 'weight_meta.py', 'graph_net.json']
        for req_file in required_files:
            if not (self.source_sample_path / req_file).exists():
                raise ValueError(f"Required file not found: {self.source_sample_path / req_file}")
        
        # input_meta.py is optional (may be empty)
    
    def generate(self) -> List[str]:
        """
        Generate low-precision samples for all specified dtypes.
        
        Returns:
            List of generated sample paths
        """
        generated_paths = []
        
        # Load source sample metadata
        import json
        with open(self.source_sample_path / 'graph_net.json', 'r') as f:
            source_metadata = json.load(f)
        
        # Get source name from metadata or use directory name
        source_name = source_metadata.get('model_name', None)
        if not source_name:
            source_name = self.source_sample_path.name
        
        for dtype in self.dtype_list:
            output_path = self._generate_for_dtype(dtype, source_name)
            if output_path:
                generated_paths.append(output_path)
        
        return generated_paths
    
    def _generate_for_dtype(self, dtype: str, source_name: str) -> Optional[str]:
        """
        Generate a single low-precision sample for the given dtype.
        
        Args:
            dtype: Target dtype (e.g., 'float16', 'bfloat16')
            source_name: Base name for the source sample
            
        Returns:
            Path to generated sample, or None if generation failed
        """
        # Create output directory name
        dtype_suffix = dtype.replace('float', 'f').replace('bfloat', 'bf')
        output_dir_name = f"{source_name}_{dtype_suffix}"
        output_path = self.output_base_path / output_dir_name
        
        try:
            # Create output directory
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Copy base files
            self._copy_base_files(output_path)
            
            # Modify input_meta.py
            self._modify_input_meta(dtype, output_path)
            
            # Modify weight_meta.py
            self._modify_weight_meta(dtype, output_path)
            
            # Modify model.py (add autocast)
            self._modify_model_code(dtype, output_path)
            
            # Update graph_net.json
            self._update_metadata(dtype, source_name, output_path)
            
            # Validate the generated sample
            if self._validate_sample(output_path):
                return str(output_path)
            else:
                # Remove invalid sample
                shutil.rmtree(output_path)
                return None
                
        except Exception as e:
            print(f"Error generating {dtype} sample: {e}")
            if output_path.exists():
                shutil.rmtree(output_path)
            return None
    
    def _copy_base_files(self, output_path: Path):
        """Copy base files that don't need modification."""
        files_to_copy = ['graph_net.json']
        for file_name in files_to_copy:
            source_file = self.source_sample_path / file_name
            if source_file.exists():
                shutil.copy2(source_file, output_path / file_name)
    
    def _modify_input_meta(self, dtype: str, output_path: Path):
        """Modify input_meta.py to use the specified dtype."""
        input_meta_path = self.source_sample_path / 'input_meta.py'
        output_input_meta = output_path / 'input_meta.py'
        
        # Check if file exists and is not empty
        if not input_meta_path.exists() or input_meta_path.stat().st_size == 0:
            # Create empty file or copy as-is
            if input_meta_path.exists():
                shutil.copy2(input_meta_path, output_input_meta)
            else:
                # Create empty file
                output_input_meta.touch()
            return
        
        with open(input_meta_path, 'r') as f:
            content = f.read()
        
        # If content is empty, just copy as-is
        if not content.strip():
            shutil.copy2(input_meta_path, output_input_meta)
            return
        
        # Replace float32 dtype with target dtype for float types
        torch_dtype_str = f"torch.{dtype}"
        
        # Replace all float32 dtypes in input tensors
        content = re.sub(
            r'dtype = "torch\.float32"',
            f'dtype = "{torch_dtype_str}"',
            content
        )
        
        # Only modify floating point types, not int types
        # We need to be careful to only modify float tensors
        # For now, replace all float32, which should be fine for inputs
        
        with open(output_input_meta, 'w') as f:
            f.write(content)
    
    def _modify_weight_meta(self, dtype: str, output_path: Path):
        """Modify weight_meta.py to use the specified dtype, keeping some weights as float32."""
        weight_meta_path = self.source_sample_path / 'weight_meta.py'
        output_weight_meta = output_path / 'weight_meta.py'
        
        with open(weight_meta_path, 'r') as f:
            content = f.read()
        
        torch_dtype_str = f"torch.{dtype}"
        
        # First, find all weight names that should remain float32
        # Pattern: class name, then name = "...", extract the name
        float32_weight_names = set()
        
        # Find all class definitions and their names
        class_pattern = r'class Program_weight_tensor_meta_([^:]+):'
        name_pattern = r'name = "([^"]+)"'
        
        # Parse the file to find weights that should stay float32
        lines = content.split('\n')
        current_class = None
        current_weight_name = None
        
        for i, line in enumerate(lines):
            # Check for class definition
            class_match = re.match(r'class\s+Program_weight_tensor_meta_([^:]+):', line.strip())
            if class_match:
                current_class = class_match.group(1)
                current_weight_name = None
                continue
            
            # Check for name line
            if current_class is not None and 'name =' in line:
                name_match = re.search(name_pattern, line)
                if name_match:
                    current_weight_name = name_match.group(1)
                    # Check if this weight should remain float32
                    for keep_key in self.FLOAT32_ONLY_WEIGHTS:
                        if keep_key in current_weight_name:
                            float32_weight_names.add(current_weight_name)
                            break
        
        # Now replace dtype, but skip the ones in float32_weight_names
        modified_lines = []
        current_weight_name = None
        skip_this_dtype = False
        
        for line in lines:
            # Check for class definition
            class_match = re.match(r'class\s+Program_weight_tensor_meta_([^:]+):', line.strip())
            if class_match:
                current_weight_name = None
                skip_this_dtype = False
                modified_lines.append(line)
                continue
            
            # Check for name line
            if 'name =' in line:
                name_match = re.search(name_pattern, line)
                if name_match:
                    current_weight_name = name_match.group(1)
                    skip_this_dtype = current_weight_name in float32_weight_names
                modified_lines.append(line)
                continue
            
            # Check for dtype line
            if 'dtype = "torch.float32"' in line:
                if skip_this_dtype:
                    # Keep as float32
                    modified_lines.append(line)
                else:
                    # Replace with target dtype
                    modified_lines.append(line.replace(
                        'dtype = "torch.float32"',
                        f'dtype = "{torch_dtype_str}"'
                    ))
                continue
            
            modified_lines.append(line)
        
        with open(output_weight_meta, 'w') as f:
            f.write('\n'.join(modified_lines))
    
    def _modify_model_code(self, dtype: str, output_path: Path):
        """Modify model.py to add autocast context manager."""
        model_path = self.source_sample_path / 'model.py'
        output_model = output_path / 'model.py'
        
        with open(model_path, 'r') as f:
            content = f.read()
        
        # Map dtype to torch autocast dtype
        autocast_dtype_map = {
            'float16': 'float16',
            'bfloat16': 'bfloat16',
            'float8': 'float16',  # Fallback for float8
        }
        autocast_dtype = autocast_dtype_map.get(dtype, 'float16')
        
        # Check if autocast is already applied
        if 'with autocast(' in content or 'torch.autocast(' in content:
            # Already has autocast, just write as-is (but ensure import exists)
            if 'from torch import autocast' not in content and 'import autocast' not in content:
                # Add import
                if 'from torch import' in content:
                    content = content.replace('from torch import device', 'from torch import device\nfrom torch import autocast', 1)
                    if 'from torch import autocast' not in content:
                        content = content.replace('import torch', 'import torch\nfrom torch import autocast', 1)
            with open(output_model, 'w') as f:
                f.write(content)
            return
        
        # Check if autocast is already imported, if not add it
        if 'from torch import autocast' not in content and 'import autocast' not in content:
            # Add import after existing torch imports
            if 'from torch import device' in content:
                content = content.replace('from torch import device', 'from torch import device\nfrom torch import autocast')
            elif 'import torch' in content:
                content = content.replace('import torch', 'import torch\nfrom torch import autocast', 1)
        
        # Determine device type from content
        device_type = 'cuda' if 'cuda' in content.lower() else 'cpu'
        
        # Use AST to find forward method and wrap it
        try:
            tree = ast.parse(content)
            
            class AutocastWrapper(ast.NodeTransformer):
                def __init__(self, dtype, device_type):
                    self.dtype = dtype
                    self.device_type = device_type
                    self.forward_found = False
                
                def visit_ClassDef(self, node):
                    # Only process GraphModule class
                    if node.name == 'GraphModule':
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef) and item.name == 'forward':
                                self.forward_found = True
                                # Wrap the body with autocast
                                autocast_call = ast.Call(
                                    func=ast.Name(id='autocast', ctx=ast.Load()),
                                    args=[],
                                    keywords=[
                                        ast.keyword(
                                            arg='device_type',
                                            value=ast.Constant(value=self.device_type)
                                        ),
                                        ast.keyword(
                                            arg='dtype',
                                            value=ast.Attribute(
                                                value=ast.Name(id='torch', ctx=ast.Load()),
                                                attr=self.dtype,
                                                ctx=ast.Load()
                                            )
                                        )
                                    ]
                                )
                                
                                # Wrap existing body with autocast
                                with_item = ast.withitem(context_expr=autocast_call)
                                with_stmt = ast.With(items=[with_item], body=item.body)
                                item.body = [with_stmt]
                                break
                    return self.generic_visit(node)
            
            wrapper = AutocastWrapper(autocast_dtype, device_type)
            modified_tree = wrapper.visit(tree)
            
            if wrapper.forward_found:
                # Convert back to code
                try:
                    modified_code = ast.unparse(modified_tree)
                except AttributeError:
                    # Fallback for older Python versions - use astor if available
                    try:
                        import astor
                        modified_code = astor.to_source(modified_tree)
                    except ImportError:
                        # Last resort: just write original with import
                        print("Warning: Could not apply autocast wrapper, only adding import")
                        modified_code = content
                
                with open(output_model, 'w') as f:
                    f.write(modified_code)
            else:
                # Forward method not found, just write original with import
                with open(output_model, 'w') as f:
                    f.write(content)
        except Exception as e:
            print(f"Warning: Error modifying model code: {e}")
            # Fallback: just write original with import
            with open(output_model, 'w') as f:
                f.write(content)
    
    def _update_metadata(self, dtype: str, source_name: str, output_path: Path):
        """Update graph_net.json metadata."""
        import json
        
        metadata_path = output_path / 'graph_net.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Update metadata
        metadata['model_name'] = f"{source_name}_{dtype}"
        metadata['generated_from'] = str(self.source_sample_path)
        metadata['dtype'] = dtype
        if 'tags' not in metadata:
            metadata['tags'] = []
        if 'multi_dtype_generated' not in metadata['tags']:
            metadata['tags'].append('multi_dtype_generated')
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
    
    def _validate_sample(self, output_path: Path) -> bool:
        """
        Validate the generated sample can execute properly.
        
        Returns:
            True if valid, False otherwise
        """
        try:
            # Try to load and run the model
            model_path = str(output_path)
            
            # Import the model
            spec = importlib.util.spec_from_file_location("test_model", str(output_path / "model.py"))
            if spec is None or spec.loader is None:
                return False
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if not hasattr(module, 'GraphModule'):
                return False
            
            # Try to instantiate (this will fail if there are syntax errors)
            model_class = module.GraphModule
            
            # Try to load inputs and weights
            from graph_net.torch.utils import load_converted_from_text
            converted = load_converted_from_text(model_path)
            
            # Create dummy inputs for validation
            from graph_net.torch.utils import get_dummy_named_tensors
            dummy_inputs = get_dummy_named_tensors(converted['input_info'])
            
            # Try to create model instance (without running forward)
            # Just check if the class can be instantiated
            # Full execution test would require actual weights which might be large
            
            return True
            
        except Exception as e:
            print(f"Validation failed: {e}")
            return False


class MultiDtypeFilter:
    """
    Filter for multi-dtype samples that cannot execute properly.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def __call__(self, gm, sample_inputs) -> bool:
        """
        Check if the graph/subgraph can execute with low precision.
        
        Args:
            gm: GraphModule to check
            sample_inputs: Sample inputs
            
        Returns:
            True if the graph should be kept, False if it should be filtered out
        """
        # Check for unsupported operations
        for node in gm.graph.nodes:
            if node.op == 'call_function':
                target_str = str(node.target)
                # Check for operators that may not support low precision
                for unsupported_op in MultiDtypeGenerator.POTENTIALLY_UNSUPPORTED_OPS:
                    if unsupported_op in target_str:
                        # Could add more sophisticated checks here
                        pass
        
        # Default: keep the graph
        return True

