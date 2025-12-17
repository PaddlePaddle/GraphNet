import os
import sys

# Ensure we can import graph_net
sys.path.append(os.getcwd())

from graph_net.agent import GraphNetAgent

def main():
    # Setup a local workspace
    # Use a writable directory instead of System32
    workspace = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_workspace")
    print(f"Using workspace: {workspace}")
    
    agent = GraphNetAgent(workspace=workspace)
    
    # Use a small model for testing
    test_model = "prajjwal1/bert-tiny"
    
    print(f"Processing model: {test_model}")
    success = agent.process_model(test_model)
    
    if success:
        print("\n[SUCCESS] Agent successfully processed the model!")
        print(f"Check results in: {workspace}/downloads/{test_model.replace('/', '_')}/extracted_sample")
    else:
        print("\n[FAILURE] Agent failed to process the model.")

if __name__ == "__main__":
    main()
