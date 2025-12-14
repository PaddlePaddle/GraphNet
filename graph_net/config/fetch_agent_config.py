import os
import yaml


def get_llm_config() -> dict:
    api_key = os.getenv("GRAPHNET_AGENT_API_KEY", None)
    base_url = os.getenv("GRAPHNET_AGENT_BASE_URL", None)
    config_dir = os.path.dirname(os.path.abspath(__file__))
    llm_config_path = os.path.join(config_dir, "config_agent_backend.yaml")
    with open(llm_config_path, "r") as file:
        llm_config = yaml.safe_load(file)
    llm_config["api_key"] = api_key
    llm_config["api_url"] = base_url
    return llm_config
