import yaml
from src.utils.paths import CONFIG_DIR


def load_config(filename: str = "config.yml") -> dict:
    config_path = CONFIG_DIR / filename
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

    