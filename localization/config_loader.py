import json
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Loads the JSON configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, 'r') as f:
        config = json.load(f)
    return config
