import json
import os
import pathlib


HF_API_TOKEN = os.environ["HF_API_TOKEN"]


def make_parents(*paths) -> None:
    """Make parent directories of a file path if they do not exist."""
    for p in paths:
        pathlib.Path(p).parent.mkdir(parents=True, exist_ok=True)


def load_config(config_path: str) -> dict:
    """Load a JSON configuration file as a Python dictionary."""
    with open(config_path) as f:
        config = json.load(f)
    return config


def main():
    pass


if __name__ == "__main__":
    main()
