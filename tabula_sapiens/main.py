import json
import os
import pathlib
import tempfile
import urllib
from itertools import product
from pathlib import Path
from typing import Tuple

import anndata as ad
import pooch
import requests
import scanpy as sc
import scvi
from scvi.hub import HubMetadata, HubModel, HubModelCardHelper


HF_API_TOKEN = os.environ["HF_API_TOKEN"]
scvi.settings.seed = 2023


def make_parents(*paths) -> None:
    """Make parent directories of a file path if they do not exist."""
    for p in paths:
        pathlib.Path(p).parent.mkdir(parents=True, exist_ok=True)


def load_config(config_path: str) -> dict:
    """Load a JSON configuration file as a Python dictionary."""
    with open(config_path) as f:
        config = json.load(f)
    return config


def get_urls(config: dict) -> Tuple[dict, dict]:
    req = requests.get(config["url"])
    adata_urls = {
        ind['key'][:-19]:ind['links']['self']
        for ind in req.json()['files'] if '.h5ad' in ind['key']
    }
    model_urls = {
        ind['key'][:-25]:ind['links']['self']
        for ind in req.json()['files'] if not '.h5ad' in ind['key']
    }

    return adata_urls, model_urls


def load_models(
    tissue: str, model_urls: dict, savedir: tempfile.TemporaryDirectory, config: dict
):
    unzipped = pooch.retrieve(
        url=model_urls[tissue],
        fname=f"{config['model_fname']}_{tissue}",
        known_hash=config["known_hash"],
        processor=pooch.Untar(),
        path=savedir.name,
    )
    return Path(unzipped[0]).parent.parent


def create_hub_model(
    tissue: str,
    model: str, 
    adata_urls: dict, 
    models_dir: str, 
    config: dict
) -> HubModel:
    """Create a HubModel object."""
    model_dir = os.path.join(models_dir, model)
    minified = model in ["scvi", "scanvi"]

    metadata = HubMetadata.from_dir(
        model_dir,
        anndata_version=ad.__version__,
        training_data_url=adata_urls[tissue],
    )

    card = HubModelCardHelper.from_dir(
        model_dir,
        license_info=config["license_info"],
        anndata_version=ad.__version__,
        data_is_minified=minified,
        data_is_annotated=True,
        tissues=config["tissues"],
        training_data_url=adata_urls[tissue],
        training_code_url=config["training_code_url"],
        description=config["description"],
        references=config["citation"],
        data_modalities=config["data_modalities"],
    )

    return HubModel(model_dir, metadata=metadata, model_card=card)


def upload_hub_model(
    hubmodel: HubModel, tissue: str, model: str, repo_token: str, config: dict
):
    """Upload the model to the HuggingFace Hub."""
    repo_name = f"{config['repo_name']}-{tissue.lower()}-{model}"
    try:
        hubmodel.push_to_huggingface_hub(
            repo_name=repo_name,
            repo_token=repo_token,
            repo_create=True,
        )
    except Exception:
        hubmodel.push_to_huggingface_hub(
            repo_name=repo_name,
            repo_token=repo_token,
            repo_create=False,
        )


def main():
    config = load_config("config.json")
    savedir = tempfile.TemporaryDirectory()

    adata_urls, model_urls = get_urls(config)

    for tissue in config["tissues"]:
        models_dir = load_models(tissue, model_urls, savedir, config)

        for model in config["models"]:
            hubmodel = create_hub_model(tissue, model, adata_urls, models_dir, config)
            upload_hub_model(hubmodel, tissue, model, HF_API_TOKEN, config)


if __name__ == "__main__":
    main()
