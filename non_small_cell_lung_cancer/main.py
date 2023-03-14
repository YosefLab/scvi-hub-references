import json
import os
import pathlib
from pathlib import Path
from typing import Tuple

import anndata as ad
import pooch
import scanpy as sc
import scvi
from scvi.hub import HubMetadata, HubModel, HubModelCardHelper

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


def load_model(config: dict) -> Tuple[scvi.model.SCANVI, ad.AnnData]:
    """Load the model and dataset."""
    model_url = config["model_url"]
    unzipped = pooch.retrieve(
        url=model_url,
        fname=config["model_fname"],
        known_hash=config["known_hash"],
        processor=pooch.Untar(),
    )[0]
    base_path = Path(unzipped).parent
    model_path = os.path.join(base_path, config["model_path"])
    adata_path = os.path.join(base_path, config["adata_path"])

    adata = sc.read(adata_path)
    model = scvi.model.SCANVI.load(model_path, adata=adata)
    
    return model


def minify_model_and_save(model: scvi.model.SCANVI, config: dict):
    """Minify the model and save it to disk."""
    latent_qzm_key = config["latent_qzm_key"]
    latent_qzv_key = config["latent_qzv_key"]

    qzm, qzv = model.get_latent_representation(return_dist=True)
    model.adata.obsm[latent_qzm_key] = qzm
    model.adata.obsm[latent_qzv_key] = qzv
    model.minify_adata(
        use_latent_qzm_key=latent_qzm_key, use_latent_qzv_key=latent_qzv_key
    )

    model_dir = os.path.join("models", config["model_dir"])
    make_parents(model_dir)
    model.save(model_dir, overwrite=True)


def create_hub_model(config: dict) -> HubModel:
    """Create a HubModel object."""
    model_dir = os.path.join("models", config["model_dir"])

    metadata = HubMetadata.from_dir(
        model_dir,
        anndata_version=ad.__version__,
        training_data_url=config["training_data_url"],
    )

    card = HubModelCardHelper.from_dir(
        model_dir,
        license_info=config["license_info"],
        anndata_version=ad.__version__,
        data_is_minified=True,
        data_is_annotated=True,
        tissues=config["tissues"],
        training_data_url=config["training_data_url"],
        training_code_url=config["training_code_url"],
        description=config["description"],
        references=config["citation"],
        data_modalities=["rna"],
    )

    return HubModel(model_dir, metadata=metadata, model_card=card)


def upload_hub_model(hubmodel: HubModel, repo_token: str, config: dict):
    """Upload the model to the HuggingFace Hub."""
    repo_name = config["repo_name"]
    try:
        hubmodel.push_to_huggingface_hub(
            repo_name=repo_name, repo_token=repo_token, repo_create=True,
        )
    except Exception as e:
        hubmodel.push_to_huggingface_hub(
            repo_name=repo_name, repo_token=repo_token, repo_create=False,
        )


def main():
    config = load_config("config.json")
    model = load_model(config)
    minify_model_and_save(model, config)
    hubmodel = create_hub_model(config)
    upload_hub_model(hubmodel, HF_API_TOKEN, config)


if __name__ == "__main__":
    main()
