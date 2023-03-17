"""Fetal immune."""
import json
import os
import pathlib
import sys
import tempfile
from pathlib import Path

import anndata as ad
import gdown
import pooch
import scanpy as sc
import scvi
from scvi.hub import HubMetadata, HubModel, HubModelCardHelper

HF_API_TOKEN = os.environ["HF_API_TOKEN"]
scvi.settings.seed = 2023
scvi.settings.reset_logging_handler()
sys.stderr = open(snakemake.log[0], "w")  # noqa: F821
sys.stdout = open(snakemake.log[1], "w")  # noqa: F821


def make_parents(*paths) -> None:
    """Make parent directories of a file path if they do not exist."""
    for p in paths:
        pathlib.Path(p).parent.mkdir(parents=True, exist_ok=True)


def load_config(config_path: str) -> dict:
    """Load a JSON configuration file as a Python dictionary."""
    with open(config_path) as f:
        config = json.load(f)
    return config


def load_adata(save_dir: str, config: dict) -> ad.AnnData:
    """Load the dataset."""
    adata_path = os.path.join(save_dir, config["adata_fname"])

    return sc.read(adata_path, backup_url=config["adata_url"])


def preprocess_adata(adata: ad.AnnData) -> ad.AnnData:
    """Preprocess the dataset."""
    pass


def convert_legacy_model(save_dir: str, config: dict) -> str:
    """Convert the legacy model to the new format."""
    legacy_model_dir = os.path.join(save_dir, config["legacy_model_dir"])
    gdown.download_folder(url=config["model_url"], output=legacy_model_dir)

    new_model_dir = os.path.join(save_dir, config["new_model_dir"])
    make_parents(new_model_dir)
    scvi.model.SCVI.convert_legacy_save(
        legacy_model_dir, new_model_dir, overwrite=True
    )

    return new_model_dir


def load_model(model_dir: str, adata: ad.AnnData) -> scvi.model.SCVI:
    """Load the model."""
    return scvi.model.SCVI.load(model_dir, adata=adata)


def minify_model_and_save(
    model: scvi.model.SCVI,
    save_dir: str,
    config: dict,
    latent_qzm_key: str = "X_scvi_qzm",
    latent_qzv_key: str = "X_scvi_qzv",
) -> str:
    """Minify the model and save it to disk."""
    (
        model.adata.obsm[latent_qzm_key],
        model.adata.obsm[latent_qzv_key],
    ) = model.get_latent_representation(give_mean=False, return_dist=True)
    model.minify_adata(
        use_latent_qzm_key=latent_qzm_key, use_latent_qzv_key=latent_qzv_key
    )

    model_dir = os.path.join(save_dir, config["minified_model_dir"])
    make_parents(model_dir)
    model.save(model_dir, overwrite=True)

    return model_dir


def create_hub_model(model_dir: str, config: str) -> HubModel:
    """Create a HubModel object."""
    metadata = HubMetadata.from_dir(
        model_dir,
        anndata_version=ad.__version__,
        training_data_url=config["adata_url"],
    )

    card = HubModelCardHelper.from_dir(
        model_dir,
        license_info=config["license_info"],
        anndata_version=ad.__version__,
        data_is_minified=True,
        data_is_annotated=True,
        tissues=config["tissues"],
        training_data_url=config["adata_url"],
        training_code_url=config["training_code_url"],
        description=config["description"],
        references=config["citation"],
        data_modalities=config["data_modalities"],
    )

    return HubModel(model_dir, metadata=metadata, model_card=card)


def upload_hub_model(hub_model: HubModel, repo_token: str, config: dict):
    """Upload the model to the HuggingFace Hub."""
    try:
        hub_model.push_to_huggingface_hub(
            repo_name=config["repo_name"],
            repo_token=repo_token,
            repo_create=True,
        )
    except Exception:
        hub_model.push_to_huggingface_hub(
            repo_name=config["repo_name"],
            repo_token=repo_token,
            repo_create=False,
        )


def main():
    """Run main."""
    config = load_config(snakemake.input[0])  # noqa: F821
    save_dir = tempfile.TemporaryDirectory().name

    adata = load_adata(save_dir, config)
    model_dir = convert_legacy_model(save_dir, config)
    model = load_model(model_dir, adata)
    model_dir = minify_model_and_save(model, save_dir, config)

    hub_model = create_hub_model(model_dir, config)
    upload_hub_model(hub_model, HF_API_TOKEN, config)


if __name__ == "__main__":
    main()
