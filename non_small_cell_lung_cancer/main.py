"""Non-small cell lung cancer (NSCLC)."""
import json
import os
import pathlib
import sys
import tempfile
from pathlib import Path

import anndata as ad
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


<<<<<<< HEAD
def load_model(save_dir: str, config: dict) -> scvi.model.SCANVI:
=======
def load_model(savedir: str, config: dict) -> scvi.model.SCANVI:
>>>>>>> main
    """Load the model and dataset."""
    unzipped = pooch.retrieve(
        url=config["model_url"],
        fname=config["model_save_dir"],
        known_hash=config["known_hash"],
        processor=pooch.Untar(),
<<<<<<< HEAD
        path=save_dir,
=======
        path=savedir,
>>>>>>> main
    )[0]
    base_path = Path(unzipped).parent
    model_path = os.path.join(base_path, config["model_dir"])
    adata_path = os.path.join(base_path, config["adata_fname"])

    adata = sc.read(adata_path)
    model = scvi.model.SCANVI.load(model_path, adata=adata)

    return model


def minify_model_and_save(
    model: scvi.model.SCANVI,
<<<<<<< HEAD
    save_dir: str,
=======
    savedir: str,
>>>>>>> main
    config: dict,
    latent_qzm_key: str = "latent_qzm",
    latent_qzv_key: str = "latent_qzv",
) -> str:
    """Minify the model and save it to disk."""
    (
        model.adata.obsm[latent_qzm_key],
        model.adata.obsm[latent_qzv_key],
    ) = model.get_latent_representation(give_mean=False, return_dist=True)
    model.minify_adata(
        use_latent_qzm_key=latent_qzm_key, use_latent_qzv_key=latent_qzv_key
    )

<<<<<<< HEAD
    model_dir = os.path.join(save_dir, config["minified_model_dir"])
=======
    model_dir = os.path.join(savedir, config["minified_model_dir"])
>>>>>>> main
    make_parents(model_dir)
    model.save(model_dir, overwrite=True)

    return model_dir


def create_hub_model(model_dir: str, config: dict) -> HubModel:
    """Create a HubModel object."""
    metadata = HubMetadata.from_dir(
        model_dir,
        anndata_version=ad.__version__,
        training_data_url=config["model_url"],
    )

    card = HubModelCardHelper.from_dir(
        model_dir,
        license_info=config["license_info"],
        anndata_version=ad.__version__,
        data_is_minified=True,
        data_is_annotated=True,
        tissues=config["tissues"],
        training_data_url=config["model_url"],
        training_code_url=config["training_code_url"],
        description=config["description"],
        references=config["citation"],
        data_modalities=config["data_modalities"],
    )

    return HubModel(model_dir, metadata=metadata, model_card=card)


def upload_hub_model(hub_model: HubModel, repo_token: str, config: dict):
    """Upload the model to the HuggingFace Hub."""
    repo_name = config["repo_name"]
    try:
        hub_model.push_to_huggingface_hub(
            repo_name=repo_name,
            repo_token=repo_token,
            repo_create=True,
        )
    except Exception:
        hub_model.push_to_huggingface_hub(
            repo_name=repo_name,
            repo_token=repo_token,
            repo_create=False,
        )


def main():
    """Run main."""
    config = load_config(snakemake.input[0])  # noqa: F821
<<<<<<< HEAD
    save_dir = tempfile.TemporaryDirectory().name

    model = load_model(save_dir, config)
    model_dir = minify_model_and_save(model, save_dir, config)
=======
    savedir = tempfile.TemporaryDirectory().name

    model = load_model(savedir, config)
    model_dir = minify_model_and_save(model, savedir, config)
>>>>>>> main
    hub_model = create_hub_model(model_dir, config)
    upload_hub_model(hub_model, HF_API_TOKEN, config)


if __name__ == "__main__":
    main()
