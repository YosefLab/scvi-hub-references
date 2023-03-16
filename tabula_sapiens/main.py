"""Tabula Sapiens."""
import json
import os
import pathlib
import sys
import tempfile
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


def get_urls(config: dict) -> Tuple[dict, dict]:
    """Get the URLs for the training data and models."""
    req = requests.get(config["url"])
    adata_urls = {
        ind["key"][:-19]: ind["links"]["self"]
        for ind in req.json()["files"]
        if ".h5ad" in ind["key"]
    }
    model_urls = {
        ind["key"][:-25]: ind["links"]["self"]
        for ind in req.json()["files"]
        if ".h5ad" not in ind["key"]
    }

    return adata_urls, model_urls


def load_models(tissue: str, model_urls: dict, savedir: str, config: dict):
    """Download the models and return the path to the models directory."""
    unzipped = pooch.retrieve(
        url=model_urls[tissue],
        fname=f"{config['model_fname']}_{tissue}",
        known_hash=config["known_hash"],
        processor=pooch.Untar(),
        path=savedir,
    )
    return Path(unzipped[0]).parent.parent


def create_hub_model(
    tissue: str, model: str, adata_urls: dict, models_dir: str, config: dict
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
        tissues=[tissue],
        training_data_url=adata_urls[tissue],
        training_code_url=config["training_code_url"],
        description=config["description"],
        references=config["citation"],
        data_modalities=config["data_modalities"],
    )

    return HubModel(model_dir, metadata=metadata, model_card=card)


def upload_hub_model(
    hub_model: HubModel, tissue: str, model: str, repo_token: str, config: dict
):
    """Upload the model to the HuggingFace Hub."""
    repo_name = f"{config['repo_name']}-{tissue.lower()}-{model}"
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


def get_retrain_urls(config: dict) -> dict:
    """Get the URLs for the retraining data."""
    req = requests.get(config["retrain_adata_urls"])
    adata_urls = {
        ind["key"][3:-14]: ind["links"]["self"] for ind in req.json()["files"]
    }
    return adata_urls


def load_adata(tissue: str, adata_urls: dict, savedir: str) -> ad.AnnData:
    """Load the training data."""
    adata_path = os.path.join(savedir, f"{tissue}.h5ad")
    return sc.read(adata_path, backup_url=adata_urls[tissue])


def preprocess_adata(adata: ad.AnnData, config: dict) -> ad.AnnData:
    """Preprocess the training data."""
    sc.experimental.pp.highly_variable_genes(
        adata,
        batch_key=config["batch_key"],
        **config["hvg_kwargs"],
    )
    return adata


def retrain_scvi_and_minify(
    adata: ad.AnnData,
    savedir: str,
    config: dict,
    latent_qzm_key: str = "X_scvi_qzm",
    latent_qzv_key: str = "X_scvi_qzv",
) -> scvi.model.SCVI:
    """Retrain scVI and minify the training data."""
    batch_key = config["batch_key"]
    labels_key = config["labels_key"]
    model_kwargs = config["model_kwargs"]
    train_kwargs = config["train_kwargs"]
    scvi_model_kwargs = model_kwargs["scvi"]
    model_kwargs["scanvi"]
    scvi_train_kwargs = train_kwargs["scvi"]

    scvi.model.SCVI.setup_anndata(
        adata,
        batch_key=batch_key,
        labels_key=labels_key,
    )
    model = scvi.model.SCVI(adata, **scvi_model_kwargs)
    model.train(**scvi_train_kwargs)

    (
        model.adata.obsm[latent_qzm_key],
        model.adata.obsm[latent_qzv_key],
    ) = model.get_latent_representation(give_mean=False, return_dist=True)
    model.minify_adata(
        use_latent_qzm_key=latent_qzm_key, use_latent_qzv_key=latent_qzv_key
    )

    model_dir = os.path.join(savedir, "scvi")
    make_parents(model_dir)
    model.save(model_dir, overwrite=True, save_anndata=True)

    return model


def retrain_scanvi_and_minify(
    model: scvi.model.SCVI,
    savedir: str,
    config: dict,
    latent_qzm_key: str = "X_scanvi_qzm",
    latent_qzv_key: str = "X_scanvi_qzv",
):
    """Retrain scANVI and minify the training data."""
    model_kwargs = config["model_kwargs"]["scanvi"]
    train_kwargs = config["train_kwargs"]["scanvi"]

    model = scvi.model.SCANVI.from_scvi_model(model, **model_kwargs)
    model.train(**train_kwargs)

    (
        model.adata.obsm[latent_qzm_key],
        model.adata.obsm[latent_qzv_key],
    ) = model.get_latent_representation(give_mean=False, return_dist=True)
    model.minify_adata(
        use_latent_qzm_key=latent_qzm_key, use_latent_qzv_key=latent_qzv_key
    )

    model_dir = os.path.join(savedir, "scanvi")
    make_parents(model_dir)
    model.save(model_dir, overwrite=True, save_anndata=True)


def retrain_condscvi(adata: ad.AnnData, savedir: str, config: dict):
    """Retrain CondSCVI."""
    labels_key = config["labels_key"]
    model_kwargs = config["model_kwargs"]["condscvi"]
    train_kwargs = config["train_kwargs"]["condscvi"]

    scvi.model.CondSCVI.setup_anndata(adata, labels_key=labels_key)
    model = scvi.model.CondSCVI(adata, **model_kwargs)
    model.train(**train_kwargs)

    model_dir = os.path.join(savedir, "condscvi")
    make_parents(model_dir)
    model.save(model_dir, overwrite=True, save_anndata=True)


def retrain_stereoscope(adata: ad.AnnData, savedir: str, config: dict):
    """Retrain Stereoscope."""
    labels_key = config["labels_key"]
    model_kwargs = config["model_kwargs"]["stereoscope"]
    train_kwargs = config["train_kwargs"]["stereoscope"]

    scvi.external.RNAStereoscope.setup_anndata(adata, labels_key=labels_key)
    model = scvi.external.RNAStereoscope(adata, **model_kwargs)
    model.train(**train_kwargs)

    model_dir = os.path.join(savedir, "stereoscope")
    make_parents(model_dir)
    model.save(model_dir, overwrite=True, save_anndata=True)


def retrain_models(
    adata: ad.AnnData, tissue: str, models: list, savedir: str, config: dict
) -> str:
    """Retrain the models and save them to disk."""
    models_dir = os.path.join(savedir, tissue)
    make_parents(models_dir)

    scvi_model = None
    if "scvi" in models:
        scvi_model = retrain_scvi_and_minify(adata, models_dir, config)

    if "scanvi" in models and scvi_model is not None:
        retrain_scanvi_and_minify(scvi_model, models_dir, config)

    if "condscvi" in models:
        retrain_condscvi(adata, models_dir, config)

    if "stereoscope" in models:
        retrain_stereoscope(adata, models_dir, config)

    return models_dir


def main():
    """Run main."""
    config = load_config(snakemake.input[0])  # noqa: F821
    retrain = config["retrain"]
    tissues = config["tissues"]
    models = config["models"]
    savedir = tempfile.TemporaryDirectory().name

    adata_urls, model_urls = get_urls(config)
    if retrain:
        retrain_adata_urls = get_retrain_urls(config)

    for tissue in tissues:
        if retrain:
            adata = load_adata(tissue, retrain_adata_urls, savedir)
            adata = preprocess_adata(adata, config)
            models_dir = retrain_models(adata, tissue, models, savedir, config)
        else:
            models_dir = load_models(tissue, model_urls, savedir, config)

        for model in models:
            try:
                hub_model = create_hub_model(
                    tissue, model, adata_urls, models_dir, config
                )
                upload_hub_model(hub_model, tissue, model, HF_API_TOKEN, config)
            except RuntimeError as e:
                print(f"Error uploading {tissue} {model}: {e}")


if __name__ == "__main__":
    main()
