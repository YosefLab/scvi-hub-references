"""Tabula Sapiens."""
import os
import sys
import tempfile
from typing import Tuple

import anndata as ad
import requests
import scanpy as sc
import scvi

from utils import (
    HF_API_TOKEN,
    create_hub_model,
    get_temp_dir,
    load_adata,
    load_config, 
    make_parents,
    minify_model_and_save,
    retrieve_model,
    save_model,
    upload_hub_model,
)

sys.stderr = open(snakemake.log[0], "w")  # noqa: F821
sys.stdout = open(snakemake.log[1], "w")  # noqa: F821


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


def get_retrain_urls(config: dict) -> dict:
    """Get the URLs for the retraining data."""
    req = requests.get(config["retrain_adata_urls"])
    adata_urls = {
        ind["key"][3:-14]: ind["links"]["self"] for ind in req.json()["files"]
    }
    return adata_urls


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
    save_dir: str,
    config: dict,
) -> scvi.model.SCVI:
    """Retrain scVI and minify the training data."""
    batch_key = config["batch_key"]
    labels_key = config["labels_key"]
    model_kwargs = config["model_kwargs"]["scvi"]
    train_kwargs = config["train_kwargs"]["scvi"]

    scvi.model.SCVI.setup_anndata(
        adata,
        batch_key=batch_key,
        labels_key=labels_key,
    )
    model = scvi.model.SCVI(adata, **model_kwargs)
    model.train(**train_kwargs)

    minify_model_and_save(
        model=model, model_dir="scvi", save_dir=save_dir, 
    )

    return model


def retrain_scanvi_and_minify(
    model: scvi.model.SCVI,
    save_dir: str,
    config: dict,
):
    """Retrain scANVI and minify the training data."""
    model_kwargs = config["model_kwargs"]["scanvi"]
    train_kwargs = config["train_kwargs"]["scanvi"]

    model = scvi.model.SCANVI.from_scvi_model(model, **model_kwargs)
    model.train(**train_kwargs)

    minify_model_and_save(
        model=model, model_dir="scanvi", save_dir=save_dir, 
    )


def retrain_condscvi(adata: ad.AnnData, save_dir: str, config: dict):
    """Retrain CondSCVI."""
    labels_key = config["labels_key"]
    model_kwargs = config["model_kwargs"]["condscvi"]
    train_kwargs = config["train_kwargs"]["condscvi"]

    scvi.model.CondSCVI.setup_anndata(adata, labels_key=labels_key)
    model = scvi.model.CondSCVI(adata, **model_kwargs)
    model.train(**train_kwargs)

    save_model(model=model, model_dir="condscvi", save_dir=save_dir)


def retrain_stereoscope(adata: ad.AnnData, savedir: str, config: dict):
    """Retrain Stereoscope."""
    labels_key = config["labels_key"]
    model_kwargs = config["model_kwargs"]["stereoscope"]
    train_kwargs = config["train_kwargs"]["stereoscope"]

    scvi.external.RNAStereoscope.setup_anndata(adata, labels_key=labels_key)
    model = scvi.external.RNAStereoscope(adata, **model_kwargs)
    model.train(**train_kwargs)

    save_model(model=model, model_dir="stereoscope", save_dir=savedir)


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
    save_dir = get_temp_dir()
    retrain = config["retrain"]
    tissues = config["tissues"]
    models = config["models"]
    savedir = tempfile.TemporaryDirectory().name

    adata_urls, model_urls = get_urls(config)
    retrain_adata_urls = get_retrain_urls(config)

    for tissue in tissues:
        if retrain:
            adata = load_adata(
                url=retrain_adata_urls[tissue],
                fname=f"{tissue}.h5ad",
                save_dir=save_dir
            )
            adata = preprocess_adata(adata, config)
            models_dir = retrain_models(adata, tissue, models, savedir, config)
        else:
            models_dir = retrieve_model(
                model_url=model_urls[tissue],
                model_dir=f"{config['model_fname']}_{tissue}",
                save_dir=save_dir,
            ).parent

        for model in models:
            hub_model = create_hub_model(
                model_dir=os.path.join(models_dir, model),
                training_data_url=adata_urls[tissue],
                training_code_url=config["training_code_url"],
                data_is_minified=model in ["scvi", "scanvi"],
                data_is_annotated=True,
                tissues=[tissue],
                data_modalities=config["data_modalities"],
                description=config["description"],
                references=config["citation"],
                license_info=config["license_info"],
            )
            upload_hub_model(
                hub_model, 
                f"{config['repo_name']}-{tissue.lower()}-{model}",
                HF_API_TOKEN
            )


if __name__ == "__main__":
    main()
