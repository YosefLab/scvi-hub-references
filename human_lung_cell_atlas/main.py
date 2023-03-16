"""Human Lung Cell Atlas."""
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


def load_adata(savedir: str, config: dict) -> ad.AnnData:
    """Load the dataset."""
    adata = scvi.data.cellxgene(
        config["adata_url"], filename=config["adata_fname"], save_path=savedir
    )
    return adata


def preprocess_adata(adata: ad.AnnData, model_dir: str) -> ad.AnnData:
    """Minimal preprocessing for the model."""
    adata.X = adata.raw.X
    adata = adata[:, adata.var.highly_variable == True].copy()  # noqa: E712
    scvi.model.base.ArchesMixin.prepare_query_anndata(adata, model_dir)

    return adata


def postprocess_adata(adata: ad.AnnData) -> ad.AnnData:
    """Postprocessing so the AnnData types are amenable to saving."""
    for k in ["feature_name", "feature_reference", "feature_biotype"]:
        new = getattr(adata.var, k).cat.add_categories("Unknown")
        setattr(adata.var, k, new)

    adata.var.fillna("Unknown", inplace=True)
    obj_cols = adata.var.select_dtypes(include="object").columns
    adata.var.loc[:, obj_cols] = adata.var.loc[:, obj_cols].astype("str")

    return adata


def convert_legacy_model(savedir: str, config: dict) -> str:
    """Convert the legacy model to the new format."""
    unzipped = pooch.retrieve(
        url=config["legacy_model_url"],
        fname=config["legacy_model_dir"],
        known_hash=config["known_hash"],
        processor=pooch.Unzip(),
        path=savedir,
    )
    legacy_model_dir = Path(unzipped[0]).parent
    new_model_dir = os.path.join(savedir, config["new_model_dir"])
    make_parents(new_model_dir)
    scvi.model.SCANVI.convert_legacy_save(legacy_model_dir, new_model_dir)

    return new_model_dir


def load_model(model_dir: str, adata: ad.AnnData) -> scvi.model.SCANVI:
    """Load the model."""
    return scvi.model.SCANVI.load(model_dir, adata=adata)


def minify_model_and_save(
    model: scvi.model.SCANVI,
    savedir: str,
    config: dict,
    latent_qzm_key: str = "X_scanvi_qzm",
    latent_qzv_key: str = "X_scanvi_qzv",
) -> str:
    """Minify the model and save it."""
    _, qzv = model.get_latent_representation(give_mean=False, return_dist=True)

    # use pre-computed qzm from HLCA team
    emb_path = os.path.join(savedir, config["emb_fname"])
    adata = sc.read(emb_path, backup_url=config["emb_url"])
    qzm = adata.X

    model.adata.obsm[latent_qzm_key] = qzm
    model.adata.obsm[latent_qzv_key] = qzv
    model.minify_adata(
        use_latent_qzm_key=latent_qzm_key, use_latent_qzv_key=latent_qzv_key
    )

    model_dir = os.path.join(savedir, config["minified_model_dir"])
    make_parents(model_dir)
    model.save(model_dir, overwrite=True, save_anndata=True)

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
    savedir = tempfile.TemporaryDirectory().name

    model_dir = convert_legacy_model(savedir, config)
    adata = load_adata(savedir, config)
    adata = preprocess_adata(adata, model_dir, config)
    model = load_model(model_dir, adata)

    model.adata = postprocess_adata(model.adata)
    model_dir = minify_model_and_save(model, savedir, config)

    hub_model = create_hub_model(model_dir, config)
    upload_hub_model(hub_model, HF_API_TOKEN, config)


if __name__ == "__main__":
    main()
