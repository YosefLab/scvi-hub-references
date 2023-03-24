import json
import os
import pathlib
import tempfile
from pathlib import Path
from typing import Callable, Optional

import anndata as ad
import numpy as np
import pooch
import scanpy as sc
import scvi
from scvi.hub import HubMetadata, HubModel, HubModelCardHelper

HF_API_TOKEN = os.environ["HF_API_TOKEN"]
scvi.settings.seed = 2023
scvi.settings.reset_logging_handler()


def make_parents(*paths) -> None:
    """Make parent directories of a file path if they do not exist."""
    for p in paths:
        pathlib.Path(p).parent.mkdir(parents=True, exist_ok=True)


def load_config(config_path: str) -> dict:
    """Load a JSON configuration file as a Python dictionary."""
    with open(config_path) as f:
        config = json.load(f)
    return config


def get_temp_dir() -> str:
    return tempfile.TemporaryDirectory().name


def load_adata(
    *,
    url: str,
    fname: str,
    save_dir: str,
    cellxgene: bool = False,
) -> ad.AnnData:
    """Load an AnnData object from a URL."""
    path = os.path.join(save_dir, fname)
    make_parents(path)
    if cellxgene:
        return scvi.data.cellxgene(url, filename=fname, save_path=save_dir)
    else:
        return sc.read(path, backup_url=url)
    

def retrieve_model(
    *,
    model_url: str,
    model_dir: str,
    save_dir: str,
    known_hash: Optional[str] = None,
    processor: Callable = pooch.Untar(),
) -> str:
    """Retrieve a model."""
    unzipped = pooch.retrieve(
        url=model_url,
        fname=model_dir,
        known_hash=known_hash,
        processor=processor,
        path=save_dir,
    )
    return Path(unzipped[0]).parent


def retrieve_and_convert_legacy_model(
    *,
    model_cls: scvi.model.base.BaseModelClass,
    model_url: str,
    legacy_model_dir: str,
    new_model_dir: str,
    save_dir: str,
    known_hash: Optional[str] = None,
    processor: Callable = pooch.Untar(),
) -> str:
    """Retrieve and convert a legacy model."""
    legacy_model_dir = retrieve_model(
        model_url=model_url,
        model_dir=legacy_model_dir,
        save_dir=save_dir,
        known_hash=known_hash,
        processor=processor,
    )
    new_model_dir = os.path.join(save_dir, new_model_dir)
    make_parents(new_model_dir)

    model_cls.convert_legacy_save(legacy_model_dir, new_model_dir)

    return new_model_dir


def save_model(
    *,
    model: scvi.model.base.BaseModelClass,
    model_dir: str,
    save_dir: str,
) -> str:
    model_dir = os.path.join(save_dir, model_dir)
    make_parents(model_dir)
    model.save(model_dir, overwrite=True, save_anndata=True)

    return model_dir


def minify_model_and_save(
    *,
    model: scvi.model.base.BaseMinifiedModeModelClass,
    model_dir: str,
    save_dir: str,
    prefix: Optional[str] = None,
    latent_qzm_key: str = "latent_qzm",
    latent_qzv_key: str = "latent_qzv",
    qzm: Optional[np.ndarray] = None,
    qzv: Optional[np.ndarray] = None,
) -> str:
    """Minify the model and save it to disk."""
    if prefix is None:
        prefix = model.__class__.__name__
    latent_qzm_key = f"{prefix}_{latent_qzm_key}"
    latent_qzv_key = f"{prefix}_{latent_qzv_key}"

    _qzm, _qzv = model.get_latent_representation(give_mean=False, return_dist=True)
    qzm = qzm if qzm is not None else _qzm
    qzv = qzv if qzv is not None else _qzv
    model.adata.obsm[latent_qzm_key] = qzm
    model.adata.obsm[latent_qzv_key] = qzv
    model.minify_adata(
        use_latent_qzm_key=latent_qzm_key, use_latent_qzv_key=latent_qzv_key
    )

    return save_model(model=model, model_dir=model_dir, save_dir=save_dir)


def create_hub_model(
    *,
    model_dir: str,
    training_data_url: str,
    training_code_url: str,
    data_is_minified: bool,
    data_is_annotated: bool,
    tissues: list,
    data_modalities: list,
    description: str,
    references: str,
    license_info: str
) -> HubModel:
    """Create a HubModel instance."""
    metadata = HubMetadata.from_dir(
        model_dir,
        training_data_url=training_data_url,
        anndata_version=ad.__version__,
    )
    card = HubModelCardHelper.from_dir(
        model_dir,
        training_data_url=training_data_url,
        training_code_url=training_code_url,
        data_is_minified=data_is_minified,
        data_is_annotated=data_is_annotated,
        tissues=tissues,
        data_modalities=data_modalities,
        description=description,
        references=references,
        license_info=license_info,
        anndata_version=ad.__version__,
    )
    return HubModel(model_dir, metadata=metadata, model_card=card)


def upload_hub_model(
    hub_model: HubModel, repo_name: str, repo_token: str
) -> HubModel:
    """Upload a HubModel to HuggingFace.
    
    If the repository already exists, a new commit will be pushed.
    """
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
    return hub_model
