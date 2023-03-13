import json
import os
import pathlib
from pathlib import Path
from typing import Tuple

import anndata as ad
import pooch
import scanpy as sc
import scvi


def make_parents(*paths) -> None:
    """Make parent directories of a file path if they do not exist."""
    for p in paths:
        pathlib.Path(p).parent.mkdir(parents=True, exist_ok=True)


def load_model_and_dataset() -> Tuple[scvi.model.SCANVI, ad.AnnData]:
    model_url = "https://zenodo.org/record/7227571/files/core_atlas_scanvi_model.tar.gz"
    unzipped = pooch.retrieve(
        url=model_url,
        fname="lung_cancer_scanvi",
        known_hash="60d6c0ccbad89178a359b3bd2f2981638c86b260011d8bd1977c989fbbc5ad7e",
        processor=pooch.Untar(),
    )[0]
    base_path = Path(unzipped).parent
    model_path = os.path.join(base_path, "full_atlas_hvg_integrated_scvi_scanvi_model")
    adata_path = os.path.join(base_path, "full_atlas_hvg_integrated_scvi_integrated_scanvi.h5ad")

    adata = sc.read(adata_path)
    model = scvi.model.SCANVI.load(model_path, adata=adata)
    
    return model, adata


def minify_model(
    model: scvi.model.SCANVI,
    latent_qzm_key: str = "X_latent_qzm",
    latent_qzv_key: str = "X_latent_qzv",
    save_dir: str = "lung_cancer_scanvi_minified"
) -> scvi.model.SCANVI:
    """Minify the model and save it to disk."""
    qzm, qzv = model.get_latent_representation(return_dist=True)
    model.adata.obsm[latent_qzm_key] = qzm
    model.adata.obsm[latent_qzv_key] = qzv
    model.minify_adata(use_latent_qzm_key=latent_qzm_key, use_latent_qzv_key=latent_qzv_key)

    model_out = os.path.join("models", save_dir)
    make_parents(model_out)
    model.save(model_out)

    return model


def create_hub_model(model_dir: str):
    model_in = os.path.join("models", model_dir)

    metadata = scvi.hub.HubMetadata.from_dir(
        model_in,
        anndata_version=ad.__version__,
        training_data_url="https://zenodo.org/record/7227571/files/core_atlas_scanvi_model.tar.gz",
    )

    citation = r""""""
    description = r""""""
    card = scvi.hub.HubModelCardHelper.from_dir(
        model_in,
        license_info="cc-by-4.0",
        anndata_version=ad.__version__,
        data_is_minified=True,
        data_is_annotated=True,
        tissues=[],
        training_data_url="https://zenodo.org/record/7227571/files/core_atlas_scanvi_model.tar.gz",
        training_code_url="https://github.com/icbi-lab/luca",
        description=description,
        references=citation,
        data_modalities=["rna"],
    )

    hubmodel = scvi.hub.HubModel(model_in, metadata=metadata, model_card=card)

    return hubmodel


def upload_hub_model(
    hubmodel: scvi.hub.HubModel,
    repo_name: str = "non_small_cell_lung_cancer",
    repo_token: str = None,
):
    hubmodel.push_to_huggingface_hub(
        repo_name=repo_name, repo_token=repo_token, repo_create=True,
    )


if __name__ == "__main__":
    model, adata = load_model_and_dataset()
    model = minify_model(model)
    hubmodel = create_hub_model(model)
    upload_hub_model(hubmodel)
