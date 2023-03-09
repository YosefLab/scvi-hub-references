import os
from pathlib import Path
from typing import Tuple

import anndata as ad
import pooch
import scanpy as sc
import scvi


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
) -> scvi.model.SCANVI:
    qzm, qzv = model.get_latent_representation(return_dist=True)
    model.adata.obsm[latent_qzm_key] = qzm
    model.adata.obsm[latent_qzv_key] = qzv
    model.minify_adata(use_latent_qzm_key=latent_qzm_key, use_latent_qzv_key=latent_qzv_key)

    return model


def create_hub_model(model: scvi.model.SCANVI):
    pass


def upload_hub_model(
    hubmodel: scvi.hub.HubModel,
    repo_name: str = "non_small_cell_lung_cancer",
    repo_token: str = None,
):
    pass


if __name__ == "__main__":
    model, adata = load_model_and_dataset()
    model = minify_model(model)
    hubmodel = create_hub_model(model)
    upload_hub_model(hubmodel)
