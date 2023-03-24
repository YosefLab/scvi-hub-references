"""Human Lung Cell Atlas."""
import os
import sys

import anndata as ad
import numpy as np
import pooch
import scanpy as sc
import scvi

from utils import (
    HF_API_TOKEN,
    create_hub_model,
    get_temp_dir,
    load_adata,
    load_config, 
    minify_model_and_save,
    retrieve_and_convert_legacy_model,
    upload_hub_model,
)

sys.stderr = open(snakemake.log[0], "w")  # noqa: F821
sys.stdout = open(snakemake.log[1], "w")  # noqa: F821


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


def main():
    """Run main."""
    config = load_config(snakemake.input[0])  # noqa: F821
    save_dir = get_temp_dir()

    model_dir = retrieve_and_convert_legacy_model(
        model_cls=scvi.model.SCANVI,
        model_url=config["legacy_model_url"],
        legacy_model_dir=config["legacy_model_dir"],
        new_model_dir=config["new_model_dir"],
        save_dir=save_dir,
        known_hash=config["known_hash"],
        processor=pooch.Unzip(),
    )
    adata = load_adata(
        url=config["adata_url"], 
        fname=config["adata_fname"], 
        save_dir=save_dir, 
        cellxgene=True,
    )
    adata = preprocess_adata(adata, model_dir)
    model = scvi.model.SCANVI.load(model_dir, adata=adata)

    model.adata = postprocess_adata(model.adata)
    qzm = load_adata(
        url=config["emb_url"], fname=config["emb_fname"], save_dir=save_dir
    ).X
    model_dir = minify_model_and_save(
        model=model,
        save_dir=save_dir,
        model_dir= config["minified_model_dir"], 
        qzm=qzm,
    )

    hub_model = create_hub_model(
        model_dir=model_dir,
        training_data_url=config["adata_url"],
        training_code_url=config["training_code_url"],
        data_is_minified=True,
        data_is_annotated=True,
        tissues=config["tissues"],
        data_modalities=config["data_modalities"],
        description=config["description"],
        references=config["citation"],
        license_info=config["license_info"],
    )
    upload_hub_model(hub_model, config["repo_name"], HF_API_TOKEN)


if __name__ == "__main__":
    main()
