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

from utils import (
    HF_API_TOKEN,
    create_hub_model,
    get_temp_dir,
    load_config, 
    minify_model_and_save,
    retrieve_model,
    upload_hub_model,
)

sys.stderr = open(snakemake.log[0], "w")  # noqa: F821
sys.stdout = open(snakemake.log[1], "w")  # noqa: F821


def main():
    """Run main."""
    config = load_config(snakemake.input[0])  # noqa: F821
    save_dir = get_temp_dir()

    base_path = retrieve_model(
        model_url=config["model_url"],
        model_dir=config["model_save_dir"],
        save_dir=save_dir,
        known_hash=config["known_hash"],
    )
    adata = sc.read(os.path.join(base_path, config["adata_fname"]))
    model_dir = os.path.join(base_path, config["model_dir"])
    model = scvi.model.SCANVI.load(model_dir, adata=adata)

    model_dir = minify_model_and_save(
        model=model,
        save_dir=save_dir,
        model_dir=config["minified_model_dir"],
    )

    hub_model = create_hub_model(
        model_dir=model_dir,
        training_data_url=config["model_url"],
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
