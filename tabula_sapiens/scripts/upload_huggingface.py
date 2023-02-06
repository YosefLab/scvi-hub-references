"""Upload Tabula sapiens."""
import anndata
import sys
import tempfile
import snakemake
from pathlib import Path
import pooch
import requests
import scvi
from scvi.data._download import _download
from scvi.model.base import ArchesMixin
import gc
from scvi.hub import HubMetadata, HubModelCardHelper, HubModel
from train_model import retrain_models

scvi.settings.seed = 0
scvi.settings.reset_logging_handler()
sys.stderr = open(snakemake.log[0], "w")  # noqa: F821
sys.stdout = open(snakemake.log[1], "w")  # noqa: F821
HF_TOKEN = snakemake.params["hf_api_token"]

# Folder structure for files and models.
model_keys = snakemake.params["model_keys"]
tissue = snakemake.params["tissue"]
retrain = snakemake.params["retrain"]
tmpdir = tempfile.TemporaryDirectory()

res = requests.get("https://zenodo.org/api/records/7608635")
adata_download_path = {
    ind['key'][:-19]:ind['links']['self']
    for ind in res.json()['files'] if '.h5ad' in ind['key']}
model_download_path = {
    ind['key'][:-25]:ind['links']['self']
    for ind in res.json()['files'] if not '.h5ad' in ind['key']}

if retrain:
    retrain_models(tissue=tissue, output_dir=tmpdir.name, model_keys=model_keys)
else:
    # TODO Model download here
    
    output_fn = f'{tmpdir.name}/TS_{tissue}.h5ad'

    # download the model
    model_url = model_download_path[tissue]
    unzipped = pooch.retrieve(
        url=model_url,
        fname=f"{tissue}_pretrained_models",
        known_hash=None,
        processor=pooch.Unzip(),
        path=tmpdir.name,
    )

# create model card
citation = r"""The Tabula Sapiens: A multi-organ, single-cell transcriptomic atlas of humans.
The Tabula Sapiens Consortium.
Science 2022.05.13; doi: https://doi.org/10.1126/science.abl4896"""
desc = r"""Tabula sapiens. An across organ dataset of cell-types in human tissues"""

for model in model_keys:
    # upload the model and minified data to hugging face
    # create metadata
    hm = HubMetadata.from_dir(
        f"{tissue}_pretrained_models/{model}",
        anndata_version = anndata.__version__,
        training_data_url=adata_download_path[tissue],
    )
    
    if model in ['scVI', 'scANVI']:
        minified = True
    else:
        minified = False
    
    hmch = HubModelCardHelper.from_dir(
        f"{tissue}_pretrained_models_{model}",
        license_info="cc-by-4.0",
        anndata_version=anndata.__version__,
        data_is_minified=minified,
        data_is_annotated=True,
        tissues = [tissue],
        training_data_url=adata_download_path[tissue],
        training_code_url="https://github.com/scvi-hub-references/tabula_sapiens/train_model.py",
        description=desc,
        references=citation,
        data_modalities=["rna"],
    )
    # create model
    hmo = HubModel(f"{tissue}_pretrained_models_{model}", metadata=hm, model_card=hmch)
    # push
    hmo.push_to_huggingface_hub(repo_name=f"scvi-tools/tabula-sapiens_{tissue}_{model}", repo_token=HF_TOKEN, repo_create=True)

