"""Upload Tabula sapiens."""
import sys
import scvi
import urllib
import logging
import requests
import shutil
from scvi.hub import HubMetadata, HubModel, HubModelCardHelper

scvi.settings.seed = 0
scvi.settings.reset_logging_handler()
sys.stderr = open(snakemake.log[0], "w")  # noqa: F821
sys.stdout = open(snakemake.log[1], "w")  # noqa: F821

# Folder structure for files and models.


model_keys = snakemake.model_keys
tissue = snakemake.tissue
models_input = snakemake.models
output_dir = snakemake.output_dir

res = requests.get("https://zenodo.org/api/records/7580683")
tissue_download_path = {ind['key'][3:-14]:ind['links']['self'] for ind in res.json()['files']}
output_fn = f'{data_dir}/TS_{tissue}.h5ad'

if models_input is not None:
    models = models_input
else:
    #TODO add link to 
    res = requests.get("https://zenodo.org/api/records/XXXX")
    model_download_path = {ind['key'][3:-14]:ind['links']['self'] for ind in res.json()['files']}
    urllib.request.urlretrieve(model_download_path[tissue], f'{output_dir}/tissue/pretrained_models.tar.gz')
    shutil.unpack_archive(f'{output_dir}/tissue/pretrained_models.tar.gz', f'{output_dir}/tissue', 'gz')
    models = {model: f'{output_dir}/tissue/{model}' for model in model_keys}
    
for model_key in models.keys():
    hm = HubMetadata.from_dir(models[model_key], anndata_version="0.8.0")

    hmch = HubModelCardHelper.from_dir(
        models[model_key],
        license_info="cc-by-4.0",
        anndata_version="0.8.0",
        data_modalities=["rna"],
        data_is_annotated=True,
        tissues=[tissue],
        training_data_url=None,
        training_code_url=None,
        description=f"This model contains a trained {model_key} model using Tabula sapiens for {tissue}. Original data was filtered and input data is available from Zenodo: https://doi.org/10.5281/zenodo.7580683",
        references="""
            The full data is hosted at https://cellxgene.cziscience.com/collections/e5f58829-1a66-40b5-a624-9046778e74f5.
            Please refer to the original publication https://www.science.org/doi/10.1126/science.abl4896 for reference and adhere to their data reuse guidelines.
            """,
    )
    hmo = HubModel(models[model_key], metadata=hm, model_card=hmch)
    logging.info(hmo)
    hmo.push_to_huggingface_hub(
        repo_name=repo_name, repo_token=repo_token, repo_create=True)