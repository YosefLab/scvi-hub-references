"""Upload HLCA."""
import sys
import scvi
import requests
import scanpy as sc
import urllib

scvi.settings.seed = 0
scvi.settings.reset_logging_handler()
sys.stderr = open(snakemake.log[0], "w")  # noqa: F821
sys.stdout = open(snakemake.log[1], "w")  # noqa: F821

tissue = snakemake.tissue

# Folder structure for files and models.
model_dir = snakemake.output_dir
data_dir = snakemake.input_data_dir
models_keys = snakemake.models_keys

# Download unprocessed data from Zenodo.
res = requests.get("https://zenodo.org/api/records/7580683")
tissue_download_path = {ind['key'][3:-14]:ind['links']['self'] for ind in res.json()['files']}
output_fn = f'{data_dir}/TS_{tissue}.h5ad'
urllib.request.urlretrieve(tissue_download_path[tissue], output_fn)

adata = sc.read(output_fn)

# Following parameters are specific to Tabula Sapiens dataset
labels_key='cell_ontology_class'
batch_key = 'donor_assay'

# Additional model parameters.
scvi_model_kwargs = {
    "dropout_rate": 0.05,
    "dispersion": "gene",
    "n_layers": 3,
    "n_latent": 20,
    "gene_likelihood": "nb",
    "use_batch_norm": "none",
    "use_layer_norm": "both",
    "encode_covariates": True,
}
scanvi_model_kwargs = {
    "unlabeled_category": 'unknown'
}
condscvi_model_kwargs = {
    "n_latent": 5,
    "n_layers": 2,
    "dropout_rate": 0.05,
    "weight_obs": False
}

# HVG gene filtering.
sc.experimental.pp.highly_variable_genes(
    adata,
    n_top_genes=4000,
    subset=True,
    flavor='pearson_residuals',
    batch_key=batch_key,
)

model = {}

# TODO upload preprocessed data to Zenodo?
# Save preprocessed data.
adata.write(model_dir + f"{tissue}/adata.h5ad")

if 'scvi' in models_keys:
    # Train scVI
    scvi.model.SCVI.setup_anndata(
                adata,
                batch_key=batch_key,
                labels_key=labels_key
            )
    scvi_model = scvi.model.SCVI(adata, **scvi_model_kwargs)
    scvi_model.train(max_epochs=100, train_size=1.0)
    # Transfer before minifying model.
    scanvi_model = scvi.model.SCANVI.from_scvi_model(
            scvi_model,
            **scanvi_model_kwargs
        )
    
    # Make models minified and save models.
    qzm, qzv = scvi_model.get_latent_representation(give_mean=False, return_dist=True)
    scvi_model.adata.obsm['X_scvi_qzm'] = qzm
    scvi_model.adata.obsm['X_scvi_qzv'] = qzv
    scvi_model.to_latent_mode(use_latent_qzm_key='X_scvi_qzm', use_latent_qzv_key='X_scvi_qzv')
    scvi_model.save(model_dir + f"{tissue}/scvi", overwrite=True, save_anndata=True)
    model['scvi'] = model_dir + f"{tissue}/scvi"

if 'scanvi' in models_keys:
    # Train scANVI
    scanvi_model.train(
                max_epochs=20,
                n_samples_per_label=20,
                train_size=1.0,
                plan_kwargs={"n_epochs_kl_warmup": 10}
            )
    qzm, qzv = scanvi_model.get_latent_representation(give_mean=False, return_dist=True)
    scanvi_model.adata.obsm['X_scanvi_qzm'] = qzm
    scanvi_model.adata.obsm['X_scanvi_qzv'] = qzv
    scanvi_model.to_latent_mode(use_latent_qzm_key='X_scanvi_qzm', use_latent_qzv_key='X_scanvi_qzv')
    scanvi_model.save(model_dir + f"{tissue}/scanvi", overwrite=True, save_anndata=True)
    model['scanvi'] = model_dir + f"{tissue}/scanvi"

if 'condscvi' in models_keys:
    # Train CondSCVI
    scvi.model.CondSCVI.setup_anndata(
        adata,
        labels_key=labels_key
    )
    condscvi_model = scvi.model.CondSCVI(
        adata, 
    )
    condscvi_model.train(max_epochs=200, train_size=1.0)
    condscvi_model.save(model_dir + f"{tissue}/condscvi", overwrite=True, save_anndata=False)
    model['condscvi'] = model_dir + f"{tissue}/condscvi"
    
if 'stereoscope' in models_keys:
    # Train Stereoscope
    scvi.external.RNAStereoscope.setup_anndata(
        adata,
        labels_key = labels_key)
    stereoscope_model = scvi.external.stereoscope.RNAStereoscope(
        adata
    )
    stereoscope_model.train()
    stereoscope_model.save(model_dir + f"{tissue}/stereoscope", overwrite=True, save_anndata=False)
    model['stereoscope'] = model_dir + f"{tissue}/stereoscope"
    
# TODO upload newly trained models to Zenodo?
# TODO use output of this function in next rule


# return ['local', model]