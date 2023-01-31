"""Upload HLCA."""
import anndata
import tempfile
import snakemake
from pathlib import Path
import pooch
import scvi
from scvi.data._download import _download
from scvi.model.base import ArchesMixin
import gc
from scvi.hub import HubMetadata, HubModelCardHelper, HubModel

scvi.settings.seed = 0
scvi.settings.reset_logging_handler()
sys.stderr = open(snakemake.log[0], "w")  # noqa: F821
sys.stdout = open(snakemake.log[1], "w")  # noqa: F821

tmpdir = tempfile.TemporaryDirectory()

# download the model
model_url = "https://zenodo.org/record/6337966/files/HLCA_reference_model.zip"
unzipped = pooch.retrieve(
    url=model_url,
    fname=f"HLCA_reference_model",
    known_hash=None,
    processor=pooch.Unzip(),
    path=tmpdir.name,
)

# convert legacy save format to new format:
legacy_model_dir = Path(unzipped[0]).parent
new_model_dir = legacy_model_dir.parent / "HLCA_reference_model_new"
scvi.model.SCANVI.convert_legacy_save(legacy_model_dir, new_model_dir)

# download the dataset
adata = scvi.data.cellxgene(
    "https://cellxgene.cziscience.com/e/066943a2-fdac-4b29-b348-40cede398e4e.cxg/",
    filename = "adata.h5ad",
    save_path= new_model_dir.parent,
)

# do some minimal processing of the data so it's ready for use with the model
# get the raw, unnormalized counts
adata.X = adata.raw.X
del adata.raw
# filter to highly variable genes
adata = adata[:, adata.var.highly_variable == True].copy()
# 4 genes are missing from cellxgene that are in the model. prepare_query_anndata will pad them with zeros.
ArchesMixin.prepare_query_anndata(adata, new_model_dir)

# load the model
model = scvi.model.SCANVI.load(new_model_dir, adata=adata)

# get the latent representation
_, qzv = model.get_latent_representation(give_mean=False, return_dist=True)
model.adata.obsm["X_latent_qzv"] = qzv
# for qzm, we'll use the qzm pre-computed by the HLCA team. We can get this from Zenodo.
emb_url = "https://zenodo.org/record/6337966/files/HLCA_emb_and_metadata.h5ad"
_download(emb_url, new_model_dir.parent, save_fn="adata_emb.h5ad")
bdata = anndata.read_h5ad(Path(new_model_dir).parent / "adata_emb.h5ad")
model.adata.obsm["X_latent_qzm"] = bdata.X
del bdata
gc.collect()

# minify the data
model.minify_adata()

# save the model and minified adata
# before we can do this we need to massage the adata so it's type are amenable to saving (str)
model.adata.var.feature_name = model.adata.var.feature_name.cat.add_categories("Unknown")
model.adata.var.feature_reference = model.adata.var.feature_reference.cat.add_categories("Unknown")
model.adata.var.feature_biotype = model.adata.var.feature_biotype.cat.add_categories("Unknown")
model.adata.var.fillna("Unknown", inplace=True)
obj_cols = model.adata.var.select_dtypes(include='object').columns
model.adata.var.loc[:, obj_cols] = model.adata.var.loc[:, obj_cols].astype("str")
# now save
model_dir = new_model_dir.parent / "HLCA_reference_model_new_minified",
model.save(
    model_dir,
    save_anndata=True,
)

# upload the model and minified data to hugging face
# create metadata
hm = HubMetadata.from_dir(
    model_dir,
    anndata_version = anndata.__version__,
    training_data_url="https://cellxgene.cziscience.com/e/066943a2-fdac-4b29-b348-40cede398e4e.cxg/",
)
# create model card
citation = r"""An integrated cell atlas of the human lung in health and disease  
L Sikkema, D Strobl, L Zappia, E Madissoon, NS Markov, L Zaragosi, M Ansari, M Arguel, L Apperloo, C Bécavin, M Berg, E Chichelnitskiy, M Chung, A Collin, ACA Gay, B Hooshiar Kashani, M Jain, T Kapellos, TM Kole, C Mayr, M von Papen, L Peter, C Ramírez-Suástegui, J Schniering, C Taylor, T Walzthoeni, C Xu, LT Bui, C de Donno, L Dony, M Guo, AJ Gutierrez, L Heumos, N Huang, I Ibarra, N Jackson, P Kadur Lakshminarasimha Murthy, M Lotfollahi, T Tabib, C Talavera-Lopez, K Travaglini, A Wilbrey-Clark, KB Worlock, M Yoshida, Lung Biological Network Consortium, T Desai, O Eickelberg, C Falk, N Kaminski, M Krasnow, R Lafyatis, M Nikolíc, J Powell, J Rajagopal, O Rozenblatt-Rosen, MA Seibold, D Sheppard, D Shepherd, SA Teichmann, A Tsankov, J Whitsett, Y Xu, NE Banovich, P Barbry, TE Duong, KB Meyer, JA Kropski, D Pe’er, HB Schiller, PR Tata, JL Schultze, AV Misharin, MC Nawijn, MD Luecken, F Theis  
bioRxiv 2022.03.10.483747; doi: https://doi.org/10.1101/2022.03.10.483747"""
desc = r"""The first integrated, universal transcriptomic reference of the human lung on the single-cell level.  
For more details, see https://github.com/LungCellAtlas/HLCA."""
hmch = HubModelCardHelper.from_dir(
    model_dir,
    license_info="cc-by-4.0",
    anndata_version=anndata.__version__,
    data_is_minified=True,
    data_is_annotated=True,
    tissues = ['nose', 'respiratory airway', 'lung parenchyma'],
    training_data_url="https://cellxgene.cziscience.com/e/066943a2-fdac-4b29-b348-40cede398e4e.cxg/",
    training_code_url="https://github.com/LungCellAtlas/HLCA_reproducibility",
    description=desc,
    references=citation,
    data_modalities=["rna"],
)
# create model
hmo = HubModel(model_dir, metadata=hm, model_card=hmch)
# push
repo_token = "token" # TODO provide your token
hmo.push_to_huggingface_hub(repo_name="scvi-tools/human-lung-cell-atlas", repo_token=repo_token, repo_create=True)

