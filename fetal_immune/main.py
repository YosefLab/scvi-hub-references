import os
from pathlib import Path

import anndata as ad
import pooch
import scanpy as sc
import scvi


def load_dataset() -> ad.AnnData:
    adata = sc.read(
        "data/fetal_immune.h5ad", 
        backup_url="https://cellgeni.cog.sanger.ac.uk/developmentcellatlas/fetal-immune/PAN.A01.v01.raw_count.20210429.HSC_IMMUNE.embedding.h5ad"
    )
    return adata


def load_model() -> scvi.model.SCVI:
    model_url = "https://cellgeni.cog.sanger.ac.uk/developmentcellatlas/fetal-immune/scVI_models/scvi_HSC_IMMUNE_model.tar.gz"
    model_path = pooch.retrieve(
        url=model_url,
        fname="fetal_immune_scvi",
        known_hash="52dcbd672d626b00a1fca2c48a466256cab15ed9e068542dfdee7cfa7a3d5d21",
        processor=pooch.Untar(),
    )[0]
    legacy_model_dir = Path(model_path).parent
    new_model_dir = os.path.join(legacy_model_dir.parent, "fetal_immune_scvi_new")

    weights_path = os.path.join(legacy_model_dir, "model.pt")
    if os.path.exists(weights_path):
        new_path = os.path.join(legacy_model_dir, "model_params.pt")
        os.rename(weights_path, new_path)

    scvi.model.SCVI.convert_legacy_save(legacy_model_dir, new_model_dir)


if __name__ == "__main__":
    adata = load_dataset()
    model = load_model()

    