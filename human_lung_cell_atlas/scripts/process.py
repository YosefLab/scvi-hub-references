"""Upload HLCA."""
import sys

import scvi

scvi.settings.seed = 0
scvi.settings.reset_logging_handler()
sys.stderr = open(snakemake.log[0], "w")  # noqa: F821
sys.stdout = open(snakemake.log[1], "w")  # noqa: F821

# download the files in a temporary directory

# create latent adata

# upload to hugging face
