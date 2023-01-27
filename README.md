# scvi-hub-references

Code for the pre-trained reference models uploaded to scvi-hub on HuggingFace.

First, intstall pre-commit from pip and run `pre-commit install` at the root of the repository.

To run the code, create a conda environment, install snakemake in this environment, then run the following command:

```
snakemake --use-conda --cores all
```

from the corresponding reference directory's workflow directory.
