# scvi-hub-references

Code for the pre-trained reference models uploaded to scvi-hub on HuggingFace.

First, install pre-commit from pip and run `pre-commit install` at the root of the repository.

To run the workflow for a particular reference model, create a conda environment with 
snakemake, activate the environment, `cd` in `workflow/`, and run the following command:

```
export HF_API_TOKEN=TOKEN_VAL
snakemake -R {model_name} --use-conda --cores all
```
