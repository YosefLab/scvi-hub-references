# scvi-hub-references

Code for the pre-trained reference models uploaded to scvi-hub on HuggingFace.

First, install pre-commit from pip and run `pre-commit install` at the root of the repository.

To run the uploading workflow for a particular reference model, create a conda 
environment snakemake, activate the environment, `cd` into the model's directory, and 
run the following command:

```
export HF_API_TOKEN=TOKEN_VAL
snakemake --forceall --use-conda --cores all --envvars HF_API_TOKEN
```
