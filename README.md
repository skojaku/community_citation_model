# community_citation_model
This is a Python repository implementing Community Citation Model.

# Reproducing the results

The code to generate the results is available in the [`repro`](./repro) directory. All scripts are written in Python and can be run with Snakemake.

## Installing the package for reproducing the results

In order to install the package for reproducing the results, run the following command:

```bash
cd repro
conda env create -f environment.yml
```

This will create a conda environment named `citationdynamics` with all the packages required to run the scripts.
Then, activate the environment with the following command:

```bash
conda activate citationdynamics
```

## Running the scripts

We built the workflows with [Snakemake](https://snakemake.readthedocs.io/en/stable/), a workflow management system.

### Set up the environment variables
First, set the environment specific variables in the [`repro/workflow/config.yaml`](./repro/config/config.yaml) file.
The variables defined in the [`repro/config/config.yaml`](./repro/config/config.yaml) file are
- `data_dir`: The directory where the raw data is stored.
- `supp_data_dir`: The directory where the supplementary data is stored (which is [repro/data_supp](./repro/data_supp)).
- `aps_data_dir`: The directory where the APS data is stored.
- `legcit_data_dir`: The directory where the Case Law data is stored.
- `uspto_data_dir`: The directory where the USPTO data is stored.

Please download the raw data from the following links and put them in the `data` directory.
- APS: https://journals.aps.org/datasets
- Case Law: https://case.law/
- USPTO: https://patentsview.org/

Alternatively, you can download the preprocessed data from the following link and put them in the `<data_dir>/<data name>/preprocessed` directory.
The `<data name>` must be one of the following: aps (for the APS), legcitv2 (for the Case Law), uspto (for the USPTO).
Due to the proprietary nature of the APS data, we cannot provide the preprocessed data.

Figshare: https://figshare.com/account/projects/202335/articles/25655514

### Data preprocessing

** Skip this step if you have downloaded the preprocessed the data.**

A data preprocessing script is provided in workflow/<data name>/Snakefile. Go to the workflow/<data name> directory and run the following command to preprocess the data:

```bash
snakemake --cores <number of cores>
```

### Running the workflows

Run the following command to run the workflows:

```bash
snakemake --cores <number of cores>
```
This will generate the results in the [`repro/results`](./repro/results) directory.


# Installing the package for the CCMs
TBD

# TODOs

- [ ] CCM
  - [x] Set up repo
  - [x] Write test script
  - [ ] Move the code and remove unused functions
  - [ ] Rename variables and functions
  - [ ] Write the installation instruction on README
  - [ ] Test
- [ ] Automatic workflow for reproduceability
  - [x] Move Snakefile and workflow scripts
  - [ ] Remove unused rules & variables
  - [ ] Pack all rules into one Snakemake
  - [ ] Replace old CCM code with new CCM code
  - [ ] Run the workflow and make sure that it works (for APS)
  - [ ] Write the instruction on README.
