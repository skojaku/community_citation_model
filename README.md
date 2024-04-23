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

First, set the environment specific variables in the [`repro/workflow/config.yaml`](./repro/config/config.yaml) file.

Second, run the following command to run the workflows:

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
