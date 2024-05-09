# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-12-27 14:19:48
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-07-06 11:46:02
from os.path import join as j
import numpy as np

#
# Reproduceability test
#
N_SIM_NET_SAMPLES = 5

## Spherical model
spherical_model_paramspace = to_paramspace(params_spherical_model)
canonical_spherical_model_paramspace = to_paramspace(params_canonical_spherical_model)
REPRO_TEST_GEOMETRIC_MODEL_FILE = j(DATA_DIR, "{data}", "derived", f"model_{spherical_model_paramspace.wildcard_pattern}.pt")
REPRO_TEST_UMAP_FILE = j(DATA_DIR, "{data}", "derived", f"umap_{spherical_model_paramspace.wildcard_pattern}.npz")
REPRO_TEST_UMAP_XNET_FILE = j(DATA_DIR, "{data}", "derived", f"xnet_{spherical_model_paramspace.wildcard_pattern}.xnet")
REPRO_TEST_GEO_SIM_NET_FILE = j(DATA_DIR, "{data}", "derived", "simulated_networks", f"net_{spherical_model_paramspace.wildcard_pattern}"+"_sample~{sample}.npz")
REPRO_TEST_GEO_SIM_NODE_FILE =j(DATA_DIR, "{data}", "derived", "simulated_networks", f"node_{spherical_model_paramspace.wildcard_pattern}"+"_sample~{sample}.csv")

## Baselines
REPRO_TEST_BASELINE_SIM_NET_FILE = j(DATA_DIR, "{data}", "derived", "simulated_networks", "net_model~{model}_sample~{sample}.npz")
REPRO_TEST_BASELINE_LTCM_MODEL_FILE = j(DATA_DIR, "{data}", "derived", "simulated_networks", "model_model~LTCM.npz")
REPRO_TEST_BASELINE_GEOMETRIC_MODEL_FILE = j(DATA_DIR, "{data}", "derived", "model~{model}_sample~{sample}_"+f"{canonical_spherical_model_paramspace.wildcard_pattern}.pt")

## cLTCM
REPRO_TEST_CLTCM_FILE = j(DATA_DIR, "{data}", "derived", f"model~cLTCM.pt")
#REPRO_TEST_CLTCM_NET_FILE = j(DATA_DIR, "{data}", "derived", "simulated_networks", f"net_model~cLTCM"+"_sample~{sample}.npz")
#REPRO_TEST_CLTCM_NODE_FILE =j(DATA_DIR, "{data}", "derived", "simulated_networks", f"node_model~cLTCM"+"_sample~{sample}.csv")


# Plot data
## Spherical model
REPRO_TEST_GEO_SIM_DATA_DEGREE_DIST = j(DATA_DIR, "{data}", "plot_data", "simulated_networks", f"degree-dist_{spherical_model_paramspace.wildcard_pattern}"+"_sample~{sample}.csv")
REPRO_TEST_GEO_SIM_DATA_EVENT_INTERVAL = j(DATA_DIR, "{data}", "plot_data", "simulated_networks", f"citation-event-interval_{spherical_model_paramspace.wildcard_pattern}"+"_sample~{sample}.csv")
REPRO_TEST_GEO_SIM_DATA_CITATION_RATE = j(DATA_DIR, "{data}", "plot_data", "simulated_networks", f"citation-rate_{spherical_model_paramspace.wildcard_pattern}"+"_sample~{sample}.csv")
REPRO_TEST_GEO_SIM_DATA_NEW_VS_ACCUM_CIT  = j(DATA_DIR, "{data}", "plot_data", "simulated_networks", f"new_vs_accumulated_citation_{spherical_model_paramspace.wildcard_pattern}"+"_sample~{sample}.csv")
REPRO_TEST_GEO_SIM_FITTED_POWER_LAW_PARAMS = j(DATA_DIR, "{data}", "plot_data", "simulated_networks", f"fitted-power-law-params_{spherical_model_paramspace.wildcard_pattern}"+"_sample~{sample}.json")
REPRO_TEST_GEO_SIM_SB_COEF_TABLE = j(DATA_DIR, "{data}", "plot_data", "simulated_networks", f"sb_coefficient_table_{spherical_model_paramspace.wildcard_pattern}"+"_sample~{sample}.csv")


## Baselines
BASELINE_MODEL_LIST = ["PA", "LTCM"]
#BASELINE_MODEL_LIST = ["PA", "LTCM", "cLTCM"]
#BASELINE_MODEL_LIST = ["PA", "LTCM", "bLTCM"]
REPRO_TEST_BASELINE_SIM_DATA_DEGREE_DIST = j(DATA_DIR, "{data}", "plot_data", "simulated_networks", "degree-dist_model~{model}_sample~{sample}.csv")
REPRO_TEST_BASELINE_SIM_DATA_EVENT_INTERVAL = j(DATA_DIR, "{data}", "plot_data", "simulated_networks", "citation-event-interval_model~{model}_sample~{sample}.csv")
REPRO_TEST_BASELINE_SIM_DATA_CITATION_RATE = j(DATA_DIR, "{data}", "plot_data", "simulated_networks", "citation-rate_model~{model}_sample~{sample}.csv")
REPRO_TEST_BASELINE_SIM_DATA_NEW_VS_ACCUM_CIT  = j(DATA_DIR, "{data}", "plot_data", "simulated_networks", "new_vs_accumulated_citation_model~{model}_sample~{sample}.csv")
REPRO_TEST_BASELINE_SIM_FITTED_POWER_LAW_PARAMS = j(DATA_DIR, "{data}", "plot_data", "simulated_networks", "fitted-power-law-params_model~{model}_sample~{sample}.json")
REPRO_TEST_BASELINE_SIM_SB_COEF_TABLE = j(DATA_DIR, "{data}", "plot_data", "simulated_networks", "sb_coefficient_table_model~{model}_sample~{sample}.csv")

## Citation radii analysis
REPRO_CITATION_RADII_FILE = j(DATA_DIR, "{data}", "plot_data", "citation_radii", f"{spherical_model_paramspace.wildcard_pattern}.csv")
REPRO_BASELINE_CITATION_RADII_FILE = j(DATA_DIR, "{data}", "plot_data", "citation_radii", "model~{model}_sample~{sample}_"+f"{canonical_spherical_model_paramspace.wildcard_pattern}.csv")

## Keyword prediction
KEYWORD_PRED_DIR = j(DATA_DIR, "{data}", "derived", "keyword_prediction")
KEYWORD_PRED_DATASET_FILE = j(KEYWORD_PRED_DIR, "dataset_categoryClass~{categoryClass}.pickle")
KEYWORD_PRED_SCORE_FILE = j(KEYWORD_PRED_DIR,"score_categoryClass~{categoryClass}_"+f"{spherical_model_paramspace.wildcard_pattern}.csv")
KEYWORD_PRED_SCORE_BY_CITATION_FILE = j(KEYWORD_PRED_DIR, "score_categoryClass~{categoryClass}_geometry~citation.csv")


# Figures ----------------
FIG_REPRO_TEST_DIR = j(FIG_DIR, "reproduceability_test", "{data}")
FIG_REPRO_TEST_CITATION_RATE = j(FIG_REPRO_TEST_DIR, f"citationRate_{spherical_model_paramspace.wildcard_pattern}.pdf")
FIG_REPRO_TEST_EVENT_INTERVAL = j(FIG_REPRO_TEST_DIR, f"eventInterval_{spherical_model_paramspace.wildcard_pattern}.pdf")
FIG_REPRO_TEST_EVENT_INTERVAL_UNNORMALIZED = j(FIG_REPRO_TEST_DIR, f"eventInterval_unnormalized_{spherical_model_paramspace.wildcard_pattern}.pdf")
FIG_REPRO_TEST_NEW_VS_ACCUM_CIT = j(FIG_REPRO_TEST_DIR, f"preferentialAttachment_{spherical_model_paramspace.wildcard_pattern}.pdf")
FIG_REPRO_TEST_NEW_VS_ACCUM_CIT_ALL_RANGE = j(FIG_REPRO_TEST_DIR, f"preferentialAttachment_all_range_{spherical_model_paramspace.wildcard_pattern}.pdf")
FIG_GEO_SIM_DATA_SB_COEFFICIENT= j(FIG_REPRO_TEST_DIR, f"SleepingBeautyCoefficient_{spherical_model_paramspace.wildcard_pattern}.pdf")
FIG_EMB_SPACE = j(FIG_DIR, "{data}", "embedding", f"embedding-{spherical_model_paramspace.wildcard_pattern}.pdf")

FIG_CITATION_RADII= j(FIG_DIR, "{data}", "embedding-stats", f"citation-radii-{canonical_spherical_model_paramspace.wildcard_pattern}-"+"model~{model}_sample~{sample}.pdf")
FIG_KEYWORD_PRED_SCORE_FILE = j(FIG_DIR, "{data}", "embedding-stats", "keywordPrediction_categoryClass~{categoryClass}_"+f"{canonical_spherical_model_paramspace.wildcard_pattern}.pdf")

rule _all:
    input:
        expand(REPRO_TEST_UMAP_FILE, **params_spherical_model, data = DATA_LIST ),
        #expand(REPRO_TEST_UMAP_XNET_FILE, **params_spherical_model, data = DATA_LIST )

rule reproduceability_test_all:
    input:
        #
        # Geomtric model
        #
        expand(REPRO_TEST_GEO_SIM_NET_FILE, **params_spherical_model, data = DATA_LIST, sample = list(range(N_SIM_NET_SAMPLES))),
        expand(REPRO_TEST_GEO_SIM_NODE_FILE, **params_spherical_model, data = DATA_LIST, sample = list(range(N_SIM_NET_SAMPLES))),
        expand(REPRO_TEST_GEO_SIM_SB_COEF_TABLE, data = DATA_LIST, **params_spherical_model, sample = list(range(N_SIM_NET_SAMPLES))),
        expand(REPRO_TEST_GEO_SIM_DATA_DEGREE_DIST, data = DATA_LIST, **params_spherical_model, sample = list(range(N_SIM_NET_SAMPLES))),
        expand(REPRO_TEST_GEO_SIM_FITTED_POWER_LAW_PARAMS, data = DATA_LIST, **params_spherical_model, sample = list(range(N_SIM_NET_SAMPLES))),
        expand(REPRO_TEST_GEO_SIM_DATA_NEW_VS_ACCUM_CIT, data = DATA_LIST, **params_spherical_model, sample = list(range(N_SIM_NET_SAMPLES))),
        #
        # Baselines
        #
        expand(REPRO_TEST_BASELINE_GEOMETRIC_MODEL_FILE, data = DATA_LIST, model =  "PA", sample = [0], **params_canonical_spherical_model),
        expand(REPRO_TEST_BASELINE_SIM_NET_FILE, data = DATA_LIST, model= BASELINE_MODEL_LIST, sample = list(range(N_SIM_NET_SAMPLES))),
        expand(REPRO_TEST_BASELINE_SIM_DATA_DEGREE_DIST, data = DATA_LIST, model= BASELINE_MODEL_LIST, sample = list(range(N_SIM_NET_SAMPLES))),
        expand(REPRO_TEST_BASELINE_SIM_DATA_EVENT_INTERVAL, data = DATA_LIST, model= BASELINE_MODEL_LIST, sample = list(range(N_SIM_NET_SAMPLES))),
        expand(REPRO_TEST_BASELINE_SIM_DATA_NEW_VS_ACCUM_CIT, data = DATA_LIST, model= BASELINE_MODEL_LIST, sample = list(range(N_SIM_NET_SAMPLES))),
        expand(REPRO_TEST_BASELINE_SIM_FITTED_POWER_LAW_PARAMS, data = DATA_LIST, model= BASELINE_MODEL_LIST, sample = list(range(N_SIM_NET_SAMPLES))),
        expand(REPRO_TEST_BASELINE_SIM_SB_COEF_TABLE, data = DATA_LIST, model= BASELINE_MODEL_LIST, sample = list(range(N_SIM_NET_SAMPLES))),
        #expand(REPRO_TEST_UMAP_FILE, **params_spherical_model, data = DATA_LIST ),
        #
        # Stats of embedding
        #
        #expand(REPRO_CITATION_RADII_FILE, data=DATA_LIST, **params_canonical_spherical_model),
        expand(KEYWORD_PRED_SCORE_FILE, data=DATA_LIST, **params_canonical_spherical_model, categoryClass = ["main", "sub"]),
        expand(KEYWORD_PRED_SCORE_BY_CITATION_FILE, data=DATA_LIST, categoryClass = ["main", "sub"] ),
        expand(REPRO_BASELINE_CITATION_RADII_FILE, data=DATA_LIST, **params_canonical_spherical_model, model=["PA"], sample = [0]),

rule reproduceability_test_figs:
    input:
        expand(FIG_REPRO_TEST_CITATION_RATE, data = DATA_LIST, **params_spherical_model),
        expand(FIG_REPRO_TEST_EVENT_INTERVAL, data = DATA_LIST, **params_spherical_model),
        expand(FIG_REPRO_TEST_NEW_VS_ACCUM_CIT, data = DATA_LIST, **params_spherical_model),
        expand(FIG_GEO_SIM_DATA_SB_COEFFICIENT, data = DATA_LIST, **params_spherical_model),
        #expand(FIG_EMB_SPACE, data = DATA_LIST, **params_spherical_model, sample = list(range(N_SIM_NET_SAMPLES))),
        #expand(FIG_CITATION_RADII, data = DATA_LIST, **params_canonical_spherical_model, model=["PA"], sample = [0]),
        expand(FIG_KEYWORD_PRED_SCORE_FILE, data = DATA_LIST, **params_canonical_spherical_model, categoryClass=["main", "sub"] )

rule test:
    input:
        expand(FIG_EMB_SPACE, data = DATA_LIST, **params_spherical_model, sample = list(range(N_SIM_NET_SAMPLES))),

# ========================================================
# Model fitting
# ========================================================

rule model_fitting_spherical_model:
    input:
        paper_table_file = PAPER_TABLE,
        net_file = CITATION_NET,
    output:
        output_file = REPRO_TEST_GEOMETRIC_MODEL_FILE
    params:
        dim = lambda wildcards : wildcards.dim,
        geometry = lambda wildcards : wildcards.geometry,
        aging = lambda wildcards : wildcards.aging,
        symmetric = lambda wildcards : wildcards.symmetric,
        fitness = lambda wildcards : wildcards.fitness,
        c0 = lambda wildcards : wildcards.c0,
    script:
        "workflow/fit-spherical-model/fitting.py"


rule model_fitting_spherical_model_to_baseline_networks:
    input:
        paper_table_file = PAPER_TABLE,
        net_file = REPRO_TEST_BASELINE_SIM_NET_FILE,
    output:
        output_file = REPRO_TEST_BASELINE_GEOMETRIC_MODEL_FILE
    params:
        dim = lambda wildcards : wildcards.dim,
        geometry = lambda wildcards : wildcards.geometry,
        aging = lambda wildcards : wildcards.aging,
        symmetric = lambda wildcards : wildcards.symmetric,
        fitness = lambda wildcards : wildcards.fitness,
        c0 = lambda wildcards : wildcards.c0,
    script:
        "workflow/fit-spherical-model/fitting.py"

rule model_fitting_cltcm:
    input:
        paper_table_file = PAPER_TABLE,
        net_file = CITATION_NET,
    output:
        output_file = REPRO_TEST_CLTCM_FILE
    params:
        c0 = 10
    script:
        "workflow/fit-spherical-model/fitting_ltcm.py"

# ========================================================
# Network generation
# ========================================================

rule fit_long_term_citations:
    input:
        paper_table_file = PAPER_TABLE,
        net_file = CITATION_NET,
    output:
        output_file = REPRO_TEST_BASELINE_LTCM_MODEL_FILE,
    script:
        "workflow/fit-spherical-model/fit-long-term-citation.py"

rule generate_networks_long_term_citations:
    input:
        input_file = REPRO_TEST_BASELINE_LTCM_MODEL_FILE,
    output:
        output_net_file = REPRO_TEST_BASELINE_SIM_NET_FILE,
    wildcard_constraints:
        model="LTCM" # long term citation model
    script:
        "workflow/fit-spherical-model/generate-networks-long-term-citation.py"

rule generate_networks_beysian_long_term_citations:
    input:
        paper_table_file = PAPER_TABLE,
        net_file = CITATION_NET,
    output:
        output_net_file = REPRO_TEST_BASELINE_SIM_NET_FILE,
    wildcard_constraints:
        model="bLTCM" # long term citation model
    script:
        "workflow/fit-spherical-model/generate-networks-bayesian-long-term-citation.py"

rule generate_networks_cltcm:
    input:
        paper_table_file = PAPER_TABLE,
        net_file = CITATION_NET,
        model_file = REPRO_TEST_CLTCM_FILE,
    output:
        output_net_file = REPRO_TEST_BASELINE_SIM_NET_FILE
        #output_net_file = REPRO_TEST_CLTCM_NET_FILE,
    wildcard_constraints:
        model="cLTCM" # long term citation model
    script:
        "workflow/fit-spherical-model/generate-networks-cltcm.py"

rule generate_networks_pa:
    input:
        paper_table_file = PAPER_TABLE,
        net_file = CITATION_NET,
    output:
        output_net_file = REPRO_TEST_BASELINE_SIM_NET_FILE,
    wildcard_constraints:
        model="PA" # long term citation model
    script:
        "workflow/fit-spherical-model/generate-networks-preferential-attachment.py"


rule generate_network_with_geometric_model:
    input:
        paper_table_file = PAPER_TABLE,
        net_file = CITATION_NET,
        model_file = REPRO_TEST_GEOMETRIC_MODEL_FILE,
    output:
        output_file = REPRO_TEST_GEO_SIM_NET_FILE,
        output_node_file = REPRO_TEST_GEO_SIM_NODE_FILE
    params:
        dim = lambda wildcards : wildcards.dim,
        geometry = lambda wildcards : wildcards.geometry,
        aging = lambda wildcards : wildcards.aging,
        fitness = lambda wildcards : wildcards.fitness,
        symmetric = lambda wildcards : wildcards.symmetric
    script:
        "workflow/fit-spherical-model/generate-networks-with-geometric-model.py"


# ========================================================
# Analyzing embedding
# ========================================================


rule pickle_fitting_results:
    input:
        paper_table_file = PAPER_TABLE,
        model_file = REPRO_TEST_GEOMETRIC_MODEL_FILE,
    output:
        output_file = REPRO_TEST_UMAP_FILE
    params:
        dim = lambda wildcards : wildcards.dim,
        symmetric = lambda wildcards : wildcards.symmetric,
    script:
        "workflow/fit-spherical-model/pickle_fitting_results.py"



rule citation_radii_analysis:
    input:
        paper_table_file = PAPER_TABLE,
        net_file = CITATION_NET,
        model_file = REPRO_TEST_GEOMETRIC_MODEL_FILE,
    output:
        output_file = REPRO_CITATION_RADII_FILE
    params:
        dim = lambda wildcards : wildcards.dim,
        symmetric = lambda wildcards : wildcards.symmetric
    script:
        "workflow/stats/citation_radii.py"

rule citation_radii_analysis_baseline:
    input:
        paper_table_file = PAPER_TABLE,
        net_file = CITATION_NET,
        model_file = REPRO_TEST_BASELINE_GEOMETRIC_MODEL_FILE
    output:
        output_file = REPRO_BASELINE_CITATION_RADII_FILE
    params:
        dim = lambda wildcards : wildcards.dim,
        symmetric = lambda wildcards : wildcards.symmetric
    script:
        "workflow/stats/citation_radii.py"

#rule generate_benchmark_dataset:
#    input:
#        citation_net_file=CITATION_NET,
#        paper_category_table_file = PAPER_CATEGORY_TABLE,
#    params:
#        categoryClass = lambda wildcards: wildcards.categoryClass,
#        min_keyword_freq=100,
#        n_splits=5,
#        max_n_samples=1000000,
#    output:
#        output_file=KEYWORD_PRED_DATASET_FILE,
#    script:
#        "workflow/fit-spherical-model/generate-benchmark-dataset.py"
#
#rule run_benchmark_with_paper_embedding:
#    input:
#        benchmark_data_file=KEYWORD_PRED_DATASET_FILE,
#        model_file=REPRO_TEST_GEOMETRIC_MODEL_FILE,
#        paper_table_file = PAPER_TABLE
#    params:
#        dim = lambda wildcards : wildcards.dim,
#    output:
#        output_file=KEYWORD_PRED_SCORE_FILE,
#    resources:
#        gpu=1,
#    script:
#        "workflow/fit-spherical-model/benchmark-keyword-prediction-by-knn.py"
#
#rule run_benchmark_with_citation:
#    input:
#        net_file=CITATION_NET,
#        benchmark_data_file=KEYWORD_PRED_DATASET_FILE,
#    output:
#        output_file=KEYWORD_PRED_SCORE_BY_CITATION_FILE,
#    script:
#        "workflow/fit-spherical-model/benchmark-keyword-prediction-by-citation.py"
#
#rule plot_benchmark_result:
#    input:
#        score_file = KEYWORD_PRED_SCORE_FILE,
#        baseline_score_file = KEYWORD_PRED_SCORE_BY_CITATION_FILE
#    output:
#        output_file=FIG_KEYWORD_PRED_SCORE_FILE,
#    params:
#        eval_metric="microf1",
#    script:
#        "workflow/plot/plot-paper-keyword-pred-result.py"

# ========================================================
# Calc net stats of the simulated net
# ========================================================

#
# For geometric models
#
use rule calc_degree_distribution as calc_degree_distribution_sim_net with:
    input:
        net_file = REPRO_TEST_GEO_SIM_NET_FILE,
        paper_table_file = REPRO_TEST_GEO_SIM_NODE_FILE,
    params:
        groupby = None
    output:
        output_file = REPRO_TEST_GEO_SIM_DATA_DEGREE_DIST

use rule calc_citation_event_interval as calc_citation_event_interval_sim_net with:
    input:
        net_file = REPRO_TEST_GEO_SIM_NET_FILE,
        paper_table_file = REPRO_TEST_GEO_SIM_NODE_FILE,
    params:
        groupby = None,
        focal_period = [2000, 2010],
        dataName = "Spherical",
    output:
        output_file = REPRO_TEST_GEO_SIM_DATA_EVENT_INTERVAL

use rule calc_citation_rate as calc_citation_rate_repro_test_geo with:
    input:
        net_file = REPRO_TEST_GEO_SIM_NET_FILE,
        paper_table_file = REPRO_TEST_GEO_SIM_NODE_FILE,
    params:
        dataName = "Spherical",
        focal_period = [1960, 2000],
    output:
        output_file = REPRO_TEST_GEO_SIM_DATA_CITATION_RATE

use rule calc_powerlaw_deg_dist_params as calc_powerlaw_deg_dist_params_sim_net with:
    input:
        net_file = REPRO_TEST_GEO_SIM_NET_FILE,
        paper_table_file = REPRO_TEST_GEO_SIM_NODE_FILE,
    params:
        groupby = None,
    output:
        output_file = REPRO_TEST_GEO_SIM_FITTED_POWER_LAW_PARAMS


use rule calc_pref_rate as calc_pref_rate_sim_net with:
    input:
        net_file = REPRO_TEST_GEO_SIM_NET_FILE,
        paper_table_file = REPRO_TEST_GEO_SIM_NODE_FILE,
    params:
        groupby = None,
        dataName = "Spherical",
    output:
        output_file = REPRO_TEST_GEO_SIM_DATA_NEW_VS_ACCUM_CIT

use rule calc_sleeping_beauty_coefficient as calc_sleeping_beauty_coefficient_sim_net with:
    input:
        net_file = REPRO_TEST_GEO_SIM_NET_FILE,
        paper_table_file = REPRO_TEST_GEO_SIM_NODE_FILE,
    params:
        dataName = "Spherical",
    output:
        output_file = REPRO_TEST_GEO_SIM_SB_COEF_TABLE
#
# For baseline models
#
use rule calc_citation_rate as calc_citation_rate_repro_test_baseline with:
    input:
        net_file = REPRO_TEST_BASELINE_SIM_NET_FILE,
        paper_table_file = PAPER_TABLE,
    params:
        dataName = lambda wildcards: wildcards.model,
        focal_period = [1960, 2000],
    output:
        output_file = REPRO_TEST_BASELINE_SIM_DATA_CITATION_RATE

use rule calc_degree_distribution_sim_net as calc_degree_distribution_sim_net_baseline with:
    input:
        net_file = REPRO_TEST_BASELINE_SIM_NET_FILE,
        paper_table_file = PAPER_TABLE,
    output:
        output_file = REPRO_TEST_BASELINE_SIM_DATA_DEGREE_DIST

use rule calc_citation_event_interval_sim_net as calc_citation_event_interval_sim_net_baseline with:
    input:
        net_file = REPRO_TEST_BASELINE_SIM_NET_FILE,
        paper_table_file = PAPER_TABLE,
    params:
        dataName = lambda wildcards: wildcards.model
    output:
        output_file = REPRO_TEST_BASELINE_SIM_DATA_EVENT_INTERVAL


use rule calc_powerlaw_deg_dist_params_sim_net as calc_powerlaw_deg_dist_params_sim_net_baseline with:
    input:
        net_file = REPRO_TEST_BASELINE_SIM_NET_FILE,
        paper_table_file = PAPER_TABLE,
    params:
        groupby = None,
    output:
        output_file = REPRO_TEST_BASELINE_SIM_FITTED_POWER_LAW_PARAMS

use rule calc_pref_rate_sim_net as calc_pref_rate_sim_net_baseline with:
    input:
        net_file = REPRO_TEST_BASELINE_SIM_NET_FILE,
        paper_table_file = PAPER_TABLE,
    params:
        dataName = lambda wildcards: wildcards.model,
        groupby = None,
    output:
        output_file = REPRO_TEST_BASELINE_SIM_DATA_NEW_VS_ACCUM_CIT

use rule calc_sleeping_beauty_coefficient_sim_net as calc_sleeping_beauty_coefficient_sim_net_baseline with:
    input:
        net_file = REPRO_TEST_BASELINE_SIM_NET_FILE,
        paper_table_file = PAPER_TABLE,
    params:
        dataName = lambda wildcards: wildcards.model,
    output:
        output_file = REPRO_TEST_BASELINE_SIM_SB_COEF_TABLE

# Plot stats of the sim networks
#
#rule plot_degree_dist_sim_net:
#    input:
#        input_file = REPRO_TEST_GEO_SIM_FITTED_POWER_LAW_PARAMS
#    output:
#        output_deg_file = FIG_GEO_SIM_DATA_DEGREE_DIST,
#    params:
#        data = lambda wildcards : wildcards.data,
#        title = lambda wildcards: " | ".join([f"{k}~{v}" for k, v in wildcards.items() if k not in ["geometry", "sample"]])
#    script:
#        "workflow/plot/plot-degree-distribution.py"
#
rule plot_citation_rate_sim_net:
    input:
        input_file = expand(REPRO_TEST_GEO_SIM_DATA_CITATION_RATE, sample = list(range(N_SIM_NET_SAMPLES))),
        empirical_baseline_file = DATA_CITATION_RATE,
        model_baseline_files = expand(REPRO_TEST_BASELINE_SIM_DATA_CITATION_RATE, model=BASELINE_MODEL_LIST, sample = list(range(N_SIM_NET_SAMPLES)))
    output:
        output_file = FIG_REPRO_TEST_CITATION_RATE
    params:
        data = lambda wildcards : wildcards.data,
        title = lambda wildcards: " | ".join([f"{k}~{v}" for k, v in wildcards.items() if k not in ["geometry", "sample"]]),
        focal_degree = 25,
    script:
        "workflow/plot/plot-citation-rate-empirical-vs-models.py"

rule plot_citation_event_interval_sim_net:
    input:
        input_file = expand(REPRO_TEST_GEO_SIM_DATA_EVENT_INTERVAL, sample = list(range(N_SIM_NET_SAMPLES))),
        empirical_baseline_file = DATA_EVENT_INTERVAL,
        model_baseline_files = expand(REPRO_TEST_BASELINE_SIM_DATA_EVENT_INTERVAL, model=BASELINE_MODEL_LIST, sample = list(range(N_SIM_NET_SAMPLES)))
    output:
        output_file = FIG_REPRO_TEST_EVENT_INTERVAL
    params:
        data = lambda wildcards : wildcards.data,
        title = lambda wildcards: " | ".join([f"{k}~{v}" for k, v in wildcards.items() if k not in ["geometry", "sample"]]),
        focal_degree = 25,
        normalize_citation_count = True
    script:
        "workflow/plot/plot-event-interval-empirical-vs-models.py"


rule plot_citation_event_interval_sim_net_unnormalized:
    input:
        input_file = expand(REPRO_TEST_GEO_SIM_DATA_EVENT_INTERVAL, sample = list(range(N_SIM_NET_SAMPLES))),
        empirical_baseline_file = DATA_EVENT_INTERVAL,
        model_baseline_files = expand(REPRO_TEST_BASELINE_SIM_DATA_EVENT_INTERVAL, model=BASELINE_MODEL_LIST, sample = list(range(N_SIM_NET_SAMPLES)))
    output:
        output_file = FIG_REPRO_TEST_EVENT_INTERVAL_UNNORMALIZED
    params:
        data = lambda wildcards : wildcards.data,
        title = lambda wildcards: " | ".join([f"{k}~{v}" for k, v in wildcards.items() if k not in ["geometry", "sample"]]),
        focal_degree = 25,
        normalize_citation_count = False
    script:
        "workflow/plot/plot-event-interval-empirical-vs-models.py"

rule plot_pref_attachment_sim_net:
    input:
        input_file = [DATA_NEW_VS_ACCUM_CIT] + expand(REPRO_TEST_GEO_SIM_DATA_NEW_VS_ACCUM_CIT, sample = list(range(N_SIM_NET_SAMPLES)))+ expand(REPRO_TEST_BASELINE_SIM_DATA_NEW_VS_ACCUM_CIT, model=BASELINE_MODEL_LIST, sample = list(range(N_SIM_NET_SAMPLES))),
    output:
        output_file =  FIG_REPRO_TEST_NEW_VS_ACCUM_CIT
    params:
        data = lambda wildcards : wildcards.data,
        title = lambda wildcards: " | ".join([f"{k}~{v}" for k, v in wildcards.items() if k not in ["geometry", "sample"]]),
        min_x = 30
    script:
        "workflow/plot/plot-new-vs-accumulated-citations.py"

rule plot_pref_attachment_sim_net_all_range:
    input:
        input_file = [DATA_NEW_VS_ACCUM_CIT] + expand(REPRO_TEST_GEO_SIM_DATA_NEW_VS_ACCUM_CIT, sample = list(range(N_SIM_NET_SAMPLES)))+ expand(REPRO_TEST_BASELINE_SIM_DATA_NEW_VS_ACCUM_CIT, model=BASELINE_MODEL_LIST, sample = list(range(N_SIM_NET_SAMPLES))),
    output:
        output_file =  FIG_REPRO_TEST_NEW_VS_ACCUM_CIT_ALL_RANGE
    params:
        data = lambda wildcards : wildcards.data,
        title = lambda wildcards: " | ".join([f"{k}~{v}" for k, v in wildcards.items() if k not in ["geometry", "sample"]]),
        min_x = 1
    script:
        "workflow/plot/plot-new-vs-accumulated-citations.py"
#
rule plot_sb_coef_dist_sim_net:
    input:
        input_file = [DATA_SB_COEF_TABLE] + expand(REPRO_TEST_GEO_SIM_SB_COEF_TABLE, sample = list(range(N_SIM_NET_SAMPLES)))+ expand(REPRO_TEST_BASELINE_SIM_SB_COEF_TABLE, model=BASELINE_MODEL_LIST, sample = list(range(N_SIM_NET_SAMPLES)))
        #sim_sb_coef_files = _expand(REPRO_TEST_GEO_SIM_SB_COEF_TABLE, sample = list(range(N_SIM_NET_SAMPLES))),
        #random_sb_coef_dir = RAND_SB_COEF_TABLE_DIR,
        #paper_table_file = GEO_SIM_NODE_FILE
    params:
        data = lambda wildcards : wildcards.data,
        offset_SB = 13,
        title = lambda wildcards: " | ".join([f"{k}~{v}" for k, v in wildcards.items() if k not in ["geometry", "sample"]])
    output:
        output_file = FIG_GEO_SIM_DATA_SB_COEFFICIENT
    script:
        "workflow/plot/plot-sb-coef-distribution.py"

#
# Spherical model
#
rule plot_embedding:
    input:
        umap_file = REPRO_TEST_UMAP_FILE,
        category_table_file = CATEGORY_TABLE,
        paper_category_table_file = PAPER_CATEGORY_TABLE,
    output:
        output_file = FIG_EMB_SPACE
    script:
        "workflow/plot/plot_embedding.py"

#
# Embedding stats
#
rule plot_citation_radii:
    input:
        data_table_file = REPRO_CITATION_RADII_FILE,
        #baseline_data_table_file =REPRO_CITATION_RADII_FILE
        baseline_data_table_file =REPRO_BASELINE_CITATION_RADII_FILE
    params:
        data = lambda wildcards : wildcards.data,
    output:
        output_file = FIG_CITATION_RADII
    script:
        "workflow/plot/plot-citation-radii.py"


#
#rule plot_awakening_time_dist_sim_net:
#    input:
#        sb_coef_file = SB_COEF_TABLE,
#        sim_sb_coef_files = _expand(REPRO_TEST_GEO_SIM_SB_COEF_TABLE, sample = list(range(N_SIM_NET_SAMPLES))),
#        random_sb_coef_dir = RAND_SB_COEF_TABLE_DIR,
#        #paper_table_file = GEO_SIM_NODE_FILE
#    params:
#        data = lambda wildcards : wildcards.data,
#        offset_awakening_time = 1,
#        title = lambda wildcards: " | ".join([f"{k}~{v}" for k, v in wildcards.items() if k not in ["geometry", "sample"]])
#    output:
#        output_file = FIG_GEO_SIM_DATA_AWAKENING_TIME
#    script:
#        "workflow/plot/plot-awakenning-time-distribution.py"
#

rule plot_sb_dist:
    input:
        input_file = [DATA_SB_COEF_TABLE] + expand(REPRO_TEST_BASELINE_SIM_SB_COEF_TABLE, model=BASELINE_MODEL_LIST, sample = list(range(N_SIM_NET_SAMPLES)))
    output:
        output_file =  FIG_DATA_SB_COEFFICIENT
    params:
        data = lambda wildcards : wildcards.data,
        offset_SB = 13,
    script:
        "workflow/plot/plot-sb-coef-distribution.py"
