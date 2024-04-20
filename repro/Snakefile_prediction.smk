# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-12-27 14:19:48
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-09-15 15:57:26
#
#
# Prediction
#
# Geometric model
citation_prediction_paramspace = to_paramspace(params_citation_prediction)
PRED_TEST_TRAIN_NET_FILE = j(DATA_DIR, "{data}", "derived", "prediction", "train_net-{t_train}.npz")
PRED_TEST_TRAIN_NODE_FILE = j(DATA_DIR, "{data}", "derived", "prediction", "train_node-{t_train}.csv")
PRED_TEST_TRAIN_GEOMETRIC_MODEL_FILE = j(DATA_DIR, "{data}", "derived", "prediction", f"model_{citation_prediction_paramspace.wildcard_pattern}.pt")
PRED_TEST_PREFERENTIAL_PRODUCTION_MODEL = j(DATA_DIR, "{data}", "derived", "prediction", f"model_preferential_production_{citation_prediction_paramspace.wildcard_pattern}.pt")
PRED_TEST_GEO_SIM_NET_FILE = j(DATA_DIR, "{data}", "derived", "prediction", "simulated_networks", f"net_{citation_prediction_paramspace.wildcard_pattern}"+"_sample~{sample}.npz")
PRED_TEST_GEO_SIM_EMB_FILE = j(DATA_DIR, "{data}", "derived", "prediction", "simulated_networks", f"emb_{citation_prediction_paramspace.wildcard_pattern}"+"_sample~{sample}.npz")

# Baseline
baseline_citation_prediction_paramspace = to_paramspace(params_baseline_citation_prediction)
PRED_TEST_PREDICTED_BASELINE_NET_FILE = j(DATA_DIR, "{data}", "derived", "prediction", "simulated_networks", f"net_{baseline_citation_prediction_paramspace.wildcard_pattern}"+"_sample~{sample}.npz")# baseline
PRED_TEST_PREDICTED_LTCM_MODEL_FILE = j(DATA_DIR, "{data}", "derived", "prediction", "simulated_networks", f"model_{baseline_citation_prediction_paramspace.wildcard_pattern}.npz")# baseline

# CLTCM
PRED_TEST_TRAIN_CLTCM_FILE = j(DATA_DIR, "{data}", "derived", "prediction", "model~cLTCM_t_train~{t_train}.pt")


# Evaluation
PRED_TEST_EVAL_FILE = j(DATA_DIR, "{data}", "derived", "prediction", "simulated_networks", f"results_{citation_prediction_paramspace.wildcard_pattern}.csv")# baseline
PRED_TEST_EVAL_BASELINE_FILE = j(DATA_DIR, "{data}", "derived", "prediction", "simulated_networks", f"results_{baseline_citation_prediction_paramspace.wildcard_pattern}.csv")# baseline

# Prediction
FIG_PRED =j(FIG_DIR, "prediction", "{data}", f"pred_result_{citation_prediction_paramspace.wildcard_pattern}_yscale="+"{yscale}.pdf")

N_SIM_NET_SAMPLES = 5

rule prediction_all:
    input:
        expand(PRED_TEST_EVAL_FILE, data = DATA_LIST, **params_citation_prediction),
        expand(PRED_TEST_EVAL_BASELINE_FILE, data = DATA_LIST, **params_baseline_citation_prediction),


rule prediction_figs:
    input:
        expand(FIG_PRED, data = DATA_LIST, **params_citation_prediction, yscale = ["linear", "log"]),
        #"figs/prediction.pdf",

# ========================================================
# Prediction
# ========================================================

rule train_test_split:
    input:
        net_file = CITATION_NET,
        paper_table_file = PAPER_TABLE,
    output:
        train_net_file = PRED_TEST_TRAIN_NET_FILE,
        train_paper_table_file = PRED_TEST_TRAIN_NODE_FILE,
    params:
        t_train = lambda wildcards: int(wildcards.t_train)
    script:
        "workflow/prediction/train_test_split.py"

rule model_fitting_spherical_model_train_net:
    input:
        paper_table_file = PRED_TEST_TRAIN_NODE_FILE,
        net_file = PRED_TEST_TRAIN_NET_FILE,
    output:
        output_file = PRED_TEST_TRAIN_GEOMETRIC_MODEL_FILE
    params:
        dim = lambda wildcards : wildcards.dim,
        geometry = lambda wildcards : wildcards.geometry,
        aging = lambda wildcards : wildcards.aging,
        symmetric = lambda wildcards : wildcards.symmetric,
        fitness = lambda wildcards : wildcards.fitness,
        c0 = lambda wildcards : float(wildcards.c0),
    script:
        "workflow/fit-spherical-model/fitting.py"

rule model_fitting_cltcm_train_net:
    input:
        paper_table_file = PRED_TEST_TRAIN_NODE_FILE,
        net_file = PRED_TEST_TRAIN_NET_FILE,
    output:
        output_file = PRED_TEST_TRAIN_CLTCM_FILE
    params:
        c0 = 10,
    script:
        "workflow/fit-spherical-model/fitting_ltcm.py"

rule estimate_kappa_paper:
    input:
        model_file = PRED_TEST_TRAIN_GEOMETRIC_MODEL_FILE,
        paper_table_file = PRED_TEST_TRAIN_NODE_FILE,
        net_file = PRED_TEST_TRAIN_NET_FILE,
    output:
        output_file = PRED_TEST_PREFERENTIAL_PRODUCTION_MODEL,
    params:
        dim = lambda wildcards : wildcards.dim,
        geometry = lambda wildcards : wildcards.geometry,
        aging = lambda wildcards : wildcards.aging,
        symmetric = lambda wildcards : wildcards.symmetric,
        fitness = lambda wildcards : wildcards.fitness,
    script:
        "workflow/fit-spherical-model/fit-paper-production-model.py"

rule simulate_networks_prediction:
    input:
        pref_prod_model = PRED_TEST_PREFERENTIAL_PRODUCTION_MODEL,
        model_file = PRED_TEST_TRAIN_GEOMETRIC_MODEL_FILE,
        paper_table_file = PAPER_TABLE,
        net_file = CITATION_NET,
        train_paper_table_file = PRED_TEST_TRAIN_NODE_FILE,
        train_net_file = PRED_TEST_TRAIN_NET_FILE,
    output:
        pred_net_file = PRED_TEST_GEO_SIM_NET_FILE,
        pred_emb_file = PRED_TEST_GEO_SIM_EMB_FILE
    params:
        dim = lambda wildcards : wildcards.dim,
        geometry = lambda wildcards : wildcards.geometry,
        aging = lambda wildcards : wildcards.aging,
        symmetric = lambda wildcards : wildcards.symmetric,
        fitness = lambda wildcards : wildcards.fitness,
        t_train = lambda wildcards : wildcards.t_train,
    script:
        "workflow/prediction/generate-predictions.py"

rule simulate_networks_cltcm_prediction:
    input:
        model_file = PRED_TEST_TRAIN_CLTCM_FILE,
        paper_table_file = PAPER_TABLE,
        net_file = CITATION_NET,
        train_paper_table_file = PRED_TEST_TRAIN_NODE_FILE,
        train_net_file = PRED_TEST_TRAIN_NET_FILE,
    output:
        pred_net_file = PRED_TEST_PREDICTED_BASELINE_NET_FILE,
    params:
        t_train = lambda wildcards : wildcards.t_train,
    wildcard_constraints:
        model="cLTCM" # long term citation model
    script:
        "workflow/prediction/generate-predictions-cltcm.py"


rule simulate_networks_prediction_preferential_attachment:
    input:
        paper_table_file = PAPER_TABLE,
        net_file = CITATION_NET,
    output:
        pred_net_file = PRED_TEST_PREDICTED_BASELINE_NET_FILE,
    params:
        t_train = lambda wildcards : wildcards.t_train,
    wildcard_constraints:
        model="PA" # long term citation model
    script:
        "workflow/prediction/generate-predictions-preferential-attachment.py"

rule simulate_networks_fitting_prediction_long_term_citations:
    input:
        paper_table_file = PAPER_TABLE,
        net_file = CITATION_NET,
    output:
        output_file = PRED_TEST_PREDICTED_LTCM_MODEL_FILE,
    params:
        t_train = lambda wildcards : wildcards.t_train,
    wildcard_constraints:
        model="LTCM" # long term citation model
    script:
        "workflow/prediction/fit-predictions-long-term-citation.py"


rule simulate_networks_prediction_long_term_citations:
    input:
        input_file = PRED_TEST_PREDICTED_LTCM_MODEL_FILE,
        net_file = CITATION_NET,
    output:
        pred_net_file = PRED_TEST_PREDICTED_BASELINE_NET_FILE,
    params:
        t_train = lambda wildcards : wildcards.t_train,
    wildcard_constraints:
        model="LTCM" # long term citation model
    script:
        "workflow/prediction/generate-predictions-long-term-citation.py"

rule simulate_networks_prediction_bayesian_long_term_citations:
    input:
        paper_table_file = PAPER_TABLE,
        net_file = CITATION_NET,
    output:
        pred_net_file = PRED_TEST_PREDICTED_BASELINE_NET_FILE,
    params:
        t_train = lambda wildcards : wildcards.t_train,
    wildcard_constraints:
        model="bLTCM" # long term citation model
    script:
        "workflow/prediction/generate-predictions-bayesian-long-term-citation.py"

#rule simulate_networks_prediction_ccm_long_term_citations:
#    input:
#        paper_table_file = PAPER_TABLE,
#        net_file = CITATION_NET,
#    output:
#        pred_net_file = PRED_TEST_PREDICTED_BASELINE_NET_FILE,
#    params:
#        t_train = lambda wildcards : wildcards.t_train,
#    wildcard_constraints:
#        model="cLTCM" # long term citation model
#    script:
#        "workflow/prediction/generate-predictions-ccm-long-term-citation.py"

rule simulate_networks_prediction_zero_citations:
    input:
        paper_table_file = PAPER_TABLE,
        net_file = CITATION_NET,
    output:
        pred_net_file = PRED_TEST_PREDICTED_BASELINE_NET_FILE,
    params:
        t_train = lambda wildcards : wildcards.t_train,
    wildcard_constraints:
        model="ZeroCitation" # long term citation model
    script:
        "workflow/prediction/generate-predictions-zero-citations.py"


# cLTCM ==================================================
#rule simulate_networks_prediction_cLTCM:
#    input:
#        paper_table_file = PAPER_TABLE,
#        net_file = CITATION_NET,
#        train_paper_table_file = PRED_TEST_TRAIN_NODE_FILE,
#        train_net_file = PRED_TEST_TRAIN_NET_FILE,
#    output:
#        pred_net_file = PRED_TEST_PREDICTED_BASELINE_NET_FILE,
#    params:
#        dim = 1,
#        geometry = "False",
#        aging = "True",
#        symmetric = "True",
#        fitness = "True",
#        t_train = lambda wildcards : wildcards.t_train,
#    wildcard_constraints:
#        model="cLTCM"
#    script:
#        "workflow/prediction/generate-predictions-ccm-long-term-citation.py"


# ========================================================

rule eval_prediction:
    input:
        paper_table_file = PAPER_TABLE,
        net_file = CITATION_NET,
        pred_net_file_list = expand(PRED_TEST_GEO_SIM_NET_FILE, sample = list(range(N_SIM_NET_SAMPLES))),
    output:
        output_file = PRED_TEST_EVAL_FILE,
    params:
        mindeg = 15,
        model_name = "Spherical",
        t_train = lambda wildcards : wildcards.t_train,
    script:
        "workflow/prediction/evaluate-prediction.py"

rule eval_prediction_baseline:
    input:
        paper_table_file = PAPER_TABLE,
        net_file = CITATION_NET,
        pred_net_file_list = expand(PRED_TEST_PREDICTED_BASELINE_NET_FILE, sample = list(range(N_SIM_NET_SAMPLES))),
    output:
        output_file = PRED_TEST_EVAL_BASELINE_FILE,
    params:
        mindeg = 15,
        model_name = lambda wildcards : wildcards.model,
        t_train = lambda wildcards : wildcards.t_train,
    script:
        "workflow/prediction/evaluate-prediction.py"



# ========================================================
#  Plot data
# ========================================================
rule plot_prediction_results:
    input:
        input_file = PRED_TEST_EVAL_FILE,
        baseline_model_files = expand(PRED_TEST_EVAL_BASELINE_FILE, model = params_baseline_citation_prediction["model"]),
    output:
        output_file=FIG_PRED,
    params:
        data = lambda wildcards : wildcards.data,
        training_period = 5,
        train =lambda wildcards: wildcards.t_train,
        yscale = lambda wildcards: wildcards.yscale,
    script:
        "workflow/plot/plot-predicted-vs-true-citation.py"
