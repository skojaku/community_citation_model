# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-12-27 14:19:48
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-01-06 16:16:16

#
# Simulation
#
sim_pa_paramspace = to_paramspace(sim_pa_params)
sim_geo_paramspace = to_paramspace(sim_geo_params)

SIM_EXP_DIR = j(DATA_DIR, "synthetic")
SIM_EXP_NET_DIR = j(SIM_EXP_DIR, "networks")
SIM_EXP_PA_NET = j(SIM_EXP_NET_DIR, f"net_model~pa_{sim_pa_paramspace.wildcard_pattern}.npz")
SIM_EXP_GEO_NET = j(SIM_EXP_NET_DIR, f"net_model~spherical_{sim_geo_paramspace.wildcard_pattern}.npz")
SIM_EXP_PA_NODE_TABLE = j(SIM_EXP_NET_DIR, f"node_model~pa-{sim_pa_paramspace.wildcard_pattern}.csv")
SIM_EXP_GEO_NODE_TABLE = j(SIM_EXP_NET_DIR, f"node_model~spherical_{sim_geo_paramspace.wildcard_pattern}.csv")

# Results
SIM_EXP_STAT_DIR = j(SIM_EXP_DIR, "stats")
SIM_EXP_GEO_DATA_EVENT_INTERVAL = j(SIM_EXP_STAT_DIR, f"stat~interval_model~spherical_{sim_geo_paramspace.wildcard_pattern}.json")
SIM_EXP_PA_DATA_EVENT_INTERVAL = j(SIM_EXP_STAT_DIR, f"stat~interval_model~pa_{sim_pa_paramspace.wildcard_pattern}.json")
SIM_EXP_GEO_DATA_CITATION_RATE = j(SIM_EXP_STAT_DIR, f"stat~citationRate_model~spherical_{sim_geo_paramspace.wildcard_pattern}.csv")
SIM_EXP_PA_DATA_CITATION_RATE = j(SIM_EXP_STAT_DIR, f"stat~citationRate_model~pa_{sim_pa_paramspace.wildcard_pattern}.csv")
SIM_EXP_GEO_DATA_SB_COEF_TABLE = j(SIM_EXP_STAT_DIR, f"stat~sbcoef_model~spherical_{sim_geo_paramspace.wildcard_pattern}.csv")
SIM_EXP_PA_DATA_SB_COEF_TABLE = j(SIM_EXP_STAT_DIR, f"stat~sbcoef_model~pa_{sim_pa_paramspace.wildcard_pattern}.csv")
SIM_EXP_GEO_DATA_DEGREE_DIST= j(SIM_EXP_STAT_DIR, f"stat~degree-dist_model~spherical_{sim_geo_paramspace.wildcard_pattern}.csv")
SIM_EXP_PA_DATA_DEGREE_DIST= j(SIM_EXP_STAT_DIR, f"stat~degree-dist_model~pa_{sim_pa_paramspace.wildcard_pattern}.csv")
SIM_EXP_GEO_DATA_NEW_VS_ACCUM_CIT=j(SIM_EXP_STAT_DIR, f"stat~NewVsOld_model~spherical_{sim_geo_paramspace.wildcard_pattern}.csv")
SIM_EXP_PA_DATA_NEW_VS_ACCUM_CIT=j(SIM_EXP_STAT_DIR, f"stat~NewVsOld_model~pa_{sim_pa_paramspace.wildcard_pattern}.csv")

#
# Validation
#
_sim_params = {k:v for k,v in sim_geo_params.items() if k in ["dim", "growthRate"]}
_sim_paramspace = to_paramspace(_sim_params)

# Ablation
FIG_SIM_EXP_NEW_VS_ACCUM_CIT = j(FIG_DIR, "synthe", f"pref_attachment_{_sim_paramspace.wildcard_pattern}.pdf")
FIG_SIM_EXP_CITATION_RATE = j(FIG_DIR, "synthe", f"event-citationRate_{_sim_paramspace.wildcard_pattern}.pdf")
FIG_SIM_EXP_SB_COEFFICIENT = j(FIG_DIR, "synthe", f"dist-sb_coefficient_{_sim_paramspace.wildcard_pattern}.pdf")

# Vs dimensions
_vs_dim_sim_params = {k:v for k,v in sim_geo_params.items() if k in ["growthRate"]}
_vs_dim_sim_paramspace = to_paramspace(_vs_dim_sim_params)
FIG_SIM_EXP_VS_DIM_NEW_VS_ACCUM_CIT = j(FIG_DIR, "synthe", f"pref_attachment_vs_dim_{_vs_dim_sim_paramspace.wildcard_pattern}.pdf")
FIG_SIM_EXP_VS_DIM_CITATION_RATE = j(FIG_DIR, "synthe", f"event-citationRate_vs_dim_{_vs_dim_sim_paramspace.wildcard_pattern}.pdf")
FIG_SIM_EXP_VS_DIM_SB_COEFFICIENT = j(FIG_DIR, "synthe", f"dist-sb_coefficient_vs_dim_{_vs_dim_sim_paramspace.wildcard_pattern}.pdf")

rule simulations_all:
    input:
        # Citation stats
        expand(SIM_EXP_GEO_DATA_EVENT_INTERVAL, **sim_geo_params),
        expand(SIM_EXP_PA_DATA_EVENT_INTERVAL, **sim_pa_params),
        expand(SIM_EXP_GEO_DATA_CITATION_RATE, **sim_geo_params),
        expand(SIM_EXP_PA_DATA_CITATION_RATE, **sim_pa_params),
        expand(SIM_EXP_GEO_DATA_SB_COEF_TABLE, **sim_geo_params),
        expand(SIM_EXP_PA_DATA_SB_COEF_TABLE, **sim_pa_params),
        expand(SIM_EXP_GEO_DATA_DEGREE_DIST, **sim_geo_params),
        expand(SIM_EXP_PA_DATA_DEGREE_DIST, **sim_pa_params),
        expand(SIM_EXP_GEO_DATA_NEW_VS_ACCUM_CIT, **sim_geo_params),
        expand(SIM_EXP_PA_DATA_NEW_VS_ACCUM_CIT, **sim_pa_params),

rule simulations_figs:
    input:
        expand(FIG_SIM_EXP_SB_COEFFICIENT, **_sim_params),
        expand(FIG_SIM_EXP_CITATION_RATE, **_sim_params),
        expand(FIG_SIM_EXP_NEW_VS_ACCUM_CIT, **_sim_params),
        expand(FIG_SIM_EXP_VS_DIM_SB_COEFFICIENT, **_vs_dim_sim_params),
        expand(FIG_SIM_EXP_VS_DIM_CITATION_RATE, **_vs_dim_sim_params),
        expand(FIG_SIM_EXP_VS_DIM_NEW_VS_ACCUM_CIT, **_vs_dim_sim_params),
        #"figs/sim_net_recency.pdf",
        #"figs/sim_net_pref_attachment.pdf",
        #"figs/sim_net_deg_dist.pdf",
        #"figs/sim_net_sb_coef.pdf",

#
# Simulations
#
rule simulate_networks_validation_geometric:
    params:
        aging = lambda wildcards: wildcards.aging,
        fitness = lambda wildcards: wildcards.fitness,
        dim = lambda wildcards: wildcards.dim,
        T = lambda wildcards: wildcards.T,
        nrefs = lambda wildcards: wildcards.nrefs,
        #kappa_paper = lambda wildcards: wildcards.kappaPaper,
        kappa_citations = lambda wildcards: wildcards.kappaCitation,
        mu = lambda wildcards: wildcards.mu,
        sigma = lambda wildcards: wildcards.sigma,
        nt = lambda wildcards: wildcards.nt,
        growthRate = lambda wildcards: wildcards.growthRate,
        c0 = lambda wildcards: wildcards.c0,
        n_samples = SIM_N_SAMPLES,
        geometry = "True"
    output:
        output_net_file = SIM_EXP_GEO_NET,
        output_node_file = SIM_EXP_GEO_NODE_TABLE
    script:
        "workflow/simulation/simulate-spherical-model.py"


rule simulate_networks_validation_pref_attachment:
    params:
        T = lambda wildcards: wildcards.T,
        nrefs = lambda wildcards: wildcards.nrefs,
        nt = lambda wildcards: wildcards.nt,
        growthRate = lambda wildcards: wildcards.growthRate,
        n_samples = SIM_N_SAMPLES
    output:
        output_net_file = SIM_EXP_PA_NET,
        output_node_file = SIM_EXP_PA_NODE_TABLE
    script:
        "workflow/simulation/simulate-pref-attachment.py"

rule calc_citation_event_interval_simulations_pref_attachment:
    input:
        net_file = SIM_EXP_PA_NET,
        paper_table_file = SIM_EXP_PA_NODE_TABLE,
    params:
        focal_period = [20, 80],
        groupby = None,
        dataName = "PA"
    output:
        output_file = SIM_EXP_PA_DATA_EVENT_INTERVAL
    script:
        "workflow/stats/calc-citation-event-interval.py"

rule calc_citation_event_interval_simulations_spherical:
    input:
        net_file = SIM_EXP_GEO_NET,
        paper_table_file = SIM_EXP_GEO_NODE_TABLE,
    params:
        focal_period = [20, 80],
        groupby=None,
        dataName = "Spherical"
    output:
        output_file = SIM_EXP_GEO_DATA_EVENT_INTERVAL
    script:
        "workflow/stats/calc-citation-event-interval.py"

use rule calc_citation_rate as calc_citation_rate_simulations_pref_attachment with:
    input:
        net_file = SIM_EXP_PA_NET,
        paper_table_file = SIM_EXP_PA_NODE_TABLE,
    params:
        focal_period = [20, 100],
        dataName = "PA",
    output:
        output_file = SIM_EXP_PA_DATA_CITATION_RATE

use rule calc_citation_rate as calc_citation_rate_simulations_spherical with:
    input:
        net_file = SIM_EXP_GEO_NET,
        paper_table_file = SIM_EXP_GEO_NODE_TABLE,
    params:
        dataName = "Spherical",
        focal_period = [20, 100],
    output:
        output_file = SIM_EXP_GEO_DATA_CITATION_RATE

rule calc_sleeping_beauty_coefficient_simulations_spherical:
    input:
        net_file = SIM_EXP_GEO_NET,
        paper_table_file = SIM_EXP_GEO_NODE_TABLE
    params:
        dataName = "Spherical"
    output:
        output_file = SIM_EXP_GEO_DATA_SB_COEF_TABLE
    script:
        "workflow/stats/calc-sb-coefficient.py"

rule calc_sleeping_beauty_coefficient_simulations_pa:
    input:
        net_file = SIM_EXP_PA_NET,
        paper_table_file = SIM_EXP_PA_NODE_TABLE
    params:
        dataName = "PA"
    output:
        output_file = SIM_EXP_PA_DATA_SB_COEF_TABLE
    script:
        "workflow/stats/calc-sb-coefficient.py"

rule calc_degree_distribution_simulations_geo:
    input:
        net_file = SIM_EXP_GEO_NET,
        paper_table_file = SIM_EXP_GEO_NODE_TABLE
    params:
        dataName = "Spherical",
        groupby = None
    output:
        output_file = SIM_EXP_GEO_DATA_DEGREE_DIST
    script:
        "workflow/stats/calc-degree-distribution.py"

rule calc_degree_distribution_simulations_pa:
    input:
        net_file = SIM_EXP_PA_NET,
        paper_table_file = SIM_EXP_PA_NODE_TABLE
    params:
        dataName = "PA"
    output:
        output_file = SIM_EXP_PA_DATA_DEGREE_DIST
    script:
        "workflow/stats/calc-degree-distribution.py"

rule calc_pref_rate_simulations_pa:
    input:
        net_file = SIM_EXP_PA_NET,
        paper_table_file = SIM_EXP_PA_NODE_TABLE
    params:
        dataName = "PA",
        groupby=None,
    output:
        output_file = SIM_EXP_PA_DATA_NEW_VS_ACCUM_CIT
    script:
        "workflow/stats/calc-new-vs-accumulated-citations.py"

rule calc_pref_rate_simulations_geo:
    input:
        net_file = SIM_EXP_GEO_NET,
        paper_table_file = SIM_EXP_GEO_NODE_TABLE
    params:
        dataName = "Spherical",
        groupby=None,
    output:
        output_file = SIM_EXP_GEO_DATA_NEW_VS_ACCUM_CIT
    script:
        "workflow/stats/calc-new-vs-accumulated-citations.py"

# ========================================================
#  Plot data
# ========================================================

#
# Ablation
#
rule plot_pref_attachment_simulation_validation:
    input:
        input_files = expand(SIM_EXP_GEO_DATA_NEW_VS_ACCUM_CIT, **sim_geo_params) + expand(SIM_EXP_PA_DATA_NEW_VS_ACCUM_CIT, **sim_pa_params)
    params:
        dim = lambda wildcards: wildcards.dim,
        growthRate = lambda wildcards: wildcards.growthRate
    output:
        output_file = FIG_SIM_EXP_NEW_VS_ACCUM_CIT
    script:
        "workflow/simulation/plot-pref-attachment.py"

rule plot_citation_rate_simulation_validation:
    input:
        input_files = expand(SIM_EXP_GEO_DATA_CITATION_RATE, **sim_geo_params) + expand(SIM_EXP_PA_DATA_CITATION_RATE, **sim_pa_params)
    params:
        dim = lambda wildcards: wildcards.dim,
        growthRate = lambda wildcards: wildcards.growthRate,
        groupby=None,
        focal_degree = 25,
    output:
        output_file = FIG_SIM_EXP_CITATION_RATE
    script:
        "workflow/simulation/plot-citation-rate.py"

rule plot_sb_coef_simulation_validation:
    input:
        input_files = expand(SIM_EXP_GEO_DATA_SB_COEF_TABLE, **sim_geo_params) + expand(SIM_EXP_PA_DATA_SB_COEF_TABLE, **sim_pa_params)
    params:
        dim = lambda wildcards: wildcards.dim,
        growthRate = lambda wildcards: wildcards.growthRate
    output:
        output_file = FIG_SIM_EXP_SB_COEFFICIENT
    script:
        "workflow/simulation/plot-sb-coef.py"

#
# Vs. Dimension
#
rule plot_pref_attachment_comp_dim_simulation_validation:
    input:
        input_files = expand(SIM_EXP_GEO_DATA_NEW_VS_ACCUM_CIT, **sim_geo_params) + expand(SIM_EXP_PA_DATA_NEW_VS_ACCUM_CIT, **sim_pa_params)
    params:
        growthRate = lambda wildcards: wildcards.growthRate
    output:
        output_file = FIG_SIM_EXP_VS_DIM_NEW_VS_ACCUM_CIT
    script:
        "workflow/simulation/plot-pref-attachment-comp-dim.py"

rule plot_citation_rate_comp_dim_simulation_validation:
    input:
        input_files = expand(SIM_EXP_GEO_DATA_CITATION_RATE, **sim_geo_params) + expand(SIM_EXP_PA_DATA_CITATION_RATE, **sim_pa_params)
    params:
        growthRate = lambda wildcards: wildcards.growthRate,
        groupby=None,
        focal_degree = 25,
    output:
        output_file = FIG_SIM_EXP_VS_DIM_CITATION_RATE
    script:
        "workflow/simulation/plot-citation-rate-comp-dim.py"

rule plot_sb_coef_comp_dim_simulation_validation:
    input:
        input_files = expand(SIM_EXP_GEO_DATA_SB_COEF_TABLE, **sim_geo_params) + expand(SIM_EXP_PA_DATA_SB_COEF_TABLE, **sim_pa_params)
    params:
        growthRate = lambda wildcards: wildcards.growthRate
    output:
        output_file = FIG_SIM_EXP_VS_DIM_SB_COEFFICIENT
    script:
        "workflow/simulation/plot-sb-coef-comp-dim.py"
