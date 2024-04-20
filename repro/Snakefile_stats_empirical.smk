DATA_NUM_PAPERS = j(DATA_DIR, "{data}", "plot_data", "num-papers.csv")
DATA_NUM_CITATIONS = j(DATA_DIR, "{data}", "plot_data", "num-citations.csv")
DATA_DEGREE_DIST = j(DATA_DIR, "{data}", "plot_data", "degree-dist.csv")
DATA_ACTIVE_AUTHOR_TABLE = j(DATA_DIR, "{data}", "plot_data", "num_active_authors.csv")
DATA_CAREER_LEN_TABLE  = j(DATA_DIR, "{data}", "plot_data", "career_age_table.csv")
DATA_EVENT_INTERVAL = j(DATA_DIR, "{data}", "plot_data", "citation-event-interval.csv")
DATA_CITATION_RATE = j(DATA_DIR, "{data}", "plot_data", "citation-rate.csv")
DATA_NEW_VS_ACCUM_CIT  = j(DATA_DIR, "{data}", "plot_data", "new_vs_accumulated_citation.csv")
DATA_SB_COEF_TABLE = j(DATA_DIR, "{data}", "plot_data", "sb_coefficient_table.csv")
FITTED_POWER_LAW_PARAMS = j(DATA_DIR, "{data}", "plot_data", "fitted-power-law-params.json")

# Q-factor model
DATA_PRODUCTIVITY = j(DATA_DIR, "{data}", "plot_data", "productivity_timeWindow~{time_window}.csv")
#DATA_AVERAGE_IMPACT = j(DATA_DIR, "{data}", "plot_data", "average-impact_timeWindow~{time_window}.csv")
DATA_TIME_HIGHEST_IMPACT = j(DATA_DIR, "{data}", "plot_data", "highest-impact_timeWindow~{time_window}.csv")

# Plot parameters
MAX_CARR_AGE = 40 # Maximum age to plot for the productivity.
MIN_CAREER_FOR_RANDOM_IMPACT_RULE = 0 # The same parameter as is used in Sinatra et al.

# ==================
# Figures
# ==================
FIG_DATA_NUM_PAPERS = j(FIG_DIR, "stat", "{data}", "num-papers.pdf")
FIG_DATA_NUM_AUTHORS = j(FIG_DIR, "stat", "{data}", "num-authors.pdf")
FIG_DATA_AVE_NUM_REFS = j(FIG_DIR, "stat", "{data}", "num-ave-references.pdf")
FIG_DATA_NUM_OUTREF_CITATIONS = j(FIG_DIR, "stat", "{data}", "num-citations.pdf")
FIG_DATA_DEGREE_DIST = j(FIG_DIR, "stat", "{data}", "degree-dist.pdf")
FIG_DATA_CITATION_RATE= j(FIG_DIR, "stat", "{data}", "citation-rate.pdf")
FIG_DATA_EVENT_INTERVAL= j(FIG_DIR, "stat", "{data}", "event-interval.pdf")
FIG_DATA_EVENT_INTERVAL_UNNORMALIZED = j(FIG_DIR, "stat", "{data}", "event-interval_unnormalized.pdf")
FIG_DATA_NEW_VS_ACCUM_CIT = j(FIG_DIR, "stat", "{data}", "pref_attachment.pdf")
FIG_DATA_SB_COEFFICIENT = j(FIG_DIR, "stat", "{data}", "dist-sb_coefficient.pdf")
FIG_DATA_AWAKENING_TIME = j(FIG_DIR, "stat", "{data}", "dist-awakening_time.pdf")

FIG_DATA_CAREER_LEN_AUTHORS = j(FIG_DIR, "stat", "{data}", "dist-career-age.pdf")
FIG_DATA_PRODUCTIVITY = j(FIG_DIR, "stat", "{data}", "productivity_timeWindow~{time_window}.pdf")
FIG_DATA_TIME_HIGHEST_IMPACT = j(FIG_DIR, "stat", "{data}", "time-highest-impact_timeWindow~{time_window}.pdf")
FIG_DATA_TIME_HIGHEST_IMPACT_UNIFORMITY= j(FIG_DIR, "stat", "{data}", "time-highest-impact_uniformity_timeWindow~{time_window}.pdf")
FIG_DATA_TIME_HIGHEST_IMPACT_SI = j(FIG_DIR, "stat", "{data}", "si-time-highest-impact_timeWindow~{time_window}.pdf")
FIG_DATA_AVERAGE_IMPACT = j(FIG_DIR, "stat", "{data}", "average-impact_timeWindow~{time_window}.pdf")
FIG_DATA_CAREER_LEN_AUTHORS = j(FIG_DIR, "stat", "{data}", "career_length_dist.pdf")

rule stats_empirical_all:
    input:
        # Citation stats
        expand(DATA_EVENT_INTERVAL, data = DATA_LIST),
        expand(DATA_CITATION_RATE, data = DATA_LIST),
        expand(DATA_SB_COEF_TABLE, data = DATA_LIST),
        expand(DATA_PRODUCTIVITY, data = DATA_LIST, time_window=CIT_TIME_WINDOW_LIST),
        expand(DATA_TIME_HIGHEST_IMPACT, data = DATA_LIST, time_window=CIT_TIME_WINDOW_LIST),
        expand(DATA_NUM_CITATIONS, data = DATA_LIST),
        expand(DATA_NUM_PAPERS, data = DATA_LIST),
        expand(DATA_DEGREE_DIST, data = DATA_LIST),
        expand(DATA_ACTIVE_AUTHOR_TABLE, data = DATA_LIST),
        expand(DATA_CAREER_LEN_TABLE, data = DATA_LIST),
        expand(FITTED_POWER_LAW_PARAMS, data = DATA_LIST),
        #expand(FITNESS_TABLE, data = DATA_LIST),
        expand(DATA_NEW_VS_ACCUM_CIT, data = DATA_LIST),
        expand(AUTHOR_TABLE_Q_FACTOR_EXTENDED, data = DATA_LIST, time_window=CIT_TIME_WINDOW_LIST),
        expand(AUTHOR_TABLE_Q_FACTOR_BEFORE_AFTER_NOMINATION, data = DATA_LIST, time_window=CIT_TIME_WINDOW_LIST),
#
# Calculate the statistics
#
rule calc_num_cases:
    input:
        net_file = CITATION_NET,
        paper_table_file = PAPER_TABLE,
    params:
        groupby = lambda wildcards :"venueType" if wildcards.data in ["legcit", "legcitv2"] else None,
        time_resol = lambda wildcards : 5 if wildcards.data in ["aps", "legcit", "legcitv2"] else 1,
    output:
        output_file = DATA_NUM_PAPERS
    script:
        "workflow/stats/calc-num-papers.py"

rule calc_num_citations:
    input:
        net_file = CITATION_NET,
        paper_table_file = PAPER_TABLE,
    params:
        groupby = lambda wildcards :"venueType" if wildcards.data in ["legcit", "legcitv2"] else None,
        time_resol = lambda wildcards : 5 if wildcards.data in ["aps", "legcit", "legcitv2"] else 1,
    output:
        output_file = DATA_NUM_CITATIONS
    script:
        "workflow/stats/calc-num-citations.py"

rule calc_degree_distribution:
    input:
        net_file = CITATION_NET,
        paper_table_file = PAPER_TABLE,
    params:
        groupby = lambda wildcards :"venueType" if wildcards.data in ["legcit", "legcitv2"] else None
    output:
        output_file = DATA_DEGREE_DIST
    script:
        "workflow/stats/calc-degree-distribution.py"

rule calc_num_active_authors:
    input:
        paper_table_file = PAPER_TABLE,
        author_table_file = AUTHOR_TABLE,
        paper_author_net_file = PAPER_AUTHOR_TABLE,
    output:
        output_file = DATA_ACTIVE_AUTHOR_TABLE,
        output_career_age_table_file = DATA_CAREER_LEN_TABLE
    script:
        "workflow/stats/calc-num-active-authors.py"


rule calc_citation_event_interval:
    input:
        net_file = CITATION_NET,
        paper_table_file = PAPER_TABLE,
    params:
        groupby = lambda wildcards :"venueType" if wildcards.data in ["legcit", "legcitv2"] else None,
        focal_period = [2000, 2010],
	    dataName = "Empirical",
        normalize_citation_count = True
    output:
        output_file = DATA_EVENT_INTERVAL
    script:
        "workflow/stats/calc-citation-event-interval.py"

rule calc_citation_rate:
    input:
        net_file = CITATION_NET,
        paper_table_file = PAPER_TABLE,
    params:
        dataName = "Empirical",
        focal_period = [1960, 2000]
    output:
        output_file = DATA_CITATION_RATE
    script:
        "workflow/stats/calc-citation-rate.py"

rule calc_powerlaw_deg_dist_params:
    input:
        net_file = CITATION_NET,
        paper_table_file = PAPER_TABLE,
    params:
        groupby = lambda wildcards :"venueType" if wildcards.data in ["legcit", "legcitv2"] else None,
    output:
        output_file = FITTED_POWER_LAW_PARAMS
    script:
        "workflow/stats/fit-powerlaw-degree-dist.py"


rule calc_pref_rate:
    input:
        net_file = CITATION_NET,
        paper_table_file = PAPER_TABLE,
    params:
        groupby = lambda wildcards :"venueType" if wildcards.data in ["legcit", "legcitv2"] else None,
        dataName = "Empirical",
    output:
        output_file = DATA_NEW_VS_ACCUM_CIT
    script:
        "workflow/stats/calc-new-vs-accumulated-citations.py"

rule calc_paper_impact:
    input:
        net_file = CITATION_NET,
        paper_table_file = PAPER_TABLE,
    params:
        time_window = lambda wildcards: int(wildcards.time_window)
    output:
        output_file = PAPER_IMPACT
    script:
        "workflow/stats/calc-paper-impact.py"

rule construct_publication_sequence:
    input:
        net_file = CITATION_NET,
        paper_table_file = PAPER_TABLE,
        author_table_file = AUTHOR_TABLE,
        paper_author_net_file = PAPER_AUTHOR_TABLE,
        paper_impact_file = PAPER_IMPACT
    output:
        output_file = PUB_SEQ
    script:
        "workflow/stats/generate-publication-sequence.py"

rule calc_productivity:
    input:
        input_file = PUB_SEQ
    output:
        output_file = DATA_PRODUCTIVITY
    script:
        "workflow/stats/calc-productivity.py"

rule calc_time_of_highest_impact_paper:
    input:
        input_file = PUB_SEQ
    output:
        output_file = DATA_TIME_HIGHEST_IMPACT
    params:
        min_career = MIN_CAREER_FOR_RANDOM_IMPACT_RULE
    script:
        "workflow/stats/calc-time-highest-impact.py"



rule calc_author_table_q_factor_extended:
    input:
        paper_author_net_file = PAPER_AUTHOR_TABLE,
        paper_table_file = PAPER_TABLE,
        author_table_file = AUTHOR_TABLE,
        paper_impact_file = PAPER_IMPACT,
        nomination_date_file = NOMINATION_DATE_FILE
    params:
        citation_time_window=lambda wildcards: int(wildcards.time_window)
    output:
        output_file = AUTHOR_TABLE_Q_FACTOR_EXTENDED,
        output_file_nomination_date = AUTHOR_TABLE_Q_FACTOR_BEFORE_AFTER_NOMINATION
    script:
        "workflow/stats/calc-Q-factor.py"

rule calc_sleeping_beauty_coefficient:
    input:
        net_file = CITATION_NET,
        paper_table_file = PAPER_TABLE,
    params:
        dataName = "Empirical"
    output:
        output_file = DATA_SB_COEF_TABLE,
    script:
        "workflow/stats/calc-sb-coefficient.py"

#rule calc_sleeping_beauty_coefficient_for_rand_net:
#    input:
#        net_file = CITATION_NET,
#        paper_table_file = PAPER_TABLE,
#    output:
#        output_file = RAND_SB_COEF_TABLE
#    params:
#        random_model = lambda wildcards : wildcards.model
#    script:
#        "workflow/stats/calc-sb-coefficient.py"

#
# Plot
#
rule plot_num_papers:
    input:
        input_file = DATA_NUM_PAPERS
    params:
        data = lambda wildcards : wildcards.data,
    output:
        output_file = FIG_DATA_NUM_PAPERS
    script:
        "workflow/plot/plot-number-of-papers.py"

rule plot_num_citations:
    input:
        citation_count_file = DATA_NUM_CITATIONS,
        paper_count_file = DATA_NUM_PAPERS,
        citation_net_file = CITATION_NET,
        paper_table_file = PAPER_TABLE,
    params:
        data = lambda wildcards : wildcards.data,
    output:
        output_outref_file = FIG_DATA_NUM_OUTREF_CITATIONS,
        output_outaveref_file = FIG_DATA_AVE_NUM_REFS
    script:
        "workflow/plot/plot-number-of-citations.py"

rule plot_pref_attachment:
    input:
        input_file = DATA_NEW_VS_ACCUM_CIT
    output:
        output_file =  FIG_DATA_NEW_VS_ACCUM_CIT
    params:
        data = lambda wildcards : wildcards.data,
        title = lambda wildcards: " | ".join([f"{k}~{v}" for k, v in wildcards.items() if k not in ["geometry", "sample"]])
    script:
        "workflow/plot/plot-new-vs-accumulated-citations-empirical.py"

rule plot_degree_distribution:
    input:
        input_file = FITTED_POWER_LAW_PARAMS,
    output:
        output_deg_file =  FIG_DATA_DEGREE_DIST
    params:
        data = lambda wildcards : wildcards.data,
    script:
        "workflow/plot/plot-degree-distribution.py"

rule plot_citation_rate:
    input:
        input_file = DATA_CITATION_RATE
    output:
        output_file =  FIG_DATA_CITATION_RATE
    params:
        data = lambda wildcards : wildcards.data,
    script:
        "workflow/plot/plot-citation-rate.py"


rule plot_citation_event_interval:
    input:
        input_file = DATA_EVENT_INTERVAL
    output:
        output_file =  FIG_DATA_EVENT_INTERVAL
    params:
        data = lambda wildcards : wildcards.data,
	normalize_citation_count = True
    script:
        "workflow/plot/plot-event-interval.py"

rule plot_citation_event_interval_unnormalized:
    input:
        input_file = DATA_EVENT_INTERVAL
    params:
        data = lambda wildcards : wildcards.data,
        normalize_citation_count = False
    output:
        output_file = FIG_DATA_EVENT_INTERVAL_UNNORMALIZED
    script:
        "workflow/plot/plot-event-interval.py"
