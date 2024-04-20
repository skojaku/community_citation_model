
######## snakemake preamble start (automatically inserted, do not edit) ########
import sys; sys.path.extend(['/home/skojaku/anaconda3/envs/legcit/lib/python3.7/site-packages', '/home/skojaku/projects/Legal-Citations/workflow/legcitv2/workflow']); import pickle; snakemake = pickle.loads(b'\x80\x03csnakemake.script\nSnakemake\nq\x00)\x81q\x01}q\x02(X\x05\x00\x00\x00inputq\x03csnakemake.io\nInputFiles\nq\x04)\x81q\x05(XL\x00\x00\x00/home/skojaku/projects/Legal-Citations/data/Data/Raw/Citation_Info_Dict.jsonq\x06XE\x00\x00\x00/home/skojaku/projects/Legal-Citations/data/Data/Raw/citations.csv.gzq\x07XI\x00\x00\x00/home/skojaku/projects/Legal-Citations/data/Data/Raw/court_hierarchy.jsonq\x08e}q\t(X\x06\x00\x00\x00_namesq\n}q\x0b(X\x0e\x00\x00\x00node_data_fileq\x0cK\x00N\x86q\rX\x0e\x00\x00\x00link_data_fileq\x0eK\x01N\x86q\x0fX\x0f\x00\x00\x00court_data_fileq\x10K\x02N\x86q\x11uX\x12\x00\x00\x00_allowed_overridesq\x12]q\x13(X\x05\x00\x00\x00indexq\x14X\x04\x00\x00\x00sortq\x15eh\x14cfunctools\npartial\nq\x16cbuiltins\ngetattr\nq\x17csnakemake.io\nNamedlist\nq\x18X\x0f\x00\x00\x00_used_attributeq\x19\x86q\x1aRq\x1b\x85q\x1cRq\x1d(h\x1b)}q\x1eX\x05\x00\x00\x00_nameq\x1fh\x14sNtq bh\x15h\x16h\x1b\x85q!Rq"(h\x1b)}q#h\x1fh\x15sNtq$bh\x0ch\x06h\x0eh\x07h\x10h\x08ubX\x06\x00\x00\x00outputq%csnakemake.io\nOutputFiles\nq&)\x81q\'(XW\x00\x00\x00/home/skojaku/projects/Legal-Citations/data/Data/legcitv2/preprocessed/citation_net.npzq(XV\x00\x00\x00/home/skojaku/projects/Legal-Citations/data/Data/legcitv2/preprocessed/paper_table.csvq)XV\x00\x00\x00/home/skojaku/projects/Legal-Citations/data/Data/legcitv2/preprocessed/court_table.csvq*e}q+(h\n}q,(X\x08\x00\x00\x00net_fileq-K\x00N\x86q.X\x0f\x00\x00\x00node_table_fileq/K\x01N\x86q0X\x10\x00\x00\x00court_table_fileq1K\x02N\x86q2uh\x12]q3(h\x14h\x15eh\x14h\x16h\x1b\x85q4Rq5(h\x1b)}q6h\x1fh\x14sNtq7bh\x15h\x16h\x1b\x85q8Rq9(h\x1b)}q:h\x1fh\x15sNtq;bh-h(h/h)h1h*ubX\x06\x00\x00\x00paramsq<csnakemake.io\nParams\nq=)\x81q>}q?(h\n}q@h\x12]qA(h\x14h\x15eh\x14h\x16h\x1b\x85qBRqC(h\x1b)}qDh\x1fh\x14sNtqEbh\x15h\x16h\x1b\x85qFRqG(h\x1b)}qHh\x1fh\x15sNtqIbubX\t\x00\x00\x00wildcardsqJcsnakemake.io\nWildcards\nqK)\x81qL}qM(h\n}qNh\x12]qO(h\x14h\x15eh\x14h\x16h\x1b\x85qPRqQ(h\x1b)}qRh\x1fh\x14sNtqSbh\x15h\x16h\x1b\x85qTRqU(h\x1b)}qVh\x1fh\x15sNtqWbubX\x07\x00\x00\x00threadsqXK\x01X\t\x00\x00\x00resourcesqYcsnakemake.io\nResources\nqZ)\x81q[(K\x01K\x01e}q\\(h\n}q](X\x06\x00\x00\x00_coresq^K\x00N\x86q_X\x06\x00\x00\x00_nodesq`K\x01N\x86qauh\x12]qb(h\x14h\x15eh\x14h\x16h\x1b\x85qcRqd(h\x1b)}qeh\x1fh\x14sNtqfbh\x15h\x16h\x1b\x85qgRqh(h\x1b)}qih\x1fh\x15sNtqjbh^K\x01h`K\x01ubX\x03\x00\x00\x00logqkcsnakemake.io\nLog\nql)\x81qm}qn(h\n}qoh\x12]qp(h\x14h\x15eh\x14h\x16h\x1b\x85qqRqr(h\x1b)}qsh\x1fh\x14sNtqtbh\x15h\x16h\x1b\x85quRqv(h\x1b)}qwh\x1fh\x15sNtqxbubX\x06\x00\x00\x00configqy}qz(X\x08\x00\x00\x00data_dirq{X0\x00\x00\x00/home/skojaku/projects/Legal-Citations/data/Dataq|X\t\x00\x00\x00paper_dirq}X,\x00\x00\x00/home/skojaku/projects/Legal-Citations/paperq~X\x0c\x00\x00\x00wos_json_dirq\x7fX \x00\x00\x00/gpfs/sciencegenome/WoSjson2019/q\x80uX\x04\x00\x00\x00ruleq\x81X\x16\x00\x00\x00construct_citation_netq\x82X\x0f\x00\x00\x00bench_iterationq\x83NX\t\x00\x00\x00scriptdirq\x84XA\x00\x00\x00/home/skojaku/projects/Legal-Citations/workflow/legcitv2/workflowq\x85ub.'); from snakemake.logging import logger; logger.printshellcmds = False; __real_file__ = __file__; __file__ = '/home/skojaku/projects/Legal-Citations/workflow/legcitv2/workflow/const-cit-net.py';
######## snakemake preamble end #########
# %%
import gzip
import sys

import networkx as nx
import numpy as np
import pandas as pd
import ujson
from scipy import sparse
from tqdm import tqdm

if "snakemake" in sys.modules:
    # Input
    node_data_file = snakemake.input["node_data_file"]
    link_data_file = snakemake.input["link_data_file"]
    court_data_file = snakemake.input["court_data_file"]
    output_net_file = snakemake.output["net_file"]
    output_node_table_file = snakemake.output["node_table_file"]
    output_court_table_file = snakemake.output["court_table_file"]
else:
    node_data_file = "../../../data/Data/Raw/Citation_Info_Dict.json"
    link_data_file = "../../../data/Data/Raw/citations.csv.gz"
    court_data_file = "../../../data/Data/Raw/court_hierarchy.json"
    output_net_file = "../../../data/Data/legcitv2/preprocessed/citation_net.npz"
    output_node_table_file = "../../../data/Data/legcitv2/preprocessed/paper_table.csv"
    output_court_table_file = "../../../data/Data/legcitv2/preprocessed/court_table.csv"

# %%
# Load
#

with open(node_data_file, "r") as f:
    node_data = ujson.load(f)

with open(court_data_file, "r") as f:
    court_data = ujson.load(f)

edges = []
with gzip.open(link_data_file, "rt") as f:
    for line in tqdm(f.readlines()):
        eles = [int(float(d)) for d in line.strip().split(",")]
        df = pd.DataFrame({"cited": eles[1:], "citing": eles[0]})
        edges.append(df)
    edge_table = pd.concat(edges)

#
# Preprocess
#
# Find all opinion ids in the data
opinion_set = np.unique(edges.values.reshape(-1))

#
# Construct the node table
#
# Assign unique consequtive ids starting from 0
opinion2id = dict(zip(opinion_set, np.arange(len(opinion_set))))
node_table = pd.DataFrame({"opinion": opinion_set})
node_table["paper_id"] = node_table["opinion"].map(opinion2id)

# append the metadata to the node_table
_node_table = (
    pd.DataFrame.from_dict(node_data, orient="index")
    .reset_index()
    .rename(columns={"index": "opinion"})
)
_node_table["opinion"] = _node_table["opinion"].astype(int)
node_table = pd.merge(node_table, _node_table, on="opinion", how="left")
node_table["date"] = pd.to_datetime(
    node_table["date"], errors="coerce"
)  # fixes some timestamp bugs
node_table["year"] = node_table["date"].dt.year
node_table["frac_year"] = (node_table["date"].dt.month - 1) / 12 + node_table[
    "date"
].dt.year

# %%
# Construct citation net
#
edge_table["citing"] = edge_table["citing"].map(opinion2id)
edge_table["cited"] = edge_table["cited"].map(opinion2id)

N = len(opinion2id)
net = sparse.csr_matrix(
    (
        np.ones(edge_table.shape[0]),
        (edge_table["citing"].values, edge_table["cited"].values),
    ),
    shape=(N, N),
)
# Construct the court hierarchy tree
edge_list = []
supreme = court_data[0][0]
edge_list += [{"venue": supreme, "parent": "", "depth": 0}]
for i in range(1, len(court_data)):
    appeal = court_data[i][0]
    district = court_data[i][1:]

    edge_list += [{"venue": appeal, "parent": supreme, "depth": 1}]
    for d in district:
        edge_list += [{"venue": d, "parent": appeal, "depth": 2}]
court_tree = pd.DataFrame(edge_list)

#
# Post-process
#
# Rename
node_table = node_table.rename(columns={"court": "venue", "opinion": "opinion_id"})

court_tree = court_tree.rename(columns={"depth": "venueType"})
court_tree["venueType"] = court_tree["venueType"].map(
    {0: "Supreme", 1: "Appeals", 2: "District"}
)
node_table = pd.merge(node_table, court_tree, on="venue", how="left")

# %%
# Save
sparse.save_npz(output_net_file, net)
node_table.to_csv(output_node_table_file, index=False)
court_tree.to_csv(output_court_table_file, index=False)

# %%
