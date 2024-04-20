
######## snakemake preamble start (automatically inserted, do not edit) ########
import sys; sys.path.extend(['/home/skojaku/anaconda3/envs/legcit/lib/python3.7/site-packages', '/home/skojaku/projects/Legal-Citations/workflow/uspto/workflow']); import pickle; snakemake = pickle.loads(b'\x80\x03csnakemake.script\nSnakemake\nq\x00)\x81q\x01}q\x02(X\x05\x00\x00\x00inputq\x03csnakemake.io\nInputFiles\nq\x04)\x81q\x05Xc\x00\x00\x00/gpfs/sciencegenome/uspto/USPTO_2020/PatentsView_20200630/PatentsView_20200630.uspatentcitation.sqlq\x06a}q\x07(X\x06\x00\x00\x00_namesq\x08}q\tX\n\x00\x00\x00input_fileq\nK\x00N\x86q\x0bsX\x12\x00\x00\x00_allowed_overridesq\x0c]q\r(X\x05\x00\x00\x00indexq\x0eX\x04\x00\x00\x00sortq\x0feh\x0ecfunctools\npartial\nq\x10cbuiltins\ngetattr\nq\x11csnakemake.io\nNamedlist\nq\x12X\x0f\x00\x00\x00_used_attributeq\x13\x86q\x14Rq\x15\x85q\x16Rq\x17(h\x15)}q\x18X\x05\x00\x00\x00_nameq\x19h\x0esNtq\x1abh\x0fh\x10h\x15\x85q\x1bRq\x1c(h\x15)}q\x1dh\x19h\x0fsNtq\x1ebh\nh\x06ubX\x06\x00\x00\x00outputq\x1fcsnakemake.io\nOutputFiles\nq )\x81q!XL\x00\x00\x00/home/skojaku/projects/Legal-Citations/data/Data/uspto/raw/usptocitation.csvq"a}q#(h\x08}q$X\x0b\x00\x00\x00output_fileq%K\x00N\x86q&sh\x0c]q\'(h\x0eh\x0feh\x0eh\x10h\x15\x85q(Rq)(h\x15)}q*h\x19h\x0esNtq+bh\x0fh\x10h\x15\x85q,Rq-(h\x15)}q.h\x19h\x0fsNtq/bh%h"ubX\x06\x00\x00\x00paramsq0csnakemake.io\nParams\nq1)\x81q2}q3(h\x08}q4h\x0c]q5(h\x0eh\x0feh\x0eh\x10h\x15\x85q6Rq7(h\x15)}q8h\x19h\x0esNtq9bh\x0fh\x10h\x15\x85q:Rq;(h\x15)}q<h\x19h\x0fsNtq=bubX\t\x00\x00\x00wildcardsq>csnakemake.io\nWildcards\nq?)\x81q@}qA(h\x08}qBh\x0c]qC(h\x0eh\x0feh\x0eh\x10h\x15\x85qDRqE(h\x15)}qFh\x19h\x0esNtqGbh\x0fh\x10h\x15\x85qHRqI(h\x15)}qJh\x19h\x0fsNtqKbubX\x07\x00\x00\x00threadsqLK\x01X\t\x00\x00\x00resourcesqMcsnakemake.io\nResources\nqN)\x81qO(K\x01K\x01e}qP(h\x08}qQ(X\x06\x00\x00\x00_coresqRK\x00N\x86qSX\x06\x00\x00\x00_nodesqTK\x01N\x86qUuh\x0c]qV(h\x0eh\x0feh\x0eh\x10h\x15\x85qWRqX(h\x15)}qYh\x19h\x0esNtqZbh\x0fh\x10h\x15\x85q[Rq\\(h\x15)}q]h\x19h\x0fsNtq^bhRK\x01hTK\x01ubX\x03\x00\x00\x00logq_csnakemake.io\nLog\nq`)\x81qa}qb(h\x08}qch\x0c]qd(h\x0eh\x0feh\x0eh\x10h\x15\x85qeRqf(h\x15)}qgh\x19h\x0esNtqhbh\x0fh\x10h\x15\x85qiRqj(h\x15)}qkh\x19h\x0fsNtqlbubX\x06\x00\x00\x00configqm}qn(X\x08\x00\x00\x00data_dirqoX0\x00\x00\x00/home/skojaku/projects/Legal-Citations/data/DataqpX\t\x00\x00\x00paper_dirqqX,\x00\x00\x00/home/skojaku/projects/Legal-Citations/paperqrX\x0c\x00\x00\x00wos_json_dirqsX \x00\x00\x00/gpfs/sciencegenome/WoSjson2019/qtuX\x04\x00\x00\x00rulequX\x15\x00\x00\x00import_citation_tableqvX\x0f\x00\x00\x00bench_iterationqwNX\t\x00\x00\x00scriptdirqxX>\x00\x00\x00/home/skojaku/projects/Legal-Citations/workflow/uspto/workflowqyub.'); from snakemake.logging import logger; logger.printshellcmds = False; __real_file__ = __file__; __file__ = '/home/skojaku/projects/Legal-Citations/workflow/uspto/workflow/make-citation-table.py';
######## snakemake preamble end #########
# %%
import sqlite3
import pandas as pd
import sys

if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    output_file = snakemake.output["output_file"]
else:
    input_file = "/gpfs/sciencegenome/uspto/USPTO_2020/PatentsView_20200630/PatentsView_20200630.uspatentcitation.sql"
    output_file = "../data/"

#
# Set up the database in RAM
#
connection = sqlite3.connect(":memory:")
cursor = connection.cursor()

# Read citation table
cursor.execute(
    """CREATE TABLE uspatentcitation(citing_patent_id, seq_id, cited_patent_id, assigned_by)"""
)
cursor.executescript(open(input_file).read())


# %%
# Save
#
df = pd.read_sql_query("SELECT * FROM uspatentcitation", connection)
df.to_csv(output_file, index=False)
