import json
import os
import re
import sys

import pandas as pd


def cleanhtml(raw_html):
    cleanr = re.compile("<.*?>")
    cleantext = re.sub(cleanr, "", raw_html)
    return cleantext


def parse_json_files(_file):
    with open(_file, "r") as f:
        data = json.load(f)
    return {
        "year": int(data["date"].split("-")[0]),
        "date": data["date"],
        "doi": data["id"],
        "journal_code": data["journal"]["id"],
        "title": cleanhtml(data["title"]["value"]),
    }


if __name__ == "__main__":

    meta_data_folder = sys.argv[1]
    out_papermeta_file = sys.argv[2]

    paper_list = []
    for root, _, files in os.walk(meta_data_folder):
        for file in files:
            if file.endswith(".json"):
                paper_list += [parse_json_files(os.path.join(root, file))]
    paper_meta_table = pd.DataFrame(paper_list)
    paper_meta_table.to_csv(out_papermeta_file, index=False)
