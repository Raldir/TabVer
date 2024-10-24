"""
Author: Rami Aly
"""

import argparse
import json
import os

"""
The name of this file is a bit misleading since the original FEVER dataset is
also in JSONL format. This script converts them into a JSONL format compatible
with anserini.
"""


def main():
    LABEL_MAPPING = {0: "REFUTES", 1: "SUPPORTS"}
    INPUT_PATH = os.path.join("data", "tabfact")

    # TABFACT_DEV_IDS = os.path.join(INPUT_PATH, "val_ids.json")
    # INPUT_PATH_CLAIMS_R1 = os.path.join(INPUT_PATH, "tabfact_r1.json")
    # INPUT_PATH_CLAIMS_R2 = os.path.join(INPUT_PATH, "tabfact_r2.json")
    # INPUT_PATH_TABLES = os.path.join(INPUT_PATH, "all_csv")

    # OUTPUT_PATH = os.path.join(INPUT_PATH, "tabfact_dev_formatted.jsonl")
    # OUTPUT_PATH_VAL_R1_IDS = os.path.join(INPUT_PATH, "tabfact_dev_r1_ids.jsonl")
    # OUTPUT_PATH_VAL_R2_IDS = os.path.join(INPUT_PATH, "tabfact_dev_r2_ids.jsonl")


    TABFACT_DEV_IDS = os.path.join(INPUT_PATH, "train_ids.json")
    INPUT_PATH_CLAIMS_R1 = os.path.join(INPUT_PATH, "tabfact_r1.json")
    INPUT_PATH_CLAIMS_R2 = os.path.join(INPUT_PATH, "tabfact_r2.json")
    INPUT_PATH_TABLES = os.path.join(INPUT_PATH, "all_csv")

    OUTPUT_PATH = os.path.join(INPUT_PATH, "tabfact_train_formatted.jsonl")
    OUTPUT_PATH_VAL_R1_IDS = os.path.join(INPUT_PATH, "tabfact_train_r1_ids.jsonl")
    OUTPUT_PATH_VAL_R2_IDS = os.path.join(INPUT_PATH, "tabfact_train_r2_ids.jsonl")

    
    formatted_tabfact = []

    with open(TABFACT_DEV_IDS, "r") as f_in:
        val_ids = json.load(f_in)

    with open(INPUT_PATH_CLAIMS_R1, "r") as f_in:
        content = json.load(f_in)

    with open(INPUT_PATH_CLAIMS_R2, "r") as f_in:
        content2 = json.load(f_in)
        r2_ids = list(content2.keys())
        content.update(content2)

    r2_ids_dev = []
    r1_ids_dev = []
    id_count = 0
    for key, value in content.items():
        if key not in val_ids:
            continue

        claims = value[0]
        labels = value[1]
        title = value[2] # or catption as described in tabfact

        table_path = os.path.join(INPUT_PATH_TABLES, key)

        table = []
        lin_table = []
        curr_buffer = []
        with open(table_path, "r") as f_in:
            table = f_in.readlines()
            headers = table[0].split("#")
            for line in table[1:]:
                cells  = line.split("#")
                assert len(cells) == len(headers)
                for i, cell in enumerate(cells):
                    cell = cell.strip()
                    header = headers[i].strip()
                    curr_buffer.append(header + " is " + cell)
                lin_table.append(" ; ".join(curr_buffer))
                curr_buffer = []

        # lin_table = " </s> ".join(lin_table) v1
        lin_table = "\n".join(lin_table)

        for i, claim in enumerate(claims):
            if key in r2_ids:
                r2_ids_dev.append(id_count)
            else:
                r1_ids_dev.append(id_count)

            dict_cust = {}
            dict_cust["claim"] = claim
            dict_cust["verdict"] = LABEL_MAPPING[labels[i]]
            dict_cust["title"] = title
            dict_cust["id"] = id_count
            dict_cust["challenge"] = "Numerical Reasoning"
            dict_cust["evidence"] = lin_table
            dict_cust["table"] = table
            dict_cust["table_id"] = key
            id_count +=1
            formatted_tabfact.append(dict_cust)
    
    with open(OUTPUT_PATH, "w") as f_out:
        for entry in formatted_tabfact:
            f_out.write("{}\n".format(json.dumps(entry)))

    with open(OUTPUT_PATH_VAL_R1_IDS, "w") as f_out:
        json.dump(r1_ids_dev, f_out)

    with open(OUTPUT_PATH_VAL_R2_IDS, "w") as f_out:
        json.dump(r2_ids_dev, f_out)

               


if __name__ == "__main__":
    main()

    print("Done!")
