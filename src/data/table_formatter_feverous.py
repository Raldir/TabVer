import os
import json
import re
import sys
import pandas as pd
from tqdm import tqdm


from src.utils.util import ROOT_DIR


class TableFormatterFEVEROUS():

    def __init__(self, input_path, tab_qa, num_samples=-1): # num_samples -1 uses all samples
        self.input_path = input_path
        self.tab_qa = tab_qa
        self.limit = num_samples


    def process_call_value(self, value, max_length=500):

        def process_match(match):
            parts = match.group(1).split('|')
            return parts[1] if len(parts) > 1 else parts[0]


        value = re.sub("\[\[(.+?)\]\]", process_match, value)
        value = value.replace(":", "").replace("\n", " ").strip()

        return value[:max_length]


    def normalize_table_representation(self, table):
        # Removes row and column spans to simplify representation
        max_row = 0
        max_column = 0

        # Compute the maximum row and column span
        for row in table:
            column_index = 0
            for cell in row:
                column_index += int(cell.get("column_span", 1))
            max_column = max(max_column, column_index)
        
        max_row = sum([int(table[j][0].get("row_span", 1)) for j in range(len(table))])

        # print("dimensions", max_column, max_row)
        expanded_table = [[None] * max_column for _ in range(max_row)]

        # Populate the expanded table
        for i, row in enumerate(table):
            row_index = 0
            for cell in row:
                column_span = int(cell.get("column_span", 1))
                row_span = int(cell.get("row_span", 1))

                # Insert the cell into the expanded table according to its row and column spans
                for _ in range(row_span):
                    for j in range(column_span):
                        if expanded_table[i + _][row_index + j] is None:
                            expanded_table[i + _][row_index + j] = cell.copy()
                        else: # If expanded table is not None (empty) at this position, shift to right until it is None.
                            for k in range(1, len(expanded_table[i + _]) - (j + row_index)):
                                if expanded_table[i + _][row_index + j + k] is None:
                                    expanded_table[i + _][row_index + j + k] = cell.copy()
                                    break
                row_index += column_span
        
        # The max_row value is an upper bound so remove any empty rows.
        expanded_table_new = []
        for i, row in enumerate(expanded_table):
            if not all(x is None for x in row):
                expanded_table_new.append(row)

        expanded_table = expanded_table_new

        return expanded_table


    def format_tabular_evidence_llm(self, table, page_title, selected_cells):
        table = self.normalize_table_representation(table)

        table_string = []

        for row in table:
            row_formatted = []
            for cell in row:
                if cell == None:
                    row_formatted.append("XX")
                    continue
                cell_id = page_title + "_" + cell["id"]
                if (cell_id not in selected_cells and "all_table" not in self.tab_qa):
                    row_formatted.append("XX")
                    continue
                row_formatted.append(self.process_call_value(cell["value"]))
            row_formatted = " | ".join(row_formatted)
            table_string.append(row_formatted)
        table_string = "Table title: {}\n{}".format(page_title, "\n".join(table_string))

        return table_string[:20000] # Arbitary cutoff if tables get too long


    def format_tabular_evidence_linearized(self, table, page_title, selected_cells):
        formatted_table = {}
        # print(table)
        for row in table:
            for cell in row:
                # print(cell)
                cell_id = page_title + "_" + cell["id"]
                if cell_id not in selected_cells and "all_table" not in self.tab_qa:
                    continue
                if "header_cell" in cell["id"] and not all(["header_cell" in  x for x in selected_cells]): # Only consider header_cell as actual cell if non other cell has been annotated
                    continue
                else:
                    if "context" not in cell or not cell["context"]: # in case the cell does not have any context, put cells of same row/column under same temporary key.
                        row_num = cell["id"].split("_")[-2]
                        col_num = cell["id"].split("_")[-1]
                        value = self.process_call_value(cell["value"])
                        if "TEMP" + row_num in formatted_table:
                            formatted_table["TEMP" + row_num].append(value)
                        elif "TEMP" + col_num in table:
                            formatted_table[  "TEMP" +  col_num].append(value)
                        else:
                            formatted_table["TEMP" + row_num] = [value]
                            formatted_table["TEMP" + col_num] = [value]
                    else:
                        for header in  cell["context"]:
                            header = self.process_call_value(header)
                            value = self.process_call_value(cell["value"])
                            # print(value)
                            if page_title + " " + header in table:
                                formatted_table[page_title + " " + header].append(value)
                            else:
                                formatted_table[page_title + " " + header] = [value]
        
        # Adjust table so that the temporary header is replaced by the first element in the list
        table_new = {}
        values_made_into_headers = set([])
        for key, value in formatted_table.items():
            if len(value) == 0:
                continue
            if value[0] in values_made_into_headers and len(value) == 1:
                continue
            elif "TEMP" in key:
                table_new[page_title + " " + value[0]] = value[1:]
                values_made_into_headers.add(value[0])
            else:
                table_new[key] = value # no need to add page_title again since already in key that do not contain TEMP.
        
        context = []
        # Iterate over each header, and linearize header and all of its cells
        for key, values in table_new.items():
            lin = ""
            for i, value in enumerate(values):
                lin += key.strip() + " is " + value.strip()
                if i + 1 < len(values):
                    lin += " ; "
                else:
                    lin += "."

            context.append(lin)

        context = " ".join(context).strip()
        return context[:20000] # Arbitary cutoff if tables get too long

    def format_tabular_evidence_table_model(self, table, page_title, selected_cells):
        table = self.normalize_table_representation(table)

        table_string = []

        for row in table:
            row_formatted = []
            for cell in row:
                if cell == None:
                    row_formatted.append("XX")
                    continue
                cell_id = page_title + "_" + cell["id"]
                if (cell_id not in selected_cells and "all_table" not in self.tab_qa):
                    row_formatted.append("XX")
                    continue
                row_formatted.append(self.process_call_value(cell["value"]))
            table_string.append(row_formatted)
        
        table_formatted_dict = {}

        transposed_table = list(zip(*table_string))
        for row in transposed_table[:20]: # Only consider 20 rows, cutoff after that
            table_formatted_dict["{}: {}".format(page_title, row[0])] = row[1:20] if len(row) > 1 else [row[0]]
            
        table_new = pd.DataFrame.from_dict(table_formatted_dict)
        table_new.fillna("",inplace=True)
        return table_new

    def load_data(self):
        tabular_evidence = {}
        tabular_evidence_titles = {}
        with open(self.input_path, "r") as f_in:
            limit = self.limit * 5 if self.limit != -1 else None
            for line in tqdm(f_in.readlines()[:limit]): # adding buffer since training data claims do not have to be taken from top always
                content = json.loads(line)
                evidence_tables = []
                evidence_titles = []
                if "llm" in self.tab_qa:
                    for table in content["table_evidence"]:
                        evidence_tab = table["content"]
                        evidence_page = "{} {}".format(table["id"].split("_")[0], " ".join(table["context"])) # Title + Table context (i.e. subsection title)
                        evidence_tab = self.format_tabular_evidence_llm(evidence_tab, evidence_page, content["selected_cells"])
                        evidence_tables.append(evidence_tab)
                        evidence_titles.append(evidence_page)
                    tabular_evidence[content["id"]] = evidence_tables[0]
                    tabular_evidence_titles[content["id"]] = evidence_titles[0]
                elif "linearized" in self.tab_qa:
                    for table in content["table_evidence"]:
                        evidence_tab = table["content"]
                        evidence_page = "{} {}".format(table["id"].split("_")[0], " ".join(table["context"])) # Title + Table context (i.e. subsection title)
                        evidence_tab = self.format_tabular_evidence_linearized(evidence_tab, evidence_page, content["selected_cells"])
                        evidence_tables.append(evidence_tab)
                        evidence_titles.append(evidence_page)
                    tabular_evidence[content["id"]] = evidence_tables[0] # It will be single sequence so use first element
                    tabular_evidence_titles[content["id"]] = evidence_titles[0]
                elif "table_model" in self.tab_qa:
                    for table in content["table_evidence"]:
                        evidence = table["content"]
                        evidence_page = "{} {}".format(table["id"].split("_")[0], " ".join(table["context"])) # Title + Table context (i.e. subsection title)
                        evidence = self.format_tabular_evidence_table_model(evidence, evidence_page, content["selected_cells"])
                        evidence_tables.append(evidence)
                        evidence_titles.append(evidence_page)

                    selected_table = 0
                    for tab in evidence_tables:
                        if tab.empty:
                            selected_table+=1   
                    selected_table = min(selected_table, len(evidence_tables) -1)

                    tabular_evidence[content["id"]] = evidence_tables[selected_table] # Simply ignore multiple evidence tables for now....
                    tabular_evidence_titles[content["id"]] = evidence_titles[selected_table]
                else:
                    print("Error. tab_qa type not recognized...")
                    sys.exit()

        contents = []
        already_seen_ids = set([])

        # TODO: Remove. Obsolete code.
        with open(self.input_path, "r") as f_in:
            for line in f_in.readlines():
                if self.limit != -1 and len(contents) >= self.limit:
                    break
                content = json.loads(line)
                keys_match = [x for x in list(tabular_evidence.keys()) if str(x) == str(content["id"])]
                keys_match = keys_match[0] # take first match, throws error if cannot be found.
                if keys_match in already_seen_ids and "baseline" in self.tab_qa: # In training data multiple proofs have same ID. Want to only unique id data.
                    continue
                already_seen_ids.add(keys_match)
                content["table"] = tabular_evidence[keys_match]
                content["title"] = tabular_evidence_titles[keys_match]
                contents.append(content)
        
        return contents

if __name__ == "__main__":
    highlighted_cells_only = False
    sample_size = 20
    data_loader_train = TableFormatterFEVEROUS(input_path=os.path.join("data", "feverous", "feverous_dev_filled_tables.jsonl"), tab_qa="baseline_all_table_linearized" if not highlighted_cells_only else "baseline_default", num_samples = sample_size)

    annotations_train = data_loader_train.load_data()
