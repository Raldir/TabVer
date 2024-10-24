import json
import sys
import pandas as pd
from tqdm import tqdm



class TableFormatterTabFact():

    def __init__(self, input_path, tab_qa, num_samples=-1): # num_samples -1 uses all samples
        self.input_path = input_path
        self.tab_qa = tab_qa
        self.limit = num_samples


    def format_tabular_evidence_llm(self, table, page_title):
        table_string = []
        for row in table:
            row_formatted = [x.strip() for x in row.split("#")]
            row_formatted = " | ".join(row_formatted)
            table_string.append(row_formatted)

        table_string = "Table title: {}\n{}".format(page_title, "\n".join(table_string))

        return table_string

    def format_tabular_evidence_table_model(self, table, page_title):
        table_formatted = []
        for i, row in enumerate(table):
            row = [x.strip() for x in row.split("#")]
            table_formatted.append(row)

        table_formatted_dict = {}
        for i, row in enumerate(table_formatted):
            for j, cell in enumerate(row):
                if i==0:
                    table_formatted_dict[cell] = []
                else:
                    table_formatted_dict[table_formatted[0][j]].append(cell)

        table_formatted_dict = {k:v for k,v in table_formatted_dict.items() if v}

        # Filling up the dict to be equal length
        max_values = 0
        for key, value in table_formatted_dict.items():
            max_values = max(max_values, len(value))
        
        for key, value in table_formatted_dict.items():
            if len(value) < max_values:
                table_formatted_dict[key] = value + [" "] * (max_values - len(value))


        # print(table_formatted_dict)

        table_new = pd.DataFrame.from_dict(table_formatted_dict)
        table_new.fillna("",inplace=True)
        return table_new

    def format_tabular_evidence_linearized(self, table, page_title):
        table_formatted = []
        for i, row in enumerate(table):
            row = [x.strip() for x in row.split("#")]
            table_formatted.append(row)

        table_formatted_dict = {}
        for i, row in enumerate(table_formatted):
            for j, cell in enumerate(row):
                if i==0:
                    table_formatted_dict[cell] = []
                else:
                    table_formatted_dict[table_formatted[0][j]].append(cell)

        table_formatted_dict = {k:v for k,v in table_formatted_dict.items() if v}

        # Filling up the dict to be equal length
        max_values = 0
        for key, value in table_formatted_dict.items():
            max_values = max(max_values, len(value))
        
        for key, value in table_formatted_dict.items():
            if len(value) < max_values:
                table_formatted_dict[key] = value + [" "] * (max_values - len(value))
        
        context = []
        # Iterate over each header, and linearize header and all of its cells
        for key, values in table_formatted_dict.items():
            lin = ""
            for i, value in enumerate(values):
                lin += key + " is " + value
                if i + 1 < len(values):
                    lin += " ; "
                else:
                    lin += "."
            context.append(lin)

        context = " ".join(context)
        return context


    def load_data(self):
        tabular_evidence = {}
        tabular_evidence_titles = {}
        with open(self.input_path, "r") as f_in:
            limit = self.limit * 5 if self.limit != -1 else None
            for line in tqdm(f_in.readlines()[:limit]):
                content = json.loads(line)
                # print(content)

                if "llm" in self.tab_qa:
                    evidence_tab = content["table"]
                    evidence_tit = content["title"]
                    evidence_tab = self.format_tabular_evidence_llm(evidence_tab, evidence_tit)
                    tabular_evidence[content["id"]] = evidence_tab
                    tabular_evidence_titles[content["id"]] = evidence_tit
                elif "linearized" in self.tab_qa:
                    evidence_tab = content["table"]
                    evidence_tit = content["title"]
                    evidence_tab = self.format_tabular_evidence_linearized(evidence_tab, evidence_tit)
                    tabular_evidence[content["id"]] = [evidence_tab][0]
                    tabular_evidence_titles[content["id"]] = [evidence_tit][0]
                elif "table_model" in self.tab_qa:
                    evidence =  content["table"]
                    evidence_tit = content["title"]
                    evidence_tab = self.format_tabular_evidence_table_model(evidence, evidence_tit)
                    tabular_evidence[content["id"]] = [evidence_tab][0]
                    tabular_evidence_titles[content["id"]] = [content["title"]][0]      
                else:
                    print("Error. tab_qa type not recognized...")
                    sys.exit()

        contents = []

        
        with open(self.input_path, "r") as f_in:
            for line in f_in.readlines():
                if self.limit != -1 and len(contents) >= self.limit:
                    break
                content = json.loads(line)
                keys_match = content["id"]  # Important to use normalized subclaim id here to find the right table.
                # keys_match = keys_match[0] # take first match, throws error if cannot be found.
                content["table"] = tabular_evidence[keys_match]
                content["title"] = tabular_evidence_titles[keys_match]
                contents.append(content)
        
        return contents
