import os
import json
import gc

import torch

from src.data.table_formatter_feverous import TableFormatterFEVEROUS
from src.data.table_formatter_tabfact import TableFormatterTabFact
from src.data.qg_and_qa_llm import QuestionGeneratorLLM, QuestionAnsweringLLM
from src.utils.util import ROOT_DIR

class ArithExpressionGenerator(object):
    
    def __init__(self, dataset, input_path, save_name, use_tab_qa, qg_model, qa_model, permissable_operators, num_samples=-1, overwrite=False):  
        self.dataset = dataset
        self.input_path = input_path
        self.use_tab_qa = use_tab_qa
        self.qg_model = qg_model
        self.qa_model = qa_model
        self.permissable_operators = permissable_operators
        self.overwrite = overwrite

        input_path_str = input_path.split("/")[-1].split(".")[0]
        qg_model_str = qg_model.split("/")[-1] if len(qg_model.split("/")) == 2 else  qg_model.split("/")[-2] # the latter is custom model so we do not want "checkpoint-60" to be the name
        qa_model_str = qa_model.split("/")[-1] if len(qa_model.split("/")) == 2 else  qa_model.split("/")[-2] # the latter is custom model so we do not want "checkpoint-60" to be the name

        self.output_path_qa = os.path.join(ROOT_DIR, "data", dataset, "arithmetic_expressions", "{}.jsonl".format(save_name))
        self.output_path_qg = os.path.join(ROOT_DIR, "data", dataset, "arithmetic_expressions", "question_generation", "question_generation_{}.jsonl".format(save_name)).replace(qa_model_str, qg_model_str)

        if dataset == "feverous":
            self.tableformatter = TableFormatterFEVEROUS(input_path, use_tab_qa, num_samples = num_samples)
        elif dataset == "tabfact":
            self.tableformatter = TableFormatterTabFact(input_path, use_tab_qa, num_samples = num_samples)

    def generate_arith_exps(self):
        if os.path.exists(self.output_path_qa):
            data = []
            with open(self.output_path_qa, "r") as f_in:
                for line in f_in.readlines():
                    content = json.loads(line)
                    data.append(content)
        else:
            data = self.tableformatter.load_data()

            llm_qg = QuestionGeneratorLLM(self.qg_model, self.output_path_qg)
            data = llm_qg.question_generation_numerical(data)

            del llm_qg
            gc.collect()
            torch.cuda.empty_cache()
            
            # In case the QA prcoess quits unexpect. reload where needed manually...
            # with open(self.output_path_qg, "r") as f_in:
            #     data = []
            #     for line in f_in.readlines():
            #         data.append(json.loads(line))
            #     data = data[1001:] #data[556:]

            llm_qa = QuestionAnsweringLLM(self.qa_model, self.output_path_qa, self.permissable_operators)
            data = llm_qa.question_answering_numerical(data)
        
        contents = {}
        for entry in data:
            contents[entry["id"]] = entry

        return contents 