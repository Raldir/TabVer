import os
import json
import argparse
import re
import sys
import pandas as pd
from tqdm import tqdm
import traceback
import copy

import torch
from transformers import AutoTokenizer
import transformers


from src.data.table_formatter_feverous import TableFormatterFEVEROUS
from src.data.table_formatter_tabfact import TableFormatterTabFact

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.constants import LABEL_LIST
from src.data.template_formatter import TemplateFormatterLLM

from src.evaluation.evaluate import load_feverous_numerical_challenges, load_tabfact_challenges



class BaselineLLM():

    def __init__(self, dataset, model, seed, evidence_mode, in_context):#, use_chat_templates):
        self.dataset = dataset
        self.model_name = model.split("/")[-1] if len(model.split("/")) == 2 else  model.split("/")[-2] # the latter is custom model so we do not want "checkpoint-60" to be the name
        self.evidence_mode = evidence_mode
        self.in_context = in_context
        self.max_seq_length = 4096
        self.seed = seed

        assert str(self.seed) in model, "Model seed and running seed diffrent, found {} and {}".format(self.seed, model) # Ensure that same model seed is used for this seed 

        torch.manual_seed(seed)

        if dataset == "feverous":
            self.formatter = TableFormatterFEVEROUS(input_path=os.path.join("data", "feverous", "feverous_dev_filled_tables.jsonl"), tab_qa='{}'.format(args.evidence_mode))
            self.num_classes = 3
            self.numerical_challenges = load_feverous_numerical_challenges()
        elif dataset == "tabfact":
            assert "gold_cells" not in args.evidence_mode, "Tabfact does not support cell level evaluation, but found {}".format(evidence_mode)
            self.formatter = TableFormatterTabFact(input_path=os.path.join("data", "tabfact", "tabfact_dev_formatted.jsonl"), tab_qa='{}'.format(args.evidence_mode))
            self.num_classes = 2
            r1_index = os.path.join("data", "tabfact", "tabfact_dev_r1_ids.jsonl")
            r2_index = os.path.join("data", "tabfact", "tabfact_dev_r2_ids.jsonl")
            self.r1_index, self.r2_index = load_tabfact_challenges()
        elif dataset == "feverous_halo":
            self.formatter = TableFormatterFEVEROUS(input_path=os.path.join("data", "feverous_halo", "feverous_dev_filled_tables_halo.jsonl"), tab_qa='{}'.format(args.evidence_mode))
            self.num_classes = 3
            self.numerical_challenges = load_feverous_numerical_challenges()
            
        self.template_formatter = TemplateFormatterLLM(scenario="verdict_prediction")

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        
        if "Meta-Llama-3" in model:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer.pad_token = "[PAD]" 



        if "8x7" in self.model_name or "70" in self.model_name:
            self.pipeline = transformers.AutoModelForCausalLM.from_pretrained(
                model,
                device_map="auto",
                bnb_4bit_compute_dtype=torch.float16,
                load_in_4bit=True)
        else:
            self.pipeline = transformers.AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype=torch.float16,
                device_map="auto")

        # if "llama" in self.config.reader_model_origin.lower() or "deci" in self.config.reader_model_origin.lower() or "mistral" in self.config.reader_model_origin.lower():
        stop = [] #if stop is None else stop
        stop = list(set(stop + ["\n", "Ċ", "ĊĊ", "<0x0A>"])) # In Llama \n is <0x0A>; In OPT \n is Ċ
        # stop = list(set(stop + ["\n", "\n\n\n", "\n\n"])) # In Llama \n is <0x0A>; In OPT \n is Ċ
        self.stop_token_ids = list(set([self.tokenizer.convert_tokens_to_ids(stop_token) for stop_token in stop] + [self.pipeline.config.eos_token_id] + [self.tokenizer.eos_token_id]))
        self.stop_token_ids.remove(self.tokenizer.unk_token_id)


        data = self.formatter.load_data()

        predictions = self.predict(data)

        scores = self.evaluate(predictions, [x["verdict"] for x in data])

        self.write_results(scores, predictions)


    def write_results(self, scores, predictions):
        out_path_raw = os.path.join("exp_out", self.dataset, "baselines", "llm_{}_{}_{}".format(self.model_name, self.evidence_mode, self.in_context), str(self.seed))
        out_path_raw = re.sub(r'seed_\d+', '', out_path_raw) # remove seed from model name since seed is already part of path.
        
        if not os.path.exists(out_path_raw):
            os.makedirs(out_path_raw)

        out_path = os.path.join(out_path_raw, "results.json")
        out_path_preds = os.path.join(out_path_raw, "preds.json")

        with open(out_path, "w") as f_out:
            json.dump(scores, f_out)
        with open(out_path_preds, "w") as f_out:
            json.dump(predictions, f_out)



    def evaluate_single(self, predictions, golds, prefix=""):
        recall = recall_score(predictions, golds, average="macro")
        precision = precision_score(predictions, golds, average="macro")
        f1 = f1_score(predictions, golds, average="macro")
        acc = accuracy_score(predictions, golds)

        scores = {prefix + "accuracy": acc,  prefix + "recall": recall,  prefix + "precision": precision, prefix + "f1": f1}
        return scores

    def evaluate(self, predictions, golds):
        predictions = [LABEL_LIST.index(pred) for pred in predictions]
        golds = [LABEL_LIST.index(gold) for gold in golds]

        scores = self.evaluate_single(predictions, golds)

        if self.dataset == "tabfact":
            predictions_r1 = [pred for i, pred in enumerate(predictions) if i in self.r1_index]
            golds_r1 = [gold for i, gold in enumerate(golds) if i in self.r1_index]

            scores_r1 = self.evaluate_single(predictions_r1, golds_r1, prefix="r1_")
            scores.update(scores_r1)
            # print("SCORES r1", scores_r1)

            predictions_r2 = [pred for i, pred in enumerate(predictions) if i in self.r2_index]
            golds_r2 = [gold for i, gold in enumerate(golds) if i in self.r2_index]

            scores_r2 = self.evaluate_single(predictions_r2, golds_r2, prefix="r2_")
            scores.update(scores_r2)
            # print("SCORES r2", scores_r2)
        
        elif self.dataset == "feverous":
            predictions_numerical = [pred for i, pred in enumerate(predictions) if i in self.numerical_challenges]
            golds_numerical = [gold for i, gold in enumerate(golds) if i in self.numerical_challenges]
            scores_numerical = self.evaluate_single(predictions_numerical, golds_numerical, prefix="numerical_")
            scores.update(scores_numerical)

        print("SCORES all", scores)

        return scores



    def extract_prediction(self, result):
        print(result)
        result = result.split("assistant")[-1].lower()
        result = result.split("[/inst]")[-1].lower()
        result = result.split("label:")[-1].lower()
        result = result.split("<|im_end|>")[0].lower()
        print(result)
        if "supports" in result:
            return "SUPPORTS"
        elif "refutes" in result:
            return "REFUTES"
        elif "not enough info" in result:
            if self.num_classes < 3:
                 return "REFUTES"
            else:
                return "NOT ENOUGH INFO"
        else:
            return "SUPPORTS" # return supports as default

    def predict(self, data):
        predictions = []
        for sample in tqdm(data):
            evidence = sample['table'] # Potentially has multiple tables but ignored for now.
            
            if self.in_context:
                input_prompt = self.template_formatter.apply_template_to_sample(claim=sample["claim"], evidence=evidence, answer=None, num_in_context_samples=3) # 3 in_context examples at most
            else:
                input_prompt = self.template_formatter.apply_template_to_sample(claim=sample["claim"], evidence=evidence, answer=None)

            final_prompt = self.tokenizer(input_prompt, return_tensors="pt", truncation=True, max_length=self.max_seq_length - 1)

            final_prompt = final_prompt.to("cuda")

            try:
                results = self.pipeline.generate(
                    **final_prompt, # Only batch size of 1 supported now
                    do_sample=False,
                    num_beams=2,
                    # remove_invalid_values=True,
                    #top_k=10,
                    temperature=1.0,
                    top_p=1.0,
                    num_return_sequences=1,
                    eos_token_id=self.stop_token_ids, #self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                    max_length=self.max_seq_length,
                    max_new_tokens=10,
                    # force_words_ids=self.force_words_ids
                )
                result = results[0]
                result = self.tokenizer.decode(result)
            except:
                print(traceback.format_exc())
                result = ""
            print("RESULT", result)
            prediction = self.extract_prediction(result)
            print("Prediction", prediction)
            print("---------")
            predictions.append(prediction)

            if (len(predictions) % 1000) == 0:
                scores = self.evaluate(predictions, [x["verdict"] for x in data[:len(predictions)]]
                )
                self.write_results(scores, predictions)
        return predictions



if __name__ == "__main__":
    # python3 -m src.models.baseline_llm --dataset tabfact --model Open-Orca/Mistral-7B-OpenOrca --evidence_mode linearized_cells
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="/path/to/data")
    parser.add_argument("--model", type=str, help="/path/to/data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evidence_mode", type=str, choices=["gold_cells_llm", "all_table_llm", "linearized_all_table","gold_cells_linearized"], help="/path/to/data")
    parser.add_argument('--in_context', action='store_true')
    # parser.add_argument('--use_chat_templates', action='store_true')

    args = parser.parse_args()

    # "Open-Orca/Mistral-7B-OpenOrca"#"mistralai/Mixtral-8x7B-Instruct-v0.1" #"Open-Orca/Mistral-7B-OpenOrca" #"bardsai/jaskier-7b-dpo-v6.1" #"Open-Orca/Mistral-7B-OpenOrca" #bardsai/jaskier-7b-dpo-v6.1 

    BaselineLLM(dataset=args.dataset, model=args.model, evidence_mode=args.evidence_mode, in_context=args.in_context, seed= args.seed)