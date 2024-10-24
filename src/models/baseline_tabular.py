import argparse
import json
import os
import re

import torch
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BartForSequenceClassification 
)
import torch

from src.utils.util import ROOT_DIR
from src.constants import TWO_CLASSES_DATASETS
from src.data.table_formatter_feverous import TableFormatterFEVEROUS
from src.data.table_formatter_tabfact import TableFormatterTabFact

from src.evaluation.evaluate import load_feverous_numerical_challenges, load_tabfact_challenges

MAP_INDEX_TO_VERDICT = None
DIRECTORY = None
CONFIG = None
DEV_DATA_IDS = []
LABELS = []

TABULAR_MODELS = ["tapex", "tapas"]
LINEARIZED_MODELS = ["deberta", "pasta", "roberta"]

class FEVEROUSDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, use_labels=True):
        self.encodings = encodings
        self.labels = labels
        self.use_labels = use_labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # print(idx)
        if self.use_labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def process_data(claim_verdict_list, map_verdict_to_index):
    text = [x["claim"] for x in claim_verdict_list]
    evidence = [x["table"] for x in claim_verdict_list]

    if CONFIG["dataset"] in TWO_CLASSES_DATASETS: # Training when model only has two classes
        labels = [map_verdict_to_index[x["verdict"]] if x["verdict"] != "NOT ENOUGH INFO" else map_verdict_to_index["REFUTES"] for x in claim_verdict_list]
    else:
        labels = [map_verdict_to_index[x["verdict"]] for x in claim_verdict_list]  # get value from enum
    return text, evidence, labels


def compute_metrics(pred):
    labels = pred.label_ids
    output_path = os.path.join(DIRECTORY, "probabilities")

    if "tapex" in CONFIG["model_name"]:
        pred.predictions = pred.predictions[0] # for BART0

    with open(output_path + ".json", "w") as f_out:
        for i in range(len(pred.predictions)):
            curr_pred = torch.from_numpy(pred.predictions[i])
            curr_pred = torch.nn.functional.softmax(curr_pred)
            curr_pred =  curr_pred.tolist()
            prob_map = {MAP_INDEX_TO_VERDICT[x]: curr_pred[x] for x in range(len(curr_pred))}
            if CONFIG["dataset"] in TWO_CLASSES_DATASETS and "NOT ENOUGH INFO" in prob_map:
                prob_map["REFUTES"] = prob_map["REFUTES"] + prob_map["NOT ENOUGH INFO"]
                del prob_map["NOT ENOUGH INFO"]
            f_out.write("{}\t{}\n".format(DEV_DATA_IDS[i], json.dumps(prob_map)))

    preds = pred.predictions.argmax(-1)
    if CONFIG["dataset"] in TWO_CLASSES_DATASETS:
        nei_index = CONFIG["map_verdict_to_index"]["NOT ENOUGH INFO"]
        refuted_index = CONFIG["map_verdict_to_index"]["REFUTES"]
        preds = [x if x != nei_index else refuted_index for x in preds]
    precision, recall, f1, _ = precision_recall_fscore_support(LABELS, preds, average="macro")
    acc = accuracy_score(LABELS, preds)
    class_labels = []
    for key in sorted(MAP_INDEX_TO_VERDICT):
        if CONFIG["dataset"] in TWO_CLASSES_DATASETS:
            if MAP_INDEX_TO_VERDICT[key] == "NOT ENOUGH INFO":
                continue
        class_labels.append(MAP_INDEX_TO_VERDICT[key])

    class_rep = classification_report(
        LABELS, preds, target_names=class_labels, labels = list(range(len(class_labels))), output_dict=True
    )

    overall_scores = {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall, "class_rep": class_rep}

    if CONFIG["dataset"] == "tabfact":
        set1_ids, set2_ids = load_tabfact_challenges()

        for i, curr_set in enumerate([set1_ids, set2_ids]):
            preds_set1 = [x for i, x in enumerate(preds) if DEV_DATA_IDS[i] in curr_set]
            gold_set1 = [x for i, x in enumerate(LABELS) if DEV_DATA_IDS[i] in curr_set]

            precision_num, recall_num, f1_num, _ = precision_recall_fscore_support(gold_set1, preds_set1, average="macro")
            acc_num = accuracy_score(gold_set1, preds_set1)

            class_rep_num = classification_report(
                gold_set1, preds_set1, target_names=class_labels, labels = list(range(len(class_labels))), output_dict=True
            )

            set_dict = {"accuracy_set{}".format(i + 1): acc_num, "f1_set{}".format(i+1): f1_num, "recall_set{}".format(i+1): recall_num, "precision_set{}".format(i+1): precision_num, "class_rep_set{}".format(i+1): class_rep_num}
            overall_scores.update(set_dict)
    elif CONFIG["dataset"] == "feverous":
        challenges_ids = load_feverous_numerical_challenges()

        preds_numerical = [x for i, x in enumerate(preds) if DEV_DATA_IDS[i] in challenges_ids]
        gold_numerical = [x for i, x in enumerate(LABELS) if DEV_DATA_IDS[i] in challenges_ids]

        precision_num, recall_num, f1_num, _ = precision_recall_fscore_support(gold_numerical, preds_numerical, average="micro")
        acc_num = accuracy_score(gold_numerical, preds_numerical)


        class_rep_num = classification_report(
            gold_numerical, preds_numerical, target_names=class_labels, labels = list(range(len(class_labels))), output_dict=True
        )
        set_dict = {"accuracy_num": acc_num, "f1_num": f1_num, "precision_num": precision_num, "recall_num": recall_num, "class_rep_num": class_rep_num}
        overall_scores.update(set_dict)

    print(overall_scores)


    return overall_scores


def model_trainer(train_dataset, test_dataset, config):
    if "tapex" in config["model_name"]:
        model = BartForSequenceClassification.from_pretrained(
            config["model_name"], num_labels=config["num_labels"], return_dict=True
        ).to(
            config["device"]
        ) 
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            config["model_name"], num_labels=config["num_labels"], return_dict=True
        ).to(
            config["device"]
        ) 

    training_args = TrainingArguments(
        output_dir=config["model_path"],  # output directory
        num_train_epochs= config["num_train_epochs"],  # total # of training epochs
        per_device_train_batch_size=config["per_device_train_batch_size"],  # batch size per device during training
        per_device_eval_batch_size=config["per_device_eval_batch_size"],  # batch size for evaluation
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        warmup_steps=config["warmup_steps"],  # number of warmup steps for learning rate scheduler
        weight_decay=config["weight_decay"],  # strength of weight decay
        logging_dir=os.path.join(config["model_path"], "logs"),  # directory for storing logs
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],  # 1200,
        learning_rate=config["learning_rate"],
        seed=config["seed"]
        # save_strategy='epoch'
    )

    if test_dataset != None:
        trainer = Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            eval_dataset=test_dataset,  # evaluation dataset
            compute_metrics=compute_metrics,
        )
    else:
        trainer = Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            compute_metrics=compute_metrics,
        )
    return trainer, model


def tokenize_input(text, evidence, tokenizer, config):
    if "table_model" in config["tab_qa_mode"]:
       text_tok = tokenizer(table=evidence, queries=text, padding="max_length", max_length=512, truncation=True)
    else:
        input_text = " </s> ".join([text, evidence])
        text_tok = tokenizer(input_text, padding="max_length", max_length=512, truncation=True)

    return text_tok

def claim_evidence_predictor(annotations_train, annotations_dev, config):
    global MAP_INDEX_TO_VERDICT
    global CONFIG
    global DIRECTORY
    global LABELS

    CONFIG = config
    MAP_INDEX_TO_VERDICT = {y:x for x,y in config["map_verdict_to_index"].items()}
    print(annotations_train[0])
    print(annotations_dev[0])

    DIRECTORY = os.path.join(ROOT_DIR, "exp_out", config["dataset"], "baselines", config["exp_name"] + "_" + str(config["num_samples"]) + "_stratified_" + str(config["stratified"]) + "_full_supervision_" + str(config["full_supervision"]) + "_zero_shot_" + str(config["zero_shot"]) + "_gold_cells_only_" + str(config["highlighted_cells_only"]), "seed_" + str(CONFIG["seed"]))

    if not os.path.exists(DIRECTORY):
        os.makedirs(DIRECTORY)

    text_train, evidence_train, labels_train = process_data(annotations_train, config["map_verdict_to_index"])

    text_test, evidence_test, labels_test = process_data(annotations_dev, config["map_verdict_to_index"])

    if config["num_labels"] == 2:
        LABELS = [config["map_verdict_to_index"][x] for x in LABELS]
    else:
        LABELS = labels_test

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    text_train_toks = {}
    text_test_toks = {}


    for i, ev_train in enumerate(tqdm(evidence_train)):
        text_train_tok = tokenize_input(text=text_train[i], evidence=ev_train, tokenizer=tokenizer, config=config)
        for key, value in text_train_tok.items():
            if key not in text_train_toks:
                text_train_toks[key] = []
            text_train_toks[key].append(value)

    for i, ev_test in enumerate(tqdm(evidence_test)):
        text_test_tok = tokenize_input(text=text_test[i], evidence=ev_test, tokenizer=tokenizer, config=config)
        for key, value in text_test_tok.items():
            if key not in text_test_toks:
                text_test_toks[key] = []
            text_test_toks[key].append(value)

    train_dataset = FEVEROUSDataset(text_train_toks, labels_train)
    # text_test = tokenizer(text_test, padding=True, truncation=True)
    test_dataset = FEVEROUSDataset(text_test_toks, labels_test)

    trainer, model = model_trainer(train_dataset, test_dataset, config)
    trainer.train()
    scores = trainer.evaluate()
    output_path = os.path.join(DIRECTORY, "dev_scores")
    with open(output_path + ".json", "w") as f_out:
        json.dump(scores, f_out)

    print(scores["eval_class_rep"])


def process_call_value(value):
    match = re.findall("\[\[(.+?)\]\]", value)
    if len(match) > 0:
        match = match[0]
        if "|" in match:
            value = match.split("|")[1].strip()
        else:
            value = match.strip()
    return value.replace(":", "").replace("\n", " ")


def format_tabular_evidence(evidence, page_title, selected_cells, use_highlighted_cells_only):
    table = {}
    for row in evidence:
        for cell in row:
            cell_id = page_title + "_" + cell["id"]
            # print(cell_id)
            if cell_id not in selected_cells and use_highlighted_cells_only:
                continue
            elif "header_cell" in cell["id"]:
                continue
            else:
                for header in  cell["context"]:
                    header = process_call_value(header)
                    value = process_call_value(cell["value"])
                    if header in table:
                        table[header].append(value)
                    else:
                        table[header] = [value]
    return table

def setup_config(dataset, config_path, sample_size, stratified, zero_shot, seed, full_supervision, highlighted_cells_only): 
    with open(os.path.join("configs", "baselines", config_path + ".json"), "r") as f:
        config = json.load(f)
    
    # Since only fever training data is used for now
    sample_text = str(sample_size)
    if stratified:
        sample_text = sample_text + "_stratified"


    if zero_shot:
        config["num_train_epochs"] = 1
    config["zero_shot"] = zero_shot
    config["dataset"] = dataset
    config["num_samples"] = sample_size
    config["stratified"] = stratified
    config["seed"] = seed
    config["full_supervision"] = full_supervision
    config["highlighted_cells_only"] = highlighted_cells_only
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["device"] = device

    tab_qa_mode = ""

    if highlighted_cells_only:
        tab_qa_mode += "default"
    else:
        tab_qa_mode += "all_table"
    
    if any([x in config["model_name"] for x in TABULAR_MODELS]):
        tab_qa_mode += "_table_model"
    else:
        tab_qa_mode += "_linearized"
    
    tab_qa_mode += "_baseline"

    config["tab_qa_mode"] = tab_qa_mode

    print(config)

    return config


def train_verdict_predictor(config) -> None:
    global DEV_DATA_IDS
    global LABELS

    if config["zero_shot"]:
        config["num_samples"] = 1
    elif config["full_supervision"]:
        config["num_samples"] = -1

    data_loader_train = TableFormatterFEVEROUS(input_path=os.path.join("data", "feverous", "feverous_train_filled_tables.jsonl"), tab_qa=config["tab_qa_mode"], num_samples = config["num_samples"])

    # data_loader_train = TableFormatterTabFact(input_path=os.path.join("data", "tabfact", "tabfact_train_formatted.jsonl"), tab_qa=config["tab_qa_mode"], num_samples = config["num_samples"]) # Can also be trained via TabFact


    if config["dataset"] == "feverous":
        data_loader_dev = TableFormatterFEVEROUS(input_path=os.path.join("data", "feverous", "feverous_dev_filled_tables.jsonl"), tab_qa=config["tab_qa_mode"])
    elif config["dataset"] == "tabfact":
        data_loader_dev = TableFormatterTabFact(input_path=os.path.join("data", "tabfact", "tabfact_dev_formatted.jsonl"), tab_qa=config["tab_qa_mode"]) #12850 / 2

    annotations_train = data_loader_train.load_data()
    annotations_dev = data_loader_dev.load_data()

    for content in annotations_dev:
        LABELS.append(content["verdict"])
        DEV_DATA_IDS.append(content["id"])

    claim_evidence_predictor(annotations_train, annotations_dev, config)


### EXAMPLE EXECUTION:
### python3 -m src.models.baseline_tabular --dataset tabfact --config_path tapas_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="/path/to/data")
    parser.add_argument("--seed", type=int, default=42, help="/path/to/data")
    parser.add_argument("--full_supervision", action='store_true', help="/path/to/data")
    parser.add_argument("--sample_size", type=int, default=32, help="/path/to/data")
    parser.add_argument("--stratified", action='store_true', help="/path/to/data")
    parser.add_argument("--config_path", type=str, required=True, help="/path/to/data")
    parser.add_argument("--zero_shot", action='store_true', help="/path/to/data")
    parser.add_argument("--highlighted_cells_only", action='store_true', help="/path/to/data")

    args = parser.parse_args()

    config = setup_config(args.dataset, args.config_path, args.sample_size, args.stratified, args.zero_shot, args.seed, args.full_supervision, args.highlighted_cells_only)

    train_verdict_predictor(config)


if __name__ == "__main__":
    main()