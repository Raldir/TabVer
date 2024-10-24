import os
import json
import datetime

import datasets
import pandas as pd
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from torch import nn
import torch
from src.constants import HF_ACCESS_TOKEN
from peft import LoraConfig, get_peft_model

from src.data.table_formatter_feverous import TableFormatterFEVEROUS
from src.data.template_formatter import TemplateFormatterLLM

def load_verdict_prediction_data(input_path, args):
    num_samples = 64
    if args.evidence_mode == "all_table_llm":
        data_loader = TableFormatterFEVEROUS(input_path=input_path, tab_qa="all_table_llm_baseline", num_samples=num_samples)
    elif args.evidence_mode =="linearized_all_table":
        data_loader = TableFormatterFEVEROUS(input_path=input_path, tab_qa="linearized_all_table_baseline", num_samples=num_samples)
    elif args.evidence_mode =="gold_cells_linearized":
        data_loader = TableFormatterFEVEROUS(input_path=input_path, tab_qa="gold_cells_linearized_baseline", num_samples=num_samples)
    else:
        data_loader = TableFormatterFEVEROUS(input_path=input_path, tab_qa="gold_cells_llm_baseline", num_samples=num_samples)

    template_formatter = TemplateFormatterLLM(scenario="verdict_prediction")

    content = data_loader.load_data()
    content_dict = []
    for item in content:
        curr_dict = {"text": [], "label": item["verdict"]}
        input_prompt = template_formatter.apply_template_to_sample(claim = item["claim"], evidence = item["table"], answer = item["verdict"], num_in_context_samples=0)
        print(input_prompt)
        curr_dict["text"] = input_prompt
        content_dict.append(curr_dict)
    
    assert len(content_dict) == num_samples, "Expected {} training samples, found {}".format(num_samples, len(content_dict))

    model_out_path = "models/{}_{}_{}_train_all_prompt_{}".format(args.mode, args.model.split("/")[-1], args.evidence_mode, args.train_on_entire_prompt)
    
    return content_dict, template_formatter, model_out_path

def load_decomposition_data(input_path, args):
    input_path_subclaims =  os.path.join("data", "feverous", "handcrafted_training_data", "training_data_combined.json")

    with open(input_path_subclaims, "r") as f_in:
        content = json.load(f_in)

    template_formatter = TemplateFormatterLLM(scenario="claim_decomposition")

    content_dict = []
    for item in content:
        curr_dict = {"text": []}
        input_prompt = template_formatter.apply_template_to_sample(claim = item["claim"], subclaims = item["subclaims"], num_in_context_samples=0) + "\n" # adding new line at end to teach model when to stop generating
        print(input_prompt)
        curr_dict["text"] = input_prompt
        content_dict.append(curr_dict)
    
    print("Num samples: {}".format(len(content_dict)))

    model_out_path = "models/{}_{}_train_all_prompt_{}".format(args.mode, args.model.split("/")[-1], args.train_on_entire_prompt)
    
    return content_dict, template_formatter, model_out_path

def load_question_generation_data(input_path, args):
    input_path_subclaims =  os.path.join("data", "feverous",  "handcrafted_training_data", "training_data_combined.json")
    with open(input_path_subclaims, "r") as f_in:
        content = json.load(f_in)

    template_formatter = TemplateFormatterLLM(scenario="question_generation")

    content_dict = []
    for item in content:
        for i, subclaim in enumerate(item["subclaims"]):
            curr_dict = {"text": []}
            print(subclaim)
            input_prompt = template_formatter.apply_template_to_sample(claim = subclaim, questions = item["questions"][i], answers = item["answers_claim"][i], num_in_context_samples=0) + "\n" # adding new line at end to teach model when to stop generating
            print(input_prompt)
            curr_dict["text"] = input_prompt
            content_dict.append(curr_dict)
    
    print("Num samples: {}".format(len(content_dict)))
    model_out_path = "models/{}_{}_train_all_prompt_{}".format(args.mode, args.model.split("/")[-1], args.train_on_entire_prompt)
    
    return content_dict, template_formatter, model_out_path



def load_question_answering_data(input_path, args):
    num_samples = 1024
    if args.evidence_mode == "all_table_llm":
        data_loader = TableFormatterFEVEROUS(input_path=input_path, tab_qa="all_table_llm_baseline", num_samples=num_samples)
    elif args.evidence_mode =="linearized_all_table":
        data_loader = TableFormatterFEVEROUS(input_path=input_path, tab_qa="linearized_all_table_baseline", num_samples=num_samples)
    elif args.evidence_mode =="gold_cells_linearized":
        data_loader = TableFormatterFEVEROUS(input_path=input_path, tab_qa="gold_cells_linearized_baseline", num_samples=num_samples)
    else:
        data_loader = TableFormatterFEVEROUS(input_path=input_path, tab_qa="gold_cells_llm_baseline", num_samples=num_samples)
    content_tables = data_loader.load_data()
    content_tables_dict = {x["id"]:x for x in content_tables}

    input_path_subclaims =  os.path.join("data", "feverous",  "handcrafted_training_data", "training_data_combined.json")
    with open(input_path_subclaims, "r") as f_in:
        content = json.load(f_in)


    template_formatter = TemplateFormatterLLM(scenario="question_answering", allowed_functions=args.qa_functions)

    content_dict = []
    for item in content:
        for i, questions in enumerate(item["questions"]):
            for j, question in enumerate(questions):
                curr_dict = {"text": []}
                evidence_table = content_tables_dict[item["qid"]]["table"]
                input_prompt = template_formatter.apply_template_to_sample(question = question, evidence_table = evidence_table, evidence_text = item["evidence"][i][j], computation = item["computation"][i], arith_exp = item["arithmetic_expressions"][i][j], num_in_context_samples=0) + "\n" # adding new line at end to teach model when to stop generating
                print(input_prompt)
                curr_dict["text"] = input_prompt
                content_dict.append(curr_dict)
    
    print("Num samples: {}".format(len(content_dict)))

    model_out_path = "models/{}_{}_{}_{}_train_all_prompt_{}".format(args.mode, args.model.split("/")[-1], args.evidence_mode, args.qa_functions, args.train_on_entire_prompt)
    
    return content_dict, template_formatter, model_out_path



def main(args):
    # dataset = load_dataset("imdb", split="train")
    input_path =  os.path.join("data", "feverous", "feverous_train_filled_tables.jsonl")

    if args.mode == "verdict_prediction":
        content_dict, template_formatter, model_out_path = load_verdict_prediction_data(input_path, args)
    elif args.mode == "claim_decomposition":
        content_dict, template_formatter, model_out_path = load_decomposition_data(input_path, args)
    elif args.mode == "question_generation":
        content_dict, template_formatter, model_out_path = load_question_generation_data(input_path, args)
    elif args.mode == "question_answering":
        content_dict, template_formatter, model_out_path = load_question_answering_data(input_path, args)
    
        
    

    dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=content_dict))

    tokenizer = AutoTokenizer.from_pretrained(args.model, device_map='auto',  torch_dtype=torch.float16,token=HF_ACCESS_TOKEN,)#load_in_8bit=True,
    if "Meta-Llama-3" in args.model:
        tokenizer.pad_token = "<|reserved_special_token_250|>" 
        tokenizer.pad_token_id = 128255
    else:
        tokenizer.pad_token = "[PAD]" 

    start_index = 1 if "Meta-Llama-3" in args.model else 2
    response_template_ids = tokenizer.encode(template_formatter.response_template, add_special_tokens=False)[start_index:]

    print(response_template_ids)

    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    current_timestamp = datetime.datetime.now()
    formatted_timestamp = current_timestamp.strftime("%Y-%m-%d-%H-%M")

    training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing =True,
    max_grad_norm= 0.3,
    num_train_epochs=10, #10, #50, 
    # num_training_steps=200,
    learning_rate=2e-4,
    # bf16=True,
    save_total_limit=2,
    save_steps=10,
    logging_steps=10,
    output_dir="{}_{}_seed_{}".format(model_out_path, formatted_timestamp, args.seed),
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    seed=args.seed
    )

    model = AutoModelForCausalLM.from_pretrained(args.model, device_map='auto', torch_dtype=torch.float16, token=HF_ACCESS_TOKEN)

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules= ["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )


    trainer = SFTTrainer(
        model,
        train_dataset=dataset,
        dataset_text_field="text",
        data_collator=None if args.train_on_entire_prompt else collator,
        max_seq_length=4096,
        peft_config=config,
        args=training_args
    )

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["Open-Orca/Mistral-7B-OpenOrca", "meta-llama/Llama-2-7b-chat-hf", "meta-llama/Meta-Llama-3-8B-Instruct", "google/gemma-7b-it"], type=str, help="/path/to/data")
    parser.add_argument("--mode", choices=["claim_decomposition", "question_generation", "question_answering", "verdict_prediction"], type=str, help="/path/to/data")
    parser.add_argument("--evidence_mode", type=str, choices=["gold_cells_llm", "all_table_llm", "gold_cells_linearized", "linearized_all_table"], help="/path/to/data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qa_functions", type=str, choices=["all", "arithmetic", "base"])
    parser.add_argument('--train_on_entire_prompt', action='store_true') # NOTE: THIS PERFORMED SUBSTENTIALLY WORSE THAN TRAINING ONLY ON COMPLETION

    args = parser.parse_args()

    main(args)
