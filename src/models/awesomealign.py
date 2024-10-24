import argparse
import itertools
import json
import os

import torch
import transformers

from src.utils.util import ROOT_DIR


class AwesomeAligner(object):
    def __init__(
        self,
        model: str = "bert",
        finetuned: bool = False,
        token_type: str = "bpe",
        distortion: float = 0.0,
        matching_methods: str = "mai",
        device: str = "cpu",
        layer: int = 8,
    ):  # layer = 8 is defaault
        model_names = {
            "medbert": "emilyalsentzer/Bio_ClinicalBERT",
            "bert": "bert-base-multilingual-cased",  # "sentence-transformers/all-mpnet-base-v2", #"bert-base-multilingual-cased",
            "xlmr": "xlm-roberta-large",  # "microsoft/deberta-v3-large" #"xlm-roberta-large"
        }
        all_matching_methods = {"a": "inter", "m": "mwmf", "i": "itermax", "f": "fwd", "r": "rev"}

        if finetuned:
            model_path = os.path.join(ROOT_DIR, "models", "awesomealign", model)
            self.model = transformers.BertModel.from_pretrained(model_path)
        else:
            self.model = transformers.BertModel.from_pretrained("bert-base-multilingual-cased")

        self.tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-multilingual-cased")

    def get_word_aligns(self, src, tgt):
        # sent_src, sent_tgt = src.strip().split(), tgt.strip().split()
        token_src, token_tgt = [self.tokenizer.tokenize(word) for word in src], [
            self.tokenizer.tokenize(word) for word in tgt
        ]
        wid_src, wid_tgt = [self.tokenizer.convert_tokens_to_ids(x) for x in token_src], [
            self.tokenizer.convert_tokens_to_ids(x) for x in token_tgt
        ]
        ids_src, ids_tgt = (
            self.tokenizer.prepare_for_model(
                list(itertools.chain(*wid_src)),
                return_tensors="pt",
                model_max_length=self.tokenizer.model_max_length,
                truncation=True,
            )["input_ids"],
            self.tokenizer.prepare_for_model(
                list(itertools.chain(*wid_tgt)),
                return_tensors="pt",
                truncation=True,
                model_max_length=self.tokenizer.model_max_length,
            )["input_ids"],
        )
        sub2word_map_src = []
        for i, word_list in enumerate(token_src):
            sub2word_map_src += [i for x in word_list]
        sub2word_map_tgt = []
        for i, word_list in enumerate(token_tgt):
            sub2word_map_tgt += [i for x in word_list]

        # alignment
        align_layer = 8
        threshold = 1e-3
        self.model.eval()
        with torch.no_grad():
            out_src = self.model(ids_src.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]
            out_tgt = self.model(ids_tgt.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]

        dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))

        softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
        softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)

        softmax_inter = (softmax_srctgt > threshold) * (softmax_tgtsrc > threshold)

        align_subwords = torch.nonzero(softmax_inter, as_tuple=False)
        align_words = list()
        for i, j in align_subwords:
            align_words.append((sub2word_map_src[i], sub2word_map_tgt[j]))

        return align_words


def execute_train(
    train_data_file,
    eval_data_file,
    output_dir,
    model_name_or_path="bert-base-multilingual-cased",
    extraction="softmax",
    do_train=True,
    train_tlm=True,
    train_so=True,
    train_mlm=False,
    train_tlm_full=False,
    train_psi=False,
    per_gpu_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-5,
    save_steps=4000,
    max_steps=20000,
    do_eval=True,
    train_co=False,
    train_gold_file=None,
    eval_gold_file=None,
    ignore_possible_alignments=False,
    gold_one_index=False,
    cache_data=False,
    align_layer=8,
    softmax_threshold=0.001,
    should_continue=False,
    mlm_probability=0.15,
    config_name=None,
    tokenizer_name=None,
    cache_dir=None,
    block_size=-1,
    seed=42,
    per_gpu_eval_batch_size=16,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    max_grad_norm=1.0,
    warmup_steps=0,
    logging_steps=500,
    save_total_limit=None,
    no_cuda=False,
    overwrite_output_dir=False,
    overwrite_cache=False,
    fp16_opt_level="01",
    local_rank=1,
):
    # Construct command
    command = f"awesome-train --output_dir={output_dir} --model_name_or_path={model_name_or_path} --extraction '{extraction}'"
    command += f" {'--do_train' if do_train else ''} {'--train_tlm' if train_tlm else ''} {'--train_so' if train_so else ''} --train_data_file={train_data_file}"
    command += f" {'--train_mlm' if train_mlm else ''} {'--train_tlm_full' if train_tlm_full else ''}{'--train_psi' if train_psi else ''}"
    command += f" --per_gpu_train_batch_size={per_gpu_train_batch_size} --gradient_accumulation_steps={gradient_accumulation_steps} --num_train_epochs={num_train_epochs}"
    command += f" --learning_rate={learning_rate} --save_steps={save_steps} --max_steps={max_steps} {'--do_eval' if do_eval else ''} --eval_data_file={eval_data_file}"

    # Execute command
    os.system(command)


def write_awesomealign_format(exp_name, in_path, out_path):
    data = []
    with open(in_path, "r") as f_in:
        lines = f_in.readlines()
        for line in lines:
            content = json.loads(line)
            if exp_name == "default_gold_no_nei" and content["verdict"] == "NOT ENOUGH INFO":
                continue
            qid = content["id"]
            claim = content["claim"]
            evidence = content["evidence"]
            input_text = "{} ||| {}".format(claim, evidence)
            data.append(input_text)
    with open(out_path, "w") as f_out:
        for content in data:
            f_out.write("{}\n".format(content))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="/path/to/data")
    parser.add_argument("--config_path", type=str, help="/path/to/data")
    parser.add_argument("--is_few_shot", action="store_true")
    args = parser.parse_args()

    with open(os.path.join(ROOT_DIR, "configs", "align_train", args.config_path + ".json"), "r") as f:
        config = json.load(f)

    if args.dataset in ["scifact"]:
        train_path = os.path.join(ROOT_DIR, "data", args.dataset, "ADD_HERE")
    elif args.dataset in ["fever"]:
        if args.is_few_shot:
            train_data_path = os.path.join(
                ROOT_DIR,
                "data",
                args.dataset,
                "processed_train-fewshot_use_retr_False_retr_evidence_2_dp_True_alignment_mode_simalign_max_chunks_6_alignment_model_bert_matching_method_mwmf_loose_matching_True.jsonl",
            )
        else:
            if config["exp_name"] == "default":
                train_data_path = os.path.join(
                    ROOT_DIR,
                    "data",
                    args.dataset,
                    "processed_train_use_retr_True_retr_evidence_2_dp_False_concat_op_s.jsonl",
                )
            else:
                train_data_path = os.path.join(
                    ROOT_DIR,
                    "data",
                    args.dataset,
                    "processed_train_use_retr_False_retr_evidence_7_dp_False_alignment_mode_proofver_max_chunks_6_alignment_model_bert_matching_method_mwmf_loose_matching_False.jsonl",
                )

    output_dir = os.path.join(
        ROOT_DIR, "models", "awesomealign", config["exp_name"] + "_few_shot_" + str(args.is_few_shot)
    )

    if not os.path.exists(output_dir + "_data"):
        os.makedirs(output_dir + "_data")

    train_data_file = os.path.join(output_dir + "_data", "training_data.txt")
    eval_data_file = os.path.join(output_dir + "_data", "training_data.txt")

    config["train_data_file"] = train_data_file
    config["eval_data_file"] = eval_data_file
    config["output_dir"] = output_dir

    write_awesomealign_format(config["exp_name"], train_data_path, train_data_file)

    config.pop("exp_name", None)

    execute_train(**config)

    # src = 'awesome-align is awesome !'
    # tgt = '牛对齐 是 牛 ！'

    # args = parser.parse_args()
    # aligner = AwesomeAligner()
    # aligner.align(src, tgt)


if __name__ == "__main__":
    main()
