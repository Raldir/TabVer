import json
import sys
import os
import torch
import gc



from src.utils.util import ROOT_DIR
from src.data.data_reader import DatasetReader
from src.utils.normalization import normalize_subclaim_id

from src.data.data_reader import DatasetReader
from src.data.claim_decomposition import ClaimDecomposer
from src.utils.normalization import normalize_subclaim_id
from src.data.claim_decomposition import ClaimDecomposer
from src.data.arith_expressions import ArithExpressionGenerator
from src.data.table_formatter_tabfact import TableFormatterTabFact


class TabFactReader(DatasetReader):

    def __init__(self, seed, split, num_samples, ev_sentence_concat_op, use_tab_qa, qg_model, qa_model, decomposition_model, permissable_operators, claim_decomposition, stratified_sampling=False, granularity=None):
        DatasetReader.__init__(self, split, granularity)
        self.dataset = "tabfact"
        self.split = split
        self.ev_sentence_concat_op = ev_sentence_concat_op
        self.use_tab_qa = use_tab_qa
        self.claim_decomposition = claim_decomposition
        self.proofver_file = os.path.join(ROOT_DIR, "data", "feverous", "handcrafted_training_data", "proofver_data", "train_with_ids_fewshot.target")
        self.stratified_sampling = stratified_sampling
        self.permissable_operators = permissable_operators
        split_str = "{}_{}_seed_{}".format(split, num_samples, seed) if "train" in split else split

        if split == "train":
            print("Training with TabFact data not supported...")
            sys.exit()
        elif split == "validation":
            self.num_samples = None
            self.claim_file = os.path.join(ROOT_DIR, "data", self.dataset, "tabfact_dev_formatted.jsonl")

        if claim_decomposition:
            decomposition_model_name = decomposition_model.split("/")[-1] if len(decomposition_model.split("/")) == 2 else  decomposition_model.split("/")[-2]
            use_tab_qa_name = use_tab_qa.replace("_replace", "")
            save_name =  "{}_{}_{}".format(split_str, claim_decomposition, decomposition_model_name)
            decomposer = ClaimDecomposer(model=decomposition_model, dataset="tabfact", input_path=self.claim_file, output_name = save_name, num_samples = self.num_samples)
            if not os.path.isfile(decomposer.out_path):
                print("Decomposition file not found at {}.\n Starting Decomposition".format(decomposer.out_path))
                decomposer.decompose_claims(max_seq_len=4096, top_p=1.0, temperature=1.0, max_batch_size=1)
            self.claim_file = decomposer.out_path
            del decomposer
            gc.collect()
            torch.cuda.empty_cache()
        
        if "llm" in use_tab_qa:
            qa_model_name = qa_model.split("/")[-1] if len(qa_model.split("/")) == 2 else  qa_model.split("/")[-2]
            use_tab_qa_name = use_tab_qa.replace("_replace", "")
            if permissable_operators == "none":
                permissable_operators = "all"
            save_name =  "{}_{}_{}_{}_{}".format(split_str, use_tab_qa_name, claim_decomposition, qa_model_name, permissable_operators)
            arithexp_generator = ArithExpressionGenerator(dataset="tabfact", input_path=self.claim_file, save_name=save_name, use_tab_qa=use_tab_qa, qg_model=qg_model, qa_model=qa_model, permissable_operators=permissable_operators)
            if not os.path.isfile(arithexp_generator.output_path_qa):
                print("Arithmetic Expressions file not found at {}.\n Starting generating arith expressions".format(arithexp_generator.output_path_qa))
                arithexp_generator.generate_arith_exps()
            self.claim_file = arithexp_generator.output_path_qa
            del arithexp_generator
            gc.collect()
            torch.cuda.empty_cache()
        else:
            table_formatter = TableFormatterTabFact(self.claim_file, tab_qa="linearized_" + use_tab_qa)
            self.data_w_formatted_tables = table_formatter.load_data()
            print("LENGTH", len(self.data_w_formatted_tables))
                

        self.granularity = granularity

    def read_annotations(self):
        annotations = []
        proofver_proofs = {}

        with open(self.claim_file, "r", encoding="utf-8") as f_in:
            # open qrels file if provided
            limit = self.num_samples if not self.claim_decomposition else None # Consider all decomposed samples since decompositon limits already if # 1000 if "train" in self.split else None
            for i, line in enumerate(f_in.readlines()[:limit]): # Only read in first 200 entries for now
                line_json = json.loads(line.strip())
                qid = line_json["id"]
                query = line_json["claim"][:300] # If query is too long
                label = line_json["verdict"] #line_json["label"] #line_json["verdict"]
                if "llm" in self.use_tab_qa:
                    table_title = line_json["title"]
                    linearized_evidence = "[ {} ] {}".format(table_title, self.ev_sentence_concat_op.join(line_json["evidence_extracted_llm"]))
                else:
                    linearized_evidence = self.data_w_formatted_tables[i]["table"]
                    if not linearized_evidence or linearized_evidence.strip() == "":
                        linearized_evidence = "N/A"
                
                # if "N/A" in linearized_evidence and: # Evidence does not contain relevant information from table
                 # label = "REFUTES"

                if "generated_answers" in line_json and not self.permissable_operators == "none":
                    arith_exp = line_json["generated_answers"]
                    assert len(arith_exp) == len(line_json["chunk_answers"]), "Number of generated Questions and answers do not aligned, found {} and {}".format(len(arith_exp), len(line_json["chunk_answers"]))
                    arith_exp = [(line_json["chunk_answers"][i], arith_exp[i]) for i in range(len(arith_exp))]
                else:
                    arith_exp = []

                proof = []
                anno = (qid, query, label, linearized_evidence, arith_exp, proof)
                annotations.append(anno)

        #TODO: Fix once possible. Add stratified sampling for both train and val.
        # if "train" in self.split_name:
        #     annotations = self.rebalance_data_train(annotations)
        
        label_stats = {"SUPPORTS": 0, "REFUTES": 0, "NOT ENOUGH INFO": 0}
        all_keys = set([])
        for anno in annotations:
            # acc_key = str(key).split("0000000000")[0]
            acc_key = normalize_subclaim_id(anno[0])
            if acc_key in all_keys:
                continue
            all_keys.add(acc_key)
            label_stats[anno[2]] += 1
        print("Label Distributions: ", label_stats)

        return annotations

            
    def read_corpus(self, input_path):
        pass