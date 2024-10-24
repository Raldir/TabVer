import json
import os
import sys
import traceback


from tqdm import tqdm

from src.data.chunking_and_alignment import DynamicSentenceAligner
from src.data.feverous_reader import FeverousReader
from src.data.tabfact_reader import TabFactReader
from src.utils.normalization import normalize_subclaim_id
from src.utils.util import ROOT_DIR

class DatasetProcessor(object):
    def __init__(
        self,
        dataset,
        split,
        num_samples=32,
        seed=42,
        stratified_sampling=False,
        is_few_shot=False,
        ev_sentence_concat_op=" </s> ",
        use_retrieved_evidence="True",
        num_retrieved_evidence=3,
        alignment_model="bert",
        overwrite_data=False,
        is_debug=False,
        use_tab_qa=True,
        claim_decomposition=False,
        qg_model="Open-Orca/Mistral-7B-OpenOrca",
        qa_model="Open-Orca/Mistral-7B-OpenOrca",
        decomposition_model="Open-Orca/Mistral-7B-OpenOrca",
        permissable_operators="all",
    ):
        assert use_retrieved_evidence in [
            "True",
            "False",
            "Only",
            True,
            False,
        ], "Value for use_retrieved_evidence not known, select from {}".format(["True", "False", "Only"])

        tab_qa_settings =["", "gold_cells_llm", "all_table_llm", "gold_cells_linearized", "linearized_all_table"]
        assert any([x in use_tab_qa for x in tab_qa_settings]), "Tab QA mode not known, found {}".format(use_tab_qa)

        self.dataset = dataset
        self.num_samples = num_samples if not is_debug else 100
        self.stratified_sampling = stratified_sampling
        self.use_retrieved_evidence = use_retrieved_evidence
        self.split = split
        self.seed = seed
        self.is_few_shot = is_few_shot
        self.split_name = split + "-fewshot" if self.is_few_shot and "train" in split else split
        self.num_retrieved_evidence = num_retrieved_evidence
        self.ev_sentence_concat_op = ev_sentence_concat_op
        self.claim_decomposition = claim_decomposition
        self.alignment_model = alignment_model
        self.use_tab_qa = use_tab_qa
        self.qg_model = qg_model
        self.qa_model = qa_model
        self.decomposition_model = decomposition_model
        self.permissable_operators = permissable_operators

        self._set_save_path()

        print("Trying to find data from {} ...".format(self.save_path))
        if os.path.exists(self.save_path) and not overwrite_data:
            print("Load existing data from {} ...".format(self.save_path))
            self.load_existing_data(is_debug)

            label_stats = {"SUPPORTS": 0, "REFUTES": 0, "NOT ENOUGH INFO": 0}
            all_keys = set([])
            for key, lab in self.labels.items():
                # acc_key = str(key).split("0000000000")[0]
                acc_key = normalize_subclaim_id(key)
                if acc_key in all_keys:
                    continue
                all_keys.add(acc_key)
                label_stats[lab] += 1
            print("Label Distributions: ", label_stats)
        else:
            self.claims = {}
            self.labels = {}
            self.sentence_evidence = {}
            self.proofver_proofs = {}
            self.arith_exps = {}
            self.claims_parsed = {}
            self.claims_parsed_hierarchy = {}
            self.alignments = {}

            self.setup_dataset()

            # Use proofver for alignment or our multi-granular alignment method
            aligner = DynamicSentenceAligner(
                dataset=self.dataset,
                split=self.split,
                alignment_model=alignment_model,
                num_retrieved_evidence=num_retrieved_evidence,
                replace_with_arith_exps = "replace" in self.use_tab_qa
            )

            for qid, claim in tqdm(self.claims.items()):
                # if qid != 134135:
                #     continue
                try:
                    print(qid, claim, self.sentence_evidence[qid])
                    aligned = aligner.align_sample(qid, claim, self.sentence_evidence[qid], self.arith_exps[qid])
                except:
                    traceback.print_exc()
                    print(
                        "Alignment Error for qid {} and claim {}, evidence {}, and arith_exp {}".format(
                            qid, claim, self.sentence_evidence[qid], self.arith_exps[qid]
                        )
                    )
                self.claims_parsed[qid] = aligned["claim_parsed"]
                self.claims_parsed_hierarchy[qid] = aligned["claim_parsed_hierarchy"]
                self.alignments[qid] = aligned["alignment"]
            self._save_data()

    def _set_save_path(self):
        dataset_path = self.dataset
        # qa_model_str = self.qa_model.split("/")[1]
        qa_model_str = self.qa_model.split("/")[-1] if len(self.qa_model.split("/")) == 2 else  self.qa_model.split("/")[-2]

        if "train" in self.split:
            samples_text = str(self.num_samples)
            if self.stratified_sampling:
                samples_text += "_stratified"
            self.samples_text = samples_text
            # Train always on FEVEROUS
            self.save_path = os.path.join(
                ROOT_DIR,
                "data",
                "feverous",
                "{}_num_samples_{}_seed_{}_use_retr_{}_retr_evidence_{}_tabqa_{}_model_{}_decomp_{}.jsonl".format(
                    self.split_name,
                    samples_text,
                    self.seed,
                    self.use_retrieved_evidence,
                    self.num_retrieved_evidence,
                    self.use_tab_qa,
                    qa_model_str,
                    self.claim_decomposition,
                ),
            )
        else:  # Do not add num samples argument since evaluation is on entire data
            self.save_path = os.path.join(
                ROOT_DIR,
                "data",
                dataset_path,
                "{}_use_retr_{}_retr_ev_{}_use_tabqa_{}_model_{}_decomp_{}_ops_{}.jsonl".format(
                    self.split_name,
                    self.use_retrieved_evidence,
                    self.num_retrieved_evidence,
                    self.use_tab_qa,
                    qa_model_str,
                    self.claim_decomposition,
                    self.permissable_operators,              
                ),
            )

    def _save_data(self):
        with open(self.save_path, "w") as f_out:
            for key in self.claims:
                dictc = {}
                if key in self.proofver_proofs:
                    claim = self.claims[key]
                    claims_parsed = self.claims_parsed[key]
                    claim_parsed_hierarchy = self.claims_parsed_hierarchy[key]
                    alignment = self.alignments[key]
                    evidence = self.sentence_evidence[key]
                    proof = self.proofver_proofs[key]
                    arith_exp = self.arith_exps[key]
                    verdict = self.labels[key]
                    dictc["id"] = key
                    dictc["claim"] = claim
                    dictc["verdict"] = verdict
                    dictc["evidence"] = evidence
                    dictc["proof"] = proof
                    dictc["arith_exp"] = arith_exp
                    dictc["claim_parsed"] = claims_parsed
                    dictc["claim_parsed_hierarchy"] = claim_parsed_hierarchy
                    dictc["alignment"] = alignment
                    f_out.write("{}\n".format(json.dumps(dictc)))
                    print(claim)
                    print(claims_parsed)
                    print(alignment)
                    print(evidence)
                    print(proof)
                    print("--------")

    def load_existing_data(self, is_debug):
        self.claims = {}
        self.labels = {}
        self.sentence_evidence = {}
        self.claims_parsed = {}
        self.claims_parsed_hierarchy = {}
        self.alignments = {}
        self.proofver_proofs = {}
        self.arith_exps = {}

        with open(self.save_path, "r") as f_in:
            lines = f_in.readlines()
            cutoff = len(lines) + 1 if not is_debug else 100
            for line in lines[:cutoff]:
                content = json.loads(line)
                qid = content["id"]
                self.claims[qid] = content["claim"]
                self.labels[qid] = content["verdict"]
                self.sentence_evidence[qid] = content["evidence"]
                self.proofver_proofs[qid] = content["proof"]
                self.arith_exps[qid] = content["arith_exp"]
                self.claims_parsed[qid] = content["claim_parsed"]
                self.claims_parsed_hierarchy[qid] = content["claim_parsed_hierarchy"]
                self.alignments[qid] = content["alignment"]


    def setup_dataset(self):
        # save_name = self.save_path.split(".jsonl")[0].split("/")[-1]
        if self.dataset == "feverous":
            self.dataset_reader = FeverousReader(self.seed, self.split, self.num_samples, ev_sentence_concat_op= self.ev_sentence_concat_op, use_tab_qa=self.use_tab_qa, qg_model = self.qg_model, qa_model = self.qa_model,  decomposition_model = self.decomposition_model, permissable_operators=self.permissable_operators, claim_decomposition=self.claim_decomposition)
            for i, anno in enumerate(self.dataset_reader.read_annotations()):
                qid, query, label, evidences, arith_exp, proof = anno
                qid = int(qid)
                self.claims[qid] = query
                self.labels[qid] = label
                self.sentence_evidence[qid] = evidences
                self.proofver_proofs[qid] = proof
                self.arith_exps[qid] = arith_exp
        elif self.dataset == "tabfact":
            self.dataset_reader = TabFactReader(self.seed, self.split, self.num_samples, ev_sentence_concat_op= self.ev_sentence_concat_op, use_tab_qa=self.use_tab_qa, qg_model = self.qg_model, qa_model = self.qa_model, decomposition_model = self.decomposition_model, permissable_operators=self.permissable_operators, claim_decomposition=self.claim_decomposition)
            for i, anno in enumerate(self.dataset_reader.read_annotations()):
                qid, query, label, evidences, arith_exp, proof = anno
                qid = int(qid)
                self.claims[qid] = query
                self.labels[qid] = label
                self.sentence_evidence[qid] = evidences
                self.proofver_proofs[qid] = proof
                self.arith_exps[qid] = arith_exp
        elif self.dataset == "feverous_halo":
            print("FEVEROUS Halo subset is generated via the file 'src/utils/generating_pragmatic_halo_data.py'. Abort...")
            sys.exit()


if __name__ == "__main__":
    dataset = sys.argv[1]

    configs = sys.argv[2]

    # Add multiple configs via "+" "dynamic_simalign_bert_mwmf_coarse+dynamic_simalign_bert_mwmf
    configs = configs.split("+")

    configs_path = os.path.join(ROOT_DIR, "configs", "alignment")

    onlyfiles = [
        os.path.join(configs_path, f)
        for f in os.listdir(configs_path)
        if os.path.isfile(os.path.join(configs_path, f))
    ]
    onlyfiles = sorted(onlyfiles)

    for config_file in onlyfiles:
        config_file_name = config_file.split("/")[-1].split(".")[0]
        if not config_file_name in configs:
            continue
        with open(config_file, "r") as f_in:
            config = json.load(f_in)
        # for split in ["train"]:
        for split in ["validation"]:
            # for split in ["train", "validation"], "symmetric"]:
            dynamic_parsing = config["dynamic_parsing"]
            use_retrieved_evidence = config["use_retrieved_evidence"]
            num_retrieved_evidence = config["num_retrieved_evidence"]
            few_shot = config["few_shot"]
            max_chunks = config["max_chunks"]
            alignment_mode = config["alignment_mode"]
            alignment_model = config["alignment_model"]
            matching_method = config["matching_method"]
            loose_matching = config["loose_matching"]
            num_samples = 32  # Fixing samples to 32

            split_name = split + "-fewshot" if few_shot and "train" in split else split

            fever_data = DatasetProcessor(
                dataset=dataset,
                split=split,
                num_samples=num_samples,
                overwrite_data=True,
                dynamic_parsing=dynamic_parsing,
                is_few_shot=few_shot,
                use_retrieved_evidence=use_retrieved_evidence,
                num_retrieved_evidence=num_retrieved_evidence,
                ev_sentence_concat_op=" </s> ",
                max_chunks=max_chunks,
                alignment_mode=alignment_mode,
                alignment_model=alignment_model,
                matching_method=matching_method,
                loose_matching=loose_matching,
            )