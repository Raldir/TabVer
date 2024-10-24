import random
import torch
from pytorch_lightning import LightningDataModule
from tqdm import tqdm

from src.constants import (
    ANSWER_CHOICES_IDS,
    CLAIM_ID,
    CLAIM_SPAN_POS,
    EVIDENCE_INFERENCE_FILL,
    IDX,
    INPUT_IDS,
    LABEL,
    LAST_SPAN,
    NEG_INDEX,
    OP,
    OP_INFERENCE_FILL,
    LABEL_LIST
)
from src.data.data_processor import DatasetProcessor
from src.data.template_formatter import TemplateFormatter

class FinetuneDataModuleJoint(LightningDataModule):
    def init_spacy(self):
        from spacy.lang.en import English

        nlp = English()
        return nlp.tokenizer

    def __init__(self, config, tokenizer, mode):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.mode = mode
        self.spacy = self.init_spacy()
        self.template_formatter = TemplateFormatter(
            setting_nr=config.template_setting_id,
            neg_token=config.neg_token,
            num_questions=config.num_questions,
            num_templates=config.num_templates,
            question_id=config.question_id,
            template_id=config.template_id,
            randomize=config.randomize_templates,
        )

    def compute_answer_choices(self, correct_answer):
        ev_spans = ["No", "Yes"]
        label_id = ev_spans.index(correct_answer)
        return [ev_spans, label_id]

    def _process_dataset_natop(self, dataset, inference=False):
        processed_data = {}
        population = [0, 1]
        prop_sum = 1 + self.config.negative_samples_ratio
        weights = [1 / prop_sum, self.config.negative_samples_ratio / prop_sum]

        qid = 0

        for key in tqdm(dataset.claims):
            claim = dataset.claims[key]
            evidence = dataset.sentence_evidence[key]
            label = dataset.labels[key]
            alignment = dataset.alignments[key] if key in dataset.alignments else []
            label = dataset.labels[key]
            # If we do not consider retrieved evidence ignore NEI instances as they have no evidence.
            if label == self.config.nei_token and not self.config.use_retrieved_evidence:
                continue

            if inference:  # Only do dynamic parsing during inference
                chunked_claim = dataset.claims_parsed[key] if key in dataset.claims_parsed else []
                proof = [(x, EVIDENCE_INFERENCE_FILL, OP_INFERENCE_FILL) for x in chunked_claim]
            else:
                proof = dataset.proofver_proofs[key]

            # print(proof, alignment)
            for k, triple in enumerate(proof):  # Split here in dp vs non-dp
                if triple[0] == ".":
                    continue
                if not inference:
                    claim_span, ev_span, op = triple
                else:
                    claim_span_proof, ev_span, op = triple
                    claim_span, ev_span = alignment[k]
                    assert claim_span_proof == claim_span, "NOT MATCHING ALIGNMENT VS ORIGINAL {}, and {}".format(
                        claim_span_proof, claim_span
                    )
                # If operator is independence and we do nto consider retrieved evidence, continue
                if op not in self.template_formatter.op_to_type_map and not self.config.use_retrieved_evidence:
                    continue
                applied_templates_s, answers_s = self.template_formatter.apply_templates_to_sample_all_ops(
                    claim_span, ev_span, op, claim, evidence
                )

                for curr_op in range(len(applied_templates_s)):
                    answer_choices, label_id = self.compute_answer_choices(answers_s[curr_op])
                    # If we do not do inference, adjust the proportion of negative samples
                    if answer_choices[label_id] == self.config.neg_token and not inference:
                        rand = random.choices(population, weights)[0]
                        if rand == 0:
                            continue

                    last_span = k == (len(proof) - 1) and curr_op == (len(applied_templates_s) - 1)
                    processed_data[qid] = {
                        IDX: qid,  # Unique id starting from 0, required by dataloader
                        CLAIM_ID: key,  # Underlying claim id in dataset
                        CLAIM_SPAN_POS: k,  # Span position within claim
                        OP: curr_op,  # Operator the question is representing
                        NEG_INDEX: len(answer_choices) - 1,  # Index of the negative answer, needed for postprocessing
                        LABEL: label_id,  # index of correct evidence span
                        LAST_SPAN: last_span,  # Indicates whether this is last span of claim
                        "input": applied_templates_s[curr_op],  # Formatted Question
                        "output": answers_s[curr_op],  # correct evidence span string
                        "answer_choices": answer_choices,  # List of all evidence answer choices
                    }
                    qid += 1
        return processed_data

        
    def _process_dataset_verdict(self, dataset, qid_start, inference=False):

        # Mapping between label and answer choice representation
        ANSWER_CHOICES = ["Yes", "No", "Neutral"]

        processed_data = {}
        qid = qid_start
        ids_processed = set([])

        for key in tqdm(dataset.claims):
            ids_processed.add(key)
            claim = dataset.claims[key]
            evidence = dataset.sentence_evidence[key]
            label = dataset.labels[key]
            label_id = LABEL_LIST.index(label)
            # If we do not consider retrieved evidence ignore NEI instances as they have no evidence.
            if label == self.config.nei_token and not self.config.use_retrieved_evidence:
                continue
            input_text = "Is the claim: {} entailed given the evidence: {} Yes, No, or Neutral?".format(
                claim, evidence
            )
            if input_text in ids_processed and not inference:  # Remove duplicates from trianing data
                continue
            ids_processed.add(input_text)
            processed_data[qid] = {
                IDX: qid,  # Unique id starting from 0, required by dataloader
                CLAIM_ID: key,  # Underlying claim id in dataset
                CLAIM_SPAN_POS: 0,  # Span position within claim
                OP: 0,  # Operator the question is representing
                NEG_INDEX: 2,  # Index of the negative answer, needed for postprocessing
                LABEL: label_id,  # index of correct evidence span
                LAST_SPAN: 0,  # Indicates whether this is last span of claim
                "input": input_text,  # Formatted Question
                "output": label,  # correct evidence span string
                "answer_choices": ANSWER_CHOICES,  # List of all evidence answer choices
            }
            qid += 1
        return processed_data

    def _process_dataset(self, dataset, mode, inference=False):
        if not inference:
            processed_data = self._process_dataset_natop(dataset, inference=inference)
            processed_data_verdict = self._process_dataset_verdict(dataset, len(processed_data), inference=inference)

            processed_data.update(processed_data_verdict)
        else:
            if mode == "verdict":
                processed_data = self._process_dataset_verdict(dataset, inference=inference, qid_start = 0)
            elif mode == "natop":
                processed_data = self._process_dataset_natop(dataset, inference=inference)
            else:
                print("Mode not recognized, abort.")

        return processed_data

    def join_splits(self):
        self.dataset_or.claims.update(self.dev_dataset_or.claims)
        self.dataset_or.labels.update(self.dev_dataset_or.labels)
        self.dataset_or.proofver_proofs.update(self.dev_dataset_or.proofver_proofs)
        self.dataset_or.sentence_evidence.update(self.dev_dataset_or.sentence_evidence)
        self.dataset_or.claims_parsed.update(self.dev_dataset_or.claims_parsed)
        self.dataset_or.claims_parsed_hierarchy.update(self.dev_dataset_or.claims_parsed_hierarchy)
        self.dataset_or.alignments.update(self.dev_dataset_or.alignments)

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP

        if stage == "fit":
            self.dataset_or = DatasetProcessor(
                dataset=self.config.dataset,
                split="train",
                num_samples=self.config.num_samples,
                seed=self.config.seed,
                stratified_sampling=self.config.stratified_sampling,
                use_retrieved_evidence=self.config.use_retrieved_evidence,
                num_retrieved_evidence=self.config.num_retrieved_evidence,
                is_debug=self.config.debug,
                is_few_shot=self.config.few_shot,
                alignment_model=self.config.alignment_model,
                use_tab_qa = self.config.use_tab_qa, # no need to do tab QA for training data
                qg_model = self.config.qg_model,
                qa_model = self.config.qa_model,
                decomposition_model = self.config.decomposition_model,
                claim_decomposition = self.config.claim_decomposition,
                permissable_operators = self.config.permissable_operators
            )
            self.train_dataset = self._process_dataset(self.dataset_or, mode = self.mode)
            self.train_dataset = FinetuneDatasetWithTemplate(
            self.train_dataset, self.tokenizer, self.config.max_seq_len, self.config.max_answer_choice_length)
            print(f"Train size {len(self.train_dataset)}")

            self.dev_dataset_or = DatasetProcessor(
                dataset=self.config.dataset,
                split="validation",
                use_retrieved_evidence=self.config.use_retrieved_evidence,
                num_retrieved_evidence=self.config.num_retrieved_evidence,
                is_debug=self.config.debug,
                is_few_shot=self.config.few_shot,
                alignment_model=self.config.alignment_model,
                use_tab_qa = self.config.use_tab_qa,
                qg_model = self.config.qg_model,
                qa_model = self.config.qa_model,
                decomposition_model = self.config.decomposition_model,
                claim_decomposition = self.config.claim_decomposition,
                permissable_operators = self.config.permissable_operators
            )
            self.dev_dataset = self._process_dataset(self.dev_dataset_or, mode = self.mode, inference=True)
            self.dev_dataset = FinetuneDatasetWithTemplate( self.dev_dataset, self.tokenizer, self.config.max_seq_len, self.config.max_answer_choice_length)
            self.join_splits()  # create a single field dataset_or that can be used later to access info
            print(f"Eval size {len(self.dev_dataset)}")
            
        elif stage in ["validate"] and not hasattr(self, "dev_dataset"):
            self.dev_dataset_or = DatasetProcessor(
                dataset=self.config.dataset,
                split="validation",
                use_retrieved_evidence=self.config.use_retrieved_evidence,
                num_retrieved_evidence=self.config.num_retrieved_evidence,
                is_debug=self.config.debug,
                is_few_shot=self.config.few_shot,
                alignment_model=self.config.alignment_model,
                use_tab_qa = self.config.use_tab_qa,
                qg_model = self.config.qg_model,
                qa_model = self.config.qa_model,
                decomposition_model = self.config.decomposition_model,
                claim_decomposition = self.config.claim_decomposition,
                permissable_operators = self.config.permissable_operators
            )
            self.dev_dataset = self._process_dataset(self.dev_dataset_or, mode = self.mode, inference=True)
            self.dev_dataset = FinetuneDatasetWithTemplate( self.dev_dataset, self.tokenizer, self.config.max_seq_len, self.config.max_answer_choice_length)
            print(f"Eval size {len(self.dev_dataset)}")

        elif stage in ["predict", "test"]:
        # no num_samples specified here since evaluation on all data
            self.dev_dataset_or = DatasetProcessor(
                dataset=self.config.dataset,
                split="validation",
                use_retrieved_evidence=self.config.use_retrieved_evidence,
                num_retrieved_evidence=self.config.num_retrieved_evidence,
                is_debug=self.config.debug,
                is_few_shot=self.config.few_shot,
                alignment_model=self.config.alignment_model,
                use_tab_qa = self.config.use_tab_qa,
                qg_model = self.config.qg_model,
                qa_model = self.config.qa_model,
                decomposition_model = self.config.decomposition_model,
                claim_decomposition = self.config.claim_decomposition,
                permissable_operators = self.config.permissable_operators
            )
            self.dev_dataset = self._process_dataset(self.dev_dataset_or, mode = self.mode, inference=True)
            self.dev_dataset = FinetuneDatasetWithTemplate(
            self.dev_dataset, self.tokenizer, self.config.max_seq_len, self.config.max_answer_choice_length
            )
            self.dataset_or = self.dev_dataset_or

            print(f"Eval size {len(self.dev_dataset)}")



    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=create_collate_fn(self.tokenizer.pad_token_id, self.tokenizer.eos_token_id),
            drop_last=True,
            num_workers=min([self.config.batch_size, self.config.num_workers]),
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dev_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            collate_fn=create_collate_fn(
                self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
            ),  # cls_token_id # eos_token_id
            num_workers=min([self.config.eval_batch_size, self.config.num_workers]),
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dev_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            collate_fn=create_collate_fn(
                self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
            ),  # cls_token_id # eos_token_id
            num_workers=min([self.config.eval_batch_size, self.config.num_workers]),
        )



class FinetuneDatasetWithTemplate(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset, tokenizer, max_seq_len, max_answer_choice_length, add_special_tokens=True):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_answer_choice_length = max_answer_choice_length
        self.add_special_tokens = add_special_tokens

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        example = self.dataset[key]
        input_str = example["input"]
        target_str = example["output"]

        answer_choices = example["answer_choices"]

        input_ids = self.tokenizer(
            input_str,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_len,
            add_special_tokens=self.add_special_tokens,
        ).input_ids.squeeze(0)

        answer_choices_ids = [
            self.tokenizer(
                answer_choice,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_answer_choice_length,
                add_special_tokens=self.add_special_tokens,
            ).input_ids.squeeze(0)
            for answer_choice in answer_choices
        ]
        label = torch.LongTensor([example[LABEL]])
        idx = torch.LongTensor([example[IDX]])
        claim_id = torch.LongTensor([example[CLAIM_ID]])
        claim_span_pos = torch.LongTensor([example[CLAIM_SPAN_POS]])
        neg_index = torch.LongTensor([example[NEG_INDEX]])
        op = torch.LongTensor([example[OP]])
        last_span = torch.BoolTensor([example[LAST_SPAN]])

        return (
            input_ids,
            answer_choices_ids,
            label,
            idx,
            claim_id,
            claim_span_pos,
            op,
            neg_index,
            last_span,
        )


def create_collate_fn(pad_token_id, cls_token_id):
    def collate_fn(batch):
        (
            input_ids,
            answer_choices_ids,
            labels,
            idx,
            claim_id,
            claim_span_pos,
            op,
            neg_index,
            last_span,
        ) = zip(*batch)

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        output_batch = {
            INPUT_IDS: input_ids,
        }

        num_choice = [len(list_choices) for list_choices in answer_choices_ids]
        max_choices = max(num_choice)
        for list_choices in answer_choices_ids:  # Add empty answer choices to ensure that every sample has same number
            diff = max_choices - len(list_choices)
            for i in range(diff):
                list_choices.append(torch.LongTensor([cls_token_id]))
        flat_answer_choice_ids = [choice for list_choices in answer_choices_ids for choice in list_choices]
        num_choice = [len(list_choices) for list_choices in answer_choices_ids]

        if max(num_choice) != min(num_choice):
            raise NotImplementedError("The collate_fn is not implmented for variable number of choices")
        flat_answer_choices_ids = torch.nn.utils.rnn.pad_sequence(
            flat_answer_choice_ids, batch_first=True, padding_value=pad_token_id
        )
        answer_choices_ids = flat_answer_choices_ids.view(len(answer_choices_ids), max(num_choice), -1).contiguous()
        labels = torch.cat(labels)
        idx = torch.cat(idx)
        claim_id = torch.cat(claim_id)
        claim_span_pos = torch.cat(claim_span_pos)
        op = torch.cat(op)
        neg_index = torch.cat(neg_index)
        last_span = torch.cat(last_span)

        output_batch.update(
            {
                ANSWER_CHOICES_IDS: answer_choices_ids,
                LABEL: labels,
                IDX: idx,
                CLAIM_ID: claim_id,
                CLAIM_SPAN_POS: claim_span_pos,
                OP: op,
                NEG_INDEX: neg_index,
                LAST_SPAN: last_span,
            }
        )

        return output_batch

    return collate_fn


if __name__ == "__main__":
    import argparse

    from transformers import AutoTokenizer

    from src.utils.util import ParseKwargs, set_seeds

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_files", required=True)
    parser.add_argument("-k", "--kwargs", nargs="*", action=ParseKwargs, default={})
    args = parser.parse_args()

    from src.utils.Config import Config

    config = Config(args.config_files, args.kwargs)

    tokenizer = AutoTokenizer.from_pretrained(config.origin_model)

    datamodule = FinetuneDataModuleJoint(config, tokenizer)
