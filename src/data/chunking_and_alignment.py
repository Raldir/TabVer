# from constituent_treelib import ConstituentTree, BracketedTree, Language, Structure
import copy
import gc
import json
import string

import torch
import torch.nn.functional as F
from flair.data import Sentence
from flair.models import SequenceTagger
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.utils.normalization import normalize_subclaim_id
from src.constants import FUNCTION_WORDS
from src.models.awesomealign import AwesomeAligner
from src.models.simalign import SentenceAligner


class DynamicSentenceAligner(object):
    def __init__(
        self,
        dataset,
        split,
        alignment_model="bert",  # "bert", bert_finetuned_paper, #xlmr
        matching_method="mwmf",  # mwmf, inter, itermax
        sentence_transformer="sentence-transformers/all-mpnet-base-v2",
        num_retrieved_evidence=2,
        max_chunks=6,
        alignment_mode="awesomealign",
        loose_matching=False,  # name misleading (legacy), but essentially allows merging of up to 4 spans, instead of only three 
        dynamic_parsing=True,  # Whether to use dyamic parsing, or only static chunking
        replace_with_arith_exps = False,
        permissable_operators="arithmetic",
    ):
        assert alignment_mode in [
            "simalign",
            "sentence_transformer",
            "awesomealign",
        ], "Selected alignment mode {} not found".format(alignment_mode)
        if alignment_mode == "simalign":
            self.model = SentenceAligner(
                model=alignment_model,
                matching_method=matching_method,
                token_type="bpe",
                matching_methods="mai",
                device="cuda:0",
            )
        elif alignment_mode == "awesomealign":
            finetuned = False
            if "finetuned_" in alignment_model:
                finetuned = True
                alignment_model = alignment_model.split("finetuned_")[1]
            self.model = AwesomeAligner(
                model=alignment_model, finetuned=finetuned, token_type="bpe", matching_methods="mai", device="cuda:0"
            )
        self.dataset = dataset
        self.split = split
        self.matching_method = matching_method
        self.replace_with_arith_exps = replace_with_arith_exps
        self.phrase_model = AutoModel.from_pretrained(sentence_transformer).to("cuda:0")
        self.phrase_tokenizer = AutoTokenizer.from_pretrained(sentence_transformer)
        self.tagger = SequenceTagger.load("flair/chunk-english")
        self.num_retrieved_evidence = num_retrieved_evidence
        self.max_chunks = max_chunks
        self.alignment_mode = alignment_mode
        self.loose_matching = loose_matching
        self.dynamic_parsing = dynamic_parsing

    def _cos_sim(self, a, b):
        """
        Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
        :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
        """
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def find_combinations(self, substrings, sentence):
        dp = {}

        def find_sentence_combinations(w):
            if w in dp:
                return dp[w]

            if not w:
                return [[]]

            result = []
            for i, substring in enumerate(substrings):
                if w.startswith(substring):
                    remaining = w[len(substring) :].lstrip()
                    remaining_combinations = find_sentence_combinations(remaining)

                    for combination in remaining_combinations:
                        result.append([i] + combination)

            dp[w] = result
            return result

        return find_sentence_combinations(sentence)

    def _find_all_options(self, spans, spans_flat_index, claim_token_to_span_map, claim_sentence):
        options = []

        if self.dynamic_parsing:
            all_combs = self.find_combinations(spans, claim_sentence)
            # print(all_combs)
            new_combs = []
            for comb in all_combs:
                # print(len(comb))
                if len(comb) < self.max_chunks:
                    new_combs.append(comb)
            options += new_combs[:5000]  # Arbitary cutoff in case too many options
            # print("-------")

            if " ".join(spans[:spans_flat_index]) == claim_sentence:
                ind = [spans.index(x) for x in spans[:spans_flat_index]]
                options.append(ind)
        else:
            if " ".join(spans) == claim_sentence:  # check if is equal sentence without dot
                ind = [spans.index(x) for x in spans]
                options.append(ind)

        # Get all span duplicates
        duplicates = {}
        for i, ele in enumerate(spans):
            if ele in spans[:i]:
                duplicates[i] = spans[:i].index(ele)

        # Remove spans that are not part of a parse
        updated_spans = []
        old_to_new_mapping = {}
        for index, span in enumerate(spans):
            in_option = False
            for option in options:
                if index in option:
                    in_option = True
            if in_option:
                updated_spans.append(span)
                old_to_new_mapping[index] = len(updated_spans) - 1

        updated_options = []
        for option in options:
            updated_option = [old_to_new_mapping[x] for x in option]
            if updated_option not in updated_options:
                updated_options.append(updated_option)

        claim_token_to_span_map_new = {}
        for key, value_list in claim_token_to_span_map.items():
            claim_token_to_span_map_new[key] = []
            for value in value_list:
                if value not in old_to_new_mapping:
                    continue
                if value in duplicates:
                    claim_token_to_span_map_new[key].append(old_to_new_mapping[duplicates[value]])
                else:
                    claim_token_to_span_map_new[key].append(old_to_new_mapping[value])

        # arith_exp_in_span = [x for x in updated_spans if arith_exp[0] in x] # Find overlapping claim spans where arith_exp expected answer occurs
        return [updated_spans, updated_options, claim_token_to_span_map_new]

    def _chunk_flair(self, sentence_or):
        aList = list()
        tokens = list()
        token_to_span_map = {}
        buffer = ""
        token_pointer = 0

        if self.dataset == "danfever":
            sentence = self.tagger.predict(sentence_or)
            nlp = self.tagger.model
            doc = nlp(sentence_or)

            curr_toks = []
            previous_tag = ""
            for token, nc in zip(doc, sentence):
                if nc == "O" and "NP" in previous_tag:  # An outside tag indicates that a span is complete
                    curr_span = " ".join(curr_toks)
                    aList.append(curr_span)
                    tokens += curr_toks
                    for ele in range(token_pointer, len(tokens), 1):
                        token_to_span_map[ele] = len(aList) - 1
                    token_pointer = len(tokens)
                    curr_toks = []
                curr_toks.append(token.text)
                previous_tag = nc
            if curr_toks:
                curr_span = " ".join(curr_toks)
                if not curr_span == ".":
                    aList.append(curr_span)
                    tokens += curr_toks
                    for ele in range(token_pointer, len(tokens), 1):
                        token_to_span_map[ele] = len(aList) - 1
                    token_pointer = len(tokens)
                    curr_toks = []

        else:
            sentence = Sentence(sentence_or)
            self.tagger.predict(sentence)

            # TODO: SOMEHOW "and" is being removed by chunking
            for i, entity in enumerate(sentence.get_spans("np")):
                span_content = entity.text
                neg_flag = False
                if span_content in ["is", "are"] and ("is not" in sentence_or or "are not" in sentence_or):
                    # Token "not" is being swallowed by chunker
                    span_content = span_content + " not"
                    neg_flag = True
                span_tokens = [
                    item.text if not neg_flag or item.text not in ["is", "are"] else item.text + " not"
                    for item in entity.tokens
                ]
                tokens += span_tokens
                span_pos = entity.get_label("np").value
                if span_pos == "PP" or all(
                    [x.lower() in FUNCTION_WORDS or x.isnumeric() or x in string.punctuation for x in span_tokens]
                ):
                    buffer += span_content + " "
                else:
                    rendered_chunk = buffer + span_content
                    rendered_chunk = rendered_chunk.strip()
                    aList.append(rendered_chunk)
                    buffer = ""
                    for ele in range(token_pointer, len(tokens), 1):
                        token_to_span_map[ele] = len(aList) - 1
                    token_pointer = len(tokens)

            if buffer:  # In case some buffer remains (e.g. Demon Albert was released.)
                rendered_chunk = buffer.strip()
                aList.append(rendered_chunk)
                buffer = ""
                for ele in range(token_pointer, len(tokens), 1):
                    token_to_span_map[ele] = len(aList) - 1
                token_pointer = len(tokens)

        sentence = " ".join(aList)
        aList_flat_index = len(aList)

        if self.dynamic_parsing:
            token_to_span_maps = {}
            token_to_span_map_inverse = {}
            for key, value in token_to_span_map.items():
                token_to_span_maps[key] = [value]
                if value in token_to_span_map_inverse:
                    token_to_span_map_inverse[value].append(key)
                else:
                    token_to_span_map_inverse[value] = [key]

            len_or = len(aList)
            for a, b in zip(aList, aList[1::]):
                aList.append(a + " " + b)
                tokens_a = token_to_span_map_inverse[aList.index(a)]
                tokens_b = token_to_span_map_inverse[aList.index(b)]
                for token in tokens_a + tokens_b:
                    token_to_span_maps[token].append(len(aList) - 1)

            if not self.loose_matching:
                for a, b, c in zip(aList[:len_or], aList[1:len_or:], aList[2:len_or:]):
                    aList.append(a + " " + b + " " + c)
                    tokens_a = token_to_span_map_inverse[aList.index(a)]
                    tokens_b = token_to_span_map_inverse[aList.index(b)]
                    tokens_c = token_to_span_map_inverse[aList.index(c)]
                    for token in tokens_a + tokens_b + tokens_c:
                        token_to_span_maps[token].append(len(aList) - 1)

                for a, b, c, d in zip(aList[:len_or], aList[1:len_or:], aList[2:len_or:], aList[3:len_or:]):
                    aList.append(a + " " + b + " " + c + " " + d)
                    tokens_a = token_to_span_map_inverse[aList.index(a)]
                    tokens_b = token_to_span_map_inverse[aList.index(b)]
                    tokens_c = token_to_span_map_inverse[aList.index(c)]
                    tokens_d = token_to_span_map_inverse[aList.index(d)]
                    for token in tokens_a + tokens_b + tokens_c + tokens_d:
                        token_to_span_maps[token].append(len(aList) - 1)

            token_to_span_map = token_to_span_maps
        else:
            for key, value in token_to_span_map.items():
                token_to_span_map[key] = [value]

        return [aList, aList_flat_index, sentence, tokens, token_to_span_map]

    def splitter(self, aList):
        for i in range(1, 3):
            start = aList[0:i]
            end = aList[i:]
            yield (start, end)
            for split in self.splitter(end):
                result = [start]
                result.extend(split)
                yield result

    def chunk_sample(self, qid, claim, evidences_processed, num_retrieved_evidence):
        chunked_data = {}

        evidences = []
        for evidence in evidences_processed.split(" </s> ")[:num_retrieved_evidence]:
            evidence = evidence.replace("[ ", "").replace(" ]", " PPPPPPPPP") # Title + content
            evidence = evidence.split(" PPPPPPPPP")  # Remove this part, could use regex lol
            if len(evidence) > 1:  # Symmetric Fever does not have seperated title
                evidences.append(evidence[0].strip() + " " + evidence[1].strip())
            else:
                evidences.append(evidence[0].strip())

        claim_spans, flat_index, claim_sentence, claim_tokens, claim_token_to_span_map = self._chunk_flair(claim)
        claim_spans, options_claim, claim_token_to_span_map = self._find_all_options(
            claim_spans, flat_index, claim_token_to_span_map, claim_sentence,
        )
        # print(claim_spans)
        # print(options_claim)

        assert len(claim_spans) > 0, "No claim spans found for id {} and claim {} with evidence {}".format(
            qid, claim, evidences_processed
        )
        assert (
            len(options_claim) > 0
        ), "No options found for id {} and claim {} with evidence {}, and claim spans {}".format(
            qid, claim, evidences_processed, claim_spans
        )

        all_evidence_parses = []
        all_evidence_tokens = []
        for evidence in evidences:
            # Parse spacy dependency tree to get spans of different granulairy instead of brute force finding options
            chunked_evidence, _, _, evidence_tokens, _ = self._chunk_flair(evidence)
            all_evidence_parses.append(chunked_evidence)
            all_evidence_tokens.append(evidence_tokens)

        chunked_data["qid"] = qid
        chunked_data["claim_tokens"] = claim_tokens
        chunked_data["claim_tokens_to_span"] = claim_token_to_span_map
        chunked_data["claim_parses"] = claim_spans
        chunked_data["evidence_parses"] = all_evidence_parses
        chunked_data["evidence"] = evidences
        chunked_data["evidence_tokens"] = all_evidence_tokens
        chunked_data["claim"] = claim
        chunked_data["claim_parsed_hierarchy"] = options_claim

        return chunked_data

    def run_sentence_alignment(self, entry, arith_exps):
        dict_aligned = {
            "qid": entry["qid"],
            "claim_parsed": entry["claim_parses"],
            "claim_parsed_hierarchy": entry["claim_parsed_hierarchy"],
            "alignment": [],
        }
        # Tokenize sentences

        claim_parsed = entry["claim_parses"]
        src_sentence = entry["claim_tokens"]
        claim_tok_to_span_map = entry["claim_tokens_to_span"]


        entry_id = entry["qid"]
        for i, ev_sentence in enumerate(entry["evidence"]):
            span_alignments = {j: [] for j, x in enumerate(claim_parsed)}

            if self.alignment_mode in ["simalign", "awesomealign"]:
                tgt_sentence = entry["evidence_tokens"][i]
                alignments = self.model.get_word_aligns(src_sentence + ["."], tgt_sentence + ["."])
                already_mapped = {x: [] for x in claim_tok_to_span_map.keys()}

                # Remove duplicate mappings
                alignments_new = []
                for alignment in alignments:
                    if alignment[0] == len(src_sentence) or alignment[1] == len(tgt_sentence):
                        continue
                    if not tgt_sentence[alignment[1]] in already_mapped[alignment[0]]:
                        alignments_new.append(alignment)
                        already_mapped[alignment[0]].append(tgt_sentence[alignment[1]])

                alignments = alignments_new

                for alignment in alignments:
                    span_mappings = claim_tok_to_span_map[alignment[0]]
                    for span_map in span_mappings:
                        span_alignments[span_map].append(alignment[1])

                # Fill out evidence from min to max
                span_alignments_text = {}
                for key, value in span_alignments.items():
                    if not value:
                        evidence_options = [(i, x) for x in entry["evidence_parses"][i]]
                        evidence_span, _ = self.select_phrase(claim_parsed[key], evidence_options)
                        evidence_span = evidence_span
                    else:
                        # evidence_span = tgt_sentence[min(value): max(value) + 1]
                        evidence_span = [tgt_sentence[val] for val in value]
                        evidence_span = " ".join(evidence_span)
                    span_alignments_text[claim_parsed[key]] = (i, evidence_span)
                    # span_alignments_text.append((claim_parsed[key], " ".join(evidence_span)))

                dict_aligned["alignment"].append(list(span_alignments_text.items()))
            elif self.alignment_mode == "sentence_transformer":
                for span in claim_parsed:
                    evidence_options = [(i, x) for x in entry["evidence_parses"][i]]
                    selected_evidence_span, _ = self.select_phrase(span, evidence_options)
                    span_alignments[span] = selected_evidence_span
                dict_aligned["alignment"].append(list(span_alignments.items()))

        combined = [item for sublist in dict_aligned["alignment"] for item in sublist]
        combined_evidence = [item for sublist in entry["evidence_parses"] for item in sublist]

        combined_ordered = []
        for span in claim_parsed:
            if span == ".":
                continue
            all_found = set([])
            for element in combined:
                if span == element[0]:
                    all_found.add(element)

            relevant_arith_exp = " ".join([x[1] for x in arith_exps if x[0] in span])
            if len(all_found) > 1:
                selected_evidence_span, _ = self.select_phrase(span, [x[1] for x in list(all_found)])
                if self.replace_with_arith_exps and len(relevant_arith_exp) > 0 and "FILTER" in relevant_arith_exp.strip():
                    combined_ordered.append((span, relevant_arith_exp.strip()))
                else:
                    combined_ordered.append((span, selected_evidence_span))
            elif len(all_found) == 1:
                al = list(all_found)[0]
                if self.replace_with_arith_exps and len(relevant_arith_exp) > 0 and "FILTER" in relevant_arith_exp.strip():
                    combined_ordered.append((al[0], relevant_arith_exp.strip()))
                else:
                    combined_ordered.append((al[0], al[1][1]))
            else:
                # print("EMPTY:", span)
                value, _ = self.select_phrase(span, combined_evidence)
                if self.replace_with_arith_exps and len(relevant_arith_exp) > 0 and "FILTER" in relevant_arith_exp.strip():
                    combined_ordered.append((span, relevant_arith_exp.strip()))
                else:
                    combined_ordered.append((span, value))

        # if not self.replace_with_arith_exps: # If not replace, add arithexp as possible additional parses
        possible_parses = entry["claim_parsed_hierarchy"]
        # Apply ArithExps to options
        curr_span_id = len(claim_parsed)
        for arith_exp in arith_exps: # All combinations of arith exps on existing spans are applied for now.
            arith_exp_in_span = [i for i, x in enumerate(claim_parsed) if arith_exp[0] in x]
            arith_exp_in_span_dict =  {x: curr_span_id + i for i, x in enumerate(arith_exp_in_span)}
            curr_span_id += len(arith_exp_in_span_dict)

            possible_parses_new = copy.deepcopy(possible_parses)

            for possible_parse in possible_parses:
                for span in arith_exp_in_span_dict:
                    if span in possible_parse:
                        new_parse = [x if x != span else arith_exp_in_span_dict[x] for x in possible_parse]
                        possible_parses_new.append(new_parse)

            possible_parses = possible_parses_new

        for arith_exp in arith_exps: #extend alignment by arith exps
            combined_ordered += [(x, arith_exp[1]) for i, x in enumerate(claim_parsed) if arith_exp[0] in x]
            claim_parsed += [x for i, x in enumerate(claim_parsed) if arith_exp[0] in x]

        assert(curr_span_id) == len(combined_ordered), "Something wrong with applying Arith Exp. Lengths are not matching {} {}. Aligned {}".format(curr_span_id -1, len(combined_ordered), combined_ordered)

        dict_aligned["claim_parsed"] = claim_parsed
        dict_aligned["claim_parsed_hierarchy"] = possible_parses

        print(possible_parses)
        print(claim_parsed)

        dict_aligned["alignment"] = combined_ordered


        return dict_aligned

    def select_phrase(self, claim_phrase, evidence_phrases):
        # print(claim_phrase, evidence_phrases)
        if len(evidence_phrases) == 1:
            return [evidence_phrases[0][1], [1]]  # ENSURE ADJUSTED FOR ADDED WEIGHTS

        # Weights (Relevance) of evidence sentences decreases with retrieval position
        weights = []
        for i, sent in enumerate(evidence_phrases):
            if sent[0] == 0:
                weights.append(1)
            else:
                weights.append(i * 1)  # 0.8 for sentences, for tables it is all same relevance, so 1

        evidence_phrases = [x[1] for x in evidence_phrases]

        # print(evidence_phrases)

        # Tokenize sentences
        encoded_input_claim = self.phrase_tokenizer(
            [claim_phrase], padding=True, truncation=True, return_tensors="pt"
        ).to("cuda:0")
        encoded_input_evidence = self.phrase_tokenizer(
            evidence_phrases, padding=True, truncation=True, return_tensors="pt"
        ).to("cuda:0")

        # Compute token embeddings
        with torch.no_grad():
            model_output_claim = self.phrase_model(**encoded_input_claim)
            model_output_evidence = self.phrase_model(**encoded_input_evidence)

        # Perform pooling
        sentence_embeddings_claim = self._mean_pooling(model_output_claim, encoded_input_claim["attention_mask"])
        sentence_embeddings_evidence = self._mean_pooling(
            model_output_evidence, encoded_input_evidence["attention_mask"]
        )

        # Normalize embeddings
        # sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        # Compute cosine-similarities
        cosine_scores = self._cos_sim(sentence_embeddings_claim, sentence_embeddings_evidence)
        cosine_scores = cosine_scores.tolist()

        cosine_scores = cosine_scores[0]  # only consider single claim phrase
        cosine_scores = [x * weights[i] for i, x in enumerate(cosine_scores)]
        max_value = max(cosine_scores)
        max_index = cosine_scores.index(max_value)

        return [evidence_phrases[max_index], cosine_scores]

    def align_sample(self, qid, claim, evidence, arith_exps):
        chunked_sample = self.chunk_sample(qid, claim, evidence, self.num_retrieved_evidence)
        aligned_sample = self.run_sentence_alignment(chunked_sample, arith_exps)

        # print(aligned_sample)

        gc.collect()

        return aligned_sample


def load_data(dataset, split, use_retrieved_evidence, num_retrieved_evidence):
    if dataset == "danfever":
        in_file = "data/danfever/processed_validation_use_retr_False_retr_evidence_2_dp_True_alignment_mode_awesomealign_max_chunks_5_alignment_model_bert_finetuned_default_gold_no_nei_few_shot_False_4000_matching_method_mwmf_loose_matching_False.jsonl"
    else:
        in_file = "data/legacy/processed_{}_use_retr_{}_retr_evidence_{}_dp_True_concat_op_s.jsonl".format(
            split, use_retrieved_evidence, num_retrieved_evidence
        )

    data = []
    with open(in_file, "r") as f_in:
        lines = f_in.readlines()
        for line in lines:
            dictc = {}
            content = json.loads(line)
            qid = content["id"]

            # if int(qid) != 63575:
            #     continue

            dictc["id"] = qid
            dictc["claim"] = content["claim"]
            dictc["evidence"] = content["evidence"]
            dictc["verdict"] = content["verdict"]
            dictc["proof"] = content["proof"]
            data.append(dictc)

    return data