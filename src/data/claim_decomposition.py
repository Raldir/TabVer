from transformers import AutoTokenizer
import transformers
import torch

import json
from tqdm import tqdm
import re
import os

from src.data.template_formatter import TemplateFormatterLLM
from src.constants import HF_ACCESS_TOKEN
from src.utils.util import postprocess_generation

class ClaimDecomposer(object):

    def __init__(self, model, dataset, input_path, output_name, num_samples=-1):
        self.input_path = input_path
        self.dataset = dataset
        self.model  = model
        self.num_samples = num_samples
        self.template_formatter = TemplateFormatterLLM(scenario="claim_decomposition")

        self.out_path = os.path.join("data", self.dataset, "arithmetic_expressions", "claim_decomposition", "claim_decomposition_{}.jsonl".format(output_name))


    def load_data(self):
        data = []
        with open(self.input_path, "r") as f_in:
            for line in f_in.readlines()[:self.num_samples]:
                content = json.loads(line)
                data.append(content)
        return data

    def extract_decomposed_claims(self, response):
        claims_list = []
        response = response.split("Subclaims:")[-1]
        claims = re.findall(r"\d\.(.*?)((?=\d\. )|$)", response)
        claims = [postprocess_generation(claim[0]) for claim in claims]
        return claims


    def decompose_claims(
        self,
        temperature: float = 1.0,
        top_p: float = 1.0, #0.9,
        max_seq_len: int = 512,
        max_batch_size: int = 8,
        max_gen_len = None,
        add_original_claim = False
    ):
        """
        Entry point of the program for generating text using a pretrained model.
        TODO: Constraint decoding. Each word has to be selected from claim itself or punctuation (.).
        """

        self.tokenizer = AutoTokenizer.from_pretrained(self.model, use_auth_token=HF_ACCESS_TOKEN) # load_in_8bit=True,
        if "Meta-Llama-3" in self.model:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer.pad_token = "[PAD]" 

        model_name = self.model.split("/")[-1] if len(self.model.split("/")) == 2 else  self.model.split("/")[-2] # the latter is custom model so we do not want "checkpoint-60" to be the name

        if "8x7" in model_name or "70" in model_name:
            self.pipeline = transformers.AutoModelForCausalLM.from_pretrained(
                self.model,
                device_map="auto",
                bnb_4bit_compute_dtype=torch.float16,
                load_in_4bit=True,
                token=HF_ACCESS_TOKEN
            )
        else:
            self.pipeline = transformers.AutoModelForCausalLM.from_pretrained(
                self.model,
                torch_dtype=torch.float16,
                # load_in_8bit=True,
                device_map="auto",
                token=HF_ACCESS_TOKEN
            )

        stop = [] #if stop is None else stop
        stop = list(set(stop + ["\n", "Ċ", "ĊĊ", "<0x0A>"])) # In Llama \n is <0x0[A>; In OPT \n is Ċ
        # stop = list(set(stop + ["\n", "\n\n\n", "\n\n"])) # In Llama \n is <0x0A>; In OPT \n is Ċ
        self.stop_token_ids = list(set([self.tokenizer.convert_tokens_to_ids(stop_token) for stop_token in stop] + [self.pipeline.config.eos_token_id] + [self.tokenizer.eos_token_id]))
        self.stop_token_ids.remove(self.tokenizer.unk_token_id)

        data = self.load_data()

        with open(self.out_path, "w") as f_out:
            for entry in tqdm(data):
                input_prompt = self.template_formatter.apply_template_to_sample(claim=entry["claim"],subclaims=None)

                final_prompt = self.tokenizer(input_prompt, return_tensors="pt", truncation=True, max_length=max_seq_len - 1)

                final_prompt = final_prompt.to("cuda")

                results = self.pipeline.generate(
                    **final_prompt, # Only batch size of 1 supported now
                    do_sample=False,
                    num_beams=2,
                    #top_k=10,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=1,
                    eos_token_id=self.stop_token_ids,
                    pad_token_id=self.tokenizer.eos_token_id,
                    max_length=max_seq_len,
                    max_new_tokens=200,
                )
                result = results[0]
                result = self.tokenizer.decode(result)
                print(result)
                decomposed_claims = self.extract_decomposed_claims(result)
                # In some instances, no suitable subclaims are found (specifically for Tabfact), resort back to original claim
                if not decomposed_claims or any(["Pavitra Hrudayalu has a soundtrack." in x for x in  decomposed_claims]):
                    decomposed_claims = [entry["claim"]]
                print(decomposed_claims)
                print("--------")

                entry["original_claim"] = entry['claim']
                qid = entry["id"]
                id_count = "1"
                for j, subclaim in enumerate(decomposed_claims[:9]):# Do not consider more than 9 suclaims
                    qid_new = int(str(qid) + "000" + id_count + "000")
                    id_count = str(int(id_count) + 1)
                    entry["id"] = qid_new
                    entry["claim"] = subclaim
                    f_out.write("{}\n".format(json.dumps(entry)))
                if add_original_claim:
                    f_out.write("{}\n".format(json.dumps(entry)))


if __name__ == "__main__":
    model_path = os.path.join("models", "claim_decomposition_Mistral-7B-OpenOrca_all_table_llm_train_on_entire_prompt_False/checkpoint-40")
    input_path = os.path.join("data", "feverous", "feverous_dev_filled_tables.jsonl")
    decomposer = ClaimDecomposer(model=model_path, dataset="feverous", input_path=input_path)
    decomposer.decompose_claims(max_seq_len=4096, top_p=1.0, temperature=1.0, max_batch_size=1, add_original_claim=False)