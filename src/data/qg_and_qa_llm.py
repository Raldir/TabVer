import json
import traceback
import re
import sys
from tqdm import tqdm
import gc

import torch
from transformers import AutoTokenizer
from guidance import models, gen, select


from src.data.template_formatter import TemplateFormatterLLM
from src.utils.util import postprocess_generation

class QuestionGeneratorLLM():

    def __init__(self, model, out_path):
        self.model_name =  model.split("/")[-1] if len(model.split("/")) == 2 else  model.split("/")[-2] # the latter is custom model so we do not want "checkpoint-60" to be the name

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if "Meta-Llama-3" in model:
            self.tokenizer.pad_token = "<|reserved_special_token_250|>" 
            self.tokenizer.pad_token_id = 128255
        else:
            self.tokenizer.pad_token = "[PAD]" 

        self.temperature = 1.0 
        self.top_p = 1.0
        self.max_seq_len = 4096
        self.max_batch_size = 1
        self.num_beams = 2
        self.max_new_tokens = 200

        self.template_formatter = TemplateFormatterLLM(scenario="question_generation")

        self.out_path_generated_questions = out_path

        self.pipeline = models.Transformers(model, torch_dtype=torch.float16, device_map="auto", echo=False)

        stop = [] #if stop is None else stop
        stop = list(set(stop + ["\n", "Ċ", "ĊĊ", "<0x0A>"])) # In Llama \n is <0x0[A>; In OPT \n is Ċ
        # stop = list(set(stop + ["\n", "\n\n\n", "\n\n"])) # In Llama \n is <0x0A>; In OPT \n is Ċ
        self.stop_token_ids = list(set([self.tokenizer.convert_tokens_to_ids(stop_token) for stop_token in stop] + [self.pipeline.model_obj.config.eos_token_id] + [self.tokenizer.eos_token_id]))
        self.stop_token_ids.remove(self.tokenizer.unk_token_id)


    def extract_question_answers(self, claim, response):
        def answer_has_numbers(answer):
            return bool(re.search(r"\b(?:[0-9]+|one|two|three|four|five|six|seven|eight|nine|ten|hundred|zero)\b", answer))

        print("-----------------------")
        # print("RESPONSE", response)
        claims_list = []
        print("RESPONSE RAW", response)
        # response = response.split("Questions:")[-1]
        # print(responses)
        questions, answers = [], []

        question_pattern = r'Questions:(.*?)Answers:(.*)'
        answer_pattern = r"Answers:(.*?)$"

        # Extracting questions
        questions_match = re.search(question_pattern, response, re.DOTALL)

        if questions_match:
            questions_text = questions_match.group(1).strip()
            questions_list = re.findall(r'\d+\.\s*(.*?\?)', questions_text, re.DOTALL)

            # Cleaning up question strings
            questions = [postprocess_generation(question.strip()) for question in questions_list]


        # Extracting answers
        answers_match = re.search(answer_pattern, response, re.DOTALL)
        if answers_match:
            answers_text = answers_match.group(1).strip()
            answers_list = re.findall(r'\d+\.(.*?)((?=\d+\. )|$)', answers_text, re.DOTALL)

            # Cleaning up answer strings
            answers = [postprocess_generation(answer[0].strip()) for answer in answers_list] # group 1 only
        
        min_len = min(len(questions), len(answers))
        questions = questions[:min_len]
        answers = answers[:min_len]

        assert len(questions) == len(answers), "Length of questions and answers do not match for {}".format(response)

        if "N/A" in questions or "N/A" in answers:
            "WHOPPPPS {} {}".format(questions, answers)
            return [], []
        else:
            filtered_questions = []
            filtered_answers = []
            for i, question in enumerate(questions):
                associated_answer = answers[i].strip()
                if associated_answer.lower() in claim.lower(): #answer_has_numbers(associated_answer) and 
                    filtered_answers.append(associated_answer)
                    filtered_questions.append(question)
            print(filtered_questions, filtered_answers)
            print("----------")
            return [filtered_questions, filtered_answers]


    def generate_sublists(self, lst):
        sublists = []
        n = len(lst)
        for i in range(n):
            for j in range(i + 1, n + 1):
                sublists.append(lst[i:j])
        return sublists


    def question_generation_numerical(self, data):
        with open(self.out_path_generated_questions, "w") as f_out:
            for entry in tqdm(data):
                input_prompt = self.template_formatter.apply_template_to_sample(claim=entry["claim"],questions=None, answers=None)
                print(input_prompt)

                cutoff = -1 if entry["claim"][-1] == "." else None
                substring_claim = self.generate_sublists(entry["claim"][:cutoff].split())
                substring_claim = ["{}".format(" ".join(x)) for x in substring_claim]

                try:
                    # Question generation
                    result = self.pipeline + f"{input_prompt} {gen(stop=':', max_tokens=100)}"
                    questions_num = len(re.findall(r'\d+\.\s*(.*?\?)', str(result), re.DOTALL))
                    if questions_num > 0: # Answer generation one by one constrained to viable options
                        for i in range(questions_num):
                            if i == 0:
                                result += f": {i+1}. {select(substring_claim)}"
                            else:
                                result += f" {i+1}. {select(substring_claim)}"
                        print(result)
                        questions, answers = self.extract_question_answers(entry["claim"], str(result))
                    else:
                        questions, answers = [], []
                except:
                    traceback.print_exception(*sys.exc_info())
                    questions, answers = self.fallback_inference(input_prompt, entry["claim"])
                entry["chunk_questions"] = questions
                entry["chunk_answers"] = answers

                f_out.write("{}\n".format(json.dumps(entry)))
        return data

    def fallback_inference(self, input_prompt, claim):
        questions, answers = [], []
        final_prompt = self.tokenizer(input_prompt, return_tensors="pt", truncation=True, max_length=self.max_seq_len - 1)
        final_prompt = final_prompt.to("cuda")

        results = self.pipeline.model_obj.generate(
            **final_prompt, # Only batch size of 1 supported now
            do_sample=False,
            num_beams=2,
            #top_k=10,
            temperature=self.temperature,
            top_p=self.top_p,
            num_return_sequences=1,
            eos_token_id=self.stop_token_ids,
            pad_token_id=self.tokenizer.eos_token_id,
            max_length=self.max_seq_len,
            max_new_tokens=200,
        )
        result = results[0]
        result = self.tokenizer.decode(result)
        try:
            questions, answers = self.extract_question_answers(claim, result)
        except:
            questions, answers = [], []
        return questions, answers


class QuestionAnsweringLLM():

    def __init__(self, model, out_path, allowed_functions):
        self.model_name =  model.split("/")[-1] if len(model.split("/")) == 2 else  model.split("/")[-2] # the latter is custom model so we do not want "checkpoint-60" to be the name
        self.tokenizer = AutoTokenizer.from_pretrained(model)

        if "Meta-Llama-3" in model:
            self.tokenizer.pad_token = "<|reserved_special_token_250|>" 
            self.tokenizer.pad_token_id = 128255
        else:
            self.tokenizer.pad_token = "[PAD]" 

        self.temperature = 1.0 #1.0
        self.top_p = 1.0 #0.9,
        self.max_seq_len = 4096 #2048
        self.max_batch_size = 1
        self.num_beams = 2
        self.max_new_tokens = 200

        print(allowed_functions)
        self.template_formatter = TemplateFormatterLLM(scenario="question_answering", allowed_functions=allowed_functions)

        self.out_path_generated_answers = out_path
        self.model = model
        # self.out_path_generated_answers = out_path.replace(".jsonl", "_generated_answers.jsonl")

        self.pipeline = models.Transformers(model, torch_dtype=torch.float16, device_map="auto", token="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", echo=False)

        stop = [] #if stop is None else stop
        stop = list(set(stop + ["\n", "Ċ", "ĊĊ", "<0x0A>"])) # In Llama \n is <0x0[A>; In OPT \n is Ċ
        # stop = list(set(stop + ["\n", "\n\n\n", "\n\n"])) # In Llama \n is <0x0A>; In OPT \n is Ċ
        self.stop_token_ids = list(set([self.tokenizer.convert_tokens_to_ids(stop_token) for stop_token in stop] + [self.pipeline.model_obj.config.eos_token_id] + [self.tokenizer.eos_token_id]))
        self.stop_token_ids.remove(self.tokenizer.unk_token_id)



    def extract_answer(self, response):
        response = response.split("Answer:")[-1]
        response = response.replace("<|im_end|>", "").replace("<|end_of_text|>", "") # Replace all EOS symbols
        print("[STARTRESPONSE] {} [ENDRESPONSE]".format(response))
            
        pattern = r'Extracted evidence from table:(.*?)Computation:(.*)Result:(.*)'

        match = re.findall(pattern, response, re.IGNORECASE)
        if match:
            evidence, _, answer = match[0]
        else:
            pattern = r'Extracted evidence from table:(.*?)Computation:(.*)'
            if match:
                evidence, _ = match[0]
                answer = "N/A"
            else:
                pattern = r'Extracted evidence from table:(.*?)$'
                if match:
                    evidence = match[0][0]
                    _, answer = "N/A"
                else:
                    evidence, _, answer = "N/A"

        answer = answer.strip()

        print(evidence,answer)
        print("-----------------")
        return [evidence, answer]

    def generate_sublists(self, lst):
        sublists = []
        n = len(lst)
        for i in range(n):
            for j in range(i + 1, n + 1):
                sublists.append(lst[i:j])
        return sublists

    def extract_numbers_from_evidence(self, evidence):
        # numbers = re.findall(r'\b\d+\.\d+|\b\d+\b', evidence)
        # Taken from https://stackoverflow.com/questions/5917082/regular-expression-to-match-numbers-with-or-without-commas-and-decimals-in-text.
        # pattern = re.compile(r"(?<!\S)(\d*\.?\d+|\d{1,3}(,\d{3})*(\.\d+)?)(?!\S)", re.MULTILINE)
        pattern = re.compile(r"(\d+(,\d+)*(\.\d+)?)", re.MULTILINE)
        numbers = re.findall(pattern, evidence)
        numbers = list(set([x[0] for x in numbers if x[0] != ""])) # remove duplicates
        return numbers

    def str_has_numbers(self, inputString):
        return any(char.isdigit() for char in inputString)

    def question_answering_numerical(self, data):
        with open(self.out_path_generated_answers, "w") as f_out:
            for i, entry in enumerate(tqdm(data)):
                evidence_extracted_llm = []
                answers = []
                evidence_numbers = self.extract_numbers_from_evidence(entry["table"])
                arithmetic_triggers = ["No"] + [value[0] for key, value in self.template_formatter.map_arithmetic_comp.items()]
                for question in entry["chunk_questions"]:
                    input_prompt = self.template_formatter.apply_template_to_sample(question=question, evidence_table=entry["table"], evidence_text=None, computation=None, arith_exp=None, num_in_context_samples=0)
                    curr_evidence_extracted_llm = "N/A"
                    curr_answer = "N/A"
                    print(input_prompt)
                    try: # Question answering. 
                        #Step 1: Evidence extraction
                        if evidence_numbers:
                            result = self.pipeline + f"{input_prompt}Extracted evidence from table: " + gen(stop_regex='(Computation:|\d)', max_tokens=100)
                            # Constrain evidence extraction to numbers occuring in table.
                            lookahead = result + gen(max_tokens=2, name="lookahead")
                            lookahead = lookahead["lookahead"]
                            remaining_loops = 150
                            while self.str_has_numbers(lookahead) and "\n" not in str(result).split("Answer:")[-1] and remaining_loops > 0:
                                result += f"{select(evidence_numbers)}"
                                result += gen(stop_regex='(Computation:|\d)', max_tokens=100)
                                lookahead = result + gen(max_tokens=2, name="lookahead")
                                lookahead = lookahead["lookahead"]
                                remaining_loops-=1
                                print(lookahead)
                                print(str(result).split("Answer")[-1])
                            if "\n" in str(result).split("Answer:")[-1]:
                                result = "\n".join(str(result).split("\n")[:-1])
                            elif remaining_loops == 0: # if was stuck in loop fallback inference
                                 raise Exception("The constrained evidence extraction step ended up degenerated, fallback to regular inference...")
                        else:
                            result = self.pipeline + f"{input_prompt}Extracted evidence from table: " + gen(stop_regex='Computation:', max_tokens=100)
                        # result = self.pipeline + f"{input_prompt} Extracted evidence from table:" + gen(stop_regex=':', max_tokens=200)
                        if "not sufficient information" not in str(result).lower():
                            # Step 2: Computation. Constrained by trigger words of functions.
                            result += f"Computation: {select(arithmetic_triggers, name='arithmetic_function')} " + gen(stop_regex=('Result|\n'), max_tokens=100)
                            # Step 3: ArithExp
                            viable_arithexp_answers = self.generate_sublists(str(result).split("Answer:")[-1].split(" "))
                            viable_arithexp_answers = ["{}".format(" ".join(x)) for x in viable_arithexp_answers]
                            viable_arithexp_answers = [x for x in viable_arithexp_answers if x != ""]
                            result += f"Result: {select(self.template_formatter.allowed_functions.split(', '))} {select(viable_arithexp_answers)}"
                            curr_evidence_extracted_llm, curr_answer = self.extract_answer(str(result))
                            del result
                    except:
                        traceback.print_exception(*sys.exc_info())
                        curr_evidence_extracted_llm, curr_answer = self.fallback_inference(input_prompt)
                    evidence_extracted_llm.append(curr_evidence_extracted_llm)
                    answers.append(curr_answer)
                entry["evidence_extracted_llm"] = evidence_extracted_llm
                entry["generated_answers"] = answers
                f_out.write("{}\n".format(json.dumps(entry)))
                self.pipeline.reset()

                if i % 50 == 0:
                    # self.pipeline.reset() # Try out to reset to fix memory issue
                    gc.collect()
                if i % 1000 == 0 and i !=0:
                    del self.pipeline
                    gc.collect()
                    self.pipeline = models.Transformers(self.model, torch_dtype=torch.float16, device_map="auto", token="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", echo=False)

                
        return data
    
    def fallback_inference(self, input_prompt):
        curr_evidence_extracted_llm, curr_answer = "N/A", "N/A"
        final_prompt = self.tokenizer(input_prompt, return_tensors="pt", truncation=True, max_length=self.max_seq_len - 1)
        final_prompt = final_prompt.to("cuda")

        results = self.pipeline.model_obj.generate(
            **final_prompt, # Only batch size of 1 supported now
            do_sample=False,
            num_beams=2,
            temperature=self.temperature,
            top_p=self.top_p,
            num_return_sequences=1,
            eos_token_id=self.stop_token_ids,
            pad_token_id=self.tokenizer.eos_token_id,
            max_length=self.max_seq_len,
            max_new_tokens=200,
        )
        result = results[0]
        result = self.tokenizer.decode(result)
        curr_evidence_extracted_llm, curr_answer = self.extract_answer(result)
        del result
        return curr_evidence_extracted_llm, curr_answer