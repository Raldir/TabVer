import json
import os

import numpy as np

from src.utils.util import ROOT_DIR

from src.constants import NATOPS_TEXT, NATOPS_TO_TEXT_MAP, NATOPS_WO_INDEP, FUNCTIONS_ALL, FUNCTIONS_ARITHMETIC, FUNCTIONS_BASE


class TemplateFormatter(object):
    def __init__(
        self,
        setting_nr,
        neg_token,
        num_templates=5,
        num_questions=5,
        template_id=0,
        question_id=0,
        randomize=True,
    ):
        self.question_types = NATOPS_TEXT
        self.op_to_type_map = NATOPS_TO_TEXT_MAP  # what about #
        self.neg_token = neg_token
        self.data_path = os.path.join(ROOT_DIR, "data", "handwritten_questions", "setting" + str(setting_nr))

        self.randomize = randomize
        self.template_id = template_id
        self.question_id = question_id

        self.questions = {
            q_type: self._read_questions_for_operator(q_type, num_questions) for q_type in self.question_types
        }
        print(self.questions)

    def _read_questions_for_operator(self, operator, num_questions):
        questions = []
        file = os.path.join(self.data_path, operator + ".csv")
        with open(file, "r") as f_in:
            lines = f_in.readlines()
            for line in lines:
                questions.append(line.strip())
        if self.randomize:
            return [np.random.choice(questions)]
        elif self.question_id >= 0:
            return [questions[self.question_id]]
        else:
            return questions[:num_questions]

    def _apply_component_to_templates(self, templates, component, component_name):
        applied_templates = []
        for template in templates:
            for question in component:
                applied_templates.append(template.replace("{{" + component_name + "}}", question))

        return applied_templates

    def apply_templates_to_sample(self, claim, claim_span, evidence, ev_span, operator):
        claim = [claim]
        evidence = [evidence]
        claim_span = [claim_span]
        applied_templates = self._apply_component_to_templates(self.templates, claim, "claim")
        applied_templates = self._apply_component_to_templates(applied_templates, evidence, "evidence")
        applied_templates = self._apply_component_to_templates(
            applied_templates, self.questions[self.op_to_type_map[operator]], "question"
        )

        # Span must be inserted after question since the question specifies the claim span position
        applied_templates = self._apply_component_to_templates(applied_templates, claim_span, "span")

        answer_list = [ev_span] * len(applied_templates)

        return [applied_templates, answer_list]

    def apply_templates_to_sample_all_ops(self, claim_span, ev_span, operator, claim, evidence):
        claim_span = [claim_span]
        ev_span = [ev_span]
        claim = [claim.strip()]
        evidence = [evidence.strip()]
        ops = NATOPS_WO_INDEP

        applied_templates_collection = []
        answer_list = []

        for op in ops:
            applied_templates_op = self._apply_component_to_templates(
                    self.questions[self.op_to_type_map[op]], claim_span, "span"
                )
            applied_templates_op = self._apply_component_to_templates(applied_templates_op, ev_span, "evidence")
            applied_templates_collection.append(applied_templates_op)

        for i in range(len(NATOPS_WO_INDEP)):
            if NATOPS_WO_INDEP[i] == operator:
                answer_list += ["Yes"] * len(applied_templates_collection[i])
            else:
                answer_list += ["No"] * len(applied_templates_collection[i])

        applied_templates = [item for sublist in applied_templates_collection for item in sublist]

        return [applied_templates, answer_list]



class TemplateFormatterLLM(object):

    def __init__(self, scenario, allowed_functions=None):
        assert scenario in ["verdict_prediction", "question_generation", "question_answering", "claim_decomposition"], "Scenario not recognized: {}.".format(scenario)
        self.scenario = scenario
        if allowed_functions == "all":
            self.allowed_functions = ", ".join(FUNCTIONS_ALL)
        elif allowed_functions == "arithmetic":
            self.allowed_functions = ", ".join(FUNCTIONS_ARITHMETIC)
        elif allowed_functions == "base":
            self.allowed_functions = ", ".join(FUNCTIONS_BASE)

        if scenario == "verdict_prediction":
            prompt_file = "baseline_verdict.json"
        elif scenario == "claim_decomposition":
            prompt_file = "claim_decomposition.json"
        elif scenario == "question_generation":
            prompt_file = "question_generation.json"
        elif scenario == "question_answering":
            prompt_file = "question_answering.json"
            self.map_arithmetic_comp = {"FILTER": ("Filtering", ",", " to"), "SUM": ("Adding", " + ", "="), "DIFF": ("Substracting", " - ", "="), "COMP": ("Comparing (Substracting)", " - ", "="), "COUNT": ("Counting the items", ",", "results in"), "AVERAGE": ("Averaging", ", ", "="), "MIN": ("Mininum", ", ", "="), "MAX": ("Maximum", ", ", "="), "SUPER": ("Superlative (deviation from zero)", " - ", "=")}
            self.map_arithmetic_comp_inv = {value[0]: key for key, value in self.map_arithmetic_comp.items()}
            self.map_arithmetic_comp_inv["No"] = "FILTER"

        prompt_path = os.path.join("data", "handwritten_questions", prompt_file)

        with open(prompt_path, "r") as f_in:
            self.prompts = json.load(f_in)

        self.response_template = self.prompts["response_template"]
    
    def apply_template_to_sample_verdict_pred(self, claim, evidence, answer, num_in_context_samples=0):
        if answer is None:
            answer = ""
        input_prompt = self.prompts["context_format"].replace("{INST}", self.prompts["task_description"])
        input_prompt = input_prompt.format(claim, evidence, answer)
        
        if num_in_context_samples > 0:
            examples = []
            for entry in self.prompts["examples"]:
                example = self.prompts["context_format"].replace("{INST}", self.prompts["task_description"])
                example = example.format(entry["claim"], "{}\n{}".format(entry["table_title"], entry["table"]), entry["label"])
                example += self.prompts["demo_sep"]
                examples.append(example)
            examples = "\n".join(examples)
            input_prompt = examples + input_prompt
        return input_prompt
    
    def apply_template_to_sample_claim_decomposition(self, claim, subclaims, num_in_context_samples=0):
        if subclaims is None:
            subclaims = ""
        else:
            subclaims = "".join(["{}. {}".format(i+1, x) for i,x in enumerate(subclaims)])
        input_prompt = self.prompts["context_format"].replace("{INST}", self.prompts["task_description"])
        input_prompt = input_prompt.format(claim, subclaims)
        
        if num_in_context_samples > 0:
            examples = []
            for entry in self.prompts["examples"]:
                example = self.prompts["context_format"].replace("{INST}", self.prompts["task_description"])
                example_subclaims = "".join(["{}. {}".format(i, x) for i,x in enumerate(entry["subclaims"])])
                example = example.format(entry["claim"], example_subclaims)
                example += self.prompts["demo_sep"]
                examples.append(example)
            examples = "\n".join(examples)
            input_prompt = examples + input_prompt
        return input_prompt

    def apply_template_to_question_generation(self, claim, questions, answers, num_in_context_samples=0):
        if questions is None:
            combined = ""
        else:
            questions = " ".join(["{}. {}".format(i+1, x) for i,x in enumerate(questions)])
            answers = " ".join(["{}. {}".format(i+1, x) for i,x in enumerate(answers)])
            combined = questions + " Answers: " + answers

        input_prompt = self.prompts["context_format"].replace("{INST}", self.prompts["task_description"])
        input_prompt = input_prompt.format(claim, combined)
        
        if num_in_context_samples > 0:
            examples = []
            for entry in self.prompts["examples"]:
                example = self.prompts["context_format"].replace("{INST}", self.prompts["task_description"])
                example_questions = " ".join(["{}. {}".format(i+1, x) for i,x in enumerate([entry["question"]])])
                example_answers = " ".join(["{}. {}".format(i+1, x) for i,x in enumerate([entry["answer"]])])
                example_combined = example_questions + " Answers: " + example_answers
                example = example.format(entry["claim"], example_combined)
                example += self.prompts["demo_sep"]
                examples.append(example)
            examples = "\n".join(examples)
            input_prompt = examples + input_prompt
        return input_prompt

    def apply_template_to_question_answering(self, question, evidence_table, evidence_text, computation, arith_exp, num_in_context_samples=0):
        def format_computation(exp, computation_map):
            func_name = computation_map[0]
            func_arguments = computation_map[1:]
            formatting = self.map_arithmetic_comp[func_name]
            # computation_text = "{}({}) {} {}".format(formatting[0], ", ".join(func_arguments), "=", exp.split(" ")[-1])
            computation_text = "{} {} {} {}".format(formatting[0], formatting[1].join(func_arguments), formatting[2], exp.split(" ")[-1])
            return computation_text

        def format_answer(evidence_text, computation, arith_exp):
            answer = self.prompts["answer_format"].replace("{EVIDENCE}", evidence_text)
            if arith_exp not in computation:
                overall_computation_text = "No computation required."
            else:
                overall_computation_text = []
                computation_map = computation[arith_exp]
                for ele in computation_map: # Only for second level, technically can be recursive for infinite depth
                    if ele in computation:
                        computation_text = format_computation(ele, computation[ele])
                        overall_computation_text.append(computation_text)
                computation_text = format_computation(arith_exp, computation_map)
                overall_computation_text.append(computation_text)
                overall_computation_text = ". ".join(overall_computation_text)
            
            answer = answer.replace("{COMPUTATION}", overall_computation_text)
            answer = answer.replace("{ARITHEXP}", arith_exp)
            
            return answer

        if evidence_text is None:
            combined = ""
        else:
            combined = format_answer(evidence_text, computation, arith_exp)

        task_description = self.prompts["task_description"].replace("{FUNCTIONS}", self.allowed_functions)
        input_prompt = self.prompts["context_format"].replace("{INST}", task_description)
        input_prompt = input_prompt.format(question, evidence_table, combined)
        
        if num_in_context_samples > 0:
            examples = []
            for entry in self.prompts["examples"]:
                example = self.prompts["context_format"].replace("{INST}", task_description)
                example_answer = self.prompts["answer_format"].replace("{EVIDENCE}", entry["explanation"]).replace("{COMPUTATION}", entry["computation"]).replace("{ARITHEXP}", entry["arith_exp"])
                example = example.format(entry["question"], entry["table"], example_answer)
                example += self.prompts["demo_sep"]
                examples.append(example)
            examples = "\n".join(examples)
            input_prompt = examples + input_prompt

        return input_prompt


    def apply_template_to_sample(self, **kwargs):
        scenario_function_mapping = {"verdict_prediction": "apply_template_to_sample_verdict_pred", "claim_decomposition": "apply_template_to_sample_claim_decomposition", "question_generation": "apply_template_to_question_generation", "question_answering": "apply_template_to_question_answering", "question_answering": "apply_template_to_question_answering"}

        template_function = getattr(self, scenario_function_mapping[self.scenario])
        input_prompt= template_function(**kwargs)
        return input_prompt



