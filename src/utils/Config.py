import ast
import json
import os


class Config(object):
    def __init__(self, filenames=None, kwargs=None):
        # Experiment configs
        self.exp_dir = None
        self.exp_name = None
        self.seed = 42
        self.run_joint = False
        self.use_cached_data = False
        self.claim_decomposition = False
        self.use_fever_training = False

        # Model Configs
        self.model = "EncDec"
        self.max_seq_len = 512
        self.max_answer_choice_length = 150
        self.origin_model = "yuchenlin/BART0pp"  # "facebook/bart-large-mnli yuchenlin/BART0pp"  # "bigscience/T0_3B"
        self.load_weight = ""
        self.nli_file_path = "deberta-v3-nli_zero_shot_False_probabilities.json"
        self.gradient_checkpointing = True

        # Dataset Configs
        self.dataset = "fever"
        self.stratified_sampling = False
        self.num_classes = 3
        self.num_samples = 32
        self.batch_size = 8
        self.eval_batch_size = 32
        self.num_workers = 8
        self.negative_samples_ratio = 1 # Number of negative samples per span
        self.use_retrieved_evidence = (
            "False"  # Select from True False Only (True is first mode, where we use gold and fill up with retrieved)
        )
        self.num_retrieved_evidence = 2
        self.neg_token = "No" # TODO: Oracle, remove
        self.nei_token = "NOT ENOUGH INFO" # TODO: Oracle, remove
        self.zero_shot = False
        self.few_shot = False


        # Template Settings
        self.randomize_templates = True # Selects one template at random, following T-Few, performs similarly well to averaging over all, much faster.
        self.template_setting_id = 1
        self.num_questions = 1 # How many questions to consider, only when randomize_templates=False
        self.num_templates = 1 # How many templates to consider, only when randomize_templates=False
        # Specify concrete question or template to be used, -1 does not select one.
        self.question_id = -1
        self.template_id = -1

        # Alignment config
        self.alignment_model = "bert_finetuned_paper"
        self.matching_method = "mwmf"  # mwmf, inter, itermax
        self.sentence_transformer = "sentence-transformers/all-mpnet-base-v2"
        self.max_chunks = 6
        self.alignment_mode = "awesomealign"  # simalign sentence_transformer proofver
        self.loose_matching = False  #  
        self.dynamic_parsing = True
        self.dynamic_parsing_nei_threshold = 0.5 # probability score od independence predictions

        # ArithExp config
        self.use_tab_qa = False
        self.qg_model = "Open-Orca/Mistral-7B-OpenOrca"
        self.qa_model = "Open-Orca/Mistral-7B-OpenOrca"
        self.decomposition_model = "Open-Orca/Mistral-7B-OpenOrca"
        self.permissable_operators="all" # all arithmetic base

        self.debug = False

        # Compute backend configs
        self.compute_precision = "bf16"
        self.compute_strategy = "none"

        # Trainer configs
        self.num_steps = 15_000  # 100_000
        self.grad_accum_factor = 4
        self.val_check_interval = 3000
        self.eval_before_training = False  # True
        self.save_model = False
        self.save_step_interval = 20_000
        self.mc_loss = 1
        self.unlikely_loss = 1
        self.length_norm = 1
        self.split_choices_at = 10  # Whether to split the answer choices during eval to lower memory usage for datasets with lots of answer choices
        self.split_option_at_inference = False  # Whether to split the answer choices during eval to lower memory usage for datasets with lots of answer choices

        # Optimization configs
        self.optimizer = "adamw"
        self.lr = 5e-5
        self.trainable_param_names = ".*"
        self.scheduler = "linear_decay_with_warmup"
        self.warmup_ratio = 0.06
        self.weight_decay = 0.3
        self.scale_parameter = True
        self.grad_clip_norm = 1

        if filenames:
            for filename in filenames.split("+"):
                if not os.path.exists(filename):
                    filename = os.path.join(os.getenv("CONFIG_PATH", default="configs"), filename)

                self.update_kwargs(json.load(open(filename)), eval=False)
        if kwargs:
            self.update_kwargs(kwargs)

        self.set_exp_dir()

    def update_kwargs(self, kwargs, eval=True):
        for k, v in kwargs.items():
            if eval:
                try:
                    if "+" in v:  # Spaces are replaced via symbol
                        v = v.replace("+", " ")
                    else:
                        v = ast.literal_eval(v)
                except ValueError:
                    v = v
            else:
                v = v
            if not hasattr(self, k):
                raise ValueError(f"{k} is not in the config")
            setattr(self, k, v)

    def set_exp_dir(self):
        """
        Updates the config default values based on parameters passed in from config file
        """

        if self.exp_name is not None:
            self.exp_dir = os.path.join(os.getenv("OUTPUT_PATH", default="exp_out"), self.exp_name)
        else:
            self.exp_dir = os.getenv("OUTPUT_PATH", default="exp_out")
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)

        if self.exp_dir is not None:
            self.train_pred_file = os.path.join(self.exp_dir, "train_pred.txt")
            self.dev_pred_file = os.path.join(self.exp_dir, "dev_pred.txt")
            self.dev_score_file = os.path.join(self.exp_dir, "dev_scores.json")
            self.test_pred_file = os.path.join(self.exp_dir, "test_pred.txt")
            self.test_score_file = os.path.join(self.exp_dir, "test_scores.json")
            self.save_config(os.path.join(self.exp_dir, os.path.join("config.json")))
            self.finish_flag_file = os.path.join(self.exp_dir, "exp_completed.txt")

    def to_json(self):
        """
        Converts parameter values in config to json
        :return: json
        """
        return json.dumps(self.__dict__, indent=4, sort_keys=False)

    def save_config(self, filename):
        """
        Saves the config
        """
        with open(filename, "w+") as fout:
            fout.write(self.to_json())
            fout.write("\n")
