import argparse
import os
from datetime import datetime

import torch
from pytorch_lightning import Trainer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.utils.util import ROOT_DIR
from src.data.data_module_joint import FinetuneDataModuleJoint

from src.evaluation.evaluate_natop import EvaluatorNatop
from src.evaluation.evaluate_natop_decomposition import EvaluatorNatopDecomposition
from src.evaluation.evaluate_verdict import EvaluatorVerdict
from src.evaluation.evaluate import Evaluator
from src.evaluation.evaluate_decomposition import EvaluatorDecomposition
from src.lit_module import LitModule
from src.models.tfew import EncoderDecoder
from src.utils.Config import Config
from src.utils.util import ParseKwargs, set_seeds


def get_transformer(config, load_model=False):
    # Selected model does not matter when loading from cache. Just filler.
    tokenizer = AutoTokenizer.from_pretrained(config.origin_model)
    tokenizer.model_max_length = config.max_seq_len
    if load_model:
        model = AutoModelForSeq2SeqLM.from_pretrained(config.origin_model, low_cpu_mem_usage=True)
        return tokenizer, model
    return tokenizer


def main(config):
    """
    Trains the model

    :param config:
    :return:
    """
    assert config.load_weight != "", "Need to specify a model to load..."

    tokenizer = get_transformer(config, load_model=False)

    datamodule_natop = FinetuneDataModuleJoint(config, tokenizer, mode="natop")
    datamodule_verdict = FinetuneDataModuleJoint(config, tokenizer, mode="verdict")

    if config.claim_decomposition and "generation" in config.claim_decomposition:
        evaluator_natop = EvaluatorNatopDecomposition(config, datamodule_natop)
    else:
        evaluator_natop = EvaluatorNatop(config, datamodule_natop)

    evaluator_verdict = EvaluatorVerdict(config, datamodule_verdict)

    _, model = get_transformer(config, load_model=True)

    model = EncoderDecoder(config, tokenizer, model)

    litmodule = LitModule(config, model, datamodule_natop, evaluator_natop)
    litmodule_verdict = LitModule(config, model, datamodule_verdict, evaluator_verdict)

    if (config.use_cached_data and os.path.exists(os.path.join(ROOT_DIR, "exp_out", config.exp_name))):
        print("Loading predictions from cache...")
        datamodule_natop.setup("validation")
        if config.claim_decomposition and "generation" in config.claim_decomposition:
            evaluator = EvaluatorDecomposition(config, datamodule_natop)
            evaluator.run_cached_data()
        else:
            evaluator = Evaluator(config, datamodule_natop)
            evaluator.run_cached_data()
    else:
        trainer = Trainer(
            enable_checkpointing=False,
            accelerator="gpu",
            devices=1,
            precision=config.compute_precision,
            strategy=config.compute_strategy if config.compute_strategy != "none" else "auto",
        )

        # Generate natop probability score and verdict score
        trainer.test(litmodule, datamodule_natop)
        trainer.test(litmodule_verdict, datamodule_verdict)

        if config.claim_decomposition and "generation" in config.claim_decomposition:
            evaluator = EvaluatorDecomposition(config, datamodule_natop)
        else:
            evaluator = Evaluator(config, datamodule_natop)
        evaluator.run_cached_data()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_files", required=True)
    parser.add_argument("-k", "--kwargs", nargs="*", action=ParseKwargs, default={})
    args = parser.parse_args()

    config = Config(args.config_files, args.kwargs)
    print(f"Start experiment {config.exp_name}")
    # Setup config
    assert config.compute_strategy in ["none", "ddp", "deepspeed_stage_3_offload", "deepspeed_stage_3"]

    print(config.to_json())

    set_seeds(config.seed)
    main(config)