## Installation

Setup a new conda environment, e.g. `tabver` (tested only Python version 3.9)

```bash
conda create -n tabver python=3.9
conda activate qa_natver
```

Install all relevant dependencies:

```bash
python3 -m pip install -e ./
```

## Download Data and Models

### Data
Download the processed FEVEROUS data by running:

```
gdown --folder https://drive.google.com/drive/folders/1ZF6klh5HcKoJH6adnoH_w8v-EbyKq_Lx?usp=sharing
```

Alternatively, these files can be compiled manually by downloading the FEVEROUS data, including the SqLite database dump from https://fever.ai/dataset/feverous.html and running the script in `src.utils.insert_tables_into_feverous_json.py` with the correct paths. The output should produce two files `data/feverous/feverous_train_filled_tables.jsonl` and `data/feverous/feverous_dev_filled_tables.jsonl`.

### Models

Download the models by running the following script, which should create a folder `./models` with four models insine of it (`Mistral-7B-OpenOrca-LoRA-QG`, `Mistral-7B-OpenOrca-LoRA-QA`, `Mistral-7B-OpenOrca-LoRA-Decomp`, `TabVer-FlanT5-xl`): 

```bash
./bin/download_models.sh
```

The script 


#### Download alignment model

Either download the alignment model and place it in ```models/awesomealign``` from here:

`https://drive.google.com/file/d/1-391W3NKTlJLMpjUwjCv03H1Qk0ygWzC/view?usp=sharing`

or simply use the non-finetuned alignment model, by calling the config file `dynamic_awesomealign_bert_mwmf_coarse_only_retrieval_5_ev` when running FEVER below.

## Run FEVEROUS


With arguments being the processed data, the dataset, the environment, the model, the sample size, and the seed, run the following command to reproduce the main results from the paper on FEVEROUS:

```
./bin/run_few_shot.sh 5_ev_concat_all_table_mistralorca_trained_ops_all_updated feverous local flant5_xl_trained 128 42
```



### LOCAL DEBUG (NOT PART OF PUBLIC REPO)å

## Training (+saving model)
/bin/train_few_shot.sh 5_ev_concat_all_table_mistralorca_trained_decomposition_ops_all_updated feverous local_saving bart0 128 42

## Inference (+saving model)
./bin/run_few_shot.sh 5_ev_concat_all_table_mistralorca_trained_decomposition_ops_all_updated feverous local bart0_trained 128 42


## On HPC
add to env file.
ABSOLUTE_PATH = "/rds/user/rmya2/hpc-work/fact_checking_qa"

# Baselines

## Tapas
python3 -m src.models.baseline_tabular --dataset feverous --sample_size 10000000 --config_path tapas_config_no_tabfact_full_supervision

## LLM
python3 -m src.models.baseline_llm --dataset feverous --model models/verdict_prediction_Meta-Llama-3-8B-Instruct_all_table_llm_train_all_prompt_False_2024-04-29-16-25_seed_42/checkpoint-80 --seed 42 --evidence_mode all_table_llm

##  LPA installation

python3 -m src.lpa.model.py --do_train --do_val --analyze --data_dir data/lpa/preprocessed_data_program_feverous/ --output_dir data/lpa/checkpoints_feverous/
python3 -m src.lpa.model.py --do_test --resume --analyze --data_dir data/lpa/preprocessed_data_program_feverous/ --output_dir data/lpa/checkpoints_feverous/
python3 -m src.lpa.evaluate_lpa

## SASP installation

Install packages in https://github.com/facebookresearch/TaBERT/blob/main/scripts/env.yml
Install packages in https://github.com/pcyin/pytorch_neural_symbolic_machines/blob/master/data/env.yml
Install pip install --editable=git+https://github.com/huggingface/transformers.git@372a5c1ceec49b52c503707e9657bfaae7c236a0#egg=pytorch_pretrained_bert
git clone TaBERT
then cd TaBERT 
pip install --editable .

./bin/prep_sasp_data.sh


### DEFAULT existing tabfact data training
python3 -m table.experiments train --seed=0 --cuda --config=data/config/config.vanilla_bert_mod.json --work-dir=runs/demo_run --extra-config='{}'

### FEVEROUS training data
python3 -m table.experiments train --seed=0 --cuda --config=data/config/config.vanilla_bert_feverous_train.json --work-dir=runs/run_feverous --extra-config='{}'

### Running Tabfact model on FEVEROUS
python -m table.experiments test --cuda --eval-batch-size=2 --eval-beam-size=4 --save-decode-to=runs/demo_run/test_result_feverous.json --model=./runs/demo_run/model.best.bin --test-file=/home/rmya2/scratch/QA-NatVer/data/sasp/data_shard_with_dep/dev.jsonl --extra-config='{"table_file": "/home/rmya2/scratch/QA-NatVer/data/sasp/tables.jsonl"}'


### Running FEVEROUS model on FEVEROUS
python -m table.experiments test --cuda --eval-batch-size=2 --eval-beam-size=4 --save-decode-to=runs/run_feverous/test_result_feverous.json --model=./runs/run_feverous/model.best.bin --test-file=/home/rmya2/scratch/QA-NatVer/data/sasp/data_shard_with_dep/dev.jsonl --extra-config='{"table_file": "/home/rmya2/scratch/QA-NatVer/data/sasp/tables.jsonl"}'

### Running FEVEROUS model on Tabfact
python -m table.experiments test --cuda --eval-batch-size=2 --eval-beam-size=4 --save-decode-to=runs/run_feverous/result_tabfact.json --model=./runs/run_feverous/model.best.bin --test-file=data/tabfact/data_shard/dev.jsonl --extra-config='{"table_file": "data/tabfact/tables.jsonl"}'
