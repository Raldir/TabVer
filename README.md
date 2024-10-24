This repository maintains the core code of TACL 2024 paper [TabVer: Tabular Fact Verification with Natural Logic](XXXX).

> Fact verification on tabular evidence incentivises the use of symbolic reasoning models
where a logical form is constructed (e.g. a LISP-style program), providing greater verifiability than fully neural approaches. However, these logical forms typically rely on well-formed tables, restricting their use in many scenarios. An emerging symbolic rea-
soning paradigm for textual evidence focuses on natural logic inference, which constructs
proofs by modelling set-theoretic relations between a claim and its evidence in natural language. This approach provides flexibility and transparency but is less compatible with tabular evidence since the relations do not extend to arithmetic functions. We propose a set-theoretic interpretation of numerals and arithmetic functions in the context of natural logic, enabling the integration of arithmetic expressions in deterministic proofs. We leverage large language models to generate arithmetic expressions by generating questions about salient parts of a claim which are answered by executing appropriate functions on tables. In a few-shot setting on FEVEROUS, we achieve an accuracy of 71.4, outperforming both fully neural and
symbolic reasoning models by 3.4 points. When evaluated on TabFact without any further training, our method remains competitive with an accuracy lead of 0.5 points.

## Installation

Setup a new conda environment, e.g. `tabver` (tested only Python version 3.9)

```bash
conda create -n tabver python=3.9
conda activate tabver
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

## Run FEVEROUS


With arguments being the processed data, the dataset, the environment, the model, the sample size, and the seed, run the following command to reproduce the main results from the paper on FEVEROUS:

```
./bin/run_few_shot.sh default feverous local flant5_xl_trained 128 42
```

## Contact

If you are facing any issues running the code or have suggestions for improvements, please either open a Github issue or send me an e-mail. 
