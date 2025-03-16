# Step 1: Create the models directory
mkdir -p models
mkdir -p models/awesomealign

# Step 2: Download the files using gdown and save them in the models directory
gdown --folder https://drive.google.com/drive/folders/16NSIq9CTXdtYJ420_t_prynnZK32drrX?usp=sharing -O models/Mistral-7B-OpenOrca-LoRA-QG
gdown --folder https://drive.google.com/drive/folders/1GAYB1RD0gtk_10ZMp2ryhLkOjb8-Fn2r?usp=sharing -O models/Mistral-7B-OpenOrca-LoRA-QA
gdown --folder https://drive.google.com/drive/folders/1giQTVhupb7nWMtoIMCZJgcx2CPOikFSr?usp=sharing -O models/Mistral-7B-OpenOrca-LoRA-Decomp
gdown --folder https://drive.google.com/drive/folders/10kmq-XmyEvqsMme2J3GxIxX6KALSiziK?usp=sharing -O models/awesomealign/paper
gdown --folder https://drive.google.com/drive/folders/1UbOW8CXT1cgpT54cMcr1ettFfzdIngp1?usp=sharing -O models/TabVer_FlanT5-xl/42

