# Law Fieldlab Project Setup Instructions
The purpose of this repository is to improve Llama 2 using finetuning and RAG. To use this code a CUDA version higher than 10.3 (Ideally 11.3) is required. Furthermore, the user should have a Hugginface and Wandb account (including access tokens for both). To get access to Llama 2 please visit: https://ai.meta.com/resources/models-and-libraries/llama-downloads/ and accept the terms and conditions. Then go to Huggingface and request access to the repository: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf. The email used when requesting access must be the same as the email from the Huggingface account.


## Clone the Repository
```
git clone https://github.com/Dimedici1/law_fieldlab.git
```

## Save Llama 2 Model
Navigate to save_model directory and clone the model from Huggingface
```
cd law_fieldlab/save_model
sudo apt-get install git-lfs
git lfs install
git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

```
Provide your Huggingface Username and Access Token. The Wandb token should be saved as well.

## Finetune Model
Clone the LMFlow repository (https://github.com/OptimalScale/LMFlow)
```
cd ..
git clone -b v0.0.5 https://github.com/OptimalScale/LMFlow.git
cd LMFlow
conda create -n lmflow python=3.9 -y
conda activate lmflow
conda install mpi4py
bash install.sh

```

Make sure to install all requirements by running
```
pip install -e .
```
There are a few problems that need to be sorted before finetuning. Run:

```
pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly
pip install datasets==2.14.6

```

Run the finetuning scripts (Here with example alpaca data from LMFlow):
```
mv $HOMElaw_fieldlab/qa_finetuning.json $HOME/law_fieldlab/LMFlow/data/qa_finetune/train/

./scripts/run_finetune_with_lora.sh \
  --model_name_or_path ${HOME}law_fieldlab/save_model/Llama-2-7b-chat-hf/ \
  --dataset_path data/qa_finetune/train \
  --output_lora_path ${HOME}law_fieldlab/run_model_files/output_models/finetuned_model

./scripts/run_finetune_with_lora_save_aggregated_weights.sh \
  --model_name_or_path ${HOME}law_fieldlab/save_model/Llama-2-7b-chat-hf/ \
  --dataset_path data/qa_finetune/train \
  --output_model_path ${HOME}law_fieldlab/run_model_files/output_models/finetuned_model

```
## Create the Database
To create the database keep the lmflow environment and run:
```
pip install langchain
pip install sentence-transformers
pip install chromadb
pip install bs4

```

Update the link collection with relevant links from EUR-Lex that contain CELEX codes. Alternatively, the code can be adapted to accept any form of string input. Please only select a part of the provided links. Trying to save all of them does not work.
```
cd ..
cd create_database
nano link_collection.py

```
Run the create_database.py file to store the vectorized data
```
python create_database.py
```
## Run Model Files
```
cd $HOME/law_fieldlab/run_model_files
python examples/chatbot_gradio.py --deepspeed $HOME/law_fieldlab/run_model_files/configs/ds_config_chatbot.json --model_name_or_path output_models/finetuned_model --max_new_tokens 500

```
