# Law Fieldlab Project Setup Instructions
The purpose of this repository is to improve Llama 2 using finetuning and RAG. To use this code a CUDA version higher than 10.3 (Ideally 11.3) is required. Furthermore, the user should have a Hugginface and Wandb account (including access tokens for both). To get access to Llama 2 please visit: https://ai.meta.com/resources/models-and-libraries/llama-downloads/ and accept the terms and conditions. Then go to Huggingface and request access to the repository: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf. The email used when requesting access must be the same as the email from the Huggingface account.


## Clone the Repository
```
git clone https://github.com/Dimedici1/law_fieldlab.git
```

## Save Llama 2 Model
Navigate to save_model directory and clone the model from Huggingface
```
cd $HOME/law_fieldlab/save_model
sudo apt-get install git-lfs
git lfs install
git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

```
Provide your Huggingface Username and Access Token. The Wandb token should be saved as well.

## Install LMFlow repository and lmflow environment
Clone the LMFlow repository (https://github.com/OptimalScale/LMFlow)
```
cd $HOME/law_fieldlab/
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

## Create the Database
To create the database keep the lmflow environment and run:
```
pip install langchain
pip install sentence-transformers
pip install chromadb
pip install bs4
pip install pypdf

```

Update the link collection with relevant links from EUR-Lex that contain CELEX codes. Alternatively, the code can be adapted to accept any form of string input. Please only select a part of the provided links. Trying to save all of them does not work.
```
cd $HOME/law_fieldlab/create_database/
python create_database.py
```
## Run Model Files

Run regular model
```
cd $HOME/law_fieldlab/run_model_files
python examples/chatbot_gradio.py --deepspeed $HOME/law_fieldlab/run_model_files/configs/ds_config_chatbot.json --model_name_or_path $HOME/law_fieldlab/save_model/Llama-2-7b-chat-hf --max_new_tokens 700

```

## Testing

Open inferencer.py from LMFLow and make the max_token and temperature variable
```
cd $HOME/law_fieldlab/LMFlow/src/lmflow/pipeline/
nano inferencer.py
```

Give the sh files permission
```
cd $HOME/law_fieldlab/run_model_files/scripts/zero_shot_prompting/
chmod +x run_chatbot_base.sh
chmod +x run_chatbot_filtered_mmr.sh
chmod +x run_chatbot_filtered_similarity.sh
chmod +x run_chatbot_mmr.sh
chmod +x run_chatbot_similarity.sh
cd $HOME/law_fieldlab/run_model_files/scripts/few_shot_prompting/
chmod +x run_chatbot_base.sh
chmod +x run_chatbot_filtered_mmr.sh
chmod +x run_chatbot_filtered_similarity.sh
chmod +x run_chatbot_mmr.sh
chmod +x run_chatbot_similarity.sh
cd $HOME/law_fieldlab/run_model_files/scripts/chain_of_thought/
chmod +x run_chatbot_base.sh
chmod +x run_chatbot_filtered_mmr.sh
chmod +x run_chatbot_filtered_similarity.sh
chmod +x run_chatbot_mmr.sh
chmod +x run_chatbot_similarity.sh
cd $HOME/law_fieldlab/run_model_files/scripts/self_consistency/
chmod +x run_chatbot_base.sh
chmod +x run_chatbot_filtered_mmr.sh
chmod +x run_chatbot_filtered_similarity.sh
chmod +x run_chatbot_mmr.sh
chmod +x run_chatbot_similarity.sh
```
Run all testing files
```
cd $HOME/law_fieldlab/run_model_files
./scripts/zero_shot_prompting/run_chatbot_base.sh $HOME/law_fieldlab/save_model/Llama-2-7b-chat-hf
./scripts/zero_shot_prompting/run_chatbot_mmr.sh $HOME/law_fieldlab/save_model/Llama-2-7b-chat-hf
./scripts/zero_shot_prompting/run_chatbot_similarity.sh $HOME/law_fieldlab/save_model/Llama-2-7b-chat-hf
./scripts/zero_shot_prompting/run_chatbot_filtered_mmr.sh $HOME/law_fieldlab/save_model/Llama-2-7b-chat-hf
./scripts/zero_shot_prompting/run_chatbot_filtered_similarity.sh $HOME/law_fieldlab/save_model/Llama-2-7b-chat-hf
./scripts/few_shot_prompting/run_chatbot_base.sh $HOME/law_fieldlab/save_model/Llama-2-7b-chat-hf
./scripts/few_shot_prompting/run_chatbot_mmr.sh $HOME/law_fieldlab/save_model/Llama-2-7b-chat-hf
./scripts/few_shot_prompting/run_chatbot_similarity.sh $HOME/law_fieldlab/save_model/Llama-2-7b-chat-hf
./scripts/few_shot_prompting/run_chatbot_filtered_mmr.sh $HOME/law_fieldlab/save_model/Llama-2-7b-chat-hf
./scripts/few_shot_prompting/run_chatbot_filtered_similarity.sh $HOME/law_fieldlab/save_model/Llama-2-7b-chat-hf
./scripts/chain_of_thought/run_chatbot_base.sh $HOME/law_fieldlab/save_model/Llama-2-7b-chat-hf
./scripts/chain_of_thought/run_chatbot_mmr.sh $HOME/law_fieldlab/save_model/Llama-2-7b-chat-hf
./scripts/chain_of_thought/run_chatbot_similarity.sh $HOME/law_fieldlab/save_model/Llama-2-7b-chat-hf
./scripts/chain_of_thought/run_chatbot_filtered_mmr.sh $HOME/law_fieldlab/save_model/Llama-2-7b-chat-hf
./scripts/chain_of_thought/run_chatbot_filtered_similarity.sh $HOME/law_fieldlab/save_model/Llama-2-7b-chat-hf
./scripts/self_consistency/run_chatbot_base.sh $HOME/law_fieldlab/save_model/Llama-2-7b-chat-hf
./scripts/self_consistency/run_chatbot_mmr.sh $HOME/law_fieldlab/save_model/Llama-2-7b-chat-hf
./scripts/self_consistency/run_chatbot_similarity.sh $HOME/law_fieldlab/save_model/Llama-2-7b-chat-hf
./scripts/self_consistency/run_chatbot_filtered_mmr.sh $HOME/law_fieldlab/save_model/Llama-2-7b-chat-hf
./scripts/self_consistency/run_chatbot_filtered_similarity.sh $HOME/law_fieldlab/save_model/Llama-2-7b-chat-hf

```
The files will be stored in the testing folder.

# Example Finetuning

Here is an example of how finetuning could be performed. This is not relevant for the rest of the repository.

### Create Finetuning Dataset
Move to finetuning_dataset directory and run generate_examples.py
```
cd $HOME/law_fieldlab/old_files/finetuning/finetuning_dataset
python generate_examples.py
```
Alternatively the alpaca finetuning can be used
```
cd $HOME/law_fieldlab/old_files/finetuning/finetuning_dataset
python alpaca_finetuning.py
```

### Finetune
Create directory to store JSON file and move the JSON file into the new directory
```
cd $HOME/law_fieldlab/LMFlow/data
mkdir qa_finetune
cd qa_finetune
mkdir train

# Choose either the qa_finetuning.json or alpaca_finetuning.json file
mv ${HOME}law_fieldlab/old_files/finetuning/qa_finetuning.json ${HOME}law_fieldlab/LMFlow/data/qa_finetune/train
mv ${HOME}law_fieldlab/old_files/finetuning/alpaca_finetuning.json ${HOME}law_fieldlab/LMFlow/data/qa_finetune/train
```
Make sure to only store one JSON file in the directory.

### Adjust epoch
The default epoch from LMFlow is 0.01. Adjust this epoch to be at least 1 for both files
```
cd $HOME/law_fieldlab/LMFlow/scripts
nano run_finetune_with_lora.sh
```
```
nano run_finetune_with_lora_save_aggregated_weights.sh
```
### Run finetuning
Run the finetuning scripts:
```
cd $HOME/law_fieldlab/LMFlow
./scripts/run_finetune_with_lora.sh \
  --model_name_or_path ${HOME}law_fieldlab/save_model/Llama-2-7b-chat-hf/ \
  --dataset_path data/qa_finetune/train \
  --output_lora_path ${HOME}law_fieldlab/run_model_files/output_models/finetuned_model

./scripts/run_finetune_with_lora_save_aggregated_weights.sh \
  --model_name_or_path ${HOME}law_fieldlab/save_model/Llama-2-7b-chat-hf/ \
  --dataset_path data/qa_finetune/train \
  --output_model_path ${HOME}law_fieldlab/run_model_files/output_models/finetuned_model
```

Run finetuned model
```
cd $HOME/law_fieldlab/run_model_files
python examples/chatbot_gradio.py --deepspeed $HOME/law_fieldlab/run_model_files/configs/ds_config_chatbot.json --model_name_or_path output_models/finetuned_model --max_new_tokens 700
```
