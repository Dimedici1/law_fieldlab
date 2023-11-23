# Law Fieldlab Project Setup Instructions

## Clone the Repository
```
git clone https://github.com/Dimedici1/law_fieldlab.git
```
## Set Up Virtual Environment
Navigate to the project directory and set up the virtual environment:
```
cd law_fieldlab
conda create -n lawlab python=3.9 -y
conda activate lawlab
conda install mpi4py

```
Install the requirements
```
pip install -r requirements.txt
```
Install additional packages that can not be installed with requirements.txt

```
pip install icetk==0.0.7
pip install chromadb==0.4.18
pip install transformers==4.35.2
pip install tokenizers==0.13.3
pip install datasets==2.14.6
pip install sentence-transformers

```
## Create the Database
Update the link collection with relevant links from EUR-Lex that contain CELEX codes.
```
cd create_database
nano link_collection.py
```
Run the create_database.py file to store the vectorized data
```
python create_database.py
```
## Save Llama 2 Model
Navigate to save_model directory and replace the Hugging Face token in save_model.py:
```
cd ..
cd save_model
nano save_model.py
```

Run the save_model.py file
```
python save_model.py
```
Move the tokenizer files to the model files
```
mv $HOME/law_fieldlab/save_model/tokenizer_for_finetune/* $HOME/law_fieldlab/save_model/model_for_finetune/
```
## Finetune Model
Move back to the law_fieldlab directory
```
cd ..
```
Clone the LMFlow repository (https://github.com/OptimalScale/LMFlow) and follow the instructions in the README file. Set the input path of the model to $HOME/law_fieldlab/save_model/model_for_finetune and output path to $HOME/law_fieldlab/run_model_files/examples/output_models/finetuned_model.

Run the finetuning scripts (Here with example alpaca data from LMFlow):
```
./scripts/run_finetune_with_lora.sh \
  --model_name_or_path $HOME/law_fieldlab/save_model/model_for_finetune \
  --dataset_path $HOME/law_fieldlab/LMFlow/data/alpaca/train \
  --output_lora_path $HOME/law_fieldlab/run_model_files/examples/output_models/finetuned_model

./scripts/run_finetune_with_lora_save_aggregated_weights.sh \
  --model_name_or_path $HOME/law_fieldlab/save_model/model_for_finetune \
  --dataset_path $HOME/law_fieldlab/LMFlow/data/alpaca/train \
  --output_lora_path $HOME/law_fieldlab/run_model_files/examples/output_models/finetuned_model
```
## Run Model Files
Update create_prompt.py in create_prompt directory to set the correct database path:
```
cd $HOME/law_fieldlab/run_model_files/examples/
nano create_prompt.py
```
# Run the model files:
```
cd $HOME/law_fieldlab/run_model_files
python examples/chatbot_gradio.py --deepspeed configs/ds_config_chatbot.json --model_name_or_path $HOME/law_fieldlab/run_model_files/output_models/finetuned_model --max_new_tokens 200
```
