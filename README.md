# Law Fieldlab Project Setup Instructions

## Clone the Repository
Clone the Github repository:
git clone https://github.com/Dimedici1/law_fieldlab.git

## Set Up Virtual Environment
Navigate to the project directory and set up the virtual environment:

cd law_fieldlab
conda create -n lawlab python=3.9 -y
conda activate lawlab
conda install mpi4py
pip install -r requirements.txt

## Create the Database
Navigate to create_database directory. Update the link collection with relevant links from EUR-Lex that contain CELEX codes. Ensure the persist path is correctly set:

cd database
# Update persist path in create_database.py to: /home/{your_username}/LawFieldlab/create_database/database
python create_database.py

## Save Model
Navigate to save_model directory. Replace the Hugging Face token in save_model.py:

cd ..
cd save_model
nano save_model.py
# Replace huggingface_token in save_model.py
python save_model.py

## Finetune Model
Clone the LMFlow repository and follow the instructions in the README file. Set the input path of the model to /home/{your_username}/LawFieldlab/save_model/model_for_finetune and output path to /home/{your_username}/LawFieldlab/run_model_files/examples/output_models/finetuned_model.

Run the finetuning scripts:

./scripts/run_finetune_with_lora.sh \
  --model_name_or_path /home/{your_username}/LawFieldlab/save_model/model_for_finetune \
  --dataset_path /home/{your_username}/LawFieldlab/LMFlow/data/alpaca/train \
  --output_lora_path /home/{your_username}/LawFieldlab/run_model_files/examples/output_models/finetuned_model

./scripts/run_finetune_with_lora_save_aggregated_weights.sh \
  --model_name_or_path /home/{your_username}/LawFieldlab/save_model/model_for_finetune \
  --dataset_path /home/{your_username}/LawFieldlab/LMFlow/data/alpaca/train \
  --output_lora_path /home/{your_username}/LawFieldlab/run_model_files/examples/output_models/finetuned_model

## Run Model Files
Update create_prompt.py in create_prompt directory to set the correct database path:

cd /home/{your_username}/LawFieldlab/run_model_files/examples/
nano create_prompt.py
# Replace the persist path with: /home/{your_username}/LawFieldlab/create_database/database

# Run the model files:

cd /home/{your_username}/LawFieldlab/run_model_files
python examples/chatbot_gradio.py --deepspeed configs/ds_config_chatbot.json --model_name_or_path /home/{your_username}/run_model_files/output_models/finetuned_model --max_new_tokens 200

