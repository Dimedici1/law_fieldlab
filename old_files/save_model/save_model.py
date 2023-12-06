import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
warnings.filterwarnings('ignore')

# HF login
login('YOUR_HUGGINGFACE_TOKEN')

# Specify the model name
model_name = "meta-llama/Llama-2-7b-chat-hf"

# Set the path to save/load the model and tokenizer
save_path_model = './model_for_finetune'
save_path_tokenizer = './tokenizer_for_finetune'

# Download and cache the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save the model and tokenizer locally
model.save_pretrained(save_path_model)
tokenizer.save_pretrained(save_path_tokenizer)

# Load the model and tokenizer from the saved local directory
model = AutoModelForCausalLM.from_pretrained(save_path_model)
tokenizer = AutoTokenizer.from_pretrained(save_path_tokenizer)
