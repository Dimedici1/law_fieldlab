from process_eurlex import process_dataset_in_chunks
from questions import generate_question
import pandas as pd
import random
import json
from datasets import load_dataset

dataset = load_dataset('eurlex')
eurlex_dataset = dataset["train"]


def generate_summary(text):
    words = text.split(" ")
    words = words[:200]
    text = ' '.join(words)
    return text


# Function to construct examples from the dataset
def construct_examples(df, contexts, questions, summaries, celex_list):
    celex_list = [item for sublist in celex_list for item in sublist]
    examples = []
    for i in range(len(df)):
        celex_id = df.iloc[i]['celex_id']
        input_text = (
            "<s>[INST] <<SYS>> You are a friendly chatbot that answers legal questions. Your main task is to assist the"
            " user as well as you can.\n"
            "Format: Based on the information in my database the answer is:\n\nAnswer to question\n\n"
            "Sources:\nlink1\nlink2\nlink3\nlink4\n\n\nLlama 2 can make mistakes. "
            "Consider checking important information."
            f"    Context: ### {' ### '.join(contexts[i])} ###\n"
            f"    <</SYS>>\n    {questions[i]}? [/INST]"
        )

        # Create additional links from celex_list
        additional_links = [f"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=celex:{celex_list[2 * i]}",
                            f"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=celex:{celex_list[2 * i + 1]}"]

        # Randomly choose 0, 1, or 2 additional links to add
        num_additional_links = random.choice([0, 1, 2])
        selected_additional_links = random.sample(additional_links, num_additional_links)

        # Combine all links
        source_links = "\n".join([f"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=celex:{celex_id}"] + selected_additional_links)
        output_text = f"Based on the information in my database the answer is:\n\n{summaries[i]}\n\n" \
                      f"Sources:\n{source_links}\n\n\nLlama 2 can make mistakes. " \
                      f"Consider checking important information."
        examples.append({"input": input_text, "output": output_text})

    return examples


# Main function to integrate all steps and generate examples
def generate_integrated_examples(chunk_size):
    data = eurlex_dataset
    df = pd.DataFrame(data)
    print(type(df))

    # Process dataset in chunks
    combined_contexts, final_celex_list = process_dataset_in_chunks(df, chunk_size)

    # Generate questions and summaries
    questions = [generate_question(title) for title in df['title']]
    summaries = [generate_summary(text) for text in df["text"]]

    # Construct and return examples
    return construct_examples(df, combined_contexts, questions, summaries, final_celex_list)


# Example usage
file_path = 'eurlex.csv'
chunk_size = 1000  # Adjust based on available memory and dataset size
examples = generate_integrated_examples(chunk_size)
# Wrap the examples in the specified format
output_data = {
    "type": "text2text",
    "instances": examples
}

# Define the filename for the output JSON file
# The file will be saved in the current working directory
output_file_name = 'qa_finetuning.json'

# Write the data to a JSON file
with open(output_file_name, 'w', encoding='utf-8') as file:
    json.dump(output_data, file, indent=4)

print(f"Data saved to {output_file_name}")
