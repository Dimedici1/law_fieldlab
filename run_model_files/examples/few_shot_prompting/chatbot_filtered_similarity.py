#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
"""A simple shell chatbot implemented with lmflow APIs.
"""
import logging
import json
import os
import sys
sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))
import warnings
import pandas as pd

from dataclasses import dataclass, field
from transformers import HfArgumentParser
from typing import Optional

from lmflow.datasets.dataset import Dataset
from lmflow.pipeline.auto_pipeline import AutoPipeline
from lmflow.models.auto_model import AutoModel
from lmflow.args import ModelArguments, DatasetArguments, AutoArguments
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
from create_prompt_filtered import get_data
from pathlib import Path

testing_data_path = str(Path.home()) + "/law_fieldlab/testing/Master_Thesis_Questions.csv"
saving_output_path = str(Path.home()) + "/law_fieldlab/testing/few_shot_prompting/filtered_similarity_responses.csv"

logging.disable(logging.ERROR)
warnings.filterwarnings("ignore")


@dataclass
class ChatbotArguments:
    prompt_structure: Optional[str] = field(
        default=
        f"<s>[INST] <<SYS>> Context: {{data}}\nHistory: {{input_text}}\n<</SYS>> {{query}} Never write more than 4 sentences. Use the context to answer the question."
        f"Name your sources in this format: \n\nlink1\n\n Name all relevant links. [/INST]",
        metadata={
            "help": "prompt structure given user's input text"
        },
)
    end_string: Optional[str] = field(
        default="\n\n\n",
        metadata={
            "help": "end string mark of the chatbot's output"
        },
    )
"""
def hist2context(history):
    input_text = ""
    for query, response in history:
        input_text += f"User: {query}\nChatbot: {response}\n"
    return input_text
"""

def choose_history(df, idx):
    # Check the format in the "Question format" column at the specified index
    question_format = df.at[idx, "Question format"]

    if question_format == "Multiple Choice":
        return """
User: What factor is essential for the effectiveness of personalized pricing in e-commerce?
(A) The geographic location of the user
(B) The digital marketing strategy
(C) Accuracy of customer data analysis
(D) The variety of products offered
Chatbot: (C) Accuracy of customer data analysis

User: Which industry is increasingly adopting dynamic pricing models based on user data and behavior?
(A) Telecommunications
(B) Online retail
(C) Automotive sales
(D) Pharmaceutical
Chatbot: (B) Online retail

User: What is a potential downside of using personalized pricing based on customer's online activity?
(A) Increased customer loyalty
(B) Reduced market competition due to price manipulation
(C) Improved product quality
(D) Enhanced user experience
Chatbot: (B) Reduced market competition due to price manipulation
"""
    elif question_format in ["Open End", "General"]:
        return """
User: In the context of e-commerce, how does price personalization impact consumer trust and brand loyalty?
Chatbot: Price personalization in e-commerce can have a complex impact on consumer trust and brand loyalty. While it can offer tailored deals to consumers, enhancing their shopping experience and loyalty, it can also raise concerns about privacy and fairness, potentially eroding trust.

User: What ethical considerations should businesses take into account when employing personalized pricing models?
Chatbot: Businesses must consider several ethical factors, including transparency, data privacy, and fairness, when using personalized pricing models. Ethical practices, such as clear communication about how personal data is used and ensuring prices are fair across different customer segments, are essential.

User: How can regulations ensure fairness in the application of personalized pricing techniques?
Chatbot: Regulations can play a critical role in ensuring fairness in personalized pricing by setting clear guidelines on data usage, prohibiting discriminatory practices, and mandating transparency about how prices are determined for individual consumers.
"""
    elif question_format == "True/False":
        return """
User: Does the European Union's GDPR potentially restrict the use of consumers' social media activity for setting personalized prices?
Chatbot: True.

User: Do personalized pricing strategies inherently increase transparency in e-commerce transactions?
Chatbot: False.

User: Do traditional brick-and-mortar stores commonly employ personalized pricing based on customer demographics?
Chatbot: False.
"""
    else:
        return "Invalid question format or format not recognized."

def main():
    # import questions to iterate through
    questions_df = pd.read_csv(testing_data_path)
    questions = questions_df['Question'].tolist()

    # DataFrame to store responses
    responses_df = pd.DataFrame(columns=['Question', 'Answer'])
    
    pipeline_name = "inferencer"
    PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)

    parser = HfArgumentParser((
        ModelArguments,
        PipelineArguments,
        ChatbotArguments,
    ))
    model_args, pipeline_args, chatbot_args = (
        parser.parse_args_into_dataclasses()
    )
    inferencer_args = pipeline_args

    with open (pipeline_args.deepspeed, "r") as f:
        ds_config = json.load(f)

    model = AutoModel.get_model(
        model_args,
        tune_strategy='none',
        ds_config=ds_config,
        device=pipeline_args.device,
    )

    # We don't need input data, we will read interactively from stdin
    data_args = DatasetArguments(dataset_path=None)
    dataset = Dataset(data_args)

    inferencer = AutoPipeline.get_pipeline(
        pipeline_name=pipeline_name,
        model_args=model_args,
        data_args=data_args,
        pipeline_args=pipeline_args,
    )

    # Chats
    model_name = model_args.model_name_or_path
    if model_args.lora_model_path is not None:
        model_name += f" + {model_args.lora_model_path}"

#    context = ""
    end_string = chatbot_args.end_string
    prompt_structure = chatbot_args.prompt_structure

    for idx, input_text in enumerate(questions):
        print(f"Question {idx}")
        context_data = get_data(input_text, 4, questions_df, idx, "similarity")
        history_text = choose_history(questions_df, idx)
        prompt = chatbot_args.prompt_structure.format(data=context_data, input_text=history_text, query=input_text)
        original_length = len(prompt)
        prompt = prompt[-model.get_max_length():]
        new_lenght = len(prompt)
        # Check if the prompt was shortened and print a message if it was
        if new_lenght < original_length:
            print(f"\n\n\nTHE PROMPT {idx} HAS BEEN SHORTENED\n\n\n")
        print(prompt)
        
        input_dataset = dataset.from_dict({
            "type": "text_only",
            "instances": [ { "text": prompt } ]
        })
        token_per_step = 4 
        response = ""
        for resp, flag_break in inferencer.stream_inference(
            context=prompt,
            model=model,
            max_new_tokens=300,
            token_per_step=token_per_step,
            temperature=inferencer_args.temperature,
            end_string=end_string,
            input_dataset=input_dataset
        ):
            response = resp
            if flag_break:
                break

        # Create a new DataFrame for the row to add
        new_row_df = pd.DataFrame({'Question': [input_text], 'Answer': [response]})
    
        # Concatenate the new DataFrame with the existing one
        responses_df = pd.concat([responses_df, new_row_df], ignore_index=True)

    # Save the responses to a CSV file
    responses_df.to_csv(saving_output_path, index=False)

if __name__ == "__main__":
    main()
