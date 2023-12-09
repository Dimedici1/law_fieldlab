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
from create_prompt import get_data
from pathlib import Path

testing_data_path = str(Path.home()) + "/law_fieldlab/Testing/"


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

def hist2context(history):
    input_text = ""
    for query, response in history:
        input_text += f"User: {query}\nChatbot: {response}\n"
    return input_text

def main():
    # import questions to iterate through
    questions_df = pd.read_csv(testing_data_path)
    questions = questions_df['Question'].tolist() 
    
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

    guide_message = (
        "\n"
        f"#############################################################################\n"
        f"##   A {model_name} chatbot is now chatting with you!\n"
        f"#############################################################################\n"
        "\n"
    )
    print(guide_message)

    # context = (
    #     "You are a helpful assistant who follows the given instructions"
    #     " unconditionally."
    # )
    context = ""

    end_string = chatbot_args.end_string
    prompt_structure = chatbot_args.prompt_structure

    history = []
    while True:
        input_text = input("User >>> ")
        if input_text == "exit":
            print("exit...")
            break
        elif input_text == "reset":
            history = []
            print("Chat history cleared")
            continue
        if not input_text:
            input_text = " "
    
        # Get context data for the current input_text
        context_data = get_data(input_text)
    
        # Build the full history text using the previous history
        history_text = hist2context(history)
    
        # Update the prompt with the new input, history, and context data
        prompt = chatbot_args.prompt_structure.format(context=context_data, history=history_text, query=input_text)
        prompt = prompt[-model.get_max_length():]  # Adjust for model's max length

        input_dataset = dataset.from_dict({
            "type": "text_only",
            "instances": [ { "text": prompt } ]
        })

        print("Bot: ", end="")
        print_index = 0

        token_per_step = 4

        for response, flag_break in inferencer.stream_inference(
            context=prompt,
            model=model,
            max_new_tokens=inferencer_args.max_new_tokens,
            token_per_step=token_per_step,
            temperature=inferencer_args.temperature,
            end_string=end_string,
            input_dataset=input_dataset
        ):
            # Prints characters in the buffer
            new_print_index = print_index
            for char in response[print_index:]:
                if end_string is not None and char == end_string[0]:
                    if new_print_index + len(end_string) >= len(response):
                        break

                new_print_index += 1
                print(char, end="", flush=True)

            print_index = new_print_index

            if flag_break:
                break
        print("\n", end="")

        # After generating the response, update the history
        history.append((input_text, response))


if __name__ == "__main__":
    main()
