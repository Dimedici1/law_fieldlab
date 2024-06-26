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
import torch
import warnings
import gradio as gr
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

MAX_BOXES = 20

logging.disable(logging.ERROR)
warnings.filterwarnings("ignore")

title = """
<h1 align="center">Law Fieldlab</h1>
<link rel="stylesheet" href="/path/to/styles/default.min.css">
<script src="/path/to/highlight.min.js"></script>
<script>hljs.highlightAll();</script>

<p align="center">This Master Thesis shows how fine-tuning and Retrieval Augmented Generation can improve Llama 2 performance.</p>

"""
css = """
#user {                                                                         
    float: right;
    position:relative;
    right:5px;
    width:auto;
    min-height:32px;
    max-width: 60%
    line-height: 32px;
    padding: 2px 8px;
    font-size: 14px;
    background:	#9DC284;
    border-radius:5px; 
    margin:10px 0px;
}
                                             
#chatbot {                                                                      
    float: left;
    position:relative;
    right:5px;
    width:auto;
    min-height:32px;
    max-width: 60%
    line-height: 32px;
    padding: 2px 8px;
    font-size: 14px;
    background:#7BA7D7;
    border-radius:5px; 
    margin:10px 0px;
}
"""


@dataclass
class ChatbotArguments:
    prompt_structure: Optional[str] = field(
        default=
        f"<s>[INST] <<SYS>> Context: {{data}}\nHistory: {{input_text}}\n<</SYS>> {{query}} Never write more than 4 sentences. Use the context to answer the question."
        f"Name your sources in this format: \n\nSource 1 \n\n Name all relevant links. [/INST]",
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
@dataclass
class ChatbotArguments:
    prompt_structure: Optional[str] = field(
        default=
        f"<s>[INST] <<SYS>> Context: {{data}} <</SYS>> {{query}} Never write more than 2 sentences. [/INST]",
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
    torch_dtype=torch.float16
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


# context = (
#     "You are a helpful assistant who follows the given instructions"
#     " unconditionally."
# )


end_string = chatbot_args.end_string
prompt_structure = chatbot_args.prompt_structure


token_per_step = 4

def hist2context(hist):
    input_text = ""
    for query, response in hist:
        input_text += f"User: {query}\nChatbot: {response}\n"
    return input_text


def chat_stream(query: str, history=None, **kwargs):
    if history is None:
        history = []

    # Get context data for the current query
    context_data = get_data(query)

    # Build the full context using the previous history
    history_text = hist2context(history)

    # Format the prompt with the history, context data, and current query
    full_prompt = prompt_structure.format(input_text=history_text, data=context_data, query=query)

    print(full_prompt)

    # Process the query and get the response
    context_ = full_prompt[-model.get_max_length():] 
    input_dataset = dataset.from_dict({
        "type": "text_only",
        "instances": [ { "text": context_ } ]
    })

    for response, flag_break in inferencer.stream_inference(context=context_, model=model, max_new_tokens=inferencer_args.max_new_tokens, 
                                    token_per_step=token_per_step, temperature=inferencer_args.temperature,
                                    end_string=end_string, input_dataset=input_dataset):
        seq = response
        
        yield response, history + [(query, response)]
        if flag_break:
            break




def predict(input, history=None): 
    if history is None:
        history = []
    for response, history in chat_stream(input, history):
        updates = []
        for query, response in history:
            updates.append(gr.update(visible=True, value="" + query))
            updates.append(gr.update(visible=True, value="" + response))
        if len(updates) < MAX_BOXES:
            updates = updates + [gr.Textbox.update(visible=False)] * (MAX_BOXES - len(updates))
        yield [history] + updates





with gr.Blocks(css=css) as demo:
    gr.HTML(title)
    state = gr.State([])
    text_boxes = []
    for i in range(MAX_BOXES):
        if i % 2 == 0:
            text_boxes.append(gr.Markdown(visible=False, label="Q:", elem_id="user"))
        else:
            text_boxes.append(gr.Markdown(visible=False, label="A:", elem_id="chatbot"))

    txt = gr.Textbox(
        show_label=False,
        placeholder="Enter text and press send.",
    )
    button = gr.Button("Send")

    button.click(predict, [txt, state], [state] + text_boxes)
    demo.queue().launch(share=True)



