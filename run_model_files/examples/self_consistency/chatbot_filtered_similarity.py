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
saving_output_path = str(Path.home()) + "/law_fieldlab/testing/self_consistency/filtered_similarity_responses.csv"

logging.disable(logging.ERROR)
warnings.filterwarnings("ignore")


@dataclass
class ChatbotArguments:
    prompt_structure: Optional[str] = field(
        default=
        f"<s>[INST] <<SYS>> Context: {{data}}\nHistory: {{input_text}}\n<</SYS>> {{query}} Never write more than 4 sentences. Use the context to answer the question. [/INST]",
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

def pick_history(df, idx):
    # Check the format in the "Question format" column at the specified index
    question_format = df.at[idx, "Question format"]

    if question_format == "Multiple Choice":
        return """
[INST] What factor is essential for the effectiveness of personalized pricing in e-commerce?
(A) The geographic location of the user
(B) The digital marketing strategy
(C) Accuracy of customer data analysis
(D) The variety of products offered [/INST] To determine the factor essential for the effectiveness of personalized pricing in e-commerce, consider what personalized pricing relies on. It's the ability to tailor prices to individual customers. This requires understanding each customer's preferences and buying habits. The most crucial aspect of this is accurate customer data analysis, as it provides insights into customer behavior. Therefore, the correct answer is (C) Accuracy of customer data analysis.

[INST] Which industry is increasingly adopting dynamic pricing models based on user data and behavior?
(A) Telecommunications
(B) Online retail
(C) Automotive sales
(D) Pharmaceutical [/INST] Dynamic pricing models are used in industries where customer data and behavior can significantly influence pricing strategies. Online retail is a field where customer browsing and purchasing data are readily available and can be used effectively for dynamic pricing. So, the answer is (B) Online retail.

[INST] What is a potential downside of using personalized pricing based on customer's online activity?
(A) Increased customer loyalty
(B) Reduced market competition due to price manipulation
(C) Improved product quality
(D) Enhanced user experience [/INST] Personalized pricing based on customer's online activity can lead to reduced market competition. This happens because personalized pricing can manipulate prices in a way that discourages customers from shopping around, which can dampen competition. Hence, the answer is (B) Reduced market competition due to price manipulation.
"""

    elif question_format in ["Open End", "General"]:
        return """
[INST] In the context of e-commerce, how does price personalization impact consumer trust and brand loyalty? [/INST] Price personalization can be a double-edged sword. On one hand, it offers deals that seem tailored to the consumer, enhancing the shopping experience and loyalty. On the other hand, it can raise privacy and fairness concerns, which may erode trust. Thus, the impact of price personalization on consumer trust and brand loyalty is complex.

[INST] What ethical considerations should businesses take into account when employing personalized pricing models? [/INST] Businesses must consider several ethical factors, including transparency, data privacy, and fairness, when using personalized pricing models. Ethical practices, such as clear communication about how personal data is used and ensuring prices are fair across different customer segments, are essential.

[INST] How can regulations ensure fairness in the application of personalized pricing techniques? [/INST] Regulations can play a critical role in ensuring fairness in personalized pricing by setting clear guidelines on data usage, prohibiting discriminatory practices, and mandating transparency about how prices are determined for individual consumers.
"""
    elif question_format == "True/False":
        return """
[INST] Does the European Union's GDPR potentially restrict the use of consumers' social media activity for setting personalized prices? [/INST] True. GDPR emphasizes user consent, data transparency, and limitation on the purpose of data use. It doesn't outright ban the use of social media activity for personalized pricing but places restrictions ensuring that any use of such data complies with these principles.

[INST] Do personalized pricing strategies inherently increase transparency in e-commerce transactions? [/INST] False. Personalized pricing strategies, if not communicated clearly, can actually reduce transparency. Customers might not be aware of how their personal data influences the prices they're offered.

[INST] Do traditional brick-and-mortar stores commonly employ personalized pricing based on customer demographics? [/INST] False. Traditional brick-and-mortar stores generally do not employ personalized pricing based on customer demographics to the extent seen in online retail. They lack the sophisticated data analytics tools that online platforms use.
"""

    else:
        return "Question format not recognized."

def main():
    # import questions to iterate through
    questions_df = pd.read_csv(testing_data_path)
    questions = questions_df['Question'].tolist()

    # DataFrame to store responses
    responses_df = pd.DataFrame(questions, columns=['Question'])
    
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

    for in in range(1, 4):
        for idx, input_text in enumerate(questions):
            print(f"Question {idx}")
            context_data = get_data(input_text, 4, questions_df, idx, "similarity")
            history_text = pick_history(questions_df, idx)
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
                temperature=0.5,
                end_string=end_string,
                input_dataset=input_dataset
            ):
                response = resp
                if flag_break:
                    break
    
            # Update the DataFrame
            responses_df.at[idx, f'Answer_{i}'] = response
    
    # Save the responses to a CSV file
    responses_df.to_csv(saving_output_path, index=False)

if __name__ == "__main__":
    main()
