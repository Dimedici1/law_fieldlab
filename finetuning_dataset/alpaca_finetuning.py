import pandas as pd
import json
from datasets import load_dataset
import random

chatbot_responses = [
    "Certainly! Let's look into that for you.",
    "Absolutely! I'll check on that right away.",
    "Of course! Let me find that information for you.",
    "Sure thing! I'll see what I can find out.",
    "Definitely! Let me dig into that.",
    "Alright! I'm on it, let me search for you.",
    "Okay! Let me explore that topic for you.",
    "Sure! I'll investigate that right now.",
    "Indeed! I'll get the details for you.",
    "Yes! Let me fetch that information.",
    "Absolutely! I'm looking that up right now.",
    "Got it! I'll find out for you.",
    "Understood! Let me see what I can discover.",
    "Alrighty! Time to find some answers for you.",
    "Sure! Let me dive into that query.",
    "Of course! Searching for that information now.",
    "Yes, indeed! I'll seek out the details.",
    "Right away! Let me pull up that information.",
    "Certainly! I'll look that up immediately.",
    "Definitely! Let me get the specifics on that.",
    "Alright! Let me see what I can uncover.",
    "Sure thing! I'll see what's available.",
    "Okay! I'm on the case to find out.",
    "Sure! Let me see what information is out there.",
    "Absolutely! I'll check that for you now.",
    "Of course! Let me see what I can find out.",
    "Got it! I'll start researching immediately.",
    "Indeed! I'll gather that information for you.",
    "Yes! Let me look into that right now.",
    "Alright! I'm searching for the answer.",
    "Sure! Let me check that information.",
    "Certainly! I'll find the answer for you.",
    "Of course! Let me get right on that.",
    "Sure thing! I'm looking into it now.",
    "Definitely! Let me see what I can dig up.",
    "Alright! Let me find the best answer.",
    "Okay! I'll see what I can learn about that.",
    "Sure! I'll find out for you.",
    "Absolutely! Let me investigate that for you.",
    "Yes! I'm checking on that now.",
    "Understood! I'll see what I can find.",
    "Alrighty! I'm on the hunt for answers.",
    "Sure! Let me get that information for you.",
    "Of course! I'm looking up that topic now.",
    "Yes, indeed! I'll get the details for you.",
    "Right away! I'm finding out for you.",
    "Certainly! I'm on it, looking for that information.",
    "Definitely! I'll provide the information shortly.",
    "Alright! Let me do a quick search.",
    "Okay! I'm fetching the details now.",
    "Sure thing! I'm gathering the facts for you.",
    "Absolutely! Let me clarify that for you.",
    "Of course! I'm on the task to find the answer.",
    "Yes! I'll see what information is available.",
    "Alright! Let me look that up for you.",
    "Certainly! I'm checking our resources now.",
    "Definitely! I'll find the most accurate information.",
    "Okay! Let's see what we can find out.",
    "Sure! I'm searching for the needed details.",
    "Understood! I'll get to the bottom of this.",
    "Indeed! Let me research that for you.",
    "Yes, indeed! I'm working on getting that information.",
    "Right away! I'll seek out the answers.",
    "Certainly! I'm finding the relevant data.",
    "Of course! Let me provide the specifics.",
    "Sure thing! I'm looking up the necessary info.",
    "Definitely! I'll scour our sources for you.",
    "Alright! I'll find the best possible answer.",
    "Okay! Let's discover the answer together.",
    "Sure! I'm on the lookout for that info.",
    "Absolutely! I'm finding out right now.",
    "Of course! I'm verifying the information.",
    "Yes! I'll pull up the latest data.",
    "Alright! Let me consult my sources.",
    "Certainly! I'm working on it right now.",
    "Definitely! Let's get you that information.",
    "Okay! I'm checking the details.",
    "Sure! Let me pinpoint the exact information.",
    "Absolutely! I'm looking for the most up-to-date info.",
    "Of course! I'm zeroing in on the details.",
    "Yes! I'm on the quest to find your answer.",
    "Alright! Let's find out what we need to know.",
    "Sure thing! I'm investigating your query.",
    "Definitely! I'm seeking the right information.",
    "Okay! I'm combing through the data.",
    "Sure! I'm checking for the most accurate answer.",
    "Absolutely! Let's get the specifics on that.",
    "Of course! I'm hunting down the information.",
    "Yes! I'm working on retrieving that for you.",
    "Alright! I'm delving into the topic.",
    "Certainly! I'm on the search for answers.",
    "Definitely! Let me grab that information for you.",
    "Okay! I'm analyzing the available data.",
    "Sure! I'm exploring all the possibilities.",
    "Absolutely! I'm ensuring to get the right info.",
    "Of course! I'm sorting through the information.",
    "Yes! Let me get the most relevant details.",
    "Alright! I'm finding the precise information.",
    "Sure thing! I'm working to get your answer.",
    "Definitely! Let me hunt down that information.",
    "Sure! Let's find the answer together.",
    "Absolutely! I'm on it.",
    "Okay! Let me look that up for you.",
    "Certainly! I'm searching for the best information.",
    "Yes! Let me get the details.",
    "Of course! I'm on the hunt for your answer.",
    "Alright! I'm checking for the most accurate answer.",
    "Indeed! Let me delve into that.",
    "Got it! I'm finding that out right now.",
    "Definitely! Let's see what I can find.",
    "Right away! I'm getting the information.",
    "Understood! I'm on the case.",
    "Sure thing! I'm gathering the facts.",
    "Okay! Let me confirm that for you.",
    "Absolutely! I'm looking into it.",
    "Certainly! Let's uncover that information.",
    "Yes, indeed! I'm working on it.",
    "Of course! Let me investigate further.",
    "Alright! Let me fetch that for you.",
    "Indeed! I'm checking our resources.",
    "Got it! I'm on the quest for answers.",
    "Definitely! Let me explore that for you.",
    "Right away! I'm seeking the information.",
    "Understood! Let me dig a little deeper.",
    "Sure thing! I'm verifying that now.",
    "Okay! I'm looking for the best response.",
    "Absolutely! Let's get to the truth.",
    "Certainly! I'm finding the most relevant information.",
    "Yes, indeed! Let me sort through the data.",
    "Of course! I'm piecing it together.",
    "Alright! I'm finding the most reliable answer.",
    "Indeed! Let me check the latest information.",
    "Got it! I'm analyzing the details.",
    "Definitely! Let me synthesize the info.",
    "Right away! I'm consulting my sources.",
    "Understood! I'm evaluating the options.",
    "Sure thing! I'm cross-referencing now.",
    "Okay! Let me get the most accurate info.",
    "Absolutely! I'm putting the pieces together.",
    "Certainly! I'm pinpointing the answer.",
    "Yes, indeed! Let me get the facts straight.",
    "Of course! I'm clarifying that for you.",
    "Alright! I'm connecting the dots.",
    "Indeed! I'm pulling up the necessary information.",
    "Got it! I'm sifting through the data.",
    "Definitely! I'm working on a clear response.",
    "Right away! I'm collating the information.",
    "Understood! I'm formulating the best answer.",
    "Sure thing! Let me refine the search.",
    "Okay! I'm tailoring the answer to your question.",
    "Absolutely! I'm zeroing in on the details.",
    "Certainly! I'm honing in on the answer.",
    "Yes, indeed! Let me draw the correct conclusion.",
    "Of course! I'm finalizing the best response.",
    "Alright! I'm extracting the key information.",
    "Indeed! I'm optimizing the search for you.",
    "Got it! I'm compiling the necessary data.",
    "Definitely! I'm narrowing down the answers.",
    "Right away! I'm focusing on getting it right.",
    "Understood! I'm making sure to be thorough.",
    "Sure thing! I'm ensuring accuracy.",
    "Okay! Let me double-check that.",
    "Absolutely! I'm confirming the details.",
    "Certainly! I'm assessing the information.",
    "Yes, indeed! Let me get the precise answer.",
    "Of course! I'm reviewing the facts.",
    "Alright! I'm getting the most relevant response.",
    "Indeed! I'm concentrating on your query.",
    "Got it! I'm doing a deep dive for you.",
    "Definitely! I'm unearthing the necessary info.",
    "Right away! I'm filtering the best answers.",
    "Understood! I'm calibrating my response.",
    "Sure thing! I'm fine-tuning the information.",
    "Okay! I'm cross-checking the facts.",
    "Absolutely! I'm ensuring the most accurate results.",
    "Certainly! I'm aligning the details.",
    "Yes, indeed! Let me bring you the best answer.",
    "Of course! I'm scrutinizing the information.",
    "Alright! I'm getting the most updated info.",
    "Indeed! I'm homing in on the answer.",
    "Got it! I'm dissecting the question.",
    "Definitely! I'm pursuing the right information.",
    "Right away! I'm fetching the latest data.",
    "Understood! I'm going the extra mile for accuracy.",
    "Sure thing! I'm validating the information.",
    "Okay! Let me provide you with the best answer.",
    "Absolutely! I'm examining every angle.",
    "Certainly! I'm ensuring thoroughness.",
    "Yes, indeed! Let me get the most precise information.",
    "Of course! I'm doing a comprehensive search.",
    "Alright! I'm gathering the essential details.",
    "Indeed! I'm selecting the most relevant information.",
    "Got it! I'm looking into every aspect.",
    "Definitely! I'm capturing the key points.",
    "Right away! I'm preparing a detailed response.",
    "Understood! I'm aiming for the most complete answer.",
    "Sure thing! I'm securing the necessary details.",
    "Okay! Let me bring you the most accurate findings.",
    "Absolutely! I'm assembling the best response.",
    "Certainly! I'm obtaining the key facts.",
    "Yes, indeed! Let me produce the most reliable answer.",
    "Of course! I'm constructing a thorough reply.",
    "Alright! I'm crafting the answer with care.",
    "Indeed! I'm piecing together the best information."
]

def generate_random_greeting(greeting_list):
    return random.choice(greeting_list)


def generate_random_url(url_li):
    return random.choice(url_li)


dataset = load_dataset('yahma/alpaca-cleaned')
alpaca_dataset = dataset["train"]

df = pd.DataFrame(alpaca_dataset)
df = df[df["input"] != ""]
print(df)

urls = load_dataset("bs-modeling-metadata/c4_newslike_url_only")
url_data = urls["train"]
url_list = url_data['url'][:50000]
print(len(url_list))

def create_prompts(df):
    examples = []
    for _, row in df.iterrows():  # Note the addition of _, to unpack the tuple
        random_url = generate_random_url(url_list)
        input_text = (f"""<s>[INST] <<SYS>> Context: {row['input']} (Source: {random_url})\n<</SYS>> {row['instruction']} [/INST]""")
        output_text = f"""{generate_random_greeting(chatbot_responses)} \n\n{row['output']}\n\nSource:\n\n{random_url}\n\nLlama 2 can make mistakes. Consider checking important information.\n\n\n"""
        examples.append({"input": input_text, "output": output_text})
    return examples


examples = create_prompts(df)
print(len(examples))
# Wrap the examples in the specified format
output_data = {
    "type": "text2text",
    "instances": examples
}

# Define the filename for the output JSON file
# The file will be saved in the current working directory
output_file_name = 'alpaca_finetuning.json'

# Write the data to a JSON file
with open(output_file_name, 'w', encoding='utf-8') as file:
    json.dump(output_data, file, indent=4)

print(f"Data saved to {output_file_name}")
