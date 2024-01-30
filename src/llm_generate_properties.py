import torch
import transformers
from transformers import AutoTokenizer

import gc
import pandas as pd
import random


input_file = "data/ufet/clean_types.txt"
# df = pd.read_csv(input_file, sep="\t", names=["type"])

with open(input_file, "r") as f:
    types = f.readlines()

types = [type.strip("\n").strip() for type in types][0:100]
print(f"num_types: {len(types)}")
print(f"sample_types: {random.sample(types, 5)}")


# prompt = f"From now on, you are an competitive contestant in the general knowledge quiz contest and always answer all kinds of common sense questions accurately. You have a broad range of general and real word knowledge. This is the final round of the quiz contest and you have to answer the question to the best of your knowledge. Your answers must be a python list. The question is - What are the five most salient properties of <CONCEPT>?"

prompt = f"List the five most salient properties of <CONCEPT>. Generate only the numbered list of properties."

prompt_list = [prompt.replace("<CONCEPT>", type) for type in types]

print("prompt_list")
print(prompt_list[0:5])


model = "meta-llama/Llama-2-13b-chat-hf"
# model = "meta-llama/Llama-2-7b-chat-hf"
# model = "meta-llama/Llama-2-13b-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

response_list = []

file_name = (
    f'outputs/generated_property_{model.replace("/", "_").replace("-", "_")}.txt'
)

with open(file_name, "w") as out_file:
    out_file.write(f"model_name: {model}")

    for prop_prompt in prompt_list:
        print(f"Concept: {prop_prompt.split()[-1]}")
        sequences = pipeline(
            prop_prompt,
            do_sample=True,
            top_k=1,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=350,
        )

        for seq in sequences:
            # response_list.append(f"{seq['generated_text']}\n\n")
            print(f"{seq['generated_text']}")

            out_file.write(f'{seq["generated_text"]}\n')

            print("===================================")

del model
torch.cuda.empty_cache()
gc.collect()
