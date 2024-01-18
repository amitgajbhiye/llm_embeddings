import torch
import transformers
from transformers import AutoTokenizer

import gc


import pandas as pd

import pandas as pd

mcrae_train25_file = "data/mcrae/most_frequent_properties/train25.tsv"
train_df = pd.read_csv(
    mcrae_train25_file, sep="\t", names=["concept", "property", "label"]
)

positive_train_df = train_df[train_df["label"] == 1]

positive_train_df.sort_values(by=["property"], inplace=True)
positive_train_df.reset_index(inplace=True, drop=True)

print("positive_train_df")
print(positive_train_df)

uniq_props = positive_train_df["property"].unique()

print("uniq_props")
print(uniq_props)

num_concepts = 5

all_concepts_list = []

for prop in uniq_props:
    concepts = positive_train_df[positive_train_df["property"] == prop][
        "concept"
    ].to_list()[:num_concepts]
    concepts_list = ", ".join(concepts)

    all_concepts_list.append((concepts_list, prop))

    # print(concepts_list)

# prompt = f"Enumerate the five most salient properties shared by the following concepts - <CONCEPT_LIST>. Generate only the numbered list of properties."

prompt = f"What are the common properties of <CONCEPT_LIST>?"

prompt_list = [
    (prompt.replace("<CONCEPT_LIST>", concept_list), original_property)
    for concept_list, original_property in all_concepts_list
]

print("prompt_list")
print(prompt_list)


# model = "meta-llama/Llama-2-13b-chat-hf"
# model = "meta-llama/Llama-2-7b-chat-hf"
model = "meta-llama/Llama-2-13b-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

response_list = []

file_name = "non_chat_llama2_13b_shared_properties_mcrae_concepts.txt"

with open(file_name, "w") as out_file:
    out_file.write(f"model_name: {model}")

    for prop_prompt, original_property in prompt_list:
        # print (concept_prompt)

        sequences = pipeline(
            prop_prompt,
            do_sample=True,
            top_k=1,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=350,
        )

        for seq in sequences:
            response_list.append(f"{seq['generated_text']}\n\n")
            print(f"Original Property: {original_property}\n")
            print(f"{seq['generated_text']}")

            out_file.write(f"\n")

            out_file.write(f"Original Property: {original_property}\n")
            out_file.write(f'{seq["generated_text"]}\n')

            print("===================================")

del model
torch.cuda.empty_cache()
gc.collect()
