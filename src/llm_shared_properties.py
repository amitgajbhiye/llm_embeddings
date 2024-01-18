import torch
import transformers
from transformers import AutoTokenizer


import pandas as pd

mcrae_train25_file = "data/mcrae/most_frequent_properties/train25.tsv"
train_df = pd.read_csv(
    mcrae_train25_file, sep="\t", names=["concept", "property", "label"]
)

positive_train_df = train_df[train_df["label"] == 1]

positive_train_df.sort_values(by=["property"], inplace=True)

print(positive_train_df)


# model = "meta-llama/Llama-2-13b-chat-hf"
model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)
