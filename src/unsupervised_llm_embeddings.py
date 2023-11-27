import torch
import pandas as pd
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# base_model_id_13b = "meta-llama/Llama-2-13b-chat-hf"

base_model_id_7b = "meta-llama/Llama-2-7b-chat-hf"
data_dir = "data/mcrae/original"


model = AutoModelForCausalLM.from_pretrained(base_model_id_7b, output_hidden_states=True, return_dict=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(base_model_id_7b, 
                                          padding_side="left",
                                          add_special_tokens = False)
tokenizer.pad_token = tokenizer.eos_token


train_file = os.path.join(data_dir, "train_mcrae.tsv")
test_file = os.path.join(data_dir, "test_mcrae.tsv")

train_df = pd.read_csv(train_file, sep="\t", names=["concept", "property", "label"])
test_df = pd.read_csv(test_file, sep="\t", names=["concept", "property", "label"])


def clean_text (text):
    return " ".join(text.replace("_", " ").split())   


train_df["concept"] = train_df["concept"].apply(clean_text)
train_df["property"] = train_df["property"].apply(clean_text)

test_df["concept"] = test_df["concept"].apply(clean_text)
test_df["property"] = test_df["property"].apply(clean_text)

print (f"train_df: {train_df}")
print (f"test_df: {test_df}")

print (f"train_df.shape: {train_df.shape}")
print (f"test_df.shape: {test_df.shape}")

all_concepts = set(list(train_df["concept"].unique()) + list(test_df["concept"].unique()))
all_properties = set(list(train_df["property"].unique()) + list(test_df["property"].unique()))

print (f"all_concepts: {len(all_concepts)}")
print (f"all_properties: {len(all_properties)}")




