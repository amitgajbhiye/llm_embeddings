import torch
import pandas as pd
import os

from argparse import ArgumentParser
from utils import read_config

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print(f"Device: {device}")


# base_model_id_13b = "meta-llama/Llama-2-13b-chat-hf"
base_model_id_7b = "meta-llama/Llama-2-7b-chat-hf"
data_dir = "data/mcrae/original"
batch_size = 16


model = AutoModelForCausalLM.from_pretrained(
    base_model_id_7b, output_hidden_states=True, return_dict=True, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id_7b, padding_side="left", add_special_tokens=False
)
tokenizer.pad_token = tokenizer.eos_token


def get_embeddings(input_list):
    for i, idx in enumerate(range(0, len(input_list), batch_size)):
        inputs = tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=concept_prompts[idx : idx + batch_size],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=32,
        )

        inputs = {key: value.to(device) for key, value in inputs.items()}

        # print (idx, idx+batch_size)
        # print (concept_prompts[idx:idx+batch_size])
        # print (inputs)

        with torch.no_grad():
            outputs = model(**inputs)

        last_token_embedding = outputs.hidden_states[-1][:, -1, :]

        print(f"i, last_token_embedding: {i}, {last_token_embedding.shape}")
        print()


def get_data(config):
    data_dir = config["data_dir"]

    train_file = os.path.join(data_dir, "train_mcrae.tsv")

    test_file = os.path.join(data_dir, "test_mcrae.tsv")

    train_df = pd.read_csv(train_file, sep="\t", names=["concept", "property", "label"])
    test_df = pd.read_csv(test_file, sep="\t", names=["concept", "property", "label"])

    def clean_text(text):
        return " ".join(text.replace("_", " ").split())

    train_df["concept"] = train_df["concept"].apply(clean_text)
    train_df["property"] = train_df["property"].apply(clean_text)

    test_df["concept"] = test_df["concept"].apply(clean_text)
    test_df["property"] = test_df["property"].apply(clean_text)

    print(f"train_df: {train_df}")
    print(f"test_df: {test_df}")

    print(f"train_df.shape: {train_df.shape}")
    print(f"test_df.shape: {test_df.shape}")

    all_concepts = set(
        list(train_df["concept"].unique()) + list(test_df["concept"].unique())
    )
    all_properties = set(
        list(train_df["property"].unique()) + list(test_df["property"].unique())
    )

    print(f"all_concepts: {len(all_concepts)}")
    print(f"all_properties: {len(all_properties)}")

    concept_prompt = f'The concept "<CONCEPT>" means in one word: "'
    concept_prompts = [concept_prompt.replace("<CONCEPT>", con) for con in all_concepts]

    print(f"concept_prompts: {concept_prompts[0:10]}")

    property_prompt = f'The property "<PROPERTY>" means in one word: "'
    property_prompts = [
        property_prompt.replace("<PROPERTY>", prop) for prop in all_properties
    ]

    print(f"property_prompts: {property_prompts[0:10]}")

    return concept_prompts, property_prompts


if __name__ == "__main__":
    parser = ArgumentParser(description="Concept Property Embeddings from LLMs.")

    parser.add_argument(
        "-c", "--config_file", required=True, help="path to the configuration file"
    )

    args = parser.parse_args()
    config = read_config(args.config_file)

    concept_prompts, property_prompts = get_data(config=config)

    print(f"concept_prompts")
    get_embeddings(concept_prompts)

    print(f"concept_prompts")
    get_embeddings(property_prompts)
