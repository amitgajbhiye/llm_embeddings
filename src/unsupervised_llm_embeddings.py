import torch
import pandas as pd
import os
import time
import logging

from argparse import ArgumentParser
from utils import read_config

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print(f"Device: {device}")


PROMPTS = {1: 'The concept "<CONCEPT>" means in one word: "'}


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


def get_embeddings(input_list, prompt_id):
    embeddings = dict()

    for i, idx in enumerate(range(0, len(input_list), batch_size)):
        print(f"Processing batch {i} of {len(input_list) // batch_size}", flush=True)

        batch = input_list[idx : idx + batch_size]
        concept_prompts = [
            PROMPTS[prompt_id].replace("<CONCEPT>", con) for con in batch
        ]

        inputs = tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=concept_prompts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=32,
        )

        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        batch_last_token_embedding = outputs.hidden_states[-1][:, -1, :]

        for con, embed in zip(batch, batch_last_token_embedding):
            embeddings[con] = embed.detach().cpu().numpy()

    print(f"{len(embeddings)}: len(embeddings)")

    return embeddings


def get_data(config):
    def clean_text(text):
        return " ".join(text.replace("_", " ").split())

    all_concepts, all_properties = set(), set()
    if config.get("train_file") is not None:
        train_file = config["train_file"]
        train_df = pd.read_csv(
            train_file, sep="\t", names=["concept", "property", "label"]
        )
        train_df["concept"] = train_df["concept"].apply(clean_text)
        train_df["property"] = train_df["property"].apply(clean_text)
        logging.info(f"train_df: {train_df}")

        all_concepts.update(train_df["concept"].unique())
        all_properties.update(train_df["property"].unique())

    if config.get("val_file") is not None:
        val_file = config["val_file"]
        val_df = pd.read_csv(val_file, sep="\t", names=["concept", "property", "label"])
        val_df["concept"] = val_df["concept"].apply(clean_text)
        val_df["property"] = val_df["property"].apply(clean_text)
        logging.info(f"val_df: {val_df}")

        all_concepts.update(val_df["concept"].unique())
        all_properties.update(val_df["property"].unique())

    if config.get("test_file") is not None:
        test_file = config["test_file"]
        test_df = pd.read_csv(
            test_file, sep="\t", names=["concept", "property", "label"]
        )
        test_df["concept"] = test_df["concept"].apply(clean_text)
        test_df["property"] = test_df["property"].apply(clean_text)
        logging.info(f"test_df: {test_df}")

        all_concepts.update(test_df["concept"].unique())
        all_properties.update(test_df["property"].unique())

    print(f"all_concepts: {all_concepts}")

    return list(all_concepts), list(all_properties), train_df, test_df


if __name__ == "__main__":
    parser = ArgumentParser(description="Concept Property Embeddings from LLMs.")

    parser.add_argument(
        "-c", "--config_file", required=True, help="path to the configuration file"
    )

    args = parser.parse_args()
    config = read_config(args.config_file)

    log_file_name = os.path.join(
        "logs",
        config.get("log_dirctory"),
        f"log_{config.get('experiment_name')}_{time.strftime('%d-%m-%Y_%H-%M-%S')}.txt",
    )
    logging.basicConfig(
        level=logging.DEBUG,
        filename=log_file_name,
        filemode="w",
        format="%(asctime)s : %(name)s : %(levelname)s - %(message)s",
    )

    concepts, properties, train_df, test_df = get_data(config=config)

    print(f"Concepts: {len(concepts)}, {concepts}")
    print(f"Properties: {len(properties)}, {properties}")

    for idx, prop in enumerate(properties):
        print(f"Processing: Prop - {idx+1} / {len(properties)} - {prop}")

        property_train_data = train_df[train_df["property"] == prop]

        print()
        print(f"property_train_data: {len(property_train_data)}")
        print(
            f"property_train_data_label_ratio: {property_train_data['label'].value_counts()}"
        )

    # embeddings = get_embeddings(input_list=concepts, prompt_id=config["prompt_id"])

    # print(f"len(embeddings): {len(embeddings)}")
