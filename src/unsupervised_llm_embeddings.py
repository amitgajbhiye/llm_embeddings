import torch
import pandas as pd
import os
import sys
import time
import logging
import numpy as np


project_path = os.path.split(os.path.abspath(os.path.realpath(__file__)))[0] + "/../"
sys.path.append(os.path.abspath(project_path))


from argparse import ArgumentParser
from utils import read_config

from sklearn.model_selection import train_test_split
from svm import train_svc, test_svc

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import gensim.downloader as api

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
        print(
            f"Getting embeddings of batch {i} of {len(input_list) // batch_size}",
            flush=True,
        )

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


def get_static_embeddings(input_list):
    embed_model = api.load(model_name)

    embeddings = dict()
    for con in input_list:
        try:
            embeddings[con] = embed_model[con]
        except KeyError:
            print("*" * 50)
            print(f"Concept {con} not found in embeding model: {model_name}")
            print("Splitting the word and then averaging...")

            embeddings[con] = np.mean(
                np.array([embed_model[word.strip()] for word in con.split()]), axis=0
            )

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
    # concept_embeddings = get_embeddings(
    #     input_list=concepts, prompt_id=config["prompt_id"]
    # )

    static_word_embedding_model_list = [
        "fasttext-wiki-news-subwords-300",
        "glove-wiki-gigaword-300",
        "word2vec-google-news-300",
    ]

    for model_name in static_word_embedding_model_list:
        print(f"model_name: {model_name}")

        concept_embeddings = get_static_embeddings(input_list=concepts)

        print(f"Concepts: {len(concepts)}, {concepts}")
        print(f"Properties: {len(properties)}, {properties}")
        res_all_prop = []

        for idx, prop in enumerate(properties):
            print()
            # print(f"For property: ************ {prop} ************ ")
            print(
                f"************ Training Classifier for Property - {idx+1} / {len(properties)} - {prop} ************"
            )

            property_train_data = train_df[train_df["property"] == prop]
            property_test_data = test_df[test_df["property"] == prop]

            print(f"property_train_data: {len(property_train_data)}")
            print(
                f"property_train_data_label_ratio: {property_train_data['label'].value_counts()}"
            )

            print(f"Spliting the Property Data into Train/Val ... ")
            train_split, val_split = train_test_split(
                property_train_data,
                test_size=0.10,
                stratify=property_train_data["label"],
            )
            print(f"train_split: {train_split.shape}")
            print(f"val_split: {val_split.shape}")

            train_con_embeddings = np.vstack(
                [concept_embeddings[con] for con in train_split["concept"]]
            )
            train_labels = train_split["label"].values

            val_con_embeddings = np.vstack(
                [concept_embeddings[con] for con in val_split["concept"]]
            )
            val_labels = val_split["label"].values

            print(f"train_con_embeddings.shape: {train_con_embeddings.shape}")
            print(f"train_labels.shape: {train_labels.shape}")

            print(f"val_con_embeddings.shape: {val_con_embeddings.shape}")
            print(f"val_labels.shape: {val_labels.shape}")

            pos_prop = property_train_data[property_train_data["label"] == 1]

            svm, th = train_svc(
                train_con_embeddings,
                val_con_embeddings,
                train_labels,
                val_labels,
                "linear",
                cv=min(3, len(pos_prop)),
            )

            print(f"svm: {svm}, th: {th}")

            print(f"Testing the Model ...")

            test_con_embeddings = np.vstack(
                [concept_embeddings[con] for con in test_df["concept"]]
            )
            test_labels = test_df["label"].values

            rr = test_svc(test_con_embeddings, test_labels, svm, th)
            print(
                str(idx + 1)
                + ", "
                + prop
                + ": map = "
                + str(rr[0])
                + ", f1 = "
                + str(rr[-1])
            )
            res_all_prop.append(rr)

        res_mean = np.mean(np.array(res_all_prop), axis=0)

        results_str = (
            str(args)
            + "\nLinear svm\n"
            + ": map = "
            + str(res_mean[0])
            + ", f1 = "
            + str(res_mean[-1])
            + "\n\n\n"
        )

        print("Final F1: ", res_mean[-1])
        print(f"Final result: {results_str}")
