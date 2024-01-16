# -*- encoding: utf-8 -*-
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from categories.category_tmmluplus import categories_tmmluplus, subcategories_tmmluplus
from categories.category_mmlu import categories_mmlu, subcategories_mmlu

choices = ["A", "B", "C", "D"]


def format_subject(subject):
    """
    Formats the subject string by replacing underscores with spaces.

    Args:
    subject (str): The subject string to format.

    Returns:
    str: The formatted subject string.
    """
    return " ".join(subject.split("_"))


def format_example(df, idx, include_answer=True):
    """
    Formats a question example from a dataframe.

    Args:
    df (pd.DataFrame): The dataframe containing the questions.
    idx (int): The index of the question in the dataframe.
    include_answer (bool): Whether to include the answer in the formatted string.

    Returns:
    str: The formatted question string.
    """
    prompt = df.iloc[idx, 0]
    num_options = df.shape[1] - 2
    for j in range(num_options):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, num_options + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    """
    Generates a prompt with multiple choice questions.

    Args:
    train_df (pd.DataFrame): The dataframe containing training data.
    subject (str): The subject of the questions.
    k (int): The number of questions to include in the prompt.

    Returns:
    str: The generated prompt.
    """
    if k == -1:
        k = train_df.shape[0]
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


@torch.no_grad()
def eval(args, subject, model, tokenizer, dev_df, test_df):
    """
    Evaluates the model on a given subject.

    Args:
    args (Namespace): Command line arguments.
    subject (str): The subject to evaluate on.
    model (AutoModelForCausalLM): The pre-trained model.
    tokenizer (AutoTokenizer): The tokenizer.
    dev_df (pd.DataFrame): The development set dataframe.
    test_df (pd.DataFrame): The test set dataframe.

    Returns:
    tuple: A tuple containing the correct answers array, accuracy, and probabilities array.
    """
    cors = []
    all_probs = []
    answers = choices[: test_df.shape[1] - 2]

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        while input_ids.shape[-1] > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
                model.device
            )

        label = test_df.iloc[i, test_df.shape[1] - 1]

        logits = model(input_ids=input_ids).logits[0, -1]

        probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits[tokenizer("A").input_ids[-1]],
                        logits[tokenizer("B").input_ids[-1]],
                        logits[tokenizer("C").input_ids[-1]],
                        logits[tokenizer("D").input_ids[-1]],
                    ]
                ).float(),
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
        )
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs


def main(args):
    """
    Main function to run the evaluation script.

    Args:
    args (Namespace): Command line arguments.
    """
    if args.category_type == "mmlu":
        categories = categories_tmmluplus
        subcategories = subcategories_tmmluplus
    elif args.category_type == "tmmluplus":
        categories = categories_mmlu
        subcategories = subcategories_mmlu
    else:
        print("Wrong category type. Please check the content.")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model.eval()
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir, "results_{}".format(args.model.split("/")[-1]))):
        os.makedirs(os.path.join(args.save_dir, "results_{}".format(args.model.split("/")[-1])))

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    start_time = time.time()
    for subject in subjects:
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )
        print("Start: {}".format(subject))
        cors, acc, probs = eval(args, subject, model, tokenizer, dev_df, test_df)
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["{}_correct".format(args.model)] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["{}_choice{}_probs".format(args.model, choice)] = probs[:, j]
        test_df.to_csv(
            os.path.join(
                args.save_dir, "results_{}".format(args.model.split("/")[-1]), "{}.csv".format(subject)
            ),
            index=None,
        )

    results = {"subcategories": {}, "categories": {}}
    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        results["subcategories"][subcat] = subcat_acc
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        results["categories"][cat] = cat_acc
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    weighted_acc = np.mean(np.concatenate(all_cors))
    results["weighted_accuracy"] = weighted_acc
    print("Average accuracy: {:.3f}".format(weighted_acc))

    results_file = os.path.join(
        args.save_dir, "accuracies_{}.json".format(args.model.replace("/", "_"))
    )
    end_time = time.time()
    results["cost_time"] = end_time - start_time
    with open(results_file, "w") as f:
        f.write(json.dumps(results, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category_type", "-c", type=str, default="tmmluplus")
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-m", type=str)
    args = parser.parse_args()
    main(args)
    
# Example command to run the script: 
# python3 evaluate_hf.py -m /workspace/models/llama2-7b-hf
