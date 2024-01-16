# -*- encoding: utf-8 -*-
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config.log_config import logging
from categories import verify_categories

choices = ["A", "B", "C", "D"]


def get_subject(data_dir):
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(data_dir, "test"))
            if "_test.csv" in f
        ]
    )
    return subjects

def format_subject(subject):
    """
    Formats the subject string by replacing underscores with spaces.

    Args:
    subject (str): The subject string to format.

    Returns:
    str: The formatted subject string.
    """
    return " ".join(subject.split("_"))

def create_result_folder(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir, "results_{}".format(args.model.split("/")[-1]))):
        os.makedirs(os.path.join(args.save_dir, "results_{}".format(args.model.split("/")[-1])))

def initial_model(args):
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
    return model, tokenizer

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
    all_times = []

    for i in range(test_df.shape[0]):
        start_time = time.time()
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
        all_times.append(time.time() - start_time)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    logging.info("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs, all_times


def main(args):
    """
    Main function to run the evaluation script.

    Args:
    args (Namespace): Command line arguments.
    """
    start_time = time.time()

    categories, subcategories, subcat_cors, cat_cors = verify_categories(args.category_type)

    model, tokenizer = initial_model(args)

    subjects = get_subject(args.data_dir)

    create_result_folder(args)

    # Initialize an empty list to store correlation scores for all subjects
    all_cors = []

    # Loop through each subject in the 'subjects' list
    for subject in subjects:
        # Log the start of processing for the current subject
        logging.info("Start the subject: {}".format(subject))

        # Read the development dataset for the current subject from CSV file
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]  # Limit the number of training samples if specified

        # Read the test dataset for the current subject from CSV file
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )

        # Evaluate the model on the current subject's data
        cors, acc, probs, all_times = eval(args, subject, model, tokenizer, dev_df, test_df)

        # Loop through subcategories related to the current subject
        subcats = subcategories[subject]
        for subcat in subcats:
            # Append the correlation scores to the respective subcategory list
            subcat_cors[subcat].append(cors)

            # Loop through main categories and assign scores to relevant categories
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)

        # Append the overall correlation scores for the subject to the all_cors list
        all_cors.append(cors)

        # Add a column to the test dataframe indicating if the model's predictions were correct
        test_df["{}_correct".format(args.model)] = cors

        # For each choice, add a column with the probabilities of that choice being correct
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["{}_choice{}_probs".format(args.model, choice)] = probs[:, j]
        
        test_df["{}_spend_time".format(args.model)] = all_times

        logging.info("Save the testing file into {}".format(os.path.join(
            args.save_dir, "results_{}".format(args.model.split("/")[-1]), "{}.csv".format(subject)
        )))

        # Save the modified test dataframe to a CSV file
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
        logging.info("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        results["categories"][cat] = cat_acc
        logging.info("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    weighted_acc = np.mean(np.concatenate(all_cors))
    results["weighted_accuracy"] = weighted_acc
    logging.info("Average accuracy: {:.3f}".format(weighted_acc))

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
# python3 evaluation_testing.py -m /workspace/models/llama2-7b-hf
