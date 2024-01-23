import argparse
import os
import pandas as pd
import json

from config.log_config import logging
from categories import verify_categories
# from categories.category_mmlu import categories_mmlu, subcategories_mmlu

def process_csv_file(file_path, model_name):
    '''
    Process a single CSV file and return the calculated statistics.
    '''
    csv_data = pd.read_csv(file_path)
    correct_column = csv_data[f'{model_name}_correct']
    time_spent_column = csv_data[f'{model_name}_spend_time']
    return correct_column.mean(), time_spent_column.sum()

def find_main_category(subcategory, categories_mmlu):
    for main_category, subcats in categories_mmlu.items():
        if subcategory in subcats:
            return main_category
    return 'Unknown'

def main(args):
    logging.info("===== Start the evaluate the category. ======")

    categories_mmlu, subcategories_mmlu, _ , _ = verify_categories(args.category_type)
    model_name = args.model
    results_dir = args.save_dir
    results = {
        "subcategories": {},
        "categories": {},
        "weighted_accuracy": 0,
        "cost_time": 0
    }
    total_questions = 0
    total_accuracy = 0

    for file_name in os.listdir(results_dir):
        if file_name.endswith('.csv'):
            file_path = os.path.join(results_dir, file_name)
            subcategory_key = file_name.replace('.csv', '')
            if subcategory_key not in subcategories_mmlu:
                continue  # Skip files without a corresponding key in subcategories_mmlu
            accuracy, time_spent = process_csv_file(file_path, model_name)
            
            # Aggregating results for subcategories
            broad_category = subcategories_mmlu[subcategory_key][0]

            results["subcategories"].setdefault(broad_category, []).append(accuracy)

            # Mapping broad category to main category
            main_category = find_main_category(broad_category, categories_mmlu)

            results["categories"].setdefault(main_category, []).append(accuracy)

            total_questions += len(pd.read_csv(file_path))
            total_accuracy += accuracy * len(pd.read_csv(file_path))
            results["cost_time"] += time_spent

    # Calculating averages for subcategories and categories
    for subcategory, accuracies in results["subcategories"].items():
        results["subcategories"][subcategory] = sum(accuracies) / len(accuracies)
    for category, accuracies in results["categories"].items():
        results["categories"][category] = sum(accuracies) / len(accuracies)

    results["weighted_accuracy"] = total_accuracy / total_questions

    # Saving the results to a JSON file
    output_file = os.path.join(results_dir, 'processed_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    logging.info("===== Finish the evaluation, please check the result json file. ======")
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Evaluation Script")
    parser.add_argument('--category_type', "-c", type=str, choices=['mmlu', 'tmmluplus'], required=True,
                        help='Type of category (mmlu or tmmluplus)')
    parser.add_argument('--model', "-m", type=str, required=True,
                        help='Model name for evaluation')
    parser.add_argument('--save_dir', "-s", type=str, default='./results',
                        help='Directory to save the results')
    args = parser.parse_args()
    main(args)