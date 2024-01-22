import argparse
import os
import pandas as pd
import json

# Load the categorization logic from a separate file
from categories.category_mmlu import categories_mmlu, subcategories_mmlu

def process_csv_file(file_path, category, model_name):
    """
    Process a single CSV file and return the calculated statistics.
    """
    csv_data = pd.read_csv(file_path)
    correct_column = csv_data[f'{model_name}_correct']
    time_spent_column = csv_data[f'{model_name}_spend_time']
    
    return {
        "accuracy": correct_column.mean(),
        "total_time_spent": time_spent_column.sum()
    }

def main(args):
    model_name = args.model.replace('/', '_').replace('.', '_')
    results = {
        "subcategories": {},
        "categories": {},
        "weighted_accuracy": 0,
        "cost_time": 0
    }
    total_questions = 0

    for file_name in os.listdir(args.save_dir):
        if file_name.endswith('.csv'):
            file_path = os.path.join(args.save_dir, file_name)
            category = subcategories_mmlu.get(file_name.split('.')[0], 'other')
            file_results = process_csv_file(file_path, category, model_name)

            results["subcategories"][category] = file_results["accuracy"]
            results["cost_time"] += file_results["total_time_spent"]
            total_questions += len(pd.read_csv(file_path))

    # Calculate weighted accuracy
    results["weighted_accuracy"] = sum(
        [results["subcategories"][cat] * len(pd.read_csv(os.path.join(args.save_dir, f"{cat}.csv")))
         for cat in results["subcategories"]]) / total_questions

    # Save the results to a JSON file
    with open(os.path.join(args.save_dir, f"{model_name}_results.json"), 'w') as f:
        json.dump(results, f, indent=4)

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
