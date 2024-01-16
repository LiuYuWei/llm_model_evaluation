from config.log_config import logging
from categories.category_tmmluplus import categories_tmmluplus, subcategories_tmmluplus
from categories.category_mmlu import categories_mmlu, subcategories_mmlu


def verify_categories(dataset_type):
    logging.info("The type of dataset: {}".format(dataset_type))
    if dataset_type == "mmlu":
        categories = categories_mmlu
        subcategories = subcategories_mmlu
    elif dataset_type == "tmmluplus":
        categories = categories_tmmluplus
        subcategories = subcategories_tmmluplus
    else:
        categories = {}
        subcategories = {}
        logging.info("Wrong category type. Please check the content.")
    
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    return categories, subcategories, subcat_cors, cat_cors