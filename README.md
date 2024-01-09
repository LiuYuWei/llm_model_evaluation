# llm_evaluation_tmmluplus

## Description
LLM Model Evaluation for tmmluplus datasets

## Dataset
tmmluplus dataset

- Introduction:
[Medium Article](https://medium.com/infuseai/tmmluplus-dataset-brief-introduction-ecfd00297838)

- huggingface dataset:
[Huggingface Dataset](https://huggingface.co/datasets/ikala/tmmluplus)

## How to use it?

- Step 1: please download the model from huggingface
The following command line is the example of mistral-7B-v0.1 model:
```bash
git lfs install
git clone https://huggingface.co/mistralai/Mistral-7B-v0.1
```

- Step 2: Please arrange the dataset from tmmluplus data folder to data_arrange folder.

- Step 3: Please run the following code:
```bash
python3 evaluate_hf.py \
    --model ./models/llama2-7b-hf \
    --data_dir ./llm_evaluation_tmmluplus/data_arrange/ \
    --save_dir ./llm_evaluation_tmmluplus/results/
```

## The example google colab code
Link: [Google Colab](https://colab.research.google.com/github/LiuYuWei/llm_evaluation_tmmluplus/blob/main/llm_evaluation_tmmluplus_example.ipynb)