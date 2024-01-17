# llm_model_evaluation

## Description
Use python script to do LLM Model Evaluation.

## Support Dataset

### I. mmlu dataset

- Introduction from paper with code:
[Paper-with-code](https://paperswithcode.com/dataset/mmlu)

### II. tmmluplus dataset

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
- mmlu dataset:
1. [Google Colab - mmlu](https://colab.research.google.com/github/LiuYuWei/llm_model_evaluation/blob/main/llm_evaluation_mmlu.ipynb)
2. [Google Colab - mmlu in phi-2 model](https://colab.research.google.com/github/LiuYuWei/llm_model_evaluation/blob/main/llm_evaluation_mmlu_phi_2.ipynb) [Colab free tier can use this Google Colab example]

- tmmluplus dataset: 
1. [Google Colab - tmmluplus](https://colab.research.google.com/github/LiuYuWei/llm_model_evaluation/blob/main/llm_evaluation_tmmluplus.ipynb)

## Evaluation Result

- mmlu dataset:

| 模型 | Weighted Accuracy | STEM | humanities | social sciences | other | Inference Time(s) |
|  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |
| Mistral-7B-v0.1 | 0.6254094858282296 | 0.5251822398939695 | 0.5636556854410202 | 0.7357816054598635 | 0.703578038247995 | 15624.038010835648 |

- tmmluplus dataset:

| 模型 | Weighted Accuracy | STEM | humanities | social sciences | other | Inference Time(s) |
|  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |
| Mistral-7B-v0.1 | - | - | - | - | - | - |