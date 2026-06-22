# Sentiment Analysis on Social Media using Language Models

Sentiment analysis involves analyzing digital text to determine whether the emotional tone is positive, negative, or neutral. This analysis provides valuable insights for companies to enhance customer service and increase brand reputation.

The project aims to explore, train, and compare multiple machine learning models for sentiment analysis using textual data sourced from social media. Specifically, it investigates the effectiveness of different pretrained language models, comparing encoder-only and decoder-only models (known as LLM) variants, and evaluates their performance, data requirements and efficiency based on parameter count.

Beyond model training, the project includes comprehensive exploratory data analysis, which provides actionable intelligence for marketing strategies using insights such as popular hashtags or emerging trends. A general analysis and detailed hashtag analysis is performed, integrating various supervised and unsupervised techniques, including graph analysis and semantic clustering.

## Data

This project uses [TweetEval](https://huggingface.co/datasets/cardiffnlp/tweet_eval) dataset, which is a collection of English datasets for seven multi-class classification tasks, all utilizing Twitter data. The tasks include irony detection, hate speech detection, offensive content identification, stance detection, emoji prediction, emotion recognition and sentiment analysis. 

Specifically, we focus on the sentiment analysis task, which aims to classify the sentiment of a tweet as negative, neutral or positive.

To download the data and load it into the project, the following code can be used:

```python
from datasets import load_dataset

dataset = load_dataset("tweet_eval", "sentiment")
```

Additionally, the dataset can be manually downloaded from the HuggingFace datasets hub.

### Preprocessing

We apply different preprocessing steps to the data to prepare it for training and evaluation depending on the type of model.

For machine learning models, we apply the following preprocessing steps:

1. Cleaning and normalization (removing non-alphabetic characters, unicode normalization, lowercasing, etc.).
2. Tokenization (word tokenization).
3. Stopwords removal.
4. Stemming (Porter stemming).
5. Vectorization (TF-IDF, word embeddings).

For encoder-only language models, we apply the following preprocessing steps:

1. Tokenization.
2. Padding.
3. Truncation.
4. Token to ID conversion.
5. Special tokens addition.
6. Attention mask creation.

For decoder-only language models (LLM), we apply the following preprocessing steps:

1. Prompt creation.
2. Stratified sampling to create subsets of the data.
3. Mapping of the data to the prompt (including system prompts).
4. Special tokens addition.
5. Tokenization.


## Models

We experiment with multiple types of models to perform sentiment analysis on the TweetEval dataset. The models are divided into two main categories: machine learning models and transformer models.

### Machine Learning Models

The explored machine learning models are:

- Logistic Regression
- Random Forest

Additionally, we experiment with VADER.

### Transformer Models

We experiment with two types of transformer models, encoder-only models and decoder-only models.

#### Encoder-only Models

The selected encoder-only models are:

- [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)
- [distilbert/distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased)
- [FacebookAI/roberta-base](https://huggingface.co/FacebookAI/roberta-base)

Additionally, we run the evaluation process for the following reference encoder-only models:

- [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
- [finiteautomata/bertweet-base-sentiment-analysis](https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis)

#### Decoder-only Models (LLM)

The selected decoder-only models are:

- [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)

Additionally, we run the evaluation process using few-shot learning for the following reference decoder-only models:

- [google/gemma-1.1-2b-it](https://huggingface.co/google/gemma-1.1-2b-it)
- [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)

## Usage

### Project setup

1. Clone this repository to your local machine. To clone it from the command line run `git clone https://github.com/acampillos/social-media-nlp`.

2. (Optional but recommended) Create a virtual environment. To create a virtual environment, run `python3.10.9 -m venv venv` in the root directory of the project. To activate the virtual environment, run `source venv/bin/activate` on Unix systems or `venv\Scripts\activate` on Windows.

3. Install dependencies. Install by running `poetry install`. Poetry can be installed by running `pip install poetry`.

4. Install [flash-attention](https://github.com/Dao-AILab/flash-attention). To install, run `pip install flash-attn --no-build-isolation`.

<br>

### Running the experiments

The available experiments are located in the `experiments` directory and divided by type of model. These are also listed in the following table:

<br>

| Model Type                          | Task                      | Command                                                 |
| ----------------------------------- | ------------------------- | ------------------------------------------------------- |
| Language Model (decoder-only)       | Training                  | `python .src/social_media_nlp/experiments/llm/train.py model_id -r r -a a -e e -t -v v` |
|                                     | Model and LoRA merging    | `python .src/social_media_nlp/experiments/llm/merge.py model_path subfolder` |
|                                     | Evaluation                | `python .src/social_media_nlp/experiments/llm/evaluate.py model_path` |
| Language Model (encoder-only)       | Training                  | `python .src/social_media_nlp/experiments/lm/train.py model_id` |
|                                     | Evaluation                | `python .src/social_media_nlp/experiments/lm/evaluation.py model_path` |
| Machine Learning                    | Training and evaluation   | `python .src/social_media_nlp/experiments/ml/tune.py` |

<br>

The parameters used are as follows:

- `r`: Rank for fine-tuning using LoRA.
- `a`: Alpha for fine-tuning using LoRA.
- `e`: Number of epochs for training.
- `t`: Number of entries to select for the training set.
- `v`: Number of entries to select for the validation set.
- `model_path`: Path to the selected model for the experiment.
- `subfolder`: Path to a specific version of the model after fine-tuning.
- `model_id`: Hugging Face identifier of the selected model or its local path

<br>

### Experiments visualization

To visualize the hyperparameter tuning logs from Optuna, follow these steps:
1. Start Optuna dashboard by running `optuna-dashboard sqlite:///mlflow.db` in the root directory of the project.
2. Access dashboard on port 8080.
3. View studies and their details (e.g., best parameters, best value, hyperparameter relationships, etc.).

To visualize the logs from MLFlow used in the training and hyperparameter tuning of the models, follow these steps:
1. Start MLFlow UI by running `mlflow ui` in the root directory of the project.
2. Access dashboard on port 5000
5. Explore dashboard (e.g., experiments, runs, metrics, etc.)

<br>

## Project structure

    ├── mlruns                                  <- MLflow logs for experiments.
    ├── models                                  <- Trained models predictions.
    ├── notebooks                               <- Jupyter notebooks for EDA and experiments.
    │   └── eda.ipynb
    ├── src
    │   └── social_media_nlp                    <- Source code for the project.
    │       ├── data                            <- Data processing scripts.
    │       │   ├── cleaning.py
    │       │   └── preprocessing.py
    │       ├── experiments                     <- Experiments scripts.
    │       │   ├── llm                         <- Decoder-only transformer experiments.
    │       │   │   ├── evaluate_few_shot.py
    │       │   │   ├── evaluate.py
    │       │   │   ├── merge_models.py
    │       │   │   └── train.py
    │       │   ├── ml                          <- Machine learning experiments.
    │       │   │   └── tune.py
    │       │   └── seq_lm                      <- Encoder-only transformer experiments.
    │       │       ├── evaluation.py
    │       │       └── train.py
    │       ├── models                          <- Model training, evaluation and inference scripts.
    │       │   ├── evaluation.py               <- Metrics evaluation.
    │       │   ├── hyperparameter_tuning.py    <- Optuna hyperparameter tuning.
    │       │   ├── ml                          <- Machine learning models' scripts.
    │       │   │   └── vader.py
    │       │   ├── transformers                <- Transformers models' scripts.
    │       │   │   ├── inference.py
    │       │   │   ├── train.py
    │       │   │   └── utils.py
    │       │   └── utils.py
    │       ├── prompts.py                      <- Prompts for the LLMs.
    │       └── visualization                   <- Scripts for exploratory analysis and results plots.
    │           ├── features.py
    │           ├── graph.py
    │           ├── hashtag_clustering.py
    │           └── similarity.py
    ├── mlflow.db                               <- Optuna database for hyperparameter tuning.
    ├── poetry.lock
    ├── pyproject.toml                          <- Project dependencies.
    └── README.md




* Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., de las Casas, D., Bressand, F., Lengyel, G., Lample, G., Saulnier, L., et al. (2023). Mistral 7B. arXiv. https://arxiv.org/abs/2310.06825

* Gemma Team, Mesnard, T., Hardin, C., Dadashi, R., Bhupatiraju, S., Pathak, S., Sifre, L., Rivière, M., Kale, M. S., Love, J., et al. (2024). Gemma: Open models based on Gemini research and technology. arXiv. https://arxiv.org/abs/2403.08295
