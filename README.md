# Aspect-Based Sentiment Analysis (ABSA) using BERT

This project fine-tunes a pre-trained BERT model to perform **Aspect-Based Sentiment Analysis (ABSA)** — identifying the sentiment expressed toward specific aspects (terms) within a sentence.

---

## What is Aspect-Based Sentiment Analysis (ABSA)

Aspect-Based Sentiment Analysis (ABSA) is a fine-grained form of sentiment analysis in Natural Language Processing (NLP). Instead of analyzing the sentiment of the entire sentence or document, ABSA identifies specific aspects or features mentioned in the text and determines the sentiment toward each aspect individually.

## Related Blog Post

I wrote a detailed Medium blog about this project:  
 [Read it here](https://medium.com/@ashok.1055/fine-tuning-bert-for-aspect-based-sentiment-analysis-absa-using-tensorflow-0875e1e5c839)


## Key Features

- Fine-tunes `bert-base-uncased` using the MAMS, Laptop, and Restaurant datasets.
- Handles sentence-aspect pair formatting: `[CLS] aspect [SEP] sentence [SEP]`.
- Trains a binary sentiment classifier (positive vs. negative).
- Includes reusable prediction and evaluation functions.

## Model Architecture

- **Encoder**: BERT (base, uncased)
- **Classification Head**: Single dense layer with 2 output logits (positive/negative)
- **Loss Function**: Sparse Categorical Crossentropy (with logits)
- **Optimizer**: AdamW with linear learning rate schedule and warmup

  ## Project Structure

```text
aspect-based-sentiment-analysis/
├── data/
│   ├── laptop_train.csv
│   ├── laptop_test.csv
│   ├── laptop_dev.csv
│   ├── rest16_train.csv
│   ├── rest16_test.csv
│   ├── rest16_dev.csv
│   ├── mams_train.csv
│   ├── mams_test.csv
│   └── mams_val.csv
│
├── notebook/
│   └── finetune_absa.ipynb          # Jupyter notebook for exploratory/interactive work
│
├── finetune_absa.py                 # Script for full fine-tuning, prediction, evaluation
├── requirements.txt                 # List of dependencies
└── README.md                        # Project overview and usage guide

```

##  How to Use

### 1. Install Dependencies

pip install pandas scikit-learn tensorflow transformers

### 2. Run Fine-Tuning

python absa_finetuning.py

This will:

Load and clean data

### 3. Inference Example

predict_sentiment("The food was amazing, but the service was slow.", "food")     # ➡ positive
predict_sentiment("The food was amazing, but the service was slow.", "service")  # ➡ negative

Encode examples with aspect + sentence pairs

Train and validate the BERT model

Save the fine-tuned model and tokenizer

