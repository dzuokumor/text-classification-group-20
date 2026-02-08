# LSTM Experiments for Text Classification

## Overview
This notebook contains experiments evaluating an LSTM (Long Short-Term Memory) model
for multi-class text classification. The goal is to compare different text representations
(embeddings) and assess their impact on LSTM performance.

## Dataset
- The dataset consists of news articles, each with a **title** and **short description**.
- We combine these fields into a single `text` column for model input.
- Labels are zero-based (`0` to `3`) representing different news categories.
- Train-validation split: 80/20 with stratification to maintain class distribution.

## Embeddings Tested
1. **TF-IDF** – classic sparse representation capturing word importance.
2. **Word2Vec (Skip-gram)** – dense semantic embeddings trained on the training data.
3. **GloVe (pretrained)** – dense embeddings capturing semantic similarity.

## Preprocessing
- Text normalized to lowercase
- URLs removed
- Extra whitespace cleaned
- Punctuation preserved
- No stemming, lemmatization, or stopword removal (to preserve sequential patterns)

## Objective
- Evaluate LSTM performance across these embeddings.
- Compare metrics: Accuracy, Precision, Recall, F1-Score.
- Identify which embedding works best with LSTM for this news classification dataset.

## Notes
- TF-IDF + LSTM achieved highest accuracy (~90.6%) across all classes.
- Word2Vec and GloVe embeddings collapsed to predicting a single class (~25% accuracy),
  highlighting that TF-IDF better captures discriminative features for short, topic-focused text.
