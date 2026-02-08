# AG News Text Classification: Comparative Analysis with Multiple Embeddings

A comprehensive comparative study of text classification performance across different word embeddings (TF-IDF, Word2Vec, GloVe) and model architectures (Logistic Regression, LSTM, GRU).

## Project Overview

This project implements and evaluates text classification systems on the AG News dataset using multiple embedding-model combinations. The goal is to understand how different word representations affect classification performance across various model architectures.

**Dataset**: AG News Classification Dataset (4 classes: World, Sports, Business, Sci/Tech)  
**Task**: Multi-class text classification  
**Embeddings**: TF-IDF, Word2Vec (Skip-gram & CBOW), GloVe  
**Models**: Logistic Regression, LSTM, GRU

## Repository Structure

```
ag_news_classification/
│
├── data/
│   ├── train_indices.npy          # Shared train split indices
│   └── val_indices.npy            # Shared validation split indices               
│
├── models/
│   ├── lr_tfidf.pkl               # Trained Logistic Regression with TF-IDF
│   ├── lr_skipgram.pkl            # Trained Logistic Regression with Word2Vec Skip-gram
│   ├── lr_cbow.pkl                # Trained Logistic Regression with Word2Vec CBOW
│   └── lr_glove.pkl               # Trained Logistic Regression with GloVe
│
├── results/
│   ├── class_distribution.png
│   ├── text_length_distribution.png
│   ├── word_frequency.png
│   ├── vocabulary_by_class.png
│   ├── lr_performance_comparison.png
│   ├── lr_confusion_matrices.png
│   ├── lr_per_class_performance.png
│   ├── logistic_regression_results.csv
│   └── lr_per_class_results.csv
│
├── hared_preprocessing.py         # Shared preprocessing pipeline
├── logistic_regression_model.py   # Logistic Regression implementation
├── 03_literature_review.md        # Comprehensive literature review
├── Logistic_Regression_Experiments.ipynb  # Main notebook for LR experiments
│
└── README.md                      # This file
```

## Dataset

**Source**: [AG News Classification Dataset on Kaggle](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)

**Statistics**:
- Training samples: 120,000
- Test samples: 7,600
- Classes: 4 (balanced distribution)
- Average text length: ~40 words

**Download Instructions**:
1. Download `train.csv` and `test.csv` from Kaggle
2. Place in `data/` directory or Google Drive (for Colab)

## Installation and Setup

### Prerequisites
```bash
Python 3.8+
Google Colab (recommended) or local Jupyter environment
```

### Required Libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn gensim
```

### For Google Colab
```python
from google.colab import drive
drive.mount('/content/drive')
```

## Shared Preprocessing Pipeline

All team members use identical preprocessing to ensure fair comparison:

### Pipeline Steps:
1. **Text Construction**: Combine title and description
2. **Text Cleaning**: 
   - Convert to lowercase
   - Remove URLs
   - Normalize whitespace
   - Preserve punctuation and word structure
3. **Label Encoding**: Convert to zero-based indices
4. **Train-Validation Split**: Stratified 80-20 split with fixed random seed

### Usage:
```python
from shared_preprocessing import SharedPreprocessor

preprocessor = SharedPreprocessor(random_state=42, val_size=0.2)
train_processed = preprocessor.preprocess_dataset(train_df)
X_train, X_val, y_train, y_val, _, _ = preprocessor.create_train_val_split(X, y)
```

## Model Implementations

### 1. Logistic Regression (Implemented)

**Embeddings**: TF-IDF, Word2Vec Skip-gram, Word2Vec CBOW, GloVe

**Key Features**:
- Hyperparameter tuning for each embedding
- Multiple evaluation metrics
- Confusion matrix analysis
- Per-class performance breakdown

**Run**:
```bash
Open Logistic_Regression_Experiments.ipynb in Google Colab
```

**Best Results**:
| Embedding | Test Accuracy | Test F1-Score |
|-----------|---------------|---------------|
| TF-IDF    | 0.898947     | 0.898735     |
| Word2Vec SG | 0.894474   | 0.894246     |
| Word2Vec CBOW | 0.869342  | 0.868962     |
| GloVe     | 0.883158     | 0.883003     |

### 2. LSTM (To be implemented by teammate)

Expected implementation using the shared preprocessing pipeline and same embeddings.

### 3. GRU (To be implemented by teammate)

Expected implementation using the shared preprocessing pipeline and same embeddings.

## Experimental Setup

### Hyperparameters Tuned

**Logistic Regression with TF-IDF**:
- Max features: [2000, 3000, 5000]
- C: [0.1, 1.0, 10.0]
- N-gram range: [(1,1), (1,2)]

**Logistic Regression with Word2Vec**:
- Vector size: [50, 100, 200]
- Window: [3, 5, 7]
- C: [0.1, 1.0, 10.0]

**Logistic Regression with GloVe**:
- Pre-trained embeddings (100-dimensional)
- C: [0.1, 1.0, 10.0, 100.0]

### Evaluation Metrics
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-Score (weighted)
- Per-class metrics
- Confusion matrices

## Results Summary

### Key Findings

1. **Best Embedding for Logistic Regression**: TF-IDF
2. **Performance Ranking**(by test accuracy): 
    1. TF-IDF: 0.8989
    2. Word2Vec Skip-gram: 0.8945
    4. GloVe: 0.8832
    3. Word2Vec CBOW: 0.8693

### Visualizations

The project includes comprehensive visualizations:
- Class distribution analysis
- Text length distribution
- Word frequency analysis
- Performance comparison plots
- Confusion matrices
- Per-class performance charts

## Team Contributions

### Member 1: Alliance Dushime
**Model**: Logistic Regression  
**Embeddings**: TF-IDF, Word2Vec (Skip-gram & CBOW), GloVe  
**Responsibilities**:
- Implemented shared preprocessing pipeline
- Developed Logistic Regression classifier with multiple embeddings
- Performed hyperparameter tuning
- Generated evaluation metrics and visualizations
- Contributed to literature review and report writing

### Member 2: Relebohile Pheko
**Model**: LSTM  
**Embeddings**: TF-IDF, Word2Vec, GloVe  
**Responsibilities**: 

### Member 3: David Zuokumor
**Model**: GRU  
**Embeddings**: TF-IDF, Word2Vec, GloVe  
**Responsibilities**: 

## Usage Instructions

### For Team Members

1. **Clone the repository**:
```bash
git clone https://github.com/dzuokumor/text-classification-group-20.git
cd ag_news_classification
```

2. **Use shared preprocessing**:
```python
from shared_preprocessing import SharedPreprocessor
preprocessor = SharedPreprocessor(random_state=42, val_size=0.2)
```

3. **Load shared train-val indices**:
```python
X_train, X_val, y_train, y_val = preprocessor.load_and_split_with_saved_indices(
    X, y, indices_dir='./data'
)
```

4. **Implement model** following the structure in `02_logistic_regression_model.py`

5. **Save results** in standardized format to `results/` directory

### For Reproducibility

All experiments use:
- **Random seed**: 42
- **Train-validation split**: 80-20
- **Stratified sampling**: Ensures balanced class distribution
- **Shared indices**: All team members use identical splits

## Literature References

The project is grounded in established research. Key references include:

### Text Classification
- Sebastiani, F. (2002). Machine learning in automated text categorization. ACM Computing Surveys.
- Aggarwal, C. C., & Zhai, C. (2012). Mining Text Data. Springer.

### Word Embeddings
- Mikolov, T., et al. (2013). Efficient estimation of word representations in vector space. arXiv.
- Pennington, J., et al. (2014). GloVe: Global vectors for word representation. EMNLP.

### Neural Architectures
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation.
- Cho, K., et al. (2014). Learning phrase representations using RNN encoder-decoder. arXiv.

**Full literature review**: See `03_literature_review.md`

## Acknowledgments

- Dataset: AG News Classification Dataset from Kaggle
- Pre-trained embeddings: GloVe from Stanford NLP
- Framework: scikit-learn, gensim

---

