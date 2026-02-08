# Literature Review for AG News Text Classification Project

## Overview
This literature review provides the theoretical foundation for comparing text classification performance across different embedding techniques and model architectures. It covers key concepts in text representation, embedding methods, and classification models relevant to your project.

---

## 1. Text Classification Fundamentals

Text classification is a supervised learning task where text documents are automatically assigned to predefined categories based on their content. The AG News dataset classification task falls under topic classification, where news articles are categorized into domains such as World, Sports, Business, and Science/Technology.

**Key References:**

- Aggarwal, C. C., & Zhai, C. (2012). *Mining Text Data*. Springer Science & Business Media. [Provides comprehensive coverage of text classification techniques and challenges]

- Sebastiani, F. (2002). Machine learning in automated text categorization. *ACM Computing Surveys*, 34(1), 1-47. [Classic survey establishing foundations of text classification approaches]

---

## 2. Word Embeddings: From Sparse to Dense Representations

### 2.1 TF-IDF (Term Frequency-Inverse Document Frequency)

TF-IDF is a statistical measure that evaluates word importance in a document relative to a corpus. It produces sparse, high-dimensional representations where each dimension corresponds to a unique term.

**Formula:**
```
TF-IDF(t,d) = TF(t,d) × IDF(t)
where IDF(t) = log(N / df(t))
```

**Key Characteristics:**
- Captures term importance through frequency weighting
- Creates sparse, interpretable feature vectors
- No semantic understanding (treats words independently)
- Works well with linear classifiers like Logistic Regression

**References:**

- Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management*, 24(5), 513-523. [Foundational work on TF-IDF weighting schemes]

- Ramos, J. (2003). Using TF-IDF to determine word relevance in document queries. *Proceedings of the First Instructional Conference on Machine Learning*, 242, 29-48. [Practical application and analysis of TF-IDF]

### 2.2 Word2Vec (Skip-gram and CBOW)

Word2Vec, introduced by Mikolov et al. (2013), learns dense vector representations that capture semantic relationships between words. It consists of two architectures:

**Skip-gram**: Predicts context words given a target word
**CBOW (Continuous Bag of Words)**: Predicts target word given context words

**Key Characteristics:**
- Creates dense, low-dimensional vectors (typically 100-300 dimensions)
- Captures semantic similarity (e.g., king - man + woman ≈ queen)
- Skip-gram: Better for rare words, larger training corpus needed
- CBOW: Faster training, better for frequent words

**References:**

- Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space. *arXiv preprint arXiv:1301.3781*. [Original Word2Vec paper introducing Skip-gram and CBOW]

- Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. *Advances in Neural Information Processing Systems*, 26. [Extensions and improvements to Word2Vec]

- Goldberg, Y., & Levy, O. (2014). word2vec Explained: deriving Mikolov et al.'s negative-sampling word-embedding method. *arXiv preprint arXiv:1402.3722*. [Detailed mathematical explanation of Word2Vec training]

### 2.3 GloVe (Global Vectors for Word Representation)

GloVe combines global matrix factorization with local context window methods to learn word vectors. Unlike Word2Vec's predictive approach, GloVe uses count-based statistics.

**Key Characteristics:**
- Leverages global word co-occurrence statistics
- Pre-trained on large corpora (Wikipedia, Common Crawl)
- Often performs better on word analogy tasks
- Faster training on large corpora

**References:**

- Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global vectors for word representation. *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 1532-1543. [Original GloVe paper]

- Levy, O., Goldberg, Y., & Dagan, I. (2015). Improving distributional similarity with lessons learned from word embeddings. *Transactions of the Association for Computational Linguistics*, 3, 211-225. [Comparative analysis of embedding methods including GloVe]

---

## 3. Classification Models

### 3.1 Logistic Regression for Text Classification

Logistic Regression is a linear classification model that estimates the probability of class membership using the logistic function. Despite its simplicity, it remains highly effective for text classification, especially with high-dimensional sparse features like TF-IDF.

**Why Logistic Regression works well for text:**
- Text data often linearly separable in high dimensions
- Handles sparse features efficiently
- Provides probabilistic predictions
- Interpretable coefficients

**References:**

- Genkin, A., Lewis, D. D., & Madigan, D. (2007). Large-scale Bayesian logistic regression for text categorization. *Technometrics*, 49(3), 291-304. [Demonstrates effectiveness of logistic regression on large text datasets]

- Zhang, T. (2004). Solving large scale linear prediction problems using stochastic gradient descent algorithms. *Proceedings of the 21st International Conference on Machine Learning*, 116. [Optimization techniques for logistic regression on text]

### 3.2 Recurrent Neural Networks (RNN, LSTM, GRU)

Recurrent architectures process sequential data by maintaining hidden states that capture temporal dependencies.

**RNN (Vanilla Recurrent Neural Network):**
- Processes sequences with recurrent connections
- Suffers from vanishing/exploding gradient problems
- Limited ability to capture long-range dependencies

**LSTM (Long Short-Term Memory):**
- Addresses vanishing gradient through gating mechanisms
- Maintains both short-term and long-term memory
- More parameters and computational cost than vanilla RNN

**GRU (Gated Recurrent Unit):**
- Simplified LSTM architecture with fewer parameters
- Combines forget and input gates into update gate
- Often comparable performance to LSTM with faster training

**Key Characteristics for Text:**
- Capture sequential nature of language
- Work well with dense word embeddings
- Can model long-range dependencies
- Require more training data than traditional methods

**References:**

- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780. [Original LSTM paper]

- Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. *arXiv preprint arXiv:1406.1078*. [Introduction of GRU architecture]

- Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). Empirical evaluation of gated recurrent neural networks on sequence modeling. *arXiv preprint arXiv:1412.3555*. [Comparative analysis of LSTM vs GRU]

- Yin, W., Kann, K., Yu, M., & Schütze, H. (2017). Comparative study of CNN and RNN for natural language processing. *arXiv preprint arXiv:1702.01923*. [Comprehensive comparison of sequence models for NLP]

---

## 4. Embedding-Model Compatibility

Different embeddings have varying compatibility with classification models based on their representation characteristics.

### 4.1 Sparse Embeddings (TF-IDF) with Linear Models

**Why TF-IDF works well with Logistic Regression:**
- High-dimensional sparse features create linearly separable decision boundaries
- Logistic Regression efficiently handles sparse inputs
- No information loss from averaging word vectors

**References:**

- Wang, S., & Manning, C. D. (2012). Baselines and bigrams: Simple, good sentiment and topic classification. *Proceedings of the 50th Annual Meeting of the Association for Computational Linguistics*, 2, 90-94. [Shows strong performance of simple features with linear models]

### 4.2 Dense Embeddings (Word2Vec, GloVe) with Neural Networks

**Why dense embeddings work better with sequence models:**
- Semantic representations capture word relationships
- Sequential models leverage temporal patterns
- Dense vectors reduce dimensionality while preserving information
- Pre-trained embeddings transfer knowledge from large corpora

**Averaging dense embeddings for traditional models:**
- Common approach: average word vectors to get document representation
- Loss of sequential information
- Still captures semantic content but ignores word order

**References:**

- Kim, Y. (2014). Convolutional neural networks for sentence classification. *arXiv preprint arXiv:1408.5882*. [Demonstrates effectiveness of pre-trained word embeddings with neural networks]

- Iyyer, M., Manjunatha, V., Boyd-Graber, J., & Daumé III, H. (2015). Deep unordered composition rivals syntactic methods for text classification. *Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics*, 1, 1681-1691. [Analysis of averaging embeddings for classification]

---

## 5. AG News Dataset Context

The AG News Corpus is a widely-used benchmark for text classification, containing news articles from over 2000 sources categorized into four topics.

**Dataset Characteristics:**
- 4 classes: World, Sports, Business, Sci/Tech
- 120,000 training samples, 7,600 test samples
- Relatively balanced class distribution
- Short documents (title + description)

**References:**

- Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level convolutional networks for text classification. *Advances in Neural Information Processing Systems*, 28. [Uses AG News as benchmark; introduces character-level approaches]

- Joulin, A., Grave, E., Bojanowski, P., & Mikolov, T. (2016). Bag of tricks for efficient text classification. *arXiv preprint arXiv:1607.01759*. [FastText experiments on AG News showing strong baselines]

---

## 6. Expected Performance Patterns

Based on the literature, we expect the following patterns in your experiments:

### For Logistic Regression:
1. **TF-IDF likely to perform best** - Linear models excel with sparse, high-dimensional features
2. **Word2Vec/GloVe averaged vectors may underperform** - Averaging loses sequential information
3. **Skip-gram vs CBOW**: Minor differences, as both are averaged for document representation

### For LSTM/GRU:
1. **Dense embeddings (Word2Vec, GloVe) likely to outperform TF-IDF** - Sequential models leverage temporal patterns better
2. **Pre-trained GloVe may perform best** - Transfers knowledge from large external corpora
3. **LSTM vs GRU**: Comparable performance, GRU slightly faster

### Cross-Model Patterns:
1. **TF-IDF + Logistic Regression** likely competitive with or superior to **Dense embeddings + RNNs** for this short-text dataset
2. **Complexity-performance tradeoff**: Simpler models may match neural networks on this structured task

**References:**

- Joulin, A., Grave, E., Bojanowski, P., & Mikolov, T. (2016). Bag of tricks for efficient text classification. *arXiv preprint arXiv:1607.01759*. [Shows simple methods can match complex models]

- Howard, J., & Ruder, S. (2018). Universal language model fine-tuning for text classification. *arXiv preprint arXiv:1801.06146*. [Analyzes when complex models outperform simple baselines]

---

## 7. Evaluation Metrics

Standard metrics for multi-class text classification:

- **Accuracy**: Overall correct predictions (suitable for balanced datasets)
- **Precision**: Proportion of true positives among predicted positives
- **Recall**: Proportion of true positives among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed per-class error analysis

**References:**

- Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance measures for classification tasks. *Information Processing & Management*, 45(4), 427-437. [Comprehensive review of classification metrics]

---

## Conclusion

This literature review establishes that:

1. **Embedding choice significantly impacts performance** based on model architecture
2. **Linear models favor sparse representations** (TF-IDF)
3. **Sequential models leverage dense semantic embeddings** (Word2Vec, GloVe)
4. **Simple baselines can be competitive** with complex models on structured tasks
5. **Pre-trained embeddings transfer external knowledge** effectively

---
