"""
Logistic Regression with Multiple Embeddings for AG News Classification
Implementation by: [Your Name]
Model: Logistic Regression
Embeddings: TF-IDF, Word2Vec (Skip-gram & CBOW), GloVe
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from gensim.models import Word2Vec
import gensim.downloader as api
import warnings
warnings.filterwarnings('ignore')


class LogisticRegressionClassifier:
    """
    Logistic Regression classifier with support for multiple embedding techniques.
    """
    
    def __init__(self, embedding_type='tfidf', max_features=5000, vector_size=100, 
                 window=5, min_count=2, workers=4, random_state=42):
        """
        Initialize the classifier.
        
        Parameters:
        -----------
        embedding_type : str
            Type of embedding ('tfidf', 'word2vec_skipgram', 'word2vec_cbow', 'glove')
        max_features : int
            Maximum features for TF-IDF
        vector_size : int
            Embedding dimension for Word2Vec
        window : int
            Context window size for Word2Vec
        min_count : int
            Minimum word frequency for Word2Vec
        workers : int
            Number of workers for Word2Vec training
        random_state : int
            Random seed for reproducibility
        """
        self.embedding_type = embedding_type
        self.max_features = max_features
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.random_state = random_state
        
        self.vectorizer = None
        self.word2vec_model = None
        self.glove_model = None
        self.classifier = None
        self.class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
        
    def _tokenize_texts(self, texts):
        """
        Tokenize texts for Word2Vec and GloVe embeddings.
        
        Parameters:
        -----------
        texts : list or pd.Series
            Text data
            
        Returns:
        --------
        list
            List of tokenized sentences
        """
        return [text.split() for text in texts]
    
    def _train_tfidf(self, texts_train):
        """
        Train TF-IDF vectorizer.
        
        Parameters:
        -----------
        texts_train : list or pd.Series
            Training texts
            
        Returns:
        --------
        np.ndarray
            TF-IDF features
        """
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        X_train = self.vectorizer.fit_transform(texts_train)
        return X_train.toarray()
    
    def _transform_tfidf(self, texts):
        """
        Transform texts using trained TF-IDF vectorizer.
        
        Parameters:
        -----------
        texts : list or pd.Series
            Texts to transform
            
        Returns:
        --------
        np.ndarray
            TF-IDF features
        """
        X = self.vectorizer.transform(texts)
        return X.toarray()
    
    def _train_word2vec(self, texts_train, sg=1):
        """
        Train Word2Vec model.
        
        Parameters:
        -----------
        texts_train : list or pd.Series
            Training texts
        sg : int
            Training algorithm (1 for skip-gram, 0 for CBOW)
            
        Returns:
        --------
        np.ndarray
            Word2Vec averaged embeddings
        """
        tokenized_texts = self._tokenize_texts(texts_train)
        
        self.word2vec_model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=sg,
            seed=self.random_state
        )
        
        return self._text_to_word2vec_vector(texts_train)
    
    def _text_to_word2vec_vector(self, texts):
        """
        Convert texts to Word2Vec averaged embeddings.
        
        Parameters:
        -----------
        texts : list or pd.Series
            Texts to convert
            
        Returns:
        --------
        np.ndarray
            Averaged Word2Vec embeddings
        """
        tokenized_texts = self._tokenize_texts(texts)
        embeddings = []
        
        for tokens in tokenized_texts:
            word_vectors = []
            for word in tokens:
                if word in self.word2vec_model.wv:
                    word_vectors.append(self.word2vec_model.wv[word])
            
            if len(word_vectors) > 0:
                embeddings.append(np.mean(word_vectors, axis=0))
            else:
                embeddings.append(np.zeros(self.vector_size))
        
        return np.array(embeddings)
    
    def _load_glove(self):
        """
        Load pre-trained GloVe embeddings.
        
        Returns:
        --------
        dict
            GloVe word vectors
        """
        print("Loading pre-trained GloVe embeddings (this may take a few minutes)...")
        self.glove_model = api.load('glove-wiki-gigaword-100')
        print("GloVe embeddings loaded successfully")
    
    def _text_to_glove_vector(self, texts):
        """
        Convert texts to GloVe averaged embeddings.
        
        Parameters:
        -----------
        texts : list or pd.Series
            Texts to convert
            
        Returns:
        --------
        np.ndarray
            Averaged GloVe embeddings
        """
        if self.glove_model is None:
            self._load_glove()
        
        tokenized_texts = self._tokenize_texts(texts)
        embeddings = []
        
        for tokens in tokenized_texts:
            word_vectors = []
            for word in tokens:
                if word in self.glove_model:
                    word_vectors.append(self.glove_model[word])
            
            if len(word_vectors) > 0:
                embeddings.append(np.mean(word_vectors, axis=0))
            else:
                embeddings.append(np.zeros(self.glove_model.vector_size))
        
        return np.array(embeddings)
    
    def fit(self, X_train, y_train):
        """
        Fit the Logistic Regression model with specified embedding.
        
        Parameters:
        -----------
        X_train : list or pd.Series
            Training texts
        y_train : array-like
            Training labels
        """
        print(f"\nTraining Logistic Regression with {self.embedding_type}...")
        
        if self.embedding_type == 'tfidf':
            X_train_embedded = self._train_tfidf(X_train)
        elif self.embedding_type == 'word2vec_skipgram':
            X_train_embedded = self._train_word2vec(X_train, sg=1)
        elif self.embedding_type == 'word2vec_cbow':
            X_train_embedded = self._train_word2vec(X_train, sg=0)
        elif self.embedding_type == 'glove':
            X_train_embedded = self._text_to_glove_vector(X_train)
        else:
            raise ValueError(f"Unknown embedding type: {self.embedding_type}")
        
        self.classifier = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            solver='lbfgs',
            multi_class='multinomial',
            C=1.0
        )
        
        self.classifier.fit(X_train_embedded, y_train)
        print(f"Training completed for {self.embedding_type}")
        
    def predict(self, X_test):
        """
        Predict labels for test data.
        
        Parameters:
        -----------
        X_test : list or pd.Series
            Test texts
            
        Returns:
        --------
        np.ndarray
            Predicted labels
        """
        if self.embedding_type == 'tfidf':
            X_test_embedded = self._transform_tfidf(X_test)
        elif self.embedding_type in ['word2vec_skipgram', 'word2vec_cbow']:
            X_test_embedded = self._text_to_word2vec_vector(X_test)
        elif self.embedding_type == 'glove':
            X_test_embedded = self._text_to_glove_vector(X_test)
        else:
            raise ValueError(f"Unknown embedding type: {self.embedding_type}")
        
        return self.classifier.predict(X_test_embedded)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.
        
        Parameters:
        -----------
        X_test : list or pd.Series
            Test texts
        y_test : array-like
            True labels
            
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        y_pred = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )
        
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            y_test, y_pred, average=None, zero_division=0
        )
        
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'confusion_matrix': cm,
            'predictions': y_pred
        }
    
    def plot_confusion_matrix(self, cm, title='Confusion Matrix'):
        """
        Plot confusion matrix heatmap.
        
        Parameters:
        -----------
        cm : np.ndarray
            Confusion matrix
        title : str
            Plot title
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        return plt.gcf()


class HyperparameterTuner:
    """
    Hyperparameter tuning for Logistic Regression with different embeddings.
    """
    
    def __init__(self, embedding_type, random_state=42):
        """
        Initialize hyperparameter tuner.
        
        Parameters:
        -----------
        embedding_type : str
            Type of embedding
        random_state : int
            Random seed
        """
        self.embedding_type = embedding_type
        self.random_state = random_state
        self.best_params = None
        self.best_score = 0
        self.results = []
        
    def tune_tfidf(self, X_train, y_train, X_val, y_val):
        """
        Tune hyperparameters for TF-IDF based model.
        
        Parameters:
        -----------
        X_train : pd.Series
            Training texts
        y_train : array-like
            Training labels
        X_val : pd.Series
            Validation texts
        y_val : array-like
            Validation labels
            
        Returns:
        --------
        dict
            Best hyperparameters
        """
        max_features_options = [2000, 3000, 5000]
        c_options = [0.1, 1.0, 10.0]
        ngram_options = [(1, 1), (1, 2)]
        
        print(f"\nTuning hyperparameters for TF-IDF...")
        
        for max_feat in max_features_options:
            for c_val in c_options:
                for ngram in ngram_options:
                    vectorizer = TfidfVectorizer(
                        max_features=max_feat,
                        ngram_range=ngram,
                        min_df=2,
                        max_df=0.95
                    )
                    
                    X_train_vec = vectorizer.fit_transform(X_train).toarray()
                    X_val_vec = vectorizer.transform(X_val).toarray()
                    
                    clf = LogisticRegression(
                        C=c_val,
                        max_iter=1000,
                        random_state=self.random_state,
                        solver='lbfgs',
                        multi_class='multinomial'
                    )
                    
                    clf.fit(X_train_vec, y_train)
                    val_score = clf.score(X_val_vec, y_val)
                    
                    self.results.append({
                        'max_features': max_feat,
                        'C': c_val,
                        'ngram_range': ngram,
                        'val_accuracy': val_score
                    })
                    
                    if val_score > self.best_score:
                        self.best_score = val_score
                        self.best_params = {
                            'max_features': max_feat,
                            'C': c_val,
                            'ngram_range': ngram
                        }
                    
                    print(f"max_features={max_feat}, C={c_val}, ngram={ngram}: {val_score:.4f}")
        
        print(f"\nBest parameters: {self.best_params}")
        print(f"Best validation accuracy: {self.best_score:.4f}")
        return self.best_params
    
    def tune_word2vec(self, X_train, y_train, X_val, y_val, sg=1):
        """
        Tune hyperparameters for Word2Vec based model.
        
        Parameters:
        -----------
        X_train : pd.Series
            Training texts
        y_train : array-like
            Training labels
        X_val : pd.Series
            Validation texts
        y_val : array-like
            Validation labels
        sg : int
            Training algorithm (1 for skip-gram, 0 for CBOW)
            
        Returns:
        --------
        dict
            Best hyperparameters
        """
        vector_sizes = [50, 100, 200]
        window_sizes = [3, 5, 7]
        c_options = [0.1, 1.0, 10.0]
        
        model_type = 'Skip-gram' if sg == 1 else 'CBOW'
        print(f"\nTuning hyperparameters for Word2Vec ({model_type})...")
        
        for vec_size in vector_sizes:
            for window in window_sizes:
                for c_val in c_options:
                    clf_model = LogisticRegressionClassifier(
                        embedding_type='word2vec_skipgram' if sg == 1 else 'word2vec_cbow',
                        vector_size=vec_size,
                        window=window,
                        random_state=self.random_state
                    )
                    
                    tokenized_train = [text.split() for text in X_train]
                    w2v_model = Word2Vec(
                        sentences=tokenized_train,
                        vector_size=vec_size,
                        window=window,
                        min_count=2,
                        workers=4,
                        sg=sg,
                        seed=self.random_state
                    )
                    
                    def text_to_vec(texts, model, vec_size):
                        embeddings = []
                        for text in texts:
                            tokens = text.split()
                            word_vectors = [model.wv[word] for word in tokens if word in model.wv]
                            if len(word_vectors) > 0:
                                embeddings.append(np.mean(word_vectors, axis=0))
                            else:
                                embeddings.append(np.zeros(vec_size))
                        return np.array(embeddings)
                    
                    X_train_vec = text_to_vec(X_train, w2v_model, vec_size)
                    X_val_vec = text_to_vec(X_val, w2v_model, vec_size)
                    
                    clf = LogisticRegression(
                        C=c_val,
                        max_iter=1000,
                        random_state=self.random_state,
                        solver='lbfgs',
                        multi_class='multinomial'
                    )
                    
                    clf.fit(X_train_vec, y_train)
                    val_score = clf.score(X_val_vec, y_val)
                    
                    self.results.append({
                        'vector_size': vec_size,
                        'window': window,
                        'C': c_val,
                        'val_accuracy': val_score
                    })
                    
                    if val_score > self.best_score:
                        self.best_score = val_score
                        self.best_params = {
                            'vector_size': vec_size,
                            'window': window,
                            'C': c_val
                        }
                    
                    print(f"vector_size={vec_size}, window={window}, C={c_val}: {val_score:.4f}")
        
        print(f"\nBest parameters: {self.best_params}")
        print(f"Best validation accuracy: {self.best_score:.4f}")
        return self.best_params
    
    def tune_glove(self, X_train, y_train, X_val, y_val):
        """
        Tune hyperparameters for GloVe based model.
        GloVe embeddings are pre-trained, so only tune classifier parameters.
        
        Parameters:
        -----------
        X_train : pd.Series
            Training texts
        y_train : array-like
            Training labels
        X_val : pd.Series
            Validation texts
        y_val : array-like
            Validation labels
            
        Returns:
        --------
        dict
            Best hyperparameters
        """
        c_options = [0.1, 1.0, 10.0, 100.0]
        
        print(f"\nTuning hyperparameters for GloVe...")
        print("Loading GloVe embeddings...")
        glove_model = api.load('glove-wiki-gigaword-100')
        
        def text_to_glove_vec(texts, model):
            embeddings = []
            for text in texts:
                tokens = text.split()
                word_vectors = [model[word] for word in tokens if word in model]
                if len(word_vectors) > 0:
                    embeddings.append(np.mean(word_vectors, axis=0))
                else:
                    embeddings.append(np.zeros(model.vector_size))
            return np.array(embeddings)
        
        X_train_vec = text_to_glove_vec(X_train, glove_model)
        X_val_vec = text_to_glove_vec(X_val, glove_model)
        
        for c_val in c_options:
            clf = LogisticRegression(
                C=c_val,
                max_iter=1000,
                random_state=self.random_state,
                solver='lbfgs',
                multi_class='multinomial'
            )
            
            clf.fit(X_train_vec, y_train)
            val_score = clf.score(X_val_vec, y_val)
            
            self.results.append({
                'C': c_val,
                'val_accuracy': val_score
            })
            
            if val_score > self.best_score:
                self.best_score = val_score
                self.best_params = {'C': c_val}
            
            print(f"C={c_val}: {val_score:.4f}")
        
        print(f"\nBest parameters: {self.best_params}")
        print(f"Best validation accuracy: {self.best_score:.4f}")
        return self.best_params


if __name__ == "__main__":
    print("Logistic Regression Classifier Module")
    print("This module implements Logistic Regression with TF-IDF, Word2Vec, and GloVe")
