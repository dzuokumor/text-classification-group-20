"""
Shared Preprocessing Pipeline for AG News Classification
This module contains the standardized preprocessing functions that all team members must use.
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split


class SharedPreprocessor:
    """
    Standardized preprocessing pipeline for AG News Classification Dataset.
    All team members must use this exact preprocessing to ensure fair comparison.
    """
    
    def __init__(self, random_state=42, val_size=0.2):
        """
        Initialize the preprocessor with fixed random state for reproducibility.
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        val_size : float
            Validation set proportion (default 0.2)
        """
        self.random_state = random_state
        self.val_size = val_size
        
    def construct_text(self, df):
        """
        Combine title and description into a single text column.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with 'Title' and 'Description' columns
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with new 'text' column
        """
        df = df.copy()
        df['Title'] = df['Title'].fillna('')
        df['Description'] = df['Description'].fillna('')
        df['text'] = df['Title'] + ' ' + df['Description']
        return df
    
    def clean_text(self, text):
        """
        Apply minimal text cleaning while preserving semantic information.
        
        Cleaning steps:
        1. Convert to lowercase
        2. Remove URLs
        3. Normalize whitespace
        4. Preserve punctuation and word structure
        
        Parameters:
        -----------
        text : str
            Input text string
            
        Returns:
        --------
        str
            Cleaned text
        """
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    def encode_labels(self, df, label_column='Class Index'):
        """
        Convert labels to zero-based indexing.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        label_column : str
            Name of the label column
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with encoded labels
        """
        df = df.copy()
        df['label'] = df[label_column] - 1
        return df
    
    def preprocess_dataset(self, df, label_column='Class Index'):
        """
        Apply full preprocessing pipeline to dataset.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw dataframe
        label_column : str
            Name of the label column
            
        Returns:
        --------
        pd.DataFrame
            Fully preprocessed dataframe
        """
        df = self.construct_text(df)
        df['text'] = df['text'].apply(self.clean_text)
        df = self.encode_labels(df, label_column)
        return df
    
    def create_train_val_split(self, X, y, save_indices=True, indices_dir='./data'):
        """
        Create stratified train-validation split with fixed random state.
        Optionally save indices to ensure all team members use identical splits.
        
        Parameters:
        -----------
        X : array-like
            Feature data
        y : array-like
            Labels
        save_indices : bool
            Whether to save split indices
        indices_dir : str
            Directory to save indices
            
        Returns:
        --------
        tuple
            (X_train, X_val, y_train, y_val, train_idx, val_idx)
        """
        X_train, X_val, y_train, y_val, train_idx, val_idx = train_test_split(
            X, y, np.arange(len(y)),
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=y
        )
        
        if save_indices:
            import os
            os.makedirs(indices_dir, exist_ok=True)
            np.save(f'{indices_dir}/train_indices.npy', train_idx)
            np.save(f'{indices_dir}/val_indices.npy', val_idx)
            
        return X_train, X_val, y_train, y_val, train_idx, val_idx
    
    def load_and_split_with_saved_indices(self, X, y, indices_dir='./data'):
        """
        Load previously saved train-validation indices.
        This ensures all team members use identical splits.
        
        Parameters:
        -----------
        X : array-like
            Feature data
        y : array-like
            Labels
        indices_dir : str
            Directory containing saved indices
            
        Returns:
        --------
        tuple
            (X_train, X_val, y_train, y_val)
        """
        train_idx = np.load(f'{indices_dir}/train_indices.npy')
        val_idx = np.load(f'{indices_dir}/val_indices.npy')
        
        if isinstance(X, pd.Series):
            X_train = X.iloc[train_idx].reset_index(drop=True)
            X_val = X.iloc[val_idx].reset_index(drop=True)
        else:
            X_train = X[train_idx]
            X_val = X[val_idx]
            
        if isinstance(y, pd.Series):
            y_train = y.iloc[train_idx].reset_index(drop=True)
            y_val = y.iloc[val_idx].reset_index(drop=True)
        else:
            y_train = y[train_idx]
            y_val = y[val_idx]
            
        return X_train, X_val, y_train, y_val


def verify_preprocessing(df_original, df_processed):
    """
    Verify that preprocessing was applied correctly.
    
    Parameters:
    -----------
    df_original : pd.DataFrame
        Original dataframe
    df_processed : pd.DataFrame
        Processed dataframe
    """
    print("Preprocessing Verification")
    print("=" * 60)
    print(f"Original shape: {df_original.shape}")
    print(f"Processed shape: {df_processed.shape}")
    print(f"\nOriginal columns: {list(df_original.columns)}")
    print(f"Processed columns: {list(df_processed.columns)}")
    print(f"\nLabel distribution:")
    print(df_processed['label'].value_counts().sort_index())
    print(f"\nSample processed text (first 3):")
    for i in range(min(3, len(df_processed))):
        print(f"\nLabel {df_processed['label'].iloc[i]}:")
        print(f"{df_processed['text'].iloc[i][:200]}...")
    print("=" * 60)


if __name__ == "__main__":
    print("Shared Preprocessing Module")
    print("This module should be imported by all team members")
    print("Ensure all team members use identical preprocessing settings")
