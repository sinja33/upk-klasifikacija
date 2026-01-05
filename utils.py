# Utility functions

import pandas as pd
import numpy as np
import re
import os
import pickle
import logging
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional

import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Keep alphanumeric and spaces (preserves numbers)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Return empty string if text too short
    if len(text) < config.MIN_TEXT_LENGTH:
        return ""
    
    return text


def load_or_download_data() -> Tuple[pd.DataFrame, list]:
    """
    Load data from cache or download from sklearn.
    
    Returns:
        Tuple of (DataFrame, category_names)
    """
    os.makedirs(config.DATA_DIR, exist_ok=True)
    
    if os.path.exists(config.DATA_PATH):
        logger.info(f"Loading cached data from {config.DATA_PATH}")
        try:
            df = pd.read_csv(config.DATA_PATH)
            df['clean_text'] = df['clean_text'].fillna("")
            
            # Validate data
            if df.empty or 'clean_text' not in df.columns:
                raise ValueError("Cached data is invalid")
            
            category_names = fetch_20newsgroups(subset='all').target_names
            logger.info(f"Loaded {len(df)} documents")
            return df, category_names
            
        except Exception as e:
            logger.warning(f"Failed to load cached data: {e}. Re-downloading...")
    
    # Download and process data
    logger.info("Downloading 20 newsgroups dataset...")
    try:
        newsgroups = fetch_20newsgroups(
            subset='all',
            remove=('headers', 'footers', 'quotes')
        )
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        raise
    
    df = pd.DataFrame({
        'text': newsgroups.data,
        'target': newsgroups.target
    })
    
    category_names = newsgroups.target_names
    df['category'] = df['target'].apply(lambda i: category_names[i])
    
    # Clean text
    logger.info("Cleaning text...")
    df['clean_text'] = df['text'].apply(clean_text)
    
    # Remove empty documents
    initial_len = len(df)
    df = df[df['clean_text'] != ""].reset_index(drop=True)
    logger.info(f"Removed {initial_len - len(df)} empty documents")
    
    # Save to cache
    try:
        df.to_csv(config.DATA_PATH, index=False)
        logger.info(f"Saved cleaned data to {config.DATA_PATH}")
    except Exception as e:
        logger.warning(f"Failed to cache data: {e}")
    
    return df, category_names


def prepare_data(df: pd.DataFrame) -> Tuple:
    """
    Prepare train/test split with proper vectorization.
    
    Args:
        df: DataFrame with 'clean_text' and 'target' columns
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, idx_train, idx_test, vectorizer)
    """
    logger.info("Preparing train/test split...")
    
    # Split BEFORE vectorization to avoid data leakage
    indices = np.arange(len(df))
    X_text_train, X_text_test, y_train, y_test, idx_train, idx_test = train_test_split(
        df['clean_text'],
        df['target'],
        indices,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=df['target']  # Ensure balanced split
    )
    
    # Vectorize - fit only on training data
    logger.info("Vectorizing text...")
    vectorizer = TfidfVectorizer(
        max_features=config.MAX_FEATURES,
        stop_words=config.STOP_WORDS,
        min_df=2,  # Ignore very rare words
        max_df=0.95  # Ignore very common words
    )
    
    X_train = vectorizer.fit_transform(X_text_train)
    X_test = vectorizer.transform(X_text_test) 
    
    logger.info(f"Training set: {X_train.shape[0]} documents")
    logger.info(f"Test set: {X_test.shape[0]} documents")
    logger.info(f"Features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test, idx_train, idx_test, vectorizer


def save_model(model, vectorizer, model_path: str = None, vectorizer_path: str = None):
    """Save trained model and vectorizer to disk."""
    model_path = model_path or config.MODEL_PATH
    vectorizer_path = vectorizer_path or config.VECTORIZER_PATH
    
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Vectorizer saved to {vectorizer_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")


def load_model(model_path: str = None, vectorizer_path: str = None) -> Tuple:
    """Load trained model and vectorizer from disk."""
    model_path = model_path or config.MODEL_PATH
    vectorizer_path = vectorizer_path or config.VECTORIZER_PATH
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        logger.info("Model and vectorizer loaded successfully")
        return model, vectorizer
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None, None


def get_category_metrics(y_test, y_pred, category_idx: int, category_names: list) -> dict:
    """
    Calculate metrics for a specific category.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        category_idx: Index of category to analyze
        category_names: List of category names
        
    Returns:
        Dictionary with metrics
    """
    # Find indices where true label matches category
    idx = [i for i, c in enumerate(y_test) if int(c) == category_idx]
    
    if not idx:
        return {
            'total': 0,
            'correct': 0,
            'incorrect': 0,
            'accuracy': 0.0,
            'indices': [],
            'wrong_predictions': []
        }
    
    # Calculate metrics
    correct = sum(int(y_test.iloc[i]) == int(y_pred[i]) for i in idx)
    total = len(idx)
    accuracy = correct / total if total > 0 else 0
    
    # Get wrong predictions
    wrong_preds = [int(y_pred[i]) for i in idx if int(y_pred[i]) != int(y_test.iloc[i])]
    
    return {
        'total': total,
        'correct': correct,
        'incorrect': total - correct,
        'accuracy': accuracy,
        'indices': idx,
        'wrong_predictions': wrong_preds
    }
