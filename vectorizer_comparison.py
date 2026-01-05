# Vectorizer Comparison Module
# Compares different text vectorization methods

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import (
    TfidfVectorizer, 
    CountVectorizer, 
    HashingVectorizer
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle


def compare_vectorizers(train_texts, test_texts, y_train, y_test, max_features=5000):
    """
    Compare different vectorization methods:
    1. TF-IDF Vectorizer
    2. Count Vectorizer (Bag of Words)
    3. Hashing Vectorizer
    
    Returns: Dictionary with results and timings
    """
    results = {}
    
    # 1. TF-IDF VECTORIZER
    start_time = time.time()
    
    tfidf_vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english'
    )
    
    # Fit and transform
    fit_start = time.time()
    X_train_tfidf = tfidf_vectorizer.fit_transform(train_texts)
    fit_time = time.time() - fit_start
    
    transform_start = time.time()
    X_test_tfidf = tfidf_vectorizer.transform(test_texts)
    transform_time = time.time() - transform_start
    
    total_time = time.time() - start_time
    
    # Train quick model to get accuracy
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y_train)
    accuracy = model.score(X_test_tfidf, y_test)
    
    results['TF-IDF'] = {
        'vectorizer': tfidf_vectorizer,
        'X_train': X_train_tfidf,
        'X_test': X_test_tfidf,
        'fit_time': fit_time,
        'transform_time': transform_time,
        'total_time': total_time,
        'accuracy': accuracy,
        'n_features': X_train_tfidf.shape[1],
        'sparsity': 1.0 - (X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1]))
    }
    

    # 2. COUNT VECTORIZER (Bag of Words)
    start_time = time.time()
    
    count_vectorizer = CountVectorizer(
        max_features=max_features,
        stop_words='english'
    )
    
    # Fit and transform
    fit_start = time.time()
    X_train_count = count_vectorizer.fit_transform(train_texts)
    fit_time = time.time() - fit_start
    
    transform_start = time.time()
    X_test_count = count_vectorizer.transform(test_texts)
    transform_time = time.time() - transform_start
    
    total_time = time.time() - start_time
    
    # Train quick model to get accuracy
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_count, y_train)
    accuracy = model.score(X_test_count, y_test)
    
    results['Count'] = {
        'vectorizer': count_vectorizer,
        'X_train': X_train_count,
        'X_test': X_test_count,
        'fit_time': fit_time,
        'transform_time': transform_time,
        'total_time': total_time,
        'accuracy': accuracy,
        'n_features': X_train_count.shape[1],
        'sparsity': 1.0 - (X_train_count.nnz / (X_train_count.shape[0] * X_train_count.shape[1]))
    }
    
    
    # 3. HASHING VECTORIZER
    start_time = time.time()
    
    hashing_vectorizer = HashingVectorizer(
        n_features=max_features,
        stop_words='english',
        alternate_sign=False  # All features are positive
    )
    
    # Transform (no fit needed!)
    fit_start = time.time()
    X_train_hash = hashing_vectorizer.transform(train_texts)
    fit_time = time.time() - fit_start
    
    transform_start = time.time()
    X_test_hash = hashing_vectorizer.transform(test_texts)
    transform_time = time.time() - transform_start
    
    total_time = time.time() - start_time
    
    # Train quick model to get accuracy
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_hash, y_train)
    accuracy = model.score(X_test_hash, y_test)
    
    results['Hashing'] = {
        'vectorizer': hashing_vectorizer,
        'X_train': X_train_hash,
        'X_test': X_test_hash,
        'fit_time': fit_time,
        'transform_time': transform_time,
        'total_time': total_time,
        'accuracy': accuracy,
        'n_features': X_train_hash.shape[1],
        'sparsity': 1.0 - (X_train_hash.nnz / (X_train_hash.shape[0] * X_train_hash.shape[1]))
    }
    
    return results


def plot_vectorizer_comparison(results, save_path='data/vectorizer_comparison.png'):
    """
    Create visualization comparing vectorizers.
    """
    
    vectorizers = list(results.keys())
    
    # Extract metrics
    fit_times = [results[v]['fit_time'] for v in vectorizers]
    transform_times = [results[v]['transform_time'] for v in vectorizers]
    total_times = [results[v]['total_time'] for v in vectorizers]
    accuracies = [results[v]['accuracy'] for v in vectorizers]
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Processing Time Comparison
    ax = axes[0, 0]
    x = np.arange(len(vectorizers))
    width = 0.35
    
    ax.bar(x - width/2, fit_times, width, label='Fit Time', color='#2196F3')
    ax.bar(x + width/2, transform_times, width, label='Transform Time', color='#4CAF50')
    
    ax.set_xlabel('Vectorizer', fontweight='bold')
    ax.set_ylabel('Time (seconds)', fontweight='bold')
    ax.set_title('Processing Time Comparison', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(vectorizers)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 2. Total Time
    ax = axes[0, 1]
    bars = ax.barh(vectorizers, total_times, color=['#2196F3', '#4CAF50', '#FF9800'])
    ax.set_xlabel('Total Time (seconds)', fontweight='bold')
    ax.set_title('Total Processing Time', fontweight='bold', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars, total_times)):
        ax.text(val + 0.01, i, f'{val:.3f}s', va='center', fontweight='bold')
    
    # 3. Accuracy Comparison
    ax = axes[1, 0]
    bars = ax.bar(vectorizers, accuracies, color=['#2196F3', '#4CAF50', '#FF9800'])
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Classification Accuracy', fontweight='bold', fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Sparsity Comparison
    ax = axes[1, 1]
    sparsities = [results[v]['sparsity'] * 100 for v in vectorizers]
    bars = ax.bar(vectorizers, sparsities, color=['#2196F3', '#4CAF50', '#FF9800'])
    ax.set_ylabel('Sparsity (%)', fontweight='bold')
    ax.set_title('Matrix Sparsity', fontweight='bold', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, sparsities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Graf shranjen: {save_path}")
    plt.close()


def save_vectorizers(results, output_dir='data'):
    """
    Save all vectorizers to disk.
    """
    
    for name, res in results.items():
        filename = f"{output_dir}/{name.lower()}_vectorizer.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(res['vectorizer'], f)
        print(f"  Shranjeno: {filename}")
    
    # Save comparison results
    comparison_file = f"{output_dir}/vectorizer_results.pkl"
    with open(comparison_file, 'wb') as f:
        save_results = {}
        for name, res in results.items():
            save_results[name] = {
                'fit_time': res['fit_time'],
                'transform_time': res['transform_time'],
                'total_time': res['total_time'],
                'accuracy': res['accuracy'],
                'n_features': res['n_features'],
                'sparsity': res['sparsity']
            }
        pickle.dump(save_results, f)
    print(f"  Shranjeno: {comparison_file}")


if __name__ == "__main__":
    import utils
    from sklearn.model_selection import train_test_split
    
    df, category_names = utils.load_or_download_data()
    
    X_text = df['clean_text']
    y = df['target']
    
    X_text_train, X_text_test, y_train, y_test = train_test_split(
        X_text, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    # Primerja vektorizatorje
    train_texts = X_text_train.tolist()
    test_texts = X_text_test.tolist()
    
    results = compare_vectorizers(
        train_texts, 
        test_texts, 
        y_train, 
        y_test,
        max_features=5000
    )
    
    # Ustvari vizualizacijo
    plot_vectorizer_comparison(results, save_path='data/vectorizer_comparison.png')
    
    # Shrani rezultate
    save_vectorizers(results, output_dir='data')
    
    print("\n ✓ Vectorizer comparison done!")

