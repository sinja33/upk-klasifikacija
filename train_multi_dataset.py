# Multi-Dataset Training Script
# Trains models on all 3 datasets

import pandas as pd
import numpy as np
import pickle
import os
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import config
import utils
import broader_categories
import slovenian_placeholder


# DATASET CONFIGURATIONS

DATASETS = {
    '20news_all': {
        'name': '20 Newsgroups (All 20 Categories)',
        'short_name': '20news_all',
        'num_categories': 20
    },
    '20news_broader': {
        'name': '20 Newsgroups (Broader - 6 Categories)',
        'short_name': '20news_broader',
        'num_categories': 6
    },
    'slovenian': {
        'name': 'Slovenski Teksti',
        'short_name': 'slovenian',
        'num_categories': 5
    }
}


# MODEL CONFIGURATIONS

MODELS_CONFIG = {
    'Naive Bayes': {
        'class': MultinomialNB,
        'params': {
            'alpha': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        }
    },
    'Logistic Regression': {
        'class': LogisticRegression,
        'params': {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'max_iter': [1000, 2000],
            'solver': ['lbfgs', 'saga'],
            'random_state': [config.RANDOM_STATE]
        }
    },
    'Linear SVM': {
        'class': LinearSVC,
        'params': {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'max_iter': [5000, 10000],
            'dual': ['auto'],
            'random_state': [config.RANDOM_STATE]
        }
    }
}


# LOAD DATASETS

def load_all_datasets():
    """
    Load all three datasets.
    
    Returns:
        datasets_dict: Dictionary with dataset info and data
    """
    
    datasets_dict = {}
    
    # 1. Load original 20 Newsgroups (All)
    df_all, category_names_all = utils.load_or_download_data()
    
    datasets_dict['20news_all'] = {
        'df': df_all,
        'category_names': category_names_all,
        'target_column': 'target',
        'info': DATASETS['20news_all']
    }
    
    
    # 2. Create broader categories version
    df_broader, category_names_broader = broader_categories.create_broader_dataset(
        df_all, category_names_all
    )
    
    # Use broader_target instead of target
    df_broader_copy = df_broader.copy()
    df_broader_copy['target'] = df_broader_copy['broader_target']
    df_broader_copy['category'] = df_broader_copy['broader_category']
    
    datasets_dict['20news_broader'] = {
        'df': df_broader_copy,
        'category_names': category_names_broader,
        'target_column': 'target',
        'info': DATASETS['20news_broader']
    }
    
    
    # 3. Load/Create Slovenian dataset
    
    slovenian_path = 'data/slovenian_news_final.csv'
    
    if os.path.exists(slovenian_path):
        print(f"   Found real dataset: {slovenian_path}")
        df_slovenian = pd.read_csv(slovenian_path, encoding='utf-8')
        category_names_slovenian = sorted(df_slovenian['category'].unique())
    else:
        print("   Real dataset not found, creating placeholder...")
        df_slovenian, category_names_slovenian = slovenian_placeholder.create_slovenian_dataset()
    
    datasets_dict['slovenian'] = {
        'df': df_slovenian,
        'category_names': category_names_slovenian,
        'target_column': 'target',
        'info': DATASETS['slovenian']
    }
    
    
    return datasets_dict


# TRAIN SINGLE MODEL ON DATASET

def train_model_on_dataset(dataset_key, dataset_info, model_name, model_config):
    """
    Train a single model on a single dataset.
    
    Returns:
        results: Dictionary with trained model and metrics
    """
    
    df = dataset_info['df']
    category_names = dataset_info['category_names']
    
    # Prepare data
    print("\nPreparing data...")
    X_text = df['clean_text']
    y = df['target']
    
    X_text_train, X_text_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y
    )
    
    # Vectorize
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    vectorizer = TfidfVectorizer(
        max_features=config.MAX_FEATURES,
        stop_words='english' if dataset_key != 'slovenian' else None,
        min_df=2,
        max_df=0.95
    )
    
    X_train = vectorizer.fit_transform(X_text_train)
    X_test = vectorizer.transform(X_text_test)
    
    
    # Grid Search
    model_class = model_config['class']
    param_grid = model_config['params']
    
    grid_search = GridSearchCV(
        model_class(),
        param_grid,
        cv=3,  # Reduced from 5 for speed
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    cv_score = grid_search.best_score_
    
    # Test
    y_pred = best_model.predict(X_test)
    test_score = accuracy_score(y_test, y_pred)
    
    
    # Save results
    results = {
        'model': best_model,
        'vectorizer': vectorizer,
        'best_params': best_params,
        'cv_score': cv_score,
        'test_score': test_score,
        'train_time': train_time,
        'y_test': y_test,
        'y_pred': y_pred,
        'category_names': category_names,
        'X_test': X_test  # Keep for later analysis
    }
    
    return results


# TRAIN ALL COMBINATIONS

def train_all_combinations(datasets_dict, output_dir='data/models'):
    """
    Train all 12 model combinations (3 datasets × 4 models).
    
    Note: BERT training is separate due to different architecture.
    """
    print("\n" + "="*70)
    print("TRAINING ALL MODEL COMBINATIONS")
    print("="*70)
    print(f"\nDatasets: {len(datasets_dict)}")
    print(f"Models: {len(MODELS_CONFIG)}")
    print(f"Total combinations: {len(datasets_dict) * len(MODELS_CONFIG)}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    
    total = len(datasets_dict) * len(MODELS_CONFIG)
    current = 0
    
    for dataset_key, dataset_info in datasets_dict.items():
        for model_name, model_config in MODELS_CONFIG.items():
            current += 1
            
            print(f"\n{'='*70}")
            print(f"PROGRESS: {current}/{total}")
            print(f"{'='*70}")
            
            # Train
            results = train_model_on_dataset(
                dataset_key, dataset_info, model_name, model_config
            )
            
            # Store
            key = f"{dataset_key}_{model_name.replace(' ', '_')}"
            all_results[key] = results
            
            # Save individual model
            model_path = os.path.join(output_dir, f"{key}.pkl")
            
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': results['model'],
                    'vectorizer': results['vectorizer'],
                    'best_params': results['best_params'],
                    'cv_score': results['cv_score'],
                    'test_score': results['test_score'],
                    'category_names': results['category_names']
                }, f)
            
            print(f"\n✓ Saved: {model_path}")
    
    # Save summary
    summary_path = os.path.join(output_dir, 'training_summary.pkl')
    
    summary = {}
    for key, results in all_results.items():
        summary[key] = {
            'cv_score': results['cv_score'],
            'test_score': results['test_score'],
            'train_time': results['train_time'],
            'best_params': results['best_params']
        }
    
    with open(summary_path, 'wb') as f:
        pickle.dump(summary, f)
    
    print(f"\n✓ Saved summary: {summary_path}")
    
    return all_results



# MAIN EXECUTION

def main():
    """
    Main training workflow.
    """
    print("\n" + "="*70)
    print("MULTI-DATASET MODEL TRAINING")
    print("="*70)
    
    # Load all datasets
    datasets_dict = load_all_datasets()
    
    # Train all combinations
    all_results = train_all_combinations(datasets_dict)
    
    # Print summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*70)
    
    summary_data = []
    for key, results in all_results.items():
        parts = key.split('_')
        dataset = parts[0] + ('_' + parts[1] if len(parts) > 2 and parts[1] == 'broader' else '')
        model = ' '.join(parts[-2:]) if 'news' in parts[0] else ' '.join(parts[1:])
        
        summary_data.append({
            'Dataset': dataset,
            'Model': model,
            'CV Score': f"{results['cv_score']:.4f}",
            'Test Score': f"{results['test_score']:.4f}",
            'Time (s)': f"{results['train_time']:.1f}"
        })
    
    df_summary = pd.DataFrame(summary_data)
    print("\n", df_summary.to_string(index=False))
    
    print("\n✓ All models trained and saved in data/models/")
    print("\nNext step: Run streamlit app with new multi-dataset support!")


if __name__ == "__main__":
    main()
