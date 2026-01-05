# Broader categories
# Maps 20 newsgroups into 6 broader categories

import pandas as pd
import numpy as np

# Mapping
BROADER_CATEGORY_MAPPING = {
    # Computers
    'comp.graphics': 'Computers',
    'comp.os.ms-windows.misc': 'Computers',
    'comp.sys.ibm.pc.hardware': 'Computers',
    'comp.sys.mac.hardware': 'Computers',
    'comp.windows.x': 'Computers',
    
    # Recreation
    'rec.autos': 'Recreation',
    'rec.motorcycles': 'Recreation',
    'rec.sport.baseball': 'Recreation',
    'rec.sport.hockey': 'Recreation',
    
    # Science
    'sci.crypt': 'Science',
    'sci.electronics': 'Science',
    'sci.med': 'Science',
    'sci.space': 'Science',
    
    # Politics
    'talk.politics.guns': 'Politics',
    'talk.politics.mideast': 'Politics',
    'talk.politics.misc': 'Politics',
    
    # Religion
    'talk.religion.misc': 'Religion',
    'soc.religion.christian': 'Religion',
    'alt.atheism': 'Religion',
    
    # Miscellaneous
    'misc.forsale': 'Miscellaneous',
}

# Broader category names
BROADER_CATEGORIES = [
    'Computers',
    'Recreation', 
    'Science',
    'Politics',
    'Religion',
    'Miscellaneous'
]


def create_broader_dataset(df, original_category_names):
    """
    Create broader category version of the dataset.
    
    Args:
        df: DataFrame with 'category' and 'target' columns
        original_category_names: List of original 20 category names
        
    Returns:
        df_broader: DataFrame with broader categories
        broader_category_names: List of 6 broader category names
    """
    df_broader = df.copy()
    
    # Map original categories to broader ones
    df_broader['broader_category'] = df_broader['category'].map(BROADER_CATEGORY_MAPPING)
    
    # Create numeric targets for broader categories
    broader_category_to_id = {cat: i for i, cat in enumerate(BROADER_CATEGORIES)}
    df_broader['broader_target'] = df_broader['broader_category'].map(broader_category_to_id)
    
    print("\nMapping summary:")
    for broader_cat in BROADER_CATEGORIES:
        original_cats = [k for k, v in BROADER_CATEGORY_MAPPING.items() if v == broader_cat]
        count = len(df_broader[df_broader['broader_category'] == broader_cat])
        print(f"  {broader_cat:15s} ({len(original_cats)} categories) → {count:5d} documents")
        for orig in original_cats:
            print(f"    - {orig}")
    
    return df_broader, BROADER_CATEGORIES


def get_broader_category_info():
    """
    Return information about broader categories for display.
    """
    info = []
    
    for broader_cat in BROADER_CATEGORIES:
        original_cats = [k for k, v in BROADER_CATEGORY_MAPPING.items() if v == broader_cat]
        info.append({
            'broader_category': broader_cat,
            'num_original': len(original_cats),
            'original_categories': original_cats
        })
    
    return info


if __name__ == "__main__":
    from sklearn.datasets import fetch_20newsgroups
    
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    
    df = pd.DataFrame({
        'text': newsgroups.data,
        'target': newsgroups.target,
        'category': [newsgroups.target_names[i] for i in newsgroups.target]
    })
    
    df_broader, broader_cats = create_broader_dataset(df, newsgroups.target_names)
    
    print("\n✓ Broader category dataset created successfully!")
