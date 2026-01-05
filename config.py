# Configuration file


import os

# Paths
DATA_DIR = "data"
DATA_PATH = os.path.join(DATA_DIR, "20newsgroups_clean.csv")
MODEL_PATH = os.path.join(DATA_DIR, "trained_model.pkl")
VECTORIZER_PATH = os.path.join(DATA_DIR, "vectorizer.pkl")

# Parameters
MAX_FEATURES = 5000
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model Configurations
MODELS_CONFIG = {
    "Naive Bayes": {
        "class": "MultinomialNB",
        "params": {"alpha": 1.0}
    },
    "Linear SVM": {
        "class": "LinearSVC",
        "params": {"max_iter": 10000, "dual": "auto", "random_state": RANDOM_STATE}
    },
    "Logistic Regression": {
        "class": "LogisticRegression",
        "params": {"max_iter": 1000, "random_state": RANDOM_STATE}
    }
}

# Text Cleaning
STOP_WORDS = 'english'
MIN_TEXT_LENGTH = 5

# Visualization
FIGSIZE_SMALL = (6, 4)
FIGSIZE_MEDIUM = (10, 6)
FIGSIZE_LARGE = (12, 8)
COLOR_CORRECT = "#4CAF50"
COLOR_INCORRECT = "#F44336"
COLOR_PALETTE = "plasma"

# Streamlit
DEFAULT_NUM_EXAMPLES = 5
MAX_NUM_EXAMPLES = 20
