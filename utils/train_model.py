import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib
import os

def train_model():
    # Read the dataset
    print("Loading dataset...")
    df = pd.read_csv('AI_Human.csv')
    
    # Assuming the CSV has 'text' and 'label' columns
    # If column names are different, adjust accordingly
    X = df['text'].astype(str)
    y = df['label'].astype(int)  # 1 for AI, 0 for human
    
    # Split the dataset
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a pipeline with TF-IDF and Random Forest
    print("Creating and training model pipeline...")
    model = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english'
        )),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            n_jobs=-1,
            random_state=42
        ))
    ])
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate the model
    print("\nModel Evaluation:")
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Training accuracy: {train_score:.4f}")
    print(f"Testing accuracy: {test_score:.4f}")
    
    # Save the model
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'text_classifier.joblib')
    print(f"\nSaving model to {model_path}")
    joblib.dump(model, model_path)
    print("Model saved successfully!")

if __name__ == '__main__':
    train_model() 