import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

class SentimentAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = LogisticRegression(max_iter=1000)
        
    def prepare_data(self, df, text_column='processed_text', label_column='sentiment'):
        """
        Prepare data for training
        """
        X = self.vectorizer.fit_transform(df[text_column])
        y = df[label_column]
        return X, y
    
    def train(self, X, y, test_size=0.2):
        """
        Train the sentiment analysis model
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred)
        
        print("Model Performance:")
        print(report)
        
        return X_test, y_test, y_pred
    
    def predict(self, text):
        """
        Predict sentiment for a single text
        """
        # Transform text
        text_vectorized = self.vectorizer.transform([text])
        
        # Make prediction
        prediction = self.model.predict(text_vectorized)
        probabilities = self.model.predict_proba(text_vectorized)
        
        return prediction[0], probabilities[0]
    
    def save_model(self, model_dir='models'):
        """
        Save the trained model and vectorizer
        """
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        joblib.dump(self.model, f'{model_dir}/sentiment_model.joblib')
        joblib.dump(self.vectorizer, f'{model_dir}/vectorizer.joblib')
        
    def load_model(self, model_dir='models'):
        """
        Load the trained model and vectorizer
        """
        self.model = joblib.load(f'{model_dir}/sentiment_model.joblib')
        self.vectorizer = joblib.load(f'{model_dir}/vectorizer.joblib')

if __name__ == "__main__":
    # Example usage
    analyzer = SentimentAnalyzer()
    
    # Example data
    data = {
        'processed_text': [
            'great product love it',
            'terrible service disappointed',
            'okay experience nothing special'
        ],
        'sentiment': [1, -1, 0]
    }
    df = pd.DataFrame(data)
    
    # Train model
    X, y = analyzer.prepare_data(df)
    X_test, y_test, y_pred = analyzer.train(X, y)
    
    # Save model
    analyzer.save_model() 