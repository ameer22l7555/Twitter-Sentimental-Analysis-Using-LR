import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

class SentimentVisualizer:
    def __init__(self):
        plt.style.use('seaborn')
        
    def plot_sentiment_distribution(self, df, sentiment_column='sentiment'):
        """
        Plot the distribution of sentiments
        """
        plt.figure(figsize=(10, 6))
        sentiment_counts = df[sentiment_column].value_counts()
        
        # Create bar plot
        bars = plt.bar(sentiment_counts.index, sentiment_counts.values)
        
        # Customize plot
        plt.title('Distribution of Sentiments', fontsize=14, pad=20)
        plt.xlabel('Sentiment', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """
        Plot confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Negative', 'Neutral', 'Positive'],
                   yticklabels=['Negative', 'Neutral', 'Positive'])
        
        plt.title('Confusion Matrix', fontsize=14, pad=20)
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_sentiment_trends(self, df, date_column='created_at', sentiment_column='sentiment'):
        """
        Plot sentiment trends over time
        """
        # Convert date column to datetime
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Group by date and calculate average sentiment
        daily_sentiment = df.groupby(df[date_column].dt.date)[sentiment_column].mean()
        
        plt.figure(figsize=(12, 6))
        plt.plot(daily_sentiment.index, daily_sentiment.values, marker='o')
        
        plt.title('Sentiment Trends Over Time', fontsize=14, pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Average Sentiment', fontsize=12)
        plt.grid(True)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_word_cloud(self, df, text_column='processed_text', sentiment=None):
        """
        Plot word cloud for positive or negative sentiments
        """
        from wordcloud import WordCloud
        
        # Filter data by sentiment if specified
        if sentiment is not None:
            df = df[df['sentiment'] == sentiment]
        
        # Combine all text
        text = ' '.join(df[text_column])
        
        # Create word cloud
        wordcloud = WordCloud(width=800, height=400,
                            background_color='white',
                            max_words=100).generate(text)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        
        sentiment_label = 'Positive' if sentiment == 1 else 'Negative' if sentiment == -1 else 'All'
        plt.title(f'Word Cloud for {sentiment_label} Sentiments', fontsize=14, pad=20)
        
        plt.tight_layout()
        return plt.gcf()

if __name__ == "__main__":
    # Example usage
    visualizer = SentimentVisualizer()
    
    # Example data
    data = {
        'sentiment': [1, -1, 0, 1, -1, 0, 1, -1],
        'processed_text': [
            'great product love it',
            'terrible service disappointed',
            'okay experience nothing special',
            'amazing quality excellent',
            'poor performance bad',
            'average service normal',
            'fantastic product perfect',
            'awful experience terrible'
        ],
        'created_at': pd.date_range(start='2024-01-01', periods=8, freq='D')
    }
    df = pd.DataFrame(data)
    
    # Create visualizations
    fig1 = visualizer.plot_sentiment_distribution(df)
    fig2 = visualizer.plot_confusion_matrix(df['sentiment'], df['sentiment'])
    fig3 = visualizer.plot_sentiment_trends(df)
    fig4 = visualizer.plot_word_cloud(df)
    
    # Save plots
    fig1.savefig('plots/sentiment_distribution.png')
    fig2.savefig('plots/confusion_matrix.png')
    fig3.savefig('plots/sentiment_trends.png')
    fig4.savefig('plots/word_cloud.png') 