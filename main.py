import os
from dotenv import load_dotenv
from src.data_collection import TwitterDataCollector
from src.preprocessing import TweetPreprocessor
from src.model import SentimentAnalyzer
from src.visualization import SentimentVisualizer

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize components
    collector = TwitterDataCollector()
    preprocessor = TweetPreprocessor()
    analyzer = SentimentAnalyzer()
    visualizer = SentimentVisualizer()
    
    # Step 1: Collect tweets
    print("🔍 Collecting tweets...")
    query = "python programming"  # Example query
    tweets_df = collector.collect_tweets(query, count=100)
    
    if tweets_df is None:
        print("❌ Failed to collect tweets. Please check your API credentials.")
        return
    
    # Step 2: Preprocess tweets
    print("🧹 Preprocessing tweets...")
    tweets_df = preprocessor.preprocess_dataframe(tweets_df)
    
    # Step 3: Train model
    print("🤖 Training sentiment analysis model...")
    X, y = analyzer.prepare_data(tweets_df)
    X_test, y_test, y_pred = analyzer.train(X, y)
    
    # Step 4: Create visualizations
    print("📊 Creating visualizations...")
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Generate and save visualizations
    fig1 = visualizer.plot_sentiment_distribution(tweets_df)
    fig2 = visualizer.plot_confusion_matrix(y_test, y_pred)
    fig3 = visualizer.plot_sentiment_trends(tweets_df)
    fig4 = visualizer.plot_word_cloud(tweets_df)
    
    # Save plots
    fig1.savefig('plots/sentiment_distribution.png')
    fig2.savefig('plots/confusion_matrix.png')
    fig3.savefig('plots/sentiment_trends.png')
    fig4.savefig('plots/word_cloud.png')
    
    # Step 5: Save model
    print("💾 Saving model...")
    analyzer.save_model()
    
    print("✅ Project execution completed successfully!")
    print("\n📈 Generated visualizations are saved in the 'plots' directory")
    print("🤖 Trained model is saved in the 'models' directory")
    print("📊 Collected data is saved in the 'data' directory")

if __name__ == "__main__":
    main() 