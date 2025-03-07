import tweepy
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TwitterDataCollector:
    def __init__(self):
        # Twitter API credentials
        self.api_key = os.getenv('TWITTER_API_KEY')
        self.api_secret = os.getenv('TWITTER_API_SECRET')
        self.access_token = os.getenv('TWITTER_ACCESS_TOKEN')
        self.access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        
        # Initialize Tweepy client
        auth = tweepy.OAuthHandler(self.api_key, self.api_secret)
        auth.set_access_token(self.access_token, self.access_token_secret)
        self.api = tweepy.API(auth)
        
    def collect_tweets(self, query, count=1000):
        """
        Collect tweets based on search query
        """
        tweets = []
        try:
            # Search tweets
            for tweet in tweepy.Cursor(
                self.api.search_tweets,
                q=query,
                lang="en",
                tweet_mode="extended"
            ).items(count):
                
                tweets.append({
                    'id': tweet.id,
                    'text': tweet.full_text,
                    'created_at': tweet.created_at,
                    'user': tweet.user.screen_name,
                    'followers_count': tweet.user.followers_count,
                    'retweet_count': tweet.retweet_count,
                    'favorite_count': tweet.favorite_count
                })
                
            # Convert to DataFrame
            df = pd.DataFrame(tweets)
            
            # Save to CSV
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'data/tweets_{timestamp}.csv'
            df.to_csv(filename, index=False)
            
            print(f"Successfully collected {len(tweets)} tweets")
            return df
            
        except Exception as e:
            print(f"Error collecting tweets: {str(e)}")
            return None

if __name__ == "__main__":
    collector = TwitterDataCollector()
    # Example usage
    tweets_df = collector.collect_tweets("python programming", count=100) 