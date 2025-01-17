import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from glob import glob

class CryptoBubbleDetector:
    def __init__(self, price_data_path: str, tweets_path: str, model_name: str = "all-MiniLM-L6-v2"):

        self.price_data_path = price_data_path
        self.tweets_path = tweets_path
        self.model = SentenceTransformer(model_name)
        self.cached_embeddings = {}
        self.historical_data = self._load_all_historical_data()
        
    def _load_all_historical_data(self) -> Dict:
        historical_data = {}
        crypto_files = glob(os.path.join(self.price_data_path, "*.csv"))
        
        for file_path in crypto_files:
            symbol = os.path.basename(file_path).replace('.csv', '')
            df = pd.read_csv(file_path)
            df['datetime'] = pd.to_datetime(df['datetime'])
            historical_data[symbol] = df
            
        return historical_data
    
    def _load_day_tweets(self, symbol: str, date_str: str) -> List[str]:
        tweet_path = os.path.join(self.tweets_path, symbol, f"{date_str}.csv")
        if os.path.exists(tweet_path):
            try:
                df = pd.read_csv(tweet_path, encoding='utf-8')
                tweets = df[df['tweet'].notna()]['tweet'].tolist()
                return tweets
            except Exception as e:
                print(f"Error loading tweets for {symbol} on {date_str}: {str(e)}")
                return []
        return []
    
    def _preprocess_tweets(self, tweets: List[str]) -> List[str]:
        processed = []
        for tweet in tweets:
            if isinstance(tweet, str):
                # Remove URLs
                tweet = ' '.join(word for word in tweet.split() if not word.startswith('http'))
                # Basic cleaning
                tweet = tweet.strip()
                if tweet:  # Only add non-empty tweets
                    processed.append(tweet)
        return processed
    
    def _create_tweet_embedding(self, tweets: List[str]) -> np.ndarray:
        if not tweets:
            return None
        processed_tweets = self._preprocess_tweets(tweets)
        if not processed_tweets:
            return None
        combined_text = " ".join(processed_tweets)
        return self.model.encode([combined_text], convert_to_tensor=True)
    
    def _find_similar_historical_periods(self, query_tweets, top_k=5):
        query_embedding = self._create_tweet_embedding(query_tweets)
        if query_embedding is None:
            return []
            
        similar_periods = []
    
        for symbol, price_df in self.historical_data.items():
            for _, row in price_df.iterrows():
                date_str = row['datetime'].strftime('%Y-%m-%d')
                historical_tweets = self._load_day_tweets(symbol, date_str)
                
                if historical_tweets:
                    cache_key = f"{symbol}_{date_str}"
                
                    if cache_key not in self.cached_embeddings:
                        historical_embedding = self._create_tweet_embedding(historical_tweets)
                        if historical_embedding is not None:
                            self.cached_embeddings[cache_key] = historical_embedding
                        else:
                            continue
                    
                    historical_embedding = self.cached_embeddings[cache_key]
                    
                    similarity = cosine_similarity(
                        query_embedding.cpu().numpy(),
                        historical_embedding.cpu().numpy()
                    )[0][0]
                    
                    similar_periods.append((symbol, date_str, similarity, row['label']))
    
        similar_periods.sort(key=lambda x: x[2], reverse=True)
        return similar_periods[:top_k]
    
    def predict_bubble(self, query_tweets: List[str]) -> Dict:

        similar_periods = self._find_similar_historical_periods(query_tweets)
        
        if not similar_periods:
            return {
                'error': 'No similar historical periods found or invalid input tweets',
                'bubble_probability': 0.0,
                'prediction': 'Unable to make prediction',
                'confidence': 0.0,
                'similar_periods': []
            }
        
        total_weight = 0
        weighted_sum = 0
        
        for _, _, similarity, label in similar_periods:
            weighted_sum += similarity * label
            total_weight += similarity
        
        bubble_probability = weighted_sum / total_weight if total_weight > 0 else 0
        
        response = {
            'bubble_probability': float(bubble_probability),
            'prediction': 'Potential bubble detected' if bubble_probability > 0.5 else 'No bubble detected',
            'confidence': float(max([s[2] for s in similar_periods]) if similar_periods else 0),
            'similar_periods': [
                {
                    'symbol': s[0],
                    'date': s[1],
                    'similarity': float(s[2]),
                    'was_bubble': bool(s[3])
                }
                for s in similar_periods
            ]
        }
        
        return response


# detector = CryptoBubbleDetector(
#     price_data_path="small_data/price_data",
#     tweets_path="small_data/tweets"
# )

# query_tweets = [
#     "Bitcoin price is skyrocketing! To the moon! 🚀",
#     "Everyone's talking about crypto nowadays, even my grandmother wants to invest",
#     "The growth is unprecedented, but looks sustainable this time"
# ]

# result = detector.predict_bubble(query_tweets)
