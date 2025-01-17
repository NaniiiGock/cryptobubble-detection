import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from glob import glob
from collections import defaultdict

class TemporalBubbleDetector:
    def __init__(self, price_data_path: str, tweets_path: str, model_name: str = "all-MiniLM-L6-v2",
                 sequence_window: int = 3):
        self.price_data_path = price_data_path
        self.tweets_path = tweets_path
        self.model = SentenceTransformer(model_name)
        self.sequence_window = sequence_window
        self.cached_embeddings = {}
        self.historical_data = self._load_all_historical_data()
        self.temporal_patterns = self._create_temporal_patterns()
        
    def _load_all_historical_data(self) -> Dict:
        historical_data = {}
        
        for file_path in glob(os.path.join(self.price_data_path, "*.csv")):
            symbol = os.path.basename(file_path).replace('.csv', '')
            df = pd.read_csv(file_path)
            df['datetime'] = pd.to_datetime(df['datetime'])
            historical_data[symbol] = df
            
        return historical_data
    
    def _create_temporal_patterns(self) -> Dict:
        patterns = defaultdict(list)
        
        for symbol, df in self.historical_data.items():
        
            for i in range(len(df) - self.sequence_window + 1):
                window = df.iloc[i:i + self.sequence_window]
                pattern_key = f"{symbol}_{window.iloc[0]['datetime'].strftime('%Y-%m-%d')}"
                patterns[pattern_key] = {
                    'dates': window['datetime'].dt.strftime('%Y-%m-%d').tolist(),
                    'labels': window['label'].tolist(),
                    'symbol': symbol
                }
                
        return patterns
    
    def _process_tweets(self, tweets: List[str]) -> np.ndarray:
        if not tweets:
            return None
            
        processed_tweets = []
        for tweet in tweets:
            if isinstance(tweet, str):
                tweet = ' '.join(word for word in tweet.split() if not word.startswith('http'))
                tweet = tweet.strip()
                if tweet:
                    processed_tweets.append(tweet)
                    
        if not processed_tweets:
            return None
            
        combined_text = " ".join(processed_tweets)
        return self.model.encode([combined_text], convert_to_tensor=True)
    
    def _get_historical_tweets(self, symbol: str, date: str) -> List[str]:
        try:
            file_path = os.path.join(self.tweets_path, symbol, f"{date}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, encoding='utf-8')
                return df[df['tweet'].notna()]['tweet'].tolist()
        except Exception as e:
            print(f"Error loading tweets for {symbol} on {date}: {str(e)}")
        return []
    
    def _calculate_sequence_similarity(self, 
                                    query_sequence: Dict[str, List[str]], 
                                    pattern: Dict) -> Tuple[float, List[float]]:
        query_dates = sorted(query_sequence.keys())
        pattern_dates = pattern['dates']
        
        if len(query_dates) != len(pattern_dates):
            return 0.0, [0.0] * len(query_dates)
            
        daily_similarities = []
        
        for query_date, pattern_date in zip(query_dates, pattern_dates):
            query_embedding = self._process_tweets(query_sequence[query_date])
            
            cache_key = f"{pattern['symbol']}_{pattern_date}"
            if cache_key not in self.cached_embeddings:
                historical_tweets = self._get_historical_tweets(pattern['symbol'], pattern_date)
                historical_embedding = self._process_tweets(historical_tweets)
                if historical_embedding is not None:
                    self.cached_embeddings[cache_key] = historical_embedding
                else:
                    daily_similarities.append(0.0)
                    continue
                    
            historical_embedding = self.cached_embeddings[cache_key]
            
            if query_embedding is not None and historical_embedding is not None:
                similarity = cosine_similarity(
                    query_embedding.cpu().numpy(),
                    historical_embedding.cpu().numpy()
                )[0][0]
                daily_similarities.append(float(similarity))
            else:
                daily_similarities.append(0.0)
                
        sequence_similarity = np.mean(daily_similarities) if daily_similarities else 0.0
        return sequence_similarity, daily_similarities
    
    def predict_bubble_sequence(self, 
                              tweet_sequence: Dict[str, List[str]], 
                              top_k: int = 5) -> Dict:

        if len(tweet_sequence) != self.sequence_window:
            return {
                'error': f'Sequence must contain exactly {self.sequence_window} days of data'
            }
            
        similar_patterns = []
        
        for pattern_key, pattern in self.temporal_patterns.items():
            sequence_similarity, daily_similarities = self._calculate_sequence_similarity(
                tweet_sequence, pattern
            )
            
            if sequence_similarity > 0:
                similar_patterns.append({
                    'pattern_key': pattern_key,
                    'symbol': pattern['symbol'],
                    'dates': pattern['dates'],
                    'labels': pattern['labels'],
                    'sequence_similarity': sequence_similarity,
                    'daily_similarities': daily_similarities
                })
                
        similar_patterns.sort(key=lambda x: x['sequence_similarity'], reverse=True)
        similar_patterns = similar_patterns[:top_k]
        
        if not similar_patterns:
            return {
                'error': 'No similar patterns found',
                'predictions': []
            }
            
        query_dates = sorted(tweet_sequence.keys())
        daily_predictions = []
        
        for day_idx in range(len(query_dates)):
            day_weights = []
            day_labels = []
            
            for pattern in similar_patterns:
                weight = pattern['daily_similarities'][day_idx]
                label = pattern['labels'][day_idx]
                day_weights.append(weight)
                day_labels.append(label)
                
            if sum(day_weights) > 0:
                bubble_probability = np.average(day_labels, weights=day_weights)
            else:
                bubble_probability = 0.0
                
            daily_predictions.append({
                'date': query_dates[day_idx],
                'bubble_probability': float(bubble_probability),
                'is_bubble': bubble_probability > 0.5
            })
            
        return {
            'daily_predictions': daily_predictions,
            'similar_patterns': [
                {
                    'symbol': p['symbol'],
                    'dates': p['dates'],
                    'labels': p['labels'],
                    'similarity': float(p['sequence_similarity']),
                    'daily_similarities': [float(s) for s in p['daily_similarities']]
                }
                for p in similar_patterns
            ],
            'overall_confidence': float(similar_patterns[0]['sequence_similarity']) if similar_patterns else 0.0
        }


detector = TemporalBubbleDetector(
    price_data_path="small_data/price_data",
    tweets_path="small_data/tweets",
    sequence_window=3  
)

query_sequence = {

}

result = detector.predict_bubble_sequence(query_sequence)
