import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime
import os
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity


# analysis for 1 day only


class CryptoRAGPipeline:
    def __init__(self, price_data_path: str, tweets_path: str, model_name: str = "all-MiniLM-L6-v2"):

        self.price_data_path = price_data_path
        self.tweets_path = tweets_path
        self.model = SentenceTransformer(model_name)
        self.price_data = {}
        self.tweet_embeddings = {}
        self.tweet_data = {}
        
    def load_price_data(self, symbol: str) -> pd.DataFrame:
        file_path = os.path.join(self.price_data_path, f"{symbol}.csv")
        df = pd.read_csv(file_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df
    
    def load_tweets(self, symbol: str, date: str) -> pd.DataFrame:
        file_path = os.path.join(self.tweets_path, symbol, f"{date}.csv")
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        return pd.DataFrame()
    
    def process_day_data(self, symbol: str, date: datetime) -> Dict:
        date_str = date.strftime('%Y-%m-%d')

        price_df = self.load_price_data(symbol)
        day_price = price_df[price_df['datetime'].dt.date == date.date()].iloc[0]
 
        tweets_df = self.load_tweets(symbol, date_str)
        tweets = tweets_df['tweet'].tolist() if not tweets_df.empty else []
        
        return {
            'price_data': day_price,
            'tweets': tweets
        }
    
    def create_embeddings(self, texts):
        return self.model.encode(texts, convert_to_tensor=True)
    
    def get_context(self, query, symbol, date, k=5):
        day_data = self.process_day_data(symbol, date)
    
        if day_data['tweets']:
            query_embedding = self.create_embeddings([query])
            
            tweet_key = f"{symbol}_{date.strftime('%Y-%m-%d')}"
            if tweet_key not in self.tweet_embeddings:
                self.tweet_embeddings[tweet_key] = self.create_embeddings(day_data['tweets'])

            similarities = cosine_similarity(
                query_embedding.cpu().numpy(),
                self.tweet_embeddings[tweet_key].cpu().numpy()
            )[0]
 
            top_k_indices = np.argsort(similarities)[-k:][::-1]
            relevant_tweets = [day_data['tweets'][i] for i in top_k_indices]
        else:
            relevant_tweets = []
        
        return day_data['price_data'], relevant_tweets

    def generate_response(self, query: str, context_price: Dict, context_tweets: List[str]) -> str:
 
        # For now, return a simple formatted response
        response = f"Price Information:\n"
        response += f"Open: {context_price['open_x']}, Close: {context_price['close_x']}\n"
        response += f"High: {context_price['high_x']}, Low: {context_price['low_x']}\n\n"
        
        if context_tweets:
            response += "Relevant Social Context:\n"
            for i, tweet in enumerate(context_tweets, 1):
                response += f"{i}. {tweet}\n"
        
        return response



rag_pipeline = CryptoRAGPipeline(
    price_data_path="small_data/price_data",
    tweets_path="small_data/tweets"
)

query = "What was the market sentiment and price movement?"
symbol = "ABT"
date = datetime(2018, 3, 2)

price_context, tweet_context = rag_pipeline.get_context(query, symbol, date)

response = rag_pipeline.generate_response(query, price_context, tweet_context)
print(response)