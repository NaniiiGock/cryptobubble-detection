import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime
import os
from typing import List, Dict, Tuple
from neo4j import GraphDatabase
from glob import glob
import json

class CryptoEmbeddingStorage:
    def __init__(self, uri: str, user: str, password: str, 
                 price_data_path: str, tweets_path: str,
                 model_name: str = "all-MiniLM-L6-v2"):

        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.model = SentenceTransformer(model_name)
        self.price_data_path = price_data_path
        self.tweets_path = tweets_path

    def close(self):
        self.driver.close()

    def _init_database(self):
        with self.driver.session() as session:
            # Create constraints
            session.run("""
                CREATE CONSTRAINT crypto_symbol IF NOT EXISTS
                FOR (c:Cryptocurrency) REQUIRE c.symbol IS UNIQUE
            """)
            
            session.run("""
                CREATE CONSTRAINT day_id IF NOT EXISTS
                FOR (d:Day) REQUIRE d.id IS UNIQUE
            """)
            
            session.run("""
                CREATE INDEX tweet_date IF NOT EXISTS
                FOR (d:Day) ON (d.date)
            """)

    def _create_embedding(self, tweets: List[str]) -> np.ndarray:
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
        embedding = self.model.encode([combined_text])[0]
        return embedding.tolist() 

    def store_data(self):
        self._init_database()
        
        for file_path in glob(os.path.join(self.price_data_path, "*.csv")):
            symbol = os.path.basename(file_path).replace('.csv', '')
            price_df = pd.read_csv(file_path)
            price_df['datetime'] = pd.to_datetime(price_df['datetime'])
            
            with self.driver.session() as session:
                session.run("""
                    MERGE (c:Cryptocurrency {symbol: $symbol})
                """, symbol=symbol)
                
                for _, row in price_df.iterrows():
                    date_str = row['datetime'].strftime('%Y-%m-%d')
                    
                    tweets = self._load_day_tweets(symbol, date_str)
                    embedding = self._create_embedding(tweets)
                    
                    if embedding:
                        session.run("""
                            MATCH (c:Cryptocurrency {symbol: $symbol})
                            MERGE (d:Day {id: $day_id})
                            SET d.date = date($date)
                            SET d.embedding = $embedding
                            SET d.label = $label
                            SET d.price_open = $open
                            SET d.price_close = $close
                            SET d.price_high = $high
                            SET d.price_low = $low
                            SET d.tweets = $tweets
                            MERGE (c)-[:HAS_DAY]->(d)
                        """, {
                            'symbol': symbol,
                            'day_id': f"{symbol}_{date_str}",
                            'date': date_str,
                            'embedding': embedding,
                            'label': float(row['label']),
                            'open': float(row['open_x']),
                            'close': float(row['close_x']),
                            'high': float(row['high_x']),
                            'low': float(row['low_x']),
                            'tweets': json.dumps(tweets) 
                        })

    def _load_day_tweets(self, symbol: str, date_str: str) -> List[str]:
        try:
            file_path = os.path.join(self.tweets_path, symbol, f"{date_str}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, encoding='utf-8')
                return df[df['tweet'].notna()]['tweet'].tolist()
        except Exception as e:
            print(f"Error loading tweets for {symbol} on {date_str}: {str(e)}")
        return []

    def find_similar_days(self, query_tweets: List[str], top_k: int = 5) -> List[Dict]:
        query_embedding = self._create_embedding(query_tweets)
        if not query_embedding:
            return []
            
        with self.driver.session() as session:
            result = session.run("""
                MATCH (d:Day)
                WITH d, gds.similarity.cosine($query_embedding, d.embedding) AS similarity
                WHERE similarity > 0.5
                RETURN d.id AS day_id, d.date AS date, d.label AS label,
                       d.tweets AS tweets, similarity
                ORDER BY similarity DESC
                LIMIT $top_k
            """, {
                'query_embedding': query_embedding,
                'top_k': top_k
            })
            
            return [dict(record) for record in result]

    def get_temporal_pattern(self, start_date: str, window_size: int = 3) -> List[Dict]:

        with self.driver.session() as session:
            result = session.run("""
                MATCH (d:Day)
                WHERE d.date >= date($start_date) AND 
                      d.date < date($start_date) + duration({days: $window_size})
                RETURN d.id AS day_id, d.date AS date, d.label AS label,
                       d.tweets AS tweets, d.embedding AS embedding
                ORDER BY d.date
            """, {
                'start_date': start_date,
                'window_size': window_size
            })
            
            return [dict(record) for record in result]


storage = CryptoEmbeddingStorage(
    uri="neo4j://localhost:7687",
    user="neo4j",
    password="12345678",
    price_data_path="small_data/price_data",
    tweets_path="small_data/tweets"
)

storage.store_data()
