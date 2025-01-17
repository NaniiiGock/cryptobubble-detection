
import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "12345678"))

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def vectorize_texts(texts):
    tokenized = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**tokenized).last_hidden_state.mean(dim=1).numpy()
    return embeddings

def load_price_data(folder):
    price_data = {}
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            crypto_name = file.split(".")[0]
            price_data[crypto_name] = pd.read_csv(os.path.join(folder, file))
    return price_data

def prepare_classification_data(price_data):
    data = []
    labels = []
    for _, df in price_data.items():
        df = df.dropna()
        for i in range(len(df) - 10): 
            window = df.iloc[i:i+10]
            embedding = vectorize_texts([str(window["close_x"].values)])
            data.append(embedding.flatten())
            labels.append(window["label"].iloc[-1])
    return np.array(data), np.array(labels)

def train_classifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(classification_report(y_test, y_pred))
    return classifier

def store_crypto_data_with_full_embeddings(driver, price_data):
    with driver.session() as session:
        for crypto, df in price_data.items():
            df = df.dropna()
            time_series_embedding = vectorize_texts([str(df["close_x"].values)])
            metadata_embedding = vectorize_texts([f"avg_close:{df['close_x'].mean()},avg_volume:{df['volumefrom_x'].mean()}"])
            combined_embedding = np.hstack((time_series_embedding.flatten(), metadata_embedding.flatten()))

            session.run(
                """
                CREATE (c:Crypto {
                    name: $name, 
                    embedding: $embedding,
                    avg_close: $avg_close, 
                    avg_volume: $avg_volume
                })
                """,
                {
                    "name": crypto,
                    "embedding": combined_embedding.tolist(),
                    "avg_close": df["close_x"].mean(),
                    "avg_volume": df["volumefrom_x"].mean()
                }
            )


def create_similarity_edges(driver, embeddings, crypto_names, threshold=0.7):
    similarity_matrix = cosine_similarity(embeddings)
    with driver.session() as session:
        for i, name1 in enumerate(crypto_names):
            for j, name2 in enumerate(crypto_names):
                if i != j and similarity_matrix[i][j] > threshold:
                    session.run(
                        """
                        MATCH (a:Crypto {name: $name1}), (b:Crypto {name: $name2})
                        CREATE (a)-[:CORRELATED {weight: $weight}]->(b)
                        """,
                        {"name1": name1, "name2": name2, "weight": similarity_matrix[i][j]}
                    )

if __name__ == "__main__":
    price_data_folder = "data/price_data"

    price_data = load_price_data(price_data_folder)
    store_crypto_data_with_full_embeddings(driver, price_data)
    X, y = prepare_classification_data(price_data)

    crypto_names = list(price_data.keys())
    crypto_embeddings = []
    for crypto in crypto_names:
        df = price_data[crypto].dropna()
        time_series_embedding = vectorize_texts([str(df["close_x"].values)])
        metadata_embedding = vectorize_texts([f"avg_close:{df['close_x'].mean()},avg_volume:{df['volumefrom_x'].mean()}"])
        combined_embedding = np.hstack((time_series_embedding.flatten(), metadata_embedding.flatten()))
        crypto_embeddings.append(combined_embedding)
    crypto_embeddings = np.array(crypto_embeddings)

    create_similarity_edges(driver, crypto_embeddings, crypto_names, threshold=0.7)

    classifier = train_classifier(X, y)

    driver.close()

