from elasticsearch import Elasticsearch
import pandas as pd
from tqdm import tqdm

# Connect to Elasticsearch
es = Elasticsearch("http://localhost:9200")

df = pd.read_csv('../meingym_sport_20250530_prepared.csv')
df = df[df['wordCount'] > 20]

for idx, row in tqdm(df.iterrows()):
    doc = {
        "title": row['title'],
        "content": row['content'],
        "category": row['category'],
    }
    es.index(index="docs", id=idx, body=doc)