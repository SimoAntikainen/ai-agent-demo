import ast
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from pymilvus import CollectionSchema, FieldSchema, DataType, MilvusClient
from tqdm import tqdm
import polars as pl

# Load environment variables and OpenAI client
load_dotenv()
openai_client = OpenAI()

THIS_DIR = Path(__file__).parent
DATA_DIR = THIS_DIR / "book_reviews"

# Milvus/Zilliz config
ZILLIZ_CLUSTER_ENDPOINT = os.getenv("ZILLIZ_CLUSTER_ENDPOINT")
ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN")
COLLECTION_NAME = "book_reviews"

# Embed a sample to determine embedding dimension
def emb_text(text):
    return (
        openai_client.embeddings.create(input=text, model="text-embedding-3-small")
        .data[0]
        .embedding
    )

test_embedding = emb_text("test vector for getting embedding dim")
embedding_dim = len(test_embedding)
print(f"embedding dim {embedding_dim}")
print(test_embedding[:10])

VECTOR_DIM = embedding_dim

# Connect to Zilliz/Milvus
client = MilvusClient(uri=ZILLIZ_CLUSTER_ENDPOINT, token=ZILLIZ_TOKEN)

# Define collection schema
fields = [
    FieldSchema(
        name="id",
        dtype=DataType.VARCHAR,
        is_primary=True,
        auto_id=False,
        max_length=200,
    ),
    FieldSchema(name="productId", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="authors", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_length=100, max_capacity=10),
    FieldSchema(name="category", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_length=100, max_capacity=10),
    FieldSchema(name="user", dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name="score", dtype=DataType.FLOAT),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=10000),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
]

schema = CollectionSchema(fields, description="Embedded book reviews")

# Index config
index_params = MilvusClient.prepare_index_params()
index_params.add_index(
    field_name="vector",
    index_type="AUTOINDEX",
    metric_type="IP",
)

# Re-create collection if exists
if client.has_collection(COLLECTION_NAME):
    client.drop_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    dimension=VECTOR_DIM,
    metric_type="IP",
    consistency_level="Eventually",
    schema=schema,
    index_params=index_params,
)

print(f"Collection '{COLLECTION_NAME}' created.")

# Load reviews, they should be preprocess by reviews_preprocess.py
schema_reviews = {
    "productId": pl.Utf8,
    "Title": pl.Utf8,
    "Price": pl.Utf8,
    "User_id": pl.Utf8,
    "profileName": pl.Utf8,
    "review/helpfulness": pl.Utf8,
    "review/score": pl.Float64,
    "review/time": pl.Int64,
    "review/summary": pl.Utf8,
    "review/text": pl.Utf8,
    "authors": pl.Utf8,
    "category": pl.Utf8,
}
reviews_df = pl.read_csv(DATA_DIR / "balanced_reviews.csv", schema_overrides=schema_reviews)


reviews_df = reviews_df
# Embed and insert in batches
batch = []
batch_size = 10

for i, row in enumerate(tqdm(reviews_df.iter_rows(named=True), desc="Embedding + Inserting")):
    try:
        # --- Safely parse authors ---
        raw_authors = row.get("authors") or "[]"
        try:
            author_list = ast.literal_eval(raw_authors)
            authors = ", ".join(author_list) if isinstance(author_list, list) else str(author_list)
        except Exception:
            authors = str(raw_authors)

        # --- Safely parse categories ---
        raw_categories = row.get("category") or "[]"
        try:
            category_list = ast.literal_eval(raw_categories)
            categories = ", ".join(category_list) if isinstance(category_list, list) else str(category_list)
        except Exception:
            categories = str(raw_categories)

        # --- Prepare combined text for embedding ---
        embedding_input = f"""
        Title: {row['Title']}
        Rating: {row["review/score"]} stars
        Authors: {authors}
        Categories: {categories}
        Review: {row['review/text']}
        """

        vector = emb_text(embedding_input.strip())

        # --- Add to Milvus batch ---
        batch.append({
            "id": f"{str(row['productId'])}_{i}",
            "productId": str(row["productId"]),
            "title": str(row["Title"]),
            "authors": author_list,
            "category": category_list,
            "user": str(row["profileName"]),
            "score": row["review/score"],
            "text": str(row["review/text"])[:10000],
            "vector": vector
        })

    except Exception as e:
        print(f"⚠️ Skipping row {i} due to error: {e}")
        continue

    if len(batch) >= batch_size or i == len(reviews_df) - 1:
        client.insert(collection_name=COLLECTION_NAME, data=batch)
        batch = []
# ensure the collection is loaded before calling it
client.load_collection(COLLECTION_NAME)
print("Upload and collection load complete.")