import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from glob import glob
from pymilvus import CollectionSchema, FieldSchema, DataType, MilvusClient
from tqdm import tqdm

load_dotenv()
openai_client = OpenAI()

THIS_DIR = Path(__file__).parent

ZILLIZ_CLUSTER_ENDPOINT = os.getenv("ZILLIZ_CLUSTER_ENDPOINT")
ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN")
COLLECTION_NAME = "ai_agent_rag"


def emb_text(text):
    return (
        openai_client.embeddings.create(input=text, model="text-embedding-3-small")
        .data[0]
        .embedding
    )

test_embedding = emb_text("test vector for getting embedding dim")  # 1536
embedding_dim = len(test_embedding)
print(f"embedding dim {embedding_dim}")
print(test_embedding[:10])

VECTOR_DIM = embedding_dim

client = MilvusClient(uri=ZILLIZ_CLUSTER_ENDPOINT, token=ZILLIZ_TOKEN)

fields = [
    FieldSchema(
        name="id",
        dtype=DataType.VARCHAR,
        is_primary=True,
        auto_id=False,
        max_length=200,
    ),
    FieldSchema(name="doc_name", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="chunk_id", dtype=DataType.INT64),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=5000),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
]

schema = CollectionSchema(fields, description="FAQ document chunks")



index_params = MilvusClient.prepare_index_params()
# AUTOINDEX is used only in zilliz cloud, change the index_type in regular milvus deployments
index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type="IP",
        )

# Check if collection already exists, if so drop it.
has = client.has_collection(COLLECTION_NAME)
if has:
    drop_result = client.drop_collection(COLLECTION_NAME)

client.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=VECTOR_DIM,
        metric_type="IP",
        consistency_level="Eventually",
        schema=schema,
        index_params=index_params
    )

print(f"Collection '{COLLECTION_NAME}' created.")


text_chunks = []

for file_path in glob("milvus_docs/en/faq/*.md", root_dir=THIS_DIR, recursive=True):
    full_path = Path(THIS_DIR) / file_path
    with open(full_path, "r") as file:
        file_text = file.read()

    chunks = file_text.split("# ")

    for i, chunk in enumerate(chunks):
        if chunk.strip():  # skip empty
            text_chunks.append(
                {
                    "id": f"{file_path}_{i}",
                    "doc_name": str(file_path),
                    "chunk_id": i,
                    "text": chunk,
                }
            )

batch = []
batch_size = 10

for i, chunk in enumerate(tqdm(text_chunks, desc="Embedding + Inserting")):
    vector = emb_text(chunk["text"])
    chunk["vector"] = vector
    batch.append(chunk)

    if len(batch) >= batch_size or i == len(text_chunks) - 1:
        client.insert(collection_name=COLLECTION_NAME, data=batch)
        batch = []  # reset batch)


# ensure the collection is loaded before calling it
client.load_collection(COLLECTION_NAME)

