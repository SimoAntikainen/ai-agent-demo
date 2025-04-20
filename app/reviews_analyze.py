import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
from pymilvus import MilvusClient

# Load env
load_dotenv()
openai_client = OpenAI()

ZILLIZ_CLUSTER_ENDPOINT = os.getenv("ZILLIZ_CLUSTER_ENDPOINT")
ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN")
COLLECTION_NAME = "balanced_book_reviews"

client = MilvusClient(uri=ZILLIZ_CLUSTER_ENDPOINT, token=ZILLIZ_TOKEN)

def get_hypothetical_answer(question: str) -> str:
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a literary reviewer. Provide a thoughtful, hypothetical book review answering the user's question."},
            {"role": "user", "content": question}
        ],
        temperature=0.7,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()

def embed_text(text):
    return openai_client.embeddings.create(
        input=text, model="text-embedding-3-small"
    ).data[0].embedding

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_reviews.py '<your question here>'")
        sys.exit(1)

    query_text = sys.argv[1]

    hypothetical_answer = get_hypothetical_answer(query_text)
    print(f"\n💡 Hypothetical Answer:\n{hypothetical_answer}\n")

    query_vector = embed_text(hypothetical_answer)

    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        limit=5,
        output_fields=["title", "user", "score", "text"],
    )

    print(f"\n🔍 Results for: '{query_text}'\n")
    for res in results[0]:
        hit = res["entity"]
        print(f"distance: {res['distance']}")
        print(f"📚 Title: {hit['title']}")
        print(f"👤 User: {hit['user']}")
        print(f"⭐ Rating: {hit['score']}")
        print(f"📝 Review: {hit['text']}\n")

if __name__ == "__main__":
    main()