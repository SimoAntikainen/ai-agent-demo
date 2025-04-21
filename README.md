# ai-agent-demo
 

## Requirements

```
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt


polars
```


## API KEYS

For the project to work you need the following API keys:
* Open AI API key https://platform.openai.com/api-keys
* Serper Dev API key for web search https://serper.dev/
* Zilliz cloud account for RAG, create a cluster and get its acccess token https://zilliz.com/ 

Set the variables into an `.env` file in the `/app` directory

```bash
OPENAI_API_KEY=...
SERPER_API_KEY=...
ZILLIZ_CLUSTER_ENDPOINT=...
ZILLIZ_TOKEN=...
```

## Creating the ZILLIZ cluster of book reviews

Download the Amazon book reviews dataset from
https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews?resource=download 

The dataset is practically a smaller subset of a larget Amazon products dataset. You can access it at
https://amazon-reviews-2023.github.io/ 


Then follow the underlying steps
```
# unpack the dataset .csv files under /app/book_reviews
unzip archive.zip -d ./app/book_reviews

# select a smaller subset of top books for indexing (feel free to increase the number of reviews/books)
python reviews_preprocess.py

# create a zilliz vector store collection on the reviews
python reviews_create_collection.py

# ask a question on the dataset to check that everything is working correctly
python reviews_analyze.py "What do readers think about the ending of Of Mice and Men?"
```

After that create fake sales data to be queried by the sql agent

```
python reviews_fake_sql_sales.py
```

## Running the app

Then Run the agent with
```
uvicorn app.chat_app_backend:app --reload
```







# Misc

Info on techology choices:
* https://datasystemreviews.com/best-open-source-vector-databases.html




##

* Use reranker on the sql tables metadata https://jina.ai/reranker/
