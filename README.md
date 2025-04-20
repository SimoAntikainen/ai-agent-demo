# ai-agent-demo
 

## Requirements

```
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
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


## Running the app

```
# populate the sqlite database
python app/populate_db.py

# create zilliz cloud collection for rag
python app/create_collection.py
```


Then Run the agent chat with
```
uvicorn app.chat_app_backend:app --reload
```







# Misc

Info on techology choices:
* https://datasystemreviews.com/best-open-source-vector-databases.html

