# ai-agent-demo
 

## Requirements

```
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Running the app

```
uvicorn app.chat_app_backend:app --reload
```


## API KEYS

For the project to work you need the following API keys:
* Open AI API key https://platform.openai.com/api-keys
* Serper Dev API key for web search https://serper.dev/

Set the variables into an `.env` file in the `/app` directory
```bash
OPENAI_API_KEY=...
SERPER_API_KEY=...
```
