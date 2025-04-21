"""Agentic AI Demo build with FastAPI.
Extended from https://ai.pydantic.dev/examples/chat-app/
"""

from __future__ import annotations as _annotations

import asyncio
import json
import os
import sqlite3
from collections.abc import AsyncIterator
from concurrent.futures.thread import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Callable,
    List,
    Literal,
    Optional,
    TypeVar,
    Union,
)

from annotated_types import MinLen
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import fastapi
from httpx import AsyncClient
import httpx
import logfire
from fastapi import Depends, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from openai import OpenAI
from pydantic import BaseModel
from pymilvus import MilvusClient
from typing_extensions import LiteralString, ParamSpec, TypedDict

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.usage import UsageLimits

import logging

load_dotenv()

THIS_DIR = Path(__file__).parent
DB_NAME = ".chat_app_backend.sqlite"
DB_PATH = THIS_DIR / DB_NAME
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

ZILLIZ_CLUSTER_ENDPOINT = os.getenv("ZILLIZ_CLUSTER_ENDPOINT")
ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN")
COLLECTION_NAME = "book_reviews" ##"balanced_book_reviews" #"ai_agent_rag"

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire="if-token-present")


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class Deps:
    client: AsyncClient
    openai_client: OpenAI
    milvus_client: MilvusClient
    serper_api_key: str | None


system_prompt=( 
    "You are a helpful assistant that uses tools when your internal knowledge is insufficient or uncertain.\n\n"

    "- Use the `web_search` tool to find up-to-date information or verify facts using Google-style search results.\n"
    "- Use the `search_the_internet` tool to fetch and read actual content from a specific website URL. It returns clean, readable text.\n"
    "- Use the `reviews_vector_database` tool to find and read real book reviews from readers. These reviews include opinions, summaries, and commentary.\n"
    "- Use the `call_sql_database_agent` tool to answer questions about book sales, pricing, and inventory from our SQL book sales database.\n\n"

    "**IMPORTANT: Do not change the capitalization or punctuation of book titles.**\n"
    "The `title` field returned by `reviews_vector_database` exactly matches the `title` field in the `Books` table of the book sales database.\n"
    "Use the title exactly as it appears, character by character — including all capital letters and special characters.\n"
    "Correct: What are the sales figures for 'OF MICE AND MEN'?\n"
    "Incorrect: What are the sales figures for 'Of Mice and Men'?\n"
    "Correct: What are the sales figures for 'Guns, Germs, and Steel: The Fates of Human Societies'?\n"
    "Incorrect: What are the sales figures for 'Guns, Germs, and Steel'?\n"
    "Preserve casing and spelling *exactly*.\n"
)

agent = Agent(
    "openai:gpt-4o",
    system_prompt=system_prompt,
    deps_type=Deps,
)


milvus_client = MilvusClient(uri=ZILLIZ_CLUSTER_ENDPOINT, token=ZILLIZ_TOKEN)
deps = Deps(
    client=httpx.AsyncClient(),
    openai_client=OpenAI(),
    milvus_client=milvus_client,
    serper_api_key=SERPER_API_KEY,
)

usage_limits = UsageLimits(request_limit=5)

# "Use the `database_schema_data` tool to fetch the current database schema. "
sql_database_agent = Agent(
    "openai:gpt-4o",
    system_prompt=(
        "You are a helpful book sales database agent. The database is SQLite3. "
        "Use the `database_schema_data` tool to fetch the current database schema. "
        "Do not call it multiple times for the same query unless the schema has changed. "
        "Use the `database_query` tool to query the book sales database."
        "ALWAYS generate a complete, executable SQL SELECT statement when constructing queries. "
        "Avoid partial queries. Ensure it starts with 'SELECT ... FROM ...'. "
        "Only use SELECT queries. Never use UPDATE, INSERT, DELETE, etc."
    ),
    usage=usage_limits,
    deps_type=Deps,
)

usage_limits_critic = UsageLimits(response_tokens_limit=300)
# literature_critic_agent is used for query rewriting  
literature_critic_agent = Agent(
    "openai:gpt-4o",
    system_prompt=(
        "You are a literary reviewer. Provide a thoughtful, hypothetical book review answering the user's question."
        "Keep it concise, under 300 tokens. "
    ),
    usage=usage_limits_critic,
    deps_type=Deps,
)



class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str


@agent.tool(retries=2)
async def web_search(ctx: RunContext[Deps], query: str) -> List[SearchResult]:
    """
    Perform a Google-style web search and return the top 3 results with title, URL, and snippet.

    Args:
        ctx: Context that includes httpx client and other deps.
        query: a Google-style web search query
    """

    API_KEY = ctx.deps.serper_api_key
    if not API_KEY:
        raise ModelRetry("Search API key not configured.")

    headers = {"X-API-KEY": API_KEY, "Content-Type": "application/json"}
    payload = {"q": query}

    response = await ctx.deps.client.post(
        url="https://google.serper.dev/search", headers=headers, json=payload
    )
    response.raise_for_status()
    data = response.json()

    top_results = data.get("organic", [])[:3]
    logging.info(top_results)
    return [
        SearchResult(
            title=result.get("title", "No Title"),
            url=result.get("link", ""),
            snippet=result.get("snippet", ""),
        )
        for result in top_results
    ]


@agent.tool(retries=2)
async def search_the_internet(ctx: RunContext[Deps], url: str) -> str:
    """
    Fetch a webpage and return its text contents.
    Returns only the cleaned readable text from the page, not HTML or raw code.

    Args:
        ctx: Context that includes httpx client and other deps.
        url: URL of the website to fetch.
    """

    try:
        r = await ctx.deps.client.get(url)
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        if status in {401, 403}:
            # Do not retry — bot blocked
            return "Could not search the web page (unauthorized or blocked)."
        else:
            # Retryable error
            raise ModelRetry(f"Server error {status}, trying again...") from e
    except httpx.RequestError as e:
        # Network or DNS issues → retry
        raise ModelRetry("Network error when fetching the web page.") from e

    soup = BeautifulSoup(r.text, "html.parser")
    main_text = soup.get_text()

    if main_text:
        return main_text[:20000]
    else:
        raise ModelRetry("Could not search the web page")


@agent.tool
async def call_sql_database_agent(ctx: RunContext[Deps], question: str) -> str:
    """Calls book sales database agent to answer question on our book sales database

    Args:
        ctx: Context that includes httpx client and other deps.
        question: Question on the product sales database from the user
    """
    print(f"question {question}")

    result = await sql_database_agent.run(question, deps=ctx.deps)
    print(f"call_sql_database_agent answer {result.output}")
    return result.output


def emb_text(client: OpenAI, text):
    return (
        client.embeddings.create(input=text, model="text-embedding-3-small")
        .data[0]
        .embedding
    )

class MilvusEntity(BaseModel):
    id: str
    productId: str
    title: Optional[str]
    authors: Optional[List[str]]
    category: Optional[List[str]]
    user: Optional[str]
    score: Optional[float]
    text: Optional[str]


class MilvusSearchHit(BaseModel):
    id: str
    distance: float
    entity: MilvusEntity


def parse_milvus_search_results(search_res: List[List[dict]]) -> List[MilvusSearchHit]:
    results: List[MilvusSearchHit] = []
    for hit_list in search_res:
        for hit in hit_list:
            print(hit)
            results.append(MilvusSearchHit(**hit))
    return results


@agent.tool
async def reviews_vector_database(
    ctx: RunContext[Deps], question: str
) -> List[MilvusSearchHit]:
    """Calls a vector database to retrieve relevant documents to answer question.
        Returns a list of top-k search results from the database

    Args:
        ctx: Context that includes httpx client and other deps.
        question: User question
    """
    print(f"query vector db: {question}")

    result = await literature_critic_agent.run(question, deps=ctx.deps)
    hypothetical_answer = result.output
    print(f"hypothetical critic answer {hypothetical_answer}")

    search_res = ctx.deps.milvus_client.search(
        collection_name=COLLECTION_NAME,
        data=[emb_text(ctx.deps.openai_client, hypothetical_answer)],
        limit=3,  # Top 3 results
        search_params={"metric_type": "IP", "params": {}},
        output_fields=["id", "productId","title", "authors", "category", "user", "score", "text"],
    )

    parsed_results = parse_milvus_search_results(search_res)

    return parsed_results


class ColumnSchema(BaseModel):
    column_id: int
    name: str
    type: str
    not_null: bool
    default_value: Optional[str]
    is_primary_key: bool


class TableSchema(BaseModel):
    table_name: str
    columns: List[ColumnSchema]


class DatabaseSchema(BaseModel):
    tables: List[TableSchema]


def get_table_schemas(db_path: str) -> DatabaseSchema:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    table_names = [row[0] for row in cursor.fetchall()]

    tables = []

    for table_name in table_names:
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()

        column_schemas = [
            ColumnSchema(
                column_id=col[0],
                name=col[1],
                type=col[2],
                not_null=bool(col[3]),
                default_value=col[4],
                is_primary_key=bool(col[5]),
            )
            for col in columns
        ]

        tables.append(TableSchema(table_name=table_name, columns=column_schemas))

    conn.close()
    return DatabaseSchema(tables=tables)


@sql_database_agent.tool
async def database_schema_data(ctx: RunContext[Deps]) -> DatabaseSchema:
    """Returns the current schema of the product sales database

    Args:
        ctx: Context that includes httpx client and other deps.
    """

    schemas = get_table_schemas(DB_PATH)
    print(schemas)
    return schemas


class QueryResult(BaseModel):
    """Response with data if SQL query was successfully executed."""

    sql_query: Annotated[str, MinLen(1)]
    rows: List[dict]
    columns: List[str]


class InvalidRequest(BaseModel):
    """Response when SQL could not be generated eg. is was invalid or unsafe"""

    error_message: str


@sql_database_agent.tool(retries=2)
async def database_query(
    ctx: RunContext[Deps], query: str
) -> Union[QueryResult, InvalidRequest]:
    """Validates that the query string is valid sqlite and safe to call against the database.
    If valid, executes the SELECT query and returns the results.

    Args:
        ctx: Context that includes httpx client and other deps.
        query: sqlite query string
    """
    print(f"validating and executing: {query}")

    # Define keywords we don't allow (write operations)
    unsafe_keywords = [
        "DROP",
        "DELETE",
        "UPDATE",
        "INSERT",
        "ALTER",
        "TRUNCATE",
        "ATTACH",
        "DETACH",
    ]
    lowered_query = query.strip().lower()

    # Quick security checks
    if not lowered_query.startswith("select"):
        return InvalidRequest(error_message="Only SELECT queries are allowed.")

    if any(keyword in lowered_query for keyword in unsafe_keywords):
        return InvalidRequest(
            error_message="Query contains potentially unsafe operations."
        )

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # Allows fetching results as dicts
        cursor = conn.cursor()

        # Validate the query
        cursor.execute(f"EXPLAIN QUERY PLAN {query}")
        cursor.fetchall()

        # Execute the actual query
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

        # Convert sqlite3.Row to plain dicts
        results = [dict(row) for row in rows]

        print(columns)
        print(results)

        return QueryResult(sql_query=query, rows=results, columns=columns)

    except sqlite3.Error as e:
        return ModelRetry(message=f"Invalid SQL query: {str(e)}")

    finally:
        conn.close()


@asynccontextmanager
async def lifespan(_app: fastapi.FastAPI):
    async with Database.connect() as db:
        yield {"db": db}


app = fastapi.FastAPI(lifespan=lifespan)
logfire.instrument_fastapi(app)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse((THIS_DIR / "chat_app.html"), media_type="text/html")


@app.get("/chat_app.ts")
async def main_ts() -> FileResponse:
    """Get the raw typescript code, it's compiled in the browser, forgive me."""
    return FileResponse((THIS_DIR / "chat_app.ts"), media_type="text/plain")


async def get_db(request: Request) -> Database:
    return request.state.db


def is_displayable_message(m: ModelMessage) -> bool:
    if isinstance(m, ModelRequest):
        return any(isinstance(p, UserPromptPart) for p in m.parts)
    elif isinstance(m, ModelResponse):
        return any(isinstance(p, TextPart) for p in m.parts)
    return False


@app.get("/chat/")
async def get_chat(database: Database = Depends(get_db)) -> Response:
    msgs = await database.get_messages()
    filtered = [m for m in msgs if is_displayable_message(m)]
    return Response(
        b"\n".join(json.dumps(to_chat_message(m)).encode("utf-8") for m in filtered),
        media_type="text/plain",
    )


@app.delete("/chat/")
async def delete_chat(database: Database = Depends(get_db)) -> dict:
    await database.delete_all_messages()
    return {"status": "deleted"}


class ChatMessage(TypedDict):
    """Format of messages sent to the browser."""

    role: Literal["user", "model"]
    timestamp: str
    content: str


def to_chat_message(m: ModelMessage) -> ChatMessage:
    if isinstance(m, ModelRequest):
        # Try to find the actual user prompt part
        for part in m.parts:
            if isinstance(part, UserPromptPart):
                return {
                    "role": "user",
                    "timestamp": part.timestamp.isoformat(),
                    "content": part.content,
                }

    elif isinstance(m, ModelResponse):
        for part in m.parts:
            if isinstance(part, TextPart):
                return {
                    "role": "model",
                    "timestamp": m.timestamp.isoformat(),
                    "content": part.content,
                }

    raise UnexpectedModelBehavior(f"Unexpected message type for chat app: {m}")


@app.post("/chat/")
async def post_chat(
    prompt: Annotated[str, fastapi.Form()], database: Database = Depends(get_db)
) -> StreamingResponse:
    async def stream_messages():
        """Streams new line delimited JSON `Message`s to the client."""
        # stream the user prompt so that can be displayed straight away
        yield (
            json.dumps(
                {
                    "role": "user",
                    "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                    "content": prompt,
                }
            ).encode("utf-8")
            + b"\n"
        )
        # get the chat history so far to pass as context to the agent
        messages = await database.get_messages()
        # run the agent with the user prompt and the chat history
        async with agent.run_stream(
            prompt, message_history=messages, deps=deps
        ) as result:
            async for text in result.stream(debounce_by=0.01):
                # text here is a `str` and the frontend wants
                # JSON encoded ModelResponse, so we create one
                m = ModelResponse(parts=[TextPart(text)], timestamp=result.timestamp())
                yield json.dumps(to_chat_message(m)).encode("utf-8") + b"\n"

        # add new messages (e.g. the user prompt and the agent response in this case) to the database
        await database.add_messages(result.new_messages_json())

    return StreamingResponse(stream_messages(), media_type="text/plain")


P = ParamSpec("P")
R = TypeVar("R")


@dataclass
class Database:
    """Rudimentary database to store chat messages in SQLite.

    The SQLite standard library package is synchronous, so we
    use a thread pool executor to run queries asynchronously.
    """

    con: sqlite3.Connection
    _loop: asyncio.AbstractEventLoop
    _executor: ThreadPoolExecutor

    @classmethod
    @asynccontextmanager
    async def connect(cls, file: Path = THIS_DIR / DB_NAME) -> AsyncIterator[Database]:
        with logfire.span("connect to DB"):
            loop = asyncio.get_event_loop()
            executor = ThreadPoolExecutor(max_workers=1)
            con = await loop.run_in_executor(executor, cls._connect, file)
            slf = cls(con, loop, executor)
        try:
            yield slf
        finally:
            await slf._asyncify(con.close)

    @staticmethod
    def _connect(file: Path) -> sqlite3.Connection:
        con = sqlite3.connect(str(file))
        con = logfire.instrument_sqlite3(con)
        cur = con.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS messages (id INT PRIMARY KEY, message_list TEXT);"
        )
        con.commit()
        return con

    async def add_messages(self, messages: bytes):
        await self._asyncify(
            self._execute,
            "INSERT INTO messages (message_list) VALUES (?);",
            messages,
            commit=True,
        )
        await self._asyncify(self.con.commit)

    async def get_messages(self) -> list[ModelMessage]:
        c = await self._asyncify(
            self._execute, "SELECT message_list FROM messages order by id"
        )
        rows = await self._asyncify(c.fetchall)
        messages: list[ModelMessage] = []
        for row in rows:
            messages.extend(ModelMessagesTypeAdapter.validate_json(row[0]))
        return messages

    async def delete_all_messages(self):
        await self._asyncify(self._execute, "DELETE FROM messages;", commit=True)

    def _execute(
        self, sql: LiteralString, *args: Any, commit: bool = False
    ) -> sqlite3.Cursor:
        cur = self.con.cursor()
        cur.execute(sql, args)
        if commit:
            self.con.commit()
        return cur

    async def _asyncify(
        self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> R:
        return await self._loop.run_in_executor(  # type: ignore
            self._executor,
            partial(func, **kwargs),
            *args,  # type: ignore
        )


for route in app.routes:
    print(route.path, route.methods)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("chat_app_backend:app", reload=True, reload_dirs=[str(THIS_DIR)])
