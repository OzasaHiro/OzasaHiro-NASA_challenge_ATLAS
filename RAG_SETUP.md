# Space Bio Graph RAG Chatbot

This project extends `SB_publication_PMC_with_abstract.csv` into a graph-based retrieval augmented generation (RAG) workflow using Neo4j and OpenAI models. The solution has two components:

1. `graph_ingest.py` – loads the CSV, generates embeddings for each abstract, and stores them in Neo4j with similarity edges.
2. `rag_chat.py` – Streamlit UI with a NASA-inspired theme that answers research questions by querying the Neo4j vector index and summarising the most relevant abstracts.

## Prerequisites

- Python 3.11 with the existing `.venv` activated (`source .venv/bin/activate`).
- A running Neo4j instance (Neo4j 5.11+ recommended) with vector indexing enabled.
- API keys stored in environment variables or `secrets/.env` (copy from `secrets/.env.example`):
  - `OPENAI_API_KEY` – OpenAI Responses + Embeddings API.
  - `NEO4J_URI` – e.g. `neo4j+s://<your-aura-host>`.
  - `NEO4J_USERNAME` – defaults to `neo4j` if omitted.
  - `NEO4J_PASSWORD` or `NEO4J_API_KEY` – authentication secret.

> `graph_ingest.py` and `rag_chat.py` automatically read `secrets/.env` if the relevant env vars are empty.

## 1. Ingest publications into Neo4j

### 接続テスト
インジェストの前に、以下のワンコマンドで Neo4J Bolt 接続が成功するか確認してください。`secrets/.env` に `NEO4J_URI` などの値を記載済みなら追加設定は不要です。

```
python test_neo4j_connection.py
```

成功すると `Connected via Bolt as ...` が表示されます。失敗する場合は `secrets/.env` のクレデンシャルを再確認してください。

```
python graph_ingest.py \
  --input SB_publication_PMC_with_abstract.csv \
  --embed-model text-embedding-3-small \
  --similar-k 5 \
  --sleep 0.4
```

Key behaviour:

- Computes embeddings for each abstract (falls back to the title if an abstract is missing).
- Creates/updates `Publication` nodes with properties: `title`, `link`, `abstract`, `embedding`, `embedding_model`.
- Builds directional `SIMILAR_TO` relationships between each paper and its top-`k` neighbours (cosine similarity score stored as `score`).
- Creates a Neo4j vector index named `publicationEmbedding` by default; override with `--index-name` if needed.
- Use `--wipe` to clear existing `Publication` nodes before ingesting. Use `--max-rows` for quick tests.

## 2. Launch the Graph RAG chatbot

```
streamlit run rag_chat.py
```

Features:

- Cosmos-inspired background and theming for a NASA user experience.
- Chat memory via Streamlit’s session state.
- Vector retrieval via `db.index.vector.queryNodes` plus neighbour expansion over `SIMILAR_TO` relationships.
- OpenAI Responses API (`CHAT_MODEL` env var, default `gpt-4.1-mini`) generates Japanese answers citing relevant titles.

Environment overrides:

- `EMBED_MODEL` – embedding model used at query time (must match ingestion dimension).
- `NEO4J_VECTOR_INDEX` – index name (defaults to `publicationEmbedding`).
- `TOP_K` – number of primary matches to retrieve (default `5`).
- `NEIGHBOR_LIMIT` – number of neighbours to show per match (default `3`).
- `CHAT_TEMPERATURE` – sampling temperature (default `0`; set `<0` to omit the parameter).

## Workflow Tips

- Re-run `graph_ingest.py` after updating abstracts or adjusting similarity parameters.
- For large batches, reduce `--sleep` only if you are within OpenAI rate limits.
- You can explore the resulting graph directly in Neo4j Browser with:
  ```cypher
  MATCH (p:Publication)-[r:SIMILAR_TO]->(q:Publication)
  RETURN p, r, q LIMIT 50;
  ```
- If you add new relationships (e.g. keyword nodes), the chatbot prompt can be extended in `build_prompt` inside `rag_chat.py`.

Enjoy exploring space bioscience knowledge with graph-powered RAG!
