#!/usr/bin/env python3
"""Ingest publication abstracts into Neo4j and create similarity relationships."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from neo4j.exceptions import ClientError
from openai import OpenAI

DEFAULT_INPUT = "SB_publication_PMC_with_abstract.csv"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_SIMILAR_K = 5


@dataclass
class Config:
    input_path: Path
    batch_size: int
    similar_k: int
    embed_model: str
    dry_run: bool
    wipe: bool
    max_rows: Optional[int]
    sleep: float
    index_name: str
    database: str | None


@dataclass
class Neo4jCreds:
    uri: str
    username: str
    password: str


def parse_args(argv: Optional[Iterable[str]] = None) -> Config:
    parser = argparse.ArgumentParser(description="Load publication abstracts into Neo4j")
    parser.add_argument("--input", default=DEFAULT_INPUT, type=Path,
                        help=f"Input CSV with abstracts (default: {DEFAULT_INPUT})")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Batch size for writing nodes to Neo4j (default: 50)")
    parser.add_argument("--similar-k", type=int, default=DEFAULT_SIMILAR_K,
                        help="Number of nearest neighbours to connect for each publication")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL,
                        help=f"Embedding model name (default: {DEFAULT_EMBED_MODEL})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip writing to Neo4j; only log planned actions")
    parser.add_argument("--wipe", action="store_true",
                        help="Delete existing Publication nodes before ingesting")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Limit number of rows ingested (for testing)")
    parser.add_argument("--sleep", type=float, default=0.4,
                        help="Seconds to sleep between embedding requests")
    parser.add_argument("--index-name", default="publicationEmbedding",
                        help="Name of the Neo4j vector index")
    parser.add_argument("--database", default=os.getenv("NEO4J_DATABASE"),
                        help="Target Neo4j database (default: env NEO4J_DATABASE or neo4j)")

    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.similar_k <= 0:
        parser.error("--similar-k must be positive")
    if args.sleep < 0:
        parser.error("--sleep must be non-negative")

    return Config(
        input_path=args.input.expanduser(),
        batch_size=args.batch_size,
        similar_k=args.similar_k,
        embed_model=args.embed_model,
        dry_run=args.dry_run,
        wipe=args.wipe,
        max_rows=args.max_rows,
        sleep=args.sleep,
        index_name=args.index_name,
        database=args.database,
    )


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def _ensure_secret(var_name: str, default: str | None = None, key_path: str | None = None) -> str:
    if key_path is None:
        key_path = os.getenv("SECRETS_FILE", "secrets/.env")
    value = os.getenv(var_name)
    if value:
        return value
    key_file = Path(key_path)
    if key_file.exists():
        with key_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith(f"{var_name}="):
                    _, raw_value = line.split("=", 1)
                    raw_value = raw_value.strip()
                    if raw_value:
                        os.environ[var_name] = raw_value
                        return raw_value
    if default is not None:
        return default
    raise RuntimeError(f"Missing required secret: {var_name}")


def load_env_credentials() -> Neo4jCreds:
    uri = _ensure_secret("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME") or _ensure_secret("NEO4J_USERNAME", default="neo4j")
    password = os.getenv("NEO4J_PASSWORD") or _ensure_secret("NEO4J_PASSWORD")
    return Neo4jCreds(uri=uri, username=username, password=password)


def ensure_openai_key(env_var: str = "OPENAI_API_KEY", key_path: str | None = None) -> None:
    if key_path is None:
        key_path = os.getenv("SECRETS_FILE", "secrets/.env")
    if os.getenv(env_var):
        return
    key_file = Path(key_path)
    if not key_file.exists():
        raise RuntimeError(f"{env_var} not set and {key_path} missing")
    with key_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(f"{env_var}="):
                _, value = line.split("=", 1)
                os.environ[env_var] = value.strip()
                logging.info("Loaded %s from %s", env_var, key_path)
                return
    raise RuntimeError(f"Could not find {env_var} in {key_path}")


def load_dataframe(config: Config) -> pd.DataFrame:
    if not config.input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {config.input_path}")
    df = pd.read_csv(config.input_path, encoding="utf-8-sig")
    required = {"Title", "Link", "abstract"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {', '.join(sorted(missing))}")
    if config.max_rows is not None:
        df = df.head(config.max_rows)
    df = df.fillna("").reset_index(drop=True)
    return df


def publication_id(row: pd.Series, idx: int) -> str:
    link = str(row.get("Link", "")).strip()
    return link or f"pub-{idx}"


def compute_embeddings(df: pd.DataFrame, model: str, sleep_seconds: float) -> np.ndarray:
    ensure_openai_key()
    client = OpenAI()
    vectors: list[list[float]] = []
    for idx, row in df.iterrows():
        text = str(row.get("abstract") or row.get("Title") or "")
        text = text.strip()
        if not text:
            vectors.append([0.0])
            continue
        logging.info("Embedding row %s", idx)
        response = client.embeddings.create(model=model, input=text)
        vector = response.data[0].embedding
        vectors.append(vector)
        if sleep_seconds:
            time.sleep(sleep_seconds)
    max_len = max(len(vec) for vec in vectors)
    for i, vec in enumerate(vectors):
        if len(vec) != max_len:
            raise ValueError(f"Embedding length mismatch at row {i}: expected {max_len}, got {len(vec)}")
    return np.array(vectors, dtype=np.float32)


def cosine_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = matrix / norms
    return normalized @ normalized.T


def prepare_similarity_edges(sim_matrix: np.ndarray, k: int) -> list[tuple[int, int, float]]:
    np.fill_diagonal(sim_matrix, -1.0)
    edges: list[tuple[int, int, float]] = []
    for i in range(sim_matrix.shape[0]):
        row = sim_matrix[i]
        if k >= len(row):
            top_indices = np.argsort(row)[::-1]
        else:
            partition = np.argpartition(row, -k)[-k:]
            top_indices = partition[np.argsort(row[partition])[::-1]]
        for j in top_indices:
            if j <= i:
                continue
            score = float(row[j])
            if score <= 0:
                continue
            edges.append((i, j, score))
            edges.append((j, i, score))
    return edges


def chunk_iterable(items: list, chunk_size: int):
    for start in range(0, len(items), chunk_size):
        yield items[start:start + chunk_size]


def ensure_indexes(session, index_name: str, dimension: int) -> None:
    session.run("CREATE CONSTRAINT publication_id IF NOT EXISTS FOR (p:Publication) REQUIRE p.id IS UNIQUE")
    result = session.run("SHOW INDEXES YIELD name WHERE name = $name", name=index_name)
    if not result.peek():
        logging.info("Creating vector index %s (dimension=%s)", index_name, dimension)
        session.run(
            "CALL db.index.vector.createNodeIndex($name, 'Publication', 'embedding', $dimension, 'cosine')",
            name=index_name,
            dimension=dimension,
        )
    else:
        logging.info("Vector index %s already exists", index_name)


def write_publications(session, df: pd.DataFrame, embeddings: np.ndarray, batch_size: int, model: str) -> None:
    records = []
    for idx, row in df.iterrows():
        record_id = publication_id(row, idx)
        record = {
            "id": record_id,
            "title": row.get("Title", ""),
            "link": row.get("Link", ""),
            "abstract": row.get("abstract", ""),
            "embedding": embeddings[idx].tolist(),
            "embedding_model": model,
        }
        records.append(record)
    for chunk in chunk_iterable(records, batch_size):
        session.run(
            "UNWIND $batch AS row "
            "MERGE (p:Publication {id: row.id}) "
            "SET p.title = row.title, "
            "    p.link = row.link, "
            "    p.abstract = row.abstract, "
            "    p.embedding = row.embedding, "
            "    p.embedding_model = row.embedding_model, "
            "    p.updated_at = timestamp()",
            batch=chunk,
        )
        logging.info("Wrote batch of %s publications", len(chunk))


def write_similarity_edges(session, df: pd.DataFrame, edges: list[tuple[int, int, float]]) -> None:
    if not edges:
        logging.warning("No similarity edges to write")
        return
    batch_payload = []
    for i, j, score in edges:
        source_row = df.iloc[i]
        target_row = df.iloc[j]
        batch_payload.append(
            {
                "source": publication_id(source_row, i),
                "target": publication_id(target_row, j),
                "score": score,
            }
        )
    for chunk in chunk_iterable(batch_payload, 100):
        session.run(
            "UNWIND $batch AS row "
            "MATCH (a:Publication {id: row.source}) "
            "MATCH (b:Publication {id: row.target}) "
            "MERGE (a)-[r:SIMILAR_TO]->(b) "
            "ON CREATE SET r.score = row.score "
            "ON MATCH SET r.score = row.score, r.updated_at = timestamp()",
            batch=chunk,
        )
        logging.info("Created %s similarity relationships", len(chunk))


def main(argv: Optional[Iterable[str]] = None) -> int:
    configure_logging()
    try:
        config = parse_args(argv)
        df = load_dataframe(config)
        logging.info("Loaded %s publications", len(df))

        if config.dry_run:
            logging.info("Dry run enabled; skipping embedding and database operations")
            return 0

        embeddings = compute_embeddings(df, config.embed_model, config.sleep)
        logging.info("Computed embeddings with shape %s", embeddings.shape)

        similarity = prepare_similarity_edges(cosine_similarity_matrix(embeddings), config.similar_k)
        logging.info("Prepared %s similarity edges", len(similarity))

        creds = load_env_credentials()
        driver = GraphDatabase.driver(creds.uri, auth=(creds.username, creds.password))
        session_kwargs = {}
        if config.database:
            session_kwargs["database"] = config.database
        with driver.session(**session_kwargs) as session:
            if config.wipe:
                logging.warning("Wiping existing Publication graph")
                session.run("MATCH (p:Publication) DETACH DELETE p")
            ensure_indexes(session, config.index_name, embeddings.shape[1])
            write_publications(session, df, embeddings, config.batch_size, config.embed_model)
            write_similarity_edges(session, df, similarity)
        driver.close()
        logging.info("Ingestion completed")
        return 0
    except Exception as exc:  # noqa: BLE001
        logging.exception("Fatal error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
