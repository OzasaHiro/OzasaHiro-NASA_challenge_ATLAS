#!/usr/bin/env python3
"""Space-themed Graph RAG chatbot powered by Neo4j and OpenAI."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from textwrap import dedent
from typing import Any, Dict, List

import numpy as np
import streamlit as st
from neo4j import GraphDatabase
from openai import OpenAI

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
INDEX_NAME = os.getenv("NEO4J_VECTOR_INDEX", "publicationEmbedding")
TOP_K = int(os.getenv("TOP_K", "5"))
NEIGHBOR_LIMIT = int(os.getenv("NEIGHBOR_LIMIT", "3"))
TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", "0"))
DATABASE = os.getenv("NEO4J_DATABASE")

st.set_page_config(
    page_title="NASA Space Bio RAG",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def ensure_openai_key(env_var: str = "OPENAI_API_KEY", key_path: str | None = None) -> None:
    if key_path is None:
        key_path = os.getenv("SECRETS_FILE", "secrets/.env")
    if os.getenv(env_var):
        return
    key_file = os.path.abspath(key_path)
    if not os.path.exists(key_file):
        raise RuntimeError(f"{env_var} not set and {key_path} missing")
    with open(key_file, "r", encoding="utf-8") as fh:
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


def ensure_secret(var_name: str, default: str | None = None, key_path: str | None = None) -> str:
    if key_path is None:
        key_path = os.getenv("SECRETS_FILE", "secrets/.env")
    value = os.getenv(var_name)
    if value:
        return value
    key_file = os.path.abspath(key_path)
    if os.path.exists(key_file):
        with open(key_file, "r", encoding="utf-8") as fh:
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
        os.environ[var_name] = default
        return default
    raise RuntimeError(f"Missing required secret: {var_name}")


def load_neo4j_creds() -> Dict[str, str]:
    uri = ensure_secret("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME") or ensure_secret("NEO4J_USERNAME", default="neo4j")
    password = os.getenv("NEO4J_PASSWORD") or ensure_secret("NEO4J_PASSWORD")
    return {"uri": uri, "username": username, "password": password}


@st.cache_resource(show_spinner=False)
def get_driver():
    creds = load_neo4j_creds()
    return GraphDatabase.driver(creds["uri"], auth=(creds["username"], creds["password"]))


@st.cache_resource(show_spinner=False)
def get_openai_client():
    ensure_openai_key()
    return OpenAI()


def embed_text(text: str) -> List[float]:
    client = get_openai_client()
    response = client.embeddings.create(model=EMBED_MODEL, input=text)
    return response.data[0].embedding


def fetch_context(question_embedding: List[float]) -> List[Dict[str, Any]]:
    driver = get_driver()
    session_kwargs = {}
    if DATABASE:
        session_kwargs["database"] = DATABASE
    with driver.session(**session_kwargs) as session:
        cypher = dedent(
            """
            CALL db.index.vector.queryNodes($index, $top_k, $embedding)
            YIELD node, score
            OPTIONAL MATCH (node)-[r:SIMILAR_TO]->(neighbor)
            WITH node, score, r, neighbor
            ORDER BY score DESC, r.score DESC
            WITH node, score,
                 collect(DISTINCT {
                     title: neighbor.title,
                     link: neighbor.link,
                     abstract: neighbor.abstract,
                     score: r.score
                 })[0..$neighbor_limit] AS neighbors
            RETURN node.title AS title,
                   node.link AS link,
                   node.abstract AS abstract,
                   score,
                   neighbors
            ORDER BY score DESC
            """
        )
        records = session.run(
            cypher,
            index=INDEX_NAME,
            top_k=TOP_K,
            embedding=question_embedding,
            neighbor_limit=NEIGHBOR_LIMIT,
        )
        return [record.data() for record in records]


def build_prompt(question: str, context: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    context_blocks: List[str] = []
    for idx, item in enumerate(context, start=1):
        neighbors_text = "".join(
            f"    - Related study: {neighbor['title']} (score: {neighbor.get('score', 0):.3f})\n"
            f"      Summary: {neighbor.get('abstract', '')[:400]}\n"
            for neighbor in item.get("neighbors", []) if neighbor.get("title")
        )
        block = dedent(
            f"""
            ### Primary Result {idx}
            Title: {item.get('title', 'Unknown')}
            Link: {item.get('link', 'Unknown')}
            Score: {item.get('score', 0):.3f}
            Summary: {item.get('abstract', '')}
            {neighbors_text}
            """
        ).strip()
        context_blocks.append(block)
    context_text = "\n\n".join(context_blocks)
    system_prompt = (
        "You are a research assistant supporting NASA space bioscience projects. "
        "Answer the user's question in English using only the provided publication summaries as evidence. "
        "Cite the supporting titles in your response and flag any conjecture explicitly."
    )
    user_content = (
        f"Question: {question}\n\n"
        "Available publication summaries:\n"
        f"{context_text if context_text else 'No matching documents were found.'}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def generate_answer(question: str, context: List[Dict[str, Any]]) -> str:
    if not context:
        return "I could not find relevant publications in the knowledge graph. Please try another question."
    client = get_openai_client()
    messages = build_prompt(question, context)
    request_kwargs = {
        "model": os.getenv("CHAT_MODEL", "gpt-4.1-mini"),
        "input": messages,
    }
    if TEMPERATURE >= 0:
        request_kwargs["temperature"] = TEMPERATURE
    response = client.responses.create(**request_kwargs)
    return response.output_text.strip()


def render_context_panels(context: List[Dict[str, Any]]):
    for item in context:
        with st.expander(f"{item.get('title', 'Unknown')} — score {item.get('score', 0):.3f}"):
            st.markdown(f"**Link:** [{item.get('link', 'Unknown')}]({item.get('link', '#')})")
            st.markdown(f"**Summary:** {item.get('abstract', '---')}")
            neighbors = item.get("neighbors", [])
            if neighbors:
                st.markdown("**Related papers:**")
                for neighbor in neighbors:
                    st.markdown(
                        f"- {neighbor.get('title', 'Unknown')} (score {neighbor.get('score', 0):.3f})\n"
                        f"  {neighbor.get('abstract', '')[:400]}..."
                    )


def main() -> None:
    st.markdown(
        """
        <style>
        body {
            background: radial-gradient(circle at 20% 20%, rgba(65,105,225,0.4), transparent 60%),
                        radial-gradient(circle at 80% 10%, rgba(138,43,226,0.35), transparent 55%),
                        linear-gradient(180deg, #05010f 0%, #02030f 45%, #020016 100%);
            color: #f0f6ff;
            font-family: 'Helvetica Neue', sans-serif;
        }
        .stApp header {background: transparent;}
        .stChatFloatingInputContainer textarea {
            background: rgba(10, 20, 60, 0.8) !important;
            color: #f0f6ff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🛰️ Space Bio Graph RAG Console")
    st.caption("English-language assistant grounded in space bioscience publication summaries")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Ask anything about space bioscience…"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Scanning orbital knowledge graph for supporting literature…"):
                embedding = embed_text(user_input)
                context = fetch_context(embedding)
                render_context_panels(context)
                answer = generate_answer(user_input, context)
                st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
