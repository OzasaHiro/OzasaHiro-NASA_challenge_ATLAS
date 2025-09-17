#!/usr/bin/env python3
"""Quick Neo4j connectivity check using credentials from secrets/.env or env vars."""
from __future__ import annotations

from graph_ingest import load_env_credentials
from neo4j import GraphDatabase


def main() -> None:
    creds = load_env_credentials()
    with GraphDatabase.driver(creds.uri, auth=(creds.username, creds.password)) as driver:
        driver.verify_connectivity()
        print(f"Connected via Bolt as {creds.username}")


if __name__ == "__main__":
    main()
