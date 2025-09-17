#!/usr/bin/env python3
"""Augment SB_publication_PMC.csv with Japanese abstracts generated via GPT-5."""
from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urlparse, urlunparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from openai import BadRequestError, OpenAI

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
)
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MODEL_FALLBACKS: list[str] = ["gpt-4.1", "gpt-4o-mini"]


class TemperatureUnsupportedError(RuntimeError):
    """Raised when the target model rejects the temperature parameter."""


class ModelUnavailableError(RuntimeError):
    """Raised when the requested model is unavailable."""


@dataclasses.dataclass
class RunConfig:
    input_path: Path
    output_path: Path
    cache_dir: Optional[Path]
    model: str
    model_fallbacks: list[str]
    start: int
    limit: Optional[int]
    sleep: float
    overwrite: bool
    dry_run: bool
    save_every: int
    max_context_chars: int
    temperature: Optional[float]


def parse_args(argv: Optional[Iterable[str]] = None) -> RunConfig:
    parser = argparse.ArgumentParser(
        description="Fetch PMC articles and generate Japanese abstracts using GPT-5",
    )
    parser.add_argument("--input", dest="input_path", default="SB_publication_PMC.csv", type=Path,
                        help="Input CSV path (default: SB_publication_PMC.csv)")
    parser.add_argument("--output", dest="output_path", default="SB_publication_PMC_with_abstract.csv", type=Path,
                        help="Output CSV path (default: SB_publication_PMC_with_abstract.csv)")
    parser.add_argument("--cache-dir", dest="cache_dir", type=Path, default=Path(".cache/pmc_html"),
                        help="Directory to cache downloaded PMC articles (default: .cache/pmc_html)")
    parser.add_argument("--model", dest="model", default=DEFAULT_MODEL,
                        help=f"Primary OpenAI model name to use (default: {DEFAULT_MODEL})")
    parser.add_argument(
        "--fallback-models",
        dest="fallback_models",
        default=",".join(DEFAULT_MODEL_FALLBACKS),
        help=(
            "Comma-separated list of fallback model names to try when the primary model is unavailable "
            f"(default: {', '.join(DEFAULT_MODEL_FALLBACKS)})"
        ),
    )
    parser.add_argument("--start", dest="start", type=int, default=0,
                        help="Row index to start processing from (default: 0)")
    parser.add_argument("--limit", dest="limit", type=int, default=None,
                        help="Maximum number of rows to process (default: all)")
    parser.add_argument("--sleep", dest="sleep", type=float, default=1.2,
                        help="Seconds to sleep between API calls (default: 1.2)")
    parser.add_argument("--overwrite", dest="overwrite", action="store_true",
                        help="Regenerate abstracts even if already present")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true",
                        help="Skip API calls and only show which rows would be processed")
    parser.add_argument("--save-every", dest="save_every", type=int, default=5,
                        help="Write intermediate CSV every N processed rows (default: 5)")
    parser.add_argument("--max-context-chars", dest="max_context_chars", type=int, default=4000,
                        help="Maximum characters of article text to send to GPT (default: 4000)")
    parser.add_argument("--temperature", dest="temperature", type=float, default=DEFAULT_TEMPERATURE,
                        help="Sampling temperature for the model; set to a negative value to disable (default: 0.2)")

    args = parser.parse_args(argv)
    cache_dir = args.cache_dir
    if cache_dir is not None:
        cache_dir = cache_dir.expanduser()

    fallback_models = [
        model.strip()
        for model in (args.fallback_models.split(",") if args.fallback_models else [])
        if model.strip()
    ]

    return RunConfig(
        input_path=args.input_path.expanduser(),
        output_path=args.output_path.expanduser(),
        cache_dir=cache_dir,
        model=args.model,
        model_fallbacks=fallback_models,
        start=max(args.start, 0),
        limit=args.limit,
        sleep=max(args.sleep, 0.0),
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        save_every=max(args.save_every, 1),
        max_context_chars=max(args.max_context_chars, 500),
        temperature=None if args.temperature is not None and args.temperature < 0 else args.temperature,
    )


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def ensure_api_key(env_var: str = "OPENAI_API_KEY", key_path: str | None = None) -> None:
    if key_path is None:
        key_path = os.getenv("SECRETS_FILE", "secrets/.env")
    if os.getenv(env_var):
        return
    key_file = Path(key_path)
    if not key_file.exists():
        raise RuntimeError(
            f"Environment variable {env_var} is not set and {key_path} was not found."
        )
    with key_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(f"{env_var}="):
                _, value = line.split("=", 1)
                value = value.strip()
                if not value:
                    continue
                os.environ[env_var] = value
                logging.info("Loaded %s from %s", env_var, key_path)
                return
    raise RuntimeError(f"Could not locate {env_var} inside {key_path}")


def load_dataframe(input_path: Path, output_path: Path) -> pd.DataFrame:
    encoding = "utf-8-sig"
    if output_path.exists():
        logging.info("Resuming from existing output file %s", output_path)
        df = pd.read_csv(output_path, encoding=encoding)
    else:
        df = pd.read_csv(input_path, encoding=encoding)
    if "abstract" not in df.columns:
        df["abstract"] = ""
    else:
        df["abstract"] = df["abstract"].astype(object).fillna("")
    return df


def normalize_url(url: str) -> str:
    if not url:
        return url
    parsed = urlparse(url)
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc
    path = parsed.path
    if "ncbi.nlm.nih.gov" in netloc:
        netloc = "pmc.ncbi.nlm.nih.gov"
        path = re.sub(r"^/pmc/articles/", "/articles/", path)
    normalized = urlunparse((scheme, netloc, path, parsed.params, parsed.query, parsed.fragment))
    return normalized


def derive_cache_path(cache_dir: Optional[Path], url: str) -> Optional[Path]:
    if cache_dir is None:
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    match = re.search(r"/articles/(PMC[0-9]+)/", url)
    if match:
        name = match.group(1)
    else:
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", url)
        name = safe[:100]
    return cache_dir / f"{name}.html"


def fetch_article_html(url: str, cache_dir: Optional[Path]) -> str:
    normalized_url = normalize_url(url)
    cache_path = derive_cache_path(cache_dir, normalized_url)
    if cache_path and cache_path.exists():
        return cache_path.read_text(encoding="utf-8", errors="ignore")

    headers = {"User-Agent": USER_AGENT}
    response = requests.get(normalized_url, headers=headers, timeout=45)
    response.raise_for_status()
    html = response.text
    if cache_path:
        cache_path.write_text(html, encoding="utf-8")
    return html


def _collect_text_from_heading(heading, max_chars: int) -> str:
    collected: list[str] = []
    for sibling in heading.next_siblings:
        if getattr(sibling, "name", None) in {"h1", "h2", "h3", "h4"}:
            break
        if getattr(sibling, "name", None) == "p":
            text = sibling.get_text(" ", strip=True)
            if text:
                collected.append(text)
        elif getattr(sibling, "name", None) in {"ul", "ol"}:
            for li in sibling.find_all("li"):
                text = li.get_text(" ", strip=True)
                if text:
                    collected.append(text)
        elif isinstance(sibling, str):
            text = sibling.strip()
            if text:
                collected.append(text)
        if sum(len(c) for c in collected) >= max_chars:
            break
    text = "\n".join(collected)
    return text[:max_chars]


def extract_article_context(html: str, max_chars: int) -> str:
    soup = BeautifulSoup(html, "html.parser")
    article = soup.find("article") or soup

    sections: list[str] = []

    # Extract title
    title_tag = article.find(["h1", "h2"], string=True)
    title = title_tag.get_text(" ", strip=True) if title_tag else ""
    if title:
        sections.append(f"タイトル: {title}")

    # Extract abstract or summary-like sections
    heading_keywords = [
        "abstract",
        "summary",
        "results",
        "findings",
        "conclusion",
        "conclusions",
        "discussion",
        "overview",
    ]
    seen_headings: set[str] = set()
    for heading in article.find_all(["h1", "h2", "h3", "h4"]):
        heading_text = heading.get_text(" ", strip=True)
        heading_lower = heading_text.lower()
        matched = None
        for keyword in heading_keywords:
            if keyword in heading_lower:
                matched = keyword
                break
        if not matched:
            continue
        if heading_text in seen_headings:
            continue
        seen_headings.add(heading_text)
        body = _collect_text_from_heading(heading, max_chars)
        if body:
            sections.append(f"{heading_text}:\n{body}")
        if sum(len(section) for section in sections) >= max_chars:
            break

    if len(sections) == 1:
        # Fallback: grab first few paragraphs if no sections beyond title
        count = 0
        for paragraph in article.find_all("p"):
            text = paragraph.get_text(" ", strip=True)
            if not text:
                continue
            sections.append(text)
            count += 1
            if count >= 5 or sum(len(section) for section in sections) >= max_chars:
                break

    combined = "\n\n".join(sections)
    if len(combined) > max_chars:
        combined = combined[:max_chars]
    return combined


def build_prompt(title: str, context: str) -> list[dict]:
    system_instruction = (
        "あなたは宇宙生物医学の専門家です。提供された論文情報をもとに、研究の目的、主要な方法や観察点、"
        "得られた成果や意義をわかりやすくまとめます。"
    )
    user_instruction = (
        "以下は論文タイトルと本文から抽出した要約材料です。研究の背景と目的、主要な実験や観測の内容、"
        "成果・結論を3〜4文で日本語説明にまとめてください。専門用語は必要に応じて使い、"
        "一般読者にも意図が伝わるようにしてください。"
        "箇条書きにはせず、自然な文章で記述し、センテンス間で論理的なつながりを示してください。"
        "また、宇宙環境に関連する場合はその点も言及してください。"
        "もし情報が不足して結論が推測に頼る場合は、その旨を明示してください。"
    )
    article_payload = f"論文タイトル: {title}\n\n抽出テキスト:\n{context.strip()}"
    return [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_instruction + "\n\n" + article_payload},
    ]


def _extract_error_message(exc: BadRequestError) -> str:
    message = ""
    try:
        response = getattr(exc, "response", None)
        if response is not None:
            json_method = getattr(response, "json", None)
            if callable(json_method):
                body = json_method()
                if isinstance(body, dict):
                    error = body.get("error")
                    if isinstance(error, dict):
                        message = str(error.get("message", ""))
    except Exception:  # noqa: BLE001
        message = ""
    if not message:
        message = str(exc)
    return message


def _is_temperature_error(exc: BadRequestError) -> bool:
    message = _extract_error_message(exc).lower()
    return "temperature" in message and "unsupported" in message


def _is_model_not_found(exc: BadRequestError) -> bool:
    message = _extract_error_message(exc).lower()
    if "model" not in message:
        return False
    keywords = ["does not exist", "not exist", "model_not_found", "not found"]
    return any(keyword in message for keyword in keywords)


def generate_summary(
    client: OpenAI,
    model: str,
    title: str,
    context: str,
    temperature: Optional[float],
) -> str:
    prompt = build_prompt(title, context)
    request_kwargs: dict[str, object] = {
        "model": model,
        "input": prompt,
    }
    if temperature is not None:
        request_kwargs["temperature"] = temperature
    try:
        response = client.responses.create(**request_kwargs)
    except BadRequestError as exc:
        if temperature is not None and _is_temperature_error(exc):
            raise TemperatureUnsupportedError("temperature parameter unsupported") from exc
        if _is_model_not_found(exc):
            raise ModelUnavailableError(f"Model {model} unavailable") from exc
        raise
    text = response.output_text.strip()
    return text


def process_rows(config: RunConfig) -> None:
    configure_logging()
    ensure_api_key()
    df = load_dataframe(config.input_path, config.output_path)

    total_rows = len(df)
    client = None if config.dry_run else OpenAI()

    model_candidates: list[str] = []
    for candidate in [config.model, *config.model_fallbacks]:
        if candidate and candidate not in model_candidates:
            model_candidates.append(candidate)
    if not model_candidates:
        raise RuntimeError("No language models configured. Specify --model or --fallback-models.")
    active_model_index = 0

    start = config.start
    end = total_rows if config.limit is None else min(total_rows, start + config.limit)
    if start >= total_rows:
        logging.warning("Start index %s is beyond the last row (%s)", start, total_rows - 1)
        return

    processed_since_save = 0

    for idx in range(start, end):
        row = df.iloc[idx]
        title = str(row.get("Title", "")).strip()
        link = str(row.get("Link", "")).strip()

        if not link:
            logging.warning("Row %s skipped: missing Link", idx)
            continue

        if not config.overwrite:
            existing_value = row.get("abstract", "")
            if isinstance(existing_value, str):
                if existing_value.strip():
                    logging.info("Row %s already has abstract, skipping", idx)
                    continue
            elif pd.notna(existing_value):
                logging.info("Row %s already has abstract, skipping", idx)
                continue

        logging.info("Processing row %s/%s: %s", idx + 1, total_rows, title or link)

        if config.dry_run:
            continue

        try:
            html = fetch_article_html(link, config.cache_dir)
            context = extract_article_context(html, config.max_context_chars)
            if not context:
                logging.warning("Row %s: could not extract context from %s", idx, link)
                continue
            summary: Optional[str] = None
            attempt_index = active_model_index
            last_error: Optional[Exception] = None
            while attempt_index < len(model_candidates):
                model_name = model_candidates[attempt_index]
                try:
                    summary = generate_summary(
                        client,
                        model_name,
                        title,
                        context,
                        config.temperature,
                    )
                    if attempt_index != active_model_index:
                        # Promote successful fallback to the front for subsequent rows.
                        chosen = model_candidates.pop(attempt_index)
                        model_candidates.insert(0, chosen)
                        logging.info(
                            "Switched active model to fallback %s",
                            chosen,
                        )
                        active_model_index = 0
                    else:
                        active_model_index = attempt_index
                    config.model = model_candidates[active_model_index]
                    break
                except TemperatureUnsupportedError:
                    logging.warning(
                        "Model %s does not support temperature control; retrying without it",
                        model_name,
                    )
                    config.temperature = None
                    continue
                except ModelUnavailableError as exc:
                    logging.warning(
                        "Model %s unavailable; trying next fallback (detail: %s)",
                        model_name,
                        exc,
                    )
                    last_error = exc
                    attempt_index += 1
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    raise

            if summary is None:
                logging.error(
                    "Row %s: all configured models failed. Last error: %s",
                    idx,
                    last_error,
                )
                continue
            df.at[idx, "abstract"] = summary
            processed_since_save += 1
            logging.info("Row %s: abstract generated (%d chars)", idx, len(summary))
            if processed_since_save % config.save_every == 0:
                df.to_csv(config.output_path, index=False, encoding="utf-8-sig")
                logging.info("Checkpoint saved to %s", config.output_path)
            time.sleep(config.sleep)
        except requests.HTTPError as exc:
            logging.error("HTTP error for %s: %s", link, exc)
        except Exception as exc:  # noqa: BLE001
            logging.exception("Failed to process row %s (%s): %s", idx, link, exc)

    if not config.dry_run:
        df.to_csv(config.output_path, index=False, encoding="utf-8-sig")
        logging.info("Final output written to %s", config.output_path)


def main(argv: Optional[Iterable[str]] = None) -> int:
    try:
        config = parse_args(argv)
        process_rows(config)
        return 0
    except KeyboardInterrupt:
        logging.warning("Interrupted by user")
        return 1
    except Exception as exc:  # noqa: BLE001
        logging.exception("Fatal error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
