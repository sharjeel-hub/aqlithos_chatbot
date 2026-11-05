#!/usr/bin/env python3
"""Collect public website pages and linked PDFs for a company chatbot knowledge base.

Example:
    python scraper.py --start-url https://example.com --max-pages 300 --delay 1.5

The script does not generate embeddings or call an LLM. It creates source documents
that can be chunked and embedded in the next RAG development step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sqlite3
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import (
    parse_qsl,
    urlencode,
    urljoin,
    urlsplit,
    urlunsplit,
)
from urllib.robotparser import RobotFileParser

import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

APP_NAME = "AqlithosKnowledgeCollector/1.0"
DEFAULT_OUTPUT = Path("data")
MAX_PDF_BYTES = 30 * 1024 * 1024  # 30 MB safety limit per PDF
TRACKING_PARAMS = {
    "fbclid", "gclid", "mc_cid", "mc_eid", "ref", "ref_src",
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
}
SKIP_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico",
    ".mp3", ".mp4", ".mov", ".avi", ".zip", ".rar", ".7z",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
}


@dataclass
class Document:
    source_url: str
    title: str
    content: str
    source_type: str
    description: str = ""
    language: str = ""
    content_hash: str = ""
    collected_at: str = ""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean_text(text: str) -> str:
    """Normalize whitespace without destroying paragraph boundaries."""
    text = text.replace("\xa0", " ").replace("\r", "\n")
    lines = [re.sub(r"[\t ]+", " ", line).strip() for line in text.splitlines()]
    cleaned: list[str] = []
    previous_blank = False
    for line in lines:
        is_blank = not line
        if is_blank:
            if not previous_blank:
                cleaned.append("")
            previous_blank = True
        else:
            cleaned.append(line)
            previous_blank = False
    return "\n".join(cleaned).strip()


def normalize_url(url: str) -> Optional[str]:
    """Remove fragments and common tracking parameters; allow only HTTP(S)."""
    try:
        parts = urlsplit(url.strip())
        if parts.scheme.lower() not in {"http", "https"} or not parts.netloc:
            return None
        query = [
            (key, value)
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
            if key.lower() not in TRACKING_PARAMS and not key.lower().startswith("utm_")
        ]
        path = parts.path or "/"
        normalized = urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, urlencode(query), ""))
        return normalized
    except ValueError:
        return None


def is_pdf_url(url: str) -> bool:
    return urlsplit(url).path.lower().endswith(".pdf")


def safe_filename(url: str) -> str:
    name = Path(urlsplit(url).path).name or "document.pdf"
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:100]
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
    return f"{digest}_{name}"


def get_domain(url: str) -> str:
    return (urlsplit(url).hostname or "").lower().removeprefix("www.")


class WebsiteCollector:
    def __init__(
        self,
        start_url: str,
        output_dir: Path,
        delay: float = 1.0,
        max_pages: int = 300,
        max_depth: int = 5,
        allow_subdomains: bool = False,
        enable_ocr: bool = False,
    ) -> None:
        normalized_start = normalize_url(start_url)
        if not normalized_start:
            raise ValueError("Start URL must be a valid http:// or https:// URL.")
        self.start_url = normalized_start
        self.start_domain = get_domain(normalized_start)
        self.output_dir = output_dir
        self.pdf_dir = output_dir / "pdfs"
        self.db_path = output_dir / "knowledge_base.sqlite3"
        self.jsonl_path = output_dir / "documents.jsonl"
        self.delay = max(0.0, delay)
        self.max_pages = max(1, max_pages)
        self.max_depth = max(0, max_depth)
        self.allow_subdomains = allow_subdomains
        self.enable_ocr = enable_ocr
        self.last_request_at = 0.0

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.session = self._make_session()
        self.robots = self._load_robots()
        self.db = sqlite3.connect(self.db_path)
        self.db.execute("PRAGMA foreign_keys = ON")
        self._initialize_database()

    @staticmethod
    def _make_session() -> requests.Session:
        session = requests.Session()
        session.headers.update({
            "User-Agent": APP_NAME,
            "Accept": "text/html,application/pdf,application/xhtml+xml,*/*;q=0.8",
        })
        retry = Retry(
            total=3,
            connect=3,
            read=2,
            backoff_factor=1,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET", "HEAD"}),
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _load_robots(self) -> RobotFileParser:
        parser = RobotFileParser()
        robots_url = urljoin(self.start_url, "/robots.txt")
        parser.set_url(robots_url)
        try:
            response = self.session.get(robots_url, timeout=15)
            if response.status_code == 200:
                parser.parse(response.text.splitlines())
                logging.info("Loaded robots.txt: %s", robots_url)
            elif response.status_code == 404:
                parser.parse([])
                logging.info("No robots.txt found (404); continuing with normal crawl limits.")
            else:
                # Conservative: if robots rules cannot be read, do not crawl.
                parser.parse(["User-agent: *", "Disallow: /"])
                logging.warning("Could not read robots.txt (HTTP %s); crawl is blocked for safety.", response.status_code)
        except requests.RequestException as exc:
            parser.parse(["User-agent: *", "Disallow: /"])
            logging.warning("Could not access robots.txt; crawl is blocked for safety: %s", exc)
        return parser

    def _initialize_database(self) -> None:
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                source_url TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                source_type TEXT NOT NULL,
                description TEXT DEFAULT '',
                language TEXT DEFAULT '',
                content_hash TEXT NOT NULL,
                collected_at TEXT NOT NULL
            )
        """)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS crawl_errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL,
                error TEXT NOT NULL,
                occurred_at TEXT NOT NULL
            )
        """)
        self.db.commit()

    def _record_error(self, url: str, message: str) -> None:
        logging.warning("%s — %s", url, message)
        self.db.execute(
            "INSERT INTO crawl_errors(url, error, occurred_at) VALUES (?, ?, ?)",
            (url, message[:1000], utc_now()),
        )
        self.db.commit()

    def _allowed_domain(self, url: str) -> bool:
        host = get_domain(url)
        if not host:
            return False
        if host == self.start_domain:
            return True
        return self.allow_subdomains and host.endswith("." + self.start_domain)

    def _polite_wait(self) -> None:
        elapsed = time.monotonic() - self.last_request_at
        wait_for = self.delay - elapsed
        if wait_for > 0:
            time.sleep(wait_for)
        self.last_request_at = time.monotonic()

    def _get(self, url: str) -> Optional[requests.Response]:
        if not self.robots.can_fetch(self.session.headers["User-Agent"], url):
            logging.info("Skipped by robots.txt: %s", url)
            return None
        self._polite_wait()
        try:
            response = self.session.get(url, timeout=(10, 30), allow_redirects=True)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            self._record_error(url, f"Request failed: {exc}")
            return None

    @staticmethod
    def _html_to_document(url: str, html: str) -> tuple[Document, list[str]]:
        soup = BeautifulSoup(html, "lxml")
        title = clean_text(soup.title.get_text(" ", strip=True)) if soup.title else ""
        description_tag = soup.find("meta", attrs={"name": re.compile("^description$", re.I)})
        description = clean_text(description_tag.get("content", "")) if description_tag else ""
        language = ""
        if soup.html:
            language = str(soup.html.get("lang", "")).strip()

        for element in soup.select(
            "script, style, noscript, svg, canvas, iframe, nav, footer, header, aside, form, button"
        ):
            element.decompose()

        # Prefer the central content region, but fall back to body for simple websites.
        main = soup.select_one("main, article, [role='main']")
        text_root = main if main and len(main.get_text(" ", strip=True)) > 200 else (soup.body or soup)
        content = clean_text(text_root.get_text(separator="\n", strip=True))

        links: list[str] = []
        for anchor in soup.find_all("a", href=True):
            absolute = normalize_url(urljoin(url, str(anchor["href"])))
            if absolute:
                links.append(absolute)

        doc = Document(
            source_url=url,
            title=title or url,
            content=content,
            source_type="webpage",
            description=description,
            language=language,
            content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            collected_at=utc_now(),
        )
        return doc, links

    def _save_document(self, doc: Document) -> None:
        self.db.execute("""
            INSERT INTO documents (
                source_url, title, content, source_type, description,
                language, content_hash, collected_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_url) DO UPDATE SET
                title=excluded.title,
                content=excluded.content,
                source_type=excluded.source_type,
                description=excluded.description,
                language=excluded.language,
                content_hash=excluded.content_hash,
                collected_at=excluded.collected_at
        """, (
            doc.source_url, doc.title, doc.content, doc.source_type,
            doc.description, doc.language, doc.content_hash, doc.collected_at,
        ))
        self.db.commit()

    def _extract_pdf_text(self, pdf_path: Path) -> tuple[str, str]:
        pages: list[str] = []
        with fitz.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                page_text = clean_text(page.get_text("text"))
                if not page_text and self.enable_ocr:
                    try:
                        import pytesseract
                        from PIL import Image
                        from io import BytesIO

                        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                        image = Image.open(BytesIO(pixmap.tobytes("png")))
                        page_text = clean_text(pytesseract.image_to_string(image))
                    except ImportError as exc:
                        raise RuntimeError(
                            "OCR requested but optional packages are missing. Install: pip install pytesseract Pillow"
                        ) from exc
                    except Exception as exc:
                        logging.warning("OCR failed on %s page %s: %s", pdf_path.name, page_number, exc)
                if page_text:
                    pages.append(f"[Page {page_number}]\n{page_text}")
            metadata = pdf.metadata or {}
            pdf_title = clean_text(metadata.get("title") or "")
        return "\n\n".join(pages), pdf_title

    def _collect_pdf(self, url: str) -> None:
        if not is_pdf_url(url):
            return
        if not self._allowed_domain(url):
            logging.info("Skipped external PDF: %s", url)
            return
        response = self._get(url)
        if response is None:
            return
        try:
            content_type = response.headers.get("Content-Type", "").lower()
            if "pdf" not in content_type and not is_pdf_url(response.url):
                self._record_error(url, f"URL looked like a PDF but returned {content_type or 'unknown content type'}")
                return
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_PDF_BYTES:
                self._record_error(url, f"PDF exceeds {MAX_PDF_BYTES // (1024 * 1024)} MB limit")
                return
            if len(response.content) > MAX_PDF_BYTES:
                self._record_error(url, f"PDF exceeds {MAX_PDF_BYTES // (1024 * 1024)} MB limit")
                return
            filename = safe_filename(url)
            pdf_path = self.pdf_dir / filename
            pdf_path.write_bytes(response.content)
            try:
                content, metadata_title = self._extract_pdf_text(pdf_path)
            except Exception as exc:
                self._record_error(url, f"PDF extraction failed: {exc}")
                return
            if not content:
                self._record_error(url, "No extractable text found; scanned PDF may need OCR")
                return
            title = metadata_title or Path(urlsplit(url).path).name or "PDF document"
            doc = Document(
                source_url=url,
                title=title,
                content=content,
                source_type="pdf",
                description=f"Downloaded file: {filename}",
                content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                collected_at=utc_now(),
            )
            self._save_document(doc)
            logging.info("Saved PDF text: %s", url)
        finally:
            response.close()

    def _collect_page(self, url: str) -> list[str]:
        response = self._get(url)
        if response is None:
            return []
        try:
            final_url = normalize_url(response.url) or url
            if not self._allowed_domain(final_url):
                logging.info("Skipped redirect outside allowed domain: %s", final_url)
                return []
            content_type = response.headers.get("Content-Type", "").lower()
            if "html" not in content_type and "xhtml" not in content_type:
                logging.info("Skipped non-HTML resource: %s (%s)", final_url, content_type)
                return []
            doc, links = self._html_to_document(final_url, response.text)
            if len(doc.content) < 30:
                logging.info("Skipped page with very little text: %s", final_url)
            else:
                self._save_document(doc)
                logging.info("Saved webpage: %s (%s chars)", final_url, len(doc.content))
            return links
        finally:
            response.close()

    def export_jsonl(self) -> None:
        with self.db:
            rows = self.db.execute("""
                SELECT source_url, title, content, source_type, description,
                       language, content_hash, collected_at
                FROM documents ORDER BY source_type, source_url
            """).fetchall()
        with self.jsonl_path.open("w", encoding="utf-8") as file:
            for row in rows:
                record = dict(zip(
                    ["source_url", "title", "content", "source_type", "description",
                     "language", "content_hash", "collected_at"],
                    row,
                ))
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        logging.info("Exported %s documents to %s", len(rows), self.jsonl_path)

    def crawl(self) -> None:
        queue: deque[tuple[str, int]] = deque([(self.start_url, 0)])
        queued = {self.start_url}
        visited: set[str] = set()
        pages_collected = 0
        pdfs_seen: set[str] = set()

        while queue and pages_collected < self.max_pages:
            url, depth = queue.popleft()
            if url in visited:
                continue
            visited.add(url)
            if not self._allowed_domain(url):
                continue
            if is_pdf_url(url):
                if url not in pdfs_seen:
                    pdfs_seen.add(url)
                    self._collect_pdf(url)
                continue

            new_links = self._collect_page(url)
            pages_collected += 1
            if depth >= self.max_depth:
                continue
            for link in new_links:
                if link in visited or link in queued or not self._allowed_domain(link):
                    continue
                path_lower = urlsplit(link).path.lower()
                if any(path_lower.endswith(ext) for ext in SKIP_EXTENSIONS):
                    continue
                if is_pdf_url(link):
                    if link not in pdfs_seen:
                        pdfs_seen.add(link)
                        self._collect_pdf(link)
                else:
                    queued.add(link)
                    queue.append((link, depth + 1))

        self.export_jsonl()
        total = self.db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        errors = self.db.execute("SELECT COUNT(*) FROM crawl_errors").fetchone()[0]
        logging.info("Finished. Documents in database: %s | Logged errors: %s", total, errors)

    def close(self) -> None:
        self.db.close()
        self.session.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect public webpages and linked PDFs for a company chatbot knowledge base."
    )
    parser.add_argument("--start-url", required=False, help="Company website URL, e.g. https://example.com")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output folder (default: ./data)")
    parser.add_argument("--delay", type=float, default=1.0, help="Minimum seconds between HTTP requests")
    parser.add_argument("--max-pages", type=int, default=300, help="Maximum HTML pages to visit")
    parser.add_argument("--max-depth", type=int, default=5, help="Maximum link depth from the homepage")
    parser.add_argument("--allow-subdomains", action="store_true", help="Also crawl subdomains of the start domain")
    parser.add_argument("--ocr", action="store_true", help="OCR scanned PDF pages (requires Tesseract installed)")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    start_url = args.start_url or input("Enter company website URL (including https://): ").strip()
    collector: Optional[WebsiteCollector] = None
    try:
        collector = WebsiteCollector(
            start_url=start_url,
            output_dir=Path(args.output),
            delay=args.delay,
            max_pages=args.max_pages,
            max_depth=args.max_depth,
            allow_subdomains=args.allow_subdomains,
            enable_ocr=args.ocr,
        )
        collector.crawl()
        return 0
    except KeyboardInterrupt:
        logging.warning("Stopped by user. Data collected so far remains saved in SQLite.")
        if collector:
            collector.export_jsonl()
        return 130
    except (ValueError, sqlite3.Error) as exc:
        logging.error("Configuration or database error: %s", exc)
        return 2
    finally:
        if collector:
            collector.close()


if __name__ == "__main__":
    sys.exit(main())
