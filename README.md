# Aqlithos Website & PDF Data Collector

A small Python collector for preparing a company's public website content and linked PDFs for a future RAG chatbot.

It collects HTML pages on the same domain, extracts readable text and basic metadata, downloads linked PDFs, extracts their text with PyMuPDF, and saves everything in SQLite plus JSONL.

**This is the data-collection stage only.** It does not create embeddings, use a vector database, or call an LLM API. No API key is needed at this stage.

## 1. Requirements

- Python 3.10 or newer
- Internet access
- Permission to collect the website content

## 2. Install

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

# Install the core packages
pip install -r requirements.txt
```

Optional OCR support for scanned/image-only PDFs:

```bash
pip install -r requirements-ocr.txt
```

OCR also requires the **Tesseract OCR application** to be installed on the computer and available on PATH. If Tesseract is not installed, do not use `--ocr`; ordinary text-based PDFs will still work.

## 3. Run for Aqlithos

Replace the example URL with the verified official Aqlithos website URL:

```bash
python scraper.py --start-url https://YOUR-AQLITHOS-DOMAIN --max-pages 300 --max-depth 5 --delay 1.5
```

Optional OCR for scanned PDFs:

```bash
python scraper.py --start-url https://YOUR-AQLITHOS-DOMAIN --ocr
```

Useful options:

```text
--start-url          Starting website URL
--output             Output directory (default: ./data)
--max-pages          Maximum HTML pages (default: 300)
--max-depth          Link depth from homepage (default: 5)
--delay              Minimum delay between requests (default: 1 second)
--allow-subdomains   Also visit subdomains of the starting domain
--ocr                OCR image-only PDF pages (Tesseract required)
```

## 4. Output

After a successful run, the `data/` folder contains:

```text
data/
├── knowledge_base.sqlite3   # searchable structured document store
├── documents.jsonl          # one JSON record per webpage/PDF; useful for RAG ingestion
└── pdfs/                    # downloaded PDF originals
```

Every document record contains:

- `source_url`: original webpage or PDF URL
- `title`: page title or PDF metadata/file name
- `content`: extracted text
- `source_type`: `webpage` or `pdf`
- `description`: meta description or source note
- `language`: HTML language attribute when available
- `content_hash`: SHA-256 of extracted text
- `collected_at`: UTC collection timestamp

The SQLite database also stores request and extraction errors in `crawl_errors`.

## 5. Important collection rules

- The collector checks `robots.txt` and stops if the rules cannot be read.
- It stays on the starting domain by default. Use `--allow-subdomains` only when those subdomains are part of the company website and should be collected.
- It uses a delay between requests, retries temporary HTTP errors, and limits each PDF to 30 MB.
- It skips common image, video, office-document, and archive file types. This first version extracts PDFs only.
- Social media sites often restrict automated scraping and may require official APIs or account exports. For the chatbot, collect approved company profile details and post text separately; do not try to bypass login, rate limits, or platform restrictions.
- Review the output before using it in a public chatbot. Remove confidential, personal, obsolete, duplicate, or unapproved information.

## 6. Suggested next step

Review `documents.jsonl`, correct company-specific information, and then build the next stage: text chunking, embeddings, ChromaDB storage, retrieval, and the LLM answer function.
