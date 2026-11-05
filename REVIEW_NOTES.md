# Review of the supplied AUST scraper

## What the original project did well

- Used a breadth-first queue to discover internal pages.
- Used `robots.txt` checks and a delay between requests.
- Stored webpage text and source URLs in SQLite.
- Downloaded linked PDFs and used PyMuPDF/Tesseract-based scripts for extraction.
- Kept the source URL, which is valuable for provenance and citations in a chatbot.

## Issues addressed in this revised starter

1. **Incomplete requirements:** `aust.py` imports `pandas` and `ratelimit`, but neither was in `requirements.txt`. The revised collector removes those unnecessary dependencies.
2. **Command-line options:** the original database path was opened before parsing `--db`, so the option did not affect the active database. Its rate-limit decorator also captured the original delay before CLI parsing. The revised collector wires settings into initialization.
3. **README/script naming mismatch:** the README referred to an underscore filename while the supplied script used hyphens. The revised project uses one clear `scraper.py` entry point.
4. **PDF handling:** the old flow used separate scripts and a hard-coded Windows Tesseract path. The revised flow extracts text-based PDFs directly and makes OCR optional, without a hard-coded install path.
5. **PDF filename collisions:** two different URLs can have the same basename. The revised collector adds a short URL hash to downloaded filenames.
6. **Crawler hygiene:** the revised collector normalizes URLs, removes fragments and common tracking parameters, restricts crawling to the chosen domain, checks content types, sets timeouts, and records errors.
7. **RAG-ready export:** the revised collector writes both SQLite and JSONL records with source URL, title, type, language, content hash, and collection time.

## Important limitations

- This is a starter collector, not a guarantee that every website page will be captured. JavaScript-rendered pages may need a browser-based crawler later.
- HTML text extraction is generic. Important pages should be manually reviewed, especially if the site has complex layouts.
- Scanned PDF OCR is optional and requires Tesseract installed separately.
- The collector does not scrape private social media content or bypass platform restrictions.
- The bundled original database was approximately 59.5 MiB and contained 1,079 page rows, 94 PDF records, and 88 table records. That is a useful existing AUST dataset, but it was not fully audited for accuracy or completeness.
