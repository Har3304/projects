# Site Intelligence Pipeline

A unified, production-grade Python pipeline designed to scrape an entire website via its sitemaps recursively, generate syntactically verified JSON-LD schema files for every page, and compile an AI-optimized `llms.txt` file designed for modern Answer Engine Optimization (AEO) and Generative Engine Optimization (GEO).

---

## Features

* **Recursive Sitemap Crawling:** Automatically navigates nested sitemaps while cleanly ignoring image assets.
* **Single-Fetch Efficiency:** Fetches every URL exactly once, parsing it simultaneously for both schema synthesis and content summarization.
* **Deterministic Type Hinting:** Evaluates page structure (URLs and sitemap sources) in native Python code to inject precise schema `@type` directives, restricting structural decisions before passing context to the LLM.
* **Automatic JSON-LD Normalization & Repair:** Guarantees strict layout standards by passing outputs through a localized parsing pass to deduplicate keys and automatically resolve syntax issues (e.g., trailing commas, missing braces).
* **Robust `llms.txt` Insertion:** Programmatically injects relative or absolute paths into the final layout using Python, guaranteeing structural integrity under the `## Schema Files` markdown heading.

---

## Installation & Setup

### 1. Clone or Move the Script
Ensure `site_intelligence_pipeline.py` is present in your working directory.

### 2. Install Dependencies
Install all required libraries using the provided `requirements.txt`:


pip install -r requirements.txt
## Technical Architecture



The workflow processes data through distinct stages to balance network efficiency with structural precision:

[Sitemap Link] ──> Recursive Crawler ──> Unique URLs ──> Single HTTP Fetch
│
┌─────────────────────────────────────────┴─────────────────────────────────────────┐
▼                                                                                   ▼
[Deterministic Type Hint]                                                                [Cleaned Body Text]
│                                                                                   │
└───────────────────────────>  SCHEMA_PROMPT  ──────────────────────────────────────┤
│                                                │
▼                                                ▼
HuggingFace Endpoint                                SUMMARY_PROMPT
(Schema Gen & Repair)                                      │
│                                                ▼
▼                                       HuggingFace Endpoint
Valid, Sanitized JSON-LD                             (Page Summary)
│                                                │
▼                                                ▼
Saved to Disk Structure                           LLMS_TXT_PROMPT
│                                                │
│                                                ▼
└────────────────────────────────────────> [llms.txt Assembly]

