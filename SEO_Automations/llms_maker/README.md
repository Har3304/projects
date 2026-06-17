# Website-to-llms.txt Generator

An automated pipeline designed to scrape content from a website using its XML sitemaps, summarize individual pages with an LLM, and synthesize the collective intelligence into a highly optimized `llms.txt` file. This file is engineered explicitly to assist LLMs, RAG applications, and autonomous AI agents in discovering, understanding, and accurately citing a brand in Generative Engine Optimization (GEO) environments.

## Features

* **Sitemap Parsing:** Automatically extracts operational links from both page and post XML sitemaps.
* **Intelligent HTML Scraping:** Strips boilerplate styling, handles image assets gracefully, and cleanses text for LLM ingestion.
* **Hierarchical Summarization:** Uses `meta-llama/Llama-3.1-8B-Instruct` via LangChain and Hugging Face to generate highly concentrated summaries of individual pages.
* **Structured Synthesis:** Consolidates individual page contexts into a unified, structured markdown framework customized for AI crawlers (`llms.txt`), mapping site entity profiles, leadership data, contact matrices, and high-signal URLs.

## Prerequisites

The script is primarily configured to run in a **Google Colab** environment and relies on a Hugging Face API token for inference.

### Hugging Face Setup
Ensure you have a Hugging Face account and an access token with read permissions (`HF_TOKEN_READ`). In Google Colab, add this token to your notebook secrets panel.

## Installation & Setup

1. Clone or download this repository containing your script.
2. Install the necessary dependencies:

```bash
pip install -r requirements.txt
