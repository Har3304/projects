# SEO Schema Maker

An automated tool built with **LangChain** and **Hugging Face (`Llama-3.1-8B-Instruct`)** that scrapes raw webpage text and transforms it into structured, production-ready SEO JSON-LD schemas. 

The pipeline implements a two-step validation process: a generation phase and a reflection/rectification phase to ensure bracket compliance and remove redundant elements.

## Features

- **Web Scraping:** Uses `Requests` and `BeautifulSoup4` to pull clean, visible text content from any given URL.
- **LLM-Powered Schema Generation:** Leverages Llama-3.1-8B via HuggingFace endpoints to analyze context and generate compliant schemas.
- **Self-Correction Pipeline:** Runs the raw schema through a dedicated "critic/error-checking" LLM prompt to fix bracket mismatches, JSON syntax errors, or duplicate keys before saving.
- **Colab Integration:** Utilizes `google.colab.userdata` for secure Hugging Face token management.

## Project Structure

```text
schema_maker/
├── schema_generator.py   # Main Python execution script containing pipeline logic
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
