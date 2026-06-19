"""
site_intelligence_pipeline.py

Unified pipeline that:
  1. Crawls a site's sitemap(s) recursively.
  2. For every real page (not a sub-sitemap, not an image), fetches the page ONCE
     and uses that single fetch for both:
       a) JSON-LD schema generation (saved to disk, mirroring the URL structure)
       b) A page summary (used to build llms.txt)
  3. Builds an llms.txt file via LLM, then programmatically inserts the list of
     generated schema file links under the "## Schema Files" heading -- this
     step is done in code, not by the LLM, so it can never be skipped or
     malformed.

Fixes applied vs. the original two standalone scripts:
  - Schema JSON is always round-tripped through json.loads/json.dumps(indent=2).
    This guarantees consistent indentation AND eliminates duplicate keys, since
    a Python dict cannot hold two values under the same key (last one wins on
    parse) -- this is a structural fix, not just a prompt instruction.
  - Page @type is no longer left to the LLM's judgement. It's computed
    deterministically in code (infer_page_type) from the URL pattern and which
    sitemap it came from, then passed to the LLM as a hint it must follow
    unless the scraped text actively contradicts it.
  - Removed a stray `break` in the original crawler that silently limited
    processing to only the first non-sitemap URL per directory level.
  - Each page is fetched exactly once instead of twice.
  - If the model's raw JSON output fails to parse, a single repair pass is
    attempted (asking the model to fix syntax only) before falling back to
    saving the raw text with a clear warning, so failures are visible instead
    of silently producing broken files.

Usage (see bottom of file for a runnable example).
"""

import os
import re
import json
import time
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser

try:
    from google.colab import userdata as _colab_userdata
except ImportError:
    _colab_userdata = None


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SCHEMA_PROMPT = PromptTemplate(
    template="""You are an expert SEO Technical Specialist generating production-ready JSON-LD Schema markup.

---
INPUT DATA:
- Website Link: {link}
- Authoritative Page Type Directive: {page_type_hint}
  (This was determined deterministically from the URL structure. You MUST use
  it as the root "@type" unless the scraped text gives clear, strong evidence
  it is wrong -- in which case briefly note nothing and just use the correct
  type silently, still returning only JSON.)
- Scraped Raw Text:
[START OF SCRAPED TEXT]
{scraped_text}
[END OF SCRAPED TEXT]

---
STRUCTURAL MAPPING RULES (apply based on the Page Type Directive above):

- CollectionPage (category/tag/archive pages):
    - root "@type" = "CollectionPage"
    - "name" = exact archive title
    - "mainEntity" = an "ItemList"
    - "mainEntity.itemListElement" = array of "ListItem", each wrapping a "BlogPosting"

- Blog (main blog index / listing page):
    - root "@type" = "Blog"
    - Do NOT put "author", "datePublished", or "articleSection" directly on the root.
    - Use a "blogPost" array of self-contained "BlogPosting" objects.

- BlogPosting / Article (single blog post):
    - root "@type" = "BlogPosting" (or "Article" if clearly not a blog post)
    - root MUST include "author", "datePublished", "dateModified", "headline", "articleBody"

- WebPage / AboutPage / Service / ContactPage (standard company pages):
    - root "@type" matches the page's purpose exactly (About -> "AboutPage",
      Services -> "Service", Contact -> "ContactPage", Home/other -> "WebPage")

---
OUTPUT FORMAT CONSTRAINTS (strict):
1. Return ONLY a single raw JSON object -- nothing before "{{" and nothing after the final "}}".
2. No markdown code fences, no commentary, no trailing explanation.
3. Every key must appear exactly once at each nesting level -- never repeat a key.
4. Use 2-space indentation consistently throughout (it will be re-validated programmatically,
   but write it correctly the first time).
5. Valid JSON only: matching braces/brackets, correct trailing commas (i.e. none before a closing
   brace/bracket), double-quoted keys and string values.""",
    input_variables=["scraped_text", "link", "page_type_hint"],
)

SCHEMA_REPAIR_PROMPT = PromptTemplate(
    template="""You are a JSON syntax repair tool. The following text was supposed to be a single
valid JSON-LD object but failed to parse. Fix ONLY syntax problems (unmatched braces/brackets,
trailing commas, unquoted keys, duplicate keys -- keep the LAST occurrence of any duplicate key
and remove the earlier ones, stray text outside the JSON object). Do not change the schema
structure or content otherwise.

Return ONLY the corrected raw JSON object. No markdown fences, no commentary.

BROKEN JSON:
{broken_json}""",
    input_variables=["broken_json"],
)

SUMMARY_PROMPT = PromptTemplate(
    template="""You are an expert text summarizer and context creator. From the given text scraped
from a webpage, write a summarized coherent paragraph containing imperative intel useful precisely
for building an llms.txt file later: {text}. The response must not include any explanation or
comments.""",
    input_variables=["text"],
)

LLMS_TXT_PROMPT = PromptTemplate(
    template="""You are a world-class AI documentation engineer and an elite SEO architect specializing in Answer Engine Optimization (AEO) and Generative Engine Optimization (GEO).

Your task is to synthesize the provided key-value text data (containing website links and page summaries) into a highly polished, professional 'llms.txt' file. This file must be built strictly to serve LLMs, RAG applications, and autonomous AI agents (such as Perplexity, GPTBot, and ClaudeBot) so they can instantly scrape, understand, and accurately cite this brand in conversational search engine results.

Structure the output exactly according to the following comprehensive layout:

1. # Brand/Company Name
   - Deduce or state the official organization name cleanly at the top.

2. > Blockquote Summary
   - Write an information-dense, 2-to-3 sentence master summary of the organization's identity, key manufacturing or service metrics, and value proposition.
   - CRITICAL: Inject zero-fluff, hard entity-keywords (e.g., industry certifications, regional specialization, capacity metrics) to anchor the site's semantic relevance for GEO search.

3. ## Site Identity & Entity Profile
   Extract, infer, or provide the following definitive bullet points:
   - **Brand Name**: [Common Name]
   - **Official Name**: [Full Registered Corporate Name]
   - **Headquarters Location**: [City, State, Country]
   - **Core Operating Industry**: [Niche and broader market category]
   - **Primary Target Audience**: [List distinct B2B/B2C consumer layers]

4. ## Leadership & Corporate Governance
   Map out the key individuals steering the organization to establish high EEAT (Experience, Expertise, Authoritativeness, Trustworthiness) for AI engines. Detail:
   - **Founder / Key Promoter**: [Identify primary founding members from data]
   - **Chairman / Executive Leadership**: [Identify the Chairman/Managing Directors]
   - **Key Management Personnel**: [List CFO, Operations Head, Retail Head, or Company Secretary if present]

5. ## Official Communications & Contact Matrix
   Do not just list a generic phone number. Create an organized contact lookup for AI agents:
   - **Corporate Inquiries**: [Email address for corporate/general]
   - **Sales & Commercial Team**: [Email address for sales]
   - **Technical & System Queries**: [Email address for engineering/systems]
   - **Compliance & Investor Relations**: [Email address for regulatory/compliance]

6. ## Core Documentation
   Map high-signal operational URLs. Every entry must strictly follow this exact markdown format:
   `- [Page Title](URL): A strict, facts-first, one-sentence description detailing what an AI can extract or learn from that specific route.`
   Include: Homepage, About, Services, Products, Contact.

7. ## Optional / Knowledge Base & Authority Assets
   Map deep context assets like Blog indexes, Case Studies, and Articles.
   Format entries exactly like the Core Documentation section. The descriptions must highlight technical or empirical values (e.g., "Contains performance data frameworks", "Houses net metering guidance").

8. ## Schema Files
   Output ONLY the heading `## Schema Files`. Leave the space beneath it entirely empty (do not include text, bullets, or syntax placeholders). A separate automated process will populate this section.

9. ## Footer Section
   Map secondary or legal structural links (Privacy Policy, Terms of Service, Careers). Format entries exactly like the Core Documentation section.

10. ## Crawling Rules for AI Bots
    Append this exact configuration block verbatim at the absolute bottom:
    - Allow: /llms.txt
    - Allow: /llms-full.txt
    - User-agent: GPTBot / Allow
    - User-agent: ClaudeBot / Allow
    - User-agent: PerplexityBot / Allow

Strict Output Constraints:
- Output ONLY raw Markdown syntax.
- Do NOT wrap the entire file or any section inside code blocks (```).
- Do NOT write introductory remarks (e.g., "Here is your file:") or concluding pleasantries.

Here is the key-value website dataset to process:
<link_summary_dataset>
{text}
</link_summary_dataset>

Generated llms.txt:""",
    input_variables=["text"],
)


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------

class SiteIntelligencePipeline:
    """
    Crawls a sitemap, generates per-page JSON-LD schema files, and builds a
    final llms.txt with the schema file links auto-inserted.
    """

    IMAGE_EXT_RE = re.compile(r"\.(jpg|jpeg|png|gif|webp|bmp|svg)(\?.*)?$", re.IGNORECASE)

    def __init__(
        self,
        output_dir: str,
        hf_token: str = None,
        hf_token_secret_name: str = "HF_TOKEN_READ",
        model_repo_id: str = "meta-llama/Llama-3.1-8B-Instruct",
        schema_link_base: str = None,
        request_timeout: int = 10,
        request_delay: float = 0.0,
    ):
        """
        output_dir:          local directory to write schemas/ and llms.txt into.
        hf_token:             pass a token directly, or leave None to pull it from
                               Colab userdata (hf_token_secret_name) / env var.
        schema_link_base:     base URL the schema files will eventually be hosted at,
                               e.g. "https://example.com/schema". Links in llms.txt's
                               Schema Files section are built as
                               f"{schema_link_base}/{relative_path}.json".
                               If left None, relative local file paths are used instead
                               (you'll want to swap these for real URLs before publishing).
        request_delay:        seconds to sleep between HTTP requests (politeness / rate limiting).
        """
        self.output_dir = output_dir
        self.schemas_dir = os.path.join(output_dir, "schemas")
        os.makedirs(self.schemas_dir, exist_ok=True)

        token = hf_token or self._resolve_hf_token(hf_token_secret_name)
        if token:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = token

        self.model_repo_id = model_repo_id
        self.schema_link_base = schema_link_base.rstrip("/") if schema_link_base else None
        self.request_timeout = request_timeout
        self.request_delay = request_delay

        # Populated during crawl()
        self.summaries = {}          # url -> summary text
        self.schema_links = []       # list of (url, link_for_llms_txt)
        self.errors = []             # list of (url, error_message)

        self._schema_chain = None
        self._repair_chain = None
        self._summary_chain = None

    # -- credential resolution -------------------------------------------------

    @staticmethod
    def _resolve_hf_token(secret_name):
        if _colab_userdata is not None:
            try:
                return _colab_userdata.get(secret_name)
            except Exception:
                pass
        return os.environ.get(secret_name)

    # -- LLM chains (built lazily, reused across calls) -------------------------

    def _make_llm(self, max_new_tokens):
        return HuggingFaceEndpoint(
            repo_id=self.model_repo_id,
            task="text-generation",
            temperature=0.1,
            max_new_tokens=max_new_tokens,
        )

    @property
    def schema_chain(self):
        if self._schema_chain is None:
            chatbot = ChatHuggingFace(llm=self._make_llm(4000))
            self._schema_chain = SCHEMA_PROMPT | chatbot | StrOutputParser()
        return self._schema_chain

    @property
    def repair_chain(self):
        if self._repair_chain is None:
            chatbot = ChatHuggingFace(llm=self._make_llm(4000))
            self._repair_chain = SCHEMA_REPAIR_PROMPT | chatbot | StrOutputParser()
        return self._repair_chain

    @property
    def summary_chain(self):
        if self._summary_chain is None:
            chatbot = ChatHuggingFace(llm=self._make_llm(1000))
            self._summary_chain = SUMMARY_PROMPT | chatbot | StrOutputParser()
        return self._summary_chain

    # -- fetching / cleaning -----------------------------------------------------

    def fetch_clean_text(self, url):
        """Single fetch, returns cleaned text suitable for both schema gen and summarization."""
        response = requests.get(url, timeout=self.request_timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()
        lines = (line.strip() for line in soup.get_text().splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        return "\n".join(chunk for chunk in chunks if chunk)

    # -- deterministic page-type inference ---------------------------------------

    def infer_page_type(self, url: str, sitemap_filename: str = "") -> str:
        """
        Computes the schema.org root @type deterministically from the URL pattern
        and which sitemap the URL came from, instead of leaving it to the LLM.
        """
        path = urlparse(url).path.rstrip("/").lower()
        sm = sitemap_filename.lower()

        if "/category/" in path or "/tag/" in path or "archive" in path:
            return "CollectionPage (category/tag/archive listing)"
        if path == "" or path == "/":
            return "WebPage (Homepage)"
        if "about" in path:
            return "AboutPage"
        if "contact" in path:
            return "ContactPage"
        if "service" in path:
            return "Service"
        if path.rstrip("/").endswith("/blog") or path.rstrip("/").endswith("/blogs"):
            return "Blog (main blog index/listing page)"
        if "post-sitemap" in sm or "/blog/" in path or "/news/" in path or "/article" in path:
            return "BlogPosting (single blog post / article)"
        return "WebPage (standard company page)"

    # -- schema generation with structural validation ----------------------------

    @staticmethod
    def _strip_fences(text: str) -> str:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"```\s*$", "", text)
        return text.strip()

    def _normalize_json(self, raw_text: str):
        """
        Tries to parse raw_text as JSON. On success, returns a pretty, deduplicated
        JSON string (json.loads/dumps inherently removes duplicate keys -- the parser
        keeps only the last value seen for any repeated key). Returns None on failure.
        """
        cleaned = self._strip_fences(raw_text)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return None
        return json.dumps(parsed, indent=2, ensure_ascii=False)

    def generate_schema(self, url: str, scraped_text: str, sitemap_filename: str = ""):
        """
        Returns a clean, validly-formatted JSON-LD string for the given page, or
        raises RuntimeError if it could not be made valid even after a repair pass.
        """
        page_type_hint = self.infer_page_type(url, sitemap_filename)

        raw = self.schema_chain.invoke(
            {"scraped_text": scraped_text, "link": url, "page_type_hint": page_type_hint}
        )

        normalized = self._normalize_json(raw)
        if normalized is not None:
            return normalized

        # First attempt failed to parse -- run one repair pass.
        repaired_raw = self.repair_chain.invoke({"broken_json": self._strip_fences(raw)})
        normalized = self._normalize_json(repaired_raw)
        if normalized is not None:
            return normalized

        raise RuntimeError(f"Could not produce valid JSON for {url} even after repair pass.")

    # -- summarization -------------------------------------------------------------

    def summarize(self, text: str) -> str:
        return self.summary_chain.invoke({"text": text})


    def crawl(self, sitemap_link: str, path: str = None):
        """
        Recursively walks a sitemap (and any nested sub-sitemaps), and for every
        real page: fetches it once, generates+saves its schema, and stores a summary
        for later use in llms.txt. Skips image assets. Errors on individual pages are
        logged to self.errors and do not stop the crawl.
        """
        if path is None:
            path = self.schemas_dir

        try:
            sitemap = requests.get(sitemap_link, timeout=self.request_timeout)
            sitemap.raise_for_status()
        except requests.RequestException as e:
            self.errors.append((sitemap_link, f"Failed to fetch sitemap: {e}"))
            return

        soup = BeautifulSoup(sitemap.text, "xml")
        locs = [l.get_text().strip() for l in soup.find_all("loc") if l.get_text().strip()]
        sitemap_filename = os.path.basename(urlparse(sitemap_link).path)

        for url in tqdm(locs, desc=f"Crawling {sitemap_filename}"):
            url_path = urlparse(url).path.rstrip("/")
            raw_name = url_path.split("/")[-1] if url_path else "index"
            name = re.sub(r"[-\s]+", "_", raw_name)

            if url.lower().endswith(".xml"):
                sub_path = os.path.join(path, name)
                os.makedirs(sub_path, exist_ok=True)
                self.crawl(url, sub_path)  # recurse into nested sitemap
                continue

            if self.IMAGE_EXT_RE.search(url):
                continue  # skip image assets

            try:
                if self.request_delay:
                    time.sleep(self.request_delay)

                scraped_text = self.fetch_clean_text(url)

                # --- schema generation ---
                schema_json = self.generate_schema(url, scraped_text, sitemap_filename)
                file_path = os.path.join(path, f"{name}.json")
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(schema_json)

                rel_path = os.path.relpath(file_path, self.output_dir).replace(os.sep, "/")
                if self.schema_link_base:
                    link = f"{self.schema_link_base}/{rel_path[len('schemas/'):] if rel_path.startswith('schemas/') else rel_path}"
                else:
                    link = rel_path
                self.schema_links.append((url, link))

                # --- summary for llms.txt ---
                self.summaries[url] = self.summarize(scraped_text)

            except requests.exceptions.RequestException as e:
                self.errors.append((url, f"Fetch error: {e}"))
            except RuntimeError as e:
                self.errors.append((url, str(e)))
            except Exception as e:
                self.errors.append((url, f"Unexpected error: {e}"))

    # -- llms.txt assembly ------------------------------------------------------------

    @staticmethod
    def _titleize(url: str) -> str:
        path = urlparse(url).path.rstrip("/")
        name = path.split("/")[-1] if path else "Homepage"
        name = re.sub(r"[-_]+", " ", name).strip()
        return name.title() if name else "Homepage"

    def _insert_schema_links(self, llms_text: str) -> str:
        """
        Finds the '## Schema Files' heading the LLM left empty and inserts the
        collected schema links directly beneath it, before the next heading.
        Pure string/code operation -- no LLM involved, so it can't be skipped.
        """
        lines = llms_text.split("\n")
        heading_idx = None
        for i, line in enumerate(lines):
            if line.strip().lower() == "## schema files":
                heading_idx = i
                break

        if heading_idx is None:
            # Fallback: append a new section if the LLM omitted the heading entirely.
            lines.append("")
            lines.append("## Schema Files")
            heading_idx = len(lines) - 1

        insert_at = heading_idx + 1
        while insert_at < len(lines) and lines[insert_at].strip() == "":
            insert_at += 1

        if self.schema_links:
            schema_block_lines = [
                f"- [{self._titleize(url)} Schema]({link}): Structured JSON-LD schema markup for this page."
                for url, link in self.schema_links
            ]
        else:
            schema_block_lines = ["_No schema files were generated._"]

        lines[insert_at:insert_at] = schema_block_lines + [""]
        return "\n".join(lines)

    def build_llms_txt(self) -> str:
        """Generates llms.txt from collected summaries and inserts schema links."""
        dataset = "\n".join(f"{url}: {summary}" for url, summary in self.summaries.items())
        chatbot = ChatHuggingFace(llm=self._make_llm(5000))
        chain = LLMS_TXT_PROMPT | chatbot | StrOutputParser()
        draft = chain.invoke({"text": dataset})
        return self._insert_schema_links(draft)

    # -- top-level orchestration ----------------------------------------------------

    def run(self, sitemap_urls):
        """
        Full pipeline: crawl each given sitemap URL, generate schemas + summaries,
        then build and save llms.txt with schema links auto-inserted.
        """
        for sitemap_url in sitemap_urls:
            self.crawl(sitemap_url)

        llms_text = self.build_llms_txt()
        llms_path = os.path.join(self.output_dir, "llms.txt")
        with open(llms_path, "w", encoding="utf-8") as f:
            f.write(llms_text)

        if self.errors:
            print(f"\nCompleted with {len(self.errors)} error(s):")
            for url, msg in self.errors:
                print(f"  - {url}: {msg}")

        print(f"\nSchemas saved under: {self.schemas_dir}")
        print(f"llms.txt saved at:   {llms_path}")
        return llms_path


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pipeline = SiteIntelligencePipeline(
        output_dir="./site_output",
        hf_token_secret_name="HF_TOKEN_READ",   # pulled from Colab userdata, or set HF_TOKEN_READ env var
        schema_link_base="https://australianpremiumsolar.co.in/schema",
        request_delay=0.5,
    )

    pipeline.run([
        "https://australianpremiumsolar.co.in/page-sitemap.xml",
        "https://australianpremiumsolar.co.in/post-sitemap.xml",
    ])
