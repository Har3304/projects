import requests
from google.colab import userdata
from bs4 import BeautifulSoup
from tqdm import tqdm
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from google.colab import userdata
import os

hf_token = userdata.get('HF_TOKEN_READ')
os.environ['HUGGINGFACEHUB_API_TOKEN'] = hf_token




sitemap_urls = ['https://australianpremiumsolar.co.in/page-sitemap.xml', 'http://australianpremiumsolar.co.in/post-sitemap.xml'] # Example page-sitemap and post-sitemap links
all_links = []

for sitemap_url in sitemap_urls:
  try:
    response = requests.get(sitemap_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'xml')
    loc_tags = soup.find_all('loc')
    all_links.extend([loc.text for loc in loc_tags])
  except requests.exceptions.RequestException as e:
    print(f"Error fetching sitemap {sitemap_url}: {e}")

links = all_links
print(links)
def summarize(text):
  prompt = PromptTemplate(
      template="""You are an expert text summarizer and context creator. From given text scraped from a webpage, make a summarized coherent paragraph containing imperative intel useful precisley to create llms.txt later: {text}. The response must not include any explanation or comments.""",
      input_variables = ['text'])
  llm = HuggingFaceEndpoint(repo_id = 'meta-llama/Llama-3.1-8B-Instruct',
                            task = 'text-generation',
                            temperature = 0.1,
                            max_new_tokens = 1000)
  ChatBot = ChatHuggingFace(llm=llm)
  chain = prompt | ChatBot
  response = chain.invoke({'text':text})
  return response.content
summaries = {}
for l in tqdm(links):
  if l.endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp')):
    print(f"Skipping image file: {l}")
    continue

  try:
    response = requests.get(l)
    response.raise_for_status()
    response = BeautifulSoup(response.text, 'html.parser')
    response = str(response.get_text()).strip(' ').split('\n')
    response = [r.strip() for r in response if not r.isspace()]
    response = ' '.join(response)
    summaries[l]=summarize(response)
  except requests.exceptions.RequestException as e:
    print(f"Error fetching or processing {l}: {e}")
  except Exception as e:
    print(f"An unexpected error occurred for {l}: {e}")

summary=''
for v in summaries.values():
  summary=summary+v+'\n'
print(summary)
hf_token = userdata.get('HF_TOKEN_READ')
os.environ['HUGGINGFACEHUB_API_TOKEN'] = hf_token

def make_llms(lis: list):
  text = ' '.join(lis)
  prompt = PromptTemplate(
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
   Output ONLY the heading `## Schema Files`. Leave the space beneath it entirely empty (do not include text, bullets, or syntax placeholders).

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
    input_variables=['text'])
  llm = HuggingFaceEndpoint(repo_id = 'meta-llama/Llama-3.1-8B-Instruct',
                              task = 'text-generation',
                            temperature = 0.1,
                            max_new_tokens = 5000)
  ChatBot = ChatHuggingFace(llm=llm)
  chain = prompt | ChatBot
  result = chain.invoke({'text':summary})
  return result
with open('llms.txt', 'w') as f:
  f.write(make_llms(response).content)
