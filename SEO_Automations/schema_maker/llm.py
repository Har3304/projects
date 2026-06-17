import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from pydantic import BaseModel, Field
from google.colab import userdata
import requests
from bs4 import BeautifulSoup

hf_token = userdata.get('HF_TOKEN_FINE')
os.environ['HUGGINGFACEHUB_API_TOKEN'] = hf_token
def make_schema(link):
  try:
    response = requests.get(link)
    response.raise_for_status() # Raise an exception for HTTP errors
    soup = BeautifulSoup(response.text, 'html.parser')
    scraped_text_str = soup.get_text() # Now get_text() is called on BeautifulSoup object
  except requests.exceptions.RequestException as e:
    print(f"Error fetching or parsing {link}: {e}")
    scraped_text_str = "" # Fallback to empty string or handle as appropriate

  prompt = PromptTemplate(
      template="""You are an expert SEO specialist. You are given a task to create page or post schema from scraped text of a website page link{link} based on of what type it is (page/blog post): {scraped_text}.\n
                  No repeatition of keyword:value. Make sure to scrutinize the result for errors before giving result. Check for mismatch of curly brackets and make it production quality. Do not generate anything else other than the requested .json format.""",
      input_variables = ['scraped_text', 'link'])

  llm = HuggingFaceEndpoint(repo_id = 'meta-llama/Llama-3.1-8B-Instruct',
                            task = 'text-generation',
                            temperature = 0.1,
                            max_new_tokens=3000)

  ChatBot = ChatHuggingFace(llm=llm)
  chain = prompt | ChatBot

  result = chain.invoke({'scraped_text':scraped_text_str,
                        'link':link})
  return result


def check_for_errors(result):
  prompt = PromptTemplate(
      template="""You are an expert 20 years experienced SEO specialist. You are given task of rectifying a schema generated earlier by a LLM: {result}.\n 
                  No repeatition of keyword:value. Make sure to scrutinize the result for errors before giving result. Check for mismatch of curly brackets and make it production quality. Do not generate anything else other than the requested .json format.""",
      input_variables = ['result'])  
  repo_id = 'meta-llama/Llama-3.1-8B-Instruct'

  llm = HuggingFaceEndpoint(repo_id = repo_id,
                            task='text-generation',
                            temperature=0.1,
                            max_new_tokens=3000)
  ChatBot = ChatHuggingFace(llm=llm)
  chain = prompt | ChatBot
  result = chain.invoke({'result': result})
  return result

with open('example_format.json', 'w') as f:
    f.write(check_for_errors(make_schema('http://australianpremiumsolar.co.in/aps-supporting-make-in-india-campaign-benefits-of-using-locally-made-solar-panels-for-your-home/')).content)
