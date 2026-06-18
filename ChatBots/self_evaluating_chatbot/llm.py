import os
import pickle
import pandas as pd
from google.colab import userdata
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.chat_history import InMemoryChatMessageHistory


pd.set_option('display.max_colwidth', None)

class ChatBotLlama():
    def __init__(self, bot_id, repo_id='meta-llama/Llama-3.1-8B-Instruct',
                 temperature=0.1, max_new_tokens=2000):
        
        self.name = f"{repo_id}_{bot_id}_{temperature}"
        
        
        tf_token = userdata.get('HF_TOKEN_READ')
        os.environ['HUGGINGFACEHUB_API_TOKEN'] = tf_token
        
        self.repo_id = repo_id
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        
        self.endpoint_llm = HuggingFaceEndpoint(
            repo_id=self.repo_id,
            task='text-generation',
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens
        )
        self.llm = ChatHuggingFace(llm=self.endpoint_llm)
        
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful and witty AI assistant."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        self.chain = self.prompt | self.llm
        self.store = {}

    def _get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    def invoke(self, user_input: str, session_id: str = 'default_session'):
        with_message_history_runnable = RunnableWithMessageHistory(
            self.chain,
            self._get_session_history, 
            input_messages_key='input',
            history_messages_key='history'
        )

        config = {'configurable': {'session_id': session_id}}
        response = with_message_history_runnable.invoke(
            {'input': user_input},
            config=config
        )
        
        
        dic = {'session_id': session_id, f'{self.name}': 1, 'response': response.content}
        df_new = pd.DataFrame([dic])
        
        if os.path.exists('chat_history.csv'):
            df_existing = pd.read_csv('chat_history.csv')
            df_final = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_final = df_new
            
        df_final.to_csv('chat_history.csv', index=False)
        return response.content


bot = ChatBotLlama(bot_id=1)
response = ''

business_problem = '''A multinational ride-sharing company wants to expand its electric vehicle fleet into a new region, but it faces a major hurdle: the local government is currently debating a strict new green energy bill that could heavily tax non-local battery manufacturers and enforce strict pricing caps on peak-hour fares by 2028. To solve this, a Government Affairs Manager (Public Policy Analyst) must track the daily progress of this moving legislation, build data models to predict the financial impact of the proposed taxes on the company's five-year profit margins, and quickly design a strategic lobbying plan to advocate for fairer tax credits before the final vote occurs.'''


initial_prompt = f"""
You are an AI Manager overseeing a workflow. Read this business problem: 
"{business_problem}"

Task:
1. Write a comprehensive system prompt that assigns an expert role (e.g., Senior Public Policy Analyst & Economic Forecaster) to a secondary AI. 
2. In that prompt, detail the problem and clearly list what data models, legislative tracking, and strategies need evaluation.
3. Add a strict completion rule: If the secondary AI successfully solves the problem, it must conclude its final answer with the single lowercase word 'exit' on a brand new line. If it fails to solve it, it must output a prompt delegating the task to the next AI agent in line.
"""

while True:
    if response == '':
        response = bot.invoke(initial_prompt)
        print("Initial Prompt Sent. Bot Responded.")
    else:
        
        cleaned_response = str(response).strip().lower()
        
        if cleaned_response == 'exit' or cleaned_response.endswith('exit'):
            print("Exit condition met. Multi-bot chain stopped.")
            break
            
        
        response = bot.invoke(response)
        print("Running next stage of evaluation...")
