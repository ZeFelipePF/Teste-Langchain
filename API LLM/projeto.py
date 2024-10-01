import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder
import os

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import torch
from langchain_huggingface import ChatHuggingFace
from langchain_community.llms import HuggingFaceHub

from dotenv import load_dotenv

load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Seu assistente virtual 🤖", page_icon="🤖")
st.title("Seu assistente virtual 🤖")
#st.button ("Botão")
#st.chat_input("Digite sua mensagem")

model_class = "openai"

def model_openai (model = "gpt-3.5-turbo", temperature = 0.1):
    llm = ChatOpenAI(model=model, temperature=temperature)
    return llm

def model_response(user_query, chat_history, model_class):
        llm = model_openai()

        #def dos prompts
        system_prompt = """
            Você é um assistente prestativo e está respondendo perguntas gerais. Responda em {language}
        """
        language = "português"

        user_prompt = "{input}"

        prompt_template = ChatPromptTemplate.from_messages([
              ("system", system_prompt),
              MessagesPlaceholder(variable_name="chat_history"),
              ("user", user_prompt)
        ])

        #criando a chain
        chain = prompt_template | llm | StrOutputParser()

        #retorno das resposta
        return chain.stream({
              "chat_history":chat_history,
              "input": user_query,
              "language": language
        })
        
if "chat_history" not in st.session_state:
      st.session_state.chat_history=[AIMessage(content="Olá, sou o seu assistente virtual. Como posso lhe ajudar ?")]

for message in st.session_state.chat_history:
      if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                  st.write(message.content)   
      elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                  st.write(message.write)   

user_query = st.chat_input("Digite sua mensagem") 
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
          st.markdown(user_query)     

    with st.chat_message("AI"):
          resp = st.write_stream(model_response(user_query,
                                                st.session_state.chat_history,
                                                model_class))
    st.session_state.chat_history.append(AIMessage(content=resp))
         
