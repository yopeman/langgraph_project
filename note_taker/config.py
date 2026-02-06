
from langchain_ollama import ChatOllama

# llm = ChatOllama(model='smollm2:135m')
# llm = ChatOllama(model='gemma3:4b')
llm = ChatOllama(model='llama3.2:3b')
config = {'configurable': {'thread_id': 't-123'}}