"""
Summarize Text

Suppose you have a set of documents (PDFs, Notion pages, customer questions, etc.)
and you want to summarize the content.

LLMs are a great tool for this given their proficiency in understanding and
synthesizing text.

In the context of retrieval-augmented generation, summarizing text can help
distill the information in a large number of retrieved documents to provide
context for a LLM.

In this walkthrough we'll go over how to summarize content from multiple
documents using LLMs.

"""
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()

model = ChatOllama(base_url='')