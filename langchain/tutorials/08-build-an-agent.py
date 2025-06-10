"""
Build an Agent

By themselves, language models can't take actions - they just output text.
A big use case for LangChain is creating agents. Agents are systems that use LLMs
as reasoning engines to determine which actions to take and the inputs necessary
to perform the action. After executing actions, the results can be fed back into
the LLM to determine whether more actions are needed, or whether it is okay to finish.
This is often achieved via tool-calling.

In this tutorial we will build an agent that can interact with a search engine.
You will be able to ask this agent questions, watch it call the search tool,
and have conversations with it.

This example demonstrates how to add system messages and context to the agent:
1. System messages: Provide instructions to the agent about its role and behavior
2. Context: Add metadata to the RunnableConfig to provide additional context

The agent_executor.invoke() method can accept:
- A single message (e.g., HumanMessage)
- A list of messages (e.g., [SystemMessage, HumanMessage])
- The config parameter can include metadata and other context information

"""
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_tavily import TavilySearch, TavilyExtract
from pprint import pprint

load_dotenv()

from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

memory = MemorySaver()
model = ChatOllama(base_url='http://obione.archive.systems:11434', model='qwen2.5:7b', temperature=0.5)

search_tool = TavilySearch(
    max_results=3,
    topic="general",
)

extract_tool = TavilyExtract(
    extract_depth="basic",
    include_images=False,
)

tools = [search_tool, extract_tool]
agent_executor = create_react_agent(model, tools, checkpointer=memory)

# Create a system message with instructions
# You can customize this message to control the agent's behavior, tone, and capabilities
system_message = SystemMessage(content="You are a helpful assistant that can search for information and extract data from websites. Always provide accurate and concise answers.")

# Create a human message with the user's question
human_message = HumanMessage(content="What is the capital of France?")

# Use the agent with system message and context
# You can customize the metadata to include any context information relevant to your application
# This context can be used for tracking conversations, user information, or any other metadata
config = RunnableConfig({
    "thread_id": "abc123",  # Unique identifier for this conversation thread
    "metadata": {
        "conversation_id": "12345",  # ID to track the conversation
        "user_id": "user_789",       # ID of the user
        "session_start_time": "2023-06-15T10:30:00Z",  # Additional context
        "user_preferences": {         # User-specific preferences
            "language": "English",
            "detail_level": "concise"
        }
    }
})

# Invoke the agent with both system and human messages
response = agent_executor.invoke([system_message, human_message], config=config)

# Print the response
print("\nAgent Response:")
print(response.content)

# Example of continuing the conversation with the same system message and context
print("\nContinuing the conversation...")
follow_up_message = HumanMessage(content="What is the population of Paris?")
follow_up_response = agent_executor.invoke([system_message, follow_up_message], config=config)
print("\nAgent Response to follow-up:")
print(follow_up_response.content)

# Summary of key concepts:
# 1. System messages provide instructions to the agent about its role and behavior
# 2. Context in RunnableConfig provides additional information for tracking and personalization
# 3. Using the same thread_id maintains conversation history across multiple interactions
# 4. You can customize both system messages and context for different use cases
#
# For more advanced usage:
# - You can add multiple system messages to provide different types of instructions
# - You can update the context between interactions to reflect changes in the conversation
# - You can use tags in RunnableConfig for filtering and organizing conversations
