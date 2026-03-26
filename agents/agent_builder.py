from langchain.agents import create_agent
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from .llm import get_llm
from .tools import get_tools


def build_agent():
    llm = get_llm()
    tools = get_tools()

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "Your name is wonderwiess,You are a hybrid AI assistant with multiple capabilities:\n"
            "1. Use the calculator tool for mathematical calculations.\n"
            "2. Use the web_search tool for real-time information from the internet.\n"
            "3. Use the search_documents tool to find information in uploaded documents and images.\n"
            "When a user asks about uploaded content, always search the documents first.\n"
            "Provide clear, helpful responses based on the available information."
        ),
    )

    message_store = {}

    def get_session_history(session_id: str):
        if session_id not in message_store:
            message_store[session_id] = InMemoryChatMessageHistory()
        return message_store[session_id]

    return RunnableWithMessageHistory(
        agent,
        get_session_history,
        input_messages_key="messages",
    )