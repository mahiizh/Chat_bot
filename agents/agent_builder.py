from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate

from .llm import get_llm
from .tools import get_tools


def build_agent():
    llm = get_llm()
    tools = get_tools()

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Your name is wonderwiess. You are a hybrid AI assistant:\n"
         "1. Use calculator for math\n"
         "2. Use web_search for real-time info\n"
         "3. Use search_documents for uploaded files\n"
         "Always prioritize document search for uploaded content."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    agent = create_openai_tools_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True
    )

    message_store = {}

    def get_session_history(session_id: str):
        if session_id not in message_store:
            message_store[session_id] = InMemoryChatMessageHistory()
        return message_store[session_id]

    return RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
    )
