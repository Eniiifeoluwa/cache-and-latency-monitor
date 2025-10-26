from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from .config import settings

llm = ChatGroq(
    api_key=settings.GROQ_API_KEY,
    model=settings.GROQ_MODEL,
    temperature=settings.GROQ_TEMPERATURE
)

def query_llm(system_prompt: str, user_prompt: str) -> str:
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    result = llm.invoke(messages)
    return result.content