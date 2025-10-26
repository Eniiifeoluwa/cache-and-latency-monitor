import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_community.chat_message_histories import ChatMessageHistory
from config import settings


class LLMClient:
    """LangChain ChatGroq client with session-based conversation memory."""

    def __init__(self):
        # --- Initialize ChatGroq model ---
        self.llm = ChatGroq(
            groq_api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_MODEL,
            temperature=settings.GROQ_TEMPERATURE,
        )

        # --- In-memory chat history ---
        self.history = ChatMessageHistory()

        # --- Define the conversation prompt ---
        self.prompt = ChatPromptTemplate.from_template(
            """You are a friendly and helpful AI assistant.

            Previous conversation:
            {history}

            Human: {input}
            AI:"""
        )

        # --- Define the runnable chain using new pipe syntax ---
        self.chain = (
            RunnableMap({
                "history": lambda _: self.format_history(),
                "input": lambda x: x["input"],
            })
            | self.prompt
            | self.llm
        )

    # -------------------------------------------------------------------------
    # Utility functions
    # -------------------------------------------------------------------------
    def format_history(self):
        """Format past chat messages into a readable string for context."""
        if not self.history.messages:
            return "No previous messages."
        return "\n".join(
            f"{msg.type.capitalize()}: {msg.content}"
            for msg in self.history.messages
        )

    def query(self, prompt: str):
        """Send a message to the model and update conversation history."""
        response = self.chain.invoke({"input": prompt})
        self.history.add_user_message(prompt)
        self.history.add_ai_message(response.content)
        return response.content

    def clear(self):
        """Clear the conversation history."""
        self.history.clear()


# -------------------------------------------------------------------------
# Simple helper functions for Streamlit integration
# -------------------------------------------------------------------------
_client = None


def get_client():
    """Get or initialize a global LLMClient (so it persists in Streamlit session)."""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


def query_llm(prompt: str):
    """Wrapper for Streamlit app to get model response."""
    client = get_client()
    return client.query(prompt)


def clear_memory():
    """Wrapper to clear model memory."""
    client = get_client()
    client.clear()
