import time
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from config import settings


class LLMClient:
    """
    ChatGroq client with lightweight conversational memory (works with LC 0.3+)
    """

    def __init__(self):
        self.llm = ChatGroq(
            groq_api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_MODEL,
            temperature=settings.GROQ_TEMPERATURE
        )

        # ✅ Custom message history to mimic ConversationBufferMemory
        self.history = ChatMessageHistory()

        # Define the conversation prompt
        self.prompt = ChatPromptTemplate.from_template(
            """You are a friendly and helpful AI assistant.
            Here is the previous context:
            {history}

            Human: {input}
            AI:"""
        )

        # Create a runnable chain manually
        self.chain = RunnableSequence([
            {
                "history": lambda _: self.format_history(),
                "input": lambda x: x["input"]
            },
            self.prompt,
            self.llm
        ])

    def format_history(self):
        """Formats previous exchanges into a readable text block."""
        lines = []
        for m in self.history.messages:
            role = "Human" if m.type == "human" else "AI"
            lines.append(f"{role}: {m.content}")
        return "\n".join(lines[-10:])  # keep last 10 turns

    def query(self, user_input: str):
        """Sends a query to Groq and updates the message history."""
        start = time.time()

        result = self.chain.invoke({"input": user_input})
        response = result.content if hasattr(result, "content") else str(result)

        # Store new messages in memory
        self.history.add_user_message(user_input)
        self.history.add_ai_message(response)

        latency = round((time.time() - start) * 1000, 2)
        return response, latency

    def clear_memory(self):
        """Clears chat history."""
        self.history.clear()
