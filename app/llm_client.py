import time
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from config import settings


class LLMClient:
    """
    ChatGroq client with memory and conversational handling.
    """

    def __init__(self):
        self.llm = ChatGroq(
            groq_api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_MODEL,
            temperature=settings.GROQ_TEMPERATURE
        )
        self.memory = ConversationBufferMemory(return_messages=True)

        self.prompt = ChatPromptTemplate.from_template(
            """The following is a friendly and thoughtful AI conversation.
            Context:
            {history}

            Human: {input}
            AI:"""
        )

        self.chain = RunnableSequence([
            {
                "history": lambda _: self.memory.load_memory_variables({})["history"],
                "input": lambda x: x["input"]
            },
            self.prompt,
            self.llm
        ])

    def query(self, user_input: str):
        """Queries Groq LLM and updates conversation memory."""
        start = time.time()
        result = self.chain.invoke({"input": user_input})
        response = result.content if hasattr(result, "content") else str(result)

        # Save in-memory conversation
        self.memory.chat_memory.add_user_message(user_input)
        self.memory.chat_memory.add_ai_message(response)

        latency = round((time.time() - start) * 1000, 2)
        return response, latency

    def clear_memory(self):
        """Clears the memory context."""
        self.memory.clear()
