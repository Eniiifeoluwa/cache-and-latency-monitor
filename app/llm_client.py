import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from config import settings

llm = ChatGroq(
    temperature=0.2,
    groq_api_key=settings.GROQ_API_KEY,
    model_name=settings.MODEL_NAME
)

memory = ConversationBufferMemory(return_messages=True)

conversation = LLMChain(
    llm=llm,
    memory=memory,
    verbose=True
)

def query_llm(prompt: str):
    return conversation.predict(input=prompt)
