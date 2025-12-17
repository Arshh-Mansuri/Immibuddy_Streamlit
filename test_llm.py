from retriever import RAGRetriever
from llm import RAGAnswerGenerator

retriever = RAGRetriever()
llm = RAGAnswerGenerator()

question = "Can international students apply for RPL?"
docs = retriever.retrieve(question)

answer = llm.answer(question, docs)
print(answer)