from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    PINECONE_INDEX_NAME,
)

# ---------------- RETRIEVER SETUP ----------------

class RAGRetriever:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=OPENAI_API_KEY,
        )

        self.vectorstore = PineconeVectorStore.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=self.embeddings,
        )

        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={
                "k": 4
            }
        )

    def retrieve(self, query: str):
        """
        Retrieve relevant document chunks for a query.
        """
        return self.retriever.invoke(query)
