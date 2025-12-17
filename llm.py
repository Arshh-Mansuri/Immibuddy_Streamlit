from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

from config import (
    OPENAI_API_KEY,
    CHAT_MODEL,
)

# ---------------- ANSWER GENERATION ----------------

class RAGAnswerGenerator:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=CHAT_MODEL,
            api_key=OPENAI_API_KEY,
        )

    def _build_context(self, docs: List[Document]) -> str:
        """
        Build a readable context string from retrieved documents.
        """
        context_blocks = []

        for d in docs:
            source = f"{d.metadata.get('institution')} | {d.metadata.get('source_type')} | {d.metadata.get('source_file')}"
            block = f"[Source: {source}]\n{d.page_content}"
            context_blocks.append(block)

        return "\n\n---\n\n".join(context_blocks)

    def answer(self, question: str, docs: List[Document]) -> str:
        """
        Generate a grounded answer using retrieved context.
        """
        context = self._build_context(docs)

        prompt = f"""
You are an assistant answering questions using retrieved documents.

Rules:
- Use the context FIRST.
- If the context does not fully answer the question, say so clearly.
- Distinguish between official policy and personal experience.
- Do NOT provide legal advice.
- Keep the answer clear and structured.

Answer format:
1. Short answer
2. What the documents say
3. Practical notes
4. What to do next
5. Disclaimer

Context:
{context}

Question:
{question}
"""

        response = self.llm.invoke(prompt)
        return response.content
