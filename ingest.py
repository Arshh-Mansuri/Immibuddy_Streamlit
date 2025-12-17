import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from config import (
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    EMBEDDING_MODEL,
    PINECONE_INDEX_NAME,
    PINECONE_DIMENSION,
    PINECONE_METRIC,
    PINECONE_CLOUD,
    PINECONE_REGION,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DATA_DIR,
    PDF_SOURCES,
)

def load_and_chunk_pdfs() -> List:
    """
    Loads PDFs from /data, chunks them, and enriches metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    all_chunks = []

    for src in PDF_SOURCES:
        pdf_path = os.path.join(DATA_DIR, src["filename"])

        if not os.path.exists(pdf_path):
            print(f"⚠️ File not found, skipping: {pdf_path}")
            continue

        print(f"📄 Loading PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        chunks = splitter.split_documents(documents)

        for chunk in chunks:
            chunk.metadata.update({
                "topic": src["topic"],
                "institution": src["institution"],
                "source_type": src["source_type"],
                "source_file": src["filename"],
            })

        all_chunks.extend(chunks)

    print(f"✅ Total chunks created: {len(all_chunks)}")
    return all_chunks

# ---------------- SETUP PINECONE ----------------

def setup_pinecone_index():
    """
    Creates Pinecone index if it does not exist.
    """
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"🆕 Creating Pinecone index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=PINECONE_DIMENSION,
            metric=PINECONE_METRIC,
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION,
            ),
        )
    else:
        print(f"✔ Pinecone index already exists: {PINECONE_INDEX_NAME}")


# ---------------- INGEST INTO PINECONE ----------------

def ingest_documents(chunks: List):
    """
    Embeds chunks and upserts them into Pinecone.
    """
    if not chunks:
        print("⚠️ No chunks to ingest.")
        return

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
    )

    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
    )

    vectorstore.add_documents(chunks)
    print(f"🚀 Upserted {len(chunks)} chunks into Pinecone.")

# ---------------- MAIN ----------------

def main():
    print("🔹 Starting ingestion pipeline")

    chunks = load_and_chunk_pdfs()
    setup_pinecone_index()
    ingest_documents(chunks)

    print("✅ Ingestion completed successfully")


if __name__ == "__main__":
    main()
