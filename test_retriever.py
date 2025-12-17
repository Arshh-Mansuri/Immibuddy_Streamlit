from retriever import RAGRetriever

retriever = RAGRetriever()
docs = retriever.retrieve("Can international students apply for RPL?")

for i, d in enumerate(docs, 1):
    print(f"\n--- Result {i} ---")
    print(d.metadata)
    print(d.page_content[:500])