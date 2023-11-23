from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_transformers import (
    LongContextReorder,
)
from pathlib import Path

def rag(query, chunk_size, chunk_overlap, number_results):
    persist_directory = str(Path.home()) + "/law_fieldlab/create_database/database"
    embeddings_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings_function)
    retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": number_results})
    
    # Get relevant documents ordered by relevance score
    docs = retriever.get_relevant_documents(query)

    if number_results >= 10:
        # Reorder the documents
        reordering = LongContextReorder()
        docs = reordering.transform_documents(docs)

    results = "Take a deep breath and look at these pieces of information step by step."
    for doc in docs:
        info = doc.page_content
        results = results + "\n###" + " " + info + "###\n"
    return results

def get_data(query: str):
    chunk_size = 512
    chunk_overlap = 50
    number_results = 4
    results = rag(query, chunk_size, chunk_overlap, number_results)
    return query, results
