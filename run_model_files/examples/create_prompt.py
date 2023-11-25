from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_transformers import (
    LongContextReorder,
)
from pathlib import Path

def rag(query, number_results):
    persist_directory = str(Path.home()) + "/law_fieldlab/create_database/database"
    embeddings_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings_function)
    retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": number_results})
    
    # Get relevant documents ordered by relevance score
    query = query.lower().strip()
    docs = retriever.get_relevant_documents(query)

    # Reorder the documents
    reordering = LongContextReorder()
    docs = reordering.transform_documents(docs)

    results = ""
    all_sources = ""
    for doc in docs:
        source = doc.metadatas
        info = doc.page_content
        info = info.lower().strip()
        results += "### " + info + " (Source: " + source + ")" + " ###"
    return results

def get_data(query: str):
    number_results = 4
    results = rag(query, number_results)
    return query, results
