from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_transformers import (
    LongContextReorder,
)
from pathlib import Path
import pandas as pd

def rag(query, number_results, df, idx, similarity):
    persist_directory = str(Path.home()) + "/law_fieldlab/create_database/database"
    embeddings_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings_function)
    
    #Get the correct paper
    filter_source = df["Paper"][idx]
    
    if filter_source == "General":
        retriever = vector_db.as_retriever(search_type=similarity, search_kwargs={"k": number_results})
    else:
        retriever = vector_db.as_retriever(search_type=similarity, search_kwargs={"k": number_results, "filter":{"title": filter_source}})
    
    # Get relevant documents ordered by relevance score
    query = query.lower().strip()
    docs = retriever.get_relevant_documents(query)

    # Reorder the documents
    reordering = LongContextReorder()
    docs = reordering.transform_documents(docs)

    results = ""
    all_sources = ""
    for doc in docs:
        source = doc.metadata["source"]
        info = doc.page_content
        info = info.lower().strip()
        results += "### " + info + " (Source: " + source + ")"
        results += " ###"

    return results

def get_data(query, number_results, all_df, idx, similarity):
    results = rag(query, number_results, all_df, idx, similarity)
    return results
