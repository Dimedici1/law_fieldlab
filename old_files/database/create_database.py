from scrape_main import scrape_full_documents
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from link_collection import link_collection
import chromadb
from pathlib import Path

persist_directory = str(Path.home()) + "/law_fieldlab/create_database/database"
ID_LIST = [0]

#chroma_client = chromadb.Client()
#collection = chroma_client.create_collection(name="EU_embeddings")

def get_document_embeddings(urls, chunk_size, chunk_overlap):
    # Get the documents and summaries
    documents, summaries, summary_urls = scrape_full_documents(urls)

    # Combine all lists
    all_documents = documents + summaries
    all_urls = urls + summary_urls

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                                   separators=["\n\n", "\n", "", " "], length_function = len)

    # List to store document splits with metadata
    split_documents_with_metadata = []

    # Split documents and create metadata
    for idx, doc in enumerate(all_documents):
        print(f"Processing Document {idx}")
        splits = text_splitter.split_text(doc)
        for split in splits:
            # For each split, create a dictionary with the text and the corresponding URL
            split_documents_with_metadata.append({
                "text": split,
                "metadata": {"source": all_urls[idx]}
            })

    # Separate the texts and their metadata for embedding
    texts_to_embed = [entry["text"] for entry in split_documents_with_metadata]
    metadata_list = [entry["metadata"] for entry in split_documents_with_metadata]

    # Get embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create a Chroma vector store from texts and metadata
    vectordb = Chroma.from_texts(texts=texts_to_embed, embedding=embeddings, metadatas=metadata_list, persist_directory=persist_directory)
    vectordb.persist()
    vectordb = None
    print("Embeddings and metadata saved locally")


def main():
    # Document URLs
    document_urls = link_collection[1:100]

    # Get embeddings
    get_document_embeddings(document_urls, chunk_size=1000, chunk_overlap=200)

    print("Embeddings saved locally.")

if __name__ == "__main__":
    main()
