from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from pathlib import Path
from langchain.document_loaders import PyPDFLoader
persist_directory = str(Path.home()) + "/law_fieldlab/create_database/database"

authors = ["Anthony Casey & Anthony Niblett",
           "Mariateresa Maggiolino",
           "Omri Ben-Shahar",
           "Christopher Townley, Eric Morrison & Karen Yeung",
           "Frederik Zuiderveen Borgesius & Joost Poort",
           "Jean‐Pierre I. van der Rest & Alan M. Sears & Li Miao & Lorna Wang",
           "Gábor Rekettye & Goran Pranjic",
           "Jerod Coker & Jean‐Manuel Izaret",
           "Joshua A. Gerlick & Stephan M. Liozu",
           "Sebastião Barros Vale",
           "Andrew Verstein",
           "Christoph Busch",
           "Pascale Chapdelaine",
           "Christophe Samuel Hutchinson & Diana Treščáková"]

publishers = ["University of Chicago Law School",
             "Boccono Legal Studies",
             "University of Chicago Law School",
             "The Dickson Poon School of Law",
             "Journal of Consumer Policy",
             "Journal of Revenue and Pricing Management",
             "University of Pécs",
             "Journal of Business Ethics",
             "Journal of Revenue and Pricing Management",
             "Eur. J. Privacy L. & Tech.",
             "The University of Chicago Law Review",
             "The University of Chicago Law Review",
             "New York University Journal of Law & Business",
             "European Competition Journal"]

titles = ["A Framework for the New Personalization of Law",
          "Ppersonalized Prices in European Competition Law",
          "Personalizing Mandatory Rules in Contract Law",
          "Big Data and Personalised Price Discrimination in EU Competition Law",
          "Online Price Discrimination and EU Data Privacy Law",
          "A note on the future of personalized pricing: cause for concern",
          "Price personalization in the Big Data and GDPR context",
          "Progressive Pricing: The Ethical Case for Price Personalization",
          "Ethical and legal considerations of artificial intelligence and algorithmic decision‐making in personalized pricing",
          "The omnibus directive and online price personalization: a mere duty to inform?",
          "Privatizing Personalized Law",
          "Implementing Personalized Law: Personalized Disclosures in Consumer Law and Data Privacy Law",
          "Algorithmic Personalized Pricing",
          "The challenges of personalized pricing to competition and personal data protection law"]

sources = ["Anthony Casey & Anthony Niblett, A Framework for the New Personalization of Law, University of Chicago Public Law & Legal Theory Paper Series, No. 696 (2018).",
           "Maggiolino, Mariateresa, Personalized Prices in European Competition Law (June 12, 2017). Bocconi Legal Studies Research Paper No. 2984840, Available at SSRN: https://ssrn.com/abstract=2984840 or http://dx.doi.org/10.2139/ssrn.2984840",
           "Omri Ben-Shahar, Personalizing Mandatory Rules in Contract Law, Public Law and Legal Theory Working Paper Series, No. 680 (2018).",
           "Christopher Townley, Eric Morrison, Karen Yeung, Big Data and Personalized Price Discrimination in EU Competition Law, Yearbook of European Law, Volume 36, 2017, Pages 683–748, https://doi.org/10.1093/yel/yex015",
           "Zuiderveen Borgesius, F., Poort, J. Online Price Discrimination and EU Data Privacy Law. J Consum Policy 40, 347–366 (2017). https://doi.org/10.1007/s10603-017-9354-z",
           "van der Rest, JP.I., Sears, A.M., Miao, L. et al. A note on the future of personalized pricing: cause for concern. J Revenue Pricing Manag 19, 113–118 (2020). https://doi.org/10.1057/s41272-020-00234-6",
           "Rekettye, G. és Pranjić, G. (2020) „Price personalization in the Big Data and GDPR context”, Marketing &amp; Menedzsment, 54(3), o. 5–14. doi: 10.15170/MM.2020.54.03.01.",
           "Coker, J., Izaret, JM. Progressive Pricing: The Ethical Case for Price Personalization. J Bus Ethics 173, 387–398 (2021). https://doi.org/10.1007/s10551-020-04545-x",
           "Gerlick, J.A., Liozu, S.M. Ethical and legal considerations of artificial intelligence and algorithmic decision-making in personalized pricing. J Revenue Pricing Manag 19, 85–98 (2020). https://doi.org/10.1057/s41272-019-00225-2",
           "Vale, Sebastião Barros. The omnibus directive and online price personalization: a mere duty to inform?. Eur. J. Privacy L. & Tech. (2020): 92.",
           "Verstein, Andrew. “Privatizing Personalized Law.” The University of Chicago Law Review, vol. 86, no. 2, 2019, pp. 551–80. JSTOR, https://www.jstor.org/stable/26590565. Accessed 6 Dec. 2023.",
           "Busch, Christoph. “Implementing Personalized Law: Personalized Disclosures in Consumer Law and Data Privacy Law.” The University of Chicago Law Review, vol. 86, no. 2, 2019, pp. 309–32. JSTOR, https://www.jstor.org/stable/26590557. Accessed 6 Dec. 2023.",
           "Chapdelaine, Pascale. (2020). Algorithmic Personalized Pricing. New York University Journal of Law & Business, 17 (1), 1-47. https://scholar.uwindsor.ca/lawpub/122",
           "Christophe Samuel Hutchinson & Diana Treščáková (2021): The challenges of personalized pricing to competition and personal data protection law, European Competition Journal, DOI: 10.1080/17441056.2021.1936400"]



def get_document_embeddings(documents, authors, publishers, titles, sources, chunk_size, chunk_overlap):
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                                   separators=["\n\n\n", "\n\n", "\n", "", " "], length_function=len)

    # List to store document splits with metadata
    split_documents_with_metadata = []

    # Process each document and its metadata
    for idx, doc in enumerate(documents):
        print(f"Processing Document {idx}")
        splits = text_splitter.split_text(doc)
        for split in splits:
            # Create a dictionary with the text and the corresponding metadata
            metadata = {
                "author": authors[idx],
                "publisher": publishers[idx],
                "title": titles[idx],
                "source": sources[idx]
            }
            split_documents_with_metadata.append({
                "text": split,
                "metadata": metadata
            })

    # Embedding and saving process
    texts_to_embed = [entry["text"] for entry in split_documents_with_metadata]
    metadata_list = [entry["metadata"] for entry in split_documents_with_metadata]

    # Get embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create a Chroma vector store from texts and metadata
    persist_directory = str(Path.home()) + "/law_fieldlab/create_database/database"
    vectordb = Chroma.from_texts(texts=texts_to_embed, embedding=embeddings, metadatas=metadata_list, persist_directory=persist_directory)
    vectordb.persist()
    print("Embeddings and metadata saved locally")


# Load all documents
all_documents = list()
for n in range(1, 15):
    loader = PyPDFLoader(f"data/Paper_{n}.pdf")
    pages = loader.load()
    all_pages = ""
    for page in pages:
        all_pages += page.page_content + "\n\n\n"
    all_documents.append(all_pages)

def main():
    # Assuming 'all_documents' is the list of document strings you provided
    get_document_embeddings(all_documents, authors, publishers, titles, sources, chunk_size=1000, chunk_overlap=200)

    print("Embeddings saved locally.")

if __name__ == "__main__":
    main()
