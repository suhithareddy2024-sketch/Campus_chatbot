from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

text = open(
    "scraper/anits_content.txt",
    encoding="utf-8"
).read()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=30
)

docs = splitter.create_documents([text])

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.from_documents(
    docs,
    embedding
)

db.save_local("vector_db")

print("Vector DB Created")