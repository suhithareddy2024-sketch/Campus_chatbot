import os
from dotenv import load_dotenv
import google.generativeai as genai

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "vector_db",
    embedding,
    allow_dangerous_deserialization=True
)

def get_answer(question, role):

    docs = db.similarity_search(
    question,
    k=10
    )

    print("\n========== RETRIEVED DOCS ==========\n")

    for i, doc in enumerate(docs):
        print(f"\nChunk {i+1}\n")
        print(doc.page_content[:500])

    context = "\n".join(
    [doc.page_content for doc in docs]
    )

    prompt = f"""
        You are ANITS Campus Assistant.

        Answer ONLY from the provided context.

        If the answer exists in the context,
        provide it clearly and directly.

        Context:
        {context}

        Question:
        {question}

        Answer:
    """

    response = model.generate_content(
        prompt + question
    )

    return response.text