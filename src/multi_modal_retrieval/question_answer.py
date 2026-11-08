import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from multi_modal_retrieval.question_embedding import (
    embed_question,
    retrieve_context_from_vectordb,
)
from multi_modal_retrieval.vectordb import run_db

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise SystemExit("Set GEMINI_API_KEY in the .env file.")

store = run_db(gemini_api_key)

question = "What was the change in finance leases between 2022 and 2023"


question_embedding = embed_question(question, api_key=gemini_api_key)

context = retrieve_context_from_vectordb(question_embedding, store)


prompt = f"""Answer the question using only the retrieved context.
If the context does not contain enough information, say context isn't sufficient.
Cite the supporting page and block UUID.

Question:
{question}

Retrieved context:
{context}
"""

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0,
    google_api_key=gemini_api_key,
)

response = llm.invoke(prompt)

print("Prompt:", prompt)
print("Answer:", response.content)
