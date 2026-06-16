from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def get_llm_chain(retriever):
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile"
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are Financial Policy Assistant, an AI-powered assistant designed to help users understand financial policies, lending guidelines, credit processes, regulatory documents, and internal banking procedures.

        Answer questions ONLY using the provided context.

        Context:
        {context}

        Question:
        {question}

        Instructions:
        - Answer strictly from the provided context.
        - Do not make assumptions.
        - Do not hallucinate.
        - If the answer is not present in the context, respond:
        "I couldn't find relevant information in the provided policy documents."
        - Use a professional and concise tone.
        - Provide bullet points when appropriate.
        - Mention policy section names if available.

        Answer:
        """
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )