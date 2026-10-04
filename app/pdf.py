

# from dotenv import load_dotenv
# load_dotenv()

# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_google_genai import (
#     GoogleGenerativeAIEmbeddings,
#     ChatGoogleGenerativeAI
# )
# from langchain_chroma import Chroma
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnableLambda
# from langchain_core.output_parsers import StrOutputParser


# # ============================================================
# # 1. LOAD PDF
# # ============================================================

# loader = PyPDFLoader("data/data_science_syllabus.pdf")

# docs = loader.load()

# print("Number of pages:", len(docs))


# # ============================================================
# # 2. SPLIT PDF INTO CHUNKS
# # ============================================================

# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200
# )

# splitter_data = splitter.split_documents(docs)

# print("Number of chunks:", len(splitter_data))


# # ============================================================
# # 3. CREATE EMBEDDINGS
# # ============================================================

# embeddings = GoogleGenerativeAIEmbeddings(
#     model="models/gemini-embedding-001"
# )


# # ============================================================
# # 4. CREATE CHROMA VECTOR DATABASE
# # ============================================================

# vector_store = Chroma.from_documents(
#     documents=splitter_data,
#     embedding=embeddings
# )

# print("Vector database created successfully!")


# # ============================================================
# # 5. RETRIEVAL FUNCTION
# # ============================================================

# def get_context(query: str):

#     retrieved_docs = vector_store.similarity_search(
#         query=query,
#         k=4
#     )

#     # Debugging: see what documents are retrieved
#     print("\n========== RETRIEVED DOCUMENTS ==========")

#     for i, doc in enumerate(retrieved_docs):
#         print(f"\n--- Document {i + 1} ---")
#         print(doc.page_content)

#     print("\n==========================================")

#     # Combine retrieved chunks
#     context = "\n\n".join(
#         doc.page_content for doc in retrieved_docs
#     )

#     return {
#         "context": context,
#         "question": query
#     }


# # ============================================================
# # 6. CREATE PROMPT
# # ============================================================

# prompt = PromptTemplate.from_template("""
# You are a helpful AI assistant.

# You have access to a PDF document.

# Instructions:
# 1. If the answer to the question is available in the provided context,
#    answer using the context.
# 2. If the answer is NOT available in the context, answer the question
#    using your general knowledge.
# 3. Do not make up information about the PDF.
# 4. If the user asks about something unrelated to the PDF, you can
#    answer normally using your general knowledge.

# Context:
# {context}

# Question:
# {question}

# Answer:
# """)


# # ============================================================
# # 7. CREATE LLM
# # ============================================================

# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash"
# )


# # ============================================================
# # 8. CREATE RAG CHAIN
# # ============================================================

# rag_chain = (
#     RunnableLambda(get_context)
#     | prompt
#     | llm
#     | StrOutputParser()
# )


# # ============================================================
# # 9. ASK QUESTION
# # ============================================================

# question = "Whst is virat kohli age?"

# response = rag_chain.invoke(question)

# print("\n\n========== FINAL ANSWER ==========")
# print(response)
# print("==================================")



import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# 1. Load environment variables
# ============================================================

load_dotenv()


# ============================================================
# 2. Streamlit page configuration
# ============================================================

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Document Assistant")
st.write("Upload a PDF and ask questions about it.")


# ============================================================
# 3. Initialize session state
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# 4. Sidebar - Upload PDF
# ============================================================

with st.sidebar:

    st.header("📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"]
    )

    if uploaded_file:
        st.success("PDF uploaded successfully! ✅")


# ============================================================
# 5. Create vector store
# ============================================================

def create_vector_store(pdf_path):

    # -------------------------
    # Load PDF
    # -------------------------

    loader = PyPDFLoader(pdf_path)

    docs = loader.load()

    if not docs:
        raise ValueError("Could not read the PDF.")

    # -------------------------
    # Check extracted text
    # -------------------------

    total_text = ""

    for doc in docs:
        total_text += doc.page_content

    if not total_text.strip():

        raise ValueError(
            "No text could be extracted from this PDF. "
            "Please upload a text-based PDF instead of a scanned image PDF."
        )

    # -------------------------
    # Split documents
    # -------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    split_docs = splitter.split_documents(docs)

    if not split_docs:
        raise ValueError(
            "No document chunks were created."
        )

    # -------------------------
    # Create embeddings
    # -------------------------

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    # -------------------------
    # Test embedding
    # -------------------------

    test_embedding = embeddings.embed_query(
        "This is a test sentence."
    )

    if not test_embedding:

        raise ValueError(
            "Gemini returned an empty embedding."
        )

    # -------------------------
    # Create Chroma database
    # -------------------------

    vector_store = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings
    )

    return vector_store


# ============================================================
# 6. Process uploaded PDF
# ============================================================

if uploaded_file is not None:

    os.makedirs("uploads", exist_ok=True)

    pdf_path = os.path.join(
        "uploads",
        uploaded_file.name
    )

    # Save uploaded file

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:

        with st.spinner("Processing PDF..."):

            vector_store = create_vector_store(
                pdf_path
            )

        st.sidebar.success(
            "Document processed successfully! 🎉"
        )

    except Exception as e:

        st.error(
            f"Error while processing PDF:\n\n{str(e)}"
        )

        st.stop()


    # ========================================================
    # 7. Retrieval function
    # ========================================================

    def get_context(question):

        docs = vector_store.similarity_search(
            query=question,
            k=4
        )

        context = ""

        for doc in docs:

            context += (
                doc.page_content
                + "\n\n"
            )

        return {
            "context": context,
            "question": question,
            "documents": docs
        }


    # ========================================================
    # 8. Prompt
    # ========================================================

    prompt = PromptTemplate.from_template(
        """
You are an intelligent AI Document Assistant.

You have access to a user's uploaded PDF.

Follow these rules:

1. First check the provided context.

2. If the answer is present in the document,
   answer using the document.

3. If the answer is not present in the document,
   you may use your general knowledge.

4. Never claim that general knowledge came from
   the uploaded document.

5. If the question is unrelated to the document,
   answer normally using your general knowledge.

6. Give a clear and concise answer.

Context:
{context}

Question:
{question}

Answer:
"""
    )


    # ========================================================
    # 9. Gemini LLM
    # ========================================================

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash"
    )


    # ========================================================
    # 10. RAG Chain
    # ========================================================

    rag_chain = (
        RunnableLambda(get_context)
        | RunnableLambda(
            lambda x: {
                "context": x["context"],
                "question": x["question"]
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )


    # ========================================================
    # 11. Display previous chat messages
    # ========================================================

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(
                message["content"]
            )


    # ========================================================
    # 12. Chat input
    # ========================================================

    question = st.chat_input(
        "Ask something about your document..."
    )


    if question:

        # -------------------------
        # Display user question
        # -------------------------

        with st.chat_message("user"):

            st.markdown(question)

        # Store user message

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )


        # -------------------------
        # Generate answer
        # -------------------------

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                try:

                    answer = rag_chain.invoke(
                        question
                    )

                    st.markdown(answer)

                except Exception as e:

                    answer = (
                        "Sorry, I encountered an error: "
                        + str(e)
                    )

                    st.error(answer)


        # Store assistant response

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


else:

    # ========================================================
    # No PDF uploaded
    # ========================================================

    st.info(
        "👈 Upload a PDF from the sidebar to start chatting."
    )