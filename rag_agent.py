import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.tools import tool
from langchain.agents import create_agent

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DB_DIR = "chroma_db"
DOCS_DIR = "docs"


def load_documents():
    docs = []
    for filename in os.listdir(DOCS_DIR):
        path = os.path.join(DOCS_DIR, filename)

        if filename.endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
            docs.extend(loader.load())

        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs.extend(loader.load())

    return docs


def build_vectorstore():
    documents = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    return vectorstore


def load_or_create_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        return Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings
        )
    return build_vectorstore()


vectorstore = load_or_create_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


@tool
def retrieve_context(query: str) -> str:
    """Recherche des informations dans les documents locaux."""
    docs = retriever.invoke(query)
    if not docs:
        return "Aucun document pertinent trouvé."
    return "\n\n".join([doc.page_content for doc in docs])


rag_model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

rag_agent = create_agent(
    model=rag_model,
    tools=[retrieve_context],
    system_prompt=(
        "Tu es un assistant RAG. "
        "Quand la question concerne les documents locaux, utilise l'outil retrieve_context. "
        "Base ta réponse sur le contexte récupéré."
    ),
)


def ask_rag(question: str) -> str:
    result = rag_agent.invoke({
        "messages": [{"role": "user", "content": question}]
    })
    return result["messages"][-1].content


if __name__ == "__main__":
    print("=== PARTIE 2 : RAG AGENT ===")
    while True:
        q = input("\nQuestion : ")
        if q.lower() in {"quit", "exit"}:
            break
        print("\nRéponse :")
        print(ask_rag(q))