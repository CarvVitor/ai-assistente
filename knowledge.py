# knowledge.py
import os  # <- era isso que estava faltando
from langchain_google_genai import ChatGoogleGenerativeAI  # para geração de respostas
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS  # FAISS agora está em langchain_community
# em vez de GoogleGenerativeAIEmbeddings use uma alternativa:
from langchain_community.embeddings import HuggingFaceEmbeddings

def carregar_base_de_conhecimento():
    pasta_docs = "docs"
    textos = []

    for arquivo in os.listdir(pasta_docs):
        caminho = os.path.join(pasta_docs, arquivo)
        with open(caminho, "r", encoding="utf-8") as f:
            textos.append(f.read())

    # divide os textos para facilitar a busca
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs_splitados = []
    for texto in textos:
        docs_splitados.extend(splitter.split_text(texto))

    # Use o modelo BERT pequeno como exemplo de embeddings locais (não há custo)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Cria o índice FAISS com os vetores
    base_conhecimento = FAISS.from_texts(docs_splitados, embeddings)
    return base_conhecimento