import os
import time
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
# from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import json
BM25_FILE = "data/bm25_chunks.json"

load_dotenv()

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
PINECONE_ENV="us-east-1"
PINECONE_INDEX_NAME="medicalindex"

os.environ["GOOGLE_API_KEY"]=GOOGLE_API_KEY

UPLOAD_DIR="./uploaded_docs"
os.makedirs(UPLOAD_DIR,exist_ok=True)


# initialize pinecone instance
pc=Pinecone(api_key=PINECONE_API_KEY)
print("Pinecone instance initialized successfully!")
print(pc.list_indexes())
spec=ServerlessSpec(cloud="aws",region=PINECONE_ENV)
existing_indexes=[i["name"] for i in pc.list_indexes()]


if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,
        metric="dotproduct",
        spec=spec
    )
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(1)


index=pc.Index(PINECONE_INDEX_NAME)

# load,split,embed and upsert pdf docs content

def load_vectorstore(uploaded_files):
    embed_model =  HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")
    file_paths = []

    for file in uploaded_files:
        save_path = Path(UPLOAD_DIR) / file.filename
        with open(save_path, "wb") as f:
            f.write(file.file.read())
        file_paths.append(str(save_path))

    for file_path in file_paths:
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        print("\n=== RAW PDF TEXT ===")
        print(documents[0].page_content[:1000])
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        chunks = splitter.split_documents(documents)
        
        ids = [
            f"{Path(file_path).stem}-page{chunks[i].metadata.get('page',0)}-chunk{i}"
            for i in range(len(chunks))
        ]
        
        metadatas = [
            {
                "chunk_id": ids[i],
                "document_id": Path(file_path).stem,
                "filename": Path(file_path).name,
                "source": chunk.metadata.get("source", ""),
                "page": chunk.metadata.get("page", 0),
                "text": chunk.page_content
            }
            for i, chunk in enumerate(chunks)
        ]
        # Save chunks for BM25 retrieval
        with open(BM25_FILE, "a", encoding="utf-8") as f:
            for i, chunk in enumerate(chunks):
                json.dump({
                    "chunk_id": ids[i],
                    "document_id": Path(file_path).stem,
                    "filename": Path(file_path).name,
                    "text": chunk.page_content,
                    "source": chunk.metadata.get("source", ""),
                    "page": chunk.metadata.get("page", 0)
                }, f)
                f.write("\n")

        texts = [chunk.page_content for chunk in chunks]
        # metadatas = [chunk.metadata for chunk in chunks]

        print(f"🔍 Embedding {len(texts)} chunks...")
        embeddings = embed_model.embed_documents(texts)

        print("📤 Uploading to Pinecone...")
        with tqdm(total=len(embeddings), desc="Upserting to Pinecone") as progress:
            index.upsert(vectors=zip(ids, embeddings, metadatas))
            progress.update(len(embeddings))

        print(f"✅ Upload complete for {file_path}")