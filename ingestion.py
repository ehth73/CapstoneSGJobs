"""
NTU SCTP Capstone - Online Support Agent
Ingestion script for building persistent Chroma vector stores from domain knowledge bases.

Supported files: .txt, .md, .csv, .xlsx, .pdf
Default expected structure:
    data/product/
    data/policy/
    data/case/

Run:
    python ingestion.py --data-dir data --persist-dir vectorstore/chroma_db
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".xlsx", ".pdf"}
DEFAULT_COLLECTIONS = {
    "product": "product_knowledge",
    "policy": "policy_knowledge",
    "case": "case_knowledge",
}


def normalise_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def csv_or_excel_to_documents(path: Path, kb_type: str) -> List[Document]:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    docs: List[Document] = []
    for idx, row in df.iterrows():
        row_pairs = [f"{col}: {normalise_text(row[col])}" for col in df.columns]
        page_content = "\n".join(row_pairs)
        docs.append(
            Document(
                page_content=page_content,
                metadata={
                    "source": str(path),
                    "row": int(idx),
                    "kb_type": kb_type,
                    "file_name": path.name,
                },
            )
        )
    return docs


def load_file(path: Path, kb_type: str) -> List[Document]:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".xlsx"}:
        return csv_or_excel_to_documents(path, kb_type)
    if suffix == ".pdf":
        docs = PyPDFLoader(str(path)).load()
    else:
        docs = TextLoader(str(path), encoding="utf-8").load()

    for doc in docs:
        doc.metadata.update({"kb_type": kb_type, "file_name": path.name, "source": str(path)})
    return docs


def iter_files(directory: Path) -> Iterable[Path]:
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def build_collection(
    docs: List[Document],
    collection_name: str,
    persist_dir: Path,
    embedding_model_name: str,
) -> None:
    if not docs:
        print(f"No documents found for collection: {collection_name}")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(persist_dir),
    )
    print(f"Built collection '{collection_name}' with {len(chunks)} chunks.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build vector stores for Online Support Agent.")
    parser.add_argument("--data-dir", default="data", help="Root directory containing product/policy/case folders.")
    parser.add_argument("--persist-dir", default=os.getenv("VECTOR_DB_DIR", "vectorstore/chroma_db"))
    parser.add_argument("--embedding-model", default=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    persist_dir = Path(args.persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    for kb_folder, collection_name in DEFAULT_COLLECTIONS.items():
        folder = data_dir / kb_folder
        folder.mkdir(parents=True, exist_ok=True)
        docs: List[Document] = []
        for file_path in iter_files(folder):
            docs.extend(load_file(file_path, kb_folder))
        build_collection(docs, collection_name, persist_dir, args.embedding_model)

    print("Ingestion completed successfully.")


if __name__ == "__main__":
    main()
