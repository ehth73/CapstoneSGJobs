"""
NTU SCTP Capstone - Online Support Agent
LangGraph + RAG multi-agent application.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Literal, TypedDict

import streamlit as st
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import END, StateGraph

try:
    from groq import Groq
except Exception:  # pragma: no cover
    Groq = None

load_dotenv()

VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "vectorstore/chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
COLLECTIONS = {
    "information": "product_knowledge",
    "policy": "policy_knowledge",
    "case": "case_knowledge",
}

POLICY_TERMS = ["policy", "refund", "return", "warranty", "privacy", "terms", "faq", "shipping", "exchange"]
CASE_TERMS = ["claim", "case", "complaint", "issue", "broken", "not working", "troubleshoot", "escalate", "return request"]
INFO_TERMS = ["recommend", "detail", "information", "compare", "product", "service", "course", "skill", "job"]
ESCALATION_TERMS = ["human", "manager", "lawyer", "legal", "angry", "urgent", "sue", "unsafe", "personal data breach"]


class AgentState(TypedDict):
    query: str
    route: Literal["information", "policy", "case", "human"]
    context: str
    answer: str
    citations: List[str]
    confidence: float


@st.cache_resource(show_spinner=False)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_resource(show_spinner=False)
def load_vectorstores() -> Dict[str, Chroma]:
    persist_path = Path(VECTOR_DB_DIR)
    if not persist_path.exists():
        return {}
    embeddings = get_embeddings()
    stores = {}
    for route, collection in COLLECTIONS.items():
        stores[route] = Chroma(
            collection_name=collection,
            persist_directory=str(persist_path),
            embedding_function=embeddings,
        )
    return stores


def keyword_score(query: str, terms: List[str]) -> int:
    q = query.lower()
    return sum(1 for term in terms if term in q)


def router_node(state: AgentState) -> AgentState:
    query = state["query"]
    scores = {
        "human": keyword_score(query, ESCALATION_TERMS),
        "case": keyword_score(query, CASE_TERMS),
        "policy": keyword_score(query, POLICY_TERMS),
        "information": keyword_score(query, INFO_TERMS),
    }
    route = max(scores, key=scores.get)
    if scores[route] == 0:
        route = "human"
    state["route"] = route  # type: ignore
    return state


def retrieve(route: str, query: str, k: int = 4) -> tuple[str, List[str], float]:
    stores = load_vectorstores()
    if route not in stores:
        return "", [], 0.0
    results = stores[route].similarity_search_with_relevance_scores(query, k=k)
    docs: List[Document] = [doc for doc, _ in results]
    scores = [score for _, score in results]
    context = "\n\n".join([doc.page_content for doc in docs])
    citations = []
    for doc in docs:
        source = doc.metadata.get("file_name") or doc.metadata.get("source", "knowledge base")
        row = doc.metadata.get("row")
        citations.append(f"{source}" + (f" row {row}" if row is not None else ""))
    confidence = max(scores) if scores else 0.0
    return context, citations, float(confidence)


def llm_generate(system_prompt: str, query: str, context: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if api_key and Groq is not None:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nCustomer query:\n{query}"},
            ],
            temperature=0.2,
            max_tokens=700,
        )
        return response.choices[0].message.content or "I have no answer."

    # Safe fallback when no API key is configured.
    if not context.strip():
        return "I have no answer from the available knowledge base. Please escalate this to a human support officer."
    short_context = re.sub(r"\s+", " ", context).strip()[:1200]
    return (
        "Based on the available knowledge base, the most relevant information is:\n\n"
        f"{short_context}\n\n"
        "Please verify this against the cited source documents before making a final customer commitment."
    )


def information_agent(state: AgentState) -> AgentState:
    context, citations, confidence = retrieve("information", state["query"])
    prompt = "You are the Information Agent. Answer only using the retrieved context. Provide helpful recommendations and say when evidence is insufficient."
    state.update({"context": context, "citations": citations, "confidence": confidence, "answer": llm_generate(prompt, state["query"], context)})
    return state


def policy_agent(state: AgentState) -> AgentState:
    context, citations, confidence = retrieve("policy", state["query"])
    prompt = "You are the Policy Agent. Explain policies, FAQs, obligations, limitations, and next steps using only the retrieved context."
    state.update({"context": context, "citations": citations, "confidence": confidence, "answer": llm_generate(prompt, state["query"], context)})
    return state


def case_agent(state: AgentState) -> AgentState:
    context, citations, confidence = retrieve("case", state["query"])
    prompt = "You are the Case Agent. Provide a numbered workflow for the customer. Ask only for minimum required information. Escalate if unresolved."
    answer = llm_generate(prompt, state["query"], context)
    if confidence < 0.25:
        answer += "\n\nRecommended next step: create a support case and route it to a human officer because confidence is low."
    state.update({"context": context, "citations": citations, "confidence": confidence, "answer": answer})
    return state


def human_escalation_agent(state: AgentState) -> AgentState:
    state.update({
        "context": "",
        "citations": [],
        "confidence": 0.0,
        "answer": (
            "This query should be escalated to a human support officer. "
            "Reason: it is out-of-scope, sensitive, urgent, or the knowledge base does not contain enough reliable evidence. "
            "Please provide contact details, order/case reference, preferred callback time, and a short issue summary."
        ),
    })
    return state


def route_condition(state: AgentState) -> str:
    return state["route"]


@st.cache_resource(show_spinner=False)
def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("router", router_node)
    workflow.add_node("information", information_agent)
    workflow.add_node("policy", policy_agent)
    workflow.add_node("case", case_agent)
    workflow.add_node("human", human_escalation_agent)
    workflow.set_entry_point("router")
    workflow.add_conditional_edges(
        "router",
        route_condition,
        {"information": "information", "policy": "policy", "case": "case", "human": "human"},
    )
    for node in ["information", "policy", "case", "human"]:
        workflow.add_edge(node, END)
    return workflow.compile()


def run_agent(query: str) -> AgentState:
    app = build_graph()
    initial_state: AgentState = {"query": query, "route": "human", "context": "", "answer": "", "citations": [], "confidence": 0.0}
    return app.invoke(initial_state)


st.set_page_config(page_title="Online Support Agent", page_icon="🤖", layout="wide")
st.title("🤖 Online Support Agent Capstone")
st.caption("LangGraph multi-agent workflow with RAG retrieval and human escalation fallback")

with st.sidebar:
    st.header("System Status")
    st.write(f"Vector DB: `{VECTOR_DB_DIR}`")
    st.write(f"Embedding model: `{EMBEDDING_MODEL}`")
    st.write("LLM: Groq" if os.getenv("GROQ_API_KEY") else "LLM: extractive fallback mode")
    st.info("Run `python ingestion.py` before launching the app.")

query = st.text_area("Customer inquiry", placeholder="Example: What course or product would you recommend for someone with strong ML and cloud skills?")

if st.button("Submit", type="primary") and query.strip():
    with st.spinner("Routing query and retrieving knowledge base context..."):
        result = run_agent(query.strip())
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Answer")
        st.write(result["answer"])
    with col2:
        st.subheader("Agent Trace")
        st.metric("Selected Agent", result["route"].title())
        st.metric("Retrieval Confidence", f"{result['confidence']:.2f}")
        if result["citations"]:
            st.write("Sources")
            for source in result["citations"]:
                st.caption(source)
else:
    st.warning("Enter a customer inquiry and click Submit.")
