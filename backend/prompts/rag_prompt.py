"""
backend/prompts/rag_prompt.py
──────────────────────────────
All LangChain prompt templates used by the application, defined as
module-level constants.

Why separate prompts from services?
  - Prompts are a product concern — they change independently of retrieval logic.
  - Decoupling lets you A/B test prompts without touching service code.
  - Centralising them here makes versioning and auditing trivial.
  - Follows the Open/Closed Principle: add new prompts without modifying services.

Usage
-----
    from backend.prompts.rag_prompt import RAG_PROMPT

    chain = RAG_PROMPT | llm | StrOutputParser()
"""

from langchain_core.prompts import ChatPromptTemplate


# ── RAG Q&A Prompt ────────────────────────────────────────────────────────────
#
# Inputs (filled at chain invocation):
#   {history}   — Formatted recent conversation turns
#   {context}   — Retrieved document chunks from Pinecone
#   {question}  — Current user question
#
RAG_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_template(
    """You are DocChat AI — a precise, helpful assistant that answers questions \
strictly based on the provided document context.

Rules:
- Answer only from the context below. Do NOT hallucinate or use outside knowledge.
- If the answer is not in the context, say: "I couldn't find that in the document."
- Be concise and clear. Prefer bullet points for lists.
- Always stay professional and factual.

Previous conversation:
{history}

Document context:
{context}

Question: {question}

Answer:"""
)


# ── Standalone Question Reformulation Prompt ────────────────────────────────
#
# Optional: Use this to reformulate a follow-up question into a standalone
# question before retrieving — improves retrieval accuracy in multi-turn chats.
#
# Inputs:
#   {history}   — Formatted recent conversation turns
#   {question}  — Raw user question (may be a follow-up)
#
QUESTION_REFORMULATION_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_template(
    """Given the conversation history below and a follow-up question, \
rewrite the follow-up question as a complete, standalone question that contains \
all necessary context for a search engine.

Conversation history:
{history}

Follow-up question: {question}

Standalone question:"""
)
