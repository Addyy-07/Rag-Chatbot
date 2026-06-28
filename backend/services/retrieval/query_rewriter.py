"""
backend/services/retrieval/query_rewriter.py
─────────────────────────────────────────────
Conversation-Aware Query Rewriting using LLM.

Given the chat history and the latest user query, this service rewrites
the query to be a standalone, optimized query suitable for vector/hybrid retrieval.
It resolves pronouns (e.g., "What did it say about X?") using the context of
the conversation history.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from backend.config.settings import settings
from backend.utils.logger import get_logger

log = get_logger(__name__)

REWRITE_PROMPT_TEMPLATE = """
Given the following conversation history and a follow-up question, rephrase the 
follow-up question to be a standalone question, in its original language, 
that captures all relevant context from the history.

If the follow-up question does not need the history (e.g. it is a completely new topic),
simply return the original follow-up question.

Conversation History:
{history}

Follow-up Question: {question}

Standalone Question:"""

REWRITE_PROMPT = PromptTemplate.from_template(REWRITE_PROMPT_TEMPLATE)


def rewrite_query(question: str, history_text: str) -> str:
    """
    Rewrite the user's question using the provided conversation history text.
    
    Args:
        question: The latest user question.
        history_text: The formatted conversation history string.
        
    Returns:
        A standalone question string optimized for retrieval.
    """
    if not history_text.strip():
        # No history to resolve against
        return question

    try:
        llm = ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=0.0, # zero temperature for deterministic rewriting
            max_tokens=200,
        )
        
        chain = REWRITE_PROMPT | llm | StrOutputParser()
        rewritten_query = chain.invoke({
            "history": history_text,
            "question": question,
        })
        
        rewritten_query = rewritten_query.strip()
        log.info("Query Rewriter: '%s' -> '%s'", question, rewritten_query)
        return rewritten_query
    except Exception as exc:
        log.error("Query rewriting failed, falling back to original query: %s", exc)
        return question
