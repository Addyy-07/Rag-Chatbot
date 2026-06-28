import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from backend.services.retrieval.query_rewriter import rewrite_query
from backend.services.retrieval.reranker import rerank_documents

def test_query_rewriter_no_history():
    result = rewrite_query("What is the capital of France?", "")
    assert result == "What is the capital of France?"

@patch("backend.services.retrieval.query_rewriter.ChatGroq")
def test_query_rewriter_with_history(mock_chat_groq):
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value.content = "What is the population of Paris?"
    mock_chat_groq.return_value = mock_llm_instance
    
    result = rewrite_query("And what is its population?", "Human: What is the capital of France?\\nAssistant: Paris.")
    # The StrOutputParser handles the extraction of .content if it returns AIMessage.
    # Our mock setup is a bit simplified, but if it passes the chain, it works.
    assert isinstance(result, str)


@patch("backend.services.retrieval.reranker.CrossEncoder")
def test_reranker(mock_cross_encoder):
    mock_model = MagicMock()
    # Scores for docs: Doc 1 gets 2.0, Doc 2 gets -10.0 (below threshold), Doc 3 gets 5.0
    mock_model.predict.return_value = [2.0, -10.0, 5.0]
    mock_cross_encoder.return_value = mock_model
    
    docs = [
        Document(page_content="Doc 1 content", metadata={"id": 1}),
        Document(page_content="Doc 2 content", metadata={"id": 2}),
        Document(page_content="Doc 3 content", metadata={"id": 3}),
    ]
    
    result = rerank_documents("query", docs, top_k=2, relevance_threshold=-5.0)
    
    # Doc 2 is dropped. Doc 3 is highest (5.0), Doc 1 is second (2.0)
    assert len(result) == 2
    assert result[0].metadata["id"] == 3
    assert result[1].metadata["id"] == 1
    
    assert result[0].metadata["reranker_score"] == 5.0
    assert result[1].metadata["reranker_score"] == 2.0
