import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from api.routes.usage_router import check_upload, track_query
from fastapi import HTTPException

@pytest.mark.asyncio
async def test_check_upload_free_tier_exceeded():
    mock_db = MagicMock()
    mock_db.document_registry.count_documents = AsyncMock(return_value=5) # 5 docs already
    
    mock_user = MagicMock()
    mock_user.tier = "free"
    
    with pytest.raises(HTTPException) as excinfo:
        await check_upload(current_user=mock_user, db=mock_db)
    
    assert excinfo.value.status_code == 429
    assert "PDF upload limit" in str(excinfo.value.detail)

@pytest.mark.asyncio
async def test_check_upload_pro_tier_allowed():
    mock_db = MagicMock()
    mock_db.document_registry.count_documents = AsyncMock(return_value=5) # 5 docs already
    
    mock_user = MagicMock()
    mock_user.tier = "pro"
    
    result = await check_upload(current_user=mock_user, db=mock_db)
    assert result["status"] == "ok"
