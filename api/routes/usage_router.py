"""
api/routes/usage_router.py
──────────────────────────
Endpoints for checking and tracking SaaS usage limits (PDFs and Queries).
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from api.database.connection import get_database
from api.middleware.auth_middleware import get_current_user
from api.models.user import UserDocument
from api.models.usage import UserUsageDocument, get_today_str

router = APIRouter(prefix="/usage", tags=["usage"])

# SaaS Limits Configuration
LIMITS = {
    "free": {
        "max_pdfs": 5,
        "max_queries_per_day": 50,
    },
    "pro": {
        "max_pdfs": 999999,
        "max_queries_per_day": 999999,
    }
}

class UsageLimitsResponse(BaseModel):
    tier: str
    max_pdfs: int
    max_queries: int
    current_pdfs: int
    current_queries: int


@router.get("/limits", response_model=UsageLimitsResponse)
async def get_usage_limits(
    current_user: UserDocument = Depends(get_current_user),
    db = Depends(get_database)
):
    """Get the user's current usage vs limits."""
    tier = current_user.tier
    limits = LIMITS.get(tier, LIMITS["free"])
    
    # 1. Count PDFs
    pdf_count = await db.document_registry.count_documents({"owner_id": str(current_user.id)})
    
    # 2. Get today's queries
    today = get_today_str()
    usage_doc = await db.user_usage.find_one({"user_id": str(current_user.id), "date": today})
    query_count = usage_doc["query_count"] if usage_doc else 0
    
    return UsageLimitsResponse(
        tier=tier,
        max_pdfs=limits["max_pdfs"],
        max_queries=limits["max_queries_per_day"],
        current_pdfs=pdf_count,
        current_queries=query_count
    )


@router.post("/track-query")
async def track_query(
    current_user: UserDocument = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Increment query count. Raises 429 if limit exceeded.
    Should be called immediately before executing an LLM query.
    """
    tier = current_user.tier
    limits = LIMITS.get(tier, LIMITS["free"])
    max_queries = limits["max_queries_per_day"]
    
    today = get_today_str()
    
    # Find current usage
    usage_doc = await db.user_usage.find_one({"user_id": str(current_user.id), "date": today})
    current_count = usage_doc["query_count"] if usage_doc else 0
    
    if current_count >= max_queries:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Daily query limit ({max_queries}) reached for {tier} tier. Upgrade to Pro."
        )
        
    # Increment
    await db.user_usage.update_one(
        {"user_id": str(current_user.id), "date": today},
        {"$inc": {"query_count": 1}},
        upsert=True
    )
    
    return {"status": "ok", "current_queries": current_count + 1}


@router.post("/check-upload")
async def check_upload(
    current_user: UserDocument = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Check if the user is allowed to upload another PDF. Raises 429 if limit exceeded.
    """
    tier = current_user.tier
    limits = LIMITS.get(tier, LIMITS["free"])
    max_pdfs = limits["max_pdfs"]
    
    pdf_count = await db.document_registry.count_documents({"owner_id": str(current_user.id)})
    
    if pdf_count >= max_pdfs:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"PDF upload limit ({max_pdfs}) reached for {tier} tier. Upgrade to Pro."
        )
        
    return {"status": "ok", "current_pdfs": pdf_count}
