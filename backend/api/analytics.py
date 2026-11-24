from fastapi import APIRouter
from analytics import get_stats

router = APIRouter()

@router.get("/analytics/stats")
def stats():
    """Return simple usage statistics for the interview demo."""
    return get_stats()
