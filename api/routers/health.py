from fastapi import APIRouter

router = APIRouter()


@router.get("")
def healthcheck() -> dict:
    return {
        "status": "ok",
        "service": "markowitz-platform-api",
    }