from fastapi import Header, HTTPException, Query, WebSocket, status

from app.config import get_settings


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> str:
    keys = get_settings().api_keys
    if not keys:
        # No keys configured → auth disabled (dev mode).
        return "anonymous"
    if x_api_key is None or x_api_key not in keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing API key",
        )
    return x_api_key


async def require_api_key_ws(
    websocket: WebSocket,
    api_key: str | None = Query(default=None),
) -> bool:
    """Return True if authorized, otherwise close the WS and return False."""
    keys = get_settings().api_keys
    if not keys:
        return True
    header_key = websocket.headers.get("x-api-key")
    supplied = api_key or header_key
    if supplied is None or supplied not in keys:
        await websocket.close(code=4401)
        return False
    return True
