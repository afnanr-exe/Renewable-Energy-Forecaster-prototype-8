from fastapi import HTTPException

def raise_bad_request(msg: str):
    raise HTTPException(status_code=400, detail=msg)

def raise_internal_error(msg: str):
    raise HTTPException(status_code=500, detail=msg)
