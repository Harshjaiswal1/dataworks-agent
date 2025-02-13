# app.py
import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse, JSONResponse
from pathlib import Path
from tasks import run_task  # Our main task router

app = FastAPI()

# The directory where all file operations are allowed.
DATA_DIR = Path("/data")  # For local testing you might change this to Path("./data")

@app.post("/run")
async def run_endpoint(task: str = Query(..., description="Plain-English task description")):
    try:
        result = run_task(task)
        return {"status": "success", "result": result}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/read", response_class=PlainTextResponse)
async def read_endpoint(path: str = Query(..., description="Path of file to read")):
    file_path = Path(path)
    if not file_path.resolve().as_posix().startswith(DATA_DIR.resolve().as_posix()):
        raise HTTPException(status_code=400, detail="Access denied. File must be under /data.")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    content = file_path.read_text()
    return content

# B10: An API endpoint to filter a CSV file and return JSON.
# Example: /filter-csv?path=/data/sample.csv&column=status&value=active
import pandas as pd
@app.get("/filter-csv")
async def filter_csv_endpoint(path: str = Query(..., description="Path to CSV file"),
                              column: str = Query(..., description="Column to filter on"),
                              value: str = Query(..., description="Value to match")):
    file_path = Path(path)
    if not file_path.resolve().as_posix().startswith(DATA_DIR.resolve().as_posix()):
        raise HTTPException(status_code=400, detail="Access denied. File must be under /data.")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="CSV file not found")
    try:
        df = pd.read_csv(file_path)
        filtered_df = df[df[column] == value]
        result = filtered_df.to_dict(orient="records")
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
