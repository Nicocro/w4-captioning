# setup basic FastAPI application
# create routes for: serving static files (frontend), handling user image uploads, running inference on uploaded images

from fastapi import FastAPI,  HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from contextlib import asynccontextmanager
import uvicorn
from pydantic import BaseModel
import torch
from PIL import Image
import requests
from io import BytesIO

from model import FlickrModel
from inf import load_model_from_checkpoint, clean_caption



model = None 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model_path = "best_model_1.pth"
    model = load_model_from_checkpoint(model_path)
    model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.eval()
    yield
    model = None

app = FastAPI(lifespan=lifespan)

static_path = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

class ImageURL(BaseModel):
    url: str

@app.get("/")
async def read_root():
    return FileResponse(static_path / "index.html")

@app.post("/api/generate-caption")
async def generate_caption(image_data: ImageURL):
    try:
        response = requests.get(image_data.url)
        image = Image.open(BytesIO(response.content))

        with torch.no_grad():
            generated_caption = model.generate_caption(image)
            clean_cap = clean_caption(generated_caption)
        
        return {"caption": clean_cap}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/test")
async def test_api():
    return {"message": "API is working"}

