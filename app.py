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
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

from model import FlickrModel
from inf import load_model_from_checkpoint, clean_caption



model = None 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
openai_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, openai_client
    model_path = "best_model_large.pth"
    model = load_model_from_checkpoint(model_path)
    model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.eval()

    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    openai_client = OpenAI(api_key=openai_api_key)

    yield
    model = None
    openai_client = None

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
            model_caption = clean_caption(generated_caption)

                # Generate caption with OpenAI
        openai_caption = "OpenAI API key not set"
    
        try:
            openai_response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": "describe briefly the content of this image."},
                            {"type": "image_url", "image_url": {"url": image_data.url}}
                        ]
                    }
                ],
                max_tokens=100
            )
            openai_caption = openai_response.choices[0].message.content

            image_response = openai_client.images.generate(
                model="dall-e-3",
                prompt=f"Generate an image based on this description: {openai_caption}",
                size="1024x1024",
            )

            generated_image_url = image_response.data[0].url

        except Exception as e:
            openai_caption = f"OpenAI error: {str(e)}"
        
        return {"model_caption": model_caption, "openai_caption": openai_caption, "generated_image_url": generated_image_url}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/test")
async def test_api():
    return {"message": "API is working"}


