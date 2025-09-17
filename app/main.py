from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import numpy as np
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

# Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # <-- allow requests from any origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load YOLO model once at startup
model = YOLO("yolo11n.pt")

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Accepts an image (any common format), runs YOLO detection,
    and returns the annotated image as JPEG.
    """
    # Read entire uploaded file into memory
    image_bytes = await file.read()

    # Open with Pillow (handles JPEG/PNG/BMP/WebP/TIFF, etc.)
    img = Image.open(BytesIO(image_bytes)).convert("RGB")

    # Run YOLO inference directly on numpy array
    results = model(np.array(img))
    result = results[0]

    # Annotate image (result.plot() gives a BGR numpy array with boxes/labels)
    annotated_bgr = result.plot()
    annotated_rgb = Image.fromarray(annotated_bgr[..., ::-1])  # convert BGR -> RGB

    # Convert to JPEG bytes for response
    buf = BytesIO()
    annotated_rgb.save(buf, format="JPEG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/jpeg")