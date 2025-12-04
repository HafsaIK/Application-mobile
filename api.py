import io
import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO

# Import refactored modules
import hybrid_malvoyant_detector_tts as hybrid

app = FastAPI()

# Global variables for models
models = {}

@app.on_event("startup")
def load_models():
    print("Loading models...")
    device = 0 if torch.cuda.is_available() else "cpu"
    models["device"] = device

    # 1. YOLO Models
    print("Loading YOLO models...")
    models["custom"] = YOLO(hybrid.CUSTOM_MODEL_PATH)
    models["coco"] = YOLO(hybrid.COCO_MODEL_PATH)
    
    print("All models loaded!")

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Process
    img_annot, sentence = hybrid.process_image_data(
        img, 
        models["custom"], 
        models["coco"], 
        models["device"]
    )

    # Encode image to JPEG
    _, img_encoded = cv2.imencode('.jpg', img_annot)
    
    import base64
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    
    return JSONResponse({
        "sentence": sentence,
        "image_base64": img_base64
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
