from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import torch
from transformers import AutoImageProcessor, AutoModel
from ultralytics import YOLO
import faiss
import numpy as np
from PIL import Image
import io
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="8X Sports ML Backend",
    description="AI-powered jersey design analysis and search",
    version="1.0.0"
)

# Add CORS middleware
cors_origins = os.getenv('CORS_ORIGINS', '*').split(',')
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,  # In production, restrict this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and index ONCE at startup
logger.info("Starting model loading process...")

try:
    # Get environment variables with defaults
    device_name = os.getenv('DEVICE', 'cpu')
    model_path = os.getenv('MODEL_PATH', 'models/deepfashion2_yolov8s-seg.pt')
    index_path = os.getenv('INDEX_PATH', 'index/jersey_index.faiss')
    metadata_path = os.getenv('METADATA_PATH', 'index/jersey_metadata.npy')
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, log_level))
    
    device = torch.device(device_name)
    logger.info(f"Using device: {device}")
    
    logger.info("Loading DINO processor...")
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    
    logger.info("Loading DINO model...")
    dino_model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
    
    logger.info("Loading YOLO model...")
    yolo_model = YOLO(model_path)
    
    logger.info("Loading FAISS index...")
    faiss_index = faiss.read_index(index_path)
    
    logger.info("Loading metadata...")
    loaded_data = np.load(metadata_path, allow_pickle=True)
    if isinstance(loaded_data, dict):
        index_to_path = {int(k): v for k, v in loaded_data.items()}
    elif isinstance(loaded_data, np.ndarray):
        index_to_path = {i: str(item) for i, item in enumerate(loaded_data)}
    else:
        index_to_path = {}
    
    logger.info("All models loaded successfully!")
    
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise Exception(f"Failed to load models: {str(e)}")

class FeaturesRequest(BaseModel):
    features: List[float] | List[List[float]]

@app.get("/")
async def root():
    return {
        "message": "8X Sports ML Backend is running!",
        "endpoints": ["/dino", "/faiss", "/yolo", "/docs"],
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": "2024-01-15T10:30:00.000Z",
        "models_loaded": True
    }

@app.post("/dino")
async def dino_inference(file: UploadFile = File(...)):
    try:
        logger.info(f"Processing DINO request for file: {file.filename}")
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt").to(device)
            outputs = dino_model(**inputs)
        
        features = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()[0]
        logger.info("DINO inference completed successfully")
        return {"features": features.tolist()}
        
    except Exception as e:
        logger.error(f"DINO inference failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"DINO inference failed: {str(e)}")

@app.post("/faiss")
async def faiss_search(request: FeaturesRequest):
    try:
        logger.info("Processing FAISS search request")
        features = request.features
        
        if isinstance(features[0], list):
            vector = np.array(features, dtype=np.float32)
        else:
            vector = np.array([features], dtype=np.float32)
            
        if vector.shape[1] != faiss_index.d:
            error_msg = f"Feature vector length {vector.shape[1]} does not match FAISS index dimension {faiss_index.d}"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
            
        faiss.normalize_L2(vector)
        distances, indices = faiss_index.search(vector, 15)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx in index_to_path:
                key = idx
                results.append({
                    "rank": i + 1,
                    "distance": float(distance),
                    "file_path": index_to_path[key],
                    "full_path": f"catalogue/{index_to_path[key]}"
                })
        
        logger.info(f"FAISS search completed, found {len(results)} results")
        return {"results": results}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"FAISS search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"FAISS search failed: {str(e)}")

@app.post("/yolo")
async def yolo_inference(file: UploadFile = File(...)):
    try:
        logger.info(f"Processing YOLO request for file: {file.filename}")
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        results = yolo_model(image, device=0 if torch.cuda.is_available() else 'cpu', verbose=False)[0]
        
        polygons = []
        if hasattr(results, 'masks') and results.masks is not None and hasattr(results.masks, 'xy'):
            for mask in results.masks.xy:
                polygons.append(mask.tolist())
        
        logger.info(f"YOLO inference completed, found {len(polygons)} polygons")
        return {"polygons": polygons}
        
    except Exception as e:
        logger.error(f"YOLO inference failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"YOLO inference failed: {str(e)}")