from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import uuid
import uvicorn
from dotenv import load_dotenv

# Import our services
from services.ocr import OCRService
from services.ai_recommendations import RecommendationService

# Load environment variables
load_dotenv()

app = FastAPI(title="Crave AI Backend")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
ocr_service = OCRService()
recommendation_service = RecommendationService()

# Define request/response models
class MenuProcessingResponse(BaseModel):
    menu_id: str
    items: List[dict]
    success: bool
    message: str

class PreferenceRequest(BaseModel):
    preferences: List[str]
    dietary_restrictions: Optional[List[str]] = None

class RecommendationResponse(BaseModel):
    recommendations: List[dict]

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "crave-ai-backend"}

# Menu processing endpoint
@app.post("/api/process-menu", response_model=MenuProcessingResponse)
async def process_menu(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Process an uploaded menu file (PDF or image)
    """
    try:
        # Validate file extension
        file_extension = file.filename.split(".")[-1].lower()
        if file_extension not in ["pdf", "png", "jpg", "jpeg"]:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Generate menu ID
        menu_id = str(uuid.uuid4())
        
        # Process file with OCR
        if file_extension == "pdf":
            text = ocr_service.process_pdf(file.file)
        else:
            # Save image temporarily for processing
            file_content = await file.read()
            with open(f"/tmp/{file.filename}", "wb") as f:
                f.write(file_content)
            
            text = ocr_service.process_image(f"/tmp/{file.filename}")
            
            # Clean up temporary file
            os.remove(f"/tmp/{file.filename}")
        
        # Extract menu items
        menu_items = ocr_service.extract_menu_items(text)
        
        # Store in vector database in background
        if background_tasks:
            background_tasks.add_task(
                recommendation_service.store_menu_items,
                menu_items=menu_items,
                menu_id=menu_id
            )
        
        return {
            "menu_id": menu_id,
            "items": menu_items,
            "success": True,
            "message": f"Menu '{file.filename}' has been processed successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Recommendation endpoint
@app.post("/api/recommendations/{menu_id}", response_model=RecommendationResponse)
async def get_recommendations(
    menu_id: str,
    request: PreferenceRequest
):
    """
    Get personalized dish recommendations based on preferences
    """
    try:
        recommendations = recommendation_service.get_recommendations(
            menu_id=menu_id,
            preferences=request.preferences,
            dietary_restrictions=request.dietary_restrictions
        )
        
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 