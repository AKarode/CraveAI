from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import uuid
import uvicorn
from dotenv import load_dotenv

# Import our services
from services.ocr import OCRService
from services.ai_recommendations import RecommendationService
from services.graph_rag import GraphRAGService

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
graph_rag_service = GraphRAGService()  # New GraphRAG service

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

class GraphRAGRequest(BaseModel):
    query: str
    constraints: Optional[Dict[str, Any]] = None

class GraphRAGResponse(BaseModel):
    success: bool
    message: str
    recommendations: List[dict]

class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    success: bool
    message: str
    answer: str
    source_items: Optional[List[dict]] = None

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "crave-ai-backend"}

# Menu processing endpoint
@app.post("/api/process-menu", response_model=MenuProcessingResponse, status_code=200)
async def process_menu(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Process an uploaded menu file (PDF or image)
    """
    # Validate file extension
    file_extension = file.filename.split(".")[-1].lower()
    if file_extension not in ["pdf", "png", "jpg", "jpeg"]:
        raise HTTPException(
            status_code=400, 
            detail="Unsupported file format"
        )
    
    try:
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

# New GraphRAG recommendation endpoint
@app.post("/api/graph-recommendations", response_model=GraphRAGResponse)
async def get_graph_recommendations(request: GraphRAGRequest):
    """
    Get enhanced food recommendations using the knowledge graph and LLM reasoning
    """
    try:
        if not graph_rag_service.knowledge_graph:
            return {
                "success": False,
                "message": "Knowledge graph not loaded. Please run the data processing script first.",
                "recommendations": []
            }
            
        result = graph_rag_service.generate_recommendations(
            query=request.query,
            constraints=request.constraints
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# New food question answering endpoint
@app.post("/api/food-question", response_model=QuestionResponse)
async def answer_food_question(request: QuestionRequest):
    """
    Answer food-related questions using the knowledge graph and LLM
    """
    try:
        result = graph_rag_service.answer_food_question(question=request.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 