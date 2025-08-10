from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import time
import logging
from model_server import ChatbotModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Custom Chatbot Model API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request/response models
class Message(BaseModel):
    role: str
    content: str

class ModelRequest(BaseModel):
    message: str
    context: Optional[List[Message]] = []
    options: Optional[Dict[str, Any]] = {}

class ModelResponse(BaseModel):
    text: str
    confidence: float
    sources: List[str]
    processingTime: int
    intent: str

# Global variable for model
model = None

@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global model
    logger.info("Loading custom trained model...")
    
    try:
        model = ChatbotModel()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        return {"status": "initializing"}
    return {"status": "healthy", "model": "custom_intent_classifier"}

@app.post("/predict", response_model=ModelResponse)
async def predict(request: ModelRequest):
    """Generate a response from the model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    start_time = time.time()
    
    try:
        # Process message with model
        result = model.process_message(request.message)
        
        # Map intent to sources (for demonstration)
        intent_to_sources = {
            "greeting": ["conversation_starters"],
            "farewell": ["conversation_closers"],
            "thanks": ["conversation_acknowledgments"],
            "app_development": ["service_catalog", "mobile_development"],
            "web_development": ["service_catalog", "web_services"],
            "pricing": ["pricing_guide", "service_rates"],
            "contact": ["contact_information"],
            "schedule_meeting": ["scheduling_system", "calendar_api"]
        }
        
        sources = intent_to_sources.get(result["intent"], ["general_knowledge"])
        
        # Calculate processing time
        process_time = int((time.time() - start_time) * 1000)
        
        return {
            "text": result["text"],
            "confidence": result["confidence"],
            "sources": sources,
            "processingTime": process_time,
            "intent": result["intent"]
        }
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn api_server:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)