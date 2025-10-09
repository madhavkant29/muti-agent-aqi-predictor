import asyncio
import os
import sys
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from typing import Dict, Any
from src.app.pipeline import create_pipeline_graph, PipelineState # Corrected import for clarity
from src.app.clients.redis_client import init_redis, close_redis
from src.app.schema import IngestRequest

# Add project root to path for imports to resolve src/app
# Note: This line might need adjustment based on the execution environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Global variable to hold the compiled LangGraph pipeline
aqi_pipeline = None 

# ----------------- FastAPI Lifespan -----------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes services (Redis) and the AQI pipeline graph."""
    print("Initializing services...")
    await init_redis()

    global aqi_pipeline
    # Create the compiled graph using the function we defined
    aqi_pipeline = create_pipeline_graph() 

    yield

    print("Shutting down services...")
    await close_redis()


app = FastAPI(title="AQI Prediction API", lifespan=lifespan)


@app.get("/")
async def read_root():
    """Simple root endpoint check."""
    return {"message": "Welcome to the AQI Prediction API"}


@app.post("/predict")
async def predict_aqi(req: IngestRequest):
    """
    Accepts an IngestRequest (lat, lon, dt_iso) and executes the AQI pipeline.
    
    The user's text query is expected to be inside req.user_query.
    """
    if aqi_pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized. Service is starting up.")

    # Initialize the state with the received request data
    initial_state = PipelineState(
        request=req,
        features={},        # empty initially
        mlp_preds={},
        dml_effects={},
        llm_response="",
        user_query=req.user_query # Use the user_query from the IngestRequest
    )

    try:
        # Run the full pipeline graph asynchronously
        final_state = await aqi_pipeline.ainvoke(initial_state)
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    return {
        "status": "success",
        "input_request": req.dict(),
        "advice": final_state["llm_response"]
    }
