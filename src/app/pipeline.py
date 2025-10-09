import asyncio
import logging
from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, END
# FIX: Corrected import path for query_llm to llm_agent
from src.app.agents.ingestion_agent import IngestionAgent
from src.app.agents.mlp_agent import MLPAgent
from src.app.agents.dml_agent import DMLAgent
from src.app.agents.llm import query_llm
from src.app.schema import IngestRequest
from datetime import datetime, timezone
import pandas as pd

log = logging.getLogger(__name__)

class PipelineState(TypedDict):
    """Represents the state of the pipeline at each step."""
    request: IngestRequest
    features: Dict[str, Any]
    mlp_preds: Dict[str, float]
    dml_effects: Dict[str, Dict[str, float]]
    llm_response: str  # text output from LLM
    user_query: str

def create_pipeline_graph():
    """
    Creates a LangGraph pipeline that fetches weather, predicts with MLP/DML,
    and passes the results to the LLM for final advice.
    """
    # Instantiate agents
    mlp_agent = MLPAgent()
    dml_agent = DMLAgent()
    ingestion_agent = IngestionAgent()

    # --- Graph Nodes ---
    async def fetch_weather_node(state: PipelineState) -> PipelineState:
        """Fetches weather data using IngestionAgent."""
        req = state["request"]
        
        # Pass the full request to the ingestion agent
        payload, features = await ingestion_agent.ingest(req)

        if not features:
            log.error(f"Ingestion failed for request: {req.dict()}")
            raise ValueError("No weather data available from ingestion agent.")
        
        # Ensure the actual user_query from the IngestRequest is stored in the state
        return {
            "features": features,
            "user_query": req.user_query
        }


    async def mlp_node(state: PipelineState) -> PipelineState:
        """Runs the MLP model for pollutant predictions."""
        features = state["features"]
        log.info("Running MLP prediction.")
        # MLP predict is synchronous, run it in a separate thread
        mlp_preds = await asyncio.to_thread(mlp_agent.predict, features)
        return {"mlp_preds": mlp_preds}

    async def dml_node(state: PipelineState) -> PipelineState:
        """Runs the DML model for causal effect estimation."""
        features = state["features"]
        log.info("Running DML estimation.")
        # DML estimate_effects is synchronous, run it in a separate thread
        dml_effects = await asyncio.to_thread(dml_agent.estimate_effects, features)
        return {"dml_effects": dml_effects}

    async def llm_node(state: PipelineState) -> PipelineState:
        """Generates the final human-readable advice using the LLM."""
        # Retrieve the original request object to get the prediction time
        req = state["request"] 
        mlp_preds = state["mlp_preds"]
        dml_effects = state["dml_effects"]
        user_query = state["user_query"]
        
        log.info("Generating LLM advice.")

        # query_llm is synchronous, run it in a separate thread
        llm_response = await asyncio.to_thread(
            query_llm, 
            user_prompt=user_query,
            mlp_output=mlp_preds,
            dml_output=dml_effects,
            # Pass the requested prediction timestamp (dt_iso)
            prediction_dt_iso=req.dt_iso 
        )
        return {"llm_response": llm_response}


    # --- Build the Graph ---
    workflow = StateGraph(PipelineState)
    workflow.add_node("fetch_weather", fetch_weather_node)
    workflow.add_node("mlp_predict", mlp_node)
    workflow.add_node("dml_effects", dml_node)
    workflow.add_node("llm_advice", llm_node)

    # Edges
    workflow.add_edge("fetch_weather", "mlp_predict")
    workflow.add_edge("fetch_weather", "dml_effects")
    
    # After both parallel branches complete, proceed to LLM
    workflow.add_edge("mlp_predict", "llm_advice")
    workflow.add_edge("dml_effects", "llm_advice") 

    # Entry and exit
    workflow.set_entry_point("fetch_weather")
    workflow.add_edge("llm_advice", END)

    return workflow.compile()