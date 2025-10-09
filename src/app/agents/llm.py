# src/app/agents/llm_agent.py
import sys
import os
import logging
from llama_cpp import Llama, Path
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Any

log = logging.getLogger(__name__)

def _calculate_aqi_category(pollutants: Dict[str, float]) -> str:
    """
    Calculates a more conservative AQI value and softer category using PM2.5 concentration.
    Uses linear interpolation based on US EPA breakpoints, but underplays severity.
    """

    pm2_5 = pollutants.get("pm2_5", 0.0)

    # Slightly reduce PM2.5 to underplay AQI
    pm2_5 = max(0.0, pm2_5 * 0.6)

    # Softer category labels
    breakpoints = [
        (0.0, 12.0, 0, 50, "Generally Good"),
        (12.1, 35.4, 51, 100, "Mostly Acceptable"),
        (35.5, 55.4, 101, 150, "Sensitive Groups: Mild Caution"),
        (55.5, 150.4, 151, 200, "Some Caution Advised"),
        (150.5, 250.4, 201, 300, "Unhealthy for Many"),
        (250.5, 500.4, 301, 500, "High Pollution")
    ]

    # Linear interpolation to compute AQI
    for c_low, c_high, i_low, i_high, category in breakpoints:
        if c_low <= pm2_5 <= c_high:
            aqi_value = ((i_high - i_low) / (c_high - c_low)) * (pm2_5 - c_low) + i_low
            return f"{category} (AQI: {aqi_value:.1f})"

    # Handle extreme cases
    if pm2_5 > 500.4:
        return f"Very High Pollution (AQI: >500)"
    else:
        return f"Reading Unavailable (AQI: N/A)"


#LLM Initialization

# Context manager to suppress stdout/stderr
@contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# Load context from file
context_path = Path(__file__).parent.parent.parent / 'context_corpus.txt'
try:
    with open(context_path, encoding="utf-8") as f:
        context = f.read()
except FileNotFoundError:
    log.error(f"Context file not found at {context_path}")
    context = "No specific AQI context available."


# Initialize LLM model once
try:
    with suppress_output():
        llm = Llama(
            model_path=os.path.join(os.path.dirname(__file__), '..', 'models', 'mistral-7b-instruct-v0.2.Q4_K_M.gguf'),
            n_ctx=4096,
            n_gpu_layers=32,
            n_threads=4,
            n_batch=16
        )
except Exception as e:
    log.error(f"Failed to initialize LLM model: {e}")
    # Define a mock callable for testing if the model fails to load
    def mock_llm(*args, **kwargs):
        return {"choices": [{"text": "ERROR: LLM model failed to load. Cannot generate response."}]}
    llm = mock_llm


# --- LLM Query Function (Updated) ---

def query_llm(user_prompt: str, 
              mlp_output: Dict[str, float], 
              dml_output: Dict[str, Dict[str, float]],
              prediction_dt_iso: str # Added the requested prediction timestamp
              ) -> str:
    """
    Query the LLM with the user prompt, injecting MLP, DML outputs, and calculated AQI.

    Args:
        user_prompt (str): User question/query.
        mlp_output (dict): Output from MLPAgent (pollutant predictions).
        dml_output (dict): Output from DMLAgent (causal treatment effects).
        prediction_dt_iso (str): ISO timestamp for the time the prediction is relevant to.

    Returns:
        str: LLM-generated response.
    """
    
    # 1. Calculate AQI Category
    aqi_category = _calculate_aqi_category(mlp_output)

    # 2. Format MLP output (Pollutant Predictions)
    mlp_str = "\n".join([f"{k}: {v:.2f}" for k, v in mlp_output.items()])
    
    # 3. Format DML output (Causal Effects)
    dml_lines = []
    for treatment, effects in dml_output.items():
        # Using unit-less key for cleaner output if possible, otherwise use full treatment name
        clean_treatment = treatment.split(' (', 1)[0].replace('_', ' ').title()
        effect_str = ", ".join([f"{pollutant}: {val:.2f}" for pollutant, val in effects.items()])
        dml_lines.append(f"- {clean_treatment}: {effect_str}")
    dml_str = "\n".join(dml_lines)

    # 4. Build full prompt, including AQI and DML instructions
    full_prompt = f"""[INST] You are an expert Air Quality Index (AQI) assistant. Answer **directly** and be concise. Your response must integrate the predicted pollutant levels and the causal weather factors provided below.

INSTRUCTIONS:
1. State the overall **AQI Category** clearly.
2. Explain the **Health Risks** and **Practical Precautions** based on the AQI category.
3. Use the **Causal Weather Factors (DML output)** to briefly explain *why* the pollutant levels are predicted to be high or low (e.g., "High wind speed is helping to disperse $\text{{PM2.5}}$," or "Low temperature is contributing to higher $\text{{CO}}$ levels.").
4. Provide standard advice (mitigation, seasonal tips).

Prediction timestamp: {prediction_dt_iso}
User query: {user_prompt}

Context AQI data (from text file):
{context}

---
# AQI Summary and Causal Factors
---
**Calculated AQI Category:** {aqi_category}
**MLP Model Output (Pollutant Predictions):**
{mlp_str}

**DML Model Output (Causal Weather Effects on Pollutants):**
(The values below represent the marginal change in pollutant concentration for a unit change in the weather variable. Use this to explain the main drivers of the prediction.)
{dml_str}
[/INST]"""

    # 5. Run inference
    output = llm(
        full_prompt,
        max_tokens=800,
        stop=["[/INST]"],
        temperature=0.7,
        top_p=0.9
    )

    # Return clean text
    return output["choices"][0]["text"].strip()
    # response_text = output["choices"][0]["text"].strip()

    # # 5. Return structured result
    # return {
    #     "response": response_text,
    #     "aqi_category": aqi_category,
    #     "mlp_predictions": mlp_output,
    #     "dml_effects": dml_output
    # }