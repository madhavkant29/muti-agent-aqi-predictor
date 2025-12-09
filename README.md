self made dataset using openaq and openmeteo.

Quantized model- mistral-7b-instruct-v0.1.Q4_K_M.gguf
souce- TheBloke/Mistral-7B-Instruct-v0.1-GGUF


# Air Quality Prediction and Analysis System

An end-to-end AI pipeline for real-time Air Quality Index (AQI) prediction and explainable analysis.  
This project integrates machine learning, causal inference, and LLM reasoning within a FastAPI backend, alongside a minimal Streamlit interface for user interaction.

---

## Overview

This system predicts pollutant levels and overall AQI for any geographic location using live weather data.  
It further provides natural-language explanations using a quantized Mistral-7B model, enabling interpretable insights into pollution drivers and health implications.

---

## Architecture

### Core Components
1. FastAPI Backend  
   Coordinates data flow, caching, and model inference.

2. LangGraph Workflow  
   Implements a modular multi-agent pipeline:  
   Weather → MLP → DML → LLM

3. Data Ingestion Pipeline  
   Fetches and merges weather data (OpenMeteo) and pollution data (OpenAQ), cleans and aligns them temporally, and stores recent results in Redis for faster reuse.

4. Machine Learning Models
   - MLP (Multilayer Perceptron, PyTorch): Predicts pollutant concentrations from meteorological variables.  
   - DML (Double Machine Learning via EconML): Estimates causal effects of weather conditions on air quality.

5. LLM Integration
   - Quantized Mistral-7B via LlamaCPP: Produces contextual AQI summaries and health recommendations.

6. Streamlit Dashboard  
   Provides a simple UI for entering latitude/longitude and interacting with the backend.

---

## Tech Stack

| Category | Tools |
|---------|-------|
| Backend | FastAPI, LangGraph |
| Frontend | Streamlit |
| ML Frameworks | PyTorch, EconML |
| LLMs | Mistral-7B (quantized via LlamaCPP) |
| Data Handling | Pandas, NumPy, scikit-learn |
| Caching | Redis |
| Deployment | Docker |
| Others | Git |

---

## Workflow

1. User Input: Latitude and longitude are provided via the Streamlit UI.  
2. Data Ingestion: Weather and pollution data are retrieved, cleaned, scaled, and aligned.  
3. Model Prediction:  
   - The MLP predicts pollutant concentrations.  
   - The DML model infers causal relationships between weather variables and AQI levels.  
4. LLM Explanation: The Mistral-7B model generates a human-readable summary describing pollution causes and health effects.  
5. Caching: Timestamped weather results are stored in Redis to reduce API latency on repeated queries.

---

## Example Output

Input:
```json
{
  "latitude": 28.6139,
  "longitude": 77.2090
}
{
  "AQI": 172,
  "Category": "Unhealthy",
  "Explanation": "High PM2.5 levels due to low wind speed and high humidity. Sensitive groups should limit outdoor activity."
}
