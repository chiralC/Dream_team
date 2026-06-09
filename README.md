# 🏏 Dream11 Next-Gen Team Builder

> **An AI-powered fantasy cricket optimization engine featuring real-time squad selection, explainable AI insights, and audio-visual strategy breakdowns.**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-blue.svg)
![Groq](https://img.shields.io/badge/AI-Llama_3.1-orange.svg)

---

## 🎯 Project Overview

Built for high-performance sports analytics, this application bridges the gap between raw predictive machine learning and human-readable strategy. It doesn't just predict player scores; it mathematically optimizes the team structure and explains exactly *why* each player was chosen using advanced SHAP analytics and broadcast-style GenAI audio commentary.

### ✨ Core Innovations
* **Leakage-Free ML Pipeline:** Robust XGBoost regressors trained on historical cricket data to predict baseline player fantasy points without overfitting to venue biasses.
* **Mathematical Optimization:** Implements Integer Linear Programming (ILP) using `PuLP` to guarantee the mathematically optimal 11-player squad under strict budget and role constraints.
* **Explainable AI (XAI):** Real-time `SHAP` (SHapley Additive exPlanations) integration provides transparent feature weights for every selection.
* **Generative AI Insights:** Utilizes a single-prompt, batched JSON request to Groq (Llama-3.1) to generate lightning-fast, highly contextual player commentary without hitting rate limits.
* **Cyberpunk UI & Data Viz:** A fully responsive, dark-themed Streamlit frontend featuring interactive `Plotly` scatter plots for deviation analysis.

---

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have Python installed, then clone this repository to your local machine.

### 2. Installation
Install all required dependencies to perfectly mirror the production environment:

    pip install -r requirements.txt

### 3. Launch the Application
Run the following command in your terminal:

    streamlit run app.py

---

## ⚙️ Configuration & Usage

To experience the full power of the Generative AI and Audio features:

1. Open the application in your browser.
2. In the left sidebar, enter your **Groq API Key** (starts with `gsk_`). *Keys can be generated at console.groq.com.*
3. Select your Match Parameters (Date, Teams, Match Type).
4. Click **Generate Dream Team**.

**🛡️ Graceful Degradation (Offline Safe):** If the application is run offline or without an API key, the system automatically catches the network error and bypasses the LLM/Audio steps. This ensures the core ML predictions and the optimized squad table are instantly delivered to the UI without application crashes.

---

## 📂 Architecture Mapping

| Component | File | Description |
| :--- | :--- | :--- |
| **Frontend** | `app.py` | Core Streamlit UI layout, state management, and interactive Plotly visualizations. |
| **ML Engine** | `predictor.py` | Deserialization of XGBoost models, prediction execution, and ILP squad optimization. |
| **GenAI Pipeline** | `ai_part_final.py` | SHAP calculation, Groq LLM integration, robust JSON parsing, and gTTS audio compilation. |