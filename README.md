# LLMSCAN: Proactive Monitoring & Alignment Verification

LLMSCAN is an advanced proactive monitoring and safety alignment system. It leverages **Causal Mediation Analysis** to track and intervene in the activation pathways of large language models (LLMs). By dynamically neutralizing safety-critical layers, LLMSCAN can detect backdoor triggers, lies, toxic behavior, and jailbreaks, steering the model back to safety.

## 🚀 Project Structure

- `backend/main.py` - The FastAPI backend server that handles model loading, causal scanning, and intervention logic.
- `frontend/app.py` - The Streamlit frontend dashboard for user interaction, prompt analysis, and metrics visualization.

## 🛠️ Prerequisites

Ensure you have Python installed (Python 3.8+ recommended). You will need the following primary packages:
- `fastapi`
- `uvicorn`
- `streamlit`
- `torch`
- `transformers`
- `plotly`
- `pandas`
- `numpy`
- `requests`

You can install dependencies using pip:
```bash
pip install fastapi uvicorn streamlit torch transformers plotly pandas numpy requests
```

## 🏃‍♂️ How to Run the Application

You will need to open **two separate terminal windows** to run the backend and frontend simultaneously.

### 1. Start the Backend Server
Open your first terminal, navigate to the `backend` folder, and start the FastAPI server:
```bash
cd backend
python main.py
```
*Wait for the terminal to show that the server is running on `http://127.0.0.1:8000`.*

### 2. Start the Frontend Application
Open a second terminal, navigate to the root directory, and launch the Streamlit app:
```bash
streamlit run frontend/app.py
```
*Your browser should automatically open to the LLMSCAN dashboard (usually `http://localhost:8501`).*

## 💡 Usage Guide
1. **Login:** Enter your authorized email address to access the workspace.
2. **Scanner Tab:** Enter a prompt and click "Run Causal Scan" to identify misbehavior probabilities and visualize causal influence across the model's layers and tokens.
3. **Intervention Lab:** Select an intervention strategy (Zero, Scale, Noise) and apply it to a targeted layer to observe how the model's output shifts from a misaligned state to a safe state.
4. **Metrics Dashboard:** View dynamic, real-time performance metrics (AUC, F1, Accuracy) based strictly on your prompt's forward passes.
