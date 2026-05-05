# api.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import data_loader
import event_detector
# Import your other TrialFlux modules here...

app = FastAPI(title="TrialFlux API")

# --- Step 2a: Enable CORS ---
# This allows your separate frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[""],  # In production, replace "" with your frontend's URL (e.g., "http://localhost:3000")
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Step 2b: Create Endpoints ---

@app.get("/")
def read_root():
    return {"message": "TrialFlux Backend is Running!"}

@app.get("/api/load-eeg")
def get_eeg_data():
    """Endpoint to trigger data loading"""
    # Use your existing data_loader.py logic
    # Make sure to format the return data as a standard Python dictionary/list
    # so FastAPI can automatically convert it to JSON for your frontend.
    data = data_loader.load_or_generate_data() 
    return {"status": "success", "data": "Replace with actual structured data"}

@app.post("/api/analyze")
def analyze_signal(payload: dict):
    """Endpoint to trigger event detection and classification"""
    # Use your event_detector.py and classifier.py logic here
    # Example: results = classifier.classify_segment(payload['signal'])
    return {"status": "success", "verdict": "Biological Event", "confidence": 0.95}
