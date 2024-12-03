from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import gzip
import joblib
import os

# Initialize the FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the input schema
class ModelInput(BaseModel):
    pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# Load the model
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "../model/diabetes_model_compressed.sav.gz")

try:
    with gzip.GzipFile(model_path, "rb") as f:
        diabetes_model = joblib.load(f)
except Exception as e:
    raise RuntimeError(f"Error loading the model: {e}")

# Define the prediction endpoint
@app.post("/diabetes_prediction")
def predict(input_data: ModelInput):
    try:
        # Convert input data to a list for prediction
        input_list = [
            input_data.pregnancies,
            input_data.Glucose,
            input_data.BloodPressure,
            input_data.SkinThickness,
            input_data.Insulin,
            input_data.BMI,
            input_data.DiabetesPedigreeFunction,
            input_data.Age,
        ]
        prediction = diabetes_model.predict([input_list])
        result = "The person is diabetic" if prediction[0] == 1 else "The person is not diabetic"
        return JSONResponse(content={"result": result})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Root route for testing
@app.get("/")
def root():
    return {"message": "Welcome to the FastAPI app!"}
