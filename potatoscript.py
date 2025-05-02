from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os

app = FastAPI()

# CORS setup
origins = ["http://localhost", "http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model with error handling
model_path = os.path.join(os.path.dirname(__file__), "potato_classifier.h5")
try:
    MODEL = tf.keras.models.load_model(model_path)
    print(f"Model loaded successfully from: {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    raise
CLASS_NAMES = ["Bacteria", "Fungi", "Healthy"]  

@app.get("/ping")
async def ping():
    return {"message": "API is alive!"}

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((256, 256))
    image = np.array(image) / 255.0  # Normalize to [0, 1] as in training
    print("Image mean after preprocessing:", image.mean())  # Debug: Should be ~0.5
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, 0)
        predictions = MODEL.predict(img_batch)
        print("Raw predictions:", predictions[0])  # Debug: Check probabilities
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))
        return {
            "class": predicted_class,
            "confidence": confidence,
            "probabilities": predictions[0].tolist()  # Return all probabilities
        }
    except Exception as e:
        print(f"Prediction error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)