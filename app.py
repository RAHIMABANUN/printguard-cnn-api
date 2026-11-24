
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import tensorflow as tf
import numpy as np
import json
import io

app = FastAPI(title="PRINTGUARD CNN API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model(
    "printguard_model.keras",
    compile=False
)

with open("class_names.json", "r") as f:
    class_names = json.load(f)

@app.get("/")
def home():
    return {"message": "PRINTGUARD CNN API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)[0]

    index = int(np.argmax(predictions))

    return {
        "defect": class_names[index],
        "confidence": round(float(predictions[index]) * 100, 2),
        "all_predictions": {
            class_names[i]: round(float(predictions[i]) * 100, 2)
            for i in range(len(class_names))
        }
    }
