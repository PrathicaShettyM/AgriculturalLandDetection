from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = load_model("ResNet50_agriculture_suitability.keras")  # Make sure this is in the same folder

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    image = Image.open(file).convert("RGB").resize((64, 64))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 64, 64, 3)

    prediction = model.predict(img_array)
    predicted_class = int(np.argmax(prediction))

    # Map predicted class to human-readable result
    if predicted_class == 1:
        result = "🔴 This land is not suitable for agriculture."
    else:
        result = "🟢  This land is suitable for agriculture."

    return jsonify({
        'class': predicted_class,
        'result': result
    })


if __name__ == '__main__':
    app.run(port=5000)
