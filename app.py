from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import io
import numpy as np


# Load the model (ensure it matches the one used in your notebook)
model = tf.keras.models.load_model('trained_model.keras')

print(model.summary())

# Define class names (same as in your training code)
class_names = [
    'Apple Apple scab', 'Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy',
    'Blueberry healthy', 'Cherry (including sour) Powdery mildew', 'Cherry (including sour) healthy',
    'Corn (maize) Cercospora leaf spot Gray leaf spot', 'Corn (maize) Common rust',
    'Corn (maize) Northern Leaf Blight', 'Corn (maize) healthy', 'Grape Black rot',
    'Grape Esca (Black Measles)', 'Grape Leaf blight (Isariopsis Leaf Spot)', 'Grape healthy',
    'Orange Haunglongbing (Citrus greening)', 'Peach Bacterial spot', 'Peach healthy',
    'Pepper, bell Bacterial spot', 'Pepper, bell healthy', 'Potato Early blight',
    'Potato Late blight', 'Potato healthy', 'Raspberry healthy', 'Soybean healthy',
    'Squash Powdery mildew', 'Strawberry Leaf scorch', 'Strawberry healthy', 'Tomato Bacterial spot',
    'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf Mold', 'Tomato Septoria leaf spot',
    'Tomato Spider mites Two-spotted spider mite', 'Tomato Target Spot', 'Tomato Tomato Yellow Leaf Curl Virus',
    'Tomato Tomato mosaic virus', 'Tomato healthy'
]

app = Flask(__name__)
CORS(app) # To enable CORS

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Convert the uploaded file to a PIL Image
        image = Image.open(io.BytesIO(file.read()))
        
        # Resize the image to match the model input size (128x128)
        image = image.resize((128, 128))

        # Convert the image to an array and normalize the pixel values (between 0 and 1)
        input_arr = tf.keras.preprocessing.image.img_to_array(image)  # Convert to array
        #input_arr = input_arr / 255.0  # Normalize pixel values to [0, 1]
        input_arr = np.array([input_arr])  # Convert single image to batch dimension

        # Perform prediction
        predictions = model.predict(input_arr)
        
        # Log the raw prediction to check if it makes sense
        print(f"Raw predictions: {predictions}")

        # Get the predicted class and confidence score
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions)

        # Return the result as a JSON response
        return jsonify({
            "predicted_class": predicted_class,
            "confidence": float(confidence)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
