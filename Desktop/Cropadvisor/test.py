import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model("model.h5")

# Load class names
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Convert index→class mapping
class_names = {v: k for k, v in class_indices.items()}

# Path to the test image
img_path = "test_leaf.jpg"   # Change to your test image path

# Load and preprocess the image
img = image.load_img(img_path, target_size=(224, 224))  
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # Normalize like in training

# Predict
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)[0]

print(f"Predicted class index: {predicted_class}")
print(f"Predicted class name: {class_names[predicted_class]}")
