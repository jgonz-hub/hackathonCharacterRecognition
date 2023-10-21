from tensorflow import keras
import cv2
import numpy as np
# Load the model from the .keras file
model = keras.models.load_model('/Users/astroworld97/Desktop/hackathon_fresh/hackathonCharacterRecognition/my_model.keras')
input_image = cv2.imread(input("Please provide path to input image: "))
desired_width = 256
desired_height = 256
resized_image = cv2.resize(input_image, (desired_width, desired_height))
resized_image = np.expand_dims(resized_image, axis=0)
prediction = model.predict(resized_image)
class_labels = ['Spongebob', 'Patrick']
predicted_label = [class_labels[i] for i in prediction.argmax(axis=1)]
print(predicted_label)