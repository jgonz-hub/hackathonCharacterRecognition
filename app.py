import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from tensorflow import keras
import cv2
import numpy as np

app = Flask(__name__, static_url_path='/static')
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
ALLOWED_EXTENSIONS = {'jpg', 'png', 'gif', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def recognize_character(image_path):
    # Load the pre-trained model
    model = keras.models.load_model('/Users/astroworld97/Desktop/hackathon_fresher/hackathonCharacterRecognition/my_model.keras')

    # Load the input image
    # input_image = cv2.imread(image_path)
    input_image = cv2.imread(input("Please provide path to input image: "))
    # input_image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

    #dimensions for resizing the image to the CNN's input layer's specifications
    desired_width = 256
    desired_height = 256

    # Resize the input image
    resized_image = cv2.resize(input_image, (desired_width, desired_height))
    resized_image = np.expand_dims(resized_image, axis=0)

    # Perform image recognition
    prediction = model.predict(resized_image)
    class_labels = ['Spongebob', 'Patrick']
    
    # Get the predicted label
    predicted_label = class_labels[prediction.argmax()]

    return predicted_label
    

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            return redirect(request.url) #used in the Flask framework to redirect the user's browser to the URL specified in the request.url variable

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # Save the uploaded file
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            # print(filename)
            # file.save(filename)

            # Perform image recognition here (not implemented in this example)

            # Redirect to the result page with the image filename
            return redirect(url_for('result_loader', filename=file.filename))

    return render_template('index.html')


@app.route('/result/<filename>')
def result(filename):
    # Perform image recognition and character detection here
    # You can pass the detected character as a variable to the template
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    detected_character = recognize_character(image_path)  # Call your recognition code
    return render_template('result.html', filename=filename, detected_character=detected_character)

@app.route('/result_loader/<filename>')
def result_loader(filename):
    return render_template('result_loader.html', filename=filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
