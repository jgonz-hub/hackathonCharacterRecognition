import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory

app = Flask(__name__, static_url_path='/static')
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'hackathonCharacterRecognition\\uploads')
ALLOWED_EXTENSIONS = {'jpg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # Save the uploaded file
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            # Perform image recognition here (not implemented in this example)

            # Redirect to the result page with the image filename
            return redirect(url_for('result_loader', filename=file.filename))

    return render_template('index.html')


@app.route('/result/<filename>')
def result(filename):
    # Perform image recognition and character detection here
    # You can pass the detected character as a variable to the template
    detected_character = "SpongeBob SquarePants"  # Example, replace with actual detection result

    return render_template('result.html', filename=filename, detected_character=detected_character)

@app.route('/result_loader/<filename>')
def result_loader(filename):
    return render_template('result_loader.html', filename=filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
