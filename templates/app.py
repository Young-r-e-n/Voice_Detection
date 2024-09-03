import base64
from flask import Flask, render_template, request, redirect, url_for
from keras.models import load_model
import numpy as np
import librosa
import os
from io import BytesIO
from pydub import AudioSegment

app = Flask(__name__)

# Load your trained model
model = load_model('trained_model.h5')

# Define a function to preprocess the uploaded or recorded audio
def preprocess_audio(file_path=None, audio_data=None):
    if file_path:
        y, sr = librosa.load(file_path, sr=None)
    else:
        audio = AudioSegment.from_file(BytesIO(base64.b64decode(audio_data.split(',')[1])), format="wav")
        y = np.array(audio.get_array_of_samples())
        sr = audio.frame_rate

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs = np.mean(mfccs.T, axis=0)
    return np.expand_dims(mfccs, axis=0)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        audio_data = request.form.get('audio_data')
        
        if file and file.filename != '':
            file_path = os.path.join('static', file.filename)
            file.save(file_path)
            X_test = preprocess_audio(file_path=file_path)
        elif audio_data:
            X_test = preprocess_audio(audio_data=audio_data)
        else:
            return redirect(request.url)

        y_pred = model.predict(X_test)
        y_pred_class = np.argmax(y_pred, axis=1)

        emotions = ['Happy', 'Sad', 'Angry', 'Neutral']  # Update with your actual labels
        emotion = emotions[y_pred_class[0]]

        return render_template('index.html', emotion=emotion, file_path=file_path if file else None)

    return render_template('index.html', emotion=None)

if __name__ == '__main__':
    app.run(debug=True)
