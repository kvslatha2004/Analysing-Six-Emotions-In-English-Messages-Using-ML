from flask import Flask, request, render_template
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the pre-trained model
with open('SixEMo\model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the pre-trained TF-IDF vectorizer
with open('SixEMo\mtfidf.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = request.form['text']

        # Calculate TF-IDF values for the input text
        tfidf_values = vectorizer.transform([text])

        # Make prediction using the model
        prediction = model.predict(tfidf_values)
        probability = model.predict_proba(tfidf_values)


        return render_template('result.html', prediction=prediction[0], probability=probability[0])

    except Exception as e:
        return 'Error: {}'.format(e)

if __name__ == '__main__':
    app.run(debug=True)
