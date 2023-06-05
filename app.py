import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# load pickle model
model = pickle.load(open('heart_model.pkl', 'rb'))





@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for (x) in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    if prediction == 1:
        prediction = 'POSITIVE'
    elif prediction == 0:
        prediction = 'NEGATIVE'

    return ({'Prediction': f'{prediction}'})


if __name__ == '__main__':
    app.run(debug=False)
