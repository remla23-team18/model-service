from flask import Flask, request
from flasgger import Swagger
import joblib
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from flask_cors import CORS

from scripts.preprocess import clean_review

app = Flask(__name__)
swagger = Swagger(app)
CORS(app)

# Loading BoW dictionary and the classifier
cv = pickle.load(open('models/c1_BoW_Sentiment_Model.pkl', "rb"))
classifier = joblib.load('models/c2_Classifier_Sentiment_Model')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Make a hardcoded prediction
    ---
    consumes:
      - application/json
    parameters:
        - name: input_data
          in: body
          description: message to be classified.
          required: True
          schema:
            type: object
            required: sms
            properties:
                msg:
                    type: string
                    example: This is an example msg.
    responses:
      200:
        description: Some result
    """
    msg = request.get_json().get('msg')
    processed_input = cv.transform([clean_review(msg)]).toarray()[0]
    prediction = classifier.predict([processed_input])[0]

    return {
        "sentiment": int(prediction),
    }

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)

