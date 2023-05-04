from flask import Flask, request
from flasgger import Swagger
import joblib
from sklearn.feature_extraction.text import CountVectorizer
import pickle

app = Flask(__name__)
swagger = Swagger(app)

# Loading BoW dictionary and the classifier
cv = pickle.load(open('models/c1_BoW_Sentiment_Model.pkl', "rb"))
classifier = joblib.load('models/c2_Classifier_Sentiment_Model')


@app.route('/', methods=['POST'])
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
    processed_input = cv.transform([msg]).toarray()[0]
    prediction = classifier.predict([processed_input])[0]

    return {
        "sentiment": int(prediction),
    }

app.run(host="0.0.0.0", port=8080, debug=True)
