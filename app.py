from flask import Flask, request
from flasgger import Swagger
import joblib
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from flask_cors import CORS

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

app = Flask(__name__)
swagger = Swagger(app)
CORS(app)

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
    processed_input = cv.transform([clean_review(msg)]).toarray()[0]
    prediction = classifier.predict([processed_input])[0]

    return {
        "sentiment": int(prediction),
    }

app.run(host="0.0.0.0", port=8080, debug=True)

all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
def clean_review(review):
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    return review