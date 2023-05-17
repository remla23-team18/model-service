from flask import Flask, request, Response
from random import random
from flasgger import Swagger
import joblib
import pickle
from flask_cors import CORS

from scripts.preprocess import clean_review
from prometheus_client import Histogram, Summary, Counter, Gauge
import prometheus_client


app = Flask(__name__)
swagger = Swagger(app)
CORS(app)

# Loading BoW dictionary and the classifier
cv = pickle.load(open('models/c1_BoW_Sentiment_Model.pkl', "rb"))
classifier = joblib.load('models/c2_Classifier_Sentiment_Model')

# Set the metrics for prometheus

# 1. Counter: predict calls
predict_counter = Counter('predict_calls_total', 'Total number of calls to the predict function')

# 2. Gauge: model accuracy
total_prediction = Counter('total_predictions', 'Total number of predictions')
correct_prediction = Counter('total_predictions_correct', 'Total number of correct predictions')
accuracy = Gauge('total_accuracy', 'The accuracy of the model')

# 3. Histogram: text length
text_length_histogram = Histogram(
    'text_length',
    'Length of the text',
    labelnames=['sentiment'],
    buckets=[10, 20, 30, 40, 50, float('inf')]
)

# 4. Summary: response time
summary_metric = Summary('response_time_seconds', 'Response time in seconds')

@summary_metric.time()
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

    # Increment the predict_counter for each call
    predict_counter.inc()

    msg = request.get_json().get('msg')
    processed_input = cv.transform([clean_review(msg)]).toarray()[0]
    prediction = classifier.predict([processed_input])[0]

    # Measure the length of the input text
    text_length = len(msg.split(' '))
    sentiment_label = 'pos' if int(prediction) == 1 else 'neg'
    text_length_histogram.labels(sentiment=sentiment_label).observe(text_length)

    return {
        "sentiment": int(prediction),
    }


@app.route('/feedback', methods=['POST'])
def feedback():
    """
    Record feedback on prediction accuracy
    ---
    consumes:
      - application/json
    parameters:
        - name: input_data
          in: body
          description: feedback data.
          required: True
          schema:
            type: object
            properties:
                sentiment:
                    type: integer
                    example: 1
                accuracy:
                    type: integer
                    example: 1
    responses:
      200:
        description: Feedback recorded
    """

    feedback = request.get_json().get('feedback')

    total_prediction.inc()

    # if sentiment is not None and feedback is not None:
    if feedback == 1:
        correct_prediction.inc()

    accuracy.set(correct_prediction._value.get() / total_prediction._value.get())

    return {
        "message": "Feedback recorded",
        "feedback": feedback
    }


@app.route('/metrics', methods=['GET'])
def metrics():

    registry = prometheus_client.CollectorRegistry()
    registry.register(text_length_histogram)
    registry.register(predict_counter)
    registry.register(correct_prediction)
    registry.register(total_prediction)
    registry.register(accuracy)
    registry.register(summary_metric)

    return Response(prometheus_client.generate_latest(registry), mimetype="text/plain")


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)

