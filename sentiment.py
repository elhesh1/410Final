from model import get_model

def predict_sentiment(text, model):
    """
    Predict the sentiment of the given text using the provided model.
    """
    sentiment_model = get_model()
    prediction = sentiment_model.predict([text])

    return prediction[0]
