from model import get_model

# Calls our model to predict sentiment text
def predict_sentiment(text):
    model, vectorizer = get_model()
    X = vectorizer.transform([text])

    # Returns predicted sentiment
    # A value of 0 is negative, 1 is positive
    return model.predict(X)[0]