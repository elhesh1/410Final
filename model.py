import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

data_file = r"data\training.1600000.processed.noemoticon.csv"

def clean_data(data):
    # Creating sentiment analysis model using the Sentiment140 dataset from Kaggle
    df = pd.read_csv(data, encoding="latin-1", header=None)
    df.columns = ["target", "id", "date", "flag", "user", "text"] 


    ''' Cleaning the data. In the data set 0 is negative and 4 is positive, but It makes more 
    sense to have 0 negative and 1 positive '''
    df["target"] = df["target"].replace({4: 1})

    '''  Get rid of non alphebetical characters and make all lower case '''
    df["clean"] = df["text"].astype(str).str.lower()
    df["clean"] = df["clean"].str.replace(r"[^a-z\s]", "", regex=True)
    df["clean"] = df["clean"].str.replace(r"\s+", " ", regex=True).str.strip()
    return df

def train_model(data_file):
    df = clean_data(data_file)
    # making an 80/20 split to create a model and then use on to test
    X_train, X_test, y_train, y_test = train_test_split( df["clean"], df["target"],test_size=0.2,)

    # creating model using TF-IDF vecotirzation

    vecto = TfidfVectorizer(max_features=25000)
    X_train_vec = vecto.fit_transform(X_train)
    X_test_vec = vecto.transform(X_test)

    # train  model 

    model = LogisticRegression(max_iter=100)
    model.fit(X_train_vec, y_train)
    return model, X_test_vec, y_test

def model_evaluation(model, X_test_vec, y_test):
    # analyzing how good the model is
    y_pred = model.predict(X_test_vec)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    
    if model.score(X_test_vec, y_test) >= 0.75:
        print("Model accuracy is acceptable, saving the model...")
        with open('data\sentiment-model.pkl','wb') as f:
            pickle.dump(model,f)
    else:
        print("Model accuracy is below acceptable threshold. Try changing model parameters or data preprocessing.")


def get_model():
    model = None
    try:
        with open('data\sentiment-model.pkl','rb') as f:
            print("Found existing model, loading it...")
            model = pickle.load(f)
    except FileNotFoundError:
        print("No existing model found, training a new model...")
        model,X_test_vec,y_test = train_model(data_file)
        model_evaluation(model, X_test_vec, y_test)
    return model

if __name__ == "__main__":
    if get_model() is None:
        print("Model training failed.")
