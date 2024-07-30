from argparse import ArgumentParser

import pandas as pd
import psycopg2
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

def get_data(save_data=False):
    if save_data:
        print("get data from db")
        db_connect = psycopg2.connect(host="localhost", database="mydatabase", user="myuser", password="mypassword")
        df = pd.read_sql("SELECT * FROM iris_data ORDER BY id DESC LIMIT 100", db_connect)
        df.to_csv("data.csv", index=False)
    else:
        print("get data from saved data")
        df = pd.read_csv("data.csv") 
    X = df.drop(["id", "timestamp", "target"], axis="columns")
    y = df["target"]
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=2022)       
    return X_train, X_valid, y_train, y_valid

def train(X_train, y_train):
    model_pipeline = Pipeline([("scaler", StandardScaler()), ("svc", SVC())])
    print("training")
    model_pipeline.fit(X_train, y_train)

    return model_pipeline

def prediction(model, X_train, X_valid):
    train_pred = model.predict(X_train)
    valid_pred = model.predict(X_valid)

    return train_pred, valid_pred

def metric(y_train, train_pred, y_valid, valid_pred):
    train_acc = accuracy_score(y_true=y_train, y_pred=train_pred)
    valid_acc = accuracy_score(y_true=y_valid, y_pred=valid_pred)

    return train_acc, valid_acc

def save_pipeline(model_pipeline):
    print("saving pipeline")
    joblib.dump(model_pipeline, "models/db_model_pipeline.joblib")

def load_pipeline():
    print("loading model")
    return joblib.load("models/db_model_pipeline.joblib")

def save():
    X_train, X_valid, y_train, y_valid = get_data(save_data=True)
    model = train(X_train, y_train)
    train_pred, valid_pred = prediction(model, X_train, X_valid)
    train_acc, valid_acc = metric(y_train, train_pred, y_valid, valid_pred)
    print(f'train acc: {train_acc}, valid acc: {valid_acc}')
    save_pipeline(model)

def load():
    X_train, X_valid, y_train, y_valid = get_data()
    model = load_pipeline()
    train_pred, valid_pred = prediction(model, X_train, X_valid)
    train_acc, valid_acc = metric(y_train, train_pred, y_valid, valid_pred)
    print(f'train acc: {train_acc}, valid acc: {valid_acc}')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-t', '--type', default='load', choices=['save', 'load'])

    args = parser.parse_args()
    if args.type == 'save':
        save()
    else:
        load()